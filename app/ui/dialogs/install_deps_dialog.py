"""Dialog for installing missing dependencies on a remote SSH host."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from app.core.resource_monitor import REMOTE_VENV_DIR, REMOTE_VENV_PYTHON

if TYPE_CHECKING:
    from app.core.ssh_client import SSHConnection


class _InstallWorker(QObject):
    line_received = Signal(str)
    finished = Signal(int)

    def __init__(self, conn: "SSHConnection") -> None:
        super().__init__()
        self._conn = conn

    def _exec(self, cmd: str) -> int:
        """Run *cmd* via SSH, stream every output line, return exit code."""
        transport = self._conn.client.get_transport()
        if transport is None:
            self.line_received.emit("错误: SSH 连接已断开")
            return -1

        channel = transport.open_session()
        channel.get_pty()
        channel.exec_command(cmd)

        buf = ""
        while not channel.exit_status_ready():
            if channel.recv_ready():
                data = channel.recv(4096).decode("utf-8", errors="replace")
                buf += data
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    self.line_received.emit(line.rstrip("\r"))
            time.sleep(0.05)

        while channel.recv_ready():
            data = channel.recv(4096).decode("utf-8", errors="replace")
            buf += data
        for remaining in buf.splitlines():
            self.line_received.emit(remaining.rstrip("\r"))

        return channel.recv_exit_status()

    def _run(self, cmd: str) -> int:
        """Emit the command line, execute it, return exit code."""
        self.line_received.emit(f"$ {cmd}")
        return self._exec(cmd)

    def run(self) -> None:
        try:
            self._do_install()
        except Exception as e:
            self.line_received.emit(f"错误: {e}")
            self.finished.emit(-1)

    def _do_install(self) -> None:
        venv = REMOTE_VENV_DIR
        vpy = REMOTE_VENV_PYTHON

        rc = self._run(f"test -d {venv}")
        if rc != 0:
            self.line_received.emit("")
            self.line_received.emit("虚拟环境不存在，正在创建……")
            rc = self._run(f"python3 -m venv {venv}")
            if rc != 0:
                self._run("rm -rf ~/.llamacpprunner")
                self.line_received.emit("")
                self.line_received.emit("虚拟环境创建失败，请确认远程设备已安装 python3-venv。")
                self.finished.emit(rc)
                return
        else:
            self.line_received.emit(f"虚拟环境已存在: {venv}")

        self.line_received.emit("")
        rc = self._run(f"{vpy} -m pip --version")
        if rc != 0:
            self.line_received.emit("")
            self.line_received.emit(
                "虚拟环境中 pip 不可用，请先在远程设备上手动安装 pip，例如：\n"
                "  sudo apt install python3-pip      (Debian/Ubuntu)\n"
                "  sudo dnf install python3-pip       (Fedora/RHEL)\n"
                "  sudo pacman -S python-pip          (Arch)"
            )
            self.finished.emit(rc)
            return

        self.line_received.emit("")
        rc = self._run(f"{vpy} -m pip install psutil")
        self.finished.emit(rc)


class InstallDepsDialog(QDialog):
    """Modal dialog that installs psutil in a remote venv via SSH."""

    def __init__(self, conn: "SSHConnection", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("安装远程依赖")
        self.setMinimumSize(620, 380)
        self.resize(620, 380)

        layout = QVBoxLayout(self)

        info = QLabel(f"正在通过 SSH 配置虚拟环境 ({REMOTE_VENV_DIR}) 并安装依赖……")
        layout.addWidget(info)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        layout.addWidget(self._log, 1)

        self._status = QLabel()
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._close_btn = QPushButton("关闭")
        self._close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._close_btn)
        layout.addLayout(btn_row)

        self._worker = _InstallWorker(conn)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._worker.line_received.connect(self._on_line)
        self._worker.finished.connect(self._on_finished)
        self._thread.started.connect(self._worker.run)
        self._thread.start()

    def _on_line(self, text: str) -> None:
        self._log.appendPlainText(text)

    def _on_finished(self, exit_code: int) -> None:
        if exit_code == 0:
            self._status.setText("安装完成")
            self._status.setStyleSheet("color: #22c55e; font-weight: bold;")
        else:
            self._status.setText(f"安装失败 (exit code: {exit_code})")
            self._status.setStyleSheet("color: #ef4444; font-weight: bold;")
        self._thread.quit()
        self._thread.wait(2000)

    def closeEvent(self, event) -> None:
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        super().closeEvent(event)
