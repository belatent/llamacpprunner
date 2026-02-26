"""SSH-based process runner for remote llama-server execution."""

from __future__ import annotations

import datetime as dt

from PySide6.QtCore import QObject, QThread, Signal, Qt

from app.core.ssh_client import SSHConnection


class _SSHExecWorker(QObject):
    """Worker that runs command via SSH and emits output."""

    line_received = Signal(str, bool)
    partial_received = Signal(str, bool)
    state_changed = Signal(str)
    process_started = Signal(str)
    process_stopped = Signal(int, float)
    error = Signal(str)
    run_finished = Signal()

    def __init__(self, conn: SSHConnection, command: list[str], cwd: str | None) -> None:
        super().__init__()
        self._conn = conn
        self._command = command
        self._cwd = cwd
        self._channel = None
        self._started_at: dt.datetime | None = None
        self._stop_requested = False

    def run(self) -> None:
        import shlex
        import time

        exit_code = -1
        try:
            if not self._conn.is_connected:
                self.error.emit("SSH 未连接")
                return

            self._started_at = dt.datetime.now()
            cmd_str = " ".join(shlex.quote(a) for a in self._command)
            if self._cwd:
                cwd = self._cwd.replace("\\", "/")
                cmd_str = f"cd {shlex.quote(cwd)} && {cmd_str}"

            self.process_started.emit(cmd_str)
            self.state_changed.emit("starting")
            self.state_changed.emit("running")

            self._channel = None
            try:
                stdin, stdout, stderr = self._conn.client.exec_command(cmd_str, get_pty=True)
                self._channel = stdout.channel

                out_buf = ""
                err_buf = ""
                while not self._stop_requested and not self._channel.exit_status_ready():
                    if self._channel.recv_ready():
                        data = self._channel.recv(4096).decode("utf-8", errors="replace")
                        out_buf = self._emit_lines(out_buf + data, False)
                    if self._channel.recv_stderr_ready():
                        data = self._channel.recv_stderr(4096).decode("utf-8", errors="replace")
                        err_buf = self._emit_lines(err_buf + data, True)
                    time.sleep(0.05)

                if self._stop_requested:
                    try:
                        self._channel.close()
                    except Exception:
                        pass

                # Drain any remaining output after the loop exits
                while self._channel.recv_ready():
                    data = self._channel.recv(4096).decode("utf-8", errors="replace")
                    out_buf = self._emit_lines(out_buf + data, False)
                while self._channel.recv_stderr_ready():
                    data = self._channel.recv_stderr(4096).decode("utf-8", errors="replace")
                    err_buf = self._emit_lines(err_buf + data, True)

                exit_code = self._channel.recv_exit_status()
                self.partial_received.emit("", False)
                self.partial_received.emit("", True)
                if out_buf.strip():
                    self.line_received.emit(out_buf.strip(), False)
                if err_buf.strip():
                    self.line_received.emit(err_buf.strip(), True)
            except Exception as e:
                exit_code = -1
                self.line_received.emit(f"错误: {e}", True)
        finally:
            elapsed = 0.0
            if self._started_at:
                elapsed = (dt.datetime.now() - self._started_at).total_seconds()
            self.state_changed.emit("stopped")
            self.process_stopped.emit(exit_code, elapsed)
            self.run_finished.emit()

    def _emit_lines(self, buf: str, is_error: bool) -> str:
        """Emit complete lines and partial tail, return remainder."""
        while "\n" in buf or "\r" in buf:
            idx = buf.find("\n")
            idx2 = buf.find("\r")
            if idx2 >= 0 and (idx < 0 or idx2 < idx):
                idx = idx2
            if idx < 0:
                break
            line = buf[:idx].rstrip("\r\n")
            buf = buf[idx + 1:]
            if line:
                self.line_received.emit(line, is_error)
        tail = buf.rstrip("\r")
        self.partial_received.emit(tail, is_error)
        return buf

    def stop(self) -> None:
        self._stop_requested = True
        if self._channel:
            try:
                self._channel.close()
            except Exception:
                pass


class SSHProcessRunner(QObject):
    """Runs remote commands via SSH, emitting same signals as ProcessRunner."""

    line_received = Signal(str, bool)
    partial_received = Signal(str, bool)
    state_changed = Signal(str)
    process_started = Signal(str)
    process_stopped = Signal(int, float)

    def __init__(self) -> None:
        super().__init__()
        self._thread: QThread | None = None
        self._worker: _SSHExecWorker | None = None
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def start(
        self,
        conn: SSHConnection,
        command: list[str],
        cwd: str | None = None,
    ) -> None:
        if self._running:
            raise RuntimeError("进程仍在运行中，不能重复启动。")
        if not command:
            raise ValueError("空命令无法启动。")

        self._running = True
        self._worker = _SSHExecWorker(conn, command, cwd)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._worker.line_received.connect(self.line_received.emit)
        self._worker.partial_received.connect(self.partial_received.emit)
        self._worker.state_changed.connect(self._on_state)
        self._worker.process_started.connect(self.process_started.emit)
        self._worker.process_stopped.connect(self._on_stopped)
        self._worker.error.connect(self._on_error)
        self._worker.run_finished.connect(self._thread.quit, Qt.DirectConnection)

        self._thread.started.connect(self._worker.run)
        self._thread.start()

    def _on_state(self, state: str) -> None:
        self._running = state in ("starting", "running", "stopping")
        self.state_changed.emit(state)

    def _on_stopped(self, exit_code: int, elapsed: float) -> None:
        self._running = False
        thread = self._thread
        self._thread = None
        self._worker = None
        if thread is not None:
            thread.quit()
            thread.wait()
            thread.deleteLater()
        self.process_stopped.emit(exit_code, elapsed)

    def _on_error(self, msg: str) -> None:
        self.line_received.emit(msg, True)

    def stop(self, force_after_ms: int = 2500) -> None:
        if not self._running or not self._worker:
            return
        self.state_changed.emit("stopping")
        self._worker.stop()
