from __future__ import annotations

import datetime as dt

from PySide6.QtCore import QObject, QProcess, QTimer, Signal


class ProcessRunner(QObject):
    line_received = Signal(str, bool)
    partial_received = Signal(str, bool)
    state_changed = Signal(str)
    process_started = Signal(str)
    process_stopped = Signal(int, float)

    def __init__(self) -> None:
        super().__init__()
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.SeparateChannels)
        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.readyReadStandardError.connect(self._read_stderr)
        self._process.finished.connect(self._on_finished)
        self._process.stateChanged.connect(self._on_qprocess_state_changed)
        self._started_at: dt.datetime | None = None
        self._stdout_buf = ""
        self._stderr_buf = ""

    def is_running(self) -> bool:
        return self._process.state() != QProcess.NotRunning

    def start(self, command: list[str], cwd: str | None = None) -> None:
        if self.is_running():
            raise RuntimeError("进程仍在运行中，不能重复启动。")
        if not command:
            raise ValueError("空命令无法启动。")
        if cwd:
            self._process.setWorkingDirectory(cwd)
        self._started_at = dt.datetime.now()
        self.process_started.emit(" ".join(command))
        self.state_changed.emit("starting")
        self._process.start(command[0], command[1:])

    def stop(self, force_after_ms: int = 2500) -> None:
        if not self.is_running():
            return
        self.state_changed.emit("stopping")
        self._process.terminate()
        QTimer.singleShot(force_after_ms, self._kill_if_still_running)

    def _kill_if_still_running(self) -> None:
        if self.is_running():
            self._process.kill()

    def _process_buf(self, buf: str, is_error: bool) -> str:
        """Emit complete lines from buf; emit the tail as partial; return the tail."""
        *lines, tail = buf.split("\n")
        for line in lines:
            line = line.rstrip("\r")
            if line:
                self.line_received.emit(line, is_error)
        tail = tail.rstrip("\r")
        self.partial_received.emit(tail, is_error)
        return tail

    def _read_stdout(self) -> None:
        raw = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._stdout_buf = self._process_buf(self._stdout_buf + raw, False)

    def _read_stderr(self) -> None:
        raw = bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace")
        self._stderr_buf = self._process_buf(self._stderr_buf + raw, True)

    def _on_qprocess_state_changed(self, state: QProcess.ProcessState) -> None:
        if state == QProcess.Running:
            self.state_changed.emit("running")

    def _on_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        for buf, is_error in ((self._stdout_buf, False), (self._stderr_buf, True)):
            self.partial_received.emit("", is_error)
            stripped = buf.rstrip("\r\n")
            if stripped:
                self.line_received.emit(stripped, is_error)
        self._stdout_buf = ""
        self._stderr_buf = ""

        elapsed = 0.0
        if self._started_at:
            elapsed = (dt.datetime.now() - self._started_at).total_seconds()
        self.state_changed.emit("stopped")
        self.process_stopped.emit(exit_code, elapsed)
