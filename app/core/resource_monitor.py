"""System resource monitor for local and remote (SSH) machines."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QThread, QTimer, Signal

if TYPE_CHECKING:
    from app.core.ssh_client import SSHConnection

STATUS_OK = "ok"
STATUS_DEPENDENCY_MISSING = "dependency_missing"

REMOTE_VENV_DIR = "~/.llamacpprunner/venv"
REMOTE_VENV_PYTHON = "~/.llamacpprunner/venv/bin/python3"


@dataclass
class GpuInfo:
    index: int
    name: str
    short_name: str
    mem_used_mb: int
    mem_total_mb: int
    power_w: float


@dataclass
class ResourceSnapshot:
    status: str = STATUS_OK
    cpu_freq_mhz: int = 0
    cpu_percent: float = 0.0
    mem_used_mb: int = 0
    mem_total_mb: int = 0
    gpus: list[GpuInfo] = field(default_factory=list)


def _shorten_gpu_name(name: str) -> str:
    """'NVIDIA GeForce RTX 4060 Ti' -> '4060Ti'"""
    name = name.strip()
    cleaned = re.sub(
        r"(?i)^(NVIDIA\s+)?(GeForce\s+)?(RTX\s+|GTX\s+)?", "", name
    ).strip()
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned or name


def _collect_local() -> ResourceSnapshot:
    import psutil

    snap = ResourceSnapshot()

    freq = psutil.cpu_freq()
    snap.cpu_freq_mhz = int(freq.current) if freq else 0
    snap.cpu_percent = psutil.cpu_percent(interval=None)

    mem = psutil.virtual_memory()
    snap.mem_used_mb = int(mem.used / 1048576)
    snap.mem_total_mb = int(mem.total / 1048576)

    snap.gpus = _query_local_gpus()
    return snap


def _query_local_gpus() -> list[GpuInfo]:
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if r.returncode != 0:
            return []
        return _parse_nvidia_csv(r.stdout)
    except Exception:
        return []


def _collect_remote(conn: "SSHConnection") -> ResourceSnapshot:
    snap = ResourceSnapshot()
    if conn is None or not conn.is_connected:
        return snap

    try:
        exit_code, out, err = conn.exec_command(
            f"{REMOTE_VENV_PYTHON} -c \""
            "import psutil, json;"
            "f=psutil.cpu_freq();"
            "print(json.dumps({'freq':int(f.current) if f else 0,"
            "'pct':psutil.cpu_percent(interval=0),"
            "'mu':int(psutil.virtual_memory().used/1048576),"
            "'mt':int(psutil.virtual_memory().total/1048576)}))"
            "\"",
            timeout=5,
        )
        if exit_code != 0:
            snap.status = STATUS_DEPENDENCY_MISSING
            return snap
        if out.strip():
            import json

            d = json.loads(out.strip().splitlines()[-1])
            snap.cpu_freq_mhz = d.get("freq", 0)
            snap.cpu_percent = d.get("pct", 0.0)
            snap.mem_used_mb = d.get("mu", 0)
            snap.mem_total_mb = d.get("mt", 0)
    except Exception:
        snap.status = STATUS_DEPENDENCY_MISSING
        return snap

    try:
        _, out, _ = conn.exec_command(
            "nvidia-smi --query-gpu=index,name,memory.used,memory.total,power.draw "
            "--format=csv,noheader,nounits",
            timeout=5,
        )
        if out.strip():
            snap.gpus = _parse_nvidia_csv(out)
    except Exception:
        pass

    return snap


def _parse_nvidia_csv(text: str) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            name = parts[1]
            mem_used = int(float(parts[2]))
            mem_total = int(float(parts[3]))
            power = float(parts[4]) if parts[4] not in ("[N/A]", "N/A", "") else 0.0
            gpus.append(
                GpuInfo(
                    index=idx,
                    name=name,
                    short_name=_shorten_gpu_name(name),
                    mem_used_mb=mem_used,
                    mem_total_mb=mem_total,
                    power_w=power,
                )
            )
        except (ValueError, IndexError):
            continue
    return gpus


class _PollWorker(QObject):
    """Collects resource data off the main thread."""

    result_ready = Signal(object)
    finished = Signal()

    def __init__(self, mode: str, conn: "SSHConnection | None") -> None:
        super().__init__()
        self._mode = mode
        self._conn = conn

    def run(self) -> None:
        try:
            if self._mode == "ssh" and self._conn is not None:
                snap = _collect_remote(self._conn)
            else:
                snap = _collect_local()
        except Exception:
            snap = ResourceSnapshot()
        self.result_ready.emit(snap)
        self.finished.emit()


class ResourceMonitor(QObject):
    """Periodically emits ResourceSnapshot every ~1 second."""

    snapshot_ready = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._poll)
        self._mode = "local"
        self._conn: SSHConnection | None = None
        self._thread: QThread | None = None
        self._worker: _PollWorker | None = None
        self._busy = False

    def start(self, mode: str = "local", conn: "SSHConnection | None" = None) -> None:
        self._mode = mode
        self._conn = conn
        self._busy = False
        if mode == "local":
            import psutil
            psutil.cpu_percent(interval=None)
        self._timer.start()
        self._poll()

    def stop(self) -> None:
        self._timer.stop()
        self._cleanup_worker()

    def set_mode(self, mode: str, conn: "SSHConnection | None" = None) -> None:
        self._mode = mode
        self._conn = conn

    def _poll(self) -> None:
        if self._busy:
            return
        self._busy = True

        self._worker = _PollWorker(self._mode, self._conn)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._worker.result_ready.connect(self._on_result)
        self._worker.finished.connect(self._cleanup_worker)
        self._thread.started.connect(self._worker.run)
        self._thread.start()

    def _on_result(self, snap: ResourceSnapshot) -> None:
        self.snapshot_ready.emit(snap)

    def _cleanup_worker(self) -> None:
        self._busy = False
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread.deleteLater()
            self._thread = None
        self._worker = None
