"""SSH client for remote server operations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paramiko import SFTPClient, SSHClient


class SSHConnection:
    """Manages SSH connection and provides remote operations."""

    def __init__(self, host: str, port: int, username: str, password: str) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self._client: SSHClient | None = None
        self._sftp: SFTPClient | None = None

    def connect(self) -> None:
        """Establish SSH connection."""
        import paramiko

        if self._client is not None:
            self.disconnect()

        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._client.connect(
            hostname=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            timeout=10,
        )
        self._sftp = self._client.open_sftp()

    def disconnect(self) -> None:
        """Close SSH connection."""
        if self._sftp:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    @property
    def client(self) -> "SSHClient | None":
        """Underlying paramiko SSHClient for advanced use."""
        return self._client

    @property
    def is_connected(self) -> bool:
        """Check if connection is active."""
        if self._client is None:
            return False
        try:
            transport = self._client.get_transport()
            return transport is not None and transport.is_active()
        except Exception:
            return False

    def exec_command(self, command: str, timeout: float | None = None) -> tuple[int, str, str]:
        """Execute command on remote host. Returns (exit_code, stdout, stderr)."""
        if not self.is_connected:
            raise RuntimeError("SSH 未连接")
        stdin, stdout, stderr = self._client.exec_command(command, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        return exit_code, out, err

    def list_dir(self, path: str) -> list[tuple[str, bool]]:
        """List directory contents. Returns list of (name, is_dir)."""
        if not self.is_connected or self._sftp is None:
            raise RuntimeError("SSH 未连接")
        result = []
        for entry in self._sftp.listdir_attr(path):
            mode = entry.st_mode
            is_dir = (mode & 0o170000) == 0o040000  # S_IFDIR
            result.append((entry.filename, is_dir))
        return sorted(result, key=lambda x: (not x[1], x[0].lower()))

    def find_gguf_files(self, base_path: str) -> list[str]:
        """Recursively find .gguf files under base_path. Returns relative paths."""
        if not self.is_connected:
            raise RuntimeError("SSH 未连接")
        result: list[str] = []

        def _walk(current: str, prefix: str) -> None:
            try:
                for name, is_dir in self.list_dir(current):
                    full = f"{current.rstrip('/')}/{name}"
                    rel = f"{prefix}/{name}" if prefix else name
                    if is_dir:
                        _walk(full, rel)
                    elif name.lower().endswith(".gguf"):
                        result.append(rel)
            except (PermissionError, OSError):
                pass

        _walk(base_path.rstrip("/") or "/", "")
        return sorted(result, key=str.casefold)

    def get_connection_string(self) -> str:
        """Return display string: user@host:port"""
        return f"{self.username}@{self.host}:{self.port}"
