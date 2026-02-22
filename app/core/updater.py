from __future__ import annotations

import json
import re
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal

GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
CUDA_WIN_PATTERN = re.compile(r"cuda.*win|win.*cuda", re.IGNORECASE)


@dataclass
class ReleaseInfo:
    tag_name: str
    name: str
    published_at: str
    assets: list[AssetInfo]


@dataclass
class AssetInfo:
    name: str
    size: int
    download_url: str


def _parse_releases(data: list[dict[str, Any]]) -> list[ReleaseInfo]:
    releases: list[ReleaseInfo] = []
    for item in data:
        assets = [
            AssetInfo(
                name=a["name"],
                size=a["size"],
                download_url=a["browser_download_url"],
            )
            for a in item.get("assets", [])
        ]
        releases.append(
            ReleaseInfo(
                tag_name=item["tag_name"],
                name=item.get("name") or item["tag_name"],
                published_at=item.get("published_at", ""),
                assets=assets,
            )
        )
    return releases


def fetch_releases(max_pages: int = 3, per_page: int = 30) -> list[ReleaseInfo]:
    releases: list[ReleaseInfo] = []
    for page in range(1, max_pages + 1):
        page_releases = fetch_releases_page(page, per_page)
        if not page_releases:
            break
        releases.extend(page_releases)
    return releases


def fetch_releases_page(page: int, per_page: int = 30) -> list[ReleaseInfo]:
    """Fetch a single page of releases from the GitHub API."""
    url = f"{GITHUB_API}?per_page={per_page}&page={page}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data: list[dict[str, Any]] = json.loads(resp.read().decode())
    if not data:
        return []
    return _parse_releases(data)


def find_cuda_assets(release: ReleaseInfo) -> list[AssetInfo]:
    """Return full llama.cpp CUDA Windows packages (excludes cudart-only runtime packs)."""
    return [
        a for a in release.assets
        if CUDA_WIN_PATTERN.search(a.name)
        and a.name.endswith(".zip")
        and not a.name.startswith("cudart-")
    ]


def find_cudart_assets(release: ReleaseInfo) -> list[AssetInfo]:
    """Return CUDA runtime-only packages (cudart-llama-bin-win-cuda-*.zip)."""
    return [
        a for a in release.assets
        if a.name.startswith("cudart-")
        and CUDA_WIN_PATTERN.search(a.name)
        and a.name.endswith(".zip")
    ]


def download_asset(
    url: str,
    dest: Path,
    progress_cb: Callable[[int, int], None] | None = None,
) -> Path:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=600) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 256 * 1024
        with open(dest, "wb") as fp:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                fp.write(chunk)
                downloaded += len(chunk)
                if progress_cb:
                    progress_cb(downloaded, total)
    return dest


def extract_and_replace(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        top_dirs = {name.split("/")[0] for name in zf.namelist() if "/" in name}
        has_single_root = len(top_dirs) == 1

        if has_single_root:
            root_prefix = top_dirs.pop() + "/"
            for info in zf.infolist():
                if info.is_dir():
                    continue
                rel = info.filename[len(root_prefix):] if info.filename.startswith(root_prefix) else info.filename
                if not rel:
                    continue
                out = target_dir / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        else:
            zf.extractall(target_dir)


class DownloadWorker(QObject):
    # progress(downloaded, total, task_idx, task_count)
    progress = Signal(int, int, int, int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, tasks: list[tuple[str, Path]], target_dir: Path) -> None:
        super().__init__()
        # tasks: list of (url, dest_path)
        self._tasks = tasks
        self._target_dir = target_dir

    def run(self) -> None:
        task_count = len(self._tasks)
        try:
            for task_idx, (url, dest) in enumerate(self._tasks):
                i, n = task_idx, task_count
                download_asset(
                    url, dest,
                    progress_cb=lambda d, t, _i=i, _n=n: self.progress.emit(d, t, _i, _n),
                )
                extract_and_replace(dest, self._target_dir)
                try:
                    dest.unlink()
                except OSError:
                    pass
            self.finished.emit(str(self._target_dir))
        except Exception as exc:
            self.error.emit(str(exc))


def start_download_thread(
    tasks: list[tuple[str, Path]],
    target_dir: Path,
) -> tuple[QThread, DownloadWorker]:
    thread = QThread()
    worker = DownloadWorker(tasks, target_dir)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    return thread, worker
