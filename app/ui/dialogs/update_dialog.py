from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.updater import (
    DownloadWorker,
    ReleaseInfo,
    fetch_releases_page,
    find_cuda_assets,
    find_cudart_assets,
    start_download_thread,
)

PER_PAGE = 30
_NO_DOWNLOAD = "不下载"


class _PageFetchWorker(QObject):
    finished = Signal(list)  # list[ReleaseInfo]
    error = Signal(str)

    def __init__(self, page: int) -> None:
        super().__init__()
        self._page = page

    def run(self) -> None:
        try:
            releases = fetch_releases_page(self._page, PER_PAGE)
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self.finished.emit(releases)


class _LazyVersionCombo(QComboBox):
    """ComboBox that triggers release fetching on first popup open."""

    popup_about_to_show = Signal()

    def showPopup(self) -> None:
        self.popup_about_to_show.emit()
        super().showPopup()


class UpdateDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, default_target_dir: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("版本更新 — llama.cpp (CUDA)")
        self.setMinimumWidth(580)
        self._releases: list[ReleaseInfo] = []
        self._current_page = 0
        self._loading = False
        self._all_loaded = False
        self._download_thread: QThread | None = None
        self._download_worker: DownloadWorker | None = None
        self._fetch_thread: QThread | None = None
        self._fetch_worker: _PageFetchWorker | None = None
        self._default_target_dir = default_target_dir
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(8)

        self.version_combo = _LazyVersionCombo()
        self.version_combo.setMinimumWidth(340)
        self.version_combo.addItem("点击下拉框加载版本列表…")
        self.version_combo.popup_about_to_show.connect(self._on_popup_about_to_show)
        self.version_combo.currentIndexChanged.connect(self._on_version_changed)
        form.addRow("Release 版本", self.version_combo)

        # Full llama.cpp CUDA package
        self.asset_combo = QComboBox()
        self.asset_combo.currentIndexChanged.connect(self._refresh_download_state)
        form.addRow("llama 包", self.asset_combo)

        # cudart runtime package (optional)
        self.cudart_combo = QComboBox()
        self.cudart_combo.addItem(_NO_DOWNLOAD)
        self.cudart_combo.currentIndexChanged.connect(self._refresh_download_state)
        form.addRow("cudart 包（可选）", self.cudart_combo)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        form.addRow("详情", self.info_label)

        target_row = QHBoxLayout()
        self.target_edit = QLineEdit(self._default_target_dir)
        self.target_edit.setPlaceholderText("选择安装目标目录")
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self._browse_target)
        target_row.addWidget(self.target_edit, 1)
        target_row.addWidget(browse_btn)
        form.addRow("目标目录", target_row)

        layout.addLayout(form)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel()
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.download_btn = QPushButton("下载并安装")
        self.download_btn.setEnabled(False)
        self.download_btn.clicked.connect(self._start_download)
        self.cancel_btn = QPushButton("关闭")
        self.cancel_btn.clicked.connect(self.close)
        btn_row.addWidget(self.download_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

    # ── version list loading ──────────────────────────────────────────────────

    def _on_popup_about_to_show(self) -> None:
        if self._loading or self._all_loaded or self._current_page > 0:
            return
        self._fetch_next_page()
        view = self.version_combo.view()
        scrollbar = view.verticalScrollBar()
        scrollbar.valueChanged.connect(self._on_version_list_scrolled)

    def _on_version_list_scrolled(self, value: int) -> None:
        scrollbar = self.version_combo.view().verticalScrollBar()
        if value >= scrollbar.maximum() - 5 and not self._loading and not self._all_loaded:
            self._fetch_next_page()

    def _fetch_next_page(self) -> None:
        if self._loading or self._all_loaded:
            return
        self._loading = True
        self._current_page += 1
        self._show_status(f"正在加载第 {self._current_page} 页版本列表…")

        self._fetch_thread = QThread()
        self._fetch_worker = _PageFetchWorker(self._current_page)
        self._fetch_worker.moveToThread(self._fetch_thread)
        self._fetch_thread.started.connect(self._fetch_worker.run)
        self._fetch_worker.finished.connect(self._on_page_fetched)
        self._fetch_worker.error.connect(self._on_fetch_error)
        self._fetch_worker.finished.connect(self._fetch_thread.quit)
        self._fetch_worker.error.connect(self._fetch_thread.quit)
        self._fetch_thread.finished.connect(self._on_fetch_thread_done)
        self._fetch_thread.finished.connect(self._fetch_thread.deleteLater)
        self._fetch_thread.start()

    def _on_page_fetched(self, releases: list) -> None:
        self._loading = False

        if not releases:
            self._all_loaded = True
            self._hide_status()
            return

        is_first_load = not self._releases
        popup_open = self.version_combo.view().isVisible()

        if popup_open:
            self.version_combo.hidePopup()

        if is_first_load:
            self.version_combo.blockSignals(True)
            self.version_combo.clear()
            self.version_combo.blockSignals(False)

        self.version_combo.blockSignals(True)
        for r in releases:
            date_str = r.published_at[:10] if r.published_at else ""
            self.version_combo.addItem(f"{r.tag_name}  ({date_str})", userData=r)
        self.version_combo.blockSignals(False)

        if popup_open:
            self.version_combo.showPopup()

        self._releases.extend(releases)

        if len(releases) < PER_PAGE:
            self._all_loaded = True

        self._hide_status()

        if is_first_load and self._releases:
            self.version_combo.setCurrentIndex(0)
            self._on_version_changed(0)

    def _on_fetch_thread_done(self) -> None:
        self._fetch_thread = None
        self._fetch_worker = None

    def _on_fetch_error(self, msg: str) -> None:
        self._loading = False
        # Reset page counter so user can retry by reopening the dropdown
        if not self._releases:
            self._current_page = 0
        self._show_status(f"获取失败: {msg}（可重新点击下拉框重试）")

    # ── asset selection ───────────────────────────────────────────────────────

    def _on_version_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._releases):
            return
        release = self._releases[index]

        # Populate llama package combo
        cuda_assets = find_cuda_assets(release)
        self.asset_combo.blockSignals(True)
        self.asset_combo.clear()
        if cuda_assets:
            for a in cuda_assets:
                size_mb = a.size / (1024 * 1024)
                self.asset_combo.addItem(f"{a.name}  ({size_mb:.1f} MB)", userData=a)
        else:
            self.asset_combo.addItem("（无完整 CUDA Windows 资源）")
        self.asset_combo.blockSignals(False)

        # Populate cudart combo (optional)
        cudart_assets = find_cudart_assets(release)
        self.cudart_combo.blockSignals(True)
        self.cudart_combo.clear()
        self.cudart_combo.addItem(_NO_DOWNLOAD)
        for a in cudart_assets:
            size_mb = a.size / (1024 * 1024)
            self.cudart_combo.addItem(f"{a.name}  ({size_mb:.1f} MB)", userData=a)
        self.cudart_combo.blockSignals(False)

        self._refresh_download_state()

    def _selected_cuda_asset(self) -> object | None:
        """Return selected AssetInfo from asset_combo, or None if unavailable."""
        ver_idx = self.version_combo.currentIndex()
        if ver_idx < 0 or ver_idx >= len(self._releases):
            return None
        release = self._releases[ver_idx]
        cuda_assets = find_cuda_assets(release)
        asset_idx = self.asset_combo.currentIndex()
        if cuda_assets and 0 <= asset_idx < len(cuda_assets):
            return cuda_assets[asset_idx]
        return None

    def _selected_cudart_asset(self) -> object | None:
        """Return selected AssetInfo from cudart_combo, or None if 不下载."""
        cidx = self.cudart_combo.currentIndex()
        if cidx <= 0:
            return None
        ver_idx = self.version_combo.currentIndex()
        if ver_idx < 0 or ver_idx >= len(self._releases):
            return None
        release = self._releases[ver_idx]
        cudart_assets = find_cudart_assets(release)
        real_idx = cidx - 1  # offset for "不下载"
        if 0 <= real_idx < len(cudart_assets):
            return cudart_assets[real_idx]
        return None

    def _refresh_download_state(self) -> None:
        cuda_asset = self._selected_cuda_asset()
        cudart_asset = self._selected_cudart_asset()

        has_any = cuda_asset is not None or cudart_asset is not None
        self.download_btn.setEnabled(has_any)

        lines: list[str] = []
        if cuda_asset is not None:
            size_mb = cuda_asset.size / (1024 * 1024)
            lines.append(f"llama:  {cuda_asset.name}  ({size_mb:.1f} MB)")
        if cudart_asset is not None:
            size_mb = cudart_asset.size / (1024 * 1024)
            lines.append(f"cudart: {cudart_asset.name}  ({size_mb:.1f} MB)")

        ver_idx = self.version_combo.currentIndex()
        if 0 <= ver_idx < len(self._releases):
            pub = self._releases[ver_idx].published_at
            lines.append(f"发布: {pub[:10] if pub else 'N/A'}")

        if not lines:
            lines.append("当前版本没有可用的 CUDA Windows 资源。")
        self.info_label.setText("\n".join(lines))

    # ── download ──────────────────────────────────────────────────────────────

    def _browse_target(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择目标目录", self.target_edit.text() or str(Path.home()))
        if d:
            self.target_edit.setText(d)

    def _start_download(self) -> None:
        target_dir = self.target_edit.text().strip()
        if not target_dir:
            QMessageBox.warning(self, "缺少目标目录", "请选择安装目标目录。")
            return

        cuda_asset = self._selected_cuda_asset()
        cudart_asset = self._selected_cudart_asset()

        tasks: list[tuple[str, Path]] = []
        if cuda_asset is not None:
            tasks.append((cuda_asset.download_url, Path(tempfile.gettempdir()) / cuda_asset.name))
        if cudart_asset is not None:
            tasks.append((cudart_asset.download_url, Path(tempfile.gettempdir()) / cudart_asset.name))

        if not tasks:
            QMessageBox.warning(self, "未选择", "请至少选择一个下载项。")
            return

        self._task_count = len(tasks)
        self.download_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._show_status("正在下载…")

        target = Path(target_dir)
        self._download_thread, self._download_worker = start_download_thread(tasks, target)
        self._download_worker.progress.connect(self._on_progress)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.error.connect(self._on_download_error)
        self._download_thread.finished.connect(self._download_thread.deleteLater)
        self._download_thread.start()

    def _on_progress(self, downloaded: int, total: int, task_idx: int, task_count: int) -> None:
        prefix = f"[{task_idx + 1}/{task_count}] " if task_count > 1 else ""
        if total > 0:
            pct = int(downloaded * 100 / total)
            self.progress_bar.setValue(pct)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            self._show_status(f"{prefix}下载中… {mb_done:.1f} / {mb_total:.1f} MB ({pct}%)")
        else:
            mb_done = downloaded / (1024 * 1024)
            self._show_status(f"{prefix}下载中… {mb_done:.1f} MB")

    def _on_download_finished(self, target_dir: str) -> None:
        self.progress_bar.setValue(100)
        self._show_status(f"安装完成: {target_dir}")
        self.download_btn.setEnabled(True)
        self._cleanup_download_thread()
        QMessageBox.information(self, "完成", f"llama.cpp 已更新至:\n{target_dir}")

    def _on_download_error(self, msg: str) -> None:
        self._show_status(f"下载失败: {msg}")
        self.download_btn.setEnabled(True)
        self._cleanup_download_thread()
        QMessageBox.warning(self, "下载失败", msg)

    def _show_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.status_label.setVisible(True)

    def _hide_status(self) -> None:
        self.status_label.setVisible(False)

    def _cleanup_download_thread(self) -> None:
        if self._download_thread and self._download_thread.isRunning():
            self._download_thread.quit()
            self._download_thread.wait(3000)
        self._download_thread = None
        self._download_worker = None

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._cleanup_download_thread()
        if self._fetch_thread and self._fetch_thread.isRunning():
            self._fetch_thread.quit()
            self._fetch_thread.wait(2000)
        super().closeEvent(event)
