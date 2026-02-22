from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFontDatabase, QFontMetrics, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

MAX_BLOCK_COUNT = 10_000
FLUSH_INTERVAL_MS = 50
PREVIEW_MAX_HEIGHT = 200
PREVIEW_MIN_HEIGHT = 38


class LogPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumWidth(440)
        self._lines: list[str] = []
        self._pending: list[str] = []
        self._pending_partial: str = ""
        self._displayed_partial: str = ""
        self._preview_visible = False
        self._build_ui()

        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(FLUSH_INTERVAL_MS)
        self._flush_timer.timeout.connect(self._flush_pending)
        self._flush_timer.start()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 8, 8, 8)
        root.setSpacing(6)

        self._build_command_preview(root)
        self._build_toolbar(root)

        self.output_edit = QPlainTextEdit()
        self.output_edit.setReadOnly(True)
        fixed_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self.output_edit.setFont(fixed_font)
        self.output_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.output_edit.setMaximumBlockCount(MAX_BLOCK_COUNT)

        root.addWidget(self.output_edit, 1)

    def _build_command_preview(self, parent_layout: QVBoxLayout) -> None:
        self.preview_toggle_btn = QPushButton("命令预览 ▼")
        self.preview_toggle_btn.setProperty("flat", True)
        self.preview_toggle_btn.clicked.connect(self._toggle_preview)

        self.preview_edit = QPlainTextEdit()
        self.preview_edit.setReadOnly(True)
        self.preview_edit.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.preview_edit.setVisible(False)
        self.preview_edit.setMaximumHeight(PREVIEW_MAX_HEIGHT)
        self.preview_edit.setMinimumHeight(0)
        self.preview_edit.document().contentsChanged.connect(self._adjust_preview_height)

        parent_layout.addWidget(self.preview_toggle_btn, 0, Qt.AlignLeft)
        parent_layout.addWidget(self.preview_edit)

    def _build_toolbar(self, parent_layout: QVBoxLayout) -> None:
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        toolbar.addWidget(QLabel("过滤"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("输入关键词过滤日志")
        self.filter_edit.textChanged.connect(self._refresh_view)
        self.auto_scroll_check = QCheckBox("自动滚动")
        self.auto_scroll_check.setChecked(True)
        self.clear_btn = QPushButton("清空")
        self.save_btn = QPushButton("保存日志")
        self.clear_btn.clicked.connect(self.clear)
        self.save_btn.clicked.connect(self.save_log)

        toolbar.addWidget(self.filter_edit, 1)
        toolbar.addWidget(self.auto_scroll_check)
        toolbar.addWidget(self.clear_btn)
        toolbar.addWidget(self.save_btn)
        parent_layout.addLayout(toolbar)

    def set_command_preview(self, text: str) -> None:
        self.preview_edit.setPlainText(text)

    def _toggle_preview(self) -> None:
        self._preview_visible = not self._preview_visible
        self.preview_edit.setVisible(self._preview_visible)
        self.preview_toggle_btn.setText(
            "命令预览 ▲" if self._preview_visible else "命令预览 ▼"
        )
        if self._preview_visible:
            self._adjust_preview_height()

    def _adjust_preview_height(self) -> None:
        if not self._preview_visible:
            return
        doc = self.preview_edit.document()
        font = self.preview_edit.font()
        fm = QFontMetrics(font)
        line_height = fm.lineSpacing()
        block_count = doc.blockCount()

        # account for wrapped lines by checking document layout
        layout_height = int(doc.size().height())
        margins = self.preview_edit.contentsMargins()
        frame_width = self.preview_edit.frameWidth() * 2
        padding = margins.top() + margins.bottom() + frame_width + 8

        # use the larger of block-based or layout-based estimate
        block_based = block_count * line_height + padding
        layout_based = layout_height + padding
        desired = max(block_based, layout_based)
        clamped = max(PREVIEW_MIN_HEIGHT, min(desired, PREVIEW_MAX_HEIGHT))
        self.preview_edit.setFixedHeight(clamped)

    def append_line(self, line: str, is_error: bool = False) -> None:
        prefix = datetime.now().strftime("%H:%M:%S")
        final = f"[{prefix}] {line}"
        self._lines.append(final)
        if self._line_visible(final):
            self._pending.append(final)

    def update_partial(self, text: str, is_error: bool = False) -> None:
        """Receive a partial (no-newline) line update; displayed in-place on next flush."""
        self._pending_partial = text

    def _flush_pending(self) -> None:
        has_new_lines = bool(self._pending)
        has_new_partial = self._pending_partial != self._displayed_partial
        if not has_new_lines and not has_new_partial:
            return

        cursor = self.output_edit.textCursor()

        # Remove the previously displayed partial line (it lives on the last block,
        # without a trailing newline, so select from StartOfBlock to End and delete).
        if self._displayed_partial:
            cursor.movePosition(QTextCursor.End)
            cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            self._displayed_partial = ""

        # Append complete lines (each ends with \n, so a new empty block is ready).
        if self._pending:
            batch = self._pending
            self._pending = []
            cursor.movePosition(QTextCursor.End)
            cursor.insertText("\n".join(batch) + "\n")

        # Append the new partial (no trailing \n — stays on the last block).
        if self._pending_partial:
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(self._pending_partial)
            self._displayed_partial = self._pending_partial

        if self.auto_scroll_check.isChecked():
            self.output_edit.verticalScrollBar().setValue(
                self.output_edit.verticalScrollBar().maximum()
            )

    def clear(self) -> None:
        self._lines.clear()
        self._pending.clear()
        self._pending_partial = ""
        self._displayed_partial = ""
        self.output_edit.clear()

    def save_log(self) -> None:
        default_name = f"llama-log-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
        path, _ = QFileDialog.getSaveFileName(self, "保存日志", str(Path(default_name)), "Log Files (*.log)")
        if not path:
            return
        try:
            Path(path).write_text("\n".join(self._lines), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "保存失败", f"保存日志失败：{exc}")

    def _line_visible(self, line: str) -> bool:
        keyword = self.filter_edit.text().strip()
        if not keyword:
            return True
        return keyword.lower() in line.lower()

    def _refresh_view(self) -> None:
        keyword = self.filter_edit.text().strip().lower()
        if not keyword:
            visible_lines = self._lines
        else:
            visible_lines = [line for line in self._lines if keyword in line.lower()]

        self.output_edit.setUpdatesEnabled(False)
        self.output_edit.setPlainText("\n".join(visible_lines))
        self._displayed_partial = ""  # display was fully rebuilt
        self.output_edit.setUpdatesEnabled(True)

        cursor = self.output_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.output_edit.setTextCursor(cursor)
