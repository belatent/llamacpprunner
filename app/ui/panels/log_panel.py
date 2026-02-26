from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFontDatabase, QKeyEvent, QTextCursor, QTextOption
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QLineEdit,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

MAX_BLOCK_COUNT = 10_000
FLUSH_INTERVAL_MS = 50
PREVIEW_MIN_HEIGHT = 38
MAX_SSH_HISTORY = 200


class _AutoHeightTextEdit(QTextEdit):
    """QTextEdit that auto-adjusts fixed height to fit all content, pushing log area down.

    Uses QTextEdit (not QPlainTextEdit) because its document().size() correctly
    accounts for word-wrap after setTextWidth(), making height calculation reliable.
    WrapAnywhere is set so that long paths without spaces also wrap at the widget edge.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.WidgetWidth)
        self.setWordWrapMode(QTextOption.WrapAnywhere)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(PREVIEW_MIN_HEIGHT)

        # Debounce timer: only recalculate after resize events settle
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(30)
        self._resize_timer.timeout.connect(self._adjust_height)

        self.document().contentsChanged.connect(self._schedule_adjust)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Debounce rapid resize events (e.g. splitter drag) so height is
        # recalculated only once the viewport width has stabilised.
        self._resize_timer.start()

    def _schedule_adjust(self) -> None:
        QTimer.singleShot(0, self._adjust_height)

    def _adjust_height(self) -> None:
        vw = self.viewport().width()
        if vw <= 0:
            return
        doc = self.document()
        doc.setTextWidth(vw)

        last_block = doc.lastBlock()
        layout = doc.documentLayout()
        if not last_block.text() and last_block.blockNumber() > 0:
            # blockBoundingRect(last_block).top() == topMargin + all-content-height.
            # Using this as the viewport target means the trailing empty block never
            # enters the visible area. We add one documentMargin as bottom padding
            # so content doesn't appear flush against the frame border.
            h = int(layout.blockBoundingRect(last_block).top()) + int(doc.documentMargin())
        else:
            h = int(doc.size().height())

        # Measure the actual frame/border overhead dynamically so the formula is
        # correct across platforms and themes, without a hard-coded fudge factor.
        overhead = self.height() - self.viewport().height()
        if overhead <= 0:
            overhead = self.frameWidth() * 2

        self.setFixedHeight(max(PREVIEW_MIN_HEIGHT, h + overhead))


class _HistoryLineEdit(QLineEdit):
    """QLineEdit with Up/Down arrow command history."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._history: list[str] = []
        self._idx = -1
        self._draft = ""

    def add_to_history(self, cmd: str) -> None:
        if cmd and (not self._history or self._history[-1] != cmd):
            self._history.append(cmd)
            if len(self._history) > MAX_SSH_HISTORY:
                self._history = self._history[-MAX_SSH_HISTORY:]
        self._idx = -1
        self._draft = ""

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Up and self._history:
            if self._idx < len(self._history) - 1:
                if self._idx == -1:
                    self._draft = self.text()
                self._idx += 1
                self.setText(self._history[-(self._idx + 1)])
            return
        if event.key() == Qt.Key_Down:
            if self._idx > 0:
                self._idx -= 1
                self.setText(self._history[-(self._idx + 1)])
            elif self._idx == 0:
                self._idx = -1
                self.setText(self._draft)
            return
        super().keyPressEvent(event)


class LogPanel(QWidget):
    ssh_command_submitted = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumWidth(440)
        self._lines: list[str] = []
        self._pending: list[str] = []
        self._pending_partial: str = ""
        self._displayed_partial: str = ""
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
        self.output_edit.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.output_edit.setMaximumBlockCount(MAX_BLOCK_COUNT)

        root.addWidget(self.output_edit, 1)

        self._build_ssh_input(root, fixed_font)

    def _build_command_preview(self, parent_layout: QVBoxLayout) -> None:
        preview_label = QLabel("命令预览")
        self.preview_edit = _AutoHeightTextEdit()
        parent_layout.addWidget(preview_label)
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

    def _build_ssh_input(self, parent_layout: QVBoxLayout, font) -> None:
        self.ssh_input_row = QWidget()
        row = QHBoxLayout(self.ssh_input_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        prompt = QLabel("$")
        prompt.setFont(font)
        self.ssh_input = _HistoryLineEdit()
        self.ssh_input.setFont(font)
        self.ssh_input.setPlaceholderText("输入命令，回车发送…")
        self.ssh_input.returnPressed.connect(self._submit_ssh_command)

        row.addWidget(prompt)
        row.addWidget(self.ssh_input, 1)

        self.ssh_input_row.setVisible(False)
        parent_layout.addWidget(self.ssh_input_row)

    def _submit_ssh_command(self) -> None:
        cmd = self.ssh_input.text().strip()
        if not cmd:
            return
        self.ssh_input.add_to_history(cmd)
        self.ssh_input.clear()
        self.ssh_command_submitted.emit(cmd)

    def set_ssh_mode(self, is_ssh: bool, connected: bool) -> None:
        """Control log panel state for SSH mode."""
        self.ssh_input_row.setVisible(is_ssh)
        if is_ssh:
            self.output_edit.setEnabled(connected)
            self.ssh_input.setEnabled(connected)
            self.filter_edit.setEnabled(connected)
            self.clear_btn.setEnabled(connected)
            self.save_btn.setEnabled(connected)
            self.auto_scroll_check.setEnabled(connected)
        else:
            self.output_edit.setEnabled(True)
            self.filter_edit.setEnabled(True)
            self.clear_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.auto_scroll_check.setEnabled(True)

    def set_ssh_input_busy(self, busy: bool) -> None:
        """Disable input while a command is executing."""
        self.ssh_input.setEnabled(not busy)

    def set_command_preview(self, text: str) -> None:
        self.preview_edit.setPlainText(text)

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
