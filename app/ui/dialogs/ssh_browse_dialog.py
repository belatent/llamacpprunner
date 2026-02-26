"""SSH remote directory browser dialog."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.ssh_client import SSHConnection


class SSHBrowseDialog(QDialog):
    """Dialog to browse and select a directory on remote SSH server."""

    def __init__(
        self,
        parent: QWidget | None,
        ssh: SSHConnection,
        initial_path: str = "/",
        pick_dir: bool = True,
    ) -> None:
        super().__init__(parent)
        self.ssh = ssh
        self._current_path = initial_path.rstrip("/") or "/"
        self._pick_dir = pick_dir
        self._selected_path: str = ""
        self.setWindowTitle("SSH 浏览远程目录")
        self.setMinimumSize(500, 400)
        self._build_ui()
        self._refresh_list()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        path_row = QHBoxLayout()

        self.back_btn = QPushButton("←")
        self.back_btn.setFixedSize(30, 28)
        self.back_btn.setToolTip("返回上一级")
        self.back_btn.setCursor(Qt.PointingHandCursor)
        self.back_btn.clicked.connect(self._go_up)
        path_row.addWidget(self.back_btn)

        path_row.addWidget(QLabel("当前路径:"))
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(False)
        self.path_edit.setText(self._current_path)
        self.path_edit.returnPressed.connect(self._on_path_edited)
        path_row.addWidget(self.path_edit, 1)

        go_btn = QPushButton("前往")
        go_btn.clicked.connect(self._on_path_edited)
        path_row.addWidget(go_btn)
        layout.addLayout(path_row)

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        if self._pick_dir:
            select_btn = QPushButton("选择当前目录")
            select_btn.clicked.connect(self._select_current)
            btn_row.addWidget(select_btn)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _go_up(self) -> None:
        """Navigate to parent directory."""
        if self._current_path == "/":
            return
        parts = [p for p in self._current_path.rstrip("/").split("/") if p]
        if len(parts) > 1:
            self._current_path = "/" + "/".join(parts[:-1])
        else:
            self._current_path = "/"
        self._refresh_list()

    def _refresh_list(self) -> None:
        self.list_widget.clear()
        self.path_edit.setText(self._current_path)
        self.back_btn.setEnabled(self._current_path != "/")

        if not self.ssh.is_connected:
            self.list_widget.addItem("（SSH 未连接）")
            return

        try:
            items = self.ssh.list_dir(self._current_path)
        except Exception as e:
            self.list_widget.addItem(f"（错误: {e}）")
            return

        for name, is_dir in items:
            if is_dir:
                item = QListWidgetItem(f"📁  {name}")
                item.setData(Qt.ItemDataRole.UserRole, (name, True))
            else:
                item = QListWidgetItem(f"    {name}")
                item.setData(Qt.ItemDataRole.UserRole, (name, False))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setForeground(QColor(150, 150, 150))
            self.list_widget.addItem(item)

    def _on_path_edited(self) -> None:
        path = self.path_edit.text().strip() or "/"
        path = "/" + path.lstrip("/")
        self._current_path = path
        self._refresh_list()

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(data, tuple):
            return
        name, is_dir = data
        if is_dir:
            new_path = f"{self._current_path.rstrip('/')}/{name}"
            self._current_path = new_path
            self._refresh_list()

    def _select_current(self) -> None:
        self._selected_path = self._current_path
        self.accept()

    def selected_path(self) -> str:
        """Return the selected directory path."""
        return self._selected_path
