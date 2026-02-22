from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication


class ThemeManager(QObject):
    theme_changed = Signal(str)

    DARK = "dark"
    LIGHT = "light"

    def __init__(self, ui_dir: Path) -> None:
        super().__init__()
        self._ui_dir = ui_dir
        self._current = self.DARK

    @property
    def current(self) -> str:
        return self._current

    def apply(self, theme: str) -> None:
        if theme not in (self.DARK, self.LIGHT):
            theme = self.DARK
        qss = self._load_qss(theme)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(qss)
        self._current = theme
        self.theme_changed.emit(theme)

    def toggle(self) -> str:
        next_theme = self.LIGHT if self._current == self.DARK else self.DARK
        self.apply(next_theme)
        return next_theme

    def _load_qss(self, theme: str) -> str:
        path = self._ui_dir / f"theme_{theme}.qss"
        if not path.exists():
            return ""
        qss = path.read_text(encoding="utf-8")
        icons_dir = str(self._ui_dir / "icons").replace("\\", "/")
        return qss.replace("{{ICONS}}", icons_dir)
