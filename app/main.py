from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow
from app.ui.theme_manager import ThemeManager


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    app = QApplication(sys.argv)
    app.setApplicationName("llama.cpp 启动器")

    theme_mgr = ThemeManager(root / "app" / "ui")
    theme_mgr.apply(ThemeManager.DARK)

    window = MainWindow(root, theme_mgr)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
