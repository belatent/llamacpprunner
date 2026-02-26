from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow
from app.ui.theme_manager import ThemeManager


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    app = QApplication(sys.argv)
    app.setApplicationName("llama.cpp 启动器")

    # 设置默认字体，避免 QComboBox 等控件因样式表 font-size 导致 point size -1 告警
    default_font = QFont("Segoe UI", 10)
    default_font.setStyleHint(QFont.SansSerif)
    app.setFont(default_font)

    theme_mgr = ThemeManager(root / "app" / "ui")
    theme_mgr.apply(ThemeManager.DARK)

    window = MainWindow(root, theme_mgr)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
