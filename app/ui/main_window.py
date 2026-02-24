from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.core.command_builder import build_command, build_command_preview
from app.core.config_schema import LlamaConfig
from app.core.process_runner import ProcessRunner
from app.core.profile_store import ProfileStore
from app.ui.panels.log_panel import LogPanel
from app.ui.panels.params_panel import ParamsPanel
from app.ui.theme_manager import ThemeManager


class MainWindow(QMainWindow):
    def __init__(self, app_root: str | Path, theme_mgr: ThemeManager) -> None:
        super().__init__()
        self.app_root = Path(app_root)
        self.profile_store = ProfileStore(self.app_root)
        self.runner = ProcessRunner()
        self.theme_mgr = theme_mgr
        self.setWindowTitle("llama.cpp 启动器")
        self.resize(1280, 640)
        self.setMinimumSize(1280, 640)

        self._build_ui()
        self._bind_events()
        self._restore_state()

    def _build_ui(self) -> None:
        self._build_toolbar()

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        self.params_panel = ParamsPanel()
        self.log_panel = LogPanel()

        left_container = QWidget()
        left_container.setMinimumWidth(720)
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(8, 8, 4, 8)
        left_layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.params_panel)

        left_layout.addWidget(scroll, 1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_container)
        splitter.addWidget(self.log_panel)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 45)
        splitter.setStretchFactor(1, 55)
        splitter.setSizes([720, 640])
        root.addWidget(splitter, 1)

        self.setCentralWidget(central)

        status = QStatusBar()
        self.status_indicator = QLabel()
        self.status_indicator.setFixedSize(10, 10)
        self._set_status_indicator("stopped")
        self.status_text = QLabel("就绪")
        status.addWidget(self.status_indicator)
        status.addWidget(self.status_text)
        self.setStatusBar(status)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Start/Stop toggle button — leftmost, double width, green/red
        self.start_stop_btn = QPushButton("启动")
        self.start_stop_btn.setObjectName("startStopBtn")
        self.start_stop_btn.setProperty("running", False)

        self.open_webui_btn = QPushButton("打开WebUI")
        self.open_webui_btn.setEnabled(False)
        self._webui_url: str = ""
        self.open_webui_btn.clicked.connect(self._open_webui)

        self.load_btn = QPushButton("打开...")
        self.save_btn = QPushButton("保存")
        self.save_as_btn = QPushButton("另存为")

        self.profile_combo = QComboBox()
        self.profile_combo.setEditable(False)
        self.profile_combo.setMinimumWidth(200)
        self.theme_toggle_btn = QPushButton("Light")
        self.theme_toggle_btn.setObjectName("themeToggleBtn")
        self.theme_toggle_btn.setFixedWidth(60)
        self.theme_toggle_btn.setToolTip("切换明暗主题")

        self.config_dir_edit = QLineEdit()
        self.config_dir_edit.setMinimumWidth(180)
        self.config_dir_edit.setMaximumWidth(300)
        self.config_dir_edit.setPlaceholderText("配置文件目录")
        self.config_dir_edit.setToolTip("配置文件的保存/加载目录（支持子文件夹扫描）")
        self.config_dir_browse_btn = QPushButton("浏览")
        self.config_dir_browse_btn.setFixedWidth(52)

        toolbar.addWidget(self.start_stop_btn)
        toolbar.addWidget(self.open_webui_btn)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("配置方案"))
        toolbar.addWidget(self.profile_combo)
        toolbar.addSeparator()
        toolbar.addWidget(self.load_btn)
        toolbar.addWidget(self.save_btn)
        toolbar.addWidget(self.save_as_btn)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("配置目录"))
        toolbar.addWidget(self.config_dir_edit)
        toolbar.addWidget(self.config_dir_browse_btn)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        toolbar.addWidget(self.theme_toggle_btn)

    def _bind_events(self) -> None:
        self.params_panel.config_changed.connect(self._on_config_changed)
        self.load_btn.clicked.connect(self._load_profile_dialog)
        self.save_btn.clicked.connect(self._save_profile_dialog)
        self.save_as_btn.clicked.connect(self._save_as_profile_dialog)
        self.start_stop_btn.clicked.connect(self._toggle_process)
        self.profile_combo.currentTextChanged.connect(self._load_profile_by_name)
        self.theme_toggle_btn.clicked.connect(self._toggle_theme)
        self.config_dir_edit.editingFinished.connect(self._on_config_dir_changed)
        self.config_dir_browse_btn.clicked.connect(self._browse_config_dir)

        self.runner.line_received.connect(self.log_panel.append_line)
        self.runner.line_received.connect(self._on_line_received)
        self.runner.partial_received.connect(self.log_panel.update_partial)
        self.runner.state_changed.connect(self._on_runner_state_changed)
        self.runner.process_started.connect(self._on_process_started)
        self.runner.process_stopped.connect(self._on_process_stopped)

    def _restore_state(self) -> None:
        state = self.profile_store.load_state()
        if "window_geometry" in state:
            self.restoreGeometry(bytes.fromhex(state["window_geometry"]))
        saved_theme = state.get("theme", ThemeManager.DARK)
        self.theme_mgr.apply(saved_theme)
        self._sync_theme_button()
        if "config_dir" in state:
            config_dir = state["config_dir"]
            p = Path(config_dir)
            if p.is_dir():
                self.config_dir_edit.setText(config_dir)
                self.profile_store.profiles_dir = p
        self._refresh_profiles()
        if "active_profile" in state:
            self.profile_combo.setCurrentText(state["active_profile"])
            self._load_profile_by_name(state["active_profile"])
        else:
            self.params_panel.from_config(LlamaConfig())
        self._on_config_changed(self._current_config())

    def _save_state(self) -> None:
        self.profile_store.save_state(
            {
                "active_profile": self.profile_combo.currentText().strip(),
                "window_geometry": self.saveGeometry().toHex().data().decode("ascii"),
                "theme": self.theme_mgr.current,
                "config_dir": self.config_dir_edit.text().strip(),
            }
        )

    def _current_config(self) -> LlamaConfig:
        return self.params_panel.to_config()

    def _on_config_changed(self, config: LlamaConfig) -> None:
        self.log_panel.set_command_preview(build_command_preview(config))

    def _refresh_profiles(self) -> None:
        current = self.profile_combo.currentText().strip()
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItems(self.profile_store.list_profiles())
        self.profile_combo.setCurrentText(current)
        self.profile_combo.blockSignals(False)

    def _new_profile_dialog(self) -> None:
        name, ok = QInputDialog.getText(self, "新建配置方案", "请输入配置方案名称：")
        if not ok or not name.strip():
            return
        name = name.strip()
        self.profile_store.save_profile(name, self._current_config())
        self._refresh_profiles()
        self.profile_combo.setCurrentText(name)
        self.status_text.setText(f"配置已保存：{name}")

    def _on_config_dir_changed(self) -> None:
        path_str = self.config_dir_edit.text().strip()
        if not path_str:
            return
        p = Path(path_str)
        if p.is_dir():
            self.profile_store.profiles_dir = p
            self._refresh_profiles()
        else:
            self.status_text.setText(f"目录不存在：{path_str}")

    def _browse_config_dir(self) -> None:
        current = self.config_dir_edit.text().strip() or str(self.profile_store.profiles_dir)
        chosen = QFileDialog.getExistingDirectory(self, "选择配置文件目录", current)
        if not chosen:
            return
        self.config_dir_edit.setText(chosen)
        self.profile_store.profiles_dir = Path(chosen)
        self._refresh_profiles()
        self.status_text.setText(f"配置目录已切换：{chosen}")

    def _load_profile_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "加载配置", str(self.profile_store.profiles_dir), "JSON Files (*.json)"
        )
        if not path:
            return
        name = Path(path).stem
        self._load_profile_by_name(name)
        self.profile_combo.setCurrentText(name)

    def _save_profile_dialog(self) -> None:
        profile_name = self.profile_combo.currentText().strip()
        if not profile_name:
            name, ok = QInputDialog.getText(self, "保存配置方案", "请输入配置方案名称：")
            if not ok or not name.strip():
                return
            profile_name = name.strip()
        self.profile_store.save_profile(profile_name, self._current_config())
        self._refresh_profiles()
        self.profile_combo.setCurrentText(profile_name)
        self.status_text.setText(f"配置已保存：{profile_name}")

    def _save_as_profile_dialog(self) -> None:
        current_name = self.profile_combo.currentText().strip()
        suggested = f"{current_name} 副本" if current_name else ""
        name, ok = QInputDialog.getText(self, "另存为", "请输入新配置方案名称：", text=suggested)
        if not ok or not name.strip():
            return
        name = name.strip()
        self.profile_store.save_profile(name, self._current_config())
        self._refresh_profiles()
        self.profile_combo.setCurrentText(name)
        self.status_text.setText(f"配置已另存为：{name}")

    def _load_profile_by_name(self, profile_name: str) -> None:
        if not profile_name:
            return
        try:
            config = self.profile_store.load_profile(profile_name)
        except FileNotFoundError:
            return
        self.params_panel.from_config(config)
        self.status_text.setText(f"已加载配置：{profile_name}")

    def _toggle_process(self) -> None:
        if self.runner.is_running():
            self._stop_process()
        else:
            self._start_process()

    def _start_process(self) -> None:
        config = self._current_config()
        errors = config.validate()
        if errors:
            QMessageBox.warning(self, "参数校验失败", "\n".join(errors))
            return
        command = build_command(config)
        try:
            self.runner.start(command, cwd=config.llama_dir or None)
        except (ValueError, RuntimeError, OSError) as exc:
            QMessageBox.critical(self, "启动失败", str(exc))

    def _stop_process(self) -> None:
        self.runner.stop()

    def _on_runner_state_changed(self, state: str) -> None:
        state_map = {
            "starting": "正在启动...",
            "running": "运行中",
            "stopping": "正在停止...",
            "stopped": "已停止",
        }
        self.status_text.setText(state_map.get(state, state))
        self._set_status_indicator(state)

        running = state in {"starting", "running", "stopping"}
        self._set_start_stop_running(running)

        if state == "running":
            config = self._current_config()
            self._webui_url = f"http://{config.host}:{config.port}"
        else:
            self._webui_url = ""
            self.open_webui_btn.setEnabled(False)

    def _set_start_stop_running(self, running: bool) -> None:
        self.start_stop_btn.setProperty("running", running)
        self.start_stop_btn.setText("停止" if running else "启动")
        # Force QSS to re-evaluate the dynamic property
        self.start_stop_btn.style().unpolish(self.start_stop_btn)
        self.start_stop_btn.style().polish(self.start_stop_btn)

    def _set_status_indicator(self, state: str) -> None:
        colors = {
            "running": "#22c55e",
            "starting": "#eab308",
            "stopping": "#f97316",
            "stopped": "#64748b",
        }
        color = colors.get(state, "#64748b")
        self.status_indicator.setStyleSheet(
            f"background: {color}; border-radius: 5px; border: none;"
        )

    def _on_process_started(self, command: str) -> None:
        self.log_panel.append_line(f"启动命令: {command}")

    def _on_line_received(self, line: str, is_error: bool) -> None:
        if not self.open_webui_btn.isEnabled() and self._webui_url:
            if "main: starting the main loop" in line:
                self.open_webui_btn.setEnabled(True)

    def _on_process_stopped(self, exit_code: int, elapsed_seconds: float) -> None:
        self.log_panel.append_line(f"进程退出，exit={exit_code}, elapsed={elapsed_seconds:.2f}s")

    def _open_webui(self) -> None:
        if self._webui_url:
            QDesktopServices.openUrl(QUrl(self._webui_url))

    def _toggle_theme(self) -> None:
        self.theme_mgr.toggle()
        self._sync_theme_button()

    def _sync_theme_button(self) -> None:
        if self.theme_mgr.current == ThemeManager.DARK:
            self.theme_toggle_btn.setText("Light")
            self.theme_toggle_btn.setToolTip("切换到灰白主题")
        else:
            self.theme_toggle_btn.setText("Dark")
            self.theme_toggle_btn.setToolTip("切换到暗黑主题")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_state()
        if self.runner.is_running():
            self.runner.stop(force_after_ms=500)
        super().closeEvent(event)
