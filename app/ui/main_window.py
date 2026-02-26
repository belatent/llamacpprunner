from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QPainter
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
from app.core.config_schema import LlamaConfig, MODE_LOCAL, MODE_SSH
from app.core.process_runner import ProcessRunner
from app.core.profile_store import ProfileStore
from app.core.resource_monitor import (
    ResourceMonitor,
    ResourceSnapshot,
    STATUS_DEPENDENCY_MISSING,
)
from app.core.ssh_client import SSHConnection
from app.core.ssh_process_runner import SSHProcessRunner
from app.ui.dialogs.install_deps_dialog import InstallDepsDialog
from app.ui.panels.log_panel import LogPanel
from app.ui.panels.params_panel import ParamsPanel
from app.ui.theme_manager import ThemeManager


class _SSHCmdWorker(QObject):
    """Executes a single SSH command and emits output."""

    output = Signal(str)
    finished = Signal()

    def __init__(self, conn: SSHConnection, cmd: str) -> None:
        super().__init__()
        self._conn = conn
        self._cmd = cmd

    def run(self) -> None:
        try:
            exit_code, stdout, stderr = self._conn.exec_command(self._cmd)
            if stdout.strip():
                self.output.emit(stdout.rstrip("\n"))
            if stderr.strip():
                self.output.emit(stderr.rstrip("\n"))
            if exit_code != 0:
                self.output.emit(f"(退出码: {exit_code})")
        except Exception as e:
            self.output.emit(f"错误: {e}")
        finally:
            self.finished.emit()


class _OverlayMask(QWidget):
    """Semi-transparent overlay that blocks mouse input on the left panel."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setStyleSheet("background: transparent;")
        self.hide()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(180, 180, 180, 84))  # ~33% gray
        painter.end()


class MainWindow(QMainWindow):
    def __init__(self, app_root: str | Path, theme_mgr: ThemeManager) -> None:
        super().__init__()
        self.app_root = Path(app_root)
        self.profile_store = ProfileStore(self.app_root)
        self.runner = ProcessRunner()
        self.ssh_runner = SSHProcessRunner()
        self.resource_monitor = ResourceMonitor(self)
        self.theme_mgr = theme_mgr
        self._ssh_cmd_thread: QThread | None = None
        self._ssh_cmd_worker: _SSHCmdWorker | None = None
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

        self.left_container = QWidget()
        self.left_container.setMinimumWidth(720)
        left_layout = QVBoxLayout(self.left_container)
        left_layout.setContentsMargins(8, 8, 4, 8)
        left_layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.params_panel)

        left_layout.addWidget(scroll, 1)

        self._overlay = _OverlayMask(self.left_container)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.left_container)
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

        self._resource_container = QWidget()
        self._resource_container.setObjectName("resourceContainer")
        res_layout = QHBoxLayout(self._resource_container)
        res_layout.setContentsMargins(0, 0, 0, 0)
        res_layout.setSpacing(4)

        self.resource_label = QLabel()
        self.resource_label.setObjectName("resourceLabel")

        self.install_link = QLabel()
        self.install_link.setObjectName("installLink")
        self.install_link.setTextFormat(Qt.RichText)
        self.install_link.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.install_link.setOpenExternalLinks(False)
        self.install_link.linkActivated.connect(self._on_install_deps_clicked)
        self.install_link.hide()

        res_layout.addStretch()
        res_layout.addWidget(self.resource_label)
        res_layout.addWidget(self.install_link)
        res_layout.addStretch()
        self._resource_container.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )

        self.mode_hint_label = QLabel()
        self.mode_hint_label.setObjectName("modeHintLabel")

        status.addWidget(self.status_indicator)
        status.addWidget(self.status_text)
        status.addWidget(self._resource_container, 1)
        status.addPermanentWidget(self.mode_hint_label)
        self.setStatusBar(status)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Start/Stop toggle button — leftmost, double width, green/red
        self.start_stop_btn = QPushButton("启动")
        self.start_stop_btn.setObjectName("startStopBtn")
        self.start_stop_btn.setProperty("running", False)
        self.start_stop_btn.setProperty("ssh_mode", False)

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
        self.params_panel.mode_changed.connect(self._on_mode_changed)
        self.params_panel.ssh_connection_changed.connect(self._on_ssh_connection_changed)
        self.log_panel.ssh_command_submitted.connect(self._on_ssh_command)
        self.load_btn.clicked.connect(self._load_profile_dialog)
        self.save_btn.clicked.connect(self._save_profile_dialog)
        self.save_as_btn.clicked.connect(self._save_as_profile_dialog)
        self.start_stop_btn.clicked.connect(self._toggle_process)
        self.profile_combo.currentTextChanged.connect(self._load_profile_by_name)
        self.theme_toggle_btn.clicked.connect(self._toggle_theme)
        self.config_dir_edit.editingFinished.connect(self._on_config_dir_changed)
        self.config_dir_browse_btn.clicked.connect(self._browse_config_dir)

        for r in (self.runner, self.ssh_runner):
            r.line_received.connect(self.log_panel.append_line)
            r.line_received.connect(self._on_line_received)
            r.partial_received.connect(self.log_panel.update_partial)
            r.state_changed.connect(self._on_runner_state_changed)
            r.process_started.connect(self._on_process_started)
            r.process_stopped.connect(self._on_process_stopped)

        self._update_mode_hint()

        self.resource_monitor.snapshot_ready.connect(self._on_resource_snapshot)

    def _start_resource_monitor(self) -> None:
        mode = self.params_panel.mode_combo.currentText()
        if mode == MODE_SSH:
            conn = self.params_panel.get_ssh_connection()
            if not conn or not conn.is_connected:
                self.resource_label.setText("等待连接...")
                self.install_link.hide()
                return
            self.resource_monitor.start("ssh", conn)
        else:
            self.resource_monitor.start("local")

    def _restart_resource_monitor(self) -> None:
        self.resource_monitor.stop()
        self.resource_label.setText("")
        self.install_link.hide()
        self._start_resource_monitor()

    def _on_resource_snapshot(self, snap: ResourceSnapshot) -> None:
        is_ssh = self.params_panel.mode_combo.currentText() == MODE_SSH

        if snap.status == STATUS_DEPENDENCY_MISSING and is_ssh:
            self.resource_monitor.stop()
            self.resource_label.setText("远程设备依赖缺失，无法显示数据……")
            self.install_link.setText(
                '<a href="install" style="color: #6366f1; '
                'text-decoration: underline;">安装依赖</a>'
            )
            self.install_link.show()
            return

        self.install_link.hide()
        prefix = "远程" if is_ssh else "本地"
        parts: list[str] = [prefix]
        parts.append(f"CPU: {snap.cpu_freq_mhz}MHz/{snap.cpu_percent:.0f}%")
        if snap.mem_total_mb:
            parts.append(
                f"RAM: {snap.mem_used_mb / 1024:.1f}/{snap.mem_total_mb / 1024:.1f}GB"
            )
        for g in snap.gpus:
            gpu_text = f"GPU {g.index}({g.short_name}): {g.mem_used_mb / 1024:.1f}GB"
            if g.power_w > 0:
                gpu_text += f"/{g.power_w:.0f}W"
            parts.append(gpu_text)
        self.resource_label.setText(" | ".join(parts))

    def _on_install_deps_clicked(self, _link: str) -> None:
        conn = self.params_panel.get_ssh_connection()
        if not conn or not conn.is_connected:
            return
        dialog = InstallDepsDialog(conn, self)
        dialog.exec()
        self._restart_resource_monitor()

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
        config = self._current_config()
        if config.mode == MODE_SSH:
            self.start_stop_btn.setProperty("ssh_mode", True)
            self.start_stop_btn.style().unpolish(self.start_stop_btn)
            self.start_stop_btn.style().polish(self.start_stop_btn)
        self._update_mode_hint()
        self._update_start_btn_enabled()
        self._update_log_panel_ssh_state()
        self._restart_resource_monitor()

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

    def _on_mode_changed(self, mode: str) -> None:
        self._update_mode_hint()
        self.log_panel.clear()
        is_ssh = mode == MODE_SSH
        self.start_stop_btn.setProperty("ssh_mode", is_ssh)
        self.start_stop_btn.style().unpolish(self.start_stop_btn)
        self.start_stop_btn.style().polish(self.start_stop_btn)
        self._update_start_btn_enabled()
        self._update_log_panel_ssh_state()
        self._restart_resource_monitor()

    def _on_ssh_connection_changed(self, connected: bool) -> None:
        self._update_mode_hint()
        self._update_start_btn_enabled()
        self._update_log_panel_ssh_state()
        self._restart_resource_monitor()

    def _update_start_btn_enabled(self) -> None:
        """Disable start button in SSH mode when not connected."""
        if self.runner.is_running() or self.ssh_runner.is_running():
            return
        is_ssh = self.params_panel.mode_combo.currentText() == MODE_SSH
        if is_ssh:
            conn = self.params_panel.get_ssh_connection()
            self.start_stop_btn.setEnabled(conn is not None)
        else:
            self.start_stop_btn.setEnabled(True)

    def _update_log_panel_ssh_state(self) -> None:
        is_ssh = self.params_panel.mode_combo.currentText() == MODE_SSH
        connected = False
        if is_ssh:
            conn = self.params_panel.get_ssh_connection()
            connected = conn is not None
        self.log_panel.set_ssh_mode(is_ssh, connected)

    def _on_ssh_command(self, cmd: str) -> None:
        conn = self.params_panel.get_ssh_connection()
        if not conn:
            return
        self.log_panel.append_line(f"$ {cmd}")
        self.log_panel.set_ssh_input_busy(True)

        worker = _SSHCmdWorker(conn, cmd)
        thread = QThread()
        worker.moveToThread(thread)

        worker.output.connect(lambda text: self.log_panel.append_line(text))
        worker.finished.connect(lambda: self._on_ssh_cmd_finished(thread, worker))
        worker.finished.connect(thread.quit, Qt.DirectConnection)

        thread.started.connect(worker.run)
        thread.start()
        self._ssh_cmd_thread = thread
        self._ssh_cmd_worker = worker

    def _on_ssh_cmd_finished(self, thread: QThread, worker: _SSHCmdWorker) -> None:
        self.log_panel.set_ssh_input_busy(False)
        thread.quit()
        thread.wait()
        thread.deleteLater()
        if self._ssh_cmd_thread is thread:
            self._ssh_cmd_thread = None
            self._ssh_cmd_worker = None

    def _update_mode_hint(self) -> None:
        if self.params_panel.mode_combo.currentText() == MODE_SSH:
            conn = self.params_panel.get_ssh_connection()
            if conn:
                self.mode_hint_label.setText(f"SSH连接成功： {conn.get_connection_string()}")
            else:
                self.mode_hint_label.setText("SSH模式")
        else:
            self.mode_hint_label.setText("本地模式")

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
        if self.runner.is_running() or self.ssh_runner.is_running():
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
            if config.mode == MODE_SSH:
                conn = self.params_panel.get_ssh_connection()
                if not conn:
                    QMessageBox.warning(self, "启动失败", "请先连接 SSH。")
                    return
                self.ssh_runner.start(conn, command, cwd=config.llama_dir or None)
            else:
                self.runner.start(command, cwd=config.llama_dir or None)
        except (ValueError, RuntimeError, OSError) as exc:
            QMessageBox.critical(self, "启动失败", str(exc))

    def _stop_process(self) -> None:
        if self.ssh_runner.is_running():
            self.ssh_runner.stop()
        if self.runner.is_running():
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
            if config.mode == MODE_SSH:
                self._webui_url = f"http://{config.ssh_host}:{config.port}"
            else:
                self._webui_url = f"http://{config.host}:{config.port}"
        else:
            self._webui_url = ""
            self.open_webui_btn.setEnabled(False)

    def _set_start_stop_running(self, running: bool) -> None:
        self.start_stop_btn.setProperty("running", running)
        self.start_stop_btn.setText("停止" if running else "启动")
        if running:
            self.start_stop_btn.setEnabled(True)
        # Force QSS to re-evaluate the dynamic property
        self.start_stop_btn.style().unpolish(self.start_stop_btn)
        self.start_stop_btn.style().polish(self.start_stop_btn)
        # Show/hide overlay mask on left panel
        self._set_left_panel_locked(running)
        if not running:
            self._update_start_btn_enabled()

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

    def _set_left_panel_locked(self, locked: bool) -> None:
        if locked:
            self._overlay.setGeometry(self.left_container.rect())
            self._overlay.raise_()
            self._overlay.show()
        else:
            self._overlay.hide()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._overlay.isVisible():
            self._overlay.setGeometry(self.left_container.rect())

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_state()
        self.resource_monitor.stop()
        if self.ssh_runner.is_running():
            self.ssh_runner.stop()
        if self.runner.is_running():
            self.runner.stop(force_after_ms=500)
        self.params_panel.disconnect_ssh()
        super().closeEvent(event)
