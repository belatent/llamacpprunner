from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSize, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core.config_schema import LlamaConfig, MODE_LOCAL, MODE_SSH
from app.core.ssh_client import SSHConnection

CTX_STEPS = [-1, 512, 8192, 16384, 32768, 65536, 131072]
CTX_DEFAULT_IDX = 2  # 8192


def _detect_gpus() -> list[str]:
    """Return list of GPU names via nvidia-smi, or empty list on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
    except Exception:
        pass
    return []


class _LazyGpuCombo(QComboBox):
    """ComboBox that auto-detects GPUs on first popup open."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._detected = False
        self._ssh_getter = None  # callable returning SSHConnection | None
        self._ssh_mode = False
        self.addItem("Auto")

    def set_ssh_getter(self, getter) -> None:
        """Set a callable that returns an active SSHConnection or None."""
        self._ssh_getter = getter

    def set_ssh_mode(self, is_ssh: bool) -> None:
        """Set whether we are in SSH mode (affects GPU detection source)."""
        self._ssh_mode = is_ssh

    def reset_detection(self) -> None:
        """Force re-detection next time popup opens."""
        self._detected = False

    def showPopup(self) -> None:
        if not self._detected:
            self._detected = True
            self._detect_and_populate()
        super().showPopup()

    def _detect_and_populate(self) -> None:
        current = self.currentText()
        gpus: list[str] = []
        ssh_conn = self._ssh_getter() if self._ssh_getter else None
        if self._ssh_mode:
            if ssh_conn and ssh_conn.is_connected:
                try:
                    _, out, _ = ssh_conn.exec_command(
                        "nvidia-smi --query-gpu=name --format=csv,noheader", timeout=5
                    )
                    gpus = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
                except Exception:
                    gpus = []
        else:
            gpus = _detect_gpus()
        self.blockSignals(True)
        self.clear()
        self.addItem("Auto")
        for i, name in enumerate(gpus):
            self.addItem(f"{i}: {name}")
        restored = False
        for idx in range(self.count()):
            text = self.itemText(idx)
            if text == current or text.split(":")[0].strip() == current:
                self.setCurrentIndex(idx)
                restored = True
                break
        if not restored:
            self.setCurrentIndex(0)
        self.blockSignals(False)


class _NoScrollFilter(QObject):
    """Blocks wheel events on input widgets to prevent accidental changes while scrolling."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Wheel:
            if isinstance(obj, (QAbstractSpinBox, QComboBox)):
                event.ignore()
                return True
        return False


_wheel_filter = _NoScrollFilter()


class _VersionFetchWorker(QObject):
    """Fetches the latest GitHub release tag in a background thread."""

    finished = Signal(str)

    def run(self) -> None:
        try:
            from app.core.updater import fetch_releases_page
            releases = fetch_releases_page(1, 1)
            tag = releases[0].tag_name if releases else ""
        except Exception:
            tag = ""
        self.finished.emit(tag)


class _CollapsibleBox(QGroupBox):
    """A QGroupBox that always shows its content (collapsibility removed)."""

    def __init__(self, title: str, initially_collapsed: bool = False, parent=None) -> None:
        super().__init__(title, parent)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(6)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 4)
        outer.setSpacing(0)
        outer.addWidget(self._content)

    def add_widget(self, widget: QWidget) -> None:
        self._content_layout.addWidget(widget)

    def add_layout(self, layout) -> None:
        self._content_layout.addLayout(layout)

    @property
    def content_layout(self) -> QVBoxLayout:
        return self._content_layout


class _FlowLayout(QLayout):
    """Arranges child widgets left-to-right, wrapping to the next row as needed."""

    def __init__(self, parent=None, h_spacing: int = 6, v_spacing: int = 4) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing

    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)
        self.invalidate()

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> QLayoutItem | None:
        if 0 <= index < len(self._items):
            item = self._items.pop(index)
            self.invalidate()
            return item
        return None

    def expandingDirections(self):
        return Qt.Orientations(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        m = self.contentsMargins()
        eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y = eff.x(), eff.y()
        line_height = 0

        for item in self._items:
            wid = item.widget()
            if wid is None:
                continue
            # When only testing height, include every item regardless of current
            # visibility: newly added chips may not have been shown yet by Qt's
            # event loop, and skipping them would yield a zero height causing the
            # widget to collapse before the chips ever get a chance to appear.
            if not test_only and not wid.isVisible():
                continue
            hint = item.sizeHint()
            iw, ih = hint.width(), hint.height()
            if iw <= 0 or ih <= 0:
                continue
            next_x = x + iw
            if next_x > eff.right() + 1 and x > eff.x():
                x = eff.x()
                y += line_height + self._v_spacing
                next_x = x + iw
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x + self._h_spacing
            line_height = max(line_height, ih)

        return y + line_height - rect.y() + m.bottom()


class _NodeChip(QWidget):
    """Rounded tag widget: address text + inline delete button."""

    delete_requested = Signal(str)

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self._text = text
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName("RpcNodeChip")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 3, 3, 3)
        lay.setSpacing(2)

        lbl = QLabel(text)
        lbl.setObjectName("RpcNodeChipLabel")

        del_btn = QPushButton("×")
        del_btn.setObjectName("RpcChipDeleteBtn")
        del_btn.setFlat(True)
        del_btn.setFixedSize(20, 20)
        del_btn.setCursor(Qt.PointingHandCursor)
        del_btn.setToolTip(f"删除 {text}")
        del_btn.clicked.connect(lambda: self.delete_requested.emit(self._text))

        lay.addWidget(lbl)
        lay.addWidget(del_btn)

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)


class _RpcNodeList(QWidget):
    """Flow-layout list of RPC node chips.

    Chips are positioned via direct setGeometry() calls in _reposition(),
    bypassing Qt's deferred layout-activation mechanism (activate /
    LayoutRequest / setGeometry on QLayout) which caused chips to appear
    as full-width blocks until a theme switch forced a full recompute.
    """

    changed = Signal()

    _H_SPACING = 6
    _V_SPACING = 4
    _MARGIN = 2

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._chips: list[_NodeChip] = []
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(0)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Chips added while hidden (config load) need one deferred pass so the
        # parent layout has assigned us a real width before we compute positions.
        QTimer.singleShot(0, self._reposition)

    def _reposition(self) -> None:
        w = self.width()
        if w <= 0:
            return
        m = self._MARGIN
        x, y = m, m
        line_h = 0
        for chip in self._chips:
            chip.ensurePolished()
            sh = chip.sizeHint()
            cw, ch = sh.width(), sh.height()
            if cw <= 0 or ch <= 0:
                continue
            if x + cw > w - m and x > m:
                x = m
                y += line_h + self._V_SPACING
                line_h = 0
            chip.setGeometry(x, y, cw, ch)
            x += cw + self._H_SPACING
            line_h = max(line_h, ch)
        h = (y + line_h + m) if self._chips else 0
        if self.height() != h:
            self.setFixedHeight(h)
            self.updateGeometry()
        self.update()

    def add_node(self, text: str) -> None:
        if any(c.property("node_text") == text for c in self._chips):
            return
        chip = _NodeChip(text)
        chip.setProperty("node_text", text)
        chip.delete_requested.connect(self._remove_chip)
        chip.setParent(self)
        chip.ensurePolished()
        chip.show()
        self._chips.append(chip)
        self._reposition()
        self.changed.emit()

    def _remove_chip(self, text: str) -> None:
        for chip in self._chips:
            if chip.property("node_text") == text:
                self._chips.remove(chip)
                chip.hide()
                chip.deleteLater()
                self._reposition()
                self.changed.emit()
                return

    def get_nodes(self) -> list[str]:
        return [c.property("node_text") for c in self._chips]

    def clear_nodes(self) -> None:
        for chip in self._chips:
            chip.hide()
            chip.deleteLater()
        self._chips.clear()
        if self.height() != 0:
            self.setFixedHeight(0)
            self.updateGeometry()

    def set_nodes(self, nodes: list[str]) -> None:
        self.clear_nodes()
        for node in nodes:
            self.add_node(node)


class ParamsPanel(QWidget):
    config_changed = Signal(object)
    mode_changed = Signal(str)  # Emits MODE_LOCAL or MODE_SSH
    ssh_connection_changed = Signal(bool)  # True when connected

    def __init__(self) -> None:
        super().__init__()
        self._github_latest: str = ""
        self._gh_fetch_thread: QThread | None = None
        self._gh_fetch_worker: _VersionFetchWorker | None = None
        self._loaded_cache_ram_mib: int = 0
        # Tracked for enable/disable toggling
        self._draft_row_widget: QWidget | None = None
        self._ngram_enable_row: QWidget | None = None
        self._ngram_group: QGroupBox | None = None
        self._ngram_params_row: QWidget | None = None
        self._draft_model_container: QWidget | None = None
        # Mode-specific path storage: {MODE_LOCAL: {llama_dir, model_dir, model_file}, MODE_SSH: {...}}
        self._mode_paths: dict[str, dict[str, str]] = {
            MODE_LOCAL: {"llama_dir": "", "model_dir": "", "model_file": "", "mmproj_file": ""},
            MODE_SSH: {"llama_dir": "", "model_dir": "", "model_file": "", "mmproj_file": ""},
        }
        self._ssh_connection = None  # SSHConnection when connected
        self._build_ui()
        self.from_config(LlamaConfig())
        self._bind_change_signals()
        self._install_wheel_filters()
        self._fetch_github_latest()

    # ──────────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setMinimumWidth(420)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)
        root.addWidget(self._build_path_group())
        root.addWidget(self._build_service_group())
        root.addStretch(1)
        self._apply_mode_ui_state(self.mode_combo.currentText())

    def _apply_mode_ui_state(self, mode: str) -> None:
        """Update enable/disable state of mode-dependent widgets."""
        is_ssh = mode == MODE_SSH
        self.ssh_ip_edit.setEnabled(is_ssh)
        self.ssh_port_spin.setEnabled(is_ssh)
        self.ssh_username_edit.setEnabled(is_ssh)
        self.ssh_password_edit.setEnabled(is_ssh)
        self.ssh_connect_btn.setEnabled(is_ssh)
        if is_ssh:
            ssh_connected = self._ssh_connection is not None and self._ssh_connection.is_connected
            self.ssh_connect_btn.setText("断开" if ssh_connected else "连接")
        for lbl in self._ssh_labels:
            lbl.setEnabled(is_ssh)
        if is_ssh:
            self.update_llamacpp_btn.setEnabled(False)
            ssh_connected = self._ssh_connection is not None and self._ssh_connection.is_connected
            self.llama_browse_btn.setEnabled(ssh_connected)
            self.model_dir_browse_btn.setEnabled(ssh_connected)
            self.model_file_refresh_btn.setEnabled(ssh_connected)
        else:
            has_dir = bool(self.llama_dir_edit.text().strip())
            self.update_llamacpp_btn.setEnabled(has_dir)
            self.llama_browse_btn.setEnabled(True)
            self.model_dir_browse_btn.setEnabled(True)
            self.model_file_refresh_btn.setEnabled(True)

    def _build_path_group(self) -> QWidget:
        group = QGroupBox("路径与模型")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Mode row: 模式 | IP 端口 账号 密码 连接
        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(4)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([MODE_LOCAL, MODE_SSH])
        self.mode_combo.setFixedWidth(70)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)

        sep_label = QLabel("|")
        sep_label.setFixedWidth(10)
        sep_label.setAlignment(Qt.AlignCenter)

        self.ssh_ip_edit = QLineEdit()
        self.ssh_ip_edit.setMinimumWidth(60)

        self.ssh_port_spin = QSpinBox()
        self.ssh_port_spin.setRange(1, 65535)
        self.ssh_port_spin.setValue(22)
        self.ssh_port_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.ssh_port_spin.setFixedWidth(56)

        self.ssh_username_edit = QLineEdit()
        self.ssh_username_edit.setMinimumWidth(50)

        self.ssh_password_edit = QLineEdit()
        self.ssh_password_edit.setEchoMode(QLineEdit.Password)
        self.ssh_password_edit.setMinimumWidth(50)

        self.ssh_connect_btn = QPushButton("连接")
        self.ssh_connect_btn.clicked.connect(self._on_ssh_connect)

        self._ssh_label_ip = QLabel("IP")
        self._ssh_label_port = QLabel("端口")
        self._ssh_label_user = QLabel("账号")
        self._ssh_label_pass = QLabel("密码")
        self._ssh_labels = [sep_label, self._ssh_label_ip, self._ssh_label_port,
                            self._ssh_label_user, self._ssh_label_pass]

        mode_layout.addWidget(self.mode_combo)
        mode_layout.addWidget(sep_label)
        mode_layout.addWidget(self._ssh_label_ip)
        mode_layout.addWidget(self.ssh_ip_edit, 2)
        mode_layout.addWidget(self._ssh_label_port)
        mode_layout.addWidget(self.ssh_port_spin)
        mode_layout.addWidget(self._ssh_label_user)
        mode_layout.addWidget(self.ssh_username_edit, 1)
        mode_layout.addWidget(self._ssh_label_pass)
        mode_layout.addWidget(self.ssh_password_edit, 1)
        mode_layout.addWidget(self.ssh_connect_btn)
        form.addRow("操作模式", mode_row)

        self.llama_dir_edit = QLineEdit()
        self.llama_dir_edit.setPlaceholderText("选择 llama.cpp 目录")
        self.llama_browse_btn = QPushButton("浏览")
        self.llama_browse_btn.clicked.connect(self._pick_llama_dir)

        llama_dir_row = QWidget()
        llama_dir_layout = QHBoxLayout(llama_dir_row)
        llama_dir_layout.setContentsMargins(0, 0, 0, 0)
        llama_dir_layout.setSpacing(6)
        llama_dir_layout.addWidget(self.llama_dir_edit, 1)
        llama_dir_layout.addWidget(self.llama_browse_btn)
        form.addRow("llama.cpp目录", llama_dir_row)

        # Action row: version labels / refresh / update
        llama_action_row = QWidget()
        act_layout = QHBoxLayout(llama_action_row)
        act_layout.setContentsMargins(0, 0, 0, 0)
        act_layout.setSpacing(6)

        self.update_llamacpp_btn = QPushButton("切换本地llama.cpp版本")
        self.update_llamacpp_btn.setEnabled(False)
        self.update_llamacpp_btn.clicked.connect(self._open_update_dialog)

        self.local_ver_label = QLabel("当前版本: —")
        self.github_ver_label = QLabel("GitHub最新: —")

        self.gh_refresh_btn = QPushButton("↻")
        self.gh_refresh_btn.setFlat(True)
        self.gh_refresh_btn.setFixedSize(30, 28)
        self.gh_refresh_btn.setToolTip("重新检查 GitHub 最新版本")
        self.gh_refresh_btn.setCursor(Qt.PointingHandCursor)
        self.gh_refresh_btn.clicked.connect(self._refresh_github_latest)

        act_layout.setSpacing(10)
        act_layout.addWidget(self.local_ver_label)
        act_layout.addWidget(QLabel(" | "))
        act_layout.addWidget(self.github_ver_label)
        act_layout.addWidget(self.gh_refresh_btn)
        act_layout.addWidget(QLabel(" | "))
        act_layout.addWidget(self.update_llamacpp_btn)
        form.addRow("", llama_action_row)

        # Model dir (full-width line edit)
        self.model_dir_edit, model_dir_row, self.model_dir_browse_btn = self._path_row(
            "选择模型目录", pick_dir=True, use_ssh_browse=True
        )
        form.addRow("模型目录", model_dir_row)

        from PySide6.QtWidgets import QGridLayout

        model_grid = QWidget()
        grid = QGridLayout(model_grid)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)

        self.model_file_combo = QComboBox()
        self.model_file_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.mmproj_enabled_check = QCheckBox("多模态")
        self.mmproj_enabled_check.setToolTip("启用后选择 mmproj 文件，传递 --mmproj 参数")
        self.mmproj_file_combo = QComboBox()
        self.mmproj_file_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mmproj_file_combo.setEnabled(False)

        self.model_file_refresh_btn = QPushButton("↻")
        self.model_file_refresh_btn.setFlat(True)
        self.model_file_refresh_btn.setFixedSize(30, 58)
        self.model_file_refresh_btn.setToolTip("刷新模型文件和多模态文件列表")
        self.model_file_refresh_btn.setCursor(Qt.PointingHandCursor)
        self.model_file_refresh_btn.clicked.connect(self._refresh_all_model_files)

        mmproj_inner = QWidget()
        mmproj_layout = QHBoxLayout(mmproj_inner)
        mmproj_layout.setContentsMargins(0, 0, 0, 0)
        mmproj_layout.setSpacing(6)
        mmproj_layout.addWidget(self.mmproj_enabled_check)
        mmproj_layout.addWidget(self.mmproj_file_combo, 1)

        grid.addWidget(self.model_file_combo, 0, 0)
        grid.addWidget(mmproj_inner, 1, 0)
        grid.addWidget(self.model_file_refresh_btn, 0, 1, 2, 1)
        grid.setColumnStretch(0, 1)

        form.addRow("模型文件", model_grid)
        return group

    def _build_service_group(self) -> QWidget:
        group = QGroupBox("服务与参数")
        outer = QVBoxLayout(group)
        outer.setSpacing(8)

        # ── Row 1: Host + 端口 + 模型别名 + 选项复选框 ──────────────────────
        net_row = QWidget()
        net_layout = QHBoxLayout(net_row)
        net_layout.setContentsMargins(0, 0, 0, 0)
        net_layout.setSpacing(6)

        self.host_edit = QLineEdit()
        self.host_edit.setPlaceholderText("127.0.0.1")
        self.host_edit.setFixedWidth(110)

        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.port_spin.setFixedWidth(60)

        self.model_alias_edit = QLineEdit()
        self.model_alias_edit.setPlaceholderText("选填")
        self.model_alias_edit.setToolTip("模型别名（--alias），留空则不传递该参数")

        self.verbose_check = QCheckBox("verbose")
        self.verbose_check.setToolTip("传递 --verbose 参数，输出详细日志")
        self.rpc_enabled_check = QCheckBox("分布式加载")
        self.rpc_enabled_check.setToolTip("启用后显示 RPC 服务器节点配置，并传递 --rpc 参数")

        net_layout.addWidget(QLabel("Host"))
        net_layout.addWidget(self.host_edit)
        net_layout.addWidget(QLabel("端口"))
        net_layout.addWidget(self.port_spin)
        net_layout.addWidget(QLabel("别名"))
        net_layout.addWidget(self.model_alias_edit, 1)
        net_layout.addWidget(self.verbose_check)
        net_layout.addWidget(self.rpc_enabled_check)
        outer.addWidget(net_row)

        # ── RPC 服务器节点（分布式加载勾选后才显示）──────────────────────────
        self._rpc_box = _CollapsibleBox("RPC 服务器节点", initially_collapsed=True)
        self._build_rpc_content(self._rpc_box)
        self._rpc_box.setVisible(False)  # 默认隐藏，由分布式加载开关控制
        outer.addWidget(self._rpc_box)

        # ── 性能参数（可折叠，包含并发/上下文/GPU/批大小/KV）───────────────
        self._perf_box = _CollapsibleBox("性能参数", initially_collapsed=False)
        self._build_perf_content(self._perf_box)
        outer.addWidget(self._perf_box)

        # ── 采样参数（可折叠）──────────────────────────────────────────────
        self._sampling_box = _CollapsibleBox("采样参数", initially_collapsed=True)
        self._build_sampling_content(self._sampling_box)
        outer.addWidget(self._sampling_box)

        # ── 高级选项（固定展开，内含可折叠的推测解码）───────────────────────
        outer.addWidget(self._build_advanced_group())

        return group

    # ── RPC content ───────────────────────────────────────────────────────────

    def _build_rpc_content(self, box: _CollapsibleBox) -> None:
        self.rpc_node_list = _RpcNodeList()
        box.add_widget(self.rpc_node_list)

        add_row = QWidget()
        add_layout = QHBoxLayout(add_row)
        add_layout.setContentsMargins(0, 0, 0, 0)
        add_layout.setSpacing(4)

        self.rpc_clear_btn = QPushButton("清空")
        self.rpc_clear_btn.setObjectName("rpcClearBtn")
        self.rpc_clear_btn.setToolTip("清空所有 RPC 节点")
        self.rpc_clear_btn.clicked.connect(self._rpc_clear_nodes)

        self.rpc_host_edit = QLineEdit()
        self.rpc_host_edit.setPlaceholderText("IP 地址")
        self.rpc_host_edit.setToolTip("远程 RPC 服务器的 IP 地址或主机名")
        self.rpc_host_edit.setFixedWidth(150)
        self.rpc_host_edit.setAlignment(Qt.AlignRight)

        self.rpc_port_spin = QSpinBox()
        self.rpc_port_spin.setRange(1, 65535)
        self.rpc_port_spin.setValue(50052)
        self.rpc_port_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.rpc_port_spin.setToolTip("远程 RPC 服务器端口（默认 50052）")
        self.rpc_port_spin.setFixedWidth(72)

        self.rpc_add_btn = QPushButton("添加")
        self.rpc_add_btn.clicked.connect(self._rpc_add_node)

        add_layout.addWidget(self.rpc_clear_btn)
        add_layout.addStretch(1)
        add_layout.addWidget(self.rpc_host_edit)
        add_layout.addWidget(QLabel(":"))
        add_layout.addWidget(self.rpc_port_spin)
        add_layout.addWidget(self.rpc_add_btn)
        box.add_widget(add_row)

    # ── 性能参数 content ──────────────────────────────────────────────────────

    def _build_perf_content(self, box: _CollapsibleBox) -> None:
        def _make_cell(label_or_widget, input_widget: QWidget) -> QWidget:
            cell = QWidget()
            cl = QHBoxLayout(cell)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(4)
            cl.addWidget(label_or_widget)
            cl.addWidget(input_widget, 1)
            return cell

        # Row 1: 批大小 | 并发数 | GPU层数 | MoE CPU卸载层数 — 等宽均分
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 65536)
        self.batch_size_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.batch_size_spin.setMinimumWidth(48)

        self.parallel_enabled_check = QCheckBox("并发数")
        self.parallel_enabled_check.setChecked(True)
        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(1, 1024)
        self.parallel_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.parallel_spin.setMinimumWidth(48)

        self.gpu_layers_enabled_check = QCheckBox("GPU层数")
        self.gpu_layers_enabled_check.setChecked(True)
        self.gpu_layers_spin = QSpinBox()
        self.gpu_layers_spin.setRange(0, 1000)
        self.gpu_layers_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.gpu_layers_spin.setMinimumWidth(48)

        self.cpu_moe_edit = QLineEdit()
        self.cpu_moe_edit.setMinimumWidth(48)
        self.cpu_moe_edit.setValidator(QIntValidator(0, 9999))
        self.cpu_moe_edit.setPlaceholderText("可选")
        self.cpu_moe_edit.setToolTip("CPU MoE层数（可选）")

        row1 = QWidget()
        r1_layout = QHBoxLayout(row1)
        r1_layout.setContentsMargins(0, 0, 0, 0)
        r1_layout.setSpacing(6)
        r1_layout.addWidget(_make_cell(QLabel("批大小"), self.batch_size_spin), 1)
        r1_layout.addWidget(_make_cell(self.parallel_enabled_check, self.parallel_spin), 1)
        r1_layout.addWidget(_make_cell(self.gpu_layers_enabled_check, self.gpu_layers_spin), 1)
        r1_layout.addWidget(_make_cell(QLabel("MoE CPU卸载层数"), self.cpu_moe_edit), 1)
        box.add_widget(row1)

        # Row 2: 上下文（带滑块和自动ctx）
        self.ctx_enabled_check = QCheckBox("上下文")
        self.ctx_enabled_check.setChecked(True)

        self.ctx_slider = QSlider(Qt.Horizontal)
        self.ctx_slider.setRange(0, len(CTX_STEPS) - 1)
        self.ctx_slider.setValue(CTX_DEFAULT_IDX)
        self.ctx_slider.setTickPosition(QSlider.TicksBelow)
        self.ctx_slider.setTickInterval(1)

        self.ctx_edit = QLineEdit()
        self.ctx_edit.setFixedWidth(64)
        self.ctx_edit.setValidator(QIntValidator(256, 2_000_000))
        self.ctx_edit.setEnabled(False)

        self.ctx_custom_check = QCheckBox("自定义")
        self.ctx_custom_check.setChecked(False)

        self._ctx_update_display()

        self.fit_ctx_check = QCheckBox("最短上下文")
        self.fit_ctx_edit = QLineEdit()
        self.fit_ctx_edit.setValidator(QIntValidator(1, 2_000_000))
        self.fit_ctx_edit.setEnabled(False)

        self.fit_target_edit = QLineEdit()
        self.fit_target_edit.setValidator(QIntValidator(1, 999_999))
        self.fit_target_edit.setEnabled(False)
        self.fit_target_edit.setPlaceholderText("可选")

        # 上下文参数与自动上下文参数合并成一行，上下文开关内联在行首
        ctx_left = QWidget()
        ctx_left_layout = QHBoxLayout(ctx_left)
        ctx_left_layout.setContentsMargins(0, 0, 0, 0)
        ctx_left_layout.setSpacing(6)
        ctx_left_layout.addWidget(self.ctx_slider, 1)
        ctx_left_layout.addWidget(self.ctx_edit)
        ctx_left_layout.addWidget(self.ctx_custom_check)

        ctx_right = QWidget()
        ctx_right_layout = QHBoxLayout(ctx_right)
        ctx_right_layout.setContentsMargins(0, 0, 0, 0)
        ctx_right_layout.setSpacing(4)
        ctx_right_layout.addWidget(self.fit_ctx_check)
        ctx_right_layout.addWidget(self.fit_ctx_edit, 1)
        ctx_right_layout.addWidget(QLabel("安全空间(MiB)"))
        ctx_right_layout.addWidget(self.fit_target_edit, 1)

        ctx_row = QWidget()
        ctx_row_layout = QHBoxLayout(ctx_row)
        ctx_row_layout.setContentsMargins(0, 0, 0, 0)
        ctx_row_layout.setSpacing(6)
        ctx_row_layout.addWidget(self.ctx_enabled_check)
        ctx_row_layout.addWidget(ctx_left, 1)
        ctx_row_layout.addWidget(QLabel(" | "))
        ctx_row_layout.addWidget(ctx_right, 1)
        box.add_widget(ctx_row)

        # Row 3: 主GPU (50%) | KV K (25%) | KV V (25%)
        kv_items = ["f16", "bf16", "f32", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"]
        self.main_gpu_combo = _LazyGpuCombo()
        self.main_gpu_combo.set_ssh_getter(self.get_ssh_connection)
        self.kv_k_combo = QComboBox()
        self.kv_k_combo.addItems(kv_items)
        self.kv_v_combo = QComboBox()
        self.kv_v_combo.addItems(kv_items)

        gpu_kv_row = QWidget()
        gkv_layout = QHBoxLayout(gpu_kv_row)
        gkv_layout.setContentsMargins(0, 0, 0, 0)
        gkv_layout.setSpacing(6)
        gkv_layout.addWidget(_make_cell(QLabel("主GPU"), self.main_gpu_combo), 2)  # 50%
        gkv_layout.addWidget(_make_cell(QLabel("KV K"), self.kv_k_combo), 1)       # 25%
        gkv_layout.addWidget(_make_cell(QLabel("KV V"), self.kv_v_combo), 1)       # 25%
        box.add_widget(gpu_kv_row)

    # ── 采样参数 content ──────────────────────────────────────────────────────

    def _build_sampling_content(self, box: _CollapsibleBox) -> None:
        enable_row = QWidget()
        enable_layout = QHBoxLayout(enable_row)
        enable_layout.setContentsMargins(0, 0, 0, 0)
        enable_layout.setSpacing(6)

        self.sampling_enabled_check = QCheckBox("启用自定义采样参数")
        self.sampling_enabled_check.setChecked(True)
        enable_layout.addWidget(self.sampling_enabled_check)
        enable_layout.addStretch(1)
        box.add_widget(enable_row)

        # All four sampling params on one row — 始终显示，未启用时置灰
        self._sampling_params_widget = QWidget()
        sp_layout = QHBoxLayout(self._sampling_params_widget)
        sp_layout.setContentsMargins(0, 0, 0, 0)
        sp_layout.setSpacing(6)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0, 5)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.temperature_spin.setMinimumWidth(52)

        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0, 1)
        self.top_p_spin.setSingleStep(0.01)
        self.top_p_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.top_p_spin.setMinimumWidth(52)

        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 5000)
        self.top_k_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.top_k_spin.setMinimumWidth(52)

        self.repeat_penalty_spin = QDoubleSpinBox()
        self.repeat_penalty_spin.setRange(0.1, 3.0)
        self.repeat_penalty_spin.setSingleStep(0.05)
        self.repeat_penalty_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.repeat_penalty_spin.setMinimumWidth(52)

        def _make_sp_cell(label_text: str, spin_widget: QWidget) -> QWidget:
            cell = QWidget()
            cl = QHBoxLayout(cell)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(4)
            cl.addWidget(QLabel(label_text))
            cl.addWidget(spin_widget, 1)
            return cell

        sp_layout.addWidget(_make_sp_cell("温度", self.temperature_spin), 1)
        sp_layout.addWidget(_make_sp_cell("Top-p", self.top_p_spin), 1)
        sp_layout.addWidget(_make_sp_cell("Top-k", self.top_k_spin), 1)
        sp_layout.addWidget(_make_sp_cell("重复惩罚", self.repeat_penalty_spin), 1)
        box.add_widget(self._sampling_params_widget)

    # ── 高级选项（含可折叠的推测解码）────────────────────────────────────────

    def _build_advanced_group(self) -> QWidget:
        adv_group = QGroupBox("高级选项")
        adv_layout = QVBoxLayout(adv_group)
        adv_layout.setSpacing(6)

        # Checkboxes grid
        adv_checks = QWidget()
        from PySide6.QtWidgets import QGridLayout
        adv_checks_layout = QGridLayout(adv_checks)
        adv_checks_layout.setContentsMargins(0, 0, 0, 0)
        adv_checks_layout.setSpacing(6)

        self.enable_jinja_check = QCheckBox("启用 Jinja")
        self.flash_attn_check = QCheckBox("Flash Attention")
        self.fit_auto_check = QCheckBox("Fit(Auto)")
        self.kv_offload_cpu_check = QCheckBox("KV放CPU")
        self.no_mmap_check = QCheckBox("禁用mmap")

        self.gpu_split_enabled_check = QCheckBox("指定GPU层比例")
        self.moe_gpu_split_edit = QLineEdit()
        self.moe_gpu_split_edit.setPlaceholderText("例如 3,1（按GPU比例分配层数）")
        self.moe_gpu_split_edit.setEnabled(False)

        gpu_split_cell = QWidget()
        gsc_layout = QHBoxLayout(gpu_split_cell)
        gsc_layout.setContentsMargins(0, 0, 0, 0)
        gsc_layout.setSpacing(6)
        gsc_layout.addWidget(self.gpu_split_enabled_check)
        gsc_layout.addWidget(self.moe_gpu_split_edit, 1)

        adv_checks_layout.addWidget(self.enable_jinja_check, 0, 0)
        adv_checks_layout.addWidget(self.flash_attn_check, 0, 1)
        adv_checks_layout.addWidget(self.fit_auto_check, 0, 2)
        adv_checks_layout.addWidget(self.kv_offload_cpu_check, 1, 0)
        adv_checks_layout.addWidget(self.no_mmap_check, 1, 1)
        adv_checks_layout.addWidget(gpu_split_cell, 1, 2)
        adv_checks_layout.setColumnStretch(0, 1)
        adv_checks_layout.setColumnStretch(1, 1)
        adv_checks_layout.setColumnStretch(2, 2)
        adv_layout.addWidget(adv_checks)

        # 推测解码（可折叠）
        self._spec_box = _CollapsibleBox("推测解码 (Speculative Decoding)", initially_collapsed=True)
        self._build_speculative_content(self._spec_box)
        adv_layout.addWidget(self._spec_box)

        # 自定义参数（最后一行，带开关）
        custom_row = QWidget()
        custom_layout = QHBoxLayout(custom_row)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        custom_layout.setSpacing(6)

        self.custom_args_enabled_check = QCheckBox("自定义参数")
        self.custom_args_enabled_check.setChecked(True)
        self.custom_args_edit = QLineEdit()
        self.custom_args_edit.setPlaceholderText("额外命令行参数，例如 --threads 16 --metrics")

        custom_layout.addWidget(self.custom_args_enabled_check)
        custom_layout.addWidget(self.custom_args_edit, 1)
        adv_layout.addWidget(custom_row)

        return adv_group

    def _build_speculative_content(self, box: _CollapsibleBox) -> None:
        # ── 启用开关 ────────────────────────────────────────────────────────
        en_row = QWidget()
        en_layout = QHBoxLayout(en_row)
        en_layout.setContentsMargins(0, 0, 0, 0)
        self.speculative_enabled_check = QCheckBox("启用推测解码")
        en_layout.addWidget(self.speculative_enabled_check)
        en_layout.addStretch(1)
        box.add_widget(en_row)

        # ── draft-max | draft-min (左33%) + 草稿模型 (右67%) 合并一行 ─────────
        self.draft_max_spin = QSpinBox()
        self.draft_max_spin.setRange(1, 512)
        self.draft_max_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.draft_max_spin.setMinimumWidth(44)

        self.draft_min_spin = QSpinBox()
        self.draft_min_spin.setRange(1, 512)
        self.draft_min_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.draft_min_spin.setMinimumWidth(44)

        draft_params = QWidget()
        dp_layout = QHBoxLayout(draft_params)
        dp_layout.setContentsMargins(0, 0, 0, 0)
        dp_layout.setSpacing(4)
        dp_layout.addWidget(QLabel("draft-max"))
        dp_layout.addWidget(self.draft_max_spin, 1)
        dp_layout.addSpacing(6)
        dp_layout.addWidget(QLabel("draft-min"))
        dp_layout.addWidget(self.draft_min_spin, 1)

        self.draft_model_edit, draft_model_inner, _ = self._path_row("选择草稿模型（可选）", pick_file=True)
        draft_model_section = QWidget()
        dms_layout = QHBoxLayout(draft_model_section)
        dms_layout.setContentsMargins(0, 0, 0, 0)
        dms_layout.setSpacing(6)
        dms_layout.addWidget(QLabel("草稿模型"))
        dms_layout.addWidget(draft_model_inner, 1)

        draft_row = QWidget()
        draft_layout = QHBoxLayout(draft_row)
        draft_layout.setContentsMargins(0, 0, 0, 0)
        draft_layout.setSpacing(6)
        draft_layout.addWidget(draft_params, 1)    # ~33%
        draft_layout.addWidget(draft_model_section, 2)  # ~67%
        self._draft_row_widget = draft_row
        self._draft_model_container = draft_row
        box.add_widget(draft_row)

        # ── 自预测配置 sub-group（启用开关收纳于组内）────────────────────────
        ngram_group = QGroupBox("自预测配置")
        ngram_layout = QVBoxLayout(ngram_group)
        ngram_layout.setSpacing(6)

        self.spec_ngram_enabled_check = QCheckBox("启用自预测")
        self.spec_ngram_enabled_check.setChecked(False)

        self.spec_type_combo = QComboBox()
        self.spec_type_combo.addItems(["ngram-cache", "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod"])
        self.spec_type_combo.setCurrentText("ngram-mod")
        self.spec_type_combo.setToolTip(
            "ngram-cache: 统计缓存\n"
            "ngram-simple: 基础模式，查找最近N-gram并以后续词元为草稿\n"
            "ngram-map-k: 仅在模式多次出现时才起草（保守型）\n"
            "ngram-map-k4v: 跟踪最多4种延续并选最频繁者（实验性）\n"
            "ngram-mod: 哈希N-gram，轻量（~16MB），所有槽位共享哈希池"
        )

        # 第一行：启用开关  算法类型
        ngram_row1 = QWidget()
        nr1_layout = QHBoxLayout(ngram_row1)
        nr1_layout.setContentsMargins(0, 0, 0, 0)
        nr1_layout.setSpacing(8)
        nr1_layout.addWidget(self.spec_ngram_enabled_check)
        nr1_layout.addWidget(QLabel(" | "))
        nr1_layout.addWidget(QLabel("算法类型"))
        nr1_layout.addWidget(self.spec_type_combo, 1)
        ngram_layout.addWidget(ngram_row1)

        self.spec_ngram_size_n_spin = QSpinBox()
        self.spec_ngram_size_n_spin.setRange(1, 200)
        self.spec_ngram_size_n_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.spec_ngram_size_n_spin.setToolTip("查找窗口：用于搜索的历史词元数量（默认 12）")

        self.spec_ngram_size_m_spin = QSpinBox()
        self.spec_ngram_size_m_spin.setRange(1, 512)
        self.spec_ngram_size_m_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.spec_ngram_size_m_spin.setToolTip("草稿长度：匹配到模式后起草的词元数量（默认 48）")

        self.spec_ngram_check_rate_spin = QSpinBox()
        self.spec_ngram_check_rate_spin.setRange(1, 1000)
        self.spec_ngram_check_rate_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.spec_ngram_check_rate_spin.setToolTip("检查频率：每隔 N 个词元搜索一次模式（默认 1）")

        self.spec_ngram_min_hits_spin = QSpinBox()
        self.spec_ngram_min_hits_spin.setRange(1, 1000)
        self.spec_ngram_min_hits_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.spec_ngram_min_hits_spin.setToolTip("最小命中数：模式至少出现多少次后才用于起草（默认 1）")

        def _make_ng_cell(label_text: str, spin_widget: QWidget) -> QWidget:
            cell = QWidget()
            cl = QHBoxLayout(cell)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(4)
            cl.addWidget(QLabel(label_text))
            cl.addWidget(spin_widget, 1)
            return cell

        # 第二行：四个参数均分
        ngram_row2 = QWidget()
        nr2_layout = QHBoxLayout(ngram_row2)
        nr2_layout.setContentsMargins(0, 0, 0, 0)
        nr2_layout.setSpacing(6)
        nr2_layout.addWidget(_make_ng_cell("查找窗口", self.spec_ngram_size_n_spin), 1)
        nr2_layout.addWidget(_make_ng_cell("草稿长度", self.spec_ngram_size_m_spin), 1)
        nr2_layout.addWidget(_make_ng_cell("检查频率", self.spec_ngram_check_rate_spin), 1)
        nr2_layout.addWidget(_make_ng_cell("最小命中数", self.spec_ngram_min_hits_spin), 1)
        ngram_layout.addWidget(ngram_row2)

        self._ngram_params_row = ngram_row2
        self._ngram_enable_row = None
        self._ngram_group = ngram_group
        box.add_widget(ngram_group)

    # ──────────────────────────────────────────────────────────────────────────
    # RPC helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _rpc_add_node(self) -> None:
        host = self.rpc_host_edit.text().strip()
        if not host:
            return
        port = self.rpc_port_spin.value()
        self.rpc_node_list.add_node(f"{host}:{port}")
        self.rpc_host_edit.clear()

    def _rpc_clear_nodes(self) -> None:
        self.rpc_node_list.clear_nodes()
        self._emit_change()

    # ──────────────────────────────────────────────────────────────────────────
    # Mode & SSH
    # ──────────────────────────────────────────────────────────────────────────

    def _on_mode_changed(self, mode: str) -> None:
        is_ssh = mode == MODE_SSH

        # Save current paths for previous mode
        prev_mode = MODE_SSH if mode == MODE_LOCAL else MODE_LOCAL
        self._mode_paths[prev_mode] = {
            "llama_dir": self.llama_dir_edit.text().strip(),
            "model_dir": self.model_dir_edit.text().strip(),
            "model_file": self.model_file_combo.currentText().strip(),
            "mmproj_file": self.mmproj_file_combo.currentText().strip(),
        }

        # Restore paths for new mode
        stored = self._mode_paths.get(mode, {})
        if stored.get("llama_dir") or stored.get("model_dir") or stored.get("model_file"):
            self.llama_dir_edit.setText(stored.get("llama_dir", ""))
            self.model_dir_edit.setText(stored.get("model_dir", ""))
            self._refresh_all_model_files()
            self.model_file_combo.setCurrentText(stored.get("model_file", ""))
            self.mmproj_file_combo.setCurrentText(stored.get("mmproj_file", ""))
        else:
            self.llama_dir_edit.clear()
            self.model_dir_edit.clear()
            self.model_file_combo.clear()
            self.model_file_combo.blockSignals(True)
            self.model_file_combo.addItems([])
            self.model_file_combo.blockSignals(False)
            self.mmproj_file_combo.clear()
            self.mmproj_file_combo.blockSignals(True)
            self.mmproj_file_combo.addItems([])
            self.mmproj_file_combo.blockSignals(False)

        if is_ssh:
            self.update_llamacpp_btn.setEnabled(False)
        else:
            self._on_llama_dir_changed(self.llama_dir_edit.text())

        self.main_gpu_combo.set_ssh_mode(is_ssh)
        self.main_gpu_combo.reset_detection()
        self._apply_mode_ui_state(mode)
        self.mode_changed.emit(mode)
        self._emit_change()

    def _on_ssh_connect(self) -> None:
        if self._ssh_connection and self._ssh_connection.is_connected:
            self._ssh_connection.disconnect()
            self._ssh_connection = None
            self.ssh_connect_btn.setText("连接")
            self.main_gpu_combo.reset_detection()
            self.ssh_connection_changed.emit(False)
            self._apply_mode_ui_state(self.mode_combo.currentText())
            return

        host = self.ssh_ip_edit.text().strip()
        if not host:
            QMessageBox.warning(self, "SSH 连接", "请填写 IP 地址。")
            return
        port = self.ssh_port_spin.value()
        username = self.ssh_username_edit.text().strip()
        if not username:
            QMessageBox.warning(self, "SSH 连接", "请填写账号。")
            return
        password = self.ssh_password_edit.text()

        try:
            conn = SSHConnection(host=host, port=port, username=username, password=password)
            conn.connect()
            self._ssh_connection = conn
            self.ssh_connect_btn.setText("断开")
            self.main_gpu_combo.reset_detection()
            self.ssh_connection_changed.emit(True)
            self._apply_mode_ui_state(self.mode_combo.currentText())
            llama_dir = self.llama_dir_edit.text().strip()
            if llama_dir:
                self._update_local_version(llama_dir)
            self._refresh_all_model_files()
        except Exception as e:
            self._ssh_connection = None
            self._apply_mode_ui_state(self.mode_combo.currentText())
            QMessageBox.warning(self, "SSH 连接失败", str(e))

    def disconnect_ssh(self) -> None:
        """Disconnect SSH (called on app close)."""
        if self._ssh_connection:
            self._ssh_connection.disconnect()
            self._ssh_connection = None

    def get_ssh_connection(self) -> SSHConnection | None:
        """Return active SSH connection, or None."""
        if self._ssh_connection and self._ssh_connection.is_connected:
            return self._ssh_connection
        return None

    def is_ssh_mode(self) -> bool:
        return self.mode_combo.currentText() == MODE_SSH

    # ──────────────────────────────────────────────────────────────────────────
    # Context helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _ctx_update_display(self) -> None:
        step = CTX_STEPS[self.ctx_slider.value()]
        self.ctx_edit.blockSignals(True)
        if step == -1:
            self.ctx_edit.setText("")
            self.ctx_edit.setPlaceholderText("默认")
        else:
            self.ctx_edit.setText(str(step))
            self.ctx_edit.setPlaceholderText("")
        self.ctx_edit.blockSignals(False)

    def _on_ctx_slider_moved(self, _idx: int) -> None:
        if self.ctx_custom_check.isChecked():
            return
        self._ctx_update_display()

    def _on_ctx_custom_toggled(self, checked: bool) -> None:
        ctx_enabled = self.ctx_enabled_check.isChecked()
        if checked:
            self.ctx_slider.setEnabled(False)
            self.ctx_edit.setEnabled(ctx_enabled)
        else:
            self.ctx_slider.setEnabled(ctx_enabled)
            self.ctx_edit.setEnabled(False)
            self._ctx_update_display()

    def _ctx_value(self) -> int:
        if self.ctx_custom_check.isChecked():
            text = self.ctx_edit.text().strip()
            if text:
                try:
                    return max(256, min(int(text), 2_000_000))
                except ValueError:
                    pass
        return CTX_STEPS[self.ctx_slider.value()]

    def _ctx_set_from_value(self, value: int) -> None:
        if value in CTX_STEPS:
            idx = CTX_STEPS.index(value)
            self.ctx_slider.blockSignals(True)
            self.ctx_slider.setValue(idx)
            self.ctx_slider.blockSignals(False)
            self.ctx_custom_check.blockSignals(True)
            self.ctx_custom_check.setChecked(False)
            self.ctx_custom_check.blockSignals(False)
            self.ctx_edit.setEnabled(False)
            self._ctx_update_display()
        elif value > 0:
            self.ctx_custom_check.blockSignals(True)
            self.ctx_custom_check.setChecked(True)
            self.ctx_custom_check.blockSignals(False)
            self.ctx_slider.setEnabled(False)
            self.ctx_edit.setEnabled(True)
            self.ctx_edit.blockSignals(True)
            self.ctx_edit.setText(str(value))
            self.ctx_edit.blockSignals(False)
        else:
            self.ctx_slider.blockSignals(True)
            self.ctx_slider.setValue(CTX_DEFAULT_IDX)
            self.ctx_slider.blockSignals(False)
            self.ctx_custom_check.blockSignals(True)
            self.ctx_custom_check.setChecked(False)
            self.ctx_custom_check.blockSignals(False)
            self.ctx_edit.setEnabled(False)
            self._ctx_update_display()

    def _on_ctx_enabled_toggled(self, checked: bool) -> None:
        custom_on = self.ctx_custom_check.isChecked()
        self.ctx_custom_check.setEnabled(checked)
        self.ctx_slider.setEnabled(checked and not custom_on)
        self.ctx_edit.setEnabled(checked and custom_on)
        self._update_fit_ctx_style()

    def _on_fit_ctx_toggled(self, checked: bool) -> None:
        self.fit_ctx_edit.setEnabled(checked)
        self.fit_target_edit.setEnabled(checked)
        if not checked:
            self.fit_ctx_edit.setStyleSheet("")
        self._update_fit_ctx_style()

    def _on_fit_ctx_text_changed(self, _text: str) -> None:
        self._update_fit_ctx_style()

    def _update_fit_ctx_style(self) -> None:
        if self.fit_ctx_check.isChecked() and not self.fit_ctx_edit.text().strip():
            self.fit_ctx_edit.setStyleSheet("border: 1px solid red;")
        else:
            self.fit_ctx_edit.setStyleSheet("")

    # ──────────────────────────────────────────────────────────────────────────
    # Toggle handlers
    # ──────────────────────────────────────────────────────────────────────────

    def _on_gpu_layers_enabled_toggled(self, checked: bool) -> None:
        self.gpu_layers_spin.setEnabled(checked)
        self.cpu_moe_edit.setEnabled(checked)

    def _on_gpu_split_enabled_toggled(self, checked: bool) -> None:
        self.moe_gpu_split_edit.setEnabled(checked)

    def _on_parallel_enabled_toggled(self, checked: bool) -> None:
        self.parallel_spin.setEnabled(checked)

    def _on_sampling_enabled_toggled(self, checked: bool) -> None:
        # 置灰而非隐藏
        self._sampling_params_widget.setEnabled(checked)

    def _on_custom_args_enabled_toggled(self, checked: bool) -> None:
        self.custom_args_edit.setEnabled(checked)

    def _on_mmproj_enabled_toggled(self, checked: bool) -> None:
        self.mmproj_file_combo.setEnabled(checked)

    def _on_rpc_enabled_toggled(self, checked: bool) -> None:
        self._rpc_box.setVisible(checked)

    def _update_spec_states(self) -> None:
        """根据推测解码和自预测的开关状态统一更新各控件的启用/禁用。"""
        spec_on = self.speculative_enabled_check.isChecked()
        ngram_on = spec_on and self.spec_ngram_enabled_check.isChecked()

        if self._draft_row_widget is not None:
            self._draft_row_widget.setEnabled(spec_on)
        if self._ngram_group is not None:
            self._ngram_group.setEnabled(spec_on)

        # 直接覆写各子控件状态，确保不受 group.setEnabled 传播的干扰
        self.spec_ngram_enabled_check.setEnabled(spec_on)
        self.spec_type_combo.setEnabled(ngram_on)
        self.spec_ngram_size_n_spin.setEnabled(ngram_on)
        self.spec_ngram_size_m_spin.setEnabled(ngram_on)
        self.spec_ngram_check_rate_spin.setEnabled(ngram_on)
        self.spec_ngram_min_hits_spin.setEnabled(ngram_on)

    def _on_speculative_enabled_toggled(self, checked: bool) -> None:
        self._update_spec_states()

    def _on_spec_ngram_enabled_toggled(self, checked: bool) -> None:
        self._update_spec_states()

    # ──────────────────────────────────────────────────────────────────────────
    # GitHub version helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _refresh_github_latest(self) -> None:
        if self._gh_fetch_thread is not None:
            return
        self.github_ver_label.setText("GitHub最新: 检查中...")
        self.gh_refresh_btn.setEnabled(False)
        self._fetch_github_latest()

    def _on_llama_dir_changed(self, text: str) -> None:
        llama_dir = text.strip()
        has_dir = bool(llama_dir)
        if not self.is_ssh_mode():
            self.update_llamacpp_btn.setEnabled(has_dir)
        if has_dir:
            self._update_local_version(llama_dir)
        else:
            self.local_ver_label.setText("当前版本: —")

    def _update_local_version(self, llama_dir: str) -> None:
        import re as _re
        if self.is_ssh_mode():
            conn = self.get_ssh_connection()
            if not conn:
                self.local_ver_label.setText("当前版本: —")
                return
            server_path = f"{llama_dir.rstrip('/')}/llama-server"
            try:
                code, out, err = conn.exec_command(f'"{server_path}" --version 2>&1', timeout=5)
                output = (out + err).strip()
                m = _re.search(r'version[:\s]+(\d+)', output, _re.IGNORECASE)
                if m:
                    self.local_ver_label.setText(f"当前版本: b{m.group(1)}")
                else:
                    self.local_ver_label.setText("当前版本: 检查失败")
            except Exception:
                self.local_ver_label.setText("当前版本: 检查失败")
            return
        server_exe = Path(llama_dir) / "llama-server.exe"
        if not server_exe.exists():
            server_exe = Path(llama_dir) / "llama-server"
        if not server_exe.exists():
            self.local_ver_label.setText("当前版本: 未找到")
            return
        try:
            result = subprocess.run(
                [str(server_exe), "--version"],
                capture_output=True, text=True, timeout=5
            )
            output = (result.stdout + result.stderr).strip()
            m = _re.search(r'version[:\s]+(\d+)', output, _re.IGNORECASE)
            if m:
                self.local_ver_label.setText(f"当前版本: b{m.group(1)}")
            else:
                self.local_ver_label.setText("当前版本: 检查失败")
        except Exception:
            self.local_ver_label.setText("当前版本: 检查失败")

    def _fetch_github_latest(self) -> None:
        self._gh_fetch_thread = QThread()
        self._gh_fetch_worker = _VersionFetchWorker()
        self._gh_fetch_worker.moveToThread(self._gh_fetch_thread)
        self._gh_fetch_thread.started.connect(self._gh_fetch_worker.run)
        self._gh_fetch_worker.finished.connect(self._on_github_latest_fetched)
        self._gh_fetch_worker.finished.connect(self._gh_fetch_thread.quit)
        self._gh_fetch_thread.finished.connect(self._gh_fetch_thread.deleteLater)
        self._gh_fetch_thread.finished.connect(self._on_gh_thread_truly_finished)
        self._gh_fetch_thread.start()

    def _on_gh_thread_truly_finished(self) -> None:
        self._gh_fetch_thread = None
        self._gh_fetch_worker = None
        self.gh_refresh_btn.setEnabled(True)

    def _on_github_latest_fetched(self, tag: str) -> None:
        self._github_latest = tag
        self.github_ver_label.setText(f"GitHub最新: {tag if tag else '—'}")

    # ──────────────────────────────────────────────────────────────────────────
    # Signal binding
    # ──────────────────────────────────────────────────────────────────────────

    def _bind_change_signals(self) -> None:
        _excluded = {self.rpc_host_edit, self.rpc_port_spin, self.mode_combo}

        for widget in self.findChildren(QLineEdit):
            if widget not in _excluded:
                widget.textChanged.connect(self._emit_change)
        for widget in self.findChildren(QSpinBox):
            if widget not in _excluded:
                widget.valueChanged.connect(self._emit_change)
        for widget in self.findChildren(QDoubleSpinBox):
            widget.valueChanged.connect(self._emit_change)
        for widget in self.findChildren(QCheckBox):
            widget.toggled.connect(self._emit_change)
        for widget in self.findChildren(QComboBox):
            if widget not in _excluded:
                widget.currentTextChanged.connect(self._emit_change)

        self.llama_dir_edit.textChanged.connect(self._on_llama_dir_changed)
        self.model_dir_edit.textChanged.connect(self._refresh_all_model_files)
        self.mmproj_enabled_check.toggled.connect(self._on_mmproj_enabled_toggled)
        self.ctx_slider.valueChanged.connect(self._on_ctx_slider_moved)
        self.ctx_slider.valueChanged.connect(self._emit_change)
        self.rpc_node_list.changed.connect(self._emit_change)

        self.ctx_enabled_check.toggled.connect(self._on_ctx_enabled_toggled)
        self.ctx_custom_check.toggled.connect(self._on_ctx_custom_toggled)
        self.fit_ctx_check.toggled.connect(self._on_fit_ctx_toggled)
        self.fit_ctx_edit.textChanged.connect(self._on_fit_ctx_text_changed)
        self.gpu_layers_enabled_check.toggled.connect(self._on_gpu_layers_enabled_toggled)
        self.gpu_split_enabled_check.toggled.connect(self._on_gpu_split_enabled_toggled)
        self.parallel_enabled_check.toggled.connect(self._on_parallel_enabled_toggled)
        self.sampling_enabled_check.toggled.connect(self._on_sampling_enabled_toggled)
        self.custom_args_enabled_check.toggled.connect(self._on_custom_args_enabled_toggled)
        self.rpc_enabled_check.toggled.connect(self._on_rpc_enabled_toggled)
        self.verbose_check.toggled.connect(self._emit_change)
        self.speculative_enabled_check.toggled.connect(self._on_speculative_enabled_toggled)
        self.spec_ngram_enabled_check.toggled.connect(self._on_spec_ngram_enabled_toggled)

    def _emit_change(self) -> None:
        self.config_changed.emit(self.to_config())

    # ──────────────────────────────────────────────────────────────────────────
    # Model scanning
    # ──────────────────────────────────────────────────────────────────────────

    def _scan_gguf_files(self) -> tuple[list[str], list[str]]:
        """Scan model directory, returning (model_files, mmproj_files)."""
        model_dir = self.model_dir_edit.text().strip()
        all_files: list[str] = []

        if self.is_ssh_mode():
            conn = self.get_ssh_connection()
            if conn and model_dir:
                try:
                    all_files = conn.find_gguf_files(model_dir)
                except Exception:
                    all_files = []
        else:
            p = Path(model_dir)
            if p.exists():
                for item in p.rglob("*.gguf"):
                    rel = str(item.relative_to(p)).replace("\\", "/")
                    all_files.append(rel)
                all_files.sort(key=str.casefold)

        models = [f for f in all_files if "mmproj" not in f.lower()]
        mmproj = [f for f in all_files if "mmproj" in f.lower()]
        return models, mmproj

    def scan_models(self) -> int:
        models, _ = self._scan_gguf_files()

        current_model = self.model_file_combo.currentText()
        self.model_file_combo.blockSignals(True)
        self.model_file_combo.clear()
        self.model_file_combo.addItems(models)
        self.model_file_combo.setCurrentText(current_model)
        self.model_file_combo.blockSignals(False)
        self._emit_change()
        return len(models)

    def scan_mmproj(self) -> int:
        _, mmproj = self._scan_gguf_files()

        current_mmproj = self.mmproj_file_combo.currentText()
        self.mmproj_file_combo.blockSignals(True)
        self.mmproj_file_combo.clear()
        self.mmproj_file_combo.addItems(mmproj)
        self.mmproj_file_combo.setCurrentText(current_mmproj)
        self.mmproj_file_combo.blockSignals(False)
        self._emit_change()
        return len(mmproj)

    def _refresh_all_model_files(self) -> None:
        """Scan once and populate both model and mmproj combos."""
        models, mmproj = self._scan_gguf_files()

        current_model = self.model_file_combo.currentText()
        self.model_file_combo.blockSignals(True)
        self.model_file_combo.clear()
        self.model_file_combo.addItems(models)
        self.model_file_combo.setCurrentText(current_model)
        self.model_file_combo.blockSignals(False)

        current_mmproj = self.mmproj_file_combo.currentText()
        self.mmproj_file_combo.blockSignals(True)
        self.mmproj_file_combo.clear()
        self.mmproj_file_combo.addItems(mmproj)
        self.mmproj_file_combo.setCurrentText(current_mmproj)
        self.mmproj_file_combo.blockSignals(False)

        self._emit_change()

    # ──────────────────────────────────────────────────────────────────────────
    # Config serialization
    # ──────────────────────────────────────────────────────────────────────────

    def to_config(self) -> LlamaConfig:
        return LlamaConfig(
            mode=self.mode_combo.currentText(),
            ssh_host=self.ssh_ip_edit.text().strip(),
            ssh_port=self.ssh_port_spin.value(),
            ssh_username=self.ssh_username_edit.text().strip(),
            ssh_password=self.ssh_password_edit.text(),
            llama_dir=self.llama_dir_edit.text().strip(),
            model_dir=self.model_dir_edit.text().strip(),
            model_file=self.model_file_combo.currentText().strip(),
            mmproj_enabled=self.mmproj_enabled_check.isChecked(),
            mmproj_file=self.mmproj_file_combo.currentText().strip(),
            host=self.host_edit.text().strip() or "127.0.0.1",
            port=self.port_spin.value(),
            parallel=self.parallel_spin.value(),
            parallel_enabled=self.parallel_enabled_check.isChecked(),
            ctx_size=self._ctx_value(),
            ctx_enabled=self.ctx_enabled_check.isChecked(),
            fit_ctx_enabled=self.fit_ctx_check.isChecked(),
            fit_ctx=self.fit_ctx_edit.text().strip(),
            fit_target=self.fit_target_edit.text().strip(),
            gpu_layers=self.gpu_layers_spin.value(),
            gpu_layers_enabled=self.gpu_layers_enabled_check.isChecked(),
            cpu_moe_layers=self.cpu_moe_edit.text().strip(),
            main_gpu=self._main_gpu_value(),
            sampling_enabled=self.sampling_enabled_check.isChecked(),
            temperature=self.temperature_spin.value(),
            top_p=self.top_p_spin.value(),
            top_k=self.top_k_spin.value(),
            repeat_penalty=self.repeat_penalty_spin.value(),
            batch_size=self.batch_size_spin.value(),
            kv_cache_type_k=self.kv_k_combo.currentText(),
            kv_cache_type_v=self.kv_v_combo.currentText(),
            cache_ram_mib=self._loaded_cache_ram_mib,
            enable_jinja=self.enable_jinja_check.isChecked(),
            enable_flash_attention=self.flash_attn_check.isChecked(),
            fit_auto=self.fit_auto_check.isChecked(),
            kv_offload_cpu=self.kv_offload_cpu_check.isChecked(),
            no_mmap=self.no_mmap_check.isChecked(),
            moe_gpu_split=self.moe_gpu_split_edit.text().strip(),
            gpu_split_enabled=self.gpu_split_enabled_check.isChecked(),
            speculative_enabled=self.speculative_enabled_check.isChecked(),
            draft_max=self.draft_max_spin.value(),
            draft_min=self.draft_min_spin.value(),
            draft_model=self.draft_model_edit.text().strip(),
            spec_ngram_enabled=self.spec_ngram_enabled_check.isChecked(),
            spec_type=self.spec_type_combo.currentText(),
            spec_ngram_size_n=self.spec_ngram_size_n_spin.value(),
            spec_ngram_size_m=self.spec_ngram_size_m_spin.value(),
            spec_ngram_check_rate=self.spec_ngram_check_rate_spin.value(),
            spec_ngram_min_hits=self.spec_ngram_min_hits_spin.value(),
            rpc_servers=self.rpc_node_list.get_nodes(),
            rpc_enabled=self.rpc_enabled_check.isChecked(),
            verbose=self.verbose_check.isChecked(),
            model_alias=self.model_alias_edit.text().strip(),
            custom_args_enabled=self.custom_args_enabled_check.isChecked(),
            custom_args=self.custom_args_edit.text().strip(),
        )

    def from_config(self, config: LlamaConfig) -> None:
        self._loaded_cache_ram_mib = config.cache_ram_mib

        self.mode_combo.blockSignals(True)
        mode = config.mode if config.mode in (MODE_LOCAL, MODE_SSH) else MODE_LOCAL
        self.mode_combo.setCurrentText(mode)
        self.mode_combo.blockSignals(False)

        self.ssh_ip_edit.setText(config.ssh_host)
        self.ssh_port_spin.setValue(config.ssh_port)
        self.ssh_username_edit.setText(config.ssh_username)
        self.ssh_password_edit.setText(config.ssh_password or "")

        self._mode_paths[mode] = {
            "llama_dir": config.llama_dir,
            "model_dir": config.model_dir,
            "model_file": config.model_file,
            "mmproj_file": config.mmproj_file,
        }

        self.llama_dir_edit.setText(config.llama_dir)
        self.model_dir_edit.setText(config.model_dir)
        self._refresh_all_model_files()
        self.model_file_combo.setCurrentText(config.model_file)
        self.mmproj_enabled_check.setChecked(config.mmproj_enabled)
        self.mmproj_file_combo.setEnabled(config.mmproj_enabled)
        self.mmproj_file_combo.setCurrentText(config.mmproj_file)

        self.host_edit.setText(config.host)
        self.port_spin.setValue(config.port)
        self.parallel_spin.setValue(config.parallel)
        self.parallel_enabled_check.setChecked(config.parallel_enabled)
        self.parallel_spin.setEnabled(config.parallel_enabled)

        self._ctx_set_from_value(config.ctx_size)
        self.ctx_enabled_check.setChecked(config.ctx_enabled)
        self.fit_ctx_check.setChecked(config.fit_ctx_enabled)
        self.fit_ctx_edit.setText(config.fit_ctx)
        self.fit_target_edit.setText(config.fit_target)
        self._on_ctx_enabled_toggled(config.ctx_enabled)
        self._on_fit_ctx_toggled(config.fit_ctx_enabled)

        self.gpu_layers_spin.setValue(config.gpu_layers)
        self.gpu_layers_enabled_check.setChecked(config.gpu_layers_enabled)
        self.cpu_moe_edit.setText(config.cpu_moe_layers)
        self._on_gpu_layers_enabled_toggled(config.gpu_layers_enabled)

        gpu_val = config.main_gpu
        combo = self.main_gpu_combo
        found = False
        for i in range(combo.count()):
            text = combo.itemText(i)
            if text == gpu_val or text.split(":")[0].strip() == gpu_val:
                combo.setCurrentIndex(i)
                found = True
                break
        if not found:
            combo.setCurrentIndex(0)

        self.sampling_enabled_check.setChecked(config.sampling_enabled)
        self._on_sampling_enabled_toggled(config.sampling_enabled)
        self.temperature_spin.setValue(config.temperature)
        self.top_p_spin.setValue(config.top_p)
        self.top_k_spin.setValue(config.top_k)
        self.repeat_penalty_spin.setValue(config.repeat_penalty)

        self.batch_size_spin.setValue(config.batch_size)
        self.kv_k_combo.setCurrentText(config.kv_cache_type_k)
        self.kv_v_combo.setCurrentText(config.kv_cache_type_v)

        self.enable_jinja_check.setChecked(config.enable_jinja)
        self.flash_attn_check.setChecked(config.enable_flash_attention)
        self.fit_auto_check.setChecked(config.fit_auto)
        self.kv_offload_cpu_check.setChecked(config.kv_offload_cpu)
        self.no_mmap_check.setChecked(config.no_mmap)
        self.moe_gpu_split_edit.setText(config.moe_gpu_split)
        self.gpu_split_enabled_check.setChecked(config.gpu_split_enabled)
        self._on_gpu_split_enabled_toggled(config.gpu_split_enabled)

        self.speculative_enabled_check.setChecked(config.speculative_enabled)
        self.draft_max_spin.setValue(config.draft_max)
        self.draft_min_spin.setValue(config.draft_min)
        self.draft_model_edit.setText(config.draft_model)
        self.spec_ngram_enabled_check.setChecked(config.spec_ngram_enabled)
        self.spec_type_combo.setCurrentText(config.spec_type)
        self.spec_ngram_size_n_spin.setValue(config.spec_ngram_size_n)
        self.spec_ngram_size_m_spin.setValue(config.spec_ngram_size_m)
        self.spec_ngram_check_rate_spin.setValue(config.spec_ngram_check_rate)
        self.spec_ngram_min_hits_spin.setValue(config.spec_ngram_min_hits)
        self._update_spec_states()

        self.rpc_node_list.set_nodes(config.rpc_servers)
        self.rpc_enabled_check.setChecked(config.rpc_enabled)
        self._rpc_box.setVisible(config.rpc_enabled)
        self.verbose_check.setChecked(config.verbose)

        self.model_alias_edit.setText(config.model_alias)

        self.custom_args_enabled_check.setChecked(config.custom_args_enabled)
        self.custom_args_edit.setEnabled(config.custom_args_enabled)
        self.custom_args_edit.setText(config.custom_args)

        self.main_gpu_combo.set_ssh_mode(mode == MODE_SSH)
        self._apply_mode_ui_state(mode)
        if mode == MODE_LOCAL and config.llama_dir:
            self._on_llama_dir_changed(config.llama_dir)

        self._emit_change()

    # ──────────────────────────────────────────────────────────────────────────
    # Utility helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _main_gpu_value(self) -> str:
        text = self.main_gpu_combo.currentText().strip()
        if text == "Auto" or not text:
            return "Auto"
        if ":" in text:
            return text.split(":")[0].strip()
        return text

    def _path_row(
        self,
        placeholder: str,
        pick_dir: bool = False,
        pick_file: bool = False,
        use_ssh_browse: bool = False,
    ) -> tuple[QLineEdit, QWidget, QPushButton]:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        button = QPushButton("浏览")

        def _pick() -> None:
            if use_ssh_browse and self.is_ssh_mode():
                conn = self.get_ssh_connection()
                if not conn:
                    QMessageBox.warning(self, "SSH 浏览", "请先连接 SSH。")
                    return
                from app.ui.dialogs.ssh_browse_dialog import SSHBrowseDialog
                d = SSHBrowseDialog(self, conn, line_edit.text() or "/", pick_dir=pick_dir)
                if d.exec() == QDialog.DialogCode.Accepted:
                    line_edit.setText(d.selected_path())
                return
            selected = ""
            if pick_dir:
                selected = QFileDialog.getExistingDirectory(self, "选择目录", line_edit.text() or str(Path.home()))
            elif pick_file:
                selected, _ = QFileDialog.getOpenFileName(self, "选择文件", line_edit.text() or str(Path.home()))
            if selected:
                line_edit.setText(selected)

        button.clicked.connect(_pick)
        row.addWidget(line_edit, 1)
        row.addWidget(button)
        return line_edit, container, button

    def _pick_llama_dir(self) -> None:
        if self.is_ssh_mode():
            conn = self.get_ssh_connection()
            if not conn:
                QMessageBox.warning(self, "SSH 浏览", "请先连接 SSH。")
                return
            from app.ui.dialogs.ssh_browse_dialog import SSHBrowseDialog
            d = SSHBrowseDialog(self, conn, self.llama_dir_edit.text() or "/", pick_dir=True)
            if d.exec() == QDialog.DialogCode.Accepted:
                self.llama_dir_edit.setText(d.selected_path())
            return
        selected = QFileDialog.getExistingDirectory(
            self, "选择 llama.cpp 目录", self.llama_dir_edit.text() or str(Path.home())
        )
        if selected:
            self.llama_dir_edit.setText(selected)

    def _open_update_dialog(self) -> None:
        from app.ui.dialogs.update_dialog import UpdateDialog
        llama_dir = self.llama_dir_edit.text().strip()
        dialog = UpdateDialog(self, default_target_dir=llama_dir)
        dialog.exec()
        target_dir = dialog.target_edit.text().strip() or llama_dir
        if target_dir:
            self._update_local_version(target_dir)

    def _install_wheel_filters(self) -> None:
        for w in self.findChildren(QAbstractSpinBox):
            w.installEventFilter(_wheel_filter)
        for w in self.findChildren(QComboBox):
            w.installEventFilter(_wheel_filter)
