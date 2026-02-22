from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QThread, Qt, Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core.config_schema import LlamaConfig

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
            return [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    except Exception:
        pass
    return []


class _LazyGpuCombo(QComboBox):
    """ComboBox that auto-detects GPUs on first popup open."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._detected = False
        self.addItem("Auto")

    def showPopup(self) -> None:
        if not self._detected:
            self._detected = True
            self._detect_and_populate()
        super().showPopup()

    def _detect_and_populate(self) -> None:
        current = self.currentText()
        gpus = _detect_gpus()
        self.blockSignals(True)
        self.clear()
        self.addItem("Auto")
        for i, name in enumerate(gpus):
            self.addItem(f"{i}: {name}")
        # Restore previous selection by index prefix or exact match
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


class ParamsPanel(QWidget):
    config_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._github_latest: str = ""
        self._gh_fetch_thread: QThread | None = None
        self._gh_fetch_worker: _VersionFetchWorker | None = None
        self._build_ui()
        self.from_config(LlamaConfig())
        self._bind_change_signals()
        self._install_wheel_filters()
        self._fetch_github_latest()

    def _build_ui(self) -> None:
        self.setMinimumWidth(400)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        root.addWidget(self._build_path_group())
        root.addWidget(self._build_service_group())
        root.addWidget(self._build_sampling_group())
        root.addWidget(self._build_perf_group())
        root.addWidget(self._build_advanced_group())
        root.addWidget(self._build_speculative_group())
        root.addWidget(self._build_custom_group())
        root.addStretch(1)

    def _build_path_group(self) -> QWidget:
        group = QGroupBox("路径与模型")
        form = QFormLayout(group)

        # llama.cpp 目录 — row 1: line edit; row 2: browse + update + version labels
        self.llama_dir_edit = QLineEdit()
        self.llama_dir_edit.setPlaceholderText("选择 llama.cpp 目录")
        form.addRow("llama.cpp目录", self.llama_dir_edit)

        llama_action_row = QWidget()
        llama_action_layout = QHBoxLayout(llama_action_row)
        llama_action_layout.setContentsMargins(0, 0, 0, 0)
        llama_action_layout.setSpacing(6)

        self.llama_browse_btn = QPushButton("浏览")
        self.llama_browse_btn.clicked.connect(self._pick_llama_dir)

        self.update_llamacpp_btn = QPushButton("版本更新")
        self.update_llamacpp_btn.setEnabled(False)
        self.update_llamacpp_btn.clicked.connect(self._open_update_dialog)

        self.local_ver_label = QLabel("本地: —")
        self.github_ver_label = QLabel("GitHub最新: —")

        llama_action_layout.addWidget(self.llama_browse_btn)
        llama_action_layout.addWidget(self.update_llamacpp_btn)
        llama_action_layout.addWidget(self.local_ver_label)
        llama_action_layout.addWidget(QLabel(" | "))
        llama_action_layout.addWidget(self.github_ver_label)
        llama_action_layout.addStretch(1)
        form.addRow("", llama_action_row)

        # 模型目录
        self.model_dir_edit, model_dir_row = self._path_row("选择模型目录", pick_dir=True)

        # 模型文件（无子文件夹复选框，无扫描按钮）
        self.model_file_combo = QComboBox()

        form.addRow("模型目录", model_dir_row)
        form.addRow("模型文件", self.model_file_combo)
        return group

    def _build_service_group(self) -> QWidget:
        group = QGroupBox("服务参数")
        form = QFormLayout(group)

        self.host_edit = QLineEdit()

        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)

        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(1, 1024)
        self.parallel_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)

        self.gpu_layers_spin = QSpinBox()
        self.gpu_layers_spin.setRange(0, 1000)
        self.gpu_layers_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)

        # Context size: fixed-step slider + manual input
        self._ctx_custom = False
        self._ctx_updating = False

        self.ctx_slider = QSlider(Qt.Horizontal)
        self.ctx_slider.setRange(0, len(CTX_STEPS) - 1)
        self.ctx_slider.setValue(CTX_DEFAULT_IDX)
        self.ctx_slider.setTickPosition(QSlider.TicksBelow)
        self.ctx_slider.setTickInterval(1)

        self.ctx_edit = QLineEdit()
        self.ctx_edit.setFixedWidth(88)
        self.ctx_edit.setValidator(QIntValidator(256, 2_000_000))
        self._ctx_update_display()  # set initial placeholder

        ctx_widget = QWidget()
        ctx_layout = QHBoxLayout(ctx_widget)
        ctx_layout.setContentsMargins(0, 0, 0, 0)
        ctx_layout.setSpacing(6)
        ctx_layout.addWidget(self.ctx_slider, 1)
        ctx_layout.addWidget(self.ctx_edit)

        self.main_gpu_combo = _LazyGpuCombo()

        form.addRow("Host", self.host_edit)
        form.addRow("端口", self.port_spin)
        form.addRow("并发数", self.parallel_spin)
        form.addRow("上下文", ctx_widget)
        form.addRow("GPU层数", self.gpu_layers_spin)
        form.addRow("主GPU", self.main_gpu_combo)
        return group

    def _build_sampling_group(self) -> QWidget:
        group = QGroupBox("采样参数")
        form = QFormLayout(group)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0, 5)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)

        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0, 1)
        self.top_p_spin.setSingleStep(0.01)
        self.top_p_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)

        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 5000)
        self.top_k_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)

        self.repeat_penalty_spin = QDoubleSpinBox()
        self.repeat_penalty_spin.setRange(0.1, 3.0)
        self.repeat_penalty_spin.setSingleStep(0.05)
        self.repeat_penalty_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)

        form.addRow("温度", self.temperature_spin)
        form.addRow("Top-p", self.top_p_spin)
        form.addRow("Top-k", self.top_k_spin)
        form.addRow("重复惩罚", self.repeat_penalty_spin)
        return group

    def _build_perf_group(self) -> QWidget:
        group = QGroupBox("性能参数")
        form = QFormLayout(group)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 65536)
        self.batch_size_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)

        self.kv_k_combo = QComboBox()
        self.kv_v_combo = QComboBox()
        kv_items = ["f16", "bf16", "f32", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"]
        self.kv_k_combo.addItems(kv_items)
        self.kv_v_combo.addItems(kv_items)

        self.cache_ram_spin = QSpinBox()
        self.cache_ram_spin.setRange(0, 65536)
        self.cache_ram_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.cache_ram_spin.setSpecialValueText("默认")

        form.addRow("批大小", self.batch_size_spin)
        form.addRow("KV-K类型", self.kv_k_combo)
        form.addRow("KV-V类型", self.kv_v_combo)
        form.addRow("RAM缓存(MiB)", self.cache_ram_spin)
        return group

    def _build_advanced_group(self) -> QWidget:
        group = QGroupBox("高级选项")
        layout = QGridLayout(group)

        self.enable_jinja_check = QCheckBox("启用 Jinja")
        self.flash_attn_check = QCheckBox("Flash Attention")
        self.fit_auto_check = QCheckBox("Fit(Auto)")
        self.kv_offload_cpu_check = QCheckBox("KV放CPU")
        self.no_mmap_check = QCheckBox("禁用mmap")

        self.expert_mode_combo = QComboBox()
        self.expert_mode_combo.addItems(["不分配", "均衡分配", "自定义分配"])
        self.moe_gpu_split_edit = QLineEdit()

        layout.addWidget(self.enable_jinja_check, 0, 0)
        layout.addWidget(self.flash_attn_check, 0, 1)
        layout.addWidget(self.fit_auto_check, 1, 0)
        layout.addWidget(self.kv_offload_cpu_check, 1, 1)
        layout.addWidget(self.no_mmap_check, 2, 0)
        layout.addWidget(QLabel("专家模式"), 3, 0)
        layout.addWidget(self.expert_mode_combo, 3, 1)
        layout.addWidget(QLabel("MoE专家GPU分配"), 4, 0)
        layout.addWidget(self.moe_gpu_split_edit, 4, 1)
        return group

    def _build_speculative_group(self) -> QWidget:
        group = QGroupBox("推测解码 (Speculative Decoding)")
        outer = QVBoxLayout(group)
        outer.setSpacing(6)

        self.speculative_enabled_check = QCheckBox("启用推测解码")
        outer.addWidget(self.speculative_enabled_check)

        form = QFormLayout()
        form.setContentsMargins(0, 4, 0, 0)
        form.setSpacing(4)

        # --- 草稿模型参数 ---
        self.draft_max_spin = QSpinBox()
        self.draft_max_spin.setRange(1, 512)
        self.draft_max_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)

        self.draft_min_spin = QSpinBox()
        self.draft_min_spin.setRange(1, 512)
        self.draft_min_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)

        self.draft_model_edit, draft_model_row = self._path_row("选择草稿模型（可选）", pick_file=True)

        form.addRow("draft-max", self.draft_max_spin)
        form.addRow("draft-min", self.draft_min_spin)
        form.addRow("草稿模型", draft_model_row)

        # --- 自预测参数 ---
        self.spec_type_combo = QComboBox()
        self.spec_type_combo.addItems(["ngram-cache", "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod"])
        self.spec_type_combo.setCurrentText("ngram-mod")
        self.spec_type_combo.setToolTip(
            "ngram-cache: 统计缓存\n"
            "ngram-simple: 基础模式，查找最近N-gram并以后续词元为草稿\n"
            "ngram-map-k: 仅在模式多次出现时才起草（保守型）\n"
            "ngram-map-k4v: 跟踪最多4种延续并选最频繁者（实验性）\n"
            "ngram-mod: 哈希N-gram，用LCG对每个N-gram计算滚动哈希并存储后续词元，推理时迭代查找；\n"
            "           轻量（~16MB），内存与复杂度恒定，草稿长度可变，所有槽位共享哈希池"
        )

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
        self.spec_ngram_check_rate_spin.setToolTip("检查频率：每隔 N 个词元搜索一次模式（默认 1，每次都搜索）")

        self.spec_ngram_min_hits_spin = QSpinBox()
        self.spec_ngram_min_hits_spin.setRange(1, 1000)
        self.spec_ngram_min_hits_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.spec_ngram_min_hits_spin.setToolTip("最小命中数：模式至少出现多少次后才用于起草（默认 1）")

        form.addRow("自预测 · 算法类型", self.spec_type_combo)
        form.addRow("自预测 · 查找窗口 N", self.spec_ngram_size_n_spin)
        form.addRow("自预测 · 草稿长度 M", self.spec_ngram_size_m_spin)
        form.addRow("自预测 · 检查频率", self.spec_ngram_check_rate_spin)
        form.addRow("自预测 · 最小命中数", self.spec_ngram_min_hits_spin)

        outer.addLayout(form)
        return group

    def _build_custom_group(self) -> QWidget:
        group = QGroupBox("自定义参数")
        layout = QVBoxLayout(group)
        self.custom_args_edit = QLineEdit()
        self.custom_args_edit.setPlaceholderText("额外命令行参数，例如 --threads 16 --metrics")
        layout.addWidget(self.custom_args_edit)
        return group

    def _path_row(self, placeholder: str, pick_dir: bool = False, pick_file: bool = False) -> tuple[QLineEdit, QWidget]:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        button = QPushButton("浏览")

        def _pick() -> None:
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
        return line_edit, container

    def _pick_llama_dir(self) -> None:
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
        # Refresh local version after dialog closes (update may have completed)
        target_dir = dialog.target_edit.text().strip() or llama_dir
        if target_dir:
            self._update_local_version(target_dir)

    def _ctx_update_display(self) -> None:
        """Update placeholder text to reflect slider's current step (non-custom mode)."""
        step = CTX_STEPS[self.ctx_slider.value()]
        self.ctx_edit.setPlaceholderText("默认" if step == -1 else str(step))

    def _on_ctx_slider_moved(self, _idx: int) -> None:
        if self._ctx_custom:
            return
        self._ctx_update_display()

    def _on_ctx_edit_changed(self, text: str) -> None:
        if self._ctx_updating:
            return
        if text.strip():
            # User entered a value → enter custom mode
            if not self._ctx_custom:
                self._ctx_custom = True
                self.ctx_slider.setEnabled(False)
        else:
            # Input cleared → exit custom mode, restore slider
            if self._ctx_custom:
                self._ctx_custom = False
                self.ctx_slider.setEnabled(True)
                self._ctx_update_display()

    def _ctx_value(self) -> int:
        text = self.ctx_edit.text().strip()
        if self._ctx_custom and text:
            try:
                return max(256, min(int(text), 2_000_000))
            except ValueError:
                pass
        step = CTX_STEPS[self.ctx_slider.value()]
        return step  # may be -1 (use llama.cpp default)

    def _ctx_set_from_value(self, value: int) -> None:
        """Restore ctx widget from a saved config value."""
        self._ctx_updating = True
        if value in CTX_STEPS:
            idx = CTX_STEPS.index(value)
            self.ctx_slider.blockSignals(True)
            self.ctx_slider.setValue(idx)
            self.ctx_slider.blockSignals(False)
            self.ctx_slider.setEnabled(True)
            self._ctx_custom = False
            self.ctx_edit.clear()
            self._ctx_update_display()
        elif value > 0:
            self._ctx_custom = True
            self.ctx_slider.setEnabled(False)
            self.ctx_edit.setText(str(value))
        else:
            # Fallback: set to default (8192)
            self.ctx_slider.blockSignals(True)
            self.ctx_slider.setValue(CTX_DEFAULT_IDX)
            self.ctx_slider.blockSignals(False)
            self.ctx_slider.setEnabled(True)
            self._ctx_custom = False
            self.ctx_edit.clear()
            self._ctx_update_display()
        self._ctx_updating = False

    def _on_llama_dir_changed(self, text: str) -> None:
        llama_dir = text.strip()
        has_dir = bool(llama_dir)
        self.update_llamacpp_btn.setEnabled(has_dir)

        # Update local version label
        if has_dir:
            self._update_local_version(llama_dir)
        else:
            self.local_ver_label.setText("本地: —")


    def _update_local_version(self, llama_dir: str) -> None:
        """Try to read local llama-server version."""
        import re as _re
        server_exe = Path(llama_dir) / "llama-server.exe"
        if not server_exe.exists():
            server_exe = Path(llama_dir) / "llama-server"
        if not server_exe.exists():
            self.local_ver_label.setText("本地: 未找到")
            return
        try:
            result = subprocess.run(
                [str(server_exe), "--version"],
                capture_output=True, text=True, timeout=5
            )
            output = (result.stdout + result.stderr).strip()
            # Extract the build number after "version:" and before any parenthesis
            m = _re.search(r'version[:\s]+(\d+)', output, _re.IGNORECASE)
            version = m.group(1) if m else ("未知" if not output else output.splitlines()[0][:20])
            self.local_ver_label.setText(f"本地: b{version}" if m else f"本地: {version}")
        except Exception:
            self.local_ver_label.setText("本地: 未知")

    def _fetch_github_latest(self) -> None:
        """Fetch the latest GitHub release tag in background."""
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

    def _on_github_latest_fetched(self, tag: str) -> None:
        self._github_latest = tag
        display = tag if tag else "—"
        self.github_ver_label.setText(f"GitHub最新: {display}")

    def _bind_change_signals(self) -> None:
        widget_types = (QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox)
        widgets = []
        for widget_type in widget_types:
            widgets.extend(self.findChildren(widget_type))
        for widget in widgets:
            if isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._emit_change)
            elif isinstance(widget, QSpinBox):
                widget.valueChanged.connect(self._emit_change)
            elif isinstance(widget, QDoubleSpinBox):
                widget.valueChanged.connect(self._emit_change)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self._emit_change)
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self._emit_change)

        self.llama_dir_edit.textChanged.connect(self._on_llama_dir_changed)
        self.model_dir_edit.textChanged.connect(self.scan_models)
        self.ctx_slider.valueChanged.connect(self._on_ctx_slider_moved)
        self.ctx_slider.valueChanged.connect(self._emit_change)
        self.ctx_edit.textChanged.connect(self._on_ctx_edit_changed)

    def _emit_change(self) -> None:
        self.config_changed.emit(self.to_config())

    def scan_models(self) -> int:
        model_dir = Path(self.model_dir_edit.text())
        models: list[str] = []
        if model_dir.exists():
            for item in model_dir.rglob("*.gguf"):
                rel = str(item.relative_to(model_dir)).replace("\\", "/")
                models.append(rel)
            models.sort(key=str.casefold)

        current_model = self.model_file_combo.currentText()

        self.model_file_combo.blockSignals(True)
        self.model_file_combo.clear()
        self.model_file_combo.addItems(models)
        self.model_file_combo.setCurrentText(current_model)
        self.model_file_combo.blockSignals(False)
        self._emit_change()
        return len(models)

    def to_config(self) -> LlamaConfig:
        return LlamaConfig(
            llama_dir=self.llama_dir_edit.text().strip(),
            model_dir=self.model_dir_edit.text().strip(),
            model_file=self.model_file_combo.currentText().strip(),
            host=self.host_edit.text().strip() or "127.0.0.1",
            port=self.port_spin.value(),
            parallel=self.parallel_spin.value(),
            ctx_size=self._ctx_value(),
            gpu_layers=self.gpu_layers_spin.value(),
            main_gpu=self._main_gpu_value(),
            temperature=self.temperature_spin.value(),
            top_p=self.top_p_spin.value(),
            top_k=self.top_k_spin.value(),
            repeat_penalty=self.repeat_penalty_spin.value(),
            batch_size=self.batch_size_spin.value(),
            kv_cache_type_k=self.kv_k_combo.currentText(),
            kv_cache_type_v=self.kv_v_combo.currentText(),
            cache_ram_mib=self.cache_ram_spin.value(),
            enable_jinja=self.enable_jinja_check.isChecked(),
            enable_flash_attention=self.flash_attn_check.isChecked(),
            fit_auto=self.fit_auto_check.isChecked(),
            kv_offload_cpu=self.kv_offload_cpu_check.isChecked(),
            no_mmap=self.no_mmap_check.isChecked(),
            moe_gpu_split=self.moe_gpu_split_edit.text().strip(),
            expert_mode=self.expert_mode_combo.currentText(),
            speculative_enabled=self.speculative_enabled_check.isChecked(),
            draft_max=self.draft_max_spin.value(),
            draft_min=self.draft_min_spin.value(),
            draft_model=self.draft_model_edit.text().strip(),
            spec_type=self.spec_type_combo.currentText(),
            spec_ngram_size_n=self.spec_ngram_size_n_spin.value(),
            spec_ngram_size_m=self.spec_ngram_size_m_spin.value(),
            spec_ngram_check_rate=self.spec_ngram_check_rate_spin.value(),
            spec_ngram_min_hits=self.spec_ngram_min_hits_spin.value(),
            custom_args=self.custom_args_edit.text().strip(),
        )

    def from_config(self, config: LlamaConfig) -> None:
        self.llama_dir_edit.setText(config.llama_dir)
        self.model_dir_edit.setText(config.model_dir)
        self.scan_models()
        self.model_file_combo.setCurrentText(config.model_file)

        self.host_edit.setText(config.host)
        self.port_spin.setValue(config.port)
        self.parallel_spin.setValue(config.parallel)
        self._ctx_set_from_value(config.ctx_size)
        self.gpu_layers_spin.setValue(config.gpu_layers)
        # Restore main_gpu: try exact match, then index-prefix match
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

        self.temperature_spin.setValue(config.temperature)
        self.top_p_spin.setValue(config.top_p)
        self.top_k_spin.setValue(config.top_k)
        self.repeat_penalty_spin.setValue(config.repeat_penalty)

        self.batch_size_spin.setValue(config.batch_size)
        self.kv_k_combo.setCurrentText(config.kv_cache_type_k)
        self.kv_v_combo.setCurrentText(config.kv_cache_type_v)
        self.cache_ram_spin.setValue(config.cache_ram_mib)

        self.enable_jinja_check.setChecked(config.enable_jinja)
        self.flash_attn_check.setChecked(config.enable_flash_attention)
        self.fit_auto_check.setChecked(config.fit_auto)
        self.kv_offload_cpu_check.setChecked(config.kv_offload_cpu)
        self.no_mmap_check.setChecked(config.no_mmap)
        self.moe_gpu_split_edit.setText(config.moe_gpu_split)
        self.expert_mode_combo.setCurrentText(config.expert_mode)

        self.speculative_enabled_check.setChecked(config.speculative_enabled)
        self.draft_max_spin.setValue(config.draft_max)
        self.draft_min_spin.setValue(config.draft_min)
        self.draft_model_edit.setText(config.draft_model)
        self.spec_type_combo.setCurrentText(config.spec_type)
        self.spec_ngram_size_n_spin.setValue(config.spec_ngram_size_n)
        self.spec_ngram_size_m_spin.setValue(config.spec_ngram_size_m)
        self.spec_ngram_check_rate_spin.setValue(config.spec_ngram_check_rate)
        self.spec_ngram_min_hits_spin.setValue(config.spec_ngram_min_hits)
        self.custom_args_edit.setText(config.custom_args)
        self._emit_change()

    def _main_gpu_value(self) -> str:
        """Return 'Auto' or the numeric GPU index string from the combo."""
        text = self.main_gpu_combo.currentText().strip()
        if text == "Auto" or not text:
            return "Auto"
        # Format may be "N: GPU Name" — extract "N"
        if ":" in text:
            return text.split(":")[0].strip()
        return text

    def _install_wheel_filters(self) -> None:
        for w in self.findChildren(QAbstractSpinBox):
            w.installEventFilter(_wheel_filter)
        for w in self.findChildren(QComboBox):
            w.installEventFilter(_wheel_filter)
