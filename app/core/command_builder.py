from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from app.core.config_schema import LlamaConfig


def _resolve_executable(llama_dir: str) -> str:
    base = Path(llama_dir) if llama_dir else Path(".")
    candidates = (
        base / "llama-server.exe",
        base / "llama-server",
        base / "llama-cli.exe",
        base / "llama-cli",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def build_command(config: LlamaConfig) -> list[str]:
    model_path = str(Path(config.model_dir) / config.model_file) if config.model_file else ""
    command: list[str] = [_resolve_executable(config.llama_dir)]

    if model_path:
        command.extend(["-m", model_path])

    command.extend(["--host", config.host, "--port", str(config.port), "-np", str(config.parallel)])
    if config.ctx_size != -1:
        command.extend(["-c", str(config.ctx_size)])
    command.extend(["-ngl", str(config.gpu_layers)])

    if config.main_gpu != "Auto":
        command.extend(["--main-gpu", str(config.main_gpu)])

    command.extend(
        [
            "--temp",
            str(config.temperature),
            "--top-p",
            str(config.top_p),
            "--top-k",
            str(config.top_k),
            "--repeat-penalty",
            str(config.repeat_penalty),
            "--batch-size",
            str(config.batch_size),
            "--cache-type-k",
            config.kv_cache_type_k,
            "--cache-type-v",
            config.kv_cache_type_v,
        ]
    )

    if config.cache_ram_mib > 0:
        command.extend(["--cache-ram", str(config.cache_ram_mib)])

    # Jinja is enabled by default; only disable explicitly
    if not config.enable_jinja:
        command.append("--no-jinja")
    # Flash attention default is 'auto'; only enable explicitly
    if config.enable_flash_attention:
        command.extend(["--flash-attn", "on"])
    # --fit is 'on' by default; only add when user disables auto-fit
    if not config.fit_auto:
        command.extend(["--fit", "off"])
    if config.kv_offload_cpu:
        command.append("--no-kv-offload")
    if config.no_mmap:
        command.append("--no-mmap")
    if config.moe_gpu_split:
        command.extend(["--tensor-split", config.moe_gpu_split])

    if config.speculative_enabled:
        command.extend(["--draft-max", str(config.draft_max), "--draft-min", str(config.draft_min)])
        if config.draft_model:
            command.extend(["--model-draft", config.draft_model])
        command.extend(["--spec-type", config.spec_type])
        command.extend(["--spec-ngram-size-n", str(config.spec_ngram_size_n)])
        command.extend(["--spec-ngram-size-m", str(config.spec_ngram_size_m)])
        if config.spec_ngram_check_rate != 1:
            command.extend(["--spec-ngram-check-rate", str(config.spec_ngram_check_rate)])
        if config.spec_ngram_min_hits != 1:
            command.extend(["--spec-ngram-min-hits", str(config.spec_ngram_min_hits)])

    if config.custom_args.strip():
        command.extend(shlex.split(config.custom_args, posix=False))

    return command


def build_command_preview(config: LlamaConfig) -> str:
    return subprocess.list2cmdline(build_command(config))
