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

    command.extend(["--host", config.host, "--port", str(config.port)])
    if config.parallel_enabled:
        command.extend(["-np", str(config.parallel)])

    if config.rpc_enabled and config.rpc_servers:
        command.extend(["--rpc", ",".join(config.rpc_servers)])

    if config.ctx_enabled:
        if config.ctx_size != -1:
            command.extend(["-c", str(config.ctx_size)])
        if config.fit_ctx_enabled and config.fit_ctx.strip():
            command.extend(["--fit-ctx", config.fit_ctx.strip()])
            if config.fit_target.strip():
                command.extend(["--fit-target", config.fit_target.strip()])

    if config.gpu_layers_enabled:
        command.extend(["-ngl", str(config.gpu_layers)])
        if config.cpu_moe_layers.strip():
            command.extend(["--cpu-moe", config.cpu_moe_layers.strip()])

    if config.main_gpu != "Auto":
        command.extend(["--main-gpu", str(config.main_gpu)])

    if config.sampling_enabled:
        command.extend(
            [
                "--temp", str(config.temperature),
                "--top-p", str(config.top_p),
                "--top-k", str(config.top_k),
                "--repeat-penalty", str(config.repeat_penalty),
            ]
        )

    command.extend(
        [
            "--batch-size", str(config.batch_size),
            "--cache-type-k", config.kv_cache_type_k,
            "--cache-type-v", config.kv_cache_type_v,
        ]
    )

    if config.cache_ram_mib > 0:
        command.extend(["--cache-ram", str(config.cache_ram_mib)])

    if not config.enable_jinja:
        command.append("--no-jinja")
    if config.enable_flash_attention:
        command.extend(["--flash-attn", "on"])
    if not config.fit_auto:
        command.extend(["--fit", "off"])
    if config.kv_offload_cpu:
        command.append("--no-kv-offload")
    if config.no_mmap:
        command.append("--no-mmap")
    if config.gpu_split_enabled and config.moe_gpu_split:
        command.extend(["--tensor-split", config.moe_gpu_split])

    if config.speculative_enabled:
        command.extend(["--draft-max", str(config.draft_max), "--draft-min", str(config.draft_min)])
        if config.draft_model:
            command.extend(["--model-draft", config.draft_model])
        if config.spec_ngram_enabled:
            command.extend(["--spec-type", config.spec_type])
            command.extend(["--spec-ngram-size-n", str(config.spec_ngram_size_n)])
            command.extend(["--spec-ngram-size-m", str(config.spec_ngram_size_m)])
            if config.spec_ngram_check_rate != 1:
                command.extend(["--spec-ngram-check-rate", str(config.spec_ngram_check_rate)])
            if config.spec_ngram_min_hits != 1:
                command.extend(["--spec-ngram-min-hits", str(config.spec_ngram_min_hits)])

    if config.model_alias.strip():
        command.extend(["--alias", config.model_alias.strip()])

    if config.custom_args_enabled and config.custom_args.strip():
        command.extend(shlex.split(config.custom_args))

    return command


def build_command_preview(config: LlamaConfig) -> str:
    """Return a human-readable command string.

    Custom args are appended verbatim so that quotes entered by the user
    are preserved in the preview instead of being lost through
    shlex.split → list2cmdline round-tripping.
    """
    base_command = build_command(config)
    # Remove the already-split custom args from the list so we can re-append
    # the raw string, preserving the user's original quoting.
    if config.custom_args_enabled and config.custom_args.strip():
        extra = shlex.split(config.custom_args)
        base_command = base_command[: len(base_command) - len(extra)]
        return subprocess.list2cmdline(base_command) + " " + config.custom_args.strip()
    return subprocess.list2cmdline(base_command)
