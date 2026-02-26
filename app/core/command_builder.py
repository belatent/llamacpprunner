from __future__ import annotations

import shlex
import subprocess
from pathlib import Path, PurePosixPath

from app.core.config_schema import LlamaConfig, MODE_SSH


def _posix_path(*parts: str) -> str:
    """Join path parts using POSIX forward-slash separators."""
    return str(PurePosixPath(*parts))


def _resolve_executable(llama_dir: str, for_linux: bool = False) -> str:
    if for_linux:
        base = PurePosixPath(llama_dir) if llama_dir else PurePosixPath(".")
        return str(base / "llama-server")
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
    for_linux = config.mode == MODE_SSH
    if for_linux:
        model_path = _posix_path(config.model_dir, config.model_file) if config.model_file else ""
    else:
        model_path = str(Path(config.model_dir) / config.model_file) if config.model_file else ""
    command: list[str] = [_resolve_executable(config.llama_dir, for_linux=for_linux)]

    if model_path:
        command.extend(["-m", model_path])

    if config.mmproj_enabled and config.mmproj_file:
        if for_linux:
            mmproj_path = _posix_path(config.model_dir, config.mmproj_file)
        else:
            mmproj_path = str(Path(config.model_dir) / config.mmproj_file)
        command.extend(["--mmproj", mmproj_path])

    host = config.host
    if for_linux:
        host = "0.0.0.0"  # Remote server must listen on all interfaces for SSH access
    command.extend(["--host", host, "--port", str(config.port)])
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

    if config.verbose:
        command.append("--verbose")

    if config.model_alias.strip():
        command.extend(["--alias", config.model_alias.strip()])

    if config.custom_args_enabled and config.custom_args.strip():
        command.extend(shlex.split(config.custom_args))

    return command


def _cmd_to_string(parts: list[str], for_linux: bool) -> str:
    if for_linux:
        return " ".join(shlex.quote(p) for p in parts)
    return subprocess.list2cmdline(parts)


def build_command_preview(config: LlamaConfig) -> str:
    """Return a human-readable command string.

    Custom args are appended verbatim so that quotes entered by the user
    are preserved in the preview instead of being lost through
    shlex.split → list2cmdline round-tripping.
    """
    for_linux = config.mode == MODE_SSH
    base_command = build_command(config)
    if config.custom_args_enabled and config.custom_args.strip():
        extra = shlex.split(config.custom_args)
        base_command = base_command[: len(base_command) - len(extra)]
        return _cmd_to_string(base_command, for_linux) + " " + config.custom_args.strip()
    return _cmd_to_string(base_command, for_linux)
