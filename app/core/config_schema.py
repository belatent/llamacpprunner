from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LlamaConfig:
    llama_dir: str = ""
    model_dir: str = ""
    model_file: str = ""

    host: str = "127.0.0.1"
    port: int = 8080
    parallel: int = 1
    parallel_enabled: bool = True
    ctx_size: int = 8192
    ctx_enabled: bool = True
    fit_ctx_enabled: bool = False
    fit_ctx: str = ""
    fit_target: str = ""
    gpu_layers: int = 1
    gpu_layers_enabled: bool = True
    cpu_moe_layers: str = ""
    main_gpu: str = "Auto"

    sampling_enabled: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 40
    repeat_penalty: float = 1.0

    batch_size: int = 4096
    kv_cache_type_k: str = "q8_0"
    kv_cache_type_v: str = "q8_0"
    cache_ram_mib: int = 0

    enable_jinja: bool = True
    enable_flash_attention: bool = False
    fit_auto: bool = True
    kv_offload_cpu: bool = False
    no_mmap: bool = False

    moe_gpu_split: str = ""
    gpu_split_enabled: bool = False

    speculative_enabled: bool = False
    draft_max: int = 16
    draft_min: int = 2
    draft_model: str = ""

    spec_ngram_enabled: bool = False
    spec_type: str = "ngram-mod"
    spec_ngram_size_n: int = 12
    spec_ngram_size_m: int = 48
    spec_ngram_check_rate: int = 1
    spec_ngram_min_hits: int = 1

    rpc_servers: list[str] = field(default_factory=list)
    rpc_enabled: bool = False

    model_alias: str = ""

    custom_args_enabled: bool = True
    custom_args: str = ""

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.llama_dir and not Path(self.llama_dir).exists():
            errors.append("Llama 目录不存在。")
        if self.model_dir and not Path(self.model_dir).exists():
            errors.append("模型目录不存在。")
        if self.model_file and self.model_dir:
            model_path = Path(self.model_dir) / self.model_file
            if not model_path.exists():
                errors.append("模型文件不存在。")
        if not (1 <= self.port <= 65535):
            errors.append("端口范围必须在 1~65535。")
        if self.parallel < 1:
            errors.append("并发数必须 >= 1。")
        if self.ctx_enabled and self.ctx_size != -1 and self.ctx_size < 256:
            errors.append("上下文大小必须 >= 256（或设为 -1 使用默认值）。")
        if self.ctx_enabled and self.fit_ctx_enabled and not self.fit_ctx.strip():
            errors.append('已勾选"向下适配"，请填写 fit-ctx 数值。')
        if self.top_k < 1:
            errors.append("Top-k 必须 >= 1。")
        if not (0.0 <= self.top_p <= 1.0):
            errors.append("Top-p 必须在 0~1 之间。")
        if self.temperature < 0:
            errors.append("温度必须 >= 0。")
        if self.repeat_penalty <= 0:
            errors.append("重复惩罚必须 > 0。")
        if self.speculative_enabled and self.draft_min > self.draft_max:
            errors.append("预测解码 draft-min 不能大于 draft-max。")
        for entry in self.rpc_servers:
            parts = entry.rsplit(":", 1)
            if len(parts) != 2 or not parts[0].strip():
                errors.append(f"RPC 节点地址格式无效（应为 host:port）：{entry}")
            else:
                try:
                    p = int(parts[1])
                    if not (1 <= p <= 65535):
                        raise ValueError
                except ValueError:
                    errors.append(f"RPC 节点端口无效（应为 1~65535）：{entry}")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LlamaConfig":
        known = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        clean_payload = {key: payload[key] for key in known if key in payload}
        return cls(**clean_payload)
