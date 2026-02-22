from __future__ import annotations

import json
from pathlib import Path

from app.core.config_schema import LlamaConfig


class ProfileStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.profiles_dir = self.root_dir / "config" / "profiles"
        self.state_path = self.root_dir / "config" / "state.json"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def list_profiles(self) -> list[str]:
        """递归扫描 profiles_dir 下所有 .json 文件，返回相对于 profiles_dir 的 POSIX 路径（不含后缀）。"""
        results = []
        for item in sorted(self.profiles_dir.rglob("*.json"), key=lambda p: str(p).casefold()):
            rel = item.relative_to(self.profiles_dir).with_suffix("")
            results.append(rel.as_posix())
        return results

    def load_profile(self, profile_name: str) -> LlamaConfig:
        profile_path = self.profiles_dir / f"{profile_name}.json"
        if not profile_path.exists():
            raise FileNotFoundError(f"未找到配置：{profile_name}")
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        return LlamaConfig.from_dict(payload)

    def save_profile(self, profile_name: str, config: LlamaConfig) -> Path:
        profile_path = self.profiles_dir / f"{profile_name}.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(
            json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return profile_path

    def load_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def save_state(self, state: dict) -> None:
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
