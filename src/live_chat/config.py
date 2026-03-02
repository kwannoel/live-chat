from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    fast_model: str = "claude-haiku-4-5-20251001"
    deep_model: str = "claude-sonnet-4-6"
    sample_rate: int = 16000
    tts_voice: str = "en_US-lessac-medium"
    persona: str | None = None
    auto_speak: bool = False
    backend: str = "api"
    cli_path: str = "claude"
    min_silence_ms: int = 600

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or Path.home() / ".live-chat" / "config.yaml"
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return cls.from_dict(data)
        return cls()
