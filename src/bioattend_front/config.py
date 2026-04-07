from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _resolve_env_file() -> Path:
    return Path(__file__).resolve().parents[2] / ".env"


load_dotenv(_resolve_env_file())


@dataclass(slots=True)
class Settings:
    debug: bool
    camera_width: int
    camera_height: int
    camera_device: str
    camera_backend: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            debug=_to_bool(os.getenv("DEBUG"), default=False),
            camera_width=_to_int(os.getenv("CAMERA_WIDTH"), default=1280),
            camera_height=_to_int(os.getenv("CAMERA_HEIGHT"), default=720),
            camera_device=os.getenv("CAMERA_DEVICE", "0"),
            camera_backend=os.getenv("CAMERA_BACKEND", "auto").strip().lower(),
        )

    def as_public_dict(self) -> dict[str, object]:
        return asdict(self)