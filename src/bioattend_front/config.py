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
    kiosk_mode: bool
    camera_mirror: bool
    camera_swap_rb: bool
    camera_jpeg_quality: int
    liveness_enabled: bool
    liveness_model_dir: str
    liveness_threshold: float
    liveness_live_class_idx: int
    camera_width: int
    camera_height: int
    camera_device: str
    camera_source: str
    camera_backend: str
    camera_warmup_ms: int
    camera_read_attempts: int
    insightface_model_name: str
    insightface_det_width: int
    insightface_det_height: int
    server_url: str
    api_token: str
    api_timeout_seconds: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            debug=_to_bool(os.getenv("DEBUG"), default=False),
            kiosk_mode=_to_bool(os.getenv("KIOSK_MODE"), default=True),
            camera_mirror=_to_bool(os.getenv("CAMERA_MIRROR"), default=True),
            camera_swap_rb=_to_bool(os.getenv("CAMERA_SWAP_RB"), default=False),
            camera_jpeg_quality=max(40, min(95, _to_int(os.getenv("CAMERA_JPEG_QUALITY"), default=68))),
            liveness_enabled=_to_bool(os.getenv("LIVENESS_ENABLED"), default=False),
            liveness_model_dir=os.getenv(
                "LIVENESS_MODEL_DIR",
                str(Path(__file__).resolve().parents[2] / "models" / "liveness"),
            ).strip(),
            liveness_threshold=float(os.getenv("LIVENESS_THRESHOLD") or "0.6"),
            liveness_live_class_idx=_to_int(os.getenv("LIVENESS_LIVE_CLASS_IDX"), default=1),
            camera_width=_to_int(os.getenv("CAMERA_WIDTH"), default=1280),
            camera_height=_to_int(os.getenv("CAMERA_HEIGHT"), default=720),
            camera_device=os.getenv("CAMERA_DEVICE", "0"),
            camera_source=os.getenv("CAMERA_SOURCE", "auto").strip().lower(),
            camera_backend=os.getenv("CAMERA_BACKEND", "auto").strip().lower(),
            camera_warmup_ms=_to_int(os.getenv("CAMERA_WARMUP_MS"), default=800),
            camera_read_attempts=_to_int(os.getenv("CAMERA_READ_ATTEMPTS"), default=10),
            insightface_model_name=os.getenv("INSIGHTFACE_MODEL_NAME", "buffalo_l").strip(),
            insightface_det_width=_to_int(os.getenv("INSIGHTFACE_DET_WIDTH"), default=640),
            insightface_det_height=_to_int(os.getenv("INSIGHTFACE_DET_HEIGHT"), default=640),
            server_url=os.getenv(
                "SERVER_URL",
                "https://bioattend.138.199.195.144.sslip.io/api/face/identify/",
            ).strip(),
            api_token=os.getenv("API_TOKEN", "").strip(),
            api_timeout_seconds=_to_int(os.getenv("API_TIMEOUT_SECONDS"), default=8),
        )

    def as_public_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["api_token"] = "***" if self.api_token else ""
        return payload