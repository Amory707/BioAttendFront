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


def _to_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _normalize_trigger_mode(value: str | None) -> str:
    raw = (value or "space").strip().lower()
    aliases = {
        "space": "space",
        "keyboard": "space",
        "clavier": "space",
        "pir": "pir",
        "ultrason": "ultrason",
        "ultrasonic": "ultrason",
        "hc-sr04": "ultrason",
    }
    return aliases.get(raw, "space")


def _resolve_env_file() -> Path:
    return Path(__file__).resolve().parents[2] / ".env"


load_dotenv(_resolve_env_file())


@dataclass(slots=True)
class Settings:
    debug: bool
    kiosk_mode: bool
    device_name: str
    camera_mirror: bool
    camera_swap_rb: bool
    camera_jpeg_quality: int
    liveness_enabled: bool
    liveness_model_dir: str
    liveness_threshold: float
    liveness_live_class_idx: int
    camera_width: int
    camera_height: int
    camera_full_fov: bool
    camera_device: str
    camera_source: str
    camera_backend: str
    camera_warmup_ms: int
    camera_read_attempts: int
    insightface_model_name: str
    insightface_det_width: int
    insightface_det_height: int
    server_url: str
    events_url: str
    api_token: str
    api_timeout_seconds: int
    pointage_trigger_mode: str
    ultrason_capture_prep_delay_ms: int
    gpio_pir: int
    gpio_ultrason_trigger: int
    gpio_ultrason_echo: int
    ultrason_distance_cm: float

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            debug=_to_bool(os.getenv("DEBUG"), default=False),
            kiosk_mode=_to_bool(os.getenv("KIOSK_MODE"), default=True),
            device_name=os.getenv("DEVICE_NAME", "bioattend-pi").strip(),
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
            camera_full_fov=_to_bool(os.getenv("CAMERA_FULL_FOV"), default=True),
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
            events_url=os.getenv("EVENTS_URL", "").strip(),
            api_token=os.getenv("API_TOKEN", "").strip(),
            api_timeout_seconds=_to_int(os.getenv("API_TIMEOUT_SECONDS"), default=8),
            pointage_trigger_mode=_normalize_trigger_mode(os.getenv("POINTAGE_TRIGGER_MODE")),
            ultrason_capture_prep_delay_ms=max(
                0,
                _to_int(os.getenv("ULTRASON_CAPTURE_PREP_DELAY_MS"), default=2600),
            ),
            gpio_pir=_to_int(os.getenv("GPIO_PIR"), default=17),
            gpio_ultrason_trigger=_to_int(os.getenv("GPIO_ULTRASON_TRIGGER"), default=18),
            gpio_ultrason_echo=_to_int(os.getenv("GPIO_ULTRASON_ECHO"), default=24),
            ultrason_distance_cm=max(2.0, _to_float(os.getenv("ULTRASON_DISTANCE_CM"), default=80.0)),
        )

    def as_public_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["api_token"] = "***" if self.api_token else ""
        return payload