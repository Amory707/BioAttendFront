from __future__ import annotations

import platform
import time
from typing import Any

import cv2

from .config import Settings


def _resolve_device(camera_device: str) -> int | str:
    stripped = camera_device.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def _resolve_backends(backend_name: str) -> list[tuple[str, int]]:
    backends: list[tuple[str, int]] = []
    if backend_name == "v4l2" and hasattr(cv2, "CAP_V4L2"):
        backends.append(("v4l2", cv2.CAP_V4L2))
    elif backend_name == "any":
        backends.append(("any", cv2.CAP_ANY))
    else:
        if hasattr(cv2, "CAP_V4L2"):
            backends.append(("v4l2", cv2.CAP_V4L2))
        backends.append(("any", cv2.CAP_ANY))
    return backends


def probe_camera(settings: Settings) -> dict[str, Any]:
    device = _resolve_device(settings.camera_device)
    backends = _resolve_backends(settings.camera_backend)
    attempts: list[dict[str, Any]] = []

    for backend_name, backend_flag in backends:
        started_at = time.monotonic()
        capture = cv2.VideoCapture(device, backend_flag)

        try:
            opened = capture.isOpened()
            attempt: dict[str, Any] = {
                "backend": backend_name,
                "device": settings.camera_device,
                "opened": opened,
                "duration_ms": round((time.monotonic() - started_at) * 1000, 2),
            }

            if not opened:
                attempts.append(attempt)
                continue

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_height)

            if settings.camera_warmup_ms > 0:
                time.sleep(settings.camera_warmup_ms / 1000)

            read_ok = False
            frame = None
            read_attempts: list[dict[str, Any]] = []

            for attempt_index in range(settings.camera_read_attempts):
                read_started_at = time.monotonic()
                current_read_ok, current_frame = capture.read()
                read_attempts.append(
                    {
                        "index": attempt_index + 1,
                        "read_ok": current_read_ok,
                        "duration_ms": round((time.monotonic() - read_started_at) * 1000, 2),
                    }
                )
                if current_read_ok and current_frame is not None:
                    read_ok = True
                    frame = current_frame
                    break
                time.sleep(0.1)

            attempt["read_ok"] = read_ok
            attempt["warmup_ms"] = settings.camera_warmup_ms
            attempt["read_attempt_count"] = settings.camera_read_attempts
            attempt["read_attempts"] = read_attempts

            if read_ok and frame is not None:
                height, width = frame.shape[:2]
                attempt["frame_shape"] = [int(height), int(width)]
                attempt["pixel_format_channels"] = int(frame.shape[2]) if len(frame.shape) == 3 else 1
                attempt["reported_width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                attempt["reported_height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                attempt["reported_fps"] = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                return {
                    "ok": True,
                    "camera": attempt,
                    "platform": platform.platform(),
                    "note": "Camera opened and a frame was captured in memory.",
                }

            attempt["error"] = "Camera opened but no frame could be read after repeated attempts."
            attempts.append(attempt)
        finally:
            capture.release()

    return {
        "ok": False,
        "platform": platform.platform(),
        "attempts": attempts,
        "error": "Unable to open the camera or read a frame.",
        "hint": "On Raspberry Pi CSI cameras, make sure the camera is enabled and exposed through a backend OpenCV can read.",
    }