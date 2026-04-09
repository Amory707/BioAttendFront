from __future__ import annotations

import importlib
import platform
import threading
import time
from typing import Any

import cv2

from .config import Settings


_persistent_camera: Any = None
_persistent_lock = threading.Lock()


def _picamera_frame_format() -> str:
    # BGR888 aligne directement la sortie Picamera2 avec OpenCV.
    return "BGR888"


def _load_picamera2_class() -> type[Any]:
    module = importlib.import_module("picamera2")
    return module.Picamera2


def _get_persistent_picamera2(settings: Settings) -> Any:
    """Retourne l'instance Picamera2 persistante, l'ouvre si nécessaire."""
    global _persistent_camera
    if _persistent_camera is not None:
        return _persistent_camera
    with _persistent_lock:
        if _persistent_camera is not None:
            return _persistent_camera
        Picamera2 = _load_picamera2_class()
        cam = Picamera2()
        configuration = cam.create_preview_configuration(
            main={
                "size": (settings.camera_width, settings.camera_height),
                "format": _picamera_frame_format(),
            }
        )
        cam.configure(configuration)
        cam.start()
        if settings.camera_warmup_ms > 0:
            time.sleep(settings.camera_warmup_ms / 1000)
        _persistent_camera = cam
    return _persistent_camera


def capture_frame_fast(settings: Settings) -> dict[str, Any]:
    """Capture rapide depuis la caméra persistante (pas de warmup)."""
    try:
        cam = _get_persistent_picamera2(settings)
        frame = cam.capture_array()
        if frame is None or getattr(frame, "size", 0) == 0:
            return {"ok": False, "error": "Frame vide"}
        return {"ok": True, "frame": _normalize_frame(frame, "picamera2", settings)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


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


def _resolve_sources(settings: Settings) -> list[str]:
    if settings.camera_source in {"opencv", "picamera2"}:
        return [settings.camera_source]

    machine = platform.machine().lower()
    if machine.startswith("arm") or machine == "aarch64":
        return ["picamera2", "opencv"]
    return ["opencv", "picamera2"]


def _normalize_frame(frame: Any, source: str, settings: Settings) -> Any:
    if frame is None:
        return None
    if hasattr(frame, "ndim") and frame.ndim == 3:
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    return frame


def _shape_to_list(frame: Any) -> list[int]:
    height, width = frame.shape[:2]
    return [int(height), int(width)]


def _probe_camera_picamera2(settings: Settings) -> dict[str, Any]:
    attempt: dict[str, Any] = {
        "source": "picamera2",
        "device": settings.camera_device,
        "read_attempt_count": settings.camera_read_attempts,
        "warmup_ms": settings.camera_warmup_ms,
    }

    try:
        Picamera2 = _load_picamera2_class()
    except Exception as exc:
        attempt["opened"] = False
        attempt["read_ok"] = False
        attempt["error"] = f"Picamera2 unavailable: {exc}"
        return {"ok": False, "attempt": attempt}

    camera = None
    try:
        started_at = time.monotonic()
        camera = Picamera2()
        configuration = camera.create_preview_configuration(
            main={
                "size": (settings.camera_width, settings.camera_height),
                "format": _picamera_frame_format(),
            }
        )
        camera.configure(configuration)
        camera.start()
        attempt["opened"] = True
        attempt["duration_ms"] = round((time.monotonic() - started_at) * 1000, 2)

        if settings.camera_warmup_ms > 0:
            time.sleep(settings.camera_warmup_ms / 1000)

        read_attempts: list[dict[str, Any]] = []
        for attempt_index in range(settings.camera_read_attempts):
            read_started_at = time.monotonic()
            frame = camera.capture_array()
            read_ok = frame is not None and getattr(frame, "size", 0) > 0
            read_attempts.append(
                {
                    "index": attempt_index + 1,
                    "read_ok": read_ok,
                    "duration_ms": round((time.monotonic() - read_started_at) * 1000, 2),
                }
            )
            if read_ok:
                height, width = frame.shape[:2]
                attempt["read_ok"] = True
                attempt["read_attempts"] = read_attempts
                attempt["frame_shape"] = [int(height), int(width)]
                attempt["pixel_format_channels"] = int(frame.shape[2]) if len(frame.shape) == 3 else 1
                return {
                    "ok": True,
                    "camera": attempt,
                    "frame": _normalize_frame(frame, "picamera2", settings),
                    "platform": platform.platform(),
                    "note": "Camera opened and a frame was captured in memory.",
                }
            time.sleep(0.1)

        attempt["read_ok"] = False
        attempt["read_attempts"] = read_attempts
        attempt["error"] = "Picamera2 started but no frame could be captured after repeated attempts."
        return {"ok": False, "attempt": attempt}
    except Exception as exc:
        attempt["read_ok"] = False
        attempt["error"] = f"Picamera2 capture failed: {exc}"
        return {"ok": False, "attempt": attempt}
    finally:
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
            try:
                camera.close()
            except Exception:
                pass


def _probe_camera_opencv(settings: Settings) -> dict[str, Any]:
    device = _resolve_device(settings.camera_device)
    backends = _resolve_backends(settings.camera_backend)
    attempts: list[dict[str, Any]] = []

    for backend_name, backend_flag in backends:
        started_at = time.monotonic()
        capture = cv2.VideoCapture(device, backend_flag)

        try:
            opened = capture.isOpened()
            attempt: dict[str, Any] = {
                "source": "opencv",
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
                frame = _normalize_frame(frame, "opencv", settings)
                height, width = frame.shape[:2]
                attempt["frame_shape"] = [int(height), int(width)]
                attempt["pixel_format_channels"] = int(frame.shape[2]) if len(frame.shape) == 3 else 1
                attempt["reported_width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                attempt["reported_height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                attempt["reported_fps"] = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                return {
                    "ok": True,
                    "camera": attempt,
                    "frame": frame,
                    "platform": platform.platform(),
                    "note": "Camera opened and a frame was captured in memory.",
                }

            attempt["error"] = "Camera opened but no frame could be read after repeated attempts."
            attempts.append(attempt)
        finally:
            capture.release()

    return {"ok": False, "attempts": attempts}


def probe_camera(settings: Settings) -> dict[str, Any]:
    result = capture_frame(settings)
    if result["ok"]:
        frame = result.pop("frame", None)
        if frame is not None:
            camera = result.get("camera", {})
            camera["frame_shape"] = _shape_to_list(frame)
        return result

    return result


def capture_frame(settings: Settings) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []

    for source in _resolve_sources(settings):
        if source == "picamera2":
            result = _probe_camera_picamera2(settings)
            if result["ok"]:
                return result
            attempts.append(result["attempt"])
            continue

        result = _probe_camera_opencv(settings)
        if result["ok"]:
            return result
        attempts.extend(result["attempts"])

    return {
        "ok": False,
        "platform": platform.platform(),
        "attempts": attempts,
        "error": "Unable to open the camera or read a frame.",
        "hint": "On Raspberry Pi CSI cameras, prefer Picamera2/libcamera over direct OpenCV capture when V4L2 opens but returns no frames.",
    }