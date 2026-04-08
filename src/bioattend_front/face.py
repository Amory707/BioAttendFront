from __future__ import annotations

import time
from typing import Any

import cv2


_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_FACE_CASCADE = cv2.CascadeClassifier(_CASCADE_PATH)


def _to_bbox(face: tuple[int, int, int, int]) -> dict[str, int]:
    x, y, w, h = face
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def _largest_face(faces: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return max(faces, key=lambda f: int(f[2]) * int(f[3]))


def detect_and_crop_face(frame: Any) -> dict[str, Any]:
    if frame is None:
        return {"ok": False, "error": "No frame provided."}

    if _FACE_CASCADE.empty():
        return {"ok": False, "error": "Face cascade classifier could not be loaded."}

    started_at = time.monotonic()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    faces = [tuple(map(int, face)) for face in detected]
    duration_ms = round((time.monotonic() - started_at) * 1000, 2)

    if not faces:
        return {
            "ok": False,
            "face_count": 0,
            "duration_ms": duration_ms,
            "error": "No face detected in frame.",
        }

    primary = _largest_face(faces)
    x, y, w, h = primary
    height, width = frame.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)
    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        return {
            "ok": False,
            "face_count": len(faces),
            "duration_ms": duration_ms,
            "error": "Detected face crop is empty.",
        }

    return {
        "ok": True,
        "duration_ms": duration_ms,
        "face_count": len(faces),
        "faces": [_to_bbox(face) for face in faces],
        "primary_face": _to_bbox(primary),
        "crop_shape": [int(face_crop.shape[0]), int(face_crop.shape[1])],
        "face_crop": face_crop,
        "note": "Face detected and cropped in memory.",
    }