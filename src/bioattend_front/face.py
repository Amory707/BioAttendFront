from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np


_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_FACE_CASCADE = cv2.CascadeClassifier(_CASCADE_PATH)

# Paramètres de cadrage utilisés seulement comme métriques de diagnostic.
_MIN_FACE_SIZE_PX = 96
_CENTER_TOLERANCE_RATIO = 0.35


def _to_bbox(face: tuple[int, int, int, int]) -> dict[str, int]:
    x, y, w, h = face
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def _largest_face(faces: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return max(faces, key=lambda f: int(f[2]) * int(f[3]))


def _select_primary_face(
    faces: list[tuple[int, int, int, int]],
    frame_w: int,
    frame_h: int,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    cx = frame_w / 2.0
    cy = frame_h / 2.0
    tol_x = frame_w * _CENTER_TOLERANCE_RATIO
    tol_y = frame_h * _CENTER_TOLERANCE_RATIO

    valid_size = [f for f in faces if int(f[2]) >= _MIN_FACE_SIZE_PX and int(f[3]) >= _MIN_FACE_SIZE_PX]

    centered = []
    for f in valid_size:
        x, y, w, h = f
        fx = x + w / 2.0
        fy = y + h / 2.0
        if abs(fx - cx) <= tol_x and abs(fy - cy) <= tol_y:
            centered.append(f)

    # Ne jamais bloquer ici: on préfère un candidat pour laisser la liveness décider.
    candidates = centered if centered else (valid_size if valid_size else faces)
    primary = _largest_face(candidates)
    return primary, {
        "min_face_size_px": _MIN_FACE_SIZE_PX,
        "center_tolerance_ratio": _CENTER_TOLERANCE_RATIO,
        "centered_candidates": len(centered),
        "size_candidates": len(valid_size),
        "all_candidates": len(faces),
    }


def _run_haar(
    gray: np.ndarray,
    *,
    scale_factor: float,
    min_neighbors: int,
    min_size: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    detected = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    return [tuple(map(int, face)) for face in detected]


def detect_and_crop_face(frame: Any) -> dict[str, Any]:
    if frame is None:
        return {"ok": False, "error": "No frame provided."}

    if _FACE_CASCADE.empty():
        return {"ok": False, "error": "Face cascade classifier could not be loaded."}

    started_at = time.monotonic()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Multi-pass OpenCV Haar pour être plus robuste en conditions réelles
    # (lumière variable, visage un peu éloigné, léger mouvement).
    gray_eq = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    passes = [
        {"gray": gray, "scale_factor": 1.10, "min_neighbors": 5, "min_size": (80, 80)},
        {"gray": gray_eq, "scale_factor": 1.08, "min_neighbors": 4, "min_size": (64, 64)},
        {"gray": gray_eq, "scale_factor": 1.05, "min_neighbors": 3, "min_size": (48, 48)},
        {"gray": gray_clahe, "scale_factor": 1.03, "min_neighbors": 2, "min_size": (36, 36)},
    ]

    faces: list[tuple[int, int, int, int]] = []
    matched_pass: dict[str, Any] | None = None
    for p in passes:
        faces = _run_haar(
            p["gray"],
            scale_factor=float(p["scale_factor"]),
            min_neighbors=int(p["min_neighbors"]),
            min_size=tuple(p["min_size"]),
        )
        if faces:
            matched_pass = {
                "scale_factor": p["scale_factor"],
                "min_neighbors": p["min_neighbors"],
                "min_size": list(p["min_size"]),
            }
            break

    duration_ms = round((time.monotonic() - started_at) * 1000, 2)

    if not faces:
        return {
            "ok": False,
            "face_count": 0,
            "duration_ms": duration_ms,
            "error": "No face detected in frame.",
        }

    height, width = frame.shape[:2]
    primary, guard = _select_primary_face(faces, width, height)
    x, y, w, h = primary

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
        "detector": "opencv_haar",
        "matched_pass": matched_pass,
        "guard": guard,
        "faces": [_to_bbox(face) for face in faces],
        "primary_face": _to_bbox(primary),
        "crop_shape": [int(face_crop.shape[0]), int(face_crop.shape[1])],
        "face_crop": face_crop,
        "note": "Face detected and cropped in memory.",
    }