from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np


_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_FACE_CASCADE = cv2.CascadeClassifier(_CASCADE_PATH)
_EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
_EYE_CASCADE = cv2.CascadeClassifier(_EYE_CASCADE_PATH)

# Taille minimale (en px) qu'un côté du visage doit atteindre pour être
# considéré comme valide. En dessous, on retourne "aucun visage".
_MIN_FACE_SIZE_PX = 80
_CENTER_TOLERANCE_RATIO = 0.35
_MIN_FACE_ASPECT_RATIO = 0.72
_MAX_FACE_ASPECT_RATIO = 1.45
_MIN_FACE_AREA_RATIO = 0.015
_MAX_FACE_AREA_RATIO = 0.65
_REQUIRE_EYE_CHECK = True


def _to_bbox(face: tuple[int, int, int, int]) -> dict[str, int]:
    x, y, w, h = face
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def _largest_face(faces: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return max(faces, key=lambda f: int(f[2]) * int(f[3]))


def _looks_like_face(
    gray: np.ndarray,
    face: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> tuple[bool, str]:
    x, y, w, h = [int(v) for v in face]
    if w < _MIN_FACE_SIZE_PX or h < _MIN_FACE_SIZE_PX:
        return False, "too_small"

    aspect_ratio = float(w) / float(max(1, h))
    if not (_MIN_FACE_ASPECT_RATIO <= aspect_ratio <= _MAX_FACE_ASPECT_RATIO):
        return False, "bad_aspect_ratio"

    area_ratio = float(w * h) / float(max(1, frame_w * frame_h))
    if not (_MIN_FACE_AREA_RATIO <= area_ratio <= _MAX_FACE_AREA_RATIO):
        return False, "bad_area_ratio"

    # Vérifie la présence d'au moins un oeil dans la moitié haute du visage.
    # Ce garde-fou réduit fortement les faux positifs (mains, objets, textures).
    if _REQUIRE_EYE_CHECK and not _EYE_CASCADE.empty():
        top_h = max(1, int(h * 0.65))
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + top_h)
        if x2 > x1 and y2 > y1:
            roi = gray[y1:y2, x1:x2]
            min_eye = max(12, int(min(w, h) * 0.12))
            eyes = _EYE_CASCADE.detectMultiScale(
                roi,
                scaleFactor=1.08,
                minNeighbors=6,
                minSize=(min_eye, min_eye),
            )
            if len(eyes) == 0:
                return False, "no_eyes"

    return True, "ok"


def _select_primary_face(
    gray: np.ndarray,
    faces: list[tuple[int, int, int, int]],
    frame_w: int,
    frame_h: int,
) -> tuple[tuple[int, int, int, int] | None, dict[str, Any]]:
    cx = frame_w / 2.0
    cy = frame_h / 2.0
    tol_x = frame_w * _CENTER_TOLERANCE_RATIO
    tol_y = frame_h * _CENTER_TOLERANCE_RATIO

    valid_shape: list[tuple[int, int, int, int]] = []
    reject_reasons: dict[str, int] = {}
    for f in faces:
        ok, reason = _looks_like_face(gray, f, frame_w, frame_h)
        if ok:
            valid_shape.append(f)
        else:
            reject_reasons[reason] = int(reject_reasons.get(reason, 0)) + 1

    guard: dict[str, Any] = {
        "min_face_size_px": _MIN_FACE_SIZE_PX,
        "center_tolerance_ratio": _CENTER_TOLERANCE_RATIO,
        "all_candidates": len(faces),
        "shape_candidates": len(valid_shape),
        "rejected_reasons": reject_reasons,
        "eye_check_enabled": _REQUIRE_EYE_CHECK and not _EYE_CASCADE.empty(),
    }

    # Si aucun visage ne dépasse le seuil minimal, on refuse proprement.
    if not valid_shape:
        guard["centered_candidates"] = 0
        guard["rejected_all_candidates"] = True
        return None, guard

    centered = []
    for f in valid_shape:
        x, y, w, h = f
        fx = x + w / 2.0
        fy = y + h / 2.0
        if abs(fx - cx) <= tol_x and abs(fy - cy) <= tol_y:
            centered.append(f)

    guard["centered_candidates"] = len(centered)
    guard["rejected_all_candidates"] = False

    candidates = centered if centered else valid_shape
    primary = _largest_face(candidates)
    return primary, guard


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
        {"gray": gray, "scale_factor": 1.10, "min_neighbors": 6, "min_size": (96, 96)},
        {"gray": gray_eq, "scale_factor": 1.08, "min_neighbors": 5, "min_size": (80, 80)},
        {"gray": gray_clahe, "scale_factor": 1.06, "min_neighbors": 4, "min_size": (64, 64)},
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
    primary, guard = _select_primary_face(gray, faces, width, height)

    if primary is None:
        return {
            "ok": False,
            "face_count": len(faces),
            "duration_ms": duration_ms,
            "guard": guard,
            "error": "No face detected in frame.",
        }

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