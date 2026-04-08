from __future__ import annotations

import time
from threading import Lock
from typing import Any

import numpy as np

from .config import Settings


_MODEL_LOCK = Lock()
_FACE_ANALYSIS: Any = None
_MODEL_SIGNATURE: tuple[str, int, int] | None = None


def _largest_face(faces: list[Any]) -> Any:
    return max(faces, key=lambda f: int((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _select_face_by_bbox(faces: list[Any], target_bbox: dict[str, int] | None) -> Any:
    if not faces:
        return None
    if not target_bbox:
        return _largest_face(faces)

    tx1 = float(target_bbox["x"])
    ty1 = float(target_bbox["y"])
    tx2 = float(target_bbox["x"] + target_bbox["w"])
    ty2 = float(target_bbox["y"] + target_bbox["h"])
    target = (tx1, ty1, tx2, ty2)

    best_face = None
    best_score = -1.0
    for candidate in faces:
        x1, y1, x2, y2 = [float(v) for v in candidate.bbox]
        score = _bbox_iou((x1, y1, x2, y2), target)
        if score > best_score:
            best_face = candidate
            best_score = score

    if best_face is not None:
        return best_face
    return _largest_face(faces)


def _get_face_analysis(settings: Settings) -> Any:
    global _FACE_ANALYSIS, _MODEL_SIGNATURE
    signature = (
        settings.insightface_model_name,
        settings.insightface_det_width,
        settings.insightface_det_height,
    )

    if _FACE_ANALYSIS is not None and _MODEL_SIGNATURE == signature:
        return _FACE_ANALYSIS

    with _MODEL_LOCK:
        if _FACE_ANALYSIS is not None and _MODEL_SIGNATURE == signature:
            return _FACE_ANALYSIS

        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name=settings.insightface_model_name,
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(settings.insightface_det_width, settings.insightface_det_height))
        _FACE_ANALYSIS = app
        _MODEL_SIGNATURE = signature
        return _FACE_ANALYSIS


def generate_embedding(
    frame: Any,
    settings: Settings,
    target_bbox: dict[str, int] | None = None,
    fallback_face_crop: Any | None = None,
) -> dict[str, Any]:
    if frame is None and fallback_face_crop is None:
        return {"ok": False, "error": "No frame or face crop provided."}

    started_at = time.monotonic()
    try:
        app = _get_face_analysis(settings)
    except Exception as exc:
        return {"ok": False, "error": f"InsightFace initialization failed: {exc}"}

    faces: list[Any] = []
    mode = "frame"
    if frame is not None:
        try:
            faces = app.get(frame)
        except Exception as exc:
            return {"ok": False, "error": f"InsightFace inference failed on frame: {exc}"}

    if not faces and fallback_face_crop is not None:
        mode = "face_crop_fallback"
        try:
            faces = app.get(fallback_face_crop)
        except Exception as exc:
            return {"ok": False, "error": f"InsightFace inference failed on face crop fallback: {exc}"}

    duration_ms = round((time.monotonic() - started_at) * 1000, 2)
    if not faces:
        return {
            "ok": False,
            "duration_ms": duration_ms,
            "error": "No embedding could be generated from frame or face crop fallback.",
        }

    primary = _select_face_by_bbox(faces, target_bbox)
    if primary is None:
        return {
            "ok": False,
            "duration_ms": duration_ms,
            "error": "Unable to select a face candidate for embedding.",
        }

    vector = getattr(primary, "normed_embedding", None)
    if vector is None:
        vector = getattr(primary, "embedding", None)

    if vector is None:
        return {
            "ok": False,
            "duration_ms": duration_ms,
            "error": "Embedding attribute is missing in InsightFace output.",
        }

    embedding = np.asarray(vector, dtype=np.float32)
    return {
        "ok": True,
        "duration_ms": duration_ms,
        "mode": mode,
        "detected_face_count": len(faces),
        "embedding_dimension": int(embedding.shape[0]),
        "embedding_l2_norm": float(np.linalg.norm(embedding)),
        "embedding_preview": [float(v) for v in embedding[:8]],
        "embedding": embedding.tolist(),
        "note": "Embedding generated in memory.",
    }