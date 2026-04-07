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


def generate_embedding(face_crop: Any, settings: Settings) -> dict[str, Any]:
    if face_crop is None:
        return {"ok": False, "error": "No face crop provided."}

    started_at = time.monotonic()
    try:
        app = _get_face_analysis(settings)
    except Exception as exc:
        return {"ok": False, "error": f"InsightFace initialization failed: {exc}"}

    try:
        faces = app.get(face_crop)
    except Exception as exc:
        return {"ok": False, "error": f"InsightFace inference failed: {exc}"}

    duration_ms = round((time.monotonic() - started_at) * 1000, 2)
    if not faces:
        return {
            "ok": False,
            "duration_ms": duration_ms,
            "error": "No embedding could be generated from the provided face crop.",
        }

    primary = _largest_face(faces)
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
        "detected_face_count": len(faces),
        "embedding_dimension": int(embedding.shape[0]),
        "embedding_l2_norm": float(np.linalg.norm(embedding)),
        "embedding_preview": [float(v) for v in embedding[:8]],
        "embedding": embedding.tolist(),
        "note": "Embedding generated in memory.",
    }