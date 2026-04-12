from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ORT_AVAILABLE = False

# Configurations de modèles Silent-Face-Anti-Spoofing.
# Le nom du fichier ONNX encode le facteur d'échelle et la taille d'entrée :
# {scale}_{width}x{height}_{Arch}.onnx
_MODEL_CONFIGS: list[dict[str, Any]] = [
    {"filename": "2.7_80x80_MiniFASNetV2.onnx",  "scale": 2.7, "input_size": 80},
    {"filename": "4_0_0_80x80_MiniFASNetV4.onnx", "scale": 4.0, "input_size": 80},
]

# Cache des sessions ONNX (une par fichier de modèle).
_sessions: dict[str, Any] = {}


def _get_session(model_path: str) -> Any:
    if model_path not in _sessions:
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        _sessions[model_path] = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
    return _sessions[model_path]


def _crop_context(frame: np.ndarray, bbox: dict, scale: float) -> np.ndarray:
    """Découpe une région centrée sur le visage avec un contexte proportionnel au scale."""
    h_img, w_img = frame.shape[:2]
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    cx = x + w / 2.0
    cy = y + h / 2.0
    size = int(max(w, h) * scale)
    x1 = max(0, int(cx - size / 2.0))
    y1 = max(0, int(cy - size / 2.0))
    x2 = min(w_img, x1 + size)
    y2 = min(h_img, y1 + size)
    return frame[y1:y2, x1:x2]


def _preprocess(region: np.ndarray, input_size: int) -> np.ndarray:
    """Pré-traitement Silent-Face : redimensionnement + normalisation [-1, 1] + CHW."""
    img = cv2.resize(region, (input_size, input_size))
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = img.transpose(2, 0, 1)  # HWC → CHW
    return np.expand_dims(img, axis=0)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def check_liveness(
    frame: Any,
    face_bbox: dict,
    model_dir: str,
    threshold: float = 0.6,
    live_class_idx: int = 1,
) -> dict[str, Any]:
    """
    Vérifie qu'un visage est bien réel (anti-spoofing) via les modèles Silent-Face ONNX.

    Paramètres
    ----------
    frame          : Image BGR complète (numpy array).
    face_bbox      : Dict avec clés x, y, w, h (résultat detect_and_crop_face).
    model_dir      : Répertoire contenant les fichiers .onnx.
    threshold      : Score minimal pour considérer le visage comme réel (0–1).
    live_class_idx : Index de la classe "visage réel" dans la sortie du modèle.
                     Ajustez si les résultats sont inversés (0 ou 1 selon le modèle).

    Retour
    ------
    {"ok": True, "is_live": bool, "score": float, "models_used": int, "duration_ms": float}
    {"ok": False, "error": str, ...} si les modèles sont absents ou si l'inférence échoue.
    """
    if not _ORT_AVAILABLE:
        return {"ok": False, "error": "onnxruntime non disponible"}

    dir_path = Path(model_dir)
    available: list[tuple[str, dict]] = [
        (str(dir_path / cfg["filename"]), cfg)
        for cfg in _MODEL_CONFIGS
        if (dir_path / cfg["filename"]).exists()
    ]

    if not available:
        return {
            "ok": False,
            "error": f"Aucun modèle liveness trouvé dans {model_dir}",
            "hint": (
                "Générez les modèles ONNX avec scripts/make_liveness_onnx.py "
                "ou placez-les manuellement dans LIVENESS_MODEL_DIR."
            ),
        }

    started_at = time.monotonic()
    scores: list[float] = []
    warnings: list[str] = []
    raw_outputs: list[list[float]] = []

    for model_path, cfg in available:
        try:
            region = _crop_context(frame, face_bbox, cfg["scale"])
            if region.size == 0:
                warnings.append(f"{cfg['filename']}: crop vide, ignoré")
                continue
            tensor = _preprocess(region, cfg["input_size"])
            session = _get_session(model_path)
            input_name = session.get_inputs()[0].name
            raw = session.run(None, {input_name: tensor})[0][0]
            probs = _softmax(raw).tolist()
            raw_outputs.append([round(float(p), 4) for p in probs])
            idx = live_class_idx if live_class_idx < len(probs) else 0
            scores.append(float(probs[idx]))
        except Exception as exc:
            warnings.append(f"{cfg['filename']}: {exc}")

    duration_ms = round((time.monotonic() - started_at) * 1000, 2)

    if not scores:
        return {
            "ok": False,
            "error": "Toutes les inférences liveness ont échoué",
            "warnings": warnings,
            "duration_ms": duration_ms,
        }

    final_score = float(np.mean(scores))
    is_live = final_score >= threshold

    result: dict[str, Any] = {
        "ok": True,
        "is_live": is_live,
        "score": round(final_score, 4),
        "threshold": threshold,
        "models_used": len(scores),
        "raw_outputs": raw_outputs,
        "duration_ms": duration_ms,
    }
    if warnings:
        result["warnings"] = warnings
    return result
