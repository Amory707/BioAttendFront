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
    {"filename": "4_0_0_80x80_MiniFASNetV1SE.onnx", "scale": 4.0, "input_size": 80},
]
_FACENOX_MODEL_FILENAME = "facenox_best_model.onnx"
_FACENOX_INPUT_SIZE = 128
_FACENOX_BBOX_EXPANSION = 1.5

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


def _context_bounds(frame: np.ndarray, bbox: dict, scale: float) -> tuple[int, int, int, int]:
    """Calcule les bornes du crop de contexte centré sur le visage."""
    h_img, w_img = frame.shape[:2]
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    cx = x + w / 2.0
    cy = y + h / 2.0
    size = int(max(w, h) * scale)
    x1 = max(0, int(cx - size / 2.0))
    y1 = max(0, int(cy - size / 2.0))
    x2 = min(w_img, x1 + size)
    y2 = min(h_img, y1 + size)
    return x1, y1, x2, y2


def _crop_context(frame: np.ndarray, bbox: dict, scale: float) -> np.ndarray:
    """Découpe une région centrée sur le visage avec un contexte proportionnel au scale."""
    x1, y1, x2, y2 = _context_bounds(frame, bbox, scale)
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


def _to_logit_threshold(probability_threshold: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, float(probability_threshold)))
    return float(np.log(p / (1.0 - p)))


def _resolve_live_class_idx(requested_idx: int, class_count: int) -> tuple[int, str | None]:
    if class_count <= 0:
        return 0, "Invalid class_count"
    if requested_idx < 0:
        return 0, f"LIVENESS_LIVE_CLASS_IDX={requested_idx} < 0, fallback to 0"
    if requested_idx >= class_count:
        return 0, f"LIVENESS_LIVE_CLASS_IDX={requested_idx} out of range for {class_count} classes, fallback to 0"
    return requested_idx, None


def _crop_square_with_reflect(frame: np.ndarray, bbox: dict, expansion_factor: float) -> tuple[np.ndarray, dict[str, int]]:
    """Extract a square crop around the face bbox; pad edges using reflection."""
    h_img, w_img = frame.shape[:2]
    x = int(bbox.get("x", 0))
    y = int(bbox.get("y", 0))
    w = int(bbox.get("w", 0))
    h = int(bbox.get("h", 0))

    if w <= 0 or h <= 0:
        raise ValueError("Invalid face bbox dimensions")

    max_dim = max(w, h)
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    crop_size = int(max_dim * float(expansion_factor))
    if crop_size <= 0:
        raise ValueError("Invalid crop size")

    x1 = int(center_x - crop_size / 2.0)
    y1 = int(center_y - crop_size / 2.0)
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w_img, x2)
    src_y2 = min(h_img, y2)

    top_pad = max(0, -y1)
    left_pad = max(0, -x1)
    bottom_pad = max(0, y2 - h_img)
    right_pad = max(0, x2 - w_img)

    if src_x2 <= src_x1 or src_y2 <= src_y1:
        region = np.zeros((1, 1, 3), dtype=frame.dtype)
    else:
        region = frame[src_y1:src_y2, src_x1:src_x2]

    crop = cv2.copyMakeBorder(
        region,
        top_pad,
        bottom_pad,
        left_pad,
        right_pad,
        cv2.BORDER_REFLECT_101,
    )

    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)

    return crop, {
        "x": int(x1),
        "y": int(y1),
        "w": int(crop_size),
        "h": int(crop_size),
    }


def _preprocess_facenox(region: np.ndarray, input_size: int) -> np.ndarray:
    """Facenox preprocess: letterbox with reflection + normalize [0,1] + CHW."""
    old_h, old_w = region.shape[:2]
    ratio = float(input_size) / float(max(old_h, old_w))
    scaled_h = max(1, int(old_h * ratio))
    scaled_w = max(1, int(old_w * ratio))
    interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
    img = cv2.resize(region, (scaled_w, scaled_h), interpolation=interpolation)

    delta_w = input_size - scaled_w
    delta_h = input_size - scaled_h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


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

    facenox_model_path = dir_path / _FACENOX_MODEL_FILENAME
    if facenox_model_path.exists():
        started_at = time.monotonic()
        warnings: list[str] = []

        try:
            region, region_bbox = _crop_square_with_reflect(frame, face_bbox, _FACENOX_BBOX_EXPANSION)
            # Facenox demo effectue l'inférence en RGB; notre frame caméra est en BGR.
            region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            tensor = _preprocess_facenox(region_rgb, _FACENOX_INPUT_SIZE)
            session = _get_session(str(facenox_model_path))
            input_name = session.get_inputs()[0].name
            raw = session.run(None, {input_name: tensor})[0][0]

            if len(raw) < 2:
                raise ValueError("Unexpected Facenox output shape")

            resolved_live_idx, idx_warning = _resolve_live_class_idx(live_class_idx, len(raw))
            if idx_warning:
                warnings.append(idx_warning)

            real_logit = float(raw[resolved_live_idx])
            other_logits = [float(raw[i]) for i in range(len(raw)) if i != resolved_live_idx]
            spoof_logit = max(other_logits) if other_logits else float(raw[resolved_live_idx])
            logit_diff = float(real_logit - spoof_logit)
            threshold_logit = _to_logit_threshold(threshold)
            is_live = logit_diff >= threshold_logit

            probs_all = _softmax(np.array(raw, dtype=np.float32))
            real_prob = float(probs_all[resolved_live_idx])
            duration_ms = round((time.monotonic() - started_at) * 1000, 2)

            result: dict[str, Any] = {
                "ok": True,
                "is_live": bool(is_live),
                "score": round(real_prob, 4),
                "threshold": threshold,
                "threshold_logit": round(threshold_logit, 4),
                "logit_diff": round(logit_diff, 4),
                "live_class_idx": int(resolved_live_idx),
                "models_used": 1,
                "backend": "facenox_best",
                "raw_outputs": [[round(float(x), 4) for x in raw]],
                "model_regions": [
                    {
                        "model": _FACENOX_MODEL_FILENAME,
                        "scale": _FACENOX_BBOX_EXPANSION,
                        "bbox": region_bbox,
                    }
                ],
                "duration_ms": duration_ms,
            }
            if warnings:
                result["warnings"] = warnings
            return result
        except Exception as exc:
            warnings.append(f"{_FACENOX_MODEL_FILENAME}: {exc}")
            duration_ms = round((time.monotonic() - started_at) * 1000, 2)
            return {
                "ok": False,
                "error": "Échec d'inférence Facenox",
                "backend": "facenox_best",
                "warnings": warnings,
                "duration_ms": duration_ms,
            }

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
                f"Placez {_FACENOX_MODEL_FILENAME} dans LIVENESS_MODEL_DIR "
                "(recommandé) ou les modèles Silent-Face classiques."
            ),
        }

    started_at = time.monotonic()
    scores: list[float] = []
    warnings: list[str] = []
    raw_outputs: list[list[float]] = []
    model_regions: list[dict[str, Any]] = []

    for model_path, cfg in available:
        try:
            x1, y1, x2, y2 = _context_bounds(frame, face_bbox, cfg["scale"])
            region = frame[y1:y2, x1:x2]
            if region.size == 0:
                warnings.append(f"{cfg['filename']}: crop vide, ignoré")
                continue

            model_regions.append(
                {
                    "model": cfg["filename"],
                    "scale": cfg["scale"],
                    "bbox": {
                        "x": int(x1),
                        "y": int(y1),
                        "w": int(max(0, x2 - x1)),
                        "h": int(max(0, y2 - y1)),
                    },
                }
            )

            tensor = _preprocess(region, cfg["input_size"])
            session = _get_session(model_path)
            input_name = session.get_inputs()[0].name
            raw = session.run(None, {input_name: tensor})[0][0]
            probs = _softmax(raw).tolist()
            raw_outputs.append([round(float(p), 4) for p in probs])
            idx, idx_warning = _resolve_live_class_idx(live_class_idx, len(probs))
            if idx_warning:
                warnings.append(f"{cfg['filename']}: {idx_warning}")
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
        "backend": "silent_face_ensemble",
        "raw_outputs": raw_outputs,
        "model_regions": model_regions,
        "duration_ms": duration_ms,
    }
    if warnings:
        result["warnings"] = warnings
    return result
