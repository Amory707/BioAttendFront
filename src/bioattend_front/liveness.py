from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from .camera import capture_frame_fast
from .config import Settings
from .embedding import _get_face_analysis


_ANTISPOOF_LOCK = threading.Lock()
_ANTISPOOF_MODEL: "AntiSpoofModel | None" = None
_ANTISPOOF_MODEL_PATH: str | None = None


class AntiSpoofModel:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, face_img: np.ndarray) -> float:
        img = cv2.resize(face_img, (80, 80))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        outputs = self.session.run(None, {self.input_name: img})
        logits = np.asarray(outputs[0])

        if logits.ndim >= 2 and logits.shape[-1] >= 2:
            return float(logits.reshape(-1, logits.shape[-1])[0][1])
        return float(np.max(logits))


class _FrameReaderAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings

    def read(self) -> tuple[bool, Any]:
        result = capture_frame_fast(self.settings)
        if not result.get("ok", False):
            return False, None
        return True, result.get("frame")


def _get_antispoof_model(model_path: str) -> AntiSpoofModel:
    global _ANTISPOOF_MODEL, _ANTISPOOF_MODEL_PATH

    if _ANTISPOOF_MODEL is not None and _ANTISPOOF_MODEL_PATH == model_path:
        return _ANTISPOOF_MODEL

    with _ANTISPOOF_LOCK:
        if _ANTISPOOF_MODEL is not None and _ANTISPOOF_MODEL_PATH == model_path:
            return _ANTISPOOF_MODEL
        _ANTISPOOF_MODEL = AntiSpoofModel(model_path)
        _ANTISPOOF_MODEL_PATH = model_path
        return _ANTISPOOF_MODEL


def crop_face(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return frame[y1:y2, x1:x2]


def blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def texture_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def screen_artifacts(face_img: np.ndarray) -> float:
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    return float(np.mean(magnitude))


def eye_ratio(kps: np.ndarray) -> float:
    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]

    eye_dist = np.linalg.norm(left_eye - right_eye)
    eye_center = (left_eye + right_eye) / 2

    if eye_dist == 0:
        return 0.0

    return float(np.linalg.norm(nose - eye_center) / eye_dist)


def detect_blink_sequence(values: list[float]) -> int:
    count = 0
    state = "open"

    for value in values:
        if state == "open" and value < 0.18:
            state = "closed"
        elif state == "closed" and value > 0.22:
            count += 1
            state = "open"

    return count


def head_yaw(kps: np.ndarray) -> float:
    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]

    center = (left_eye + right_eye) / 2
    return float(nose[0] - center[0])


def detect_head_movement(values: list[float]) -> bool:
    if not values:
        return False
    return (max(values) - min(values)) > 10


def depth_variation(seq: list[np.ndarray]) -> float:
    if len(seq) < 2:
        return 0.0

    diffs: list[float] = []
    for index in range(1, len(seq)):
        prev = seq[index - 1]
        curr = seq[index]

        d1 = np.linalg.norm(prev[2] - prev[0])
        d2 = np.linalg.norm(curr[2] - curr[0])

        diffs.append(abs(float(d2 - d1)))

    return float(np.mean(diffs))


def motion_score(seq: list[np.ndarray]) -> float:
    motions: list[float] = []
    for index in range(1, len(seq)):
        motions.append(float(np.linalg.norm(seq[index] - seq[index - 1])))
    return float(np.mean(motions)) if motions else 0.0


def check_liveness(
    camera: Any,
    insight_model: Any,
    anti_spoof: AntiSpoofModel,
    timeout: float = 5.0,
    min_score: float = 0.65,
) -> dict[str, Any]:
    started_at = time.monotonic()

    ear_vals: list[float] = []
    yaw_vals: list[float] = []
    kps_seq: list[np.ndarray] = []

    last_frame: np.ndarray | None = None
    last_face: Any = None

    while True:
        if time.monotonic() - started_at > timeout:
            break

        ok, frame = camera.read()
        if not ok or frame is None:
            return {"ok": False, "error": "camera error"}

        faces = insight_model.get(frame)
        if not faces:
            continue

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        kps = np.asarray(face.kps)

        last_frame = frame
        last_face = face

        ear_vals.append(eye_ratio(kps))
        yaw_vals.append(head_yaw(kps))
        kps_seq.append(kps)

        time.sleep(0.05)

    duration_ms = round((time.monotonic() - started_at) * 1000, 2)

    if last_frame is None or last_face is None:
        return {"ok": False, "duration_ms": duration_ms, "error": "no face"}

    face_img = crop_face(last_frame, np.asarray(last_face.bbox))
    if face_img.size == 0:
        return {"ok": False, "duration_ms": duration_ms, "error": "empty face crop"}

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    anti_score = anti_spoof.predict(face_img)
    screen_score_val = screen_artifacts(face_img)
    texture = texture_score(gray)
    blur = blur_score(gray)

    blink = detect_blink_sequence(ear_vals)
    head_move = detect_head_movement(yaw_vals)

    depth = depth_variation(kps_seq)
    motion = motion_score(kps_seq)

    score = 0.0

    if anti_score > 0.5:
        score += 0.3
    if screen_score_val < 120:
        score += 0.1
    if texture > 20:
        score += 0.1
    if depth > 1.0:
        score += 0.2
    if motion > 0.5:
        score += 0.1
    if blink >= 1:
        score += 0.1
    if head_move:
        score += 0.1

    return {
        "ok": True,
        "duration_ms": duration_ms,
        "is_live": score > min_score,
        "score": round(score, 2),
        "min_score": min_score,
        "frame": last_frame,
        "face_bbox": [float(v) for v in np.asarray(last_face.bbox).tolist()],
        "anti_spoof": anti_score,
        "screen_score": screen_score_val,
        "texture": texture,
        "blur": blur,
        "blink_count": blink,
        "head_movement": head_move,
        "depth_variation": depth,
        "motion": motion,
    }


def run_liveness_check(settings: Settings, include_frame: bool = False) -> dict[str, Any]:
    if not settings.liveness_enabled:
        return {
            "ok": True,
            "enabled": False,
            "is_live": True,
            "score": 1.0,
            "note": "Liveness disabled by configuration.",
        }

    if not settings.liveness_antispoof_model_path:
        return {
            "ok": False,
            "enabled": True,
            "error": "Liveness is enabled but LIVENESS_ANTISPOOF_MODEL_PATH is empty.",
        }

    try:
        anti_spoof = _get_antispoof_model(settings.liveness_antispoof_model_path)
        insight_model = _get_face_analysis(settings)
        camera = _FrameReaderAdapter(settings)
    except Exception as exc:
        return {
            "ok": False,
            "enabled": True,
            "error": f"Liveness initialization failed: {exc}",
        }

    result = check_liveness(
        camera=camera,
        insight_model=insight_model,
        anti_spoof=anti_spoof,
        timeout=float(settings.liveness_timeout_seconds),
        min_score=float(settings.liveness_min_score),
    )
    if not include_frame:
        result.pop("frame", None)
    result["enabled"] = True
    return result
