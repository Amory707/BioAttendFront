from __future__ import annotations

import time
import cv2
from flask import Flask, Response, jsonify, request

from .api_client import identify_embedding, report_event
from .camera import capture_frame, capture_frame_fast, probe_camera
from .config import Settings
from .embedding import generate_embedding
from .face import detect_and_crop_face
from .liveness import check_liveness

_UI_HTML = """\
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BioAttend | Station de pointage</title>
  <style>
    :root {
      --bg-a: #0d1a23;
      --bg-b: #071018;
      --panel: rgba(4, 12, 18, 0.76);
      --line: rgba(255, 255, 255, 0.18);
      --text: #ecf4f7;
      --muted: #a8bdc8;
      --accent: #18a5b2;
      --accent-strong: #0f8792;
      --ok: #3ad17c;
      --bad: #ff6d6d;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    html,
    body {
      width: 100%;
      height: 100%;
      overflow: hidden;
    }

    body {
      background:
        radial-gradient(1000px 500px at 82% -10%, rgba(24, 165, 178, 0.18), transparent 60%),
        radial-gradient(820px 420px at -5% 102%, rgba(58, 209, 124, 0.12), transparent 60%),
        linear-gradient(155deg, var(--bg-a), var(--bg-b));
      color: var(--text);
      font-family: "Segoe UI", "Noto Sans", sans-serif;
      user-select: none;
    }

    .screen {
      width: 100vw;
      height: 100dvh;
      padding: clamp(12px, 2.3vw, 28px);
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: clamp(10px, 1.6vw, 20px);
    }

    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 14px;
    }

    .brand {
      font-size: clamp(1.02rem, 2vw, 1.5rem);
      font-weight: 700;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: #d4e7ee;
      opacity: 0.95;
    }

    .clock {
      font-variant-numeric: tabular-nums;
      color: var(--muted);
      font-size: clamp(0.96rem, 1.6vw, 1.15rem);
      text-align: right;
    }

    .main {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(250px, 28vw);
      gap: clamp(10px, 1.8vw, 20px);
    }

    .camera-card {
      position: relative;
      border-radius: 24px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: #05090d;
      min-height: 0;
    }

    #feed {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
      background: #000;
    }

    .scan-oval {
      position: absolute;
      left: 50%;
      top: 50%;
      width: clamp(220px, 28vw, 360px);
      aspect-ratio: 0.76;
      transform: translate(-50%, -56%);
      border: 4px solid rgba(255, 255, 255, 0.62);
      border-radius: 50%;
      pointer-events: none;
      box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.2) inset;
      transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }

    .scan-oval.success {
      border-color: var(--ok);
      box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.13) inset, 0 0 38px rgba(58, 209, 124, 0.35);
    }

    .scan-oval.error {
      border-color: var(--bad);
      box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.18) inset, 0 0 36px rgba(255, 109, 109, 0.32);
    }

    .guide {
      position: absolute;
      bottom: 16px;
      left: 50%;
      transform: translateX(-50%);
      padding: 8px 14px;
      border-radius: 999px;
      font-size: 0.9rem;
      color: #dbe8ee;
      background: rgba(4, 10, 16, 0.52);
      border: 1px solid rgba(255, 255, 255, 0.18);
      backdrop-filter: blur(4px);
    }

    .side {
      border-radius: 20px;
      border: 1px solid var(--line);
      background: var(--panel);
      backdrop-filter: blur(8px);
      padding: clamp(14px, 2vw, 20px);
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 14px;
      min-height: 0;
    }

    .status {
      border-radius: 14px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: rgba(255, 255, 255, 0.04);
      min-height: 118px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      text-align: center;
      transition: background 0.25s ease, border-color 0.25s ease;
    }

    .status.success {
      background: rgba(58, 209, 124, 0.13);
      border-color: rgba(58, 209, 124, 0.45);
    }

    .status.error {
      background: rgba(255, 109, 109, 0.12);
      border-color: rgba(255, 109, 109, 0.45);
    }

    .status-title {
      font-size: clamp(1.05rem, 1.7vw, 1.4rem);
      font-weight: 700;
      line-height: 1.2;
      color: #eef6fa;
    }

    .status-info {
      margin-top: 6px;
      color: var(--muted);
      font-size: clamp(0.9rem, 1.35vw, 1.02rem);
    }

    .actions {
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }

    .btn {
      width: 100%;
      border: 0;
      border-radius: 12px;
      padding: 14px 16px;
      font-size: clamp(1rem, 1.5vw, 1.18rem);
      font-weight: 700;
      letter-spacing: 0.09em;
      text-transform: uppercase;
      cursor: pointer;
      transition: transform 0.15s ease, background 0.2s ease, opacity 0.2s ease;
    }

    .btn:active:not(:disabled) {
      transform: scale(0.985);
    }

    .btn:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }

    .btn-primary {
      background: linear-gradient(180deg, var(--accent), var(--accent-strong));
      color: #f3fcff;
    }

    .btn-secondary {
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.22);
      color: #e8f3f7;
    }

    .hint {
      color: #94aebb;
      font-size: 0.9rem;
      text-align: center;
      line-height: 1.35;
    }

    .footer {
      text-align: center;
      color: #88a2af;
      font-size: 0.84rem;
      opacity: 0.9;
    }

    @media (max-width: 980px) {
      .main {
        grid-template-columns: 1fr;
        grid-template-rows: 1fr auto;
      }

      .camera-card {
        min-height: 52vh;
      }
    }

    @media (max-width: 560px) {
      .screen {
        padding: 10px;
      }

      .brand {
        letter-spacing: 0.12em;
      }
    }
  </style>
</head>
<body>
  <div class="screen">
    <header class="topbar">
      <h1 class="brand">BioAttend</h1>
      <div class="clock" id="clock">--:--:--</div>
    </header>

    <main class="main">
      <section class="camera-card">
        <img id="feed" src="/snapshot" alt="Flux camera">
        <div class="scan-oval" id="scanOval"></div>
        <div class="guide">Placez votre visage dans l'ovale</div>
      </section>

      <aside class="side">
        <div class="status" id="status">
          <div class="status-title" id="statusTitle">Pret pour pointage</div>
          <div class="status-info" id="statusInfo">Appuyez sur Pointer ou sur Espace</div>
        </div>
        <div></div>
        <div class="actions">
          <button class="btn btn-primary" id="btnPointage" type="button">Pointer</button>
          <button class="btn btn-secondary" id="btnFullscreen" type="button">Plein ecran</button>
            <p class="hint" id="kioskHint">Conseil: lancez le navigateur en mode kiosk pour un vrai plein ecran permanent.</p>
        </div>
      </aside>
    </main>

    <footer class="footer">Station de pointage locale</footer>
  </div>

  <script>
    var feed = document.getElementById('feed');
    var btnPointage = document.getElementById('btnPointage');
    var btnFullscreen = document.getElementById('btnFullscreen');
    var scanOval = document.getElementById('scanOval');
    var status = document.getElementById('status');
    var statusTitle = document.getElementById('statusTitle');
    var statusInfo = document.getElementById('statusInfo');
    var clock = document.getElementById('clock');
    var kioskHint = document.getElementById('kioskHint');
    var KIOSK_MODE = __KIOSK_MODE__;
    var CAMERA_MIRROR = __CAMERA_MIRROR__;
    var SHOW_BOXES = new URLSearchParams(window.location.search).get('boxes') === '1';

    var streamRunning = false;
    var streamTimer = null;

    function setStatus(mode, title, info) {
      status.className = 'status' + (mode ? ' ' + mode : '');
      scanOval.className = 'scan-oval' + (mode ? ' ' + mode : '');
      statusTitle.textContent = title;
      statusInfo.textContent = info;
    }

    function scheduleNextFrame(delay) {
      if (!streamRunning) {
        return;
      }
      clearTimeout(streamTimer);
      streamTimer = setTimeout(function() {
        feed.src = '/snapshot?boxes=' + (SHOW_BOXES ? '1' : '0') + '&t=' + Date.now();
      }, delay);
    }

    function startStream() {
      if (streamRunning) {
        return;
      }
      streamRunning = true;
      scheduleNextFrame(0);
    }

    function stopStream() {
      streamRunning = false;
      clearTimeout(streamTimer);
      streamTimer = null;
    }

    feed.addEventListener('load', function() {
      scheduleNextFrame(90);
    });

    feed.addEventListener('error', function() {
      scheduleNextFrame(180);
    });

    function updateClock() {
      var now = new Date();
      var time = now.toLocaleTimeString('fr-FR');
      var date = now.toLocaleDateString('fr-FR', {
        weekday: 'short',
        day: '2-digit',
        month: '2-digit',
        year: 'numeric'
      });
      clock.textContent = time + '  |  ' + date;
    }

    async function enterFullscreen() {
      var el = document.documentElement;
      if (!document.fullscreenElement && el.requestFullscreen) {
        try {
          await el.requestFullscreen();
        } catch (e) {
          return;
        }
      }
    }

    async function startPointage() {
      btnPointage.disabled = true;
      stopStream();
      setStatus('', 'Identification en cours...', 'Veuillez patienter');
      try {
        var resp = await fetch('/pointage', { method: 'POST' });
        var data = await resp.json();
        if (data.ok && data.matched) {
          var type = data.pointage_type === 'ENTREE' ? 'Entr\u00e9e' : 'Sortie';
          var heure = new Date().toLocaleTimeString('fr-FR');
          setStatus('success', 'Identifie: ' + data.full_name, type + ' a ' + heure);
        } else {
          var errorType = data.error_type || 'recognition_failed';
          var titleByType = {
            no_face_detected: 'Visage non detecte',
            recognition_failed: 'Echec de reconnaissance',
            unknown_user: 'Utilisateur inconnu',
            spoof_attempt: 'Tentative d\'usurpation detectee'
          };
          var loggedText = data.event_logged === true ? 'journalise plateforme: oui' : 'journalise plateforme: non';
          setStatus('error', titleByType[errorType] || 'Non reconnu', (data.error || 'Veuillez reessayer') + ' | ' + loggedText);
        }
      } catch (e) {
        setStatus('error', 'Erreur reseau', 'Connexion API indisponible');
      }
      setTimeout(function() {
        setStatus('', 'Pret pour pointage', 'Appuyez sur Pointer ou sur Espace');
        btnPointage.disabled = false;
        startStream();
      }, 3500);
    }

    btnPointage.addEventListener('click', startPointage);
    btnFullscreen.addEventListener('click', enterFullscreen);

    document.addEventListener('keydown', function(e) {
      if ((e.code === 'Space' || e.code === 'Enter') && !btnPointage.disabled) {
        e.preventDefault();
        startPointage();
      }
      if (e.key === 'f' || e.key === 'F') {
        enterFullscreen();
      }
    });

    if (KIOSK_MODE) {
      btnFullscreen.style.display = 'none';
      kioskHint.textContent = 'Mode kiosk actif';

      document.addEventListener('pointerdown', function autoKiosk() {
        enterFullscreen();
        document.removeEventListener('pointerdown', autoKiosk);
      }, { once: true });

      document.addEventListener('keydown', function autoKioskKey() {
        enterFullscreen();
        document.removeEventListener('keydown', autoKioskKey);
      }, { once: true });
    }

    if (CAMERA_MIRROR) {
      feed.style.transform = 'scaleX(-1)';
    }

    updateClock();
    setInterval(updateClock, 1000);
    startStream();
  </script>
</body>
</html>
"""


def create_app() -> Flask:
    settings = Settings.from_env()
    app = Flask(__name__)
    app.config["BIOATTEND_SETTINGS"] = settings
    platform_event_types = {"recognition_failed", "unknown_user", "spoof_attempt"}

    def _bbox_iou(a: dict, b: dict) -> float:
        ax1, ay1 = int(a.get("x", 0)), int(a.get("y", 0))
        ax2, ay2 = ax1 + int(a.get("w", 0)), ay1 + int(a.get("h", 0))
        bx1, by1 = int(b.get("x", 0)), int(b.get("y", 0))
        bx2, by2 = bx1 + int(b.get("w", 0)), by1 + int(b.get("h", 0))

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return float(inter / union)

    def _emit_platform_event(
        event_type: str,
        *,
        status: str,
        message: str,
        details: dict | None = None,
    ) -> tuple[str, dict]:
        normalized_type = event_type if event_type in platform_event_types else "recognition_failed"
        event_result = report_event(
            normalized_type,
            settings,
            status=status,
            message=message,
            details=details,
        )
        return normalized_type, event_result

    def _is_reliable_face_result(face_result: dict, frame_w: int, frame_h: int) -> bool:
        primary = face_result.get("primary_face") or {}
        guard = face_result.get("guard") or {}

        w = int(primary.get("w", 0))
        h = int(primary.get("h", 0))
        x = int(primary.get("x", 0))
        y = int(primary.get("y", 0))

        if w < 72 or h < 72:
            return False

        if int(guard.get("centered_candidates", 0)) <= 0:
            return False

        margin_x = max(2, int(frame_w * 0.02))
        margin_y = max(2, int(frame_h * 0.02))
        if x <= margin_x or y <= margin_y or (x + w) >= (frame_w - margin_x) or (y + h) >= (frame_h - margin_y):
            return False

        return True

    def _capture_with_face(max_attempts: int = 6, delay_ms: int = 70) -> dict:
        last_capture: dict | None = None
        last_face: dict | None = None
        pending_bbox: dict | None = None
        for attempt_idx in range(max_attempts):
            capture_result = capture_frame_fast(settings)
            if not capture_result.get("ok", False):
                capture_result.pop("frame", None)
                last_capture = capture_result
                pending_bbox = None
            else:
                frame = capture_result.pop("frame")
                face_result = detect_and_crop_face(frame)
                if face_result.get("ok", False):
                    frame_h, frame_w = frame.shape[:2]
                    if not _is_reliable_face_result(face_result, frame_w, frame_h):
                        pending_bbox = None
                        face_result["ok"] = False
                        face_result["error"] = "No reliable centered face detected."
                    else:
                        current_bbox = face_result.get("primary_face") or {}
                        if pending_bbox is None:
                            pending_bbox = current_bbox
                            face_result.pop("face_crop", None)
                            last_capture = capture_result
                            last_face = face_result
                        else:
                            iou = _bbox_iou(pending_bbox, current_bbox)
                            if iou >= 0.18:
                                return {
                                    "ok": True,
                                    "frame": frame,
                                    "camera": capture_result.get("camera"),
                                    "platform": capture_result.get("platform"),
                                    "face": face_result,
                                    "attempts_used": attempt_idx + 1,
                                    "face_confirmed_iou": round(iou, 4),
                                }
                            pending_bbox = current_bbox
                face_result.pop("face_crop", None)
                last_capture = capture_result
                last_face = face_result

            if attempt_idx < max_attempts - 1 and delay_ms > 0:
                time.sleep(delay_ms / 1000)

        if last_capture is None:
            last_capture = {}
        return {
            "ok": False,
            "error_code": "no_face" if last_face is not None else "capture_failed",
            "error": "No face detected in sampled frames." if last_face is not None else "Capture failed.",
            "camera": last_capture.get("camera"),
            "platform": last_capture.get("platform"),
            "face": last_face,
            "attempts_used": max_attempts,
        }

    @app.get("/health")
    def health() -> tuple[object, int]:
        return jsonify({"ok": True, "service": "bioattend-front"}), 200

    @app.get("/diagnostics/config")
    def diagnostics_config() -> tuple[object, int]:
        return jsonify({"ok": True, "config": settings.as_public_dict()}), 200

    @app.post("/diagnostics/camera")
    @app.get("/diagnostics/camera")
    def diagnostics_camera() -> tuple[object, int]:
        result = probe_camera(settings)
        status_code = 200 if result["ok"] else 503
        return jsonify(result), status_code

    @app.post("/diagnostics/face")
    @app.get("/diagnostics/face")
    def diagnostics_face() -> tuple[object, int]:
        sample = _capture_with_face()
        if not sample.get("ok", False):
            status_code = 422 if sample.get("error_code") == "no_face" else 503
            return jsonify(sample), status_code

        face_result = sample["face"]
        face_result.pop("face_crop", None)
        response = {
            "ok": face_result.get("ok", False),
            "camera": sample.get("camera"),
            "face": face_result,
            "platform": sample.get("platform"),
            "attempts_used": sample.get("attempts_used"),
        }
        return jsonify(response), 200

    @app.post("/diagnostics/embedding")
    @app.get("/diagnostics/embedding")
    def diagnostics_embedding() -> tuple[object, int]:
        sample = _capture_with_face()
        if not sample.get("ok", False):
            status_code = 422 if sample.get("error_code") == "no_face" else 503
            return jsonify(sample), status_code

        frame = sample["frame"]
        face_result = sample["face"]
        face_crop = face_result.pop("face_crop")
        embedding_result = generate_embedding(
            frame=frame,
            settings=settings,
            target_bbox=face_result.get("primary_face"),
            fallback_face_crop=face_crop,
        )
        embedding_result.pop("embedding", None)

        response = {
            "ok": embedding_result.get("ok", False),
            "camera": sample.get("camera"),
            "face": face_result,
            "embedding": embedding_result,
            "platform": sample.get("platform"),
            "attempts_used": sample.get("attempts_used"),
        }
        status_code = 200 if response["ok"] else 503
        return jsonify(response), status_code

    @app.post("/diagnostics/identify")
    @app.get("/diagnostics/identify")
    def diagnostics_identify() -> tuple[object, int]:
        sample = _capture_with_face()
        if not sample.get("ok", False):
            status_code = 422 if sample.get("error_code") == "no_face" else 503
            return jsonify(sample), status_code

        frame = sample["frame"]
        face_result = sample["face"]
        face_crop = face_result.pop("face_crop")
        embedding_result = generate_embedding(
            frame=frame,
            settings=settings,
            target_bbox=face_result.get("primary_face"),
            fallback_face_crop=face_crop,
        )
        if not embedding_result.get("ok", False):
            embedding_result.pop("embedding", None)
            response = {
                "ok": False,
                "camera": sample.get("camera"),
                "face": face_result,
                "embedding": embedding_result,
                "platform": sample.get("platform"),
                "attempts_used": sample.get("attempts_used"),
            }
            return jsonify(response), 503

        embedding_vector = embedding_result.pop("embedding")
        api_result = identify_embedding(embedding_vector, settings)
        response = {
            "ok": api_result.get("ok", False),
            "camera": sample.get("camera"),
            "face": face_result,
            "embedding": embedding_result,
            "api": api_result,
            "platform": sample.get("platform"),
            "attempts_used": sample.get("attempts_used"),
        }
        status_code = 200 if response["ok"] else 503
        return jsonify(response), status_code

    @app.get("/")
    def index() -> tuple[str, int, dict[str, str]]:
        html = _UI_HTML
        html = html.replace("__KIOSK_MODE__", "true" if settings.kiosk_mode else "false")
        html = html.replace("__CAMERA_MIRROR__", "true" if settings.camera_mirror else "false")
        return html, 200, {"Content-Type": "text/html; charset=utf-8"}

    @app.post("/diagnostics/liveness")
    @app.get("/diagnostics/liveness")
    def diagnostics_liveness() -> tuple[object, int]:
        if not settings.liveness_enabled:
            return jsonify({"ok": True, "skipped": True, "reason": "LIVENESS_ENABLED=false"}), 200

        sample = _capture_with_face()
        if not sample.get("ok", False):
            status_code = 422 if sample.get("error_code") == "no_face" else 503
            return jsonify(sample), status_code

        frame = sample["frame"]
        face_result = sample["face"]
        face_result.pop("face_crop", None)
        liveness_result = check_liveness(
            frame=frame,
            face_bbox=face_result["primary_face"],
            model_dir=settings.liveness_model_dir,
            threshold=settings.liveness_threshold,
            live_class_idx=settings.liveness_live_class_idx,
        )

        response = {
            "ok": liveness_result.get("ok", False),
            "liveness": liveness_result,
            "face": face_result,
            "attempts_used": sample.get("attempts_used"),
        }
        status_code = 200 if liveness_result.get("ok") and liveness_result.get("is_live") else 422
        return jsonify(response), status_code

    @app.get("/diagnostics/liveness/preview")
    def diagnostics_liveness_preview() -> object:
        sample = _capture_with_face()
        if not sample.get("ok", False):
            status_code = 422 if sample.get("error_code") == "no_face" else 503
            return jsonify(sample), status_code

        frame = sample["frame"].copy()
        face_result = sample["face"]
        face_bbox = face_result.get("primary_face") or {}
        if not face_bbox:
            return jsonify({"ok": False, "error": "Missing primary face bbox."}), 503

        x = int(face_bbox.get("x", 0))
        y = int(face_bbox.get("y", 0))
        w = int(face_bbox.get("w", 0))
        h = int(face_bbox.get("h", 0))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 220, 80), 2)
        cv2.putText(frame, "face", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 80), 2)

        liveness_result = check_liveness(
          frame=sample["frame"],
          face_bbox=face_bbox,
          model_dir=settings.liveness_model_dir,
          threshold=settings.liveness_threshold,
          live_class_idx=settings.liveness_live_class_idx,
        )
        regions = liveness_result.get("model_regions") or []
        colors = [(255, 170, 0), (0, 200, 255), (180, 120, 255)]
        for idx, region in enumerate(regions):
          bbox = region.get("bbox") or {}
          rx = int(bbox.get("x", 0))
          ry = int(bbox.get("y", 0))
          rw = int(bbox.get("w", 0))
          rh = int(bbox.get("h", 0))
          color = colors[idx % len(colors)]
          cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)
          label = str(region.get("model", "model"))
          cv2.putText(frame, label, (rx, max(20, ry - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        score = liveness_result.get("score")
        is_live = liveness_result.get("is_live")
        backend = liveness_result.get("backend", "unknown")
        text = f"{backend} | liveness: {score} ({'live' if is_live else 'spoof'})" if score is not None else f"{backend} | liveness: n/a"
        cv2.putText(frame, text, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 2)

        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, settings.camera_jpeg_quality])
        if not ok:
            return jsonify({"ok": False, "error": "Failed to encode preview image."}), 503
        return Response(
            jpeg.tobytes(),
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/snapshot")
    def snapshot() -> object:
        capture_result = capture_frame_fast(settings)
        if not capture_result["ok"]:
            return ("", 503)
        frame = capture_result["frame"]

        show_boxes = request.args.get("boxes", "0") == "1"
        if show_boxes:
            face_result = detect_and_crop_face(frame)
            if face_result.get("ok", False):
                face_bbox = face_result.get("primary_face") or {}
                x = int(face_bbox.get("x", 0))
                y = int(face_bbox.get("y", 0))
                w = int(face_bbox.get("w", 0))
                h = int(face_bbox.get("h", 0))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 220, 80), 2)
                cv2.putText(frame, "face", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 80), 2)

                if settings.liveness_enabled:
                    liveness_result = check_liveness(
                        frame=frame,
                        face_bbox=face_bbox,
                        model_dir=settings.liveness_model_dir,
                        threshold=settings.liveness_threshold,
                        live_class_idx=settings.liveness_live_class_idx,
                    )
                    regions = liveness_result.get("model_regions") or []
                    colors = [(255, 170, 0), (0, 200, 255), (180, 120, 255)]
                    for idx, region in enumerate(regions):
                        bbox = region.get("bbox") or {}
                        rx = int(bbox.get("x", 0))
                        ry = int(bbox.get("y", 0))
                        rw = int(bbox.get("w", 0))
                        rh = int(bbox.get("h", 0))
                        color = colors[idx % len(colors)]
                        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)
                        label = str(region.get("model", "model"))
                        cv2.putText(frame, label, (rx, max(20, ry - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

                    score = liveness_result.get("score")
                    is_live = liveness_result.get("is_live")
                    backend = liveness_result.get("backend", "unknown")
                    text = (
                        f"{backend} | liveness: {score} ({'live' if is_live else 'spoof'})"
                        if score is not None
                        else f"{backend} | liveness: n/a"
                    )
                    cv2.putText(frame, text, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 2)

        _, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, settings.camera_jpeg_quality],
        )
        return Response(
            jpeg.tobytes(),
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/pointage")
    def pointage() -> tuple[object, int]:
      sample = _capture_with_face()
      liveness_result: dict | None = None
      if not sample.get("ok", False):
        event_type, event_result = _emit_platform_event(
          "recognition_failed",
          status="error",
          message="Aucun visage détecté avant identification" if sample.get("error_code") == "no_face" else "Capture échouée avant identification",
          details={"stage": "capture_or_face", "sample": sample},
        )
        if sample.get("error_code") == "no_face":
          return jsonify({
            "ok": False,
            "matched": False,
            "error": "Aucun visage détecté",
            "error_type": "no_face_detected",
            "platform_event_type": event_type,
            "event_logged": bool(event_result.get("ok")),
            "event_result": event_result,
          }), 422
        return jsonify({
          "ok": False,
          "matched": False,
          "error": "Capture échouée",
          "error_type": "recognition_failed",
          "platform_event_type": event_type,
          "event_logged": bool(event_result.get("ok")),
          "event_result": event_result,
        }), 503

      frame = sample["frame"]
      face_result = sample["face"]
      face_crop = face_result.pop("face_crop")

      if settings.liveness_enabled:
        liveness_result = check_liveness(
          frame=frame,
          face_bbox=face_result["primary_face"],
          model_dir=settings.liveness_model_dir,
          threshold=settings.liveness_threshold,
          live_class_idx=settings.liveness_live_class_idx,
        )
        if not liveness_result.get("ok", False):
          event_type, event_result = _emit_platform_event(
            "recognition_failed",
            status="error",
            message="Liveness indisponible, pointage bloqué",
            details={"stage": "liveness", "liveness": liveness_result},
          )
          return jsonify({
            "ok": False,
            "matched": False,
            "error": "Liveness indisponible, pointage bloqué",
            "error_type": "recognition_failed",
            "platform_event_type": event_type,
            "liveness_details": liveness_result,
            "event_logged": bool(event_result.get("ok")),
            "event_result": event_result,
          }), 503
        # Blocage strict: on accepte uniquement le booléen True explicite.
        if liveness_result.get("is_live") is not True:
          event_type, event_result = _emit_platform_event(
            "spoof_attempt",
            status="blocked",
            message="Tentative d'usurpation détectée par la liveness",
            details={
              "stage": "liveness",
              "liveness_score": liveness_result.get("score"),
              "liveness": liveness_result,
            },
          )
          return jsonify({
            "ok": False,
            "matched": False,
            "error": "Tentative d'usurpation détectée",
            "error_type": "spoof_attempt",
            "platform_event_type": event_type,
            "liveness_score": liveness_result.get("score"),
            "event_logged": bool(event_result.get("ok")),
            "event_result": event_result,
          }), 401

      embedding_result = generate_embedding(
        frame=frame,
        settings=settings,
        target_bbox=face_result.get("primary_face"),
        fallback_face_crop=face_crop,
      )
      if not embedding_result.get("ok", False):
        event_type, event_result = _emit_platform_event(
          "recognition_failed",
          status="error",
          message="Échec de génération d'embedding",
          details={"stage": "embedding", "embedding": embedding_result},
        )
        return jsonify({
          "ok": False,
          "matched": False,
          "error": "Échec d'embedding",
          "error_type": "recognition_failed",
          "platform_event_type": event_type,
          "event_logged": bool(event_result.get("ok")),
          "event_result": event_result,
        }), 503

      embedding_vector = embedding_result.pop("embedding")
      api_result = identify_embedding(embedding_vector, settings)
      api_response = api_result.get("response", {})
      if api_result.get("ok") and api_response.get("matched"):
        return jsonify({
          "ok": True,
          "matched": True,
          "full_name": api_response.get("full_name"),
          "pointage_type": api_response.get("pointage_type"),
          "pointage_id": api_response.get("pointage_id"),
          "liveness": {
            "enabled": bool(settings.liveness_enabled),
            "is_live": None if liveness_result is None else liveness_result.get("is_live"),
            "score": None if liveness_result is None else liveness_result.get("score"),
            "backend": None if liveness_result is None else liveness_result.get("backend"),
          },
        }), 200

      event_type = "unknown_user" if api_result.get("status_code") in {401, 404} else "recognition_failed"
      event_type, event_result = _emit_platform_event(
        event_type,
        status="rejected",
        message=api_response.get("error", "Identité non reconnue"),
        details={"stage": "identify", "identify": api_result},
      )
      return jsonify({
        "ok": False,
        "matched": False,
        "error": api_response.get("error", "Identité non reconnue"),
        "error_type": event_type,
        "platform_event_type": event_type,
        "event_logged": bool(event_result.get("ok")),
        "event_result": event_result,
      }), 401

    return app


app = create_app()