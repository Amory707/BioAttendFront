from __future__ import annotations

import cv2
from flask import Flask, Response, jsonify

from .api_client import identify_embedding
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
        feed.src = '/snapshot?' + Date.now();
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
          setStatus('error', 'Non reconnu', data.error || 'Veuillez reessayer');
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
        capture_result = capture_frame(settings)
        if not capture_result["ok"]:
            capture_result.pop("frame", None)
            return jsonify(capture_result), 503

        frame = capture_result.pop("frame")
        face_result = detect_and_crop_face(frame)
        face_result.pop("face_crop", None)

        response = {
            "ok": face_result.get("ok", False),
            "camera": capture_result.get("camera"),
            "face": face_result,
            "platform": capture_result.get("platform"),
        }

        if response["ok"]:
            return jsonify(response), 200
        return jsonify(response), 422

    @app.post("/diagnostics/embedding")
    @app.get("/diagnostics/embedding")
    def diagnostics_embedding() -> tuple[object, int]:
        capture_result = capture_frame(settings)
        if not capture_result["ok"]:
            capture_result.pop("frame", None)
            return jsonify(capture_result), 503

        frame = capture_result.pop("frame")
        face_result = detect_and_crop_face(frame)
        if not face_result.get("ok", False):
            face_result.pop("face_crop", None)
            response = {
                "ok": False,
                "camera": capture_result.get("camera"),
                "face": face_result,
                "platform": capture_result.get("platform"),
            }
            return jsonify(response), 422

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
            "camera": capture_result.get("camera"),
            "face": face_result,
            "embedding": embedding_result,
            "platform": capture_result.get("platform"),
        }

        if response["ok"]:
            return jsonify(response), 200
        return jsonify(response), 503

    @app.post("/diagnostics/identify")
    @app.get("/diagnostics/identify")
    def diagnostics_identify() -> tuple[object, int]:
        capture_result = capture_frame(settings)
        if not capture_result["ok"]:
            capture_result.pop("frame", None)
            return jsonify(capture_result), 503

        frame = capture_result.pop("frame")
        face_result = detect_and_crop_face(frame)
        if not face_result.get("ok", False):
            face_result.pop("face_crop", None)
            response = {
                "ok": False,
                "camera": capture_result.get("camera"),
                "face": face_result,
                "platform": capture_result.get("platform"),
            }
            return jsonify(response), 422

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
                "camera": capture_result.get("camera"),
                "face": face_result,
                "embedding": embedding_result,
                "platform": capture_result.get("platform"),
            }
            return jsonify(response), 503

        embedding_vector = embedding_result.pop("embedding")
        api_result = identify_embedding(embedding_vector, settings)

        response = {
            "ok": api_result.get("ok", False),
            "camera": capture_result.get("camera"),
            "face": face_result,
            "embedding": embedding_result,
            "api": api_result,
            "platform": capture_result.get("platform"),
        }

        if response["ok"]:
            return jsonify(response), 200
        return jsonify(response), 503

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

        capture_result = capture_frame(settings)
        if not capture_result["ok"]:
            capture_result.pop("frame", None)
            return jsonify(capture_result), 503

        frame = capture_result.pop("frame")
        face_result = detect_and_crop_face(frame)
        if not face_result.get("ok", False):
            face_result.pop("face_crop", None)
            return jsonify({"ok": False, "face": face_result}), 422

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
        }
        if liveness_result.get("ok") and liveness_result.get("is_live"):
            return jsonify(response), 200
        return jsonify(response), 422

    @app.get("/snapshot")
    def snapshot() -> object:
        capture_result = capture_frame_fast(settings)
        if not capture_result["ok"]:
            return ("", 503)
        frame = capture_result["frame"]
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
        capture_result = capture_frame_fast(settings)
        if not capture_result["ok"]:
            return jsonify({"ok": False, "matched": False, "error": "Capture échouée"}), 503
        frame = capture_result.pop("frame")
        face_result = detect_and_crop_face(frame)
        if not face_result.get("ok", False):
            face_result.pop("face_crop", None)
            return jsonify({"ok": False, "matched": False, "error": "Aucun visage détecté"}), 422
        face_crop = face_result.pop("face_crop")

        # ── Liveness (anti-spoofing) ──────────────────────────────────────────
        if settings.liveness_enabled:
            liveness_result = check_liveness(
                frame=frame,
                face_bbox=face_result["primary_face"],
                model_dir=settings.liveness_model_dir,
                threshold=settings.liveness_threshold,
                live_class_idx=settings.liveness_live_class_idx,
            )
            if liveness_result.get("ok") and not liveness_result.get("is_live", True):
                return jsonify({
                    "ok": False,
                    "matched": False,
                    "error": "Tentative d'usurpation détectée",
                    "liveness_score": liveness_result.get("score"),
                }), 401

        embedding_result = generate_embedding(
            frame=frame,
            settings=settings,
            target_bbox=face_result.get("primary_face"),
            fallback_face_crop=face_crop,
        )
        if not embedding_result.get("ok", False):
            return jsonify({"ok": False, "matched": False, "error": "Échec d'embedding"}), 503
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
            }), 200
        return jsonify({
            "ok": False,
            "matched": False,
            "error": api_response.get("error", "Identité non reconnue"),
        }), 401

    return app


app = create_app()