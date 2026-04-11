from __future__ import annotations

import cv2
from flask import Flask, Response, jsonify

from .api_client import identify_embedding
from .camera import capture_frame, capture_frame_fast, probe_camera
from .config import Settings
from .embedding import generate_embedding
from .face import detect_and_crop_face
from .anti_spoofing import AntiSpoofPredict

_UI_HTML = """\
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BioAttend</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0d0d0d; color: #fff;
      font-family: 'Segoe UI', Arial, sans-serif;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      min-height: 100vh; gap: 20px;
    }
    h1 { font-size: 1.3em; letter-spacing: 6px; color: #888; text-transform: uppercase; }
    .camera-wrap { position: relative; width: 640px; height: 360px; }
    #feed {
      width: 100%; height: 100%; object-fit: cover;
      border-radius: 12px; display: block; background: #111;
    }
    .oval {
      position: absolute; top: 50%; left: 50%;
      width: 220px; height: 290px;
      transform: translate(-50%, -58%);
      border: 3px solid rgba(255,255,255,0.55);
      border-radius: 50%; pointer-events: none;
      transition: border-color 0.4s, box-shadow 0.4s;
    }
    .oval.success { border-color: #4CAF50; box-shadow: 0 0 24px rgba(76,175,80,0.5); }
    .oval.error   { border-color: #f44336; box-shadow: 0 0 24px rgba(244,67,54,0.4); }
    .status {
      width: 640px; min-height: 70px; padding: 14px 24px;
      border-radius: 10px; background: #181818; border: 1px solid #2a2a2a;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      text-align: center; transition: background 0.4s, border-color 0.4s;
    }
    .status.success { background: #193319; border-color: #4CAF50; }
    .status.error   { background: #331919; border-color: #f44336; }
    .status .name { font-size: 1.4em; font-weight: 600; }
    .status .info { font-size: 0.9em; color: #aaa; margin-top: 5px; }
    button {
      padding: 13px 52px; font-size: 1.05em; letter-spacing: 2px;
      background: #1565C0; color: #fff; border: none; border-radius: 8px;
      cursor: pointer; text-transform: uppercase; transition: background 0.2s;
    }
    button:hover:not(:disabled) { background: #1976D2; }
    button:disabled { background: #2a2a2a; color: #555; cursor: not-allowed; }
    .hint { font-size: 0.72em; color: #444; }
  </style>
</head>
<body>
  <h1>BioAttend</h1>
  <div class="camera-wrap">
    <img id="feed" src="/snapshot" alt="Camera">
    <div class="oval" id="oval"></div>
  </div>
  <div class="status" id="status">
    <span class="info">Positionnez votre visage dans le cadre</span>
  </div>
  <button id="btn" onclick="startPointage()">Pointer</button>
  <p class="hint">ou appuyez sur Espace</p>
  <script>
    var timer = null;
    function startRefresh() {
      if (timer) return;
      timer = setInterval(function() {
        document.getElementById('feed').src = '/snapshot?' + Date.now();
      }, 200);
    }
    function stopRefresh() { clearInterval(timer); timer = null; }
    startRefresh();
    document.addEventListener('keydown', function(e) {
      if (e.code === 'Space' && !document.getElementById('btn').disabled) {
        e.preventDefault();
        startPointage();
      }
    });
    async function startPointage() {
      var btn = document.getElementById('btn');
      var status = document.getElementById('status');
      var oval = document.getElementById('oval');
      btn.disabled = true;
      stopRefresh();
      oval.className = 'oval';
      status.className = 'status';
      status.innerHTML = '<span class="info">Identification en cours\u2026</span>';
      try {
        var resp = await fetch('/pointage', { method: 'POST' });
        var data = await resp.json();
        if (data.ok && data.matched) {
          var type = data.pointage_type === 'ENTREE' ? 'Entr\u00e9e' : 'Sortie';
          var heure = new Date().toLocaleTimeString('fr-FR');
          oval.className = 'oval success';
          status.className = 'status success';
          status.innerHTML = '<span class="name">\u2713 ' + data.full_name + '</span>' +
            '<span class="info">' + type + ' \u2014 ' + heure + '</span>';
        } else {
          oval.className = 'oval error';
          status.className = 'status error';
          status.innerHTML = '<span class="name">\u2717 Non reconnu</span>' +
            '<span class="info">' + (data.error || 'Veuillez r\u00e9essayer') + '</span>';
        }
      } catch(e) {
        oval.className = 'oval error';
        status.className = 'status error';
        status.innerHTML = '<span class="name">\u2717 Erreur r\u00e9seau</span>';
      }
      setTimeout(function() {
        oval.className = 'oval';
        status.className = 'status';
        status.innerHTML = '<span class="info">Positionnez votre visage dans le cadre</span>';
        btn.disabled = false;
        startRefresh();
      }, 4000);
    }
  </script>
</body>
</html>
"""

# Initialized once at module level — safe because AntiSpoofPredict
# only loads the model on first predict() call, not on __init__.
_anti_spoof = AntiSpoofPredict()
_ANTI_SPOOF_MODEL_PATH = (
    "./resources/anti_spoof_models/"
    "2.7182818284590452353602874713527_MiniFASNetV2.pth"
)
_ANTI_SPOOF_THRESHOLD = 0.7  # real_score must exceed this to pass


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
        return _UI_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

    @app.get("/snapshot")
    def snapshot() -> object:
        capture_result = capture_frame_fast(settings)
        if not capture_result["ok"]:
            return ("", 503)
        frame = capture_result["frame"]
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return Response(
            jpeg.tobytes(),
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/pointage")
    def pointage() -> tuple[object, int]:
        # ── 1. Capture ────────────────────────────────────────────────────
        capture_result = capture_frame_fast(settings)
        if not capture_result["ok"]:
            return jsonify({"ok": False, "matched": False, "error": "Capture échouée"}), 503

        frame = capture_result.pop("frame")

        # ── 2. Face detection ─────────────────────────────────────────────
        face_result = detect_and_crop_face(frame)
        if not face_result.get("ok", False):
            face_result.pop("face_crop", None)
            return jsonify({"ok": False, "matched": False, "error": "Aucun visage détecté"}), 422

        face_crop = face_result.pop("face_crop")

        # ── 3. Anti-spoofing (liveness check) ────────────────────────────
        spoof_result = _anti_spoof.predict(frame, _ANTI_SPOOF_MODEL_PATH)
        real_score = float(spoof_result[0][1])
        if real_score < _ANTI_SPOOF_THRESHOLD:
            return jsonify({
                "ok": False,
                "matched": False,
                "error": "Liveness check failed",
            }), 403

        # ── 4. Embedding ──────────────────────────────────────────────────
        embedding_result = generate_embedding(
            frame=frame,
            settings=settings,
            target_bbox=face_result.get("primary_face"),
            fallback_face_crop=face_crop,
        )
        if not embedding_result.get("ok", False):
            return jsonify({"ok": False, "matched": False, "error": "Échec d'embedding"}), 503

        embedding_vector = embedding_result.pop("embedding")

        # ── 5. API identification ─────────────────────────────────────────
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