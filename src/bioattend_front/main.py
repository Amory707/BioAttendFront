from __future__ import annotations

from flask import Flask, jsonify

from .camera import capture_frame, probe_camera
from .config import Settings
from .face import detect_and_crop_face


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

    return app


app = create_app()