from __future__ import annotations

from flask import Flask, jsonify

from .camera import probe_camera
from .config import Settings


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

    return app


app = create_app()