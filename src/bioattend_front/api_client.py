from __future__ import annotations

import time
from typing import Any

import requests

from .config import Settings


def identify_embedding(embedding: list[float], settings: Settings) -> dict[str, Any]:
    if not settings.server_url:
        return {"ok": False, "error": "SERVER_URL is empty."}

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.api_token:
        headers["X-API-Key"] = settings.api_token

    payload = {"embedding": embedding}
    started_at = time.monotonic()

    try:
        response = requests.post(
            settings.server_url,
            json=payload,
            headers=headers,
            timeout=settings.api_timeout_seconds,
        )
    except requests.RequestException as exc:
        duration_ms = round((time.monotonic() - started_at) * 1000, 2)
        return {
            "ok": False,
            "duration_ms": duration_ms,
            "error": f"API request failed: {exc}",
            "target": settings.server_url,
        }

    duration_ms = round((time.monotonic() - started_at) * 1000, 2)
    body: Any
    try:
        body = response.json()
    except ValueError:
        body = {"raw": response.text}

    return {
        "ok": 200 <= response.status_code < 300,
        "duration_ms": duration_ms,
        "status_code": response.status_code,
        "target": settings.server_url,
        "response": body,
    }
