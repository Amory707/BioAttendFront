from __future__ import annotations

import time
from typing import Any

import requests

from .config import Settings


def _try_identify(embedding: list[float], settings: Settings) -> dict[str, Any]:
    """Single attempt — makes one HTTP request and returns the result dict."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.api_token:
        headers["Authorization"] = f"Bearer {settings.api_token}"
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
    try:
        body: Any = response.json()
    except ValueError:
        body = {"raw": response.text}

    return {
        "ok": 200 <= response.status_code < 300,
        "duration_ms": duration_ms,
        "status_code": response.status_code,
        "target": settings.server_url,
        "response": body,
    }


def identify_embedding(
    embedding: list[float],
    settings: Settings,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Public entry point — retries on transient network failures with backoff."""
    if not settings.server_url:
        return {"ok": False, "error": "SERVER_URL is empty."}

    result: dict[str, Any] = {}
    for attempt in range(max_retries + 1):
        result = _try_identify(embedding, settings)
        if result["ok"] or attempt == max_retries:
            return result
        time.sleep(0.3 * (attempt + 1))  # 0.3 s, then 0.6 s

    return result