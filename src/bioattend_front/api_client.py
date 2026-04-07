from __future__ import annotations

import time
from typing import Any

import requests

from .config import Settings


def _mask_token(token: str) -> str:
    if not token:
        return ""
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}...{token[-4:]}"


def identify_embedding(embedding: list[float], settings: Settings) -> dict[str, Any]:
    if not settings.server_url:
        return {"ok": False, "error": "SERVER_URL is empty."}

    headers: dict[str, str] = {"Content-Type": "application/json"}
    auth_mode = "none"
    if settings.api_token:
        headers["Authorization"] = f"Bearer {settings.api_token}"
        headers["X-API-Key"] = settings.api_token
        auth_mode = "bearer+x-api-key"

    auth_debug = {
        "auth_mode": auth_mode,
        "token_masked": _mask_token(settings.api_token),
        "token_len": len(settings.api_token),
        "headers_sent": sorted(list(headers.keys())),
    }

    if settings.debug:
        print(
            "[identify_embedding] auth_mode=",
            auth_mode,
            "token_masked=",
            _mask_token(settings.api_token),
            "token_len=",
            len(settings.api_token),
            "target=",
            settings.server_url,
        )

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
        result = {
            "ok": False,
            "duration_ms": duration_ms,
            "error": f"API request failed: {exc}",
            "target": settings.server_url,
        }
        result["auth_debug"] = auth_debug
        return result

    duration_ms = round((time.monotonic() - started_at) * 1000, 2)
    body: Any
    try:
        body = response.json()
    except ValueError:
        body = {"raw": response.text}

    result = {
        "ok": 200 <= response.status_code < 300,
        "duration_ms": duration_ms,
        "status_code": response.status_code,
        "target": settings.server_url,
        "response": body,
    }
    result["auth_debug"] = auth_debug
    return result
