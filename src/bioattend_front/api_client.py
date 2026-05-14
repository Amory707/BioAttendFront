from __future__ import annotations

import time
from typing import Any

import requests

from .config import Settings


def _build_headers(settings: Settings) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.api_token:
        headers["Authorization"] = f"Bearer {settings.api_token}"
        headers["X-API-Key"] = settings.api_token
    return headers



def identify_embedding(embedding: list[float], settings: Settings) -> dict[str, Any]:
    if not settings.server_url:
        return {"ok": False, "error": "SERVER_URL is empty."}

    headers = _build_headers(settings)

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


def report_event(
    event_type: str,
    settings: Settings,
    *,
    status: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not settings.events_url:
        return {"ok": False, "skipped": True, "reason": "EVENTS_URL is empty."}

    payload: dict[str, Any] = {
        "event_type": event_type,
        "status": status,
        "message": message,
        "device_name": settings.device_name,
    }
    if details:
        payload["details"] = details

    def _post_event(url: str) -> dict[str, Any]:
        started_at = time.monotonic()
        try:
            response = requests.post(
                url,
                json=payload,
                headers=_build_headers(settings),
                timeout=settings.api_timeout_seconds,
            )
        except requests.RequestException as exc:
            duration_ms = round((time.monotonic() - started_at) * 1000, 2)
            return {
                "ok": False,
                "duration_ms": duration_ms,
                "error": f"Event API request failed: {exc}",
                "target": url,
                "payload": payload,
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
            "target": url,
            "response": body,
            "payload": payload,
        }

    first = _post_event(settings.events_url)
    if first.get("ok"):
        return first

    if int(first.get("status_code") or 0) == 404:
        base_url = settings.events_url.strip()
        alternate_url = base_url[:-1] if base_url.endswith("/") else f"{base_url}/"
        if alternate_url and alternate_url != base_url:
            second = _post_event(alternate_url)
            second["retry_from"] = base_url
            if second.get("ok"):
                second["url_autofixed"] = True
            return second

    return first

