from __future__ import annotations

import time
import cv2
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file

from .api_client import identify_embedding, report_event
from .camera import capture_frame, capture_frame_fast, probe_camera
from .config import Settings
from .embedding import generate_embedding
from .face import detect_and_crop_face
from .liveness import check_liveness

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_UI_HTML = """\
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BioAttend | Pointeuse intelligente</title>
  <style>
    :root {
      --ink: #f4f8fa;
      --muted: #a9bfcd;
      --panel: rgba(8, 16, 23, 0.74);
      --line: rgba(255, 255, 255, 0.2);
      --ok: #43d68a;
      --bad: #ff7a7a;
      --bg-a: #09111a;
      --bg-b: #02070b;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    html, body {
      width: 100%;
      height: 100%;
      overflow: hidden;
    }

    body {
      color: var(--ink);
      background: radial-gradient(1200px 600px at 10% -20%, rgba(67, 214, 138, 0.14), transparent 65%),
        radial-gradient(900px 540px at 95% 108%, rgba(255, 184, 90, 0.16), transparent 62%),
        linear-gradient(150deg, var(--bg-a), var(--bg-b));
      font-family: "Bahnschrift", "Trebuchet MS", sans-serif;
      user-select: none;
    }

    .app {
      position: relative;
      width: 100vw;
      height: 100dvh;
      overflow: hidden;
    }

    .app::before {
      content: "";
      position: absolute;
      inset: 0;
      z-index: 0;
      pointer-events: none;
      background: radial-gradient(120% 90% at 50% 30%, rgba(6, 14, 24, 0.18), rgba(1, 6, 10, 0.78));
    }

    .logo-bg {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      pointer-events: none;
      opacity: 0.28;
      overflow: hidden;
      z-index: 0;
    }

    .logo-bg img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      object-position: center;
      transform: scale(1.04);
      filter: saturate(0.92) brightness(0.72) contrast(1.03);
    }

    .view {
      position: absolute;
      inset: 0;
      padding: clamp(12px, 2.4vw, 28px);
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.25s ease;
      z-index: 1;
    }

    .view.active {
      opacity: 1;
      pointer-events: auto;
    }

    .view-standard {
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: clamp(10px, 2vw, 22px);
    }

    .std-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }

    .std-brand {
      letter-spacing: 0.22em;
      text-transform: uppercase;
      font-weight: 700;
      font-size: clamp(1.06rem, 2.1vw, 1.56rem);
    }

    .badge {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 14px;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      background: rgba(255, 255, 255, 0.06);
      backdrop-filter: blur(7px);
    }

    .std-main {
      display: grid;
      justify-items: center;
      align-content: center;
      text-align: center;
      gap: clamp(10px, 2.2vw, 22px);
      width: min(1180px, 100%);
      margin: 0 auto;
      border-radius: 26px;
      border: 1px solid rgba(255, 255, 255, 0.15);
      background: linear-gradient(150deg, rgba(6, 14, 24, 0.52), rgba(4, 10, 17, 0.38));
      backdrop-filter: blur(6px);
      box-shadow: 0 20px 80px rgba(0, 0, 0, 0.35);
      padding: clamp(16px, 2.5vw, 28px);
    }

    .clock-time {
      font-size: clamp(2.4rem, 11vw, 8rem);
      font-variant-numeric: tabular-nums;
      font-weight: 800;
      line-height: 0.94;
      letter-spacing: 0.02em;
      text-shadow: 0 10px 34px rgba(0, 0, 0, 0.45);
    }

    .clock-date {
      color: var(--muted);
      font-size: clamp(1rem, 2vw, 1.44rem);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .std-cards {
      width: min(860px, 100%);
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: clamp(10px, 1.8vw, 18px);
    }

    .card {
      border-radius: 20px;
      border: 1px solid rgba(255, 255, 255, 0.22);
      background: linear-gradient(145deg, rgba(8, 16, 25, 0.76), rgba(6, 13, 21, 0.64));
      backdrop-filter: blur(10px);
      padding: 16px;
      min-height: 126px;
      display: grid;
      align-content: center;
      justify-items: center;
      gap: 8px;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04);
    }

    .card-title {
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.76rem;
      color: #d4e4ee;
    }

    .card-value {
      font-size: clamp(1.02rem, 1.7vw, 1.24rem);
      text-align: center;
      color: #edf6fa;
    }

    .std-footer {
      text-align: center;
      color: #8fa8b7;
      font-size: clamp(0.92rem, 1.5vw, 1.04rem);
    }

    .key {
      display: inline-block;
      border: 1px solid rgba(255, 255, 255, 0.38);
      border-bottom-width: 3px;
      border-radius: 10px;
      padding: 4px 10px;
      margin: 0 2px;
      color: #f7fbff;
      font-size: 0.92em;
    }

    .view-capture {
      background: linear-gradient(180deg, rgba(2, 7, 11, 0.34), rgba(2, 7, 11, 0.76));
    }

    .capture-wrap {
      position: relative;
      width: 100%;
      height: 100%;
      border-radius: 26px;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.2);
      background: #010203;
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
      width: clamp(250px, 34vw, 430px);
      aspect-ratio: 0.76;
      transform: translate(-50%, -54%);
      border-radius: 50%;
      border: 4px solid rgba(255, 255, 255, 0.72);
      box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.28) inset;
      pointer-events: none;
      transition: border-color 0.2s ease;
    }

    .scan-oval.success { border-color: var(--ok); }
    .scan-oval.error { border-color: var(--bad); }

    .scan-line {
      position: absolute;
      left: 50%;
      top: 50%;
      width: clamp(210px, 28vw, 352px);
      height: 2px;
      transform: translate(-50%, -70px);
      background: linear-gradient(90deg, transparent, rgba(67, 214, 138, 0.9), transparent);
      filter: drop-shadow(0 0 8px rgba(67, 214, 138, 0.7));
      animation: scan 2.1s ease-in-out infinite;
      pointer-events: none;
    }

    @keyframes scan {
      0% { transform: translate(-50%, -96px); opacity: 0.2; }
      50% { transform: translate(-50%, 96px); opacity: 0.95; }
      100% { transform: translate(-50%, -96px); opacity: 0.2; }
    }

    .capture-overlay {
      position: absolute;
      top: 14px;
      left: 14px;
      right: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }

    .capture-title {
      font-size: clamp(1.08rem, 2vw, 1.58rem);
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: #ecf8ff;
      text-shadow: 0 6px 20px rgba(0, 0, 0, 0.55);
    }

    .capture-help {
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.2);
      background: rgba(4, 12, 19, 0.6);
      padding: 8px 14px;
      color: #d8e8f1;
      font-size: 0.92rem;
      backdrop-filter: blur(6px);
    }

    .capture-status {
      position: absolute;
      bottom: 16px;
      left: 50%;
      transform: translateX(-50%);
      min-width: min(90vw, 560px);
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.2);
      background: rgba(3, 9, 14, 0.7);
      backdrop-filter: blur(6px);
      padding: 10px 16px;
      text-align: center;
      color: #e2eef4;
    }

    .capture-status-main {
      font-size: clamp(1rem, 1.8vw, 1.3rem);
      font-weight: 700;
    }

    .capture-status-sub {
      margin-top: 4px;
      font-size: clamp(0.88rem, 1.2vw, 1rem);
      color: var(--muted);
    }

    .view-result {
      display: grid;
      align-items: center;
      justify-items: center;
    }

    .result-bg-word {
      position: absolute;
      inset: auto 0 11%;
      text-align: center;
      font-size: clamp(3rem, 17vw, 16rem);
      font-weight: 800;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: rgba(255, 255, 255, 0.08);
      pointer-events: none;
    }

    .result-card {
      width: min(96vw, 960px);
      border-radius: 22px;
      border: 1px solid rgba(255, 255, 255, 0.26);
      background: rgba(6, 14, 20, 0.78);
      backdrop-filter: blur(7px);
      box-shadow: 0 16px 60px rgba(0, 0, 0, 0.45);
      padding: clamp(20px, 3.2vw, 38px);
      text-align: center;
      display: grid;
      gap: 10px;
    }

    .result-card.success {
      border-color: rgba(67, 214, 138, 0.55);
      background: linear-gradient(180deg, rgba(18, 48, 36, 0.82), rgba(8, 25, 18, 0.8));
    }

    .result-card.error {
      border-color: rgba(255, 122, 122, 0.55);
      background: linear-gradient(180deg, rgba(64, 22, 22, 0.8), rgba(30, 11, 11, 0.78));
    }

    .result-tag {
      justify-self: center;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.32);
      background: rgba(255, 255, 255, 0.08);
      color: #f5fbff;
      font-size: 0.78rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      padding: 8px 14px;
    }

    .result-greeting { font-size: clamp(1.3rem, 2.5vw, 2rem); color: #e8f5fc; }

    .result-kind {
      margin-top: 4px;
      font-size: clamp(2.3rem, 8.5vw, 6rem);
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #f9fcff;
      text-shadow: 0 6px 24px rgba(0, 0, 0, 0.45);
    }

    .result-meta {
      color: #d5e5ef;
      font-size: clamp(0.95rem, 1.55vw, 1.1rem);
    }

    .result-next { color: #93a8b8; font-size: 0.92rem; }

    @media (max-width: 980px) {
      .std-cards { grid-template-columns: 1fr; }
      .capture-overlay { flex-direction: column; align-items: flex-start; }
    }

    @media (max-width: 560px) {
      .view { padding: 10px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="logo-bg" aria-hidden="true"><img src="/assets/logo-projet" alt=""></div>

    <section class="view view-standard active" id="viewStandard">
      <header class="std-header">
        <h1 class="std-brand">BioAttend</h1>
        <div class="badge">Mode standard</div>
      </header>

      <main class="std-main">
        <div class="clock-time" id="clockTime">--:--:--</div>
        <div class="clock-date" id="clockDate">--</div>
        <div class="std-cards">
          <article class="card"><div class="card-title">Jour</div><div class="card-value" id="dayLabel">--</div></article>
          <article class="card"><div class="card-title">Meteo</div><div class="card-value" id="weatherLabel">Mise a jour...</div></article>
        </div>
      </main>

      <footer class="std-footer" id="stdFooter">Appuyez sur <span class="key">Espace</span> pour lancer la capture</footer>
    </section>

    <section class="view view-capture" id="viewCapture">
      <div class="capture-wrap">
        <img id="feed" src="/snapshot" alt="Flux camera">
        <div class="scan-oval" id="scanOval"></div>
        <div class="scan-line" aria-hidden="true"></div>
        <div class="capture-overlay">
          <div class="capture-title">Mode capture</div>
          <div class="capture-help">Centrez votre visage dans l'ovale</div>
        </div>
        <div class="capture-status">
          <div class="capture-status-main" id="captureStatusMain">Preparation de la reconnaissance...</div>
          <div class="capture-status-sub" id="captureStatusSub">Ne bougez pas pendant la lecture</div>
        </div>
      </div>
    </section>

    <section class="view view-result" id="viewResult">
      <div class="result-bg-word">BioAttend</div>
      <div class="result-card" id="resultCard">
        <div class="result-tag" id="resultTag">Resultat</div>
        <div class="result-greeting" id="resultGreeting">Traitement en cours...</div>
        <div class="result-kind" id="resultKind">--</div>
        <div class="result-meta" id="resultMeta">--</div>
        <div class="result-next">Retour automatique au mode standard</div>
      </div>
    </section>
  </div>

  <script>
    var feed = document.getElementById('feed');
    var scanOval = document.getElementById('scanOval');
    var viewStandard = document.getElementById('viewStandard');
    var viewCapture = document.getElementById('viewCapture');
    var viewResult = document.getElementById('viewResult');
    var clockTime = document.getElementById('clockTime');
    var clockDate = document.getElementById('clockDate');
    var dayLabel = document.getElementById('dayLabel');
    var weatherLabel = document.getElementById('weatherLabel');
    var stdFooter = document.getElementById('stdFooter');
    var captureStatusMain = document.getElementById('captureStatusMain');
    var captureStatusSub = document.getElementById('captureStatusSub');
    var resultCard = document.getElementById('resultCard');
    var resultTag = document.getElementById('resultTag');
    var resultGreeting = document.getElementById('resultGreeting');
    var resultKind = document.getElementById('resultKind');
    var resultMeta = document.getElementById('resultMeta');

    var KIOSK_MODE = __KIOSK_MODE__;
    var CAMERA_MIRROR = __CAMERA_MIRROR__;
    var SHOW_BOXES = new URLSearchParams(window.location.search).get('boxes') === '1';

    var streamRunning = false;
    var streamTimer = null;
    var recognitionInProgress = false;
    var resultTimer = null;

    function setMode(mode) {
      viewStandard.classList.toggle('active', mode === 'standard');
      viewCapture.classList.toggle('active', mode === 'capture');
      viewResult.classList.toggle('active', mode === 'result');
    }

    function setCaptureStatus(mainText, subText, mood) {
      captureStatusMain.textContent = mainText;
      captureStatusSub.textContent = subText;
      scanOval.className = 'scan-oval' + (mood ? ' ' + mood : '');
    }

    function scheduleNextFrame(delay) {
      if (!streamRunning) return;
      clearTimeout(streamTimer);
      streamTimer = setTimeout(function() {
        feed.src = '/snapshot?boxes=' + (SHOW_BOXES ? '1' : '0') + '&t=' + Date.now();
      }, delay);
    }

    function startStream() {
      if (streamRunning) return;
      streamRunning = true;
      scheduleNextFrame(0);
    }

    function stopStream() {
      streamRunning = false;
      clearTimeout(streamTimer);
      streamTimer = null;
    }

    feed.addEventListener('load', function() { scheduleNextFrame(90); });
    feed.addEventListener('error', function() { scheduleNextFrame(180); });

    function updateClock() {
      var now = new Date();
      clockTime.textContent = now.toLocaleTimeString('fr-FR');
      clockDate.textContent = now.toLocaleDateString('fr-FR', {
        weekday: 'short', day: '2-digit', month: 'long', year: 'numeric'
      });
      dayLabel.textContent = now.toLocaleDateString('fr-FR', { weekday: 'long' });
    }

    async function enterFullscreen() {
      var el = document.documentElement;
      if (!document.fullscreenElement && el.requestFullscreen) {
        try { await el.requestFullscreen(); } catch (e) { return; }
      }
    }

    function updateWeather() {
      if (!('geolocation' in navigator)) {
        weatherLabel.textContent = 'GPS indisponible';
        return;
      }
      navigator.geolocation.getCurrentPosition(async function(pos) {
        var lat = pos.coords.latitude;
        var lon = pos.coords.longitude;
        try {
          var url = 'https://api.open-meteo.com/v1/forecast?latitude=' + lat + '&longitude=' + lon + '&current=temperature_2m&timezone=auto';
          var resp = await fetch(url);
          if (!resp.ok) throw new Error('meteo');
          var data = await resp.json();
          if (!data.current || typeof data.current.temperature_2m !== 'number') throw new Error('meteo');
          weatherLabel.textContent = Math.round(data.current.temperature_2m) + '°C';
        } catch (e) {
          weatherLabel.textContent = 'Meteo indisponible';
        }
      }, function() {
        weatherLabel.textContent = 'Localisation refusee';
      }, { timeout: 7000, maximumAge: 600000 });
    }

    function showResultSuccess(data) {
      var type = data.pointage_type === 'ENTREE' ? 'ENTREE' : 'SORTIE';
      var now = new Date();
      resultCard.className = 'result-card success';
      resultTag.textContent = 'Pointage valide';
      resultGreeting.textContent = 'Bonjour ' + data.full_name;
      resultKind.textContent = type;
      resultMeta.textContent = 'Heure: ' + now.toLocaleTimeString('fr-FR') + ' | Date: ' + now.toLocaleDateString('fr-FR');
    }

    function showResultError(data) {
      var errorType = data.error_type || 'recognition_failed';
      var titleByType = {
        no_face_detected: 'Visage non detecte',
        recognition_failed: 'Echec de reconnaissance',
        unknown_user: 'Utilisateur inconnu',
        spoof_attempt: 'Anti-spoof'
      };
      resultCard.className = 'result-card error';
      resultTag.textContent = titleByType[errorType] || 'Pointage refuse';
      resultGreeting.textContent = data.error || 'Veuillez recommencer';
      resultKind.textContent = 'ECHEC';
      resultMeta.textContent = 'Heure: ' + new Date().toLocaleTimeString('fr-FR');
    }

    function backToStandardSoon() {
      clearTimeout(resultTimer);
      resultTimer = setTimeout(function() {
        recognitionInProgress = false;
        setMode('standard');
        setCaptureStatus('Preparation de la reconnaissance...', 'Ne bougez pas pendant la lecture', '');
        stdFooter.innerHTML = 'Appuyez sur <span class="key">Espace</span> pour lancer la capture';
      }, 4200);
    }

    async function startCaptureFlow() {
      if (recognitionInProgress) return;
      recognitionInProgress = true;
      clearTimeout(resultTimer);
      setMode('capture');
      setCaptureStatus('Preparation de la reconnaissance...', 'Cadrez votre visage dans l\'ovale', '');
      startStream();
      await enterFullscreen();
      await new Promise(function(resolve) { setTimeout(resolve, 700); });

      stopStream();
      setCaptureStatus('Identification en cours...', 'Veuillez patienter', '');
      try {
        var resp = await fetch('/pointage', { method: 'POST' });
        var data = await resp.json();
        setMode('result');
        if (data.ok && data.matched) {
          showResultSuccess(data);
        } else {
          showResultError(data);
        }
      } catch (e) {
        setMode('result');
        showResultError({ error: 'Connexion API indisponible', error_type: 'recognition_failed' });
      }
      backToStandardSoon();
    }

    viewStandard.addEventListener('pointerdown', function() {
      if (!recognitionInProgress) startCaptureFlow();
    });

    document.addEventListener('keydown', function(e) {
      if ((e.code === 'Space' || e.code === 'Enter') && !recognitionInProgress) {
        e.preventDefault();
        startCaptureFlow();
      }
      if (e.key === 'f' || e.key === 'F') enterFullscreen();
    });

    if (KIOSK_MODE) {
      stdFooter.innerHTML = 'Mode kiosk actif | Lancez la capture avec <span class="key">Espace</span>';
      document.addEventListener('pointerdown', function autoKiosk() {
        enterFullscreen();
        document.removeEventListener('pointerdown', autoKiosk);
      }, { once: true });
      document.addEventListener('keydown', function autoKioskKey() {
        enterFullscreen();
        document.removeEventListener('keydown', autoKioskKey);
      }, { once: true });
    }

    if (CAMERA_MIRROR) feed.style.transform = 'scaleX(-1)';

    updateClock();
    setInterval(updateClock, 1000);
    updateWeather();
    setInterval(updateWeather, 300000);
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

    @app.get("/assets/logo-projet")
    def logo_projet() -> object:
        logo_path = _PROJECT_ROOT / "Logo projet.png"
        if not logo_path.exists():
            return ("", 404)
        return send_file(logo_path)

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