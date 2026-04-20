from __future__ import annotations

import statistics
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
      --ok: #3de89a;
      --bad: #ff6b6b;
      --accent: #42c8de;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    html, body { width: 100%; height: 100%; overflow: hidden; }

    body {
      background: #06101a;
      font-family: "Bahnschrift", "Segoe UI", "Trebuchet MS", sans-serif;
      user-select: none;
      color: #f0f8ff;
    }

    /* ── Conteneur racine ── */
    .app {
      position: relative;
      width: 100vw;
      height: 100dvh;
      overflow: hidden;
    }

    /* ── Fond plein écran ── */
    .logo-bg {
      position: absolute;
      inset: 0;
      z-index: 0;
      overflow: hidden;
      pointer-events: none;
    }

    .logo-bg img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      object-position: center;
      opacity: 0.34;
      filter: saturate(1.02) brightness(1.0);
    }

    /* Vignette douce : assombrit seulement les bords haut/bas */
    .logo-bg::after {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      background:
        linear-gradient(to bottom,
          rgba(4,10,18,0.10) 0%,
          rgba(4,10,18,0.0) 16%,
          rgba(4,10,18,0.0) 80%,
          rgba(4,10,18,0.12) 100%);
    }

    /* ── Vues ── */
    .view {
      position: absolute;
      inset: 0;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.3s ease;
      z-index: 1;
    }
    .view.active { opacity: 1; pointer-events: auto; }

    /* ════════════════════════
       MODE STANDARD
    ════════════════════════ */
    .view-standard {
      display: flex;
      flex-direction: column;
      padding: clamp(20px, 3.5vw, 52px) clamp(24px, 5vw, 68px);
    }

    /* En-tête : marque discrète */
    .std-header { display: flex; align-items: center; }

    .std-brand {
      font-size: clamp(1.18rem, 2.1vw, 1.8rem);
      letter-spacing: 0.38em;
      text-transform: uppercase;
      font-weight: 700;
      color: rgba(255, 255, 255, 0.35);
    }

    /* Zone centrale : contenu flottant, pas de panneau */
    .std-main {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: clamp(12px, 2.2vw, 28px);
      text-align: center;
    }

    .std-bottom {
      display: flex;
      justify-content: center;
      padding-top: clamp(10px, 2vh, 20px);
    }

    /* Horloge héro */
    .clock-time {
      font-size: clamp(1.9rem, 6vw, 4.6rem);
      font-weight: 700;
      letter-spacing: 0.02em;
      line-height: 1.12;
      white-space: pre-line;
      color: #ffffff;
      text-shadow:
        0 0 44px rgba(66, 200, 222, 0.35),
        0 3px 24px rgba(0, 0, 0, 0.9);
    }

    /* Météo : pilule légère */
    .std-cards {
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .card {
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.14);
      background: rgba(8, 18, 30, 0.52);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      padding: 12px 32px;
      display: flex;
      align-items: center;
      gap: 16px;
    }

    .card-title {
      font-size: 0.82rem;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      color: rgba(255, 255, 255, 0.35);
    }

    .card-value {
      font-size: clamp(1.18rem, 1.9vw, 1.45rem);
      color: #e6f6ff;
      font-weight: 600;
    }

    /* Pied : indice Espace */
    .std-footer {
      text-align: center;
      color: rgba(255, 255, 255, 0.28);
      font-size: clamp(0.76rem, 1.05vw, 0.9rem);
    }

    .key {
      display: inline-block;
      border: 1px solid rgba(255, 255, 255, 0.25);
      border-bottom-width: 2px;
      border-radius: 6px;
      padding: 2px 9px;
      margin: 0 3px;
      color: rgba(255, 255, 255, 0.6);
      font-size: 0.86em;
    }

    /* ════════════════════════
       MODE CAPTURE
    ════════════════════════ */
    .view-capture { padding: 0; }

    .capture-wrap {
      position: relative;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #000;
    }

    #feed {
      width: 100%;
      height: 100%;
      object-fit: contain;
      object-position: center center;
      background: #000;
      display: block;
    }

    .scan-oval {
      position: absolute;
      left: 50%;
      top: 50%;
      width: clamp(240px, 33vw, 420px);
      aspect-ratio: 0.76;
      transform: translate(-50%, -54%);
      border-radius: 50%;
      border: 3px solid rgba(255, 255, 255, 0.72);
      box-shadow:
        0 0 0 9999px rgba(0, 0, 0, 0.38),
        0 0 28px rgba(66, 200, 222, 0.22);
      pointer-events: none;
      transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }

    .scan-oval.success {
      border-color: var(--ok);
      box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.32), 0 0 44px rgba(61, 232, 154, 0.55);
    }
    .scan-oval.error {
      border-color: var(--bad);
      box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.32), 0 0 44px rgba(255, 107, 107, 0.55);
    }

    .scan-line {
      position: absolute;
      left: 50%;
      top: 50%;
      width: clamp(200px, 27vw, 340px);
      height: 2px;
      background: linear-gradient(90deg, transparent, rgba(61, 232, 154, 0.95), transparent);
      filter: drop-shadow(0 0 7px rgba(61, 232, 154, 0.8));
      animation: scan 2.2s ease-in-out infinite;
      pointer-events: none;
    }

    @keyframes scan {
      0%   { transform: translate(-50%, -90px); opacity: 0.1; }
      50%  { transform: translate(-50%, 90px);  opacity: 1; }
      100% { transform: translate(-50%, -90px); opacity: 0.1; }
    }

    .capture-overlay {
      position: absolute;
      top: 18px;
      left: 18px;
      right: 18px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .capture-debug {
      border-radius: 10px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: rgba(6, 14, 22, 0.62);
      backdrop-filter: blur(8px);
      padding: 6px 10px;
      color: rgba(255, 255, 255, 0.74);
      font-size: 0.74rem;
      font-family: "Consolas", "Liberation Mono", monospace;
      letter-spacing: 0.02em;
      text-transform: none;
      max-width: min(92vw, 560px);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .capture-status {
      position: absolute;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%);
      min-width: min(88vw, 520px);
      border-radius: 16px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      background: rgba(4, 10, 18, 0.58);
      backdrop-filter: blur(12px);
      padding: 12px 20px;
      text-align: center;
    }

    .capture-status-main {
      font-size: clamp(1rem, 1.7vw, 1.25rem);
      font-weight: 700;
    }

    .capture-status-sub {
      margin-top: 4px;
      font-size: clamp(0.8rem, 1.1vw, 0.94rem);
      color: rgba(255, 255, 255, 0.42);
    }

    /* ════════════════════════
       MODE RÉSULTAT
    ════════════════════════ */
    .view-result {
      display: grid;
      align-items: center;
      justify-items: center;
    }

    .result-bg-word {
      position: absolute;
      inset: auto 0 7%;
      text-align: center;
      font-size: clamp(4rem, 18vw, 18rem);
      font-weight: 900;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: rgba(255, 255, 255, 0.04);
      pointer-events: none;
    }

    .result-card {
      width: min(94vw, 860px);
      border-radius: 28px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: rgba(7, 14, 22, 0.68);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.5);
      padding: clamp(28px, 4.2vw, 56px) clamp(22px, 4.5vw, 58px);
      text-align: center;
      display: grid;
      gap: 12px;
    }

    .result-card.success {
      border-color: rgba(61, 232, 154, 0.28);
      box-shadow: 0 0 70px rgba(61, 232, 154, 0.1), 0 24px 80px rgba(0, 0, 0, 0.48);
    }
    .result-card.error {
      border-color: rgba(255, 107, 107, 0.28);
      box-shadow: 0 0 70px rgba(255, 107, 107, 0.1), 0 24px 80px rgba(0, 0, 0, 0.48);
    }

    .result-tag {
      justify-self: center;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.18);
      background: rgba(255, 255, 255, 0.05);
      color: rgba(255, 255, 255, 0.75);
      font-size: 0.7rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      padding: 7px 16px;
    }

    .result-greeting {
      font-size: clamp(1.3rem, 2.8vw, 2.2rem);
      color: rgba(255, 255, 255, 0.88);
      font-weight: 400;
    }

    .result-kind {
      font-size: clamp(2.8rem, 9.5vw, 7.5rem);
      font-weight: 900;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #ffffff;
      text-shadow: 0 4px 32px rgba(0, 0, 0, 0.65);
    }

    .result-card.success .result-kind { color: var(--ok); }
    .result-card.error   .result-kind { color: var(--bad); }

    .result-meta {
      color: rgba(255, 255, 255, 0.45);
      font-size: clamp(1.3rem, 2.6vw, 1.95rem);
      line-height: 1.35;
      white-space: pre-line;
    }

    .result-schedule {
      display: grid;
      gap: 12px;
      justify-items: center;
      margin-top: 4px;
    }

    .result-flags {
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 10px;
    }

    .result-flag {
      border-radius: 999px;
      padding: 8px 16px;
      border: 1px solid rgba(255, 255, 255, 0.18);
      background: rgba(255, 255, 255, 0.06);
      color: rgba(255, 255, 255, 0.82);
      font-size: clamp(0.82rem, 1.7vw, 1rem);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .result-flag.warning {
      border-color: rgba(255, 196, 87, 0.36);
      background: rgba(255, 196, 87, 0.14);
      color: #ffd36f;
    }

    .result-feedback {
      color: rgba(255, 255, 255, 0.74);
      font-size: clamp(1rem, 2vw, 1.18rem);
      line-height: 1.45;
    }

    @media (max-width: 560px) {
      .view-standard { padding: 14px 16px; }
      .capture-overlay { flex-direction: column; align-items: flex-start; gap: 6px; }
      .capture-debug { white-space: normal; }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="logo-bg" aria-hidden="true"><img src="/assets/logo-projet" alt=""></div>

    <section class="view view-standard active" id="viewStandard">
      <header class="std-header">
        <h1 class="std-brand">BioAttend</h1>
      </header>

      <main class="std-main">
        <div class="clock-time" id="clockTime">--:--:-- | --- | --/--/----</div>
      </main>

      <div class="std-bottom">
        <div class="std-cards">
          <article class="card"><div class="card-title">Meteo</div><div class="card-value" id="weatherLabel">Mise a jour...</div></article>
        </div>
      </div>
    </section>

    <section class="view view-capture" id="viewCapture">
      <div class="capture-wrap">
        <img id="feed" src="/snapshot" alt="Flux camera">
        <div class="scan-oval" id="scanOval"></div>
        <div class="scan-line" aria-hidden="true"></div>
        <div class="capture-overlay">
          <div class="capture-debug" id="captureDebug" hidden>Debug flux</div>
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
        <div class="result-schedule" id="resultSchedule" hidden></div>
      </div>
    </section>
  </div>

  <script>
    var feed = document.getElementById("feed");
    var scanOval = document.getElementById("scanOval");
    var viewStandard = document.getElementById("viewStandard");
    var viewCapture = document.getElementById("viewCapture");
    var viewResult = document.getElementById("viewResult");
    var clockTime = document.getElementById("clockTime");
    var weatherLabel = document.getElementById("weatherLabel");
    var captureStatusMain = document.getElementById("captureStatusMain");
    var captureStatusSub = document.getElementById("captureStatusSub");
    var captureDebug = document.getElementById("captureDebug");
    var resultCard = document.getElementById("resultCard");
    var resultTag = document.getElementById("resultTag");
    var resultGreeting = document.getElementById("resultGreeting");
    var resultKind = document.getElementById("resultKind");
    var resultMeta = document.getElementById("resultMeta");
    var resultSchedule = document.getElementById("resultSchedule");

    var KIOSK_MODE = "__KIOSK_MODE__" === "true";
    var CAMERA_MIRROR = "__CAMERA_MIRROR__" === "true";
    var POINTAGE_TRIGGER_MODE = "__POINTAGE_TRIGGER_MODE__";
    var ULTRASON_DISTANCE_CM = parseFloat("__ULTRASON_DISTANCE_CM__") || 80;
    var ULTRASON_CAPTURE_PREP_DELAY_MS = parseInt("__ULTRASON_CAPTURE_PREP_DELAY_MS__", 10);
    var ULTRASON_PRESENCE_COOLDOWN_MS = parseInt("__ULTRASON_PRESENCE_COOLDOWN_MS__", 10);
    var MANUAL_TRIGGER_ENABLED = POINTAGE_TRIGGER_MODE === "space";
    var AUTO_TRIGGER_ENABLED = !MANUAL_TRIGGER_ENABLED;

    function getQueryParam(name) {
      var search = window.location.search || "";
      if (!search || search.length < 2) return null;
      var items = search.substring(1).split("&");
      for (var i = 0; i < items.length; i++) {
        var pair = items[i].split("=");
        if (decodeURIComponent(pair[0] || "") === name) {
          return decodeURIComponent(pair[1] || "");
        }
      }
      return null;
    }

    var SHOW_BOXES = getQueryParam("boxes") === "1";
    var SHOW_DEBUG = getQueryParam("debug") === "1";

    var streamRunning = false;
    var streamTimer = null;
    var recognitionInProgress = false;
    var resultTimer = null;
    var CAPTURE_PREP_DELAY_MS = 1800;
    if (POINTAGE_TRIGGER_MODE === "ultrason" && !isNaN(ULTRASON_CAPTURE_PREP_DELAY_MS) && ULTRASON_CAPTURE_PREP_DELAY_MS >= 0) {
      CAPTURE_PREP_DELAY_MS = ULTRASON_CAPTURE_PREP_DELAY_MS;
    }

    function setMode(mode) {
      viewStandard.classList.toggle("active", mode === "standard");
      viewCapture.classList.toggle("active", mode === "capture");
      viewResult.classList.toggle("active", mode === "result");
    }

    function setCaptureStatus(mainText, subText, mood) {
      captureStatusMain.textContent = mainText;
      captureStatusSub.textContent = subText;
      scanOval.className = "scan-oval" + (mood ? " " + mood : "");
    }

    function scheduleNextFrame(delay) {
      if (!streamRunning) return;
      clearTimeout(streamTimer);
      streamTimer = setTimeout(function() {
        feed.src = "/snapshot?boxes=" + (SHOW_BOXES ? "1" : "0") + "&t=" + Date.now();
      }, delay);
    }

    function updateCaptureDebug() {
      if (!SHOW_DEBUG || !captureDebug || !feed) return;

      var camW = feed.naturalWidth || 0;
      var camH = feed.naturalHeight || 0;
      var viewW = feed.clientWidth || window.innerWidth || 0;
      var viewH = feed.clientHeight || window.innerHeight || 0;

      var camRatio = camW > 0 && camH > 0 ? (camW / camH).toFixed(3) : "n/a";
      var viewRatio = viewW > 0 && viewH > 0 ? (viewW / viewH).toFixed(3) : "n/a";

      captureDebug.textContent =
        "cam: " + camW + "x" + camH + " (r=" + camRatio + ") | " +
        "zone: " + viewW + "x" + viewH + " (r=" + viewRatio + ") | " +
        "fit: contain (no crop)";
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

    feed.addEventListener("load", function() {
      updateCaptureDebug();
      scheduleNextFrame(90);
    });
    feed.addEventListener("error", function() { scheduleNextFrame(180); });
    window.addEventListener("resize", updateCaptureDebug);

    var _DAYS = ["Dimanche", "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"];
    var _MONTHS = ["janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre", "octobre", "novembre", "decembre"];

    function _pad(n) { return String(n).padStart(2, "0"); }

    function _fmtTime(d) {
      return _pad(d.getHours()) + ":" + _pad(d.getMinutes()) + ":" + _pad(d.getSeconds());
    }

    function _fmtDateEuroLong(d) {
      return _DAYS[d.getDay()] + " " + _pad(d.getDate()) + " " + _MONTHS[d.getMonth()] + " " + d.getFullYear();
    }

    function updateClock() {
      try {
        var now = new Date();
        if (clockTime) {
          clockTime.textContent = _fmtDateEuroLong(now) + "\\n" + _fmtTime(now);
        }
      } catch(e) {}
    }

    function enterFullscreen() {
      var el = document.documentElement;
      if (!document.fullscreenElement && el.requestFullscreen) {
        try {
          el.requestFullscreen();
        } catch (e) {}
      }
    }

    function _jsonGet(url, onSuccess, onError) {
      try {
        var xhr = new XMLHttpRequest();
        xhr.open("GET", url, true);
        xhr.onreadystatechange = function() {
          if (xhr.readyState !== 4) return;
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              var payload = JSON.parse(xhr.responseText || "{}");
              onSuccess(payload);
            } catch (err) {
              onError(err);
            }
            return;
          }
          onError(new Error("http_" + xhr.status));
        };
        xhr.onerror = function() { onError(new Error("network")); };
        xhr.send();
      } catch (err) {
        onError(err);
      }
    }

    function _jsonPost(url, body, onSuccess, onHttpError, onNetworkError) {
      try {
        var xhr = new XMLHttpRequest();
        xhr.open("POST", url, true);
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.onreadystatechange = function() {
          if (xhr.readyState !== 4) return;
          var payload = {};
          try {
            payload = JSON.parse(xhr.responseText || "{}");
          } catch (err) {
            payload = {};
          }

          if (xhr.status >= 200 && xhr.status < 300) {
            onSuccess(payload, xhr.status);
            return;
          }
          onHttpError(payload, xhr.status);
        };
        xhr.onerror = function() { onNetworkError(new Error("network")); };
        xhr.send(body ? JSON.stringify(body) : "{}");
      } catch (err) {
        onNetworkError(err);
      }
    }

    var FALLBACK_WEATHER_CITY = "Mons";
    var FALLBACK_WEATHER_LAT = 50.4542;
    var FALLBACK_WEATHER_LON = 3.9523;

    function _weatherIconFromCode(code) {
      // Codes meteo Open-Meteo: https://open-meteo.com/en/docs
      if (typeof code !== "number") return "\u2601";
      if (code === 0) return "\u2600"; // ciel degage
      if (code === 1 || code === 2) return "\u26c5"; // peu nuageux
      if (code === 3) return "\u2601"; // couvert
      if (code === 45 || code === 48) return "\u2601"; // brouillard
      if (code === 51 || code === 53 || code === 55 || code === 56 || code === 57) return "\u2614"; // bruine
      if (code === 61 || code === 63 || code === 65 || code === 66 || code === 67 || code === 80 || code === 81 || code === 82) return "\u2614"; // pluie
      if (code === 71 || code === 73 || code === 75 || code === 77 || code === 85 || code === 86) return "\u2744"; // neige
      if (code === 95 || code === 96 || code === 99) return "\u26a1"; // orage
      return "\u2601";
    }

    function fetchWeatherFor(lat, lon, sourceName, onSuccess, onError) {
      var url = "https://api.open-meteo.com/v1/forecast?latitude=" + lat + "&longitude=" + lon + "&current=temperature_2m,weather_code&timezone=auto";
      _jsonGet(url, function(data) {
        if (!data.current || typeof data.current.temperature_2m !== "number") {
          onError(new Error("meteo"));
          return;
        }
        var weatherCode = typeof data.current.weather_code === "number" ? data.current.weather_code : null;
        var weatherIcon = _weatherIconFromCode(weatherCode);
        if (weatherLabel) {
          weatherLabel.textContent = weatherIcon + " " + Math.round(data.current.temperature_2m) + "\u00b0C";
          if (sourceName) weatherLabel.title = sourceName;
        }
        onSuccess();
      }, onError);
    }

    function updateWeather() {
      if (!("geolocation" in navigator)) {
        fetchWeatherFor(FALLBACK_WEATHER_LAT, FALLBACK_WEATHER_LON, FALLBACK_WEATHER_CITY, function() {}, function() {
          if (weatherLabel) weatherLabel.textContent = "Meteo indisponible";
        });
        return;
      }
      try {
        navigator.geolocation.getCurrentPosition(function(pos) {
          fetchWeatherFor(pos.coords.latitude, pos.coords.longitude, "Position actuelle", function() {}, function() {
            if (weatherLabel) weatherLabel.textContent = "Meteo indisponible";
          });
        }, function() {
          fetchWeatherFor(FALLBACK_WEATHER_LAT, FALLBACK_WEATHER_LON, FALLBACK_WEATHER_CITY, function() {}, function() {
            if (weatherLabel) weatherLabel.textContent = "Meteo indisponible";
          });
        }, { timeout: 7000, maximumAge: 600000 });
      } catch (e) {
        fetchWeatherFor(FALLBACK_WEATHER_LAT, FALLBACK_WEATHER_LON, FALLBACK_WEATHER_CITY, function() {}, function() {
          if (weatherLabel) weatherLabel.textContent = "Meteo indisponible";
        });
      }
    }

    function showResultSuccess(data) {
      var type = data.pointage_type === "ENTREE" ? "ENTREE" : "SORTIE";
      var now = new Date();
      var metaLines = [
        "Heure: " + _fmtTime(now),
        "Date: " + _fmtDateEuroLong(now)
      ];

      if (type === "SORTIE" && data.worked_duration_display) {
        metaLines.push("Temps effectif: " + data.worked_duration_display);
      }

      resultCard.className = "result-card success";
      resultTag.textContent = "Pointage valide";
      resultGreeting.textContent = "Bonjour " + data.full_name;
      resultKind.textContent = type;
      resultMeta.textContent = metaLines.join("\n");
      renderScheduleDetails(data);
    }

    function _scheduleFlagLabel(flag) {
      var labels = {
        "RETARD": "Retard",
        "DEPART_ANTICIPE": "Depart anticipe",
        "JOURNEE_COURTE": "Journee courte"
      };
      return labels[flag] || flag;
    }

    function renderScheduleDetails(data) {
      var flags = Array.isArray(data.schedule_flags) ? data.schedule_flags.filter(Boolean) : [];
      var feedback = Array.isArray(data.schedule_feedback) ? data.schedule_feedback.filter(Boolean) : [];

      resultSchedule.innerHTML = "";
      resultSchedule.hidden = true;

      if (!flags.length && !feedback.length) {
        return;
      }

      if (flags.length) {
        var flagsWrap = document.createElement("div");
        flagsWrap.className = "result-flags";
        flags.forEach(function(flag) {
          var badge = document.createElement("span");
          badge.className = "result-flag" + (flag === "RETARD" || flag === "DEPART_ANTICIPE" || flag === "JOURNEE_COURTE" ? " warning" : "");
          badge.textContent = _scheduleFlagLabel(flag);
          flagsWrap.appendChild(badge);
        });
        resultSchedule.appendChild(flagsWrap);
      }

      if (feedback.length) {
        var feedbackNode = document.createElement("div");
        feedbackNode.className = "result-feedback";
        feedbackNode.textContent = feedback.join(" | ");
        resultSchedule.appendChild(feedbackNode);
      }

      resultSchedule.hidden = false;
    }

    function showResultError(data) {
      var errorType = data.error_type || "recognition_failed";
      var titleByType = {
        "no_face_detected": "Visage non detecte",
        "recognition_failed": "Echec de reconnaissance",
        "unknown_user": "Utilisateur inconnu",
        "spoof_attempt": "Anti-spoof"
      };
      resultCard.className = "result-card error";
      resultTag.textContent = titleByType[errorType] || "Pointage refuse";
      resultGreeting.textContent = data.error || "Veuillez recommencer";
      resultKind.textContent = "ECHEC";
      resultMeta.textContent = "Heure: " + _fmtTime(new Date());
      resultSchedule.innerHTML = "";
      resultSchedule.hidden = true;
    }

    function backToStandardSoon() {
      clearTimeout(resultTimer);
      resultTimer = setTimeout(function() {
        recognitionInProgress = false;
        setMode("standard");
        setCaptureStatus("Preparation de la reconnaissance...", "Ne bougez pas pendant la lecture", "");
      }, 4200);
    }

    function startCaptureFlow() {
      if (recognitionInProgress) return;
      recognitionInProgress = true;
      clearTimeout(resultTimer);
      setMode("capture");
      setCaptureStatus("Preparation de la reconnaissance...", "Cadrez votre visage dans l\u2019ovale", "");
      startStream();
      enterFullscreen();
      setTimeout(function() {
        stopStream();
        setCaptureStatus("Identification en cours...", "Veuillez patienter", "");
        _jsonPost("/pointage", {}, function(data) {
          setMode("result");
          if (data.ok && data.matched) {
            showResultSuccess(data);
          } else {
            showResultError(data);
          }
          backToStandardSoon();
        }, function(data) {
          setMode("result");
          showResultError(data && typeof data === "object" ? data : { error: "Pointage refuse", error_type: "recognition_failed" });
          backToStandardSoon();
        }, function() {
          setMode("result");
          showResultError({ error: "Connexion API indisponible", error_type: "recognition_failed" });
          backToStandardSoon();
        });
      }, CAPTURE_PREP_DELAY_MS);
    }

    
    function triggerCaptureFromUserInput(e) {
      if (e && typeof e.preventDefault === "function") e.preventDefault();
      if (!recognitionInProgress) startCaptureFlow();
    }

    if (viewStandard) {
      if (MANUAL_TRIGGER_ENABLED) {
        if ("onpointerdown" in window) {
          viewStandard.addEventListener("pointerdown", triggerCaptureFromUserInput);
        }
        viewStandard.addEventListener("click", triggerCaptureFromUserInput);
        viewStandard.addEventListener("touchstart", triggerCaptureFromUserInput, false);
      }
    }

    function onKeydown(e) {
      if (e.repeat) return;
      var pressedKey = (e.key || "").toLowerCase();
      var code = e.code || "";
      var isSpace = code === "Space" || pressedKey === " " || pressedKey === "spacebar" || e.keyCode === 32;
      var isEnter = code === "Enter" || pressedKey === "enter" || e.keyCode === 13;
      if (MANUAL_TRIGGER_ENABLED && (isSpace || isEnter)) {
        e.preventDefault();
        if (!recognitionInProgress) startCaptureFlow();
        return;
      }
      if (pressedKey === "f") enterFullscreen();
    }

    document.addEventListener("keydown", onKeydown);
    window.addEventListener("keydown", onKeydown);

    document.body.addEventListener("click", function() { document.body.focus(); });

    if (KIOSK_MODE && MANUAL_TRIGGER_ENABLED) {
      document.addEventListener("pointerdown", function() {
        enterFullscreen();
      }, false);
    }

    if (CAMERA_MIRROR) feed.style.transform = "scaleX(-1)";

    if (SHOW_DEBUG && captureDebug) {
      captureDebug.hidden = false;
      updateCaptureDebug();
      setInterval(updateCaptureDebug, 1000);
    }

    // ── Polling capteur de presence (PIR / ultrason) ───────────────────────
    if (AUTO_TRIGGER_ENABLED) {
      var _presenceLastDetected = false;
      var _presenceCooldownUntil = 0;
      var PRESENCE_POLL_INTERVAL_MS = 400;
      var PRESENCE_COOLDOWN_MS = !isNaN(ULTRASON_PRESENCE_COOLDOWN_MS) && ULTRASON_PRESENCE_COOLDOWN_MS >= 0
        ? ULTRASON_PRESENCE_COOLDOWN_MS
        : 8000;

      function _presencePoll() {
        if (recognitionInProgress) return;
        var now = Date.now();
        if (now < _presenceCooldownUntil) return;
        var xhr = new XMLHttpRequest();
        xhr.open("GET", "/presence/status", true);
        xhr.timeout = 350;
        xhr.onload = function() {
          if (xhr.status !== 200) return;
          try {
            var data = JSON.parse(xhr.responseText);
            var detected = data.ok && data.detected;
            if (detected && !_presenceLastDetected && !recognitionInProgress && Date.now() >= _presenceCooldownUntil) {
              _presenceCooldownUntil = Date.now() + PRESENCE_COOLDOWN_MS;
              startCaptureFlow();
            }
            _presenceLastDetected = detected;
          } catch(e) {}
        };
        xhr.send();
      }

      setInterval(_presencePoll, PRESENCE_POLL_INTERVAL_MS);
    }
    // ─────────────────────────────────────────────────────────────────────────

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

    # ── Initialisation GPIO capteurs de présence ─────────────────────────────
    _gpio_module = None
    _gpio_pir_available = False
    _gpio_ultrason_available = False
    if settings.pointage_trigger_mode in {"pir", "ultrason"}:
        try:
            import RPi.GPIO as _gpio  # type: ignore[import]
            _gpio.setmode(_gpio.BCM)
            _gpio_module = _gpio
            if settings.pointage_trigger_mode == "pir":
                _gpio.setup(settings.gpio_pir, _gpio.IN)
                _gpio_pir_available = True
            elif settings.pointage_trigger_mode == "ultrason":
                _gpio.setup(settings.gpio_ultrason_trigger, _gpio.OUT)
                _gpio.setup(settings.gpio_ultrason_echo, _gpio.IN)
                _gpio.output(settings.gpio_ultrason_trigger, False)
                time.sleep(0.05)
                _gpio_ultrason_available = True
        except Exception:
            pass
    # ────────────────────────────────────────────────────────────────────────

    def _read_ultrason_distance_cm(timeout_s: float = 0.03) -> float | None:
        if not _gpio_ultrason_available or _gpio_module is None:
            return None

        trigger_pin = settings.gpio_ultrason_trigger
        echo_pin = settings.gpio_ultrason_echo

        try:
            _gpio_module.output(trigger_pin, False)
            time.sleep(0.000002)
            _gpio_module.output(trigger_pin, True)
            time.sleep(0.00001)
            _gpio_module.output(trigger_pin, False)

            wait_start = time.perf_counter()
            pulse_start = wait_start
            while _gpio_module.input(echo_pin) == 0:
                pulse_start = time.perf_counter()
                if (pulse_start - wait_start) > timeout_s:
                    return None

            pulse_end = pulse_start
            while _gpio_module.input(echo_pin) == 1:
                pulse_end = time.perf_counter()
                if (pulse_end - pulse_start) > timeout_s:
                    return None

            duration_s = pulse_end - pulse_start
            distance_cm = (duration_s * 34300.0) / 2.0
            if distance_cm <= 0.0 or distance_cm > 600.0:
                return None
            return float(distance_cm)
        except Exception:
            return None

    def _ultrason_status_payload() -> tuple[dict[str, object], int]:
        if settings.pointage_trigger_mode != "ultrason":
            return {"ok": False, "reason": "mode_not_ultrason"}, 400
        if not _gpio_ultrason_available:
            return {"ok": False, "reason": "gpio_unavailable"}, 503

        samples: list[float] = []
        for _ in range(3):
            measure = _read_ultrason_distance_cm()
            if measure is not None:
                samples.append(measure)
            time.sleep(0.01)

        if not samples:
            return {"ok": False, "reason": "no_signal"}, 503

        distance_cm = round(float(statistics.median(samples)), 1)
        threshold_cm = float(settings.ultrason_distance_cm)
        detected = distance_cm <= threshold_cm
        return {
            "ok": True,
            "detected": detected,
            "distance_cm": distance_cm,
            "threshold_cm": threshold_cm,
            "mode": "ultrason",
        }, 200

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

    def _scan_oval_geometry(frame_w: int, frame_h: int) -> tuple[float, float, float, float]:
        # Aligne la zone d'acceptation backend sur l'ovale affiché dans l'UI.
        oval_w = max(240.0, min(float(frame_w) * 0.33, 420.0))
        oval_h = oval_w / 0.76
        cx = float(frame_w) * 0.5
        cy = (float(frame_h) * 0.5) - (oval_h * 0.04)
        rx = oval_w * 0.5
        ry = oval_h * 0.5
        return cx, cy, rx, ry

    def _point_in_oval(px: float, py: float, cx: float, cy: float, rx: float, ry: float) -> bool:
        if rx <= 0 or ry <= 0:
            return False
        dx = (px - cx) / rx
        dy = (py - cy) / ry
        return (dx * dx + dy * dy) <= 1.0

    def _is_reliable_face_result(face_result: dict, frame_w: int, frame_h: int) -> bool:
        primary = face_result.get("primary_face") or {}

        w = int(primary.get("w", 0))
        h = int(primary.get("h", 0))
        x = int(primary.get("x", 0))
        y = int(primary.get("y", 0))

        if w < 72 or h < 72:
            return False

        margin_x = max(2, int(frame_w * 0.01))
        margin_y = max(2, int(frame_h * 0.01))
        if x <= margin_x or y <= margin_y or (x + w) >= (frame_w - margin_x) or (y + h) >= (frame_h - margin_y):
            return False

        cx, cy, rx, ry = _scan_oval_geometry(frame_w, frame_h)
        face_cx = x + (w * 0.5)
        face_cy = y + (h * 0.5)

        # Le centre du visage doit être dans l'ovale de scan.
        if not _point_in_oval(face_cx, face_cy, cx, cy, rx, ry):
            return False

        return True

    def _capture_with_face(max_attempts: int = 8, delay_ms: int = 70) -> dict:
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
                            if iou >= 0.10:
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

    @app.get("/pir/status")
    def pir_status() -> tuple[object, int]:
        if settings.pointage_trigger_mode != "pir":
            return jsonify({"ok": False, "reason": "mode_not_pir"}), 400
        if not _gpio_pir_available or _gpio_module is None:
            return jsonify({"ok": False, "reason": "gpio_unavailable"}), 503
        try:
            detected = bool(_gpio_module.input(settings.gpio_pir))
            return jsonify({"ok": True, "detected": detected}), 200
        except Exception as exc:
            return jsonify({"ok": False, "reason": str(exc)}), 500

    @app.get("/ultrason/status")
    def ultrason_status() -> tuple[object, int]:
        payload, status = _ultrason_status_payload()
        return jsonify(payload), status

    @app.get("/presence/status")
    def presence_status() -> tuple[object, int]:
        if settings.pointage_trigger_mode == "pir":
            if not _gpio_pir_available or _gpio_module is None:
                return jsonify({"ok": False, "reason": "gpio_unavailable"}), 503
            try:
                detected = bool(_gpio_module.input(settings.gpio_pir))
                return jsonify({"ok": True, "detected": detected, "mode": "pir"}), 200
            except Exception as exc:
                return jsonify({"ok": False, "reason": str(exc)}), 500

        if settings.pointage_trigger_mode == "ultrason":
            payload, status = _ultrason_status_payload()
            return jsonify(payload), status

        return jsonify({"ok": False, "reason": "mode_not_sensor"}), 400

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
        html = html.replace("__POINTAGE_TRIGGER_MODE__", settings.pointage_trigger_mode)
        html = html.replace("__ULTRASON_DISTANCE_CM__", str(settings.ultrason_distance_cm))
        html = html.replace("__ULTRASON_CAPTURE_PREP_DELAY_MS__", str(settings.ultrason_capture_prep_delay_ms))
        html = html.replace("__ULTRASON_PRESENCE_COOLDOWN_MS__", str(settings.ultrason_presence_cooldown_ms))
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
            capture_result = capture_frame(settings)
            if not capture_result.get("ok", False):
                return jsonify(capture_result), 503
        frame = capture_result["frame"]

        show_boxes = request.args.get("boxes", "0") == "1"
        show_liveness_overlay = request.args.get("liveness", "0") == "1"
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

                # En mode flux camera, la liveness reste désactivée par défaut.
                # Elle ne s'affiche qu'en diagnostic explicite (?boxes=1&liveness=1).
                if settings.liveness_enabled and show_liveness_overlay:
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
      app.logger.info(
        "[pointage] identify result target=%s status=%s ok=%s matched=%s full_name=%s pointage_type=%s payload=%s",
        api_result.get("target"),
        api_result.get("status_code"),
        api_result.get("ok"),
        api_response.get("matched"),
        api_response.get("full_name"),
        api_response.get("pointage_type"),
        api_response,
      )
      if api_result.get("ok") and api_response.get("matched"):
        return jsonify({
          "ok": True,
          "matched": True,
          "user_id": api_response.get("user_id"),
          "username": api_response.get("username"),
          "full_name": api_response.get("full_name"),
          "distance": api_response.get("distance"),
          "pointage_type": api_response.get("pointage_type"),
          "pointage_id": api_response.get("pointage_id"),
          "schedule_feedback": api_response.get("schedule_feedback", []),
          "schedule_flags": api_response.get("schedule_flags", []),
          "worked_duration_display": api_response.get("worked_duration_display"),
          "liveness": {
            "enabled": bool(settings.liveness_enabled),
            "is_live": None if liveness_result is None else liveness_result.get("is_live"),
            "score": None if liveness_result is None else liveness_result.get("score"),
            "backend": None if liveness_result is None else liveness_result.get("backend"),
          },
        }), 200

      event_type = "unknown_user" if api_result.get("status_code") in {401, 404} else "recognition_failed"
      app.logger.warning(
        "[pointage] identify rejected event_type=%s status=%s ok=%s matched=%s error=%s",
        event_type,
        api_result.get("status_code"),
        api_result.get("ok"),
        api_response.get("matched"),
        api_response.get("error"),
      )
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