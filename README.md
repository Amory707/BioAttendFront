# BioAttend Front — Pointeuse / Interface locale

BioAttendFront est l'application légère qui tourne sur la borne (Raspberry Pi).  
Elle gère la capture caméra, la détection et recadrage du visage, la génération d'embeddings via InsightFace, la vérification de liveness (optionnelle), l'appel au serveur central (BioAttend) pour identification et la présentation d'une UI kiosque (Flask + page HTML/JS unique).

- Langage principal : Python (Flask)
- UI : page HTML embarquée (single-file template dans `main.py`)
- Usage typique : tourner sur un Raspberry Pi connecté à une caméra pour fonctionner en mode kiosque.

---

## Table des matières

- [Quickstart local](#quickstart-local)
- [Variables d'environnement / configuration](#variables-denvironnement--configuration)
- [Dépendances principales](#dépendances-principales)
- [Structure du projet](#structure-du-projet)
- [Endpoints exposés](#endpoints-exposés)
- [Flux principal (capture → identification)](#flux-principal-capture--identification)
- [Système de liveness & modèles ONNX](#système-de-liveness--modèles-onnx)
- [GitHub Action : déploiement sur Raspberry Pi (détail)](#github-action-déploiement-sur-raspberry-pi-détail)
- [Exemples d'utilisation (cURL) et debugging](#exemples-ducurl-et-debugging)
- [Conseils d'exploitation / sécurité](#conseils-dexploitation--sécurité)

---

## Quickstart local

1. Cloner le dépôt et positionner-toi à la racine du projet.
2. Créer un fichier `.env` (voir section variables).
3. Installer les dépendances dans un venv :
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. Lancer l'application (mode développement) :
   ```bash
   # méthode 1 (recommandé pour dev)
   export FLASK_APP=app
   flask --app app run --host 0.0.0.0 --port 5000

   # ou méthode directe (si python trouve app.app)
   python -m flask --app app run --host 0.0.0.0 --port 5000
   ```
5. Ouvrir `http://<pi-ip>:5000/` dans un navigateur pour l'UI kiosque.

Remarque : le projet lit `.env` via python-dotenv (fichier résolu dans `config.py`).

---

## Variables d'environnement / configuration

La configuration est centralisée dans `src/bioattend_front/config.py`. Les variables sont lues depuis `.env`. Voici les plus importantes (avec leurs valeurs par défaut si présentes) :

- DEBUG (bool) — par défaut False
- KIOSK_MODE (bool) — default True
- DEVICE_NAME — nom de l’appareil (par défaut `bioattend-pi`)
- CAMERA_MIRROR (bool) — mirror horizontal du feed (True par défaut)
- CAMERA_SWAP_RB (bool)
- CAMERA_JPEG_QUALITY (int) — 40..95 (default 68)
- LIVENESS_ENABLED (bool) — active la vérification anti-spoof
- LIVENESS_MODEL_DIR — dossier des modèles liveness (défaut `./models/liveness`)
- LIVENESS_THRESHOLD (float) — seuil
- LIVENESS_LIVE_CLASS_IDX (int) — index de la classe live
- CAMERA_WIDTH / CAMERA_HEIGHT — résolution par défaut (1280x720)
- CAMERA_DEVICE / CAMERA_SOURCE / CAMERA_BACKEND — source de la caméra
- CAMERA_WARMUP_MS / CAMERA_READ_ATTEMPTS
- INSIGHTFACE_MODEL_NAME — modèle InsightFace utilisé (`buffalo_l` par défaut)
- INSIGHTFACE_DET_WIDTH / INSIGHTFACE_DET_HEIGHT
- SERVER_URL — URL d’identification (par défaut fourni dans le code, remplacer par votre serveur)
- EVENTS_URL — URL pour poster les events (ex: `/api/front/events/`)
- API_TOKEN — token utilisé pour Authorization / X-API-Key
- API_TIMEOUT_SECONDS — timeouts pour appels HTTP (default 8s)
- POINTAGE_TRIGGER_MODE — `space` | `pir` | `ultrason` (par défaut `space`)
- ULTRASON_CAPTURE_PREP_DELAY_MS — délai before taking picture (pour ultrason)
- ULTRASON_PRESENCE_COOLDOWN_MS — cooldown
- GPIO_PIR — pin BCM pour PIR
- GPIO_ULTRASON_TRIGGER / GPIO_ULTRASON_ECHO — pins pour HC-SR04
- ULTRASON_DISTANCE_CM — distance seuil pour présence (default 80.0)

Fonction utilitaire : `Settings.from_env()` (voir `config.py`) charge et normalise ces valeurs. `Settings.as_public_dict()` donne une version masquée (utile pour diagnostics).

Exemple minimal `.env` :
```dotenv
DEBUG=true
KIOSK_MODE=true
DEVICE_NAME=bioattend-pi-01
SERVER_URL=https://mon-bioattend.example.com/api/face/identify/
EVENTS_URL=https://mon-bioattend.example.com/api/front/events/
API_TOKEN=MaCleApiPourLesPointeuses
LIVENESS_ENABLED=false
```

---

## Dépendances principales

Les dépendances requises (essentielles observées dans le code) :
- Flask
- requests
- python-dotenv
- opencv-python (cv2)
- numpy
- insightface (pour génération d'embeddings côté front si besoin)
- (facultatif) RPi.GPIO — si tu utilises les GPIO du Pi (PIR / ultrason)
- (pour conversion/ONNX) onnx, onnxscript, torch (CPU wheel)
- autres utilitaires selon `requirements.txt`

Installe via `pip install -r requirements.txt`. Sur Raspberry Pi, préférer des wheels compatibles CPU et installer les paquets système requis pour OpenCV si besoin.

---

## Structure du projet (principaux fichiers)

- `app.py` — bootstrap : ajoute `src` au PYTHONPATH et importe `create_app()`.
- `src/bioattend_front/`
  - `main.py` — définition complète de l'app Flask (routes, logique capture, pointage, diagnostics).
  - `api_client.py` — fonctions HTTP vers le serveur central (`identify_embedding`, `report_event`).
  - `config.py` — chargement et normalisation des variables d'environnement (`Settings`).
  - `camera.py` — logique de capture caméra (frame capture, fast capture, platform metadata).
  - `face.py` — détection & recadrage de visage (renvoie bbox, face_crop).
  - `embedding.py` — génération d'embedding à partir d'un crop (insightface/onnx).
  - `liveness.py` — vérification anti-spoof (liveness) ; wrapper pour modèles installés.
  - `models/` — (dans root) modèles liveness ONNX (déployés par l'action).
- `Logo projet.png` — logo affiché en fond UI.
- `.github/workflows/pi.yaml` — workflow GitHub Actions pour déployer sur le Pi.
- `requirements.txt` — listes des packages.

---

## Endpoints exposés (serveur local sur le Pi)

Routes HTTP utiles exposées par l'app Flask :

- GET `/`  
  - UI kiosque HTML (page unique embarquée).

- GET `/health`  
  - Test de vie : `{ "ok": true, "service": "bioattend-front" }`

- GET `/pir/status`  
  - Statut du capteur PIR (mode `pir`). Réponses : 200 `{ok: true, detected: bool}` | 400 mode incorrect | 503 gpio indisponible.

- GET `/ultrason/status`  
  - Statut capteur ultrason (mode `ultrason`). Retourne JSON `{ok, detected, distance_cm, threshold_cm, mode}`

- GET `/presence/status`  
  - Statut combiné (renvoie le mode actif et `detected`).

- GET `/diagnostics/config`  
  - Retourne la config publique `{ ok: true, config: {...} }`

- GET/POST `/diagnostics/camera`  
  - Test de la caméra (probe), renvoie `{ok: true, ...}` ou 503.

- GET/POST `/diagnostics/face`  
  - Capture et detection de face, renvoie informations sur le visage détecté.

- GET/POST `/diagnostics/embedding`  
  - Génère embedding depuis un face crop (retire le vecteur brut dans la réponse par défaut pour sécurité).

- GET/POST `/diagnostics/identify`  
  - Capture, génère embedding et appelle le serveur central d'identification (report_result inclus).

- GET `/diagnostics/liveness`  
  - Test liveness (si activé) — renvoie score et verdict.

- GET `/diagnostics/liveness/preview`  
  - Renvoie un JPEG annoté (bbox & regions) pour debug.

- GET `/snapshot`  
  - Renvoie la dernière image encodée en JPEG (option `?boxes=1` pour dessiner bbox, `?liveness=1` pour overlay liveness en diagnostic).

- POST `/pointage`  
  - Flux principal déclenché par l'UI : effectue capture(s) jusqu'à confirma­tion de face, (liveness si activé), génération d'embedding, appel API centrale (`identify_embedding`) :
    - Succès 200 : JSON `{ ok: true, matched: true, user_id, username, full_name, pointage_type, ... }`
    - Erreurs diverses : 422 (no face), 503 (capture/embedding/liveness error), 401 (unknown/spoof), etc.
  - En cas d'échec l'app appelle `report_event()` pour journaliser l'événement vers `EVENTS_URL`.

- GET `/assets/logo-projet`  
  - Sert l'image `Logo projet.png`

---

## Flux principal (capture → identification) résumé

1. L'UI déclenche `/pointage`.
2. Le serveur exécute `_capture_with_face()` : prend plusieurs frames rapides jusqu'à confirmation d'une face stable (IoU entre bbox successives).
3. Si `liveness_enabled` : exécute `check_liveness(...)`. Si échec → événement `spoof_attempt` ou rejet.
4. Génère l'embedding via `generate_embedding(...)` (insightface / ONNX).
5. Appelle `identify_embedding(embedding, settings)` dans `api_client.py` qui POSTe vers `SERVER_URL` (ton serveur central — ex `/api/face/identify/`) avec headers Authorization/X-API-Key (API_TOKEN).
6. Si identifiée : renvoie success au frontend (UI) qui affiche carte verte et informations.
7. Si non identifiée ou erreur : renvoie message d'erreur et journalise l'événement (report_event) auprès du serveur central (EVENTS_URL).

---

## Système de liveness & modèles ONNX

- `liveness.py` encapsule la logique d'anti-spoof. Les modèles attendus sont des fichiers `.onnx` placés dans `models/liveness`.
- L'action GitHub (décrite ci‑dessous) contient un step "Build liveness ONNX models" qui :
  - Clone Silent-Face-Anti-Spoofing (SFA),
  - Lance `scripts/make_liveness_onnx.py` pour produire les .onnx,
  - Upload les modèles générés sur le Pi dans le dossier `${DEPLOY_PATH}/models/liveness`.
- Sur le Pi, `LIVENESS_MODEL_DIR` (variable) doit pointer sur ce dossier.
- `LIVENESS_THRESHOLD` et `LIVENESS_LIVE_CLASS_IDX` contrôlent la décision `is_live`.

---

## GitHub Action — déploiement sur le Raspberry Pi (détail)

Le workflow est `.github/workflows/pi.yaml`. Principales caractéristiques et étapes :

### Déclenchement
- Événement : `push` sur n’importe quelle branche (`'**'`) et `workflow_dispatch` (manuel).
- Permissions : `contents: read`, `id-token: write`.

### Secrets requis (à configurer dans GitHub)
- `PI_HOST` — adresse/hostname Tailscale ou IP du Pi.
- `TAILSCALE_AUTHKEY` — clé d'auth Tailscale pour connecter le runner au réseau Tailscale.
- `PI_PASSWORD` — mot de passe SSH du user `bioattend` sur le Pi (utilisé par sshpass).
  - Note : le workflow définit `PI_USER` à `bioattend` en dur.

> Les secrets sont référencés dans l'action via `${{ secrets.NAME }}`.

### Étapes clefs du job `deploy`
1. Checkout du dépôt.
2. Installation de `sshpass` (permet fournir le mot de passe SSH via variable d'environnement `SSHPASS`).
3. Connexion du runner à Tailscale (`tailscale/github-action@v2`) avec `authkey`.
4. Vérification connectivité Tailscale & ping vers `${PI_HOST}`.
5. Boucle d'attente SSH : tente une connexion SSH (max 15 essais) pour s'assurer que le Pi est joignable.
6. Création d'un répertoire de déploiement sur le Pi :
   - `DEPLOY_PATH="/home/bioattend/${BRANCH}"` où `${BRANCH}` est la branche courante (`github.ref_name`).
   - Cela permet avoir un déploiement isolé par branche sur le Pi.
7. Synchronisation des fichiers avec `rsync` (exclut `.git`, `.github`, `venv`, `__pycache__`, `*.pyc`).
8. Installation des dépendances Python sur le Pi :
   - Crée/active `venv` dans `${DEPLOY_PATH}/venv` (avec `--system-site-packages`).
   - `pip install -r requirements.txt` (vérifie que `requirements.txt` est présent).
9. Build / upload modèles `liveness` ONNX (si absents) :
   - Si le Pi n'a pas au moins 2 .onnx dans `${DEPLOY_PATH}/models/liveness`, le runner :
     - Installe numpy/onnx/onnxscript, torch (CPU) localement,
     - Clone `Silent-Face-Anti-Spoofing` en /tmp,
     - Lance `scripts/make_liveness_onnx.py` pour générer les modèles,
     - Envoie les fichiers `.onnx` vers `${MODELS_PATH}` sur le Pi via rsync.
   - Si la conversion échoue, le workflow continue (liveness peut rester désactivé).
10. Démarrage automatique de l’application sur le Pi :
    - Crée logs dans `/home/bioattend/.bioattend/`
    - Utilise `nohup` pour lancer `venv/bin/python -m flask --app app run --host 0.0.0.0 --port 5000` en arrière-plan.
    - Stocke PID dans `/home/bioattend/.bioattend/app-<branch_slug>.pid`
    - Teste la santé via `curl http://127.0.0.1:5000/health` pour valider le démarrage.
11. Configuration du mode kiosk (autostart) sur le Pi :
    - Génère un script `bioattend-kiosk.sh` dans `~/.local/bin/` qui attend le service et lance Chromium en mode `--kiosk --app=http://localhost:5000`.
    - Met à jour `~/.config/lxsession/LXDE-pi/autostart` (et `/home/bioattend/.config/labwc/autostart`) pour que Chromium se lance automatiquement en session graphique.
    - Le script gère logs, attente du service et lance Chromium avec options Wayland / kiosk.

### Variables/Environnements passés au job
- `PI_HOST` (depuis secret)
- `PI_USER` = `"bioattend"`
- `TAILSCALE_AUTHKEY` (secret)
- `SSHPASS` = `PI_PASSWORD` (secret)

### Remarques de sécurité / opération
- Le workflow utilise `sshpass` et mot de passe SSH : en production, préférer clé SSH privée dans GitHub Secrets et usage `ssh -i` pour sécurité renforcée.
- Tailscale permet joindre le Pi même derrière NAT — assure-toi de protéger la clé d'auth.
- Le déploiement synchronise tout le repo vers le Pi (exclusion list incluse), puis installe requirements. Veiller à ne pas exposer secrets en clair dans le dépôt (jamais committer `.env`).
- Lancer `migrations` / opérations lourdes côté serveur central n'est pas géré ici (ce workflow concerne uniquement l'app front).

---

## Exemples d'utilisation & debugging

- Vérifier la santé :
  ```bash
  curl http://<pi-host>:5000/health
  ```

- Tester snapshot :
  ```bash
  curl -fsS http://<pi-host>:5000/snapshot > last.jpg
  ```

- Lancer un diagnostic caméra :
  ```bash
  curl -X POST http://<pi-host>:5000/diagnostics/camera
  ```

- Appeler la route pointage (simuler) : normalement l’UI appelle `/pointage` en POST sans corps (le serveur effectue capture localement). On peut appeler `/diagnostics/identify` pour simuler capture+identification et obtenir plus d'informations.

- Voir config publique (utile pour vérifier que les variables d'env sur le Pi sont correctement prises) :
  ```bash
  curl http://<pi-host>:5000/diagnostics/config
  ```

---

## Conseils d'exploitation & sécurité

- Ne jamais stocker `.env` dans le repo.
- Préférer l'authentification par clé SSH plutôt que mot de passe pour le déploiement.
- Protéger `API_TOKEN` utilisé entre le Pi et le serveur central. Dans le serveur central, utiliser une clé dédiée par Pi et prévoir rotation.
- Sur Pi, exécuter dans un utilisateur non-root (ici `bioattend`).
- Mettre en place logging central (logs écrits dans `/home/bioattend/.bioattend` par le workflow).
- Superviser l'état du service (systemd, supervisord ou vérifier périodiquement `/health`).

---

## Annexes

- Fichier de configuration principal : `src/bioattend_front/config.py`
- Client API : `src/bioattend_front/api_client.py`
- Logic capture/mobile : `src/bioattend_front/camera.py`
- Détection face : `src/bioattend_front/face.py`
- Embedding : `src/bioattend_front/embedding.py`
- Liveness : `src/bioattend_front/liveness.py`
- Entrypoint : `app.py` (create_app)