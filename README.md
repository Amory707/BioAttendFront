# BioAttendFront

Partie embarquée du projet **BioAttend**, exécutée sur Raspberry Pi.

Ce dépôt implémente un client léger chargé de piloter le matériel, capturer une image exploitable, préparer les données pour la reconnaissance faciale et communiquer avec l'API distante. **Toute la logique métier reste côté serveur.**

---

## Vue d'ensemble du projet

Le système de pointage fonctionne selon l'architecture suivante :

```
Raspberry Pi (ce dépôt)  →  API distante (Django + IA)
```

Le Raspberry Pi est responsable de :

1. Détecter une présence via un capteur PIR
2. Activer la caméra uniquement quand nécessaire
3. Détecter et extraire le visage
4. Effectuer la liveness detection (anti-spoofing)
5. Générer un embedding via InsightFace
6. Envoyer cet embedding à l'API distante
7. Afficher un retour utilisateur local (nom, heure, type de pointage)

---

## Structure du dépôt

```
app.py                          # Point d'entrée Flask
requirements.txt                # Dépendances Python
.env                            # Configuration locale (à créer, non versionné)
src/
  bioattend_front/
    __init__.py                 # Fabrique de l'application Flask
    config.py                   # Chargement de la config depuis .env
    camera.py                   # Capture de frame (Picamera2 + OpenCV)
    face.py                     # Détection et crop du visage
    embedding.py                # Génération de l'embedding (InsightFace)
    api_client.py               # Appel HTTP vers l'API distante
    main.py                     # Routes Flask (interface + diagnostics)
```

---

## Prérequis

- Python 3.10+
- Sur Raspberry Pi : `libcamera` et `picamera2` installés via le système
- Sur PC de développement : une webcam USB suffit pour tester

---

## Installation

```bash
git clone https://github.com/Nde-Code/BioAttendFront.git
cd BioAttendFront

# Créer un environnement virtuel (recommandé)
python3 -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

> **Note Raspberry Pi :** `picamera2` s'installe via apt et non pip :
> ```bash
> sudo apt install python3-picamera2
> ```

---

## Configuration — fichier `.env`

Créer un fichier `.env` à la racine du projet. Ce fichier **ne doit jamais être versionné** (il est dans `.gitignore`).

```env
# ── Mode debug Flask ──────────────────────────────────────────────
DEBUG=false
KIOSK_MODE=true          # Active l'UI borne plein écran
POINTAGE_TRIGGER_MODE=space   # "space" (clavier/touch) ou "pir" (declenchement capteur presence)

# ── Caméra ───────────────────────────────────────────────────────
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_MIRROR=true       # Inverse horizontalement l'image (effet miroir)
CAMERA_SWAP_RB=false     # Mettre a true si les couleurs sont inversees (peau bleue, jaunes, etc.)
CAMERA_JPEG_QUALITY=68   # 40-95: plus bas = plus fluide, plus haut = meilleure qualite
CAMERA_DEVICE=0           # Index du device vidéo (ex: 0, 1…)
CAMERA_SOURCE=auto        # "picamera2" sur Raspberry Pi, "opencv" sur PC, "auto" = détection automatique
CAMERA_BACKEND=auto       # Backend OpenCV : "v4l2", "any", "auto"
CAMERA_WARMUP_MS=800      # Temps de chauffe caméra en millisecondes
CAMERA_READ_ATTEMPTS=10   # Nombre de tentatives de lecture de frame

# ── InsightFace (génération d'embeddings) ─────────────────────────
INSIGHTFACE_MODEL_NAME=buffalo_l
INSIGHTFACE_DET_WIDTH=640
INSIGHTFACE_DET_HEIGHT=640

# ── API distante ──────────────────────────────────────────────────
SERVER_URL=https://bioattend.138.199.195.144.sslip.io/api/face/identify/
EVENTS_URL=https://bioattend.138.199.195.144.sslip.io/api/front/events/
API_TOKEN=votre_token_ici
API_TIMEOUT_SECONDS=8
DEVICE_NAME=bioattend-pi

# ── Liveness (anti-spoofing) ─────────────────────────────────────
LIVENESS_ENABLED=true
LIVENESS_MODEL_DIR=models/liveness
LIVENESS_THRESHOLD=0.80
LIVENESS_LIVE_CLASS_IDX=0
```

### Variables importantes

| Variable | Description | Valeur conseillée |
|---|---|---|
| `CAMERA_SOURCE` | Source de capture | `picamera2` sur Raspberry Pi, `opencv` sur PC |
| `CAMERA_MIRROR` | Active l'effet miroir horizontal | `true` pour cadrage type selfie, `false` pour image réelle |
| `CAMERA_SWAP_RB` | Inverse les canaux rouge/bleu si les couleurs paraissent fausses | `true` uniquement si l'image a des couleurs inversees |
| `CAMERA_JPEG_QUALITY` | Qualité JPEG du flux live | `60-70` sur Raspberry Pi pour plus de fluidité |
| `API_TOKEN` | Token d'authentification de l'API | Récupérer auprès du responsable backend |
| `SERVER_URL` | URL de l'endpoint d'identification | Ne pas modifier sauf changement de déploiement |
| `EVENTS_URL` | URL de journalisation des tentatives front | Laisser vide si le backend ne l'expose pas encore |
| `DEVICE_NAME` | Nom logique de la pointeuse | `bioattend-pi` ou un identifiant unique |
| `DEBUG` | Active le mode debug Flask | `false` en production |
| `KIOSK_MODE` | Active le comportement borne (plein écran auto) | `true` sur Raspberry Pi |
| `POINTAGE_TRIGGER_MODE` | Source de déclenchement du pointage | `space` ou `pir` |

---

## Lancer le serveur

```bash
# Activer l'environnement virtuel si ce n'est pas déjà fait
source .venv/bin/activate

# Lancer Flask
python app.py
```

Le serveur démarre sur `http://0.0.0.0:5000`.

- Sur Raspberry Pi : accessible depuis un navigateur sur le même réseau à `http://<ip-du-raspberry>:5000`
- Sur PC : ouvrir `http://localhost:5000`

### Mode kiosk Raspberry Pi (recommandé)

Pour un rendu station de pointage sans barre navigateur, lancez Chromium en mode kiosk :

```bash
chromium-browser --kiosk --app=http://localhost:5000
```

Avec `KIOSK_MODE=true`, l'interface masque le bouton "Plein écran" et force le comportement borne.

---

## Interface utilisateur

La page principale (`/`) affiche :

- Le flux de la caméra en temps réel avec un ovale de cadrage
- Un bouton **Pointer** (ou touche `Espace`) pour déclencher l'identification
- Un retour visuel : **vert** si reconnu (nom + heure + type de pointage), **rouge** sinon

---

## Routes de diagnostic

Ces routes permettent de tester chaque brique du pipeline de façon isolée. Utiles pour déboguer.

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/health` | Vérifie que le service tourne |
| `GET` | `/config` | Affiche la configuration active (token masqué) |
| `GET` | `/diagnostics/camera` | Teste la capture d'une frame |
| `GET` | `/diagnostics/face` | Teste la détection + crop du visage |
| `GET` | `/diagnostics/embedding` | Teste la génération du vecteur |
| `GET/POST` | `/diagnostics/identify` | Teste le pipeline complet jusqu'à l'appel API |
| `GET` | `/snapshot` | Retourne une frame JPEG brute (utilisé par l'UI) |
| `POST` | `/pointage` | Déclenche un pointage complet |

---

## Philosophie de développement

Le projet est bâti de façon **incrémentale** : chaque brique est validée séparément avant d'être intégrée au pipeline. Sur Raspberry Pi, si tout est assemblé d'un coup, il devient très difficile de savoir d'où vient une panne (matériel, caméra, modèle IA, réseau…).

**Règles à respecter :**

- Le Raspberry Pi est un **client léger** : aucune logique métier locale
- **Aucune image ne doit être stockée** sur le disque (respect RGPD)
- Le pipeline d'identification est **séquentiel et strict** : PIR → caméra → visage → liveness → embedding → API → affichage
- Avancer par petites étapes testables via les routes `/diagnostics/*`

---

## Choix techniques

### Picamera2 pour la capture

Sur Raspberry Pi avec caméra CSI, OpenCV seul n'est pas fiable (la caméra est détectée mais aucune frame exploitable n'est retournée). **Picamera2** s'appuie sur la pile libcamera native du Raspberry et permet une capture fiable. OpenCV prend ensuite le relais for le traitement.

Sur PC de développement, `CAMERA_SOURCE=opencv` utilise directement OpenCV avec la webcam.

### InsightFace pour les embeddings

InsightFace transforme le visage en un vecteur numérique (embedding). C'est ce vecteur, et non l'image, qui est envoyé à l'API — plus léger et plus respectueux de la vie privée.

---

## Déploiement automatique

Une GitHub Action déploie automatiquement le dépôt sur le Raspberry Pi cible à chaque push sur la branche `main`.

Le workflow :

1. recupere le code de la branche poussee
2. rejoint le Raspberry via Tailscale et SSH
3. synchronise les fichiers dans un dossier dedie a la branche
4. cree ou reutilise un environnement virtuel Python sur le Pi
5. installe les dependances depuis requirements.txt

Ce choix permet :

1. de tester rapidement sur le materiel reel
2. de travailler branche par branche sans casser un etat stable
3. de garder un cycle simple : modification, push, deploiement, verification sur Pi

## Etat actuel du depot

Les briques minimales en place sont :

1. chargement de configuration depuis .env
2. application Flask minimale
3. endpoint de diagnostic camera
4. fallback de capture Picamera2 pour Raspberry Pi
5. endpoint de diagnostic detection/crop visage
6. endpoint de diagnostic embedding InsightFace
7. endpoint de diagnostic identify vers l'API distante

Fichiers principaux :

1. app.py
2. src/bioattend_front/main.py
3. src/bioattend_front/config.py
4. src/bioattend_front/camera.py

## Etapes suivantes prevues

L'ordre retenu pour avancer proprement est :

1. validation du capteur PIR
2. declenchement conditionnel de la capture camera sur presence
3. detection et extraction du visage
4. liveness detection avec le modele impose par la partie IA
5. generation de l'embedding
6. appel a l'API distante
7. interface utilisateur locale

Cet ordre permet de respecter le pipeline cible tout en gardant des points de test simples sur le Raspberry.

## Lancement local de l'application

Depuis le depot :

```bash
python -m flask --app app run --host 0.0.0.0 --port 5000
```

Routes utiles :

1. GET /health
2. GET /diagnostics/config
3. GET /diagnostics/camera
4. GET /diagnostics/face
5. GET /diagnostics/embedding
6. GET /diagnostics/identify

## Dependances

Les dependances Python du projet sont listees dans requirements.txt.

Note importante pour le Raspberry Pi :

1. certaines briques materielles peuvent dependre de paquets systeme presents sur Raspberry Pi OS
2. le venv de deploiement est cree avec acces aux paquets systeme
3. cela est particulierement utile pour la pile camera Raspberry

## Ce que le projet ne fait pas encore

A ce stade, le depot ne fait pas encore :

1. la lecture du PIR
2. la liveness
3. l'affichage local sur l'ecran de la pointeuse

Ce n'est pas un oubli. C'est un choix de sequence pour valider d'abord la base materielle et logicielle.