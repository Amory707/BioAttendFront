# BioAttendFront

Partie embarquee du projet BioAttend, executee sur Raspberry Pi.

Ce depot implemente un client leger charge de piloter le materiel, capturer une image exploitable, preparer les donnees pour la reconnaissance faciale et communiquer avec l'API distante. La logique metier reste cote serveur.

## Objectif

Le Raspberry Pi doit a terme :

1. detecter une presence via un capteur PIR
2. activer la capture camera uniquement quand c'est necessaire
3. detecter et extraire le visage
4. effectuer la liveness detection
5. generer un embedding via InsightFace
6. envoyer cet embedding a l'API distante
7. afficher un retour utilisateur local

Le pipeline final doit rester strict et sequenciel.

## Principes de conception

Les choix actuels suivent ces contraintes :

1. Raspberry Pi = client leger
2. aucune logique metier locale
3. aucune image stockee localement
4. optimisation pour ressources limitees
5. fiabilite avant sophistication

Consequence directe : on avance par petites etapes testables sur le Raspberry au lieu d'assembler tout le pipeline d'un coup.

## Pourquoi une approche incremental

Le projet melange materiel, capture video, traitements IA et communication reseau. Sur Raspberry Pi, si tout est integre en une seule fois, il devient difficile de savoir si une panne vient :

1. du capteur PIR
2. de la camera
3. de la pile logicielle Raspberry
4. d'OpenCV ou du backend video
5. du modele IA
6. de l'API distante

Le depot est donc construit pour valider chaque brique separement avant d'enchainer avec la suivante.

## Choix techniques deja valides

### Flask pour la plateforme locale

Flask est utilise comme couche locale minimale pour exposer des routes de diagnostic et de test.

Pourquoi ce choix :

1. il est leger et simple a lancer sur Raspberry Pi
2. il permet de tester chaque composant via HTTP sans interface graphique immediate
3. il sert de socle pour les diagnostics, puis pour une future interface locale si necessaire

Etat actuel :

1. une route de sante confirme que le service tourne
2. une route de configuration expose les parametres non sensibles charges depuis l'environnement
3. une route de diagnostic camera teste l'acquisition d'une frame en memoire
4. une route de diagnostic visage teste detection + crop en memoire

## Pourquoi Picamera2 pour la capture camera

Le cahier cible OpenCV pour le traitement du flux video, ce qui reste coherent pour les etapes de vision.

En revanche, sur Raspberry Pi avec une camera CSI, l'acquisition directe via OpenCV peut etre instable ou incomplete. C'est exactement ce qui a ete observe pendant les tests :

1. la camera est detectee par la pile Raspberry
2. les devices video existent bien
3. OpenCV ouvre le peripherique mais ne lit aucune frame exploitable

Picamera2 a donc ete introduit comme couche de capture prioritaire sur Raspberry, pour une raison precise : il s'appuie sur la pile camera native actuelle du Raspberry, basee sur libcamera.

Cela signifie :

1. capture fiable de la frame via Picamera2
2. traitement de cette frame ensuite avec OpenCV
3. aucun changement de logique metier, uniquement une meilleure couche d'acces au materiel

Ce choix ne remplace donc pas OpenCV dans le projet. Il se limite a la partie acquisition, la plus sensible au materiel.

## Resultat des tests camera

Le diagnostic camera a permis d'etablir les points suivants :

1. le service Flask demarre correctement sur le Raspberry Pi
2. la camera est bien detectee par la stack Raspberry via libcamera
3. l'acquisition via OpenCV seul n'etait pas fiable dans cette configuration
4. l'acquisition via Picamera2 fonctionne et retourne une frame en memoire en 1280x720

Conclusion : la couche de capture camera est validee.

## Deploiement automatique vers le Raspberry Pi

Une GitHub Action est configuree pour deployer automatiquement le depot sur le Raspberry Pi a chaque push.

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
3. la generation d'embedding
4. l'appel API final de reconnaissance
5. l'affichage local sur l'ecran de la pointeuse

Ce n'est pas un oubli. C'est un choix de sequence pour valider d'abord la base materielle et logicielle.