# IPSSI Mémoire – Détection de Volant de Badminton

Ce projet vise à détecter et suivre le volant de badminton et les joueurs dans des vidéos YouTube à l'aide de modèles de deep learning (YOLO). Il a été réalisé dans le cadre du mémoire IPSSI.

## 🚀 Fonctionnalités

- **Détection en temps réel** : Détection du volant de badminton et des joueurs
- **Interface Streamlit** : Interface web moderne et intuitive
- **Versions OpenCV multiples** : Applications desktop avec contrôles clavier
- **Streaming YouTube** : Analyse directe des vidéos YouTube
- **Suivi de trajectoire** : Visualisation du parcours du volant
- **Système de blacklist** : Élimination des fausses détections statiques
- **Optimisations** : Traitement optimisé pour les performances

## 📁 Structure du projet

```
ipssi_mémoire/
├── app/
│   ├── app.py              # Interface Streamlit
│   ├── ytb/
│   │   ├── detection.py    # Version OpenCV de base
│   │   ├── detection_v2.py # Version avec suivi de trajectoire
│   │   └── detection_v3.py # Version avec blacklist et optimisations
│   └── model/
│   │   └── weights_v1.pt   # modele fine tuner version 1
│   │   └── weights_v2.pt   # modele fine tuner version 2
│   │   └── yolov8n.pt      # modele yolo
├── data/                   # Jeux de données
├── output/                 # Résultats des détections
├── requirements.txt        # Dépendances Python
└── readme.md              # Ce fichier
```

## 🛠️ Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/TON-UTILISATEUR/ipssi_memoire.git
cd ipssi_memoire
```

### 2. Créer un environnement virtuel (recommandé)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## 🎯 Utilisation

### Option 1 : Interface Streamlit (Recommandée)

Lancez l'interface web moderne avec Streamlit :

```bash
cd app
streamlit run app.py
```

**Fonctionnalités :**
- Interface web intuitive
- Analyse en temps réel
- URL YouTube personnalisable
- Vidéo de démo intégrée
- Affichage optimisé

**Contrôles :**
- Entrez une URL YouTube ou utilisez la vidéo par défaut
- Cliquez sur "Démarrer l'analyse" pour commencer
- L'analyse se lance automatiquement

### Option 2 : Version OpenCV - Base (detection.py)

Lancez l'application desktop de base :

```bash
cd app/ytb
python detection.py
```

Ou avec une URL personnalisée :
```bash
python detection.py --url "https://www.youtube.com/watch?v=VOTRE_VIDEO"
```

### Option 3 : Version OpenCV - Suivi de trajectoire (detection_v2.py)

Version améliorée avec suivi du parcours du volant :

```bash
cd app/ytb
python detection_v2.py
```

**Nouvelles fonctionnalités :**
- Visualisation de la trajectoire du volant
- Filtrage des détections statiques
- Amélioration de la précision

### Option 4 : Version OpenCV - Blacklist (detection_v3.py) ⭐ RECOMMANDÉE

Version optimisée avec système de blacklist :

```bash
cd app/ytb
python detection_v3.py
```

**Fonctionnalités avancées :**
- Système de blacklist pour éliminer les fausses détections
- Suivi de trajectoire amélioré
- Détection d'immobilité intelligente
- Visualisation des zones blacklistées

**Contrôles clavier (toutes versions OpenCV) :**
- **ESPACE** : Pause/Play
- **Flèche droite** ou **L** : Avancer de 5 secondes
- **Flèche gauche** ou **J** : Reculer de 5 secondes
- **Q** : Quitter

## ⚙️ Configuration

### Paramètres modifiables

Dans `app/app.py` (Streamlit) :
```python
CONFIDENCE_SHUTTLE = 0.15    # Seuil de confiance pour le volant
CONFIDENCE_PLAYER = 0.40      # Seuil de confiance pour les joueurs
IMG_SIZE = 416               # Taille d'image pour l'inférence
SKIP_FRAMES = 3              # Traitement toutes les N frames
```

Dans `app/ytb/detection_v3.py` (Version recommandée) :
```python
CONFIDENCE_SHUTTLE = 0.10    # Seuil de confiance pour le volant
CONFIDENCE_PLAYER = 0.40     # Seuil de confiance pour les joueurs
IMG_SIZE = 1080              # Taille d'image pour l'inférence
ADVANCE_SEC = 5              # Secondes d'avance/retour
IMMOBILE_THRESHOLD = 10      # Pixels de tolérance pour l'immobilité
IMMOBILE_DURATION = 3.0      # Secondes avant blacklist
BLACKLIST_RADIUS = 25        # Rayon de la zone blacklistée
MAX_TRAJ_DURATION = 2.0      # Durée d'affichage de la trajectoire
```

## 🔧 Dépendances

Les principales dépendances sont :
- `torch` : PyTorch pour l'inférence
- `ultralytics` : Framework YOLO
- `opencv-python` : Traitement vidéo
- `streamlit` : Interface web
- `yt-dlp` : Téléchargement YouTube
- `numpy` : Calculs numériques

## 📊 Résultats

Le système détecte :
- **Volant de badminton** : Boîtes vertes avec score de confiance
- **Joueurs** : Boîtes bleues avec score de confiance
- **Zone de jeu** : Masque automatique du terrain
- **Trajectoire** : Ligne verte montrant le parcours du volant (v2/v3)
- **Zones blacklistées** : Cercles rouges pour les zones ignorées (v3)

## 🚨 Dépannage

### Problèmes courants

1. **Erreur de modèles manquants**
   ```
   FileNotFoundError: ./model/weights_v2.pt
   ```
   **Solution** : Les modèles sont inclus dans le projet, vérifiez le chemin

2. **Erreur de dépendances**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Problème de performance**
   - Réduisez `IMG_SIZE` dans les paramètres
   - Augmentez `SKIP_FRAMES` pour traiter moins de frames
   - Utilisez un GPU si disponible

4. **Erreur YouTube**
   - Vérifiez la validité de l'URL
   - Assurez-vous que la vidéo est publique
   - Mettez à jour `yt-dlp` : `pip install --upgrade yt-dlp`

## 🎯 Exemples d'utilisation

### Vidéo de démo
```bash
# Streamlit
streamlit run app.py

# OpenCV - Version recommandée
python detection_v3.py

# OpenCV - Avec trajectoire
python detection_v2.py

# OpenCV - Version de base
python detection.py
```

### Vidéo personnalisée
```bash
# OpenCV avec URL personnalisée
python detection_v3.py --url "https://www.youtube.com/watch?v=VOTRE_VIDEO"
```

## 📝 Notes techniques

### Évolutions des versions

**Version 1 (detection.py) :**
- Détection de base du volant et des joueurs
- Interface OpenCV simple

**Version 2 (detection_v2.py) :**
- Ajout du suivi de trajectoire du volant
- Filtrage des détections statiques
- Amélioration de la précision

**Version 3 (detection_v3.py) :**
- Système de blacklist pour éliminer les fausses détections
- Détection d'immobilité intelligente
- Visualisation des zones blacklistées
- Optimisations de performance

### Optimisations
- **Backend CUDA** : Automatique si disponible
- **Cache des modèles** : Pour de meilleures performances
- **Streaming direct** : Sans téléchargement
- **Interface moderne** : Thème sombre avec animations

---

**Projet réalisé dans le cadre du mémoire IPSSI**  
*Détection de volant de badminton avec YOLO et deep learning*
