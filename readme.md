# IPSSI Mémoire – Détection de Volant de Badminton

Ce projet vise à détecter et suivre le volant de badminton et les joueurs dans des vidéos YouTube à l'aide de modèles de deep learning (YOLO). Il a été réalisé dans le cadre du mémoire IPSSI.

## 🚀 Fonctionnalités

- **Détection en temps réel** : Détection du volant de badminton et des joueurs
- **Interface Streamlit** : Interface web moderne et intuitive
- **Version OpenCV** : Application desktop avec contrôles clavier
- **Streaming YouTube** : Analyse directe des vidéos YouTube
- **Optimisations** : Traitement optimisé pour les performances

## 📁 Structure du projet

```
ipssi_mémoire/
├── app/
│   ├── app.py              # Interface Streamlit
│   ├── ytb/
│   │   └── detection.py    # Version OpenCV
│   └── model/
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

### 4. Ajouter les modèles YOLO
Placez vos modèles dans le dossier `app/model/` :
- `weights_v2.pt` : Modèle pour la détection du volant
- `yolov8n.pt` : Modèle YOLOv8 pour la détection des joueurs

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

### Option 2 : Version OpenCV (Desktop)

Lancez l'application desktop avec OpenCV :

```bash
cd app/ytb
python detection.py
```

Ou avec une URL personnalisée :
```bash
python detection.py --url "https://www.youtube.com/watch?v=VOTRE_VIDEO"
```

**Contrôles clavier :**
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

Dans `app/ytb/detection.py` (OpenCV) :
```python
CONFIDENCE_SHUTTLE = 0.10    # Seuil de confiance pour le volant
CONFIDENCE_PLAYER = 0.40     # Seuil de confiance pour les joueurs
IMG_SIZE = 1080              # Taille d'image pour l'inférence
ADVANCE_SEC = 5              # Secondes d'avance/retour
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

## 🚨 Dépannage

### Problèmes courants

1. **Erreur de modèles manquants**
   ```
   FileNotFoundError: ./model/weights_v2.pt
   ```
   **Solution** : Ajoutez les fichiers de modèles dans `app/model/`

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

# OpenCV
python detection.py
```

### Vidéo personnalisée
```bash
# OpenCV avec URL personnalisée
python detection.py --url "https://www.youtube.com/watch?v=VOTRE_VIDEO"
```

## 📝 Notes techniques

- **Optimisations** : Backend CUDA automatique si disponible
- **Mémoire** : Cache des modèles pour de meilleures performances
- **Réseau** : Streaming direct sans téléchargement
- **Interface** : Thème sombre moderne avec animations

---

**Projet réalisé dans le cadre du mémoire IPSSI**  
*Détection de volant de badminton avec YOLO et deep learning*
