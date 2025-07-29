# IPSSI Mémoire – Détection de Volant de Badminton

Ce projet vise à détecter et suivre le volant de badminton dans des vidéos à l'aide de modèles de deep learning (YOLO, etc.). Il a été réalisé dans le cadre du mémoire IPSSI.

## Structure du projet
- `app/` : Scripts Python pour l'extraction de frames, la prédiction, l'API, etc.
- `data/` : Jeux de données, images, vidéos et annotations.
- `output/` : Résultats des détections et images extraites.
- `requirements.txt` : Dépendances Python du projet.

## Installation rapide
1. Clone le dépôt :
   ```sh
   git clone https://github.com/TON-UTILISATEUR/ipssi_memoire.git
   cd ipssi_memoire
   ```
2. Installe les dépendances :
   ```sh
   pip install -r requirements.txt
   ```

## Utilisation
- Lance les scripts dans `app/` pour extraire des frames, faire des prédictions ou utiliser l'API.
- Les résultats sont enregistrés dans le dossier `output/`.

## Remarques
- Les poids des modèles (`*.pt`) ne sont pas inclus dans le dépôt.
- Pense à configurer un fichier `.env` si besoin.

---
Projet réalisé par [Ton Nom] dans le cadre du mémoire IPSSI.
