from ultralytics import YOLO
import cv2

# Charger le modèle (remplace par ton chemin local)
Fmodel = YOLO("weights.pt")

# Prédiction sur une image
results = Fmodel("../../output/frame_000711.jpg", conf=0.5)

# Afficher les résultats
results[0].show()          # affiche avec bounding boxes
results[0].save("pred.jpg")  # sauvegarde l’image avec détection
