"""
yolo_video_tracker.py
---------------------
Détection + tracking (joueurs, volant, balle) sur une vidéo badminton.

• Compatible CPU **et** GPU : choix auto de l’appareil.
• Vise ~30 FPS : détection YOLOv8n toutes les N images, tracking intermédiaire.
• Quitter : touche « q ».

Dépendances : ultralytics >= 8.1, opencv-python, torch, numpy
GPU : installe torch + CUDA 11.8 ou 12.1 :
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
"""

import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO

# ------------------------ configuration ------------------------
VIDEO_PATH      = "../data/badminton_video_test.mp4"  # chemin vidéo
MODEL_PATH      = "yolov8n.pt"                # version nano (rapide)
DETECT_EVERY    = 1                           # 1 détection toutes les N frames
TARGET_FPS      = 15
IMG_SIZE        = 640                         # redimension pour YOLO
# classes COCO utiles
CLS_PERSON      = 0
CLS_SHUTTLE     = 32                          # sports racket ≈ volant
CLS_BALL        = 37                          # sports ball
# ----------------------------------------------------------------

device   = "cuda" if torch.cuda.is_available() else "cpu"
model    = YOLO(MODEL_PATH).to(device)
print(f"✓ Modèle chargé sur {device}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Impossible d’ouvrir {VIDEO_PATH}")

# ---------- variables du script ----------
frame_id           = 0
prev_time_ms       = cv2.getTickCount() / cv2.getTickFrequency() * 1000
ms_per_frame       = 1000 / TARGET_FPS
clicks             = []          # pour choisir manuellement les 2 joueurs
player_histograms  = [None, None]
player_boxes       = [None, None]

def get_hist(image, bbox):
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def compare_hist(h1, h2):
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
        clicks.append((x, y))
        print(f"Clic pour joueur {len(clicks)} : {x}, {y}")

# ---------- détection initiale pour sélectionner les joueurs ----------
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Lecture de la première frame impossible")

people = []
for b in model(first_frame, imgsz=IMG_SIZE, device=device)[0].boxes:
    if int(b.cls[0]) == CLS_PERSON:
        people.append(b.xyxy[0].cpu().numpy().astype(int))

preview = first_frame.copy()
for box in people:
    cv2.rectangle(preview, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
cv2.putText(preview, "Cliquez Joueur 1 puis Joueur 2",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Selection joueurs", preview)
cv2.setMouseCallback("Selection joueurs", mouse_cb)
while len(clicks) < 2:
    cv2.waitKey(1)
cv2.destroyWindow("Selection joueurs")

# associe chaque clic au joueur le plus proche
for i, (x, y) in enumerate(clicks):
    best, dmin = None, 1e9
    for box in people:
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        d = np.hypot(x - cx, y - cy)
        if d < dmin:
            best, dmin = box, d
    player_boxes[i]      = best
    player_histograms[i] = get_hist(first_frame, best)

# ---------------------- boucle vidéo ------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # détection yolo toutes les DETECT_EVERY frames
    if frame_id % DETECT_EVERY == 0:
        result  = model.track(frame, persist=True,
                              tracker="bytetrack.yaml",
                              imgsz=IMG_SIZE, device=device,
                              verbose=False)
        boxes   = result[0].boxes
    else:
        model.track(frame, persist=True, stream=True, verbose=False)  # MAJ tracker
        boxes = None  # inutilisé ici

    best_players = [None, None]
    best_scores  = [-1, -1]
    shuttle_box  = None
    ball_box     = None

    if boxes is not None:
        for b in boxes:
            cls  = int(b.cls[0])
            xyxy = b.xyxy[0].cpu().numpy().astype(int)

            if cls == CLS_PERSON:
                hist = get_hist(frame, xyxy)
                for i in range(2):
                    s = compare_hist(hist, player_histograms[i])
                    if s > best_scores[i]:
                        best_scores[i], best_players[i] = s, xyxy

            elif cls == CLS_SHUTTLE:
                shuttle_box = xyxy
            elif cls == CLS_BALL:
                ball_box = xyxy

    # ----------- dessin -----------
    colors = [(0, 255, 0), (255, 0, 0)]  # joueur1 vert, joueur2 bleu
    for i, box in enumerate(best_players):
        if box is not None:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), colors[i], 2)
            cv2.putText(frame, f"Joueur {i+1}", (box[0], box[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)

    if shuttle_box is not None:
        cv2.rectangle(frame, (shuttle_box[0], shuttle_box[1]),
                      (shuttle_box[2], shuttle_box[3]), (0, 0, 255), 2)
        cv2.putText(frame, "Volant", (shuttle_box[0], shuttle_box[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ----------- affichage + régulation FPS -----------
    cv2.imshow("Badminton Tracker", frame)
    now_ms  = cv2.getTickCount() / cv2.getTickFrequency() * 1000
    delay   = max(1, int(ms_per_frame - (now_ms - prev_time_ms)))
    prev_time_ms = now_ms
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
