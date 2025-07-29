#!/usr/bin/env python
"""
yolo_video_tracker.py
---------------------
Détection + tracking (joueurs + volant) sur une vidéo de badminton, avec :
    • Sélection de 4 coins du terrain au démarrage → homographie pixels → mètres
    • Sélection interactive des 2 joueurs
    • Suivi BoT‑SORT (Re‑ID) + histogramme couleur pour plus de robustesse
    • Compatible CPU / GPU automatique
    • Vise ~15 FPS (détection YOLOv8n toutes les N images, update tracker entre‑temps)

Dépendances : ultralytics >= 8.1, opencv-python, torch, numpy
Si GPU (CUDA 12.1) :
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ------------------------ configuration ------------------------
VIDEO_PATH   = "../data/badminton_video_test.mp4"  # chemin vidéo
MODEL_PATH   = "yolov8n.pt"                        # YOLO nano (rapide)
IMG_SIZE     = 640                                  # redimension YOLO
DETECT_EVERY = 1                                    # détection toutes les N frames
TARGET_FPS   = 15

# Classes COCO utiles
CLS_PERSON  = 0
CLS_SHUTTLE = 32  # sports racket ≈ volant

# Dimensions du terrain simple badminton (m)
COURT_W = 5.18
COURT_H = 13.40

BOTSORT_CFG = "cfg/trackers/botsort.yaml"

# ------------------------- initialisation ----------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = YOLO(MODEL_PATH).to(device)
print(f"✓ Modèle chargé sur {device}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Impossible d’ouvrir {VIDEO_PATH}")

# ---------- variables globales ----------
corner_clicks     = []  # 4 coins du court
player_clicks     = []  # clic joueur 1 & 2
player_boxes      = [None, None]
player_hists      = [None, None]
player_ids        = {}  # {0: id_j1, 1: id_j2}
id_to_hist        = {}
H                 = None  # homographie pixel→m

frame_id     = 0
ms_per_frame = 1000 / TARGET_FPS
prev_time_ms = cv2.getTickCount() / cv2.getTickFrequency() * 1000

# ----------------- fonctions utilitaires -----------------------

def mouse_cb(event, x, y, flags, param):
    phase = param["phase"]
    if event == cv2.EVENT_LBUTTONDOWN:
        if phase == "corners" and len(corner_clicks) < 4:
            corner_clicks.append((x, y))
            print(f"Coin {len(corner_clicks)}/4 : {x}, {y}")
        elif phase == "players" and len(player_clicks) < 2:
            player_clicks.append((x, y))
            print(f"Clic Joueur {len(player_clicks)} : {x}, {y}")

def get_hist(image, bbox):
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def compare_hist(h1, h2):
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

def px_to_m(pts):
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    m   = cv2.perspectiveTransform(pts, H)
    return m.reshape(-1, 2)

def bbox_center_m(box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return px_to_m([(cx, cy)])[0]

# ------------------ 1. calibration court -----------------------
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Lecture première frame impossible")

cal_view = first_frame.copy()
cv2.putText(cal_view, "Cliquez 4 coins du court (sens horaire)", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Calibration", cal_view)
cv2.setMouseCallback("Calibration", mouse_cb, {"phase": "corners"})
while len(corner_clicks) < 4:
    cv2.waitKey(1)
cv2.destroyWindow("Calibration")

src_px = np.float32(corner_clicks)
dst_m  = np.float32([(0, 0), (0, COURT_H), (COURT_W, COURT_H), (COURT_W, 0)])
H, _ = cv2.findHomography(src_px, dst_m)
print("✓ Homographie calculée")

# --------------- 2. détection initiale joueurs -----------------
people = []
for b in model(first_frame, imgsz=IMG_SIZE, device=device)[0].boxes:
    if int(b.cls[0]) == CLS_PERSON:
        people.append(b.xyxy[0].cpu().numpy().astype(int))

prev = first_frame.copy()
for box in people:
    cv2.rectangle(prev, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
cv2.putText(prev, "Cliquez Joueur 1 puis Joueur 2", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Selection joueurs", prev)
cv2.setMouseCallback("Selection joueurs", mouse_cb, {"phase": "players"})
while len(player_clicks) < 2:
    cv2.waitKey(1)
cv2.destroyWindow("Selection joueurs")

for i, (x, y) in enumerate(player_clicks):
    best, dmin = None, 1e9
    for box in people:
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        d = np.hypot(x - cx, y - cy)
        if d < dmin:
            best, dmin = box, d
    player_boxes[i] = best
    player_hists[i] = get_hist(first_frame, best)

print("✓ Joueurs sélectionnés")

# ---------------------- 3. boucle vidéo ------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # détection yolo selon DETECT_EVERY
    if frame_id % DETECT_EVERY == 0:
        result = model.track(frame, imgsz=IMG_SIZE, device=device,
                             persist=True, tracker=BOTSORT_CFG)
        boxes = result[0].boxes
    else:
        model.track(frame, persist=True, stream=True)  # maj interne tracker
        boxes = None  # inutilisé ici

    best_players = [None, None]
    best_scores  = [-1, -1]
    shuttle_box  = None

    if boxes is not None:
        for b in boxes:
            cls  = int(b.cls[0])
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            track_id = int(b.id[0]) if b.id is not None else -1

            if cls == CLS_PERSON:
                hist = get_hist(frame, xyxy)
                if track_id not in id_to_hist:
                    id_to_hist[track_id] = hist.copy()

                # attribution joueur 0/1 (mix ID + couleur)
                for i in range(2):
                    score = 0.0
                    if track_id == player_ids.get(i):
                        score = 2.0  # priorité ID correspondant
                    else:
                        score = compare_hist(hist, player_hists[i])
                    if score > best_scores[i]:
                        best_scores[i]  = score
                        best_players[i] = xyxy
                        if track_id != -1:
                            player_ids[i] = track_id

            elif cls == CLS_SHUTTLE:
                shuttle_box = xyxy

    # ------------ dessin ------------
    colors = [(0, 255, 0), (255, 0, 0)]
    for i, box in enumerate(best_players):
        if box is not None:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), colors[i], 2)
            cv2.putText(frame, f"Joueur {i+1}", (box[0], box[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)

    if shuttle_box is not None:
        cv2.rectangle(frame, (shuttle_box[0], shuttle_box[1]),
                      (shuttle_box[2], shuttle_box[3]), (0, 0, 255), 2)
        cv2.putText(frame, "Volant", (shuttle_box[0], shuttle_box[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ----------- affichage + régulation FPS -----------
    cv2.imshow("Badminton Tracker", frame)
    now_ms = cv2.getTickCount() / cv2.getTickFrequency() * 1000
    delay  = max(1, int(ms_per_frame - (now_ms - prev_time_ms)))
    prev_time_ms = now_ms
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

    frame_id += 1

# --------------------- cleanup ---------------------------
cap.release()
cv2.destroyAllWindows()

print("✓ Terminé")
