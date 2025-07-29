import cv2
import json
import time
import numpy as np
import torch
from datetime import datetime
from ultralytics import YOLO
import yt_dlp

# === CONFIGURATION ===
YOUTUBE_URL = "https://www.youtube.com/watch?v=PQmBTcnz998&ab_channel=BadmintonHam"
MODEL_PATH = "yolov8n.pt"
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
OUTPUT_JSON = "../output/mp4_result/detections.json"
IMG_SIZE = 640
TARGET_FPS = 40
DETECT_EVERY = 1
CONFIDENCE_THRESHOLD = 0.2
MIN_HIST_SCORE = 0.3  # score minimal pour valider un joueur

CLS_PERSON = 0
CLS_SHUTTLE = 32
CLS_BALL = 37

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH).to(device)

print(f"✓ Modèle YOLO chargé sur {device}")

# === YOUTUBE STREAM URL ===
def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        'quiet': True,
        'format': 'bestvideo[height>=720][ext=mp4]',
        'noplaylist': True,
        'skip_download': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

# === HISTOGRAMME COULEUR ===
def get_hist(image, bbox):
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def compare_hist(h1, h2):
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

# === MOUSE CALLBACK ===
clicks = []
def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
        clicks.append((x, y))
        print(f"🖱️ Clic joueur {len(clicks)} : {x}, {y}")

# === FONCTION DE SÉLECTION DES JOUEURS ===
def relabel_players(frame):
    global clicks, player_boxes, player_histograms
    clicks = []
    people = []

    detections = model(frame, imgsz=IMG_SIZE, device=device, conf=CONFIDENCE_THRESHOLD)[0].boxes
    for b in detections:
        if int(b.cls[0]) == CLS_PERSON:
            people.append(b.xyxy[0].cpu().numpy().astype(int))

    preview = frame.copy()
    for box in people:
        cv2.rectangle(preview, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
    cv2.putText(preview, "Cliquez Joueur 1 puis Joueur 2",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Selection joueurs", preview)
    cv2.setMouseCallback("Selection joueurs", mouse_cb)
    while len(clicks) < 2:
        cv2.waitKey(1)
    cv2.destroyWindow("Selection joueurs")

    player_boxes = [None, None]
    player_histograms = [None, None]
    for i, (x, y) in enumerate(clicks):
        best, dmin = None, 1e9
        for box in people:
            cx, cy = (box[0]+box[2])//2, (box[1]+box[3])//2
            d = np.hypot(x - cx, y - cy)
            if d < dmin:
                best, dmin = box, d
        player_boxes[i] = best
        player_histograms[i] = get_hist(frame, best)

# === INITIALISATION ===
print("🔗 Chargement du flux YouTube...")
stream_url = get_youtube_stream_url(YOUTUBE_URL)
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    raise RuntimeError("❌ Impossible d’ouvrir le flux vidéo")

fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = int(1000 / fps) if fps > 0 else 1
ms_per_frame = 1000 / TARGET_FPS
prev_time_ms = cv2.getTickCount() / cv2.getTickFrequency() * 1000

# Première frame + sélection des joueurs
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("❌ Lecture de la première frame impossible")
first_frame = cv2.resize(first_frame, (FRAME_WIDTH, FRAME_HEIGHT))
relabel_players(first_frame)

# === MAIN LOOP ===
frame_id = 0
all_detections = []

print("🚀 Début du tracking YOLO...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    result = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=IMG_SIZE,
        device=device,
        conf=CONFIDENCE_THRESHOLD,
        stream=True,
        verbose=False
    )
    boxes = list(result)[0].boxes

    best_players = [None, None]
    best_scores = [-1, -1]
    shuttle_box = None

    for b in boxes:
        cls = int(b.cls[0])
        xyxy = b.xyxy[0].cpu().numpy().astype(int)

        if cls == CLS_PERSON:
            hist = get_hist(frame, xyxy)
            for i in range(2):
                score = compare_hist(hist, player_histograms[i])
                if score > best_scores[i]:
                    best_scores[i], best_players[i] = score, xyxy

        elif cls == CLS_SHUTTLE:
            shuttle_box = xyxy

        # Enregistrement JSON
        detection = {
            "frame_id": frame_id,
            "timestamp": datetime.now().isoformat(),
            "class_id": cls,
            "confidence": float(b.conf[0]),
            "bbox": [float(x) for x in xyxy.tolist()]
        }
        all_detections.append(detection)

    # === AFFICHAGE ===
    colors = [(0, 255, 0), (255, 0, 0)]
    for i, box in enumerate(best_players):
        if box is not None and best_scores[i] > MIN_HIST_SCORE:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), colors[i], 2)
            cv2.putText(frame, f"Joueur {i+1}", (box[0], box[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
        else:
            cv2.putText(frame, f"Joueur {i+1} ?", (50, 50 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if shuttle_box is not None:
        cv2.rectangle(frame, (shuttle_box[0], shuttle_box[1]),
                      (shuttle_box[2], shuttle_box[3]), (0, 0, 255), 2)
        cv2.putText(frame, "Volant", (shuttle_box[0], shuttle_box[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Badminton Tracker (YouTube)", frame)

    key = cv2.waitKey(wait_time)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('r'):
        print("🔁 Re-sélection des joueurs...")
        relabel_players(frame)

    now_ms = cv2.getTickCount() / cv2.getTickFrequency() * 1000
    delay = max(1, int(ms_per_frame - (now_ms - prev_time_ms)))
    prev_time_ms = now_ms
    frame_id += 1

cap.release()
cv2.destroyAllWindows()

# === SAVE DETECTIONS ===
with open(OUTPUT_JSON, "w") as f:
    json.dump(all_detections, f, indent=4)

print(f"✅ Analyse terminée. Résultats sauvegardés dans {OUTPUT_JSON}")
