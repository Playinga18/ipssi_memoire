import cv2
import json
import time
import torch
from datetime import datetime
from ultralytics import YOLO
import yt_dlp

# === CONFIGURATION ===
YOUTUBE_URL = "https://www.youtube.com/watch?v=zYqgJo1L5uM&t=1s&ab_channel=BadmintonHam"
SHUTTLE_MODEL_PATH = "weights.pt"
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
OUTPUT_JSON = "detections_shuttlecock.json"
IMG_SIZE = 640
CONFIDENCE_THRESHOLD = 0.2
TARGET_FPS = 40

# === CHARGEMENT DU MODÈLE ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(SHUTTLE_MODEL_PATH).to(device)
print(f"✅ Modèle chargé sur {device}")
print(f"📦 Classes du modèle : {model.names}")

# Récupération dynamique de l'index de la classe "Shuttlecock"
shuttle_class_index = None
for k, v in model.names.items():
    if v.lower() == "shuttlecock":
        shuttle_class_index = k
        break

if shuttle_class_index is None:
    raise ValueError("❌ Classe 'Shuttlecock' introuvable dans le modèle. Vérifie que ton modèle contient cette classe.")

# === FONCTION STREAM YOUTUBE ===
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

# === LECTURE VIDÉO ===
print("🔗 Connexion au flux YouTube...")
stream_url = get_youtube_stream_url(YOUTUBE_URL)
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    raise RuntimeError("❌ Impossible d’ouvrir le flux vidéo")

fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = int(1000 / fps) if fps > 0 else 1

frame_id = 0
detections = []

print("🎯 Début de la détection du volant...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        device=device,
        stream=True,
        verbose=False
    )

    boxes = list(results)[0].boxes
    for box in boxes:
        cls = int(box.cls[0])
        if cls == shuttle_class_index:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            detections.append({
                "frame_id": frame_id,
                "timestamp": datetime.now().isoformat(),
                "class_id": cls,
                "class_name": model.names[cls],
                "confidence": conf,
                "bbox": [float(x) for x in xyxy.tolist()]
            })

            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
            cv2.putText(frame, f"Shuttlecock ({conf:.2f})", (xyxy[0], xyxy[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Détection Shuttlecock", frame)
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()

# === SAVE JSON ===
with open(OUTPUT_JSON, "w") as f:
    json.dump(detections, f, indent=4)

print(f"✅ Détection terminée. Résultats enregistrés dans {OUTPUT_JSON}")
