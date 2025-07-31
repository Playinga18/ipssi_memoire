import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import yt_dlp
from PIL import Image

# === CONFIGURATION ===
YOUTUBE_URL = "https://www.youtube.com/watch?v=PQmBTcnz998&ab_channel=BadmintonHam"
MODEL_PATH = "yolov8n.pt"
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
IMG_SIZE = 640
CONFIDENCE_THRESHOLD = 0.2
CLS_PERSON = 0
CLS_SHUTTLE = 32

# === UTILS ===
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

def get_hist(image, bbox):
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def compare_hist(h1, h2):
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

# === INITIALISATION ===
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH).to(device)
    return model, device

model, device = load_model()

# === SESSION STATE ===
if 'paused' not in st.session_state:
    st.session_state.paused = False
if 'reselect' not in st.session_state:
    st.session_state.reselect = False
if 'player_boxes' not in st.session_state:
    st.session_state.player_boxes = [None, None]
if 'player_histograms' not in st.session_state:
    st.session_state.player_histograms = [None, None]
if 'clicks' not in st.session_state:
    st.session_state.clicks = []
if 'frame_id' not in st.session_state:
    st.session_state.frame_id = 0

st.title("Badminton YOLO Streamlit Front")

# === CONTROLS ===
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("⏯️ Pause/Reprendre"):
        st.session_state.paused = not st.session_state.paused
with col2:
    if st.button("🔁 Re-sélection joueurs"):
        st.session_state.reselect = True
        st.session_state.clicks = []
with col3:
    st.write(f"Frame: {st.session_state.frame_id}")

# === VIDEO CAPTURE ===
@st.cache_resource
def get_video_capture():
    stream_url = get_youtube_stream_url(YOUTUBE_URL)
    cap = cv2.VideoCapture(stream_url)
    return cap

cap = get_video_capture()

# === FRAME LOOP ===
frame_placeholder = st.empty()
info_placeholder = st.empty()

while True:
    if st.session_state.paused:
        st.stop()
    ret, frame = cap.read()
    if not ret:
        info_placeholder.error("❌ Impossible de lire la vidéo.")
        break
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Sélection des joueurs
    if st.session_state.reselect or st.session_state.player_boxes == [None, None]:
        # Détection des personnes
        detections = model(frame, imgsz=IMG_SIZE, device=device, conf=CONFIDENCE_THRESHOLD)[0].boxes
        people = []
        for b in detections:
            if int(b.cls[0]) == CLS_PERSON:
                people.append(b.xyxy[0].cpu().numpy().astype(int))
        preview = frame.copy()
        for box in people:
            cv2.rectangle(preview, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
        st.info("Cliquez sur l'image pour sélectionner Joueur 1 puis Joueur 2.")
        img = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        clicked = st.session_state.clicks
        click = st.image(img, channels="RGB", use_column_width=True)
        # Récupérer les clics
        coords = st.experimental_get_query_params().get('coords', [])
        if len(coords) > len(clicked):
            x, y = map(int, coords[-1].split(','))
            clicked.append((x, y))
        if len(clicked) == 2:
            player_boxes = [None, None]
            player_histograms = [None, None]
            for i, (x, y) in enumerate(clicked):
                best, dmin = None, 1e9
                for box in people:
                    cx, cy = (box[0]+box[2])//2, (box[1]+box[3])//2
                    d = np.hypot(x - cx, y - cy)
                    if d < dmin:
                        best, dmin = box, d
                player_boxes[i] = best
                player_histograms[i] = get_hist(frame, best)
            st.session_state.player_boxes = player_boxes
            st.session_state.player_histograms = player_histograms
            st.session_state.reselect = False
            st.session_state.clicks = []
        st.stop()

    # Tracking et affichage
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
                score = compare_hist(hist, st.session_state.player_histograms[i])
                if score > best_scores[i]:
                    best_scores[i], best_players[i] = score, xyxy
        elif cls == CLS_SHUTTLE:
            shuttle_box = xyxy
    # Affichage
    colors = [(0, 255, 0), (255, 0, 0)]
    for i, box in enumerate(best_players):
        if box is not None and best_scores[i] > 0.3:
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
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(img, channels="RGB", use_column_width=True)
    st.session_state.frame_id += 1
    # Pause courte pour simuler la vidéo
    if not st.session_state.paused:
        cv2.waitKey(30) 