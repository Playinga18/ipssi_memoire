import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import yt_dlp
from PIL import Image
import time
import torch

DEFAULT_URL = "https://www.youtube.com/watch?v=GXLFKy0GYJI"
SHUTTLE_MODEL = "./model/weights_v2.pt"
PLAYER_MODEL = "./model/yolov8n.pt"
FRAME_SIZE = 800
IMG_SIZE = 416  # encore plus rapide ! (essaye 320 si vraiment besoin)
CONFIDENCE_SHUTTLE = 0.15
CONFIDENCE_PLAYER = 0.40
SHUTTLE_CLASS_NAME = "shuttlecock"
PLAYER_CLASS_ID = 0
SKIP_FRAMES = 3  # Prédit toutes les 3 frames (fluidité ++)

# Force le backend torch optimal
torch.backends.cudnn.benchmark = True

st.set_page_config(page_title="Badminton Detection", layout="centered", initial_sidebar_state="collapsed")

# ---- THEME & STYLE ----
st.markdown("""
    <style>
    body, .stApp { background-color: #15191C; color: #F1F1F1; }
    .css-18e3th9 { background-color: #15191C !important; }
    .stButton button {
        background: linear-gradient(90deg, #1874CD 10%, #00C78C 90%);
        color: white;
        border-radius: 12px;
        padding: 0.8em 2em;
        font-size: 1.08em;
        margin: 0.25em 0.6em 1.1em 0.6em;
        border: none;
        transition: 0.2s;
        font-weight: 600;
        letter-spacing: 0.02em;
        display: block;
        width: 100%;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #1874CD 10%, #4CB944 90%);
        color: #FFF;
        transform: scale(1.03);
        box-shadow: 0 2px 12px #00C78C44;
    }
    .block-container {
        padding-top: 2.5rem;
        max-width: 1000px !important;
    }
    .stTextInput>div>div>input {
        background: #252A2E;
        color: #FFF;
        border-radius: 8px;
        border: 1px solid #444;
        font-size: 1.07em;
    }
    .main-title {
        color: #34E0A1;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        letter-spacing: 0.02em;
        margin-bottom: 0.15em;
        margin-top: 0.1em;
        line-height: 1.1;
        text-shadow: 0 2px 12px #00c78c33, 0 1px 0 #222;
    }
    .main-subtitle {
        text-align: center;
        font-size: 1.18em;
        color: #b6ffe0;
        margin-bottom: 1.2em;
        font-weight: 400;
    }
    .center-buttons-vertical {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1em;
        margin-bottom: 1.2em;
        margin-top: 0.8em;
        max-width: 360px;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_shuttle = YOLO(SHUTTLE_MODEL).to(device)
    model_player = YOLO(PLAYER_MODEL).to(device)
    return model_shuttle, model_player, device

def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        'format': 'best[ext=mp4][vcodec^=avc1]/bestvideo[ext=mp4][vcodec^=avc1]/best',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        if 'url' in info:
            return info['url']
        elif 'formats' in info:
            for f in info['formats']:
                if f.get('vcodec', '').startswith('avc1') and f.get('ext') == 'mp4':
                    return f['url']
            return info['formats'][0]['url']
        else:
            raise Exception("Stream URL introuvable.")

def terrain_mask(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = int(w*0.15), int(h*0.12), int(w*0.85), int(h*0.85)
    mask[y1:y2, x1:x2] = 255
    return mask

def is_on_terrain(box, mask):
    h, w = mask.shape
    x1, y1, x2, y2 = [max(0, min(int(val), w if i%2==0 else h)) for i, val in enumerate(box)]
    box_mask = mask[y1:y2, x1:x2]
    if box_mask.size == 0:
        return False
    return np.mean(box_mask) > 180

def process_frame(frame, model_shuttle, model_player, device):
    frame_proc = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    mask = terrain_mask(frame_proc)
    # Shuttlecock
    results_shuttle = model_shuttle.predict(frame_proc, conf=CONFIDENCE_SHUTTLE, imgsz=IMG_SIZE, device=device, verbose=False)
    shuttle_boxes = []
    for r in results_shuttle:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = r.names[cls] if hasattr(r, "names") else str(cls)
            if SHUTTLE_CLASS_NAME in label.lower() and conf > CONFIDENCE_SHUTTLE:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                shuttle_boxes.append((x1, y1, x2, y2, conf))
    # Joueurs
    results_player = model_player.predict(frame_proc, conf=CONFIDENCE_PLAYER, imgsz=IMG_SIZE, device=device, verbose=False, classes=[PLAYER_CLASS_ID])
    player_boxes = []
    for r in results_player:
        for box in r.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            box_area = (x2-x1) * (y2-y1)
            if box_area < (IMG_SIZE*IMG_SIZE)*0.01:
                continue
            if is_on_terrain([x1, y1, x2, y2], mask):
                player_boxes.append((x1, y1, x2, y2, conf))
    # Draw
    for x1, y1, x2, y2, conf in shuttle_boxes:
        cv2.rectangle(frame_proc, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame_proc, f"Shuttlecock {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    for x1, y1, x2, y2, conf in player_boxes:
        cv2.rectangle(frame_proc, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.putText(frame_proc, f"Player {conf:.2f}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    return frame_proc

# ----------- UI -----------
st.markdown('<div class="main-title">Détection IA Badminton<br>Shuttlecock & Joueurs</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Analyse automatique sur vidéo YouTube.<br>Entrez votre lien ou utilisez la vidéo de démo</div>', unsafe_allow_html=True)

url = st.text_input("", value=DEFAULT_URL, key="yturl", help="Coller ici l’URL de la vidéo YouTube à analyser.")

st.markdown('<div class="center-buttons-vertical">', unsafe_allow_html=True)
use_default = st.button("🎬 Vidéo par défaut", key="default_btn")
start_analysis = st.button("🚀 Démarrer l’analyse", key="start_btn")
st.markdown('</div>', unsafe_allow_html=True)

if use_default:
    url = DEFAULT_URL
    st.experimental_rerun()

if start_analysis:
    st.write("Chargement des modèles, connexion à la vidéo...")
    model_shuttle, model_player, device = load_models()
    try:
        stream_url = get_youtube_stream_url(url)
    except Exception as e:
        st.error(f"Erreur d'accès à la vidéo : {e}")
        st.stop()
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        st.error("Impossible d'ouvrir la vidéo.")
        st.stop()

    frame_placeholder = st.empty()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue  # Attend la prochaine frame valide
        frame_idx += 1
        if frame_idx % SKIP_FRAMES != 0:
            continue
        # Prediction optimisée
        processed = process_frame(frame, model_shuttle, model_player, device)
        # Respect du ratio (pas d'image déformée)
        img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        ratio = FRAME_SIZE / w
        new_w = FRAME_SIZE
        new_h = int(h * ratio)
        display_img = Image.fromarray(img_rgb).resize((new_w, new_h))
        frame_placeholder.image(display_img, use_column_width=True)
        # Pas de sleep ou minimal
    cap.release()
    st.success("Analyse terminée !")
