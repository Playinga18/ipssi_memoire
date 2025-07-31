import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import yt_dlp
from PIL import Image
import time

DEFAULT_URL = "https://www.youtube.com/watch?v=GXLFKy0GYJI"
SHUTTLE_MODEL = "./model/weights_v2.pt"
PLAYER_MODEL = "yolov8n.pt"
FRAME_SIZE = 800
IMG_SIZE = 720
CONFIDENCE_SHUTTLE = 0.15
CONFIDENCE_PLAYER = 0.40
SHUTTLE_CLASS_NAME = "shuttlecock"
PLAYER_CLASS_ID = 0  # YOLOv8: class 'person'

st.set_page_config(page_title="Badminton Detection", layout="centered", initial_sidebar_state="collapsed")

# ---- THEME PERSONNALISÉ ----
st.markdown("""
    <style>
    body, .stApp {
        background-color: #15191C;
        color: #F1F1F1;
    }
    .css-18e3th9 { background-color: #15191C !important; }
    .stButton button {
        background: linear-gradient(90deg, #1874CD 30%, #00C78C 90%);
        color: white;
        border-radius: 10px;
        padding: 0.75em 2em;
        font-size: 1.1em;
        margin: 0.5em 0.5em 1em 0.5em;
        border: none;
        transition: 0.2s;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #1874CD 10%, #4CB944 90%);
        color: #FFF;
        transform: scale(1.03);
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1000px !important;
    }
    .stTextInput>div>div>input {
        background: #252A2E;
        color: #FFF;
        border-radius: 8px;
        border: 1px solid #444;
    }
    .css-1offfwp { color: #93FFD8; font-size: 2rem; text-align: center; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model_shuttle = YOLO(SHUTTLE_MODEL)
    model_player = YOLO(PLAYER_MODEL)
    return model_shuttle, model_player

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

def process_frame(frame, model_shuttle, model_player):
    frame_proc = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_blur = cv2.GaussianBlur(frame_proc, (3,3), 0)
    mask = terrain_mask(frame_proc)
    # Détection Shuttlecock
    results_shuttle = model_shuttle.predict(frame_blur, conf=CONFIDENCE_SHUTTLE, imgsz=IMG_SIZE, verbose=False)
    shuttle_boxes = []
    for r in results_shuttle:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = r.names[cls] if hasattr(r, "names") else str(cls)
            if SHUTTLE_CLASS_NAME in label.lower() and conf > CONFIDENCE_SHUTTLE:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                shuttle_boxes.append((x1, y1, x2, y2, conf))
    # Détection Joueurs
    results_player = model_player.predict(frame_blur, conf=CONFIDENCE_PLAYER, imgsz=IMG_SIZE, verbose=False, classes=[PLAYER_CLASS_ID])
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
    # Dessin des boxes
    for x1, y1, x2, y2, conf in shuttle_boxes:
        cv2.rectangle(frame_proc, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame_proc, f"Shuttlecock {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    for x1, y1, x2, y2, conf in player_boxes:
        cv2.rectangle(frame_proc, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.putText(frame_proc, f"Player {conf:.2f}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    return frame_proc

# --- UI épurée ---
st.markdown('<h1 class="css-1offfwp">Détection Badminton IA : Shuttlecock & Joueurs</h1>', unsafe_allow_html=True)
st.markdown("<div style='text-align:center; margin-bottom:1.5em;'>\
    <b>Analyse automatique sur vidéo YouTube. <br> Entrer votre lien ou cliquez pour utiliser la vidéo de démo.</b></div>", unsafe_allow_html=True)

url = st.text_input("", value=DEFAULT_URL, key="yturl", help="Coller ici l'URL de la vidéo YouTube à analyser.")
if st.button("Utiliser la vidéo par défaut 🎥", use_container_width=True):
    url = DEFAULT_URL

if st.button("Démarrer l'analyse vidéo 🚀", use_container_width=True):
    st.write("Chargement des modèles, connexion à la vidéo...")
    model_shuttle, model_player = load_models()
    try:
        stream_url = get_youtube_stream_url(url)
    except Exception as e:
        st.error(f"Erreur d'accès à la vidéo : {e}")
        st.stop()
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        st.error("Impossible d'ouvrir la vidéo.")
        st.stop()
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_placeholder = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Fin du flux vidéo ou problème de lecture.")
            break
        processed = process_frame(frame, model_shuttle, model_player)
        img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        display_img = Image.fromarray(img_rgb)
        display_img = display_img.resize((FRAME_SIZE, int(FRAME_SIZE*img_rgb.shape[0]/img_rgb.shape[1])))
        frame_placeholder.image(display_img, caption=f"Frame / {total_frames if total_frames>0 else '?'}", use_column_width=True)
        time.sleep(0.03)  # Pour éviter d'exploser le CPU
    cap.release()
    st.success("Analyse terminée !")
