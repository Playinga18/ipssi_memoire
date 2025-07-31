import cv2
from ultralytics import YOLO
import yt_dlp
import argparse
import numpy as np

DEFAULT_URL = "https://www.youtube.com/watch?v=GXLFKy0GYJI"
SHUTTLE_MODEL = "../model/weights_v2.pt"
PLAYER_MODEL = "../model/yolov8n.pt"
IMG_SIZE = 1080
CONFIDENCE_SHUTTLE = 0.10
CONFIDENCE_PLAYER = 0.40  # Peut-être ajuster, 0.3 à 0.5 est souvent optimal
SHUTTLE_CLASS_NAME = "shuttlecock"
PLAYER_CLASS_ID = 0  # YOLOv8, la classe "person" est 0
ADVANCE_SEC = 5

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
    """Simple heuristique : on ne conserve que la partie centrale (terrain de badminton).
    (Affinable selon vidéo)"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    # Le terrain occupe souvent la partie centrale (ajuste si besoin)
    x1, y1, x2, y2 = int(w*0.15), int(h*0.12), int(w*0.85), int(h*0.85)
    mask[y1:y2, x1:x2] = 255
    return mask

def is_on_terrain(box, mask):
    # box = [x1, y1, x2, y2]
    h, w = mask.shape
    x1, y1, x2, y2 = [max(0, min(int(val), w if i%2==0 else h)) for i, val in enumerate(box)]
    box_mask = mask[y1:y2, x1:x2]
    # Garder si 60% de la boîte est sur le terrain (à affiner si besoin)
    if box_mask.size == 0:
        return False
    return np.mean(box_mask) > 180

def main():
    parser = argparse.ArgumentParser(description="Detection multi-objet badminton YouTube")
    parser.add_argument('--url', type=str, help="URL de la vidéo YouTube", default=DEFAULT_URL)
    args = parser.parse_args()
    youtube_url = args.url

    # YOLO Shuttlecock + Player
    model_shuttle = YOLO(SHUTTLE_MODEL)
    model_player = YOLO(PLAYER_MODEL)

    stream_url = get_youtube_stream_url(youtube_url)
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Erreur lors de l'ouverture du flux YouTube.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    paused = False
    frame_pos = 0
    last_frame = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Fin ou coupure du flux.")
                break
            frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            last_frame = frame

            # Resize pour 1080x1080 (adaptable en 1920x1080 si besoin)
            frame_proc = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame_blur = cv2.GaussianBlur(frame_proc, (3,3), 0)

            # ==== Détection Shuttlecock ====
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

            # ==== Détection Joueurs ====
            results_player = model_player.predict(frame_blur, conf=CONFIDENCE_PLAYER, imgsz=IMG_SIZE, verbose=False, classes=[PLAYER_CLASS_ID])
            player_boxes = []
            mask = terrain_mask(frame_proc)
            for r in results_player:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    # Préfiltrage sur la taille (exclure petites box = spectateurs/arbitres)
                    box_area = (x2-x1) * (y2-y1)
                    if box_area < (IMG_SIZE*IMG_SIZE)*0.01:  # Ignore très petites détections
                        continue
                    # Garder uniquement les personnes sur le terrain
                    if is_on_terrain([x1, y1, x2, y2], mask):
                        player_boxes.append((x1, y1, x2, y2, conf))

            # ==== Affichage ====
            # Shuttlecock (vert)
            for x1, y1, x2, y2, conf in shuttle_boxes:
                cv2.rectangle(frame_proc, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame_proc, f"Shuttlecock {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # Joueurs (bleu)
            for x1, y1, x2, y2, conf in player_boxes:
                cv2.rectangle(frame_proc, (x1, y1), (x2, y2), (255,0,0), 2)
                cv2.putText(frame_proc, f"Player {conf:.2f}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            # Optionnel: masque terrain (pour debug, enlever le commentaire)
            # mask_3c = cv2.merge([mask, mask, mask])
            # frame_proc = cv2.addWeighted(frame_proc, 0.8, mask_3c, 0.2, 0)

            # Aide contrôles
            cv2.rectangle(frame_proc, (0,0), (540,70), (0,0,0), -1)
            cv2.putText(frame_proc, "ESPACE: Pause/Play", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame_proc, "<-/j: -5s    ->/l: +5s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame_proc, "Q: Quitter", (380, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Badminton Detection (YouTube Streaming)", frame_proc)
        else:
            # Pause: affiche la dernière frame
            if last_frame is not None:
                frame_proc = cv2.resize(last_frame, (IMG_SIZE, IMG_SIZE))
                cv2.imshow("Badminton Detection (PAUSE)", frame_proc)

        key = cv2.waitKey(20) & 0xFF
        if key == ord(' '):  # Pause / Play
            paused = not paused
        elif key == ord('q'):
            break
        elif key == 83 or key == ord('l'):  # Flèche droite ou 'l'
            frame_to_go = int(frame_pos + fps * ADVANCE_SEC)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_go)
            paused = False
        elif key == 81 or key == ord('j'):  # Flèche gauche ou 'j'
            frame_to_go = max(0, int(frame_pos - fps * ADVANCE_SEC))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_go)
            paused = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
