import cv2
from ultralytics import YOLO
import yt_dlp
import argparse

# ========== PARAMÈTRES ==========
DEFAULT_URL = "https://www.youtube.com/watch?v=GXLFKy0GYJI"
YOLO_WEIGHTS = "../model/weights_v2.pt"
IMG_SIZE = 1080
CONFIDENCE = 0.10
SHUTTLE_CLASS_NAME = "shuttlecock"
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

def main():
    # 1. Parser les arguments
    parser = argparse.ArgumentParser(description="Detection de volant de badminton en streaming YouTube")
    parser.add_argument('--url', type=str, help="URL de la vidéo YouTube", default=DEFAULT_URL)
    args = parser.parse_args()
    youtube_url = args.url

    # 2. Charger YOLO
    model = YOLO(YOLO_WEIGHTS)

    # 3. Stream la vidéo YT
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

            # Redimensionner la frame pour l'affichage 1080x1080
            frame_proc = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame_proc = cv2.GaussianBlur(frame_proc, (3,3), 0)

            # Détection YOLO
            results = model.predict(frame_proc, conf=CONFIDENCE, imgsz=IMG_SIZE, verbose=False)
            boxes = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = r.names[cls] if hasattr(r, "names") else str(cls)
                    if SHUTTLE_CLASS_NAME in label.lower() and conf > CONFIDENCE:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                        boxes.append((x1, y1, x2, y2, conf))

            for x1, y1, x2, y2, conf in boxes:
                cv2.rectangle(frame_proc, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame_proc, f"Shuttlecock {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Affichage aide
            cv2.rectangle(frame_proc, (0,0), (480,70), (0,0,0), -1)
            cv2.putText(frame_proc, "ESPACE: Pause/Play", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame_proc, "<-5s:j     l:5s->", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame_proc, "Q: Quitter", (340, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Shuttlecock Detection (YouTube Streaming)", frame_proc)
        else:
            # Pause: affiche la dernière frame
            if last_frame is not None:
                frame_proc = cv2.resize(last_frame, (IMG_SIZE, IMG_SIZE))
                cv2.imshow("Shuttlecock Detection (PAUSE)", frame_proc)

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
