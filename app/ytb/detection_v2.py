import cv2
from ultralytics import YOLO
import yt_dlp
import argparse
import numpy as np
from collections import deque
import time

DEFAULT_URL = "https://www.youtube.com/watch?v=GXLFKy0GYJI"
SHUTTLE_MODEL = "../model/weights_v2.pt"
PLAYER_MODEL = "../model/yolov8n.pt"
IMG_SIZE = 1080
CONFIDENCE_SHUTTLE = 0.10
CONFIDENCE_PLAYER = 0.40
SHUTTLE_CLASS_NAME = "shuttlecock"
PLAYER_CLASS_ID = 0
ADVANCE_SEC = 5

MAX_TRAJ_DURATION = 2.0    # secondes pour la trajectoire affichée
MAX_DIST_FOR_LINE = 80     # pixels, pour relier deux points proches seulement

# Nouveaux paramètres d'immobilité
IMMOBILE_THRESHOLD = 10    # px (zone de tolérance)
IMMOBILE_DURATION = 3.0    # sec (temps avant d'ignorer cette position)

class SimpleShuttleTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.last_seen = time.time()
        self.tracking = False

    def update(self, x, y):
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(measured)
        self.last_seen = time.time()
        self.tracking = True

    def predict(self):
        prediction = self.kalman.predict()
        return int(prediction[0]), int(prediction[1])

    def is_lost(self, max_lost=0.5):
        return (time.time() - self.last_seen) > max_lost

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

def main():
    parser = argparse.ArgumentParser(description="Detection multi-objet badminton YouTube")
    parser.add_argument('--url', type=str, help="URL de la vidéo YouTube", default=DEFAULT_URL)
    args = parser.parse_args()
    youtube_url = args.url

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

    shuttle_tracker = SimpleShuttleTracker()
    shuttle_trajectory = deque(maxlen=500)

    last_shuttle_pos = None        # Pour la gestion d'immobilité
    immobile_since = None          # Horodatage début d'immobilité
    ignore_immobile = False        # Faut-il ignorer la détection actuelle

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Fin ou coupure du flux.")
                break
            frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            last_frame = frame

            frame_proc = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame_blur = cv2.GaussianBlur(frame_proc, (3,3), 0)

            # === Détection shuttlecock ===
            results_shuttle = model_shuttle.predict(frame_blur, conf=CONFIDENCE_SHUTTLE, imgsz=IMG_SIZE, verbose=False)
            shuttle_detected = False
            shuttle_center = None

            for r in results_shuttle:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = r.names[cls] if hasattr(r, "names") else str(cls)
                    if SHUTTLE_CLASS_NAME in label.lower() and conf > CONFIDENCE_SHUTTLE:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                        cx, cy = (x1 + x2)//2, (y1 + y2)//2
                        current_time = time.time()
                        # -- IMMOBILE FILTER --
                        if last_shuttle_pos is not None and \
                            abs(cx - last_shuttle_pos[0]) <= IMMOBILE_THRESHOLD and \
                            abs(cy - last_shuttle_pos[1]) <= IMMOBILE_THRESHOLD:
                            # Toujours dans la même zone
                            if immobile_since is None:
                                immobile_since = current_time
                            elif (current_time - immobile_since) >= IMMOBILE_DURATION:
                                ignore_immobile = True
                            else:
                                ignore_immobile = False
                        else:
                            # La position a changé (reset)
                            immobile_since = None
                            ignore_immobile = False

                        if ignore_immobile:
                            continue  # Ignore toute détection à cette position (immobile trop longtemps)
                        # Sinon, c'est ok : enregistrer le point
                        last_shuttle_pos = (cx, cy)
                        shuttle_center = (cx, cy)
                        shuttle_detected = True
                        shuttle_tracker.update(cx, cy)

            # Tracking si perdu temporairement
            if not shuttle_detected and shuttle_tracker.tracking and not shuttle_tracker.is_lost():
                cx, cy = shuttle_tracker.predict()
                shuttle_center = (cx, cy)
            else:
                if shuttle_tracker.is_lost():
                    shuttle_tracker.tracking = False
                    shuttle_center = None

            # Trajectoire (quand on a une position valide)
            if shuttle_center is not None:
                now = time.time()
                shuttle_trajectory.append((shuttle_center[0], shuttle_center[1], now))
                # On garde que les 2 dernières secondes
                while shuttle_trajectory and (now - shuttle_trajectory[0][2]) > MAX_TRAJ_DURATION:
                    shuttle_trajectory.popleft()

            # === Détection joueurs ===
            results_player = model_player.predict(frame_blur, conf=CONFIDENCE_PLAYER, imgsz=IMG_SIZE, verbose=False, classes=[PLAYER_CLASS_ID])
            player_boxes = []
            mask = terrain_mask(frame_proc)
            for r in results_player:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    box_area = (x2-x1) * (y2-y1)
                    if box_area < (IMG_SIZE*IMG_SIZE)*0.01:
                        continue
                    if is_on_terrain([x1, y1, x2, y2], mask):
                        player_boxes.append((x1, y1, x2, y2, conf))

            # === Affichage ===
            # Shuttlecock (vert) : seulement s'il n'est pas "ignoré"
            if shuttle_center is not None:
                cv2.circle(frame_proc, shuttle_center, 8, (0,255,0), -1)
                cv2.putText(frame_proc, f"Shuttlecock", (shuttle_center[0], shuttle_center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Joueurs (bleu)
            for x1, y1, x2, y2, conf in player_boxes:
                cv2.rectangle(frame_proc, (x1, y1), (x2, y2), (255,0,0), 2)
                cv2.putText(frame_proc, f"Player {conf:.2f}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            # Trajectoire (jaune)
            points = list(shuttle_trajectory)
            for i in range(1, len(points)):
                x1, y1, t1 = points[i-1]
                x2, y2, t2 = points[i]
                dist = np.hypot(x2 - x1, y2 - y1)
                if dist < MAX_DIST_FOR_LINE:
                    cv2.line(frame_proc, (x1, y1), (x2, y2), (0,255,255), 2)

            # Contrôles
            cv2.rectangle(frame_proc, (0,0), (540,70), (0,0,0), -1)
            cv2.putText(frame_proc, "ESPACE: Pause/Play", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame_proc, "<-/j: -5s    ->/l: +5s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame_proc, "Q: Quitter", (380, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Badminton Detection (YouTube Streaming)", frame_proc)
        else:
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
