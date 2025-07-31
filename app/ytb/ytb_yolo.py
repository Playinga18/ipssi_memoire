import cv2
import json
import time
import torch
import argparse
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import yt_dlp
from collections import deque

# === ARGUMENTS ===
parser = argparse.ArgumentParser(description="Détection des 2 joueurs avec sélection par clic")
parser.add_argument('--url', required=True, help='URL de la vidéo YouTube à analyser')
args = parser.parse_args()
YOUTUBE_URL = args.url

# === CONFIGURATION ===
PLAYER_MODEL_PATH = "./model/yolov8n.pt"
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
IMG_SIZE = 640
CONFIDENCE_THRESHOLD = 0.3
OUTPUT_JSON = "detections_selected_players.json"
TARGET_FPS = 30
MIN_HIST_SCORE = 0.25  # Score minimal pour valider un joueur par histogramme
CLS_PERSON = 0

# === VARIABLES GLOBALES ===
clicks = []
player_boxes = [None, None]
player_histograms = [None, None]
selection_active = False

# === CHARGEMENT DU MODÈLE ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(PLAYER_MODEL_PATH).to(device)
print(f"✅ Modèle chargé sur {device}")

# === CLASSE POUR TRACKER LES JOUEURS ===
class PlayerTracker:
    def __init__(self):
        self.player_positions = [deque(maxlen=20), deque(maxlen=20)]  # Historique positions
        self.lost_frames = [0, 0]  # Compteur frames perdues
        self.max_lost_frames = 10
        self.max_distance_jump = 200  # Distance max entre 2 frames
    
    def update_positions(self, boxes):
        """Met à jour les positions des joueurs"""
        for i, box in enumerate(boxes):
            if box is not None:
                center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
                self.player_positions[i].append(center)
                self.lost_frames[i] = 0
            else:
                self.lost_frames[i] += 1
    
    def is_valid_position(self, player_idx, new_box):
        """Vérifie si la nouvelle position est cohérente"""
        if not self.player_positions[player_idx]:
            return True
        
        last_pos = self.player_positions[player_idx][-1]
        new_center = ((new_box[0] + new_box[2]) // 2, (new_box[1] + new_box[3]) // 2)
        distance = np.sqrt((new_center[0] - last_pos[0])**2 + (new_center[1] - last_pos[1])**2)
        
        return distance < self.max_distance_jump
    
    def get_tracking_status(self):
        """Retourne le statut du tracking pour les 2 joueurs"""
        return [
            self.lost_frames[0] < self.max_lost_frames,
            self.lost_frames[1] < self.max_lost_frames
        ]

# === FONCTIONS HISTOGRAMME ===
def get_hist(image, bbox):
    """Calcule l'histogramme couleur d'une région"""
    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def compare_hist(h1, h2):
    """Compare deux histogrammes"""
    if h1 is None or h2 is None:
        return 0.0
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

# === CALLBACK SOURIS ===
def mouse_callback(event, x, y, flags, param):
    global clicks, selection_active
    if event == cv2.EVENT_LBUTTONDOWN and selection_active and len(clicks) < 2:
        clicks.append((x, y))
        print(f"🖱️ Clic joueur {len(clicks)} : ({x}, {y})")

# === FONCTION DE SÉLECTION ===
def select_players(frame):
    """Interface de sélection des joueurs par clic"""
    global clicks, player_boxes, player_histograms, selection_active
    
    clicks = []
    selection_active = True
    
    # Détecter toutes les personnes
    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        device=device,
        verbose=False
    )
    
    people = []
    for box in results[0].boxes:
        if int(box.cls[0]) == CLS_PERSON:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            people.append(xyxy)
    
    if len(people) == 0:
        print("❌ Aucun joueur détecté")
        return False
    
    # Interface de sélection
    preview = frame.copy()
    for i, box in enumerate(people):
        cv2.rectangle(preview, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
        cv2.putText(preview, f"#{i+1}", (box[0], box[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Instructions
    instructions = [
        "SELECTION DES JOUEURS:",
        "1. Cliquez sur le PREMIER joueur",
        "2. Cliquez sur le DEUXIEME joueur",
        "3. Appuyez sur ESPACE pour valider"
    ]
    for i, text in enumerate(instructions):
        color = (0, 255, 0) if i == 0 else (255, 255, 255)
        cv2.putText(preview, text, (10, 30 + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imshow("Selection des joueurs", preview)
    cv2.setMouseCallback("Selection des joueurs", mouse_callback)
    
    # Attendre 2 clics
    while len(clicks) < 2:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            selection_active = False
            cv2.destroyWindow("Selection des joueurs")
            return False
        
        # Redessiner avec les clics actuels
        if len(clicks) > 0:
            preview_temp = preview.copy()
            for j, (cx, cy) in enumerate(clicks):
                cv2.circle(preview_temp, (cx, cy), 10, (0, 255, 0), -1)
                cv2.putText(preview_temp, f"J{j+1}", (cx-15, cy-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(preview_temp, f"Joueurs selectionnes: {len(clicks)}/2",
                       (10, FRAME_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Selection des joueurs", preview_temp)
    
    # Attendre validation
    cv2.putText(preview, "Appuyez sur ESPACE pour valider, 'r' pour recommencer",
               (10, FRAME_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Selection des joueurs", preview)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Validation
            break
        elif key == ord('r'):  # Recommencer
            return select_players(frame)
        elif key == ord('q'):
            selection_active = False
            cv2.destroyWindow("Selection des joueurs")
            return False
    
    # Associer les clics aux boîtes les plus proches
    player_boxes = [None, None]
    player_histograms = [None, None]
    
    for i, (click_x, click_y) in enumerate(clicks):
        best_box = None
        min_distance = float('inf')
        
        for box in people:
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
            distance = np.sqrt((click_x - center_x)**2 + (click_y - center_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                best_box = box
        
        if best_box is not None:
            player_boxes[i] = best_box
            hist = get_hist(frame, best_box)
            player_histograms[i] = hist
            print(f"✅ Joueur {i+1} sélectionné : bbox {best_box}")
    
    selection_active = False
    cv2.destroyWindow("Selection des joueurs")
    return True

# === FONCTION POUR RÉCUPÉRER LE FLUX YOUTUBE ===
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

# === PROGRAMME PRINCIPAL ===
print("🔗 Connexion au flux YouTube...")
stream_url = get_youtube_stream_url(YOUTUBE_URL)
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    raise RuntimeError("❌ Impossible d'ouvrir le flux vidéo")

fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = max(1, int(1000 / TARGET_FPS))

# Lecture de la première frame pour sélection
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("❌ Impossible de lire la première frame")

first_frame = cv2.resize(first_frame, (FRAME_WIDTH, FRAME_HEIGHT))

# Sélection initiale des joueurs
print("👆 Phase de sélection des joueurs...")
if not select_players(first_frame):
    print("❌ Sélection annulée")
    cap.release()
    exit()

# Initialisation du tracker
tracker = PlayerTracker()
frame_id = 0
detections = []
paused = False

print("🎯 Début du suivi des joueurs sélectionnés...")
print("Contrôles: ESPACE=pause, 'r'=resélectionner, 'q'=quitter")

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Détection de toutes les personnes
        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            device=device,
            verbose=False
        )
        
        # Trouver les meilleurs matchs pour nos 2 joueurs
        current_players = [None, None]
        best_scores = [-1, -1]
        
        for box in results[0].boxes:
            if int(box.cls[0]) == CLS_PERSON:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                
                # Calculer l'histogramme de la détection actuelle
                current_hist = get_hist(frame, xyxy)
                
                # Comparer avec les histogrammes de référence
                for i in range(2):
                    if player_histograms[i] is not None:
                        hist_score = compare_hist(current_hist, player_histograms[i])
                        
                        # Vérifier cohérence de position
                        position_valid = tracker.is_valid_position(i, xyxy)
                        
                        # Score combiné (histogramme + cohérence position)
                        final_score = hist_score if position_valid else hist_score * 0.5
                        
                        if final_score > best_scores[i] and final_score > MIN_HIST_SCORE:
                            best_scores[i] = final_score
                            current_players[i] = {
                                'bbox': xyxy,
                                'confidence': conf,
                                'hist_score': hist_score
                            }
        
        # Mettre à jour le tracker
        tracker.update_positions([p['bbox'] if p else None for p in current_players])
        tracking_status = tracker.get_tracking_status()
        
        # Affichage et enregistrement
        colors = [(0, 255, 0), (255, 0, 0)]  # Vert et Rouge
        labels = ["Joueur 1", "Joueur 2"]
        
        for i, player in enumerate(current_players):
            if player is not None and tracking_status[i]:
                bbox = player['bbox']
                color = colors[i]
                
                # Encadrer le joueur
                thickness = 3
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
                
                # Label avec informations
                label = f"{labels[i]} ({player['confidence']:.2f}|{player['hist_score']:.2f})"
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Enregistrer la détection
                detection = {
                    "frame_id": frame_id,
                    "timestamp": datetime.now().isoformat(),
                    "player_id": i + 1,
                    "class_name": "person",
                    "confidence": player['confidence'],
                    "hist_score": player['hist_score'],
                    "bbox": [float(x) for x in bbox.tolist()],
                    "tracking_status": "active"
                }
                detections.append(detection)
            else:
                # Joueur perdu
                status_color = (0, 0, 255)  # Rouge
                status_text = f"{labels[i]} - PERDU ({tracker.lost_frames[i]} frames)"
                cv2.putText(frame, status_text, (10, 60 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Informations générales
        info_text = f"Frame: {frame_id} | Joueurs actifs: {sum(tracking_status)}/2"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        frame_id += 1

    # Affichage
    cv2.imshow("Suivi des 2 joueurs sélectionnés", frame)
    key = cv2.waitKey(wait_time) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord(' '):  # Pause
        paused = not paused
        print("⏸️ Pause" if paused else "▶️ Reprise")
    elif key == ord('r'):  # Resélection
        print("🔄 Resélection des joueurs...")
        if select_players(frame):
            tracker = PlayerTracker()  # Reset tracker
            print("✅ Nouveaux joueurs sélectionnés")
        else:
            print("❌ Resélection annulée")

cap.release()
cv2.destroyAllWindows()

# === SAUVEGARDE JSON ===
with open(OUTPUT_JSON, "w") as f:
    json.dump(detections, f, indent=4)

print(f"✅ Analyse terminée. Résultats enregistrés dans {OUTPUT_JSON}")
print(f"📊 Total détections: {len(detections)}")
if detections:
    player1_count = len([d for d in detections if d['player_id'] == 1])
    player2_count = len([d for d in detections if d['player_id'] == 2])
    print(f"📊 Joueur 1: {player1_count} détections")
    print(f"📊 Joueur 2: {player2_count} détections")