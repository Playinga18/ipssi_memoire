from flask import Flask, Response, jsonify, request
from ultralytics import YOLO
import cv2

app = Flask(__name__)
model = YOLO("weights.pt")  # ton modèle

@app.route("/detect", methods=["POST"])
def detect_video():
    video_path = request.json.get("video_url")
    if not video_path:
        return jsonify({"error": "video_url manquant"}), 400

    cap = cv2.VideoCapture(video_path)
    detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for r in results:
            for box in r.boxes:
                cls = model.names[int(box.cls)]
                detections.append({
                    "label": cls,
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()
                })

    cap.release()
    return jsonify(detections)
