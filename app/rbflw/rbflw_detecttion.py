from ultralytics import YOLO

def main():
    model = YOLO('yolov8s.pt')
    results = model.train(
        data='../data/dataset/data.yaml',
        epochs=30,              # Mets plus d’epochs pour bien converger
        imgsz=320,              # Baisse la taille pour matcher le 360p natif
        batch=4,                # Ajuste à ta RAM
        workers=4,              # Ajuste si plantage RAM
        optimizer='Adam',       # Peut aider sur petit dataset/noisy
        amp=True                # Training plus rapide si GPU compatible
    )

    # Inférence test
    model.predict('../output/frame_000711.jpg', save=True)

if __name__ == '__main__':
    main()
