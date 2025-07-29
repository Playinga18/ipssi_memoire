#!/usr/bin/env python3
"""
Usage :
    python extract_frame_youtube.py <youtube_url> <nb_frames> <output_dir>

Exemple :
    python extract_frame_youtube.py "https://www.youtube.com/watch?v=MhIXLtGdKlo&t=507s" 100 ./frames_out
"""

import cv2
import random
import os
import sys
from pathlib import Path
from yt_dlp import YoutubeDL


def get_youtube_stream_url(youtube_url: str) -> str:
    """Récupère l'URL du flux vidéo lisible par OpenCV."""
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'quiet': True,
        'noplaylist': True,
        'skip_download': True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        return info_dict['url']


def extract_random_frames(video_stream_url: str,
                          n_frames: int,
                          output_dir: str,
                          img_extension: str = "jpg",
                          seed: int | None = 42) -> None:
    """Extrait n_frames images aléatoires depuis un flux vidéo YouTube."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_stream_url)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d’ouvrir la vidéo : {video_stream_url}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise RuntimeError("La vidéo semble vide ou illisible.")
    n_frames = min(n_frames, total)

    if seed is not None:
        random.seed(seed)
    frame_ids = random.sample(range(total), k=n_frames)

    print(f"🎞️ Total frames dans la vidéo : {total}")
    print(f"→ Extraction de {n_frames} frames vers « {output_dir} »…")

    for i, fid in enumerate(frame_ids, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️  Frame {fid} illisible, ignorée.")
            continue
        filename = f"frame_{fid:06d}.{img_extension}"
        cv2.imwrite(os.path.join(output_dir, filename), frame)
        print(f"[{i}/{n_frames}]  {filename}")

    cap.release()
    print("✅ Extraction terminée.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    youtube_url = sys.argv[1]
    try:
        nb_frames = int(sys.argv[2])
    except ValueError:
        print("Le paramètre <nb_frames> doit être un entier.")
        sys.exit(1)
    output_dir = sys.argv[3]

    try:
        print("🔗 Récupération du flux vidéo YouTube...")
        stream_url = get_youtube_stream_url(youtube_url)
        extract_random_frames(
            video_stream_url=stream_url,
            n_frames=nb_frames,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"❌ Erreur : {e}")
        sys.exit(1)
