from pytube import YouTube

def download_youtube_video(url, output_path="."):
    yt = YouTube(url)
    # Télécharger la meilleure qualité vidéo + audio (progressive)
    stream = yt.streams.get_highest_resolution()
    print(f"Téléchargement de : {yt.title}")
    stream.download(output_path=output_path)
    print("Téléchargement terminé !")

if __name__ == "__main__":
    # Exemple d'URL (remplace par ton URL)
    video_url = "https://www.youtube.com/watch?v=zYqgJo1L5uM&ab_channel=BadmintonHam"
    # Dossier de sortie (par défaut : dossier courant)
    output_folder = "./output/video/"
    download_youtube_video(video_url, output_folder)
