import os
import subprocess
from datasets import Dataset
import json
import yt_dlp

num_proc = os.cpu_count()  # Total available CPU coresnum_proc


def download_youtube_video(youtube_id, youtube_url, save_dir):
    """ Downloads a YouTube video using yt-dlp in MP4 format with audio merged. """
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{youtube_id.replace('v_', '')}")
    #print('youtube_url', youtube_url)

    command = [
        "yt-dlp",
        "-f", "best[ext=mp4]",  # Download best MP4 format'
        "--max-filesize", "500M",  # Avoid large file downloads
        "--quiet",  # Suppress output
        "--no-warnings",  # Suppress warnings
        "-o", output_path,  # Save as output_path
        youtube_url
    ]
    # Create a yt-dlp downloader instance
    try:
        subprocess.run(command, check=True)
        return output_path if os.path.exists(output_path) else None
    except subprocess.CalledProcessError:
        return None



    

def download_video(example):
    video_id = example["video_path"]
    
    # Skip if invalid format
    if not isinstance(video_id, str) or not video_id.startswith("v_"):
        return {"video_path": None}
    
    youtube_url = f"https://www.youtube.com/watch?{video_id.replace('v_', 'v=')}"
    save_dir = "source_data_new1"
    saved_path = download_youtube_video(video_id, youtube_url, save_dir)

    return {"video_path": saved_path}

dataset1 = dataset.map(download_video, num_proc=num_proc-4)

dataset1 = dataset1.filter(lambda example: example["video_path"] is not None)
dataset1 = dataset1.shuffle(seed=42)

