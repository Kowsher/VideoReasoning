import json
import yt_dlp
import os
import concurrent.futures
import subprocess
from datasets import Dataset
from tqdm.auto import tqdm

def extract_reasoning_chain(reasoning_tree):
    """ Recursively extract the reasoning chain as a single string """
    reasoning_chain = []
    def traverse(node):
        reasoning_chain.append(node["reasoning"])
        if "next" in node:
            for key in node["next"]:
                traverse(node["next"][key])
    
    traverse(reasoning_tree)
    return " -> ".join(reasoning_chain)

def download_youtube_video(youtube_id, youtube_url, save_dir):
    """ Downloads a YouTube video using yt-dlp in MP4 format with audio merged. """
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{youtube_id}")

    command = [
        "yt-dlp",
        "-f", "best[ext=mp4]",  # Download best MP4 format
        "--max-filesize", "50M",  # Avoid large file downloads
        "-o", output_path,  # Save as output_path
        youtube_url
    ]
    
    try:
        subprocess.run(command, check=True)
        return output_path if os.path.exists(output_path) else None
    except subprocess.CalledProcessError:
        return None




def download_videos_and_create_dataset(json_path: str, save_dir: str):
    max_workers = os.cpu_count() // 2 if os.cpu_count() else 4
    print(f"Using {max_workers} parallel workers for downloading videos.")
    
    # Load dataset from JSON file
    with open(json_path, 'r') as f:
        dataset = json.load(f)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Store successfully downloaded videos
    successful_downloads = {}
    skipped_videos = 0
    failed_downloads = 0
    total_videos = len(dataset["videos"])
    
    # Get list of already downloaded videos
    existing_videos = {file.split('.')[0] for file in os.listdir(save_dir)}
    
    # Prepare download tasks
    download_tasks = {}
    for vid, data in dataset["videos"].items():
        youtube_id = vid.split("_")[1]  # Extract youtube_id from video id
        
        # Check if video already exists
        if youtube_id in existing_videos:
            skipped_videos += 1
            continue
        
        url = f'https://www.youtube.com/watch?v={youtube_id}'
        download_tasks[youtube_id] = (url, vid, data)
    
    # Execute parallel downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {executor.submit(download_youtube_video, yt_id, url, save_dir): (yt_id, vid, data)
                           for yt_id, (url, vid, data) in download_tasks.items()}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_video), total=len(future_to_video), desc="Downloading Videos"):
            yt_id, vid, data = future_to_video[future]
            video_path = future.result()
            if video_path:
                successful_downloads[vid] = {
                    "video_path": video_path,
                    "question": data["question"],
                    "options": data["options"],
                    "cot_A": extract_reasoning_chain(data["reasoning_tree"]["Perception & Scene Understanding"]),
                    "cot_B": extract_reasoning_chain(data["reasoning_tree"]["Incorrect Path 1"]["Perception & Scene Understanding"]),
                    "cot_C": extract_reasoning_chain(data["reasoning_tree"]["Incorrect Path 2"]["Event Segmentation & Action Recognition"]),
                    "cot_D": extract_reasoning_chain(data["reasoning_tree"]["Incorrect Path 3"]["Temporal & Causal Relationship Analysis"]),
                    "correct_answer": data["correct_answer"]
                }
            else:
                failed_downloads += 1
    
    # Generate summary report
    downloaded_videos = len(successful_downloads)
    print("\nDownload Report:")
    print(f"Total videos in dataset: {total_videos}")
    print(f"Successfully downloaded: {downloaded_videos}")
    print(f"Already existed (skipped): {skipped_videos}")
    print(f"Failed downloads: {failed_downloads}")
    
    # Convert to Hugging Face Dataset format
    dataset_list = []
    for vid, info in successful_downloads.items():
        dataset_list.append(info)
    
    hf_dataset = Dataset.from_list(dataset_list)
    
    # Save the dataset
    dataset_path = os.path.join(save_dir, "hf_video_dataset")
    hf_dataset.save_to_disk(dataset_path)
    print("Hugging Face dataset saved successfully!")
    
    return hf_dataset

download_videos_and_create_dataset('train.json', 'videos')
