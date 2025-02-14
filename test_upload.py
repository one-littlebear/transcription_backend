import requests
import time
import os

def test_video_processing(api_url, video_path):
    # Get filename from path
    filename = os.path.basename(video_path)
    
    # Step 1: Get upload URL
    print("Getting upload URL...")
    response = requests.post(f"{api_url}/upload/request", params={"filename": filename})
    upload_data = response.json()
    
    # Step 2: Upload video
    print("Uploading video...")
    with open(video_path, 'rb') as video_file:
        upload_response = requests.put(
            upload_data["upload_url"],
            data=video_file,
            headers={
                "x-ms-blob-type": "BlockBlob",
                "Content-Type": "video/mp4"
            }
        )
    
    if upload_response.status_code != 201:
        print(f"Upload failed: {upload_response.text}")
        return
    
    # Step 3: Start processing
    print("Starting processing...")
    process_response = requests.post(
        f"{api_url}/process/{upload_data['cleaned_filename']}"
    )
    
    if process_response.status_code != 200:
        print(f"Processing failed: {process_response.text}")
        return
    
    # Step 4: Check status until complete
    print("Checking status...")
    while True:
        status_response = requests.get(
            f"{api_url}/upload/status/{upload_data['cleaned_filename']}"
        )
        status_data = status_response.json()
        
        if status_data["status"] == "completed":
            print("Processing complete!")
            print(f"Download your transcript at: {status_data['transcript_url']}")
            break
        elif status_data["status"] == "processing":
            print("Still processing...")
            time.sleep(10)  # Wait 10 seconds before checking again
        else:
            print(f"Unexpected status: {status_data}")
            break

if __name__ == "__main__":
    API_URL = "https://transcription-backend.livelywater-0798357b.germanywestcentral.azurecontainerapps.io"
    VIDEO_PATH = "videos/NVIDIA BUSTED Devin with DEEPSEEK R1!!!.mp4"  # Replace with your video path
    
    test_video_processing(API_URL, VIDEO_PATH) 