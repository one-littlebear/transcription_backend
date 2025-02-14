import requests
import time
import os
from dotenv import load_dotenv
import httpx
import asyncio
from pathlib import Path

# Load environment variables
load_dotenv()

# API endpoint (local FastAPI server)
BASE_URL = "http://localhost:8000"

async def test_api_endpoints():
    """Test basic API endpoints without file processing"""
    try:
        async with httpx.AsyncClient() as client:
            # Test root endpoint
            print("\nTesting root endpoint...")
            response = await client.get(f"{BASE_URL}/")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")

            # Test list videos endpoint
            print("\nTesting list videos endpoint...")
            response = await client.get(f"{BASE_URL}/videos")
            print(f"Status: {response.status_code}")
            videos = response.json()
            print("Available videos:")
            for video in videos:
                print(f"- {video}")

    except Exception as e:
        print(f"Error during API testing: {e}")

def process_video(video_path):
    """Process a single video file"""
    filename = os.path.basename(video_path)
    print(f"\nProcessing video: {filename}")
    
    # Step 1: Get upload URL
    print("Getting upload URL...")
    response = requests.post(f"{BASE_URL}/upload/request", params={"filename": filename})
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
        f"{BASE_URL}/process/{upload_data['cleaned_filename']}"
    )
    
    if process_response.status_code != 200:
        print(f"Processing failed: {process_response.text}")
        return
    
    # Step 4: Check status until complete
    print("Checking processing status...")
    while True:
        status_response = requests.get(
            f"{BASE_URL}/upload/status/{upload_data['cleaned_filename']}"
        )
        
        if status_response.status_code != 200:
            print(f"Status check failed: {status_response.text}")
            return
            
        status_data = status_response.json()
        
        if status_data.get("status") == "completed":
            print("✅ Processing complete!")
            if status_data.get("transcript_url"):
                print(f"Download your transcript at: {status_data['transcript_url']}")
            break
        elif status_data.get("status") == "processing":
            print("⏳ Still processing...")
            time.sleep(10)  # Wait 10 seconds before checking again
        elif status_data.get("status") == "failed":
            print(f"❌ Processing failed: {status_data.get('message', 'Unknown error')}")
            break
        else:
            print(f"❌ Unexpected status: {status_data}")
            break

if __name__ == "__main__":
    print("FastAPI Testing Tool")
    print("===================")
    
    # First, check if server is running and test basic endpoints
    try:
        response = httpx.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ Server is running!")
            # Run the API endpoint tests
            asyncio.run(test_api_endpoints())
            
            # Then process the video
            video_path = "videos/FULL SPEECH： President Donald Trump's inauguration speech.mp4"
            if os.path.exists(video_path):
                process_video(video_path)
            else:
                print(f"❌ Video file not found: {video_path}")
        else:
            print("❌ Server returned unexpected status code:", response.status_code)
    except httpx.ConnectError:
        print("❌ Error: Could not connect to the server.")
        print("Please make sure the FastAPI server is running (python main.py)") 
        
    print("===================") 