import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from moviepy import VideoFileClip
from pathlib import Path
from openai import OpenAI
from pydub import AudioSegment
from utils.cleanup import split_into_sentences, ensure_output_directory
from docx import Document
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import time
import math
from typing import Optional, List
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceExistsError
from pydantic import BaseModel
from utils.video_upload import generate_sas_token, check_file_status, UploadResponse, UploadStatus

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Processing API",
    description="API for processing videos, extracting audio, and generating transcriptions",
    version="1.0.0"
)

class ProcessingStatus(BaseModel):
    status: str
    message: str

def extract_audio(video_path, output_path):
    """Extract audio from video file and save as MP3."""
    try:
        # Create the video clip
        video = VideoFileClip(video_path)
        try:
            # Get the audio and write it to file
            audio = video.audio
            audio.write_audiofile(output_path)
            return True
        finally:
            # Ensure resources are properly closed even if an error occurs
            if hasattr(video, 'close'):
                video.close()
            if hasattr(video.audio, 'close'):
                video.audio.close()
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        return False

def split_audio(audio_path):
    """Split audio file into 10-minute chunks."""
    try:
        audio = AudioSegment.from_mp3(audio_path)
        chunk_length = 10 * 60 * 1000  # 10 minutes in milliseconds
        chunks = []
        
        # Split audio into chunks
        for i in range(0, len(audio), chunk_length):
            chunk = audio[i:i + chunk_length]
            chunk_path = f"{audio_path.stem}_chunk_{i//chunk_length}{audio_path.suffix}"
            chunk_path = audio_path.parent / chunk_path
            chunk.export(str(chunk_path), format="mp3")
            chunks.append(chunk_path)
            logger.debug(f"Created audio chunk: {chunk_path}")
            
        return chunks
    except Exception as e:
        logger.error(f"Error splitting audio: {e}")
        return []

def transcribe_with_retry(audio_path: Path, client: OpenAI, max_retries: int = 3) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            with audio_path.open("rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            logger.debug(f"Successfully transcribed {audio_path}")
            return transcript
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {audio_path}: {e}. Retrying...")
            if attempt == max_retries - 1:
                logger.error(f"Final retry failed for {audio_path}: {e}")
                return None
            time.sleep(math.pow(2, attempt))  # Exponential backoff

def get_blob_service_client():
    """Initialize Azure Blob Service Client."""
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("Azure Storage connection string not found in environment variables")
    return BlobServiceClient.from_connection_string(connection_string)

def ensure_containers(blob_service_client):
    """Ensure required containers exist."""
    containers = {
        'videos': 'video-uploads',
        'audio': 'audio-temp',
        'transcripts': 'transcriptions'
    }
    
    for container_name in containers.values():
        try:
            blob_service_client.create_container(container_name)
            logger.info(f"Created container: {container_name}")
        except ResourceExistsError:
            logger.debug(f"Container already exists: {container_name}")
    
    return containers

def list_videos_in_container(blob_service_client, container_name):
    """List all MP4 files in the specified container."""
    container_client = blob_service_client.get_container_client(container_name)
    return [blob for blob in container_client.list_blobs() if blob.name.lower().endswith('.mp4')]

def process_video(video_name, client, blob_service_client, containers):
    """Process a single video file directly from Azure Storage."""
    try:
        # Get container clients
        video_container = blob_service_client.get_container_client(containers['videos'])
        audio_container = blob_service_client.get_container_client(containers['audio'])
        
        # Create temporary local paths for processing
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_video_path = temp_dir / video_name
        temp_audio_path = temp_video_path.with_suffix('.mp3')
        
        # Download video for processing
        logger.info(f"Downloading video: {video_name}")
        with open(temp_video_path, "wb") as video_file:
            video_file.write(video_container.download_blob(video_name).readall())
        
        logger.info(f"Starting processing of {video_name}")
        logger.info(f"Extracting audio...")
        if not extract_audio(str(temp_video_path), str(temp_audio_path)):
            return False
            
        # Upload audio for tracking
        with temp_audio_path.open('rb') as audio_file:
            audio_container.upload_blob(temp_audio_path.name, audio_file, overwrite=True)
        
        logger.info("Splitting audio into 10-minute chunks...")
        audio_chunks = split_audio(temp_audio_path)
        
        full_transcript = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            transcribe_func = partial(transcribe_with_retry, client=client)
            results = list(executor.map(transcribe_func, audio_chunks))
            
            full_transcript = [t for t in results if t is not None]
            
            for chunk_path in audio_chunks:
                os.remove(chunk_path)
                logger.debug(f"Removed chunk file: {chunk_path}")
        
        if full_transcript:
            # Save transcript to Azure
            transcript_container = blob_service_client.get_container_client(containers['transcripts'])
            output_filename = f"{video_name}_cleaned.docx"
            
            doc = Document()
            text = '\n'.join(full_transcript)
            sentences = split_into_sentences(text)
            for sentence in sentences:
                doc.add_paragraph(sentence)
            
            # Save locally temporarily
            temp_doc_path = Path("temp_doc.docx")
            doc.save(str(temp_doc_path))
            
            # Upload to Azure
            with temp_doc_path.open('rb') as doc_file:
                transcript_container.upload_blob(output_filename, doc_file, overwrite=True)
            
            # Clean up local temp file
            temp_doc_path.unlink()
            logger.info(f"Transcription saved to Azure: {output_filename}")
        
        # Clean up all temporary files and Azure blobs
        temp_video_path.unlink()
        temp_audio_path.unlink()
        logger.debug("Cleaned up local temporary files")
            
        # Clean up from Azure Storage
        audio_container.delete_blob(temp_audio_path.name)
        logger.debug(f"Cleaned up Azure audio file: {temp_audio_path.name}")
        
        video_container.delete_blob(video_name)
        logger.debug(f"Cleaned up Azure video file: {video_name}")
        
        # Remove temp directory if empty
        if not any(temp_dir.iterdir()):
            temp_dir.rmdir()
            logger.debug("Removed temporary directory")
        
        return True

    except Exception as e:
        logger.error(f"Error processing video {video_name}: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Initializing services...")
    # Create temp directory if it doesn't exist
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

@app.get("/", response_model=dict)
async def root():
    """Root endpoint to check if the API is running"""
    return {"status": "running", "message": "Video processing API is operational"}

@app.get("/videos", response_model=List[str])
async def list_videos():
    """List all available videos in Azure Storage"""
    try:
        blob_service_client = get_blob_service_client()
        containers = ensure_containers(blob_service_client)
        videos = list_videos_in_container(blob_service_client, containers['videos'])
        return [video.name for video in videos]
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process/{video_name}", response_model=ProcessingStatus)
async def process_video_endpoint(video_name: str, background_tasks: BackgroundTasks):
    """
    Start processing a video
    """
    try:
        client = OpenAI()
        blob_service_client = get_blob_service_client()
        containers = ensure_containers(blob_service_client)
        
        # Check if video exists
        video_container = blob_service_client.get_container_client(containers['videos'])
        if not any(b.name == video_name for b in video_container.list_blobs()):
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Add processing to background tasks
        background_tasks.add_task(
            process_video,
            video_name,
            client,
            blob_service_client,
            containers
        )
        
        return {
            "status": "processing",
            "message": f"Processing of {video_name} started"
        }
    
    except Exception as e:
        logger.error(f"Error initiating video processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/request", response_model=UploadResponse)
async def request_upload(filename: str):
    """Get a pre-signed URL for uploading a video file"""
    try:
        return generate_sas_token(filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/upload/status/{filename}", response_model=UploadStatus)
async def get_upload_status(filename: str):
    """Check the status of a video upload and processing"""
    try:
        return check_file_status(filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
