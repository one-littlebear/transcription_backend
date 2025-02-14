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
    temp_files = []  # Keep track of all temporary files
    try:
        # Get container clients
        logger.info(f"Starting video processing pipeline for: {video_name}")
        video_container = blob_service_client.get_container_client(containers['videos'])
        audio_container = blob_service_client.get_container_client(containers['audio'])
        
        # Create temporary local paths for processing
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_video_path = temp_dir / video_name
        temp_audio_path = temp_video_path.with_suffix('.mp3')
        temp_files.extend([temp_video_path, temp_audio_path])
        
        # Download video for processing
        logger.info(f"Downloading video from Azure Storage: {video_name}")
        with open(temp_video_path, "wb") as video_file:
            video_file.write(video_container.download_blob(video_name).readall())
        logger.info(f"Video download completed: {video_name}")
        
        logger.info(f"Starting audio extraction from video: {video_name}")
        if not extract_audio(str(temp_video_path), str(temp_audio_path)):
            logger.error(f"Audio extraction failed for: {video_name}")
            return False
        logger.info(f"Audio extraction completed: {video_name}")
            
        # Upload audio for tracking
        logger.info(f"Uploading extracted audio to Azure Storage: {temp_audio_path.name}")
        with temp_audio_path.open('rb') as audio_file:
            audio_container.upload_blob(temp_audio_path.name, audio_file, overwrite=True)
        logger.info(f"Audio upload completed: {temp_audio_path.name}")
        
        # Keep track of audio chunks for cleanup
        logger.info(f"Starting audio splitting into 10-minute chunks: {video_name}")
        audio_chunks = split_audio(temp_audio_path)
        temp_files.extend(audio_chunks)  # Add chunks to cleanup list
        logger.info(f"Created {len(audio_chunks)} audio chunks for processing")
        
        logger.info("Starting transcription of audio chunks...")
        full_transcript = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            transcribe_func = partial(transcribe_with_retry, client=client)
            results = list(executor.map(transcribe_func, audio_chunks))
            
            full_transcript = [t for t in results if t is not None]
            logger.info(f"Transcription completed: {len(full_transcript)} chunks successfully transcribed")
            
            for chunk_path in audio_chunks:
                os.remove(chunk_path)
        
        if full_transcript:
            # Save transcript to Azure
            logger.info("Creating Word document from transcription...")
            transcript_container = blob_service_client.get_container_client(containers['transcripts'])
            output_filename = f"{video_name}_cleaned.docx"
            
            doc = Document()
            text = '\n'.join(full_transcript)
            sentences = split_into_sentences(text)
            for sentence in sentences:
                doc.add_paragraph(sentence)
            
            # Save locally temporarily
            temp_doc_path = Path("temp_doc.docx")
            temp_files.append(temp_doc_path)  # Add doc to cleanup list
            doc.save(str(temp_doc_path))
            
            # Upload to Azure
            logger.info(f"Uploading transcript document to Azure Storage: {output_filename}")
            with temp_doc_path.open('rb') as doc_file:
                transcript_container.upload_blob(output_filename, doc_file, overwrite=True)
            logger.info(f"Transcript upload completed: {output_filename}")
            
            # Enhanced cleanup process
            logger.info("Starting cleanup process...")
            
            # Clean up all temporary local files
            for temp_file in temp_files:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                        logger.info(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    logger.error(f"Error removing temporary file {temp_file}: {e}")
            
            # Clean up from Azure Storage
            try:
                audio_container.delete_blob(temp_audio_path.name)
                logger.info(f"Removed audio blob: {temp_audio_path.name}")
            except Exception as e:
                logger.error(f"Error removing audio blob {temp_audio_path.name}: {e}")
            
            try:
                video_container.delete_blob(video_name)
                logger.info(f"Removed video blob: {video_name}")
            except Exception as e:
                logger.error(f"Error removing video blob {video_name}: {e}")
            
            # Remove temp directory if empty
            try:
                if temp_dir.exists() and not any(temp_dir.iterdir()):
                    temp_dir.rmdir()
                    logger.info("Removed empty temp directory")
            except Exception as e:
                logger.error(f"Error removing temp directory: {e}")
        
        logger.info(f"Video processing completed successfully for: {video_name}")
        return True

    except Exception as e:
        logger.error(f"Error processing video {video_name}: {e}")
        # Attempt cleanup even if processing failed
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.info(f"Cleaned up temporary file after error: {temp_file}")
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup of {temp_file}: {cleanup_error}")
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
