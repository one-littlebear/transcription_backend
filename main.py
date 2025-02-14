import logging
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
from typing import Optional
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceExistsError

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

def main():
    logger.info("Starting video processing script")
    client = OpenAI()
    
    # Initialize Azure Storage
    blob_service_client = get_blob_service_client()
    containers = ensure_containers(blob_service_client)
    
    # List videos directly from Azure Storage
    video_files = list_videos_in_container(blob_service_client, containers['videos'])
    logger.info(f"Found {len(video_files)} MP4 files to process in Azure Storage")
    
    for video_blob in video_files:
        logger.info(f"\nProcessing {video_blob.name}...")
        process_video(video_blob.name, client, blob_service_client, containers)
    
    logger.info("Video processing completed")

if __name__ == "__main__":
    main()
