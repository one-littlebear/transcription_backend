import logging
from pathlib import Path
from typing import Optional, List
from moviepy import VideoFileClip
from pydub import AudioSegment
from openai import OpenAI
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
from docx import Document
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import time
import math
import os
import asyncio
from utils.cleanup import split_into_sentences

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Suppress Azure SDK logging
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.blob_service_client = self._get_blob_service_client()
        self.containers = self._ensure_containers()
        # Create a thread pool for CPU-bound tasks
        self.executor = ThreadPoolExecutor(max_workers=3)

    def _get_blob_service_client(self):
        """Initialize Azure Blob Service Client."""
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("Azure Storage connection string not found")
        return BlobServiceClient.from_connection_string(connection_string)

    def _ensure_containers(self):
        """Ensure required containers exist."""
        containers = {
            'videos': 'video-uploads',
            'audio': 'audio-temp',
            'transcripts': 'transcriptions'
        }
        
        for container_name in containers.values():
            try:
                self.blob_service_client.create_container(container_name)
                logger.info(f"Created container: {container_name}")
            except ResourceExistsError:
                logger.debug(f"Container already exists: {container_name}")
        
        return containers

    @staticmethod
    def extract_audio(video_path, output_path):
        """Extract audio from video file and save as MP3."""
        try:
            video = VideoFileClip(video_path)
            try:
                audio = video.audio
                audio.write_audiofile(output_path)
                return True
            finally:
                if hasattr(video, 'close'):
                    video.close()
                if hasattr(video.audio, 'close'):
                    video.audio.close()
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return False

    @staticmethod
    def split_audio(audio_path):
        """Split audio file into 10-minute chunks."""
        try:
            audio = AudioSegment.from_mp3(audio_path)
            chunk_length = 10 * 60 * 1000  # 10 minutes in milliseconds
            chunks = []
            
            for i in range(0, len(audio), chunk_length):
                chunk = audio[i:i + chunk_length]
                chunk_path = f"{audio_path.stem}_chunk_{i//chunk_length}{audio_path.suffix}"
                chunk_path = audio_path.parent / chunk_path
                chunk.export(str(chunk_path), format="mp3")
                chunks.append(chunk_path)
                
            return chunks
        except Exception as e:
            logger.error(f"Error splitting audio: {e}")
            return []

    async def process_video(self, video_name: str, client: OpenAI) -> bool:
        """Process a single video file from Azure Storage."""
        temp_files = []
        try:
            # Setup container clients and temp directory
            video_container = self.blob_service_client.get_container_client(self.containers['videos'])
            audio_container = self.blob_service_client.get_container_client(self.containers['audio'])
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            # Create temp paths
            temp_video_path = temp_dir / video_name
            temp_audio_path = temp_video_path.with_suffix('.mp3')
            temp_files.extend([temp_video_path, temp_audio_path])

            # Download video (I/O operation, can be async)
            logger.info(f"Processing video: {video_name}")
            video_data = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: video_container.download_blob(video_name).readall()
            )
            
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: temp_video_path.write_bytes(video_data)
            )

            # Extract audio (CPU-bound, run in thread pool)
            if not await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.extract_audio,
                str(temp_video_path),
                str(temp_audio_path)
            ):
                return False

            # Upload audio (I/O operation)
            audio_data = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: temp_audio_path.read_bytes()
            )
            
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: audio_container.upload_blob(temp_audio_path.name, audio_data, overwrite=True)
            )

            # Split audio (CPU-bound)
            audio_chunks = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.split_audio,
                temp_audio_path
            )
            temp_files.extend(audio_chunks)

            # Transcribe chunks (I/O bound, can be parallel)
            transcribe_tasks = []
            for chunk_path in audio_chunks:
                task = self.transcribe_with_retry(chunk_path, client)
                transcribe_tasks.append(task)
            
            results = await asyncio.gather(*transcribe_tasks)
            full_transcript = [t for t in results if t is not None]

            if full_transcript:
                # Create document
                transcript_container = self.blob_service_client.get_container_client(self.containers['transcripts'])
                output_filename = f"{video_name}_cleaned.docx"
                
                # Create document in memory
                doc = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._create_document,
                    full_transcript
                )
                
                # Save temporarily and upload
                temp_doc_path = Path("temp_doc.docx")
                temp_files.append(temp_doc_path)
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    doc.save,
                    str(temp_doc_path)
                )

                doc_data = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: temp_doc_path.read_bytes()
                )
                
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: transcript_container.upload_blob(output_filename, doc_data, overwrite=True)
                )

            # Cleanup
            await self._cleanup_files(temp_files, video_name, temp_audio_path.name)
            return True

        except Exception as e:
            logger.error(f"Error processing video {video_name}: {e}")
            await self._cleanup_files(temp_files)
            return False

    async def _cleanup_files(self, temp_files, video_name=None, audio_name=None):
        """Clean up temporary files and blobs."""
        for temp_file in temp_files:
            try:
                if isinstance(temp_file, Path) and temp_file.exists():
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        temp_file.unlink
                    )
            except Exception as e:
                logger.error(f"Error cleaning up {temp_file}: {e}")

        if video_name and audio_name:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.blob_service_client.get_container_client(self.containers['videos']).delete_blob(video_name)
                )
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.blob_service_client.get_container_client(self.containers['audio']).delete_blob(audio_name)
                )
            except Exception as e:
                logger.error(f"Error cleaning up blobs: {e}")

    def _create_document(self, transcript_list):
        """Create a Word document from transcripts"""
        doc = Document()
        text = '\n'.join(transcript_list)
        sentences = split_into_sentences(text)
        for sentence in sentences:
            doc.add_paragraph(sentence)
        return doc

    async def transcribe_with_retry(self, audio_path: Path, client: OpenAI, max_retries: int = 3) -> Optional[str]:
        """Async version of transcribe with retry"""
        for attempt in range(max_retries):
            try:
                transcript = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_path.open("rb"),
                        response_format="text"
                    )
                )
                return transcript
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {audio_path}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Final retry failed for {audio_path}: {e}")
                    return None
                await asyncio.sleep(math.pow(2, attempt))

    def list_videos(self) -> List[str]:
        """List all MP4 files in the videos container."""
        container_client = self.blob_service_client.get_container_client(self.containers['videos'])
        return [blob.name for blob in container_client.list_blobs() if blob.name.lower().endswith('.mp4')]

# Add this test section at the end of the file
if __name__ == "__main__":
    """
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Initialize video processor
    processor = VideoProcessor()
    
    def test_video_processing():
        '''Test the video processing pipeline'''
        try:
            # List available videos
            print("Available videos:")
            videos = processor.list_videos()
            for i, video in enumerate(videos, 1):
                print(f"{i}. {video}")
            
            if not videos:
                print("No videos found in storage.")
                return
            
            # Let user select a video
            while True:
                try:
                    choice = int(input("\nEnter the number of the video to process (0 to exit): ")) - 1
                    if choice == -1:
                        return
                    if 0 <= choice < len(videos):
                        break
                    print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
            
            video_name = videos[choice]
            print(f"\nProcessing video: {video_name}")
            
            # Process the video
            success = processor.process_video(video_name, client)
            
            if success:
                print(f"\nSuccessfully processed {video_name}")
                print("Check the 'transcripts' container in Azure Storage for the output.")
            else:
                print(f"\nFailed to process {video_name}")
                
        except Exception as e:
            print(f"Error during testing: {e}")
    
    # Run the test
    print("Video Processing Test Tool")
    print("=========================")
    test_video_processing() 
    """
    pass