import logging
from pathlib import Path
from typing import Optional, List
from moviepy import VideoFileClip
from pydub import AudioSegment
from openai import OpenAI
from docx import Document
from concurrent.futures import ThreadPoolExecutor
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

logger = logging.getLogger(__name__)

class LocalVideoProcessor:
    def __init__(self):
        # Create a thread pool for CPU-bound tasks
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Ensure required directories exist
        self.videos_dir = Path("videos")
        self.output_dir = Path("output")
        self.temp_dir = Path("temp")
        
        # Create directories if they don't exist
        self.videos_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

    @staticmethod
    def extract_audio(video_path: Path, output_path: Path) -> bool:
        """Extract audio from video file and save as MP3."""
        try:
            video = VideoFileClip(str(video_path))
            try:
                audio = video.audio
                audio.write_audiofile(str(output_path))
                return True
            finally:
                if hasattr(video, 'close'):
                    video.close()
                if hasattr(video.audio, 'close'):
                    video.audio.close()
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return False

    async def process_video(self, video_name: str, client: OpenAI) -> bool:
        """Process a single video file from local directory."""
        temp_files = []
        try:
            start_time = time.time()
            
            # Setup paths
            video_path = self.videos_dir / video_name
            if not video_path.exists():
                logger.error(f"Video file not found: {video_path}")
                return False

            temp_audio_path = self.temp_dir / f"{video_name}.mp3"
            temp_files.append(temp_audio_path)

            # Extract audio
            extract_start = time.time()
            logger.info("Starting audio extraction")
            success = self.extract_audio(video_path, temp_audio_path)
            if not success:
                logger.error("Failed to extract audio")
                return False
            logger.info(f"Audio extraction took {time.time() - extract_start:.2f} seconds")

            # Split and transcribe
            split_start = time.time()
            audio = AudioSegment.from_mp3(temp_audio_path)
            chunk_length = 5 * 60 * 1000  # 5 minutes
            chunks = []
            
            # Create chunks
            for i in range(0, len(audio), chunk_length):
                chunk_path = self.temp_dir / f"chunk_{i//chunk_length}.mp3"
                audio[i:i + chunk_length].export(chunk_path, format="mp3")
                chunks.append(chunk_path)
                temp_files.append(chunk_path)
            
            logger.info(f"Split into {len(chunks)} chunks")

            # Transcribe chunks in parallel
            sem = asyncio.Semaphore(10)
            
            async def transcribe_chunk(chunk_path):
                async with sem:
                    return await self.transcribe_with_retry(chunk_path, client)

            # Create and run tasks in parallel
            transcribe_tasks = [transcribe_chunk(chunk) for chunk in chunks]
            logger.info(f"Starting parallel transcription of {len(chunks)} chunks")
            results = await asyncio.gather(*transcribe_tasks)
            logger.info("Completed all transcriptions")
            full_transcript = [t for t in results if t is not None]

            # Create and save document
            if full_transcript:
                doc = Document()
                text = '\n'.join(full_transcript)
                sentences = split_into_sentences(text)
                for sentence in sentences:
                    doc.add_paragraph(sentence)

                output_filename = f"{video_name}_transcript.docx"
                output_path = self.output_dir / output_filename
                doc.save(str(output_path))
                logger.info(f"Saved transcript to: {output_path}")

            # Cleanup
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()

            total_time = time.time() - start_time
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            return True

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            # Cleanup on error
            for temp_file in temp_files:
                if isinstance(temp_file, Path) and temp_file.exists():
                    temp_file.unlink()
            return False

    async def transcribe_with_retry(self, audio_path: Path, client: OpenAI, max_retries: int = 3) -> Optional[str]:
        """Transcribe audio file with retry logic"""
        async def _transcribe():
            logger.info(f"Starting transcription of {audio_path}")
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: client.audio.transcriptions.create(
                    model="whisper-1",
                    file=open(audio_path, "rb"),
                    response_format="text"
                )
            )
            logger.info(f"Completed transcription of {audio_path}")
            return result

        for attempt in range(max_retries):
            try:
                transcript = await _transcribe()
                return transcript
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {audio_path}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Final retry failed for {audio_path}: {e}")
                    return None
                await asyncio.sleep(math.pow(2, attempt))

    def list_videos(self) -> List[str]:
        """List all MP4 files in the videos directory."""
        return [f.name for f in self.videos_dir.glob("*.mp4")]

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import asyncio
    
    # Load environment variables
    load_dotenv()
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Initialize video processor
    processor = LocalVideoProcessor()
    
    async def test_video_processing():
        '''Test the video processing pipeline'''
        try:
            # List available videos
            print("\nAvailable videos in 'videos' directory:")
            videos = processor.list_videos()
            for i, video in enumerate(videos, 1):
                print(f"{i}. {video}")
            
            if not videos:
                print("No MP4 files found in the 'videos' directory.")
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
            success = await processor.process_video(video_name, client)
            
            if success:
                print(f"\nSuccessfully processed {video_name}")
                print(f"Check the 'output' directory for the transcript.")
            else:
                print(f"\nFailed to process {video_name}")
                
        except Exception as e:
            print(f"Error during testing: {e}")
    
    # Run the test
    print("Local Video Processing Tool")
    print("==========================")
    asyncio.run(test_video_processing()) 