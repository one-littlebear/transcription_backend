import os
import logging
from pathlib import Path
from moviepy import VideoFileClip

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_audio(video_path: str, output_path: str) -> bool:
    """Extract audio from video file and save as MP3.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str): Path where to save the output MP3 file
    
    Returns:
        bool: True if successful, False otherwise
    """
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
        print(f"Error extracting audio: {e}")
        return False

if __name__ == "__main__":
    # Get input from user
    video_file = "videos/How Startup Funding works： Seed money, Angel Investors and Venture Capitalists explained.mp4"
    output_file = "output/test.mp3"
    
    # If no output path specified, use input filename with .mp3 extension
    if not output_file:
        output_file = str(Path(video_file).with_suffix('.mp3'))
    
    if extract_audio(video_file, output_file):
        print(f"✅ Audio extracted successfully to: {output_file}")
    else:
        print("❌ Audio extraction failed")
