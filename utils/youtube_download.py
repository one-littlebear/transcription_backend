import yt_dlp
import os
import re

def clean_title(title):
    """
    Cleans up video title by removing emojis, extra whitespace, and special characters.
    
    Args:
        title (str): The original video title
    Returns:
        str: Cleaned up title
    """
    # Remove emojis and special characters
    title = re.sub(r'[^\w\s-]', '', title)
    # Replace multiple spaces with single space and strip
    title = ' '.join(title.split())
    # Replace spaces with underscores
    title = title.replace(' ', '_')
    return title

def download_video(url, output_path="videos"):
    """
    Downloads a YouTube video and saves it to the specified output path.
    
    Args:
        url (str): The YouTube video URL
        output_path (str): The folder where videos will be saved (default: 'videos')
    """
    try:
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        # Configure yt-dlp options with custom output template
        ydl_opts = {
            'format': 'best',  # Download best quality
            'outtmpl': f'{output_path}/%(title)s.%(ext)s',  # Output template
            'progress_hooks': [lambda d: print(f"Downloading: {d['_percent_str']} of {d['_total_bytes_str']}")],
        }
        
        # Create a yt-dlp object and download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=True)
            
            # Get the original filename and extension
            original_title = info['title']
            ext = info['ext']
            original_path = os.path.join(output_path, f"{original_title}.{ext}")
            
            # Create new filename with cleaned title
            cleaned_title = clean_title(original_title)
            new_path = os.path.join(output_path, f"{cleaned_title}.{ext}")
            
            # Rename the file
            if os.path.exists(original_path):
                os.rename(original_path, new_path)
                print(f"File renamed to: {cleaned_title}.{ext}")
            
        print(f"\nDownload completed! Video saved to: {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Example usage
    #video_url = "https://www.youtube.com/watch?v=zWecbmgHNVY"
    #download_video(video_url)
    pass