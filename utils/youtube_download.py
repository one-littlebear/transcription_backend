import yt_dlp
import os

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
            
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'best',  # Download best quality
            'outtmpl': f'{output_path}/%(title)s.%(ext)s',  # Output template
            'progress_hooks': [lambda d: print(f"Downloading: {d['_percent_str']} of {d['_total_bytes_str']}")],
        }
        
        # Create a yt-dlp object and download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            print(f"Downloading: {info['title']}")
            print(f"Quality: {info['format']}") if 'format' in info else None
            
            # Download the video
            ydl.download([url])
            
        print(f"\nDownload completed! Video saved to: {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Example usage
    video_url = "https://www.youtube.com/watch?v=mM4NWS3o2Lo"
    download_video(video_url)
