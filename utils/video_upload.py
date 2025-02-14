import os
from pathlib import Path
import logging
from azure.storage.blob import BlobClient, BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
from dotenv import load_dotenv
import re
from typing import Optional
from pydantic import BaseModel

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UploadResponse(BaseModel):
    upload_url: str
    sas_token: str
    blob_url: str
    cleaned_filename: str

class UploadStatus(BaseModel):
    status: str
    message: str
    transcript_url: str | None = None

def clean_filename(filename: str) -> str:
    """
    Clean filename by:
    1. Converting to lowercase
    2. Replacing spaces with underscores
    3. Removing special characters
    4. Ensuring it ends with .mp4
    """
    # Get the base name and extension
    base_name, ext = os.path.splitext(filename.lower())
    
    # Remove special characters and replace spaces
    cleaned_name = re.sub(r'[^\w\s-]', '', base_name)  # Remove special chars except spaces and hyphens
    cleaned_name = re.sub(r'[-\s]+', '_', cleaned_name)  # Replace spaces and hyphens with underscore
    
    # Ensure valid extension
    valid_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
    if ext.lower() not in valid_extensions:
        ext = '.mp4'  # Default to .mp4 if not a valid video extension
    
    # Ensure the filename isn't too long (Azure has a 1024 char limit)
    max_length = 100  # Reasonable length for filenames
    if len(cleaned_name) > max_length:
        cleaned_name = cleaned_name[:max_length]
    
    return f"{cleaned_name}{ext}"

def generate_sas_token(blob_name: str, container_name: str = "video-uploads") -> UploadResponse:
    """Generate a SAS token for client-side upload."""
    try:
        # Clean the blob name first
        cleaned_blob_name = clean_filename(blob_name)
        
        # Get account details from connection string
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        account_key = dict(item.split('=', 1) for item in conn_str.split(';'))['AccountKey']
        account_name = blob_service_client.account_name
        
        # Generate SAS token
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=cleaned_blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(write=True, create=True),
            expiry=datetime.utcnow() + timedelta(minutes=30)
        )
        
        # Generate the full URL for the client
        blob_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{cleaned_blob_name}"
        
        return UploadResponse(
            upload_url=f"{blob_url}?{sas_token}",
            sas_token=sas_token,
            blob_url=blob_url,
            cleaned_filename=cleaned_blob_name
        )
        
    except Exception as e:
        logger.error(f"Failed to generate SAS token: {str(e)}")
        raise

def check_file_status(filename: str) -> UploadStatus:
    """Check the status of a file's processing."""
    try:
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        
        # Check if transcript exists
        transcript_container = blob_service_client.get_container_client("transcriptions")
        transcript_name = f"{filename}_cleaned.docx"
        
        if any(blob.name == transcript_name for blob in transcript_container.list_blobs()):
            # Get transcript URL with SAS token
            transcript_blob = transcript_container.get_blob_client(transcript_name)
            sas_token = generate_blob_sas(
                account_name=blob_service_client.account_name,
                container_name="transcriptions",
                blob_name=transcript_name,
                account_key=dict(item.split('=', 1) for item in conn_str.split(';'))['AccountKey'],
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=1)
            )
            transcript_url = f"{transcript_blob.url}?{sas_token}"
            return UploadStatus(
                status="completed",
                message="File processed successfully",
                transcript_url=transcript_url
            )
        
        # Check if video is still in processing
        video_container = blob_service_client.get_container_client("video-uploads")
        if any(blob.name == filename for blob in video_container.list_blobs()):
            return UploadStatus(
                status="processing",
                message="File is being processed"
            )
            
        return UploadStatus(
            status="not_found",
            message="File not found"
        )
        
    except Exception as e:
        logger.error(f"Error checking file status: {e}")
        raise

def get_upload_url(filename: str) -> dict:
    """Get a secure upload URL for the frontend."""
    try:
        return generate_sas_token(filename)
    except Exception as e:
        logger.error(f"Error generating upload URL: {str(e)}")
        raise

def upload_file(file_path: Path, container_name: str = "video-uploads") -> bool:
    """Upload a file to Azure Blob Storage using the simplest modern approach."""
    try:
        # Clean the filename before upload
        cleaned_filename = clean_filename(file_path.name)
        
        # Create blob client directly - one connection, no container client needed
        blob = BlobClient.from_connection_string(
            conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
            container_name=container_name,
            blob_name=cleaned_filename
        )
        
        file_size = file_path.stat().st_size
        logger.info(f"Starting upload of {cleaned_filename} ({file_size / (1024*1024):.2f} MB)")
        
        # Upload with standard blob tier for better performance
        with open(file_path, "rb") as data:
            blob.upload_blob(
                data,
                overwrite=True,
                max_concurrency=4,
                timeout=600  # 10 minutes
            )
            
        logger.info(f"Successfully uploaded {cleaned_filename}")
        return True
        
    except Exception as e:
        logger.error(f"Upload failed for {file_path.name}: {str(e)}")
        return False

def main():
    upload_dir = Path("videos")
    upload_dir.mkdir(exist_ok=True)
    
    files = list(upload_dir.glob("*.*"))
    if not files:
        logger.info(f"No files found in {upload_dir}. Please add files and try again.")
        return
    
    successful = 0
    for file_path in files:
        if upload_file(file_path):
            successful += 1
            
    logger.info(f"Upload complete! Successfully uploaded {successful} out of {len(files)} files")

if __name__ == "__main__":
    main()