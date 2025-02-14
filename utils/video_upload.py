import os
from pathlib import Path
import logging
from azure.storage.blob import BlobClient, BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sas_token(blob_name: str, container_name: str = "video-uploads") -> dict:
    """Generate a SAS token for client-side upload."""
    try:
        # Get account details from connection string
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        account_key = dict(item.split('=', 1) for item in conn_str.split(';'))['AccountKey']
        account_name = blob_service_client.account_name
        
        # Generate SAS token
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(write=True, create=True),
            expiry=datetime.utcnow() + timedelta(minutes=30)
        )
        
        # Generate the full URL for the client
        blob_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}"
        
        return {
            "sasToken": sas_token,
            "uploadUrl": f"{blob_url}?{sas_token}",
            "blobUrl": blob_url
        }
        
    except Exception as e:
        logger.error(f"Failed to generate SAS token: {str(e)}")
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
        # Create blob client directly - one connection, no container client needed
        blob = BlobClient.from_connection_string(
            conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
            container_name=container_name,
            blob_name=file_path.name
        )
        
        file_size = file_path.stat().st_size
        logger.info(f"Starting upload of {file_path.name} ({file_size / (1024*1024):.2f} MB)")
        
        # Upload with standard blob tier for better performance
        with open(file_path, "rb") as data:
            blob.upload_blob(
                data,
                overwrite=True,
                max_concurrency=4,
                timeout=600  # 10 minutes
            )
            
        logger.info(f"Successfully uploaded {file_path.name}")
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