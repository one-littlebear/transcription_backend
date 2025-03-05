import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel
from video_processor import VideoProcessor
from utils.video_upload import generate_sas_token, check_file_status, UploadResponse, UploadStatus
import os
import asyncio
from datetime import datetime

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Suppress Azure SDK logging
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class ProcessingStatus(BaseModel):
    status: str
    message: str
    file_type: str = "video"  # Add file type to track whether it's video or audio

class TaskManager:
    def __init__(self):
        self.processing_tasks: Dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def start_task(self, video_name: str) -> bool:
        async with self._lock:
            if video_name in self.processing_tasks:
                return False
            self.processing_tasks[video_name] = {
                "status": "processing",
                "start_time": datetime.now(),
                "completed": False
            }
            return True

    async def complete_task(self, video_name: str):
        async with self._lock:
            if video_name in self.processing_tasks:
                self.processing_tasks[video_name]["completed"] = True
                self.processing_tasks[video_name]["status"] = "completed"

    async def fail_task(self, video_name: str, error: str):
        async with self._lock:
            if video_name in self.processing_tasks:
                self.processing_tasks[video_name]["status"] = "failed"
                self.processing_tasks[video_name]["error"] = error

    def get_task_status(self, video_name: str) -> dict:
        return self.processing_tasks.get(video_name, {"status": "not_found"})

app = FastAPI(
    title="Video Processing API",
    description="API for processing videos, extracting audio, and generating transcriptions",
    version="1.0.0"
)

# Configure CORS
allowed_origins = [
    os.getenv("FRONTEND_URL", "http://localhost:3000"),  # Production URL
    "http://localhost:3000",  # Development URL
]

if os.getenv("ADDITIONAL_CORS_ORIGINS"):
    # Add any additional origins from environment variable
    # Format should be comma-separated URLs
    additional_origins = os.getenv("ADDITIONAL_CORS_ORIGINS").split(",")
    allowed_origins.extend([origin.strip() for origin in additional_origins])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize shared resources
video_processor = VideoProcessor()
task_manager = TaskManager()

async def process_video_task(video_name: str, client: OpenAI):
    """Background task for processing videos"""
    try:
        success = await video_processor.process_video(video_name, client)
        if success:
            await task_manager.complete_task(video_name)
        else:
            await task_manager.fail_task(video_name, "Processing failed")
    except Exception as e:
        logger.error(f"Error processing video {video_name}: {e}")
        await task_manager.fail_task(video_name, str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Initializing services...")
    # Ensure required environment variables are set
    required_vars = [
        "AZURE_STORAGE_CONNECTION_STRING",
        "OPENAI_API_KEY",
        "FRONTEND_URL"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    Path("temp").mkdir(exist_ok=True)

@app.get("/", response_model=dict)
async def root():
    """Root endpoint to check if the API is running"""
    return {"status": "running", "message": "Video processing API is operational"}

@app.get("/videos", response_model=List[str])
async def list_videos():
    """List all available videos in Azure Storage"""
    try:
        return video_processor.list_videos()
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process/{file_name}", response_model=ProcessingStatus)
async def process_file_endpoint(file_name: str, background_tasks: BackgroundTasks):
    """Start processing a video or audio file"""
    try:
        # Check if file exists
        if file_name not in video_processor.list_files():
            raise HTTPException(status_code=404, detail="File not found")

        # Determine file type
        file_type = get_file_type(file_name)

        # Check if file is already being processed
        can_start = await task_manager.start_task(file_name)
        if not can_start:
            status = task_manager.get_task_status(file_name)
            if status["status"] == "processing":
                return {
                    "status": "processing",
                    "message": f"{file_type.capitalize()} {file_name} is already being processed",
                    "file_type": file_type
                }
            elif status["status"] == "completed":
                return {
                    "status": "completed",
                    "message": f"{file_type.capitalize()} {file_name} has already been processed",
                    "file_type": file_type
                }

        # Initialize OpenAI client
        client = OpenAI()
        
        # Add processing to background tasks
        if file_type == 'audio':
            background_tasks.add_task(video_processor.process_audio, file_name, client)
        else:
            background_tasks.add_task(process_video_task, file_name, client)
        
        return {
            "status": "processing",
            "message": f"Processing of {file_name} started",
            "file_type": file_type
        }
    
    except Exception as e:
        logger.error(f"Error initiating file processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add supported file types
SUPPORTED_FILE_TYPES = {
    'video': ['.mp4', '.avi', '.mov', '.mkv'],
    'audio': ['.mp3', '.wav']
}

def get_file_type(filename: str) -> str:
    """Determine if the file is a video or audio file based on extension"""
    ext = Path(filename).suffix.lower()
    if ext in SUPPORTED_FILE_TYPES['video']:
        return 'video'
    elif ext in SUPPORTED_FILE_TYPES['audio']:
        return 'audio'
    else:
        raise ValueError(f"Unsupported file type: {ext}")

@app.post("/upload/request", response_model=UploadResponse)
async def request_upload(filename: str):
    """Get a pre-signed URL for uploading a video or audio file"""
    try:
        # Validate file type
        file_type = get_file_type(filename)
        if file_type not in ['video', 'audio']:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {SUPPORTED_FILE_TYPES}"
            )
        return generate_sas_token(filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/upload/status/{filename}", response_model=UploadStatus)
async def get_upload_status(filename: str):
    """Check the status of a video upload and processing"""
    try:
        status = task_manager.get_task_status(filename)
        if status["status"] == "not_found":
            return check_file_status(filename)
        
        # Get transcript URL if completed
        transcript_url = None
        if status["status"] == "completed":
            try:
                # Use the check_file_status function to get the URL with SAS token
                completed_status = check_file_status(filename)
                transcript_url = completed_status.transcript_url
            except Exception as e:
                logger.warning(f"Failed to generate transcript URL: {e}")
        
        return UploadStatus(
            status=status["status"],
            message=status.get("error", "Processing in progress"),
            transcript_url=transcript_url
        )
    except Exception as e:
        logger.error(f"Error in status check: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 