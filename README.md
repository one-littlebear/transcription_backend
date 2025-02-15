# Video Transcription API

A powerful FastAPI-based service that processes videos, extracts audio, and generates transcriptions using OpenAI's Whisper model. This project is designed to handle video uploads, process them in the background, and provide transcription results through a RESTful API.

## Features

- 🎥 Video upload to Azure Blob Storage
- 🔊 Audio extraction from videos
- 📝 Transcription generation using OpenAI's Whisper model
- 🔄 Background task processing
- 📊 Status tracking for uploads and processing
- 🔐 Secure file handling with SAS tokens

## Prerequisites

- Python 3.11 or higher
- Azure Storage Account
- OpenAI API key
- FFmpeg (for video processing)

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
AZURE_STORAGE_CONNECTION_STRING=your_azure_storage_connection_string
OPENAI_API_KEY=your_openai_api_key
FRONTEND_URL=your_frontend_url
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd video-transcription-api
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## API Endpoints

### Root Endpoint
- **GET /** 
  - Check if the API is running
  - Response: `{"status": "running", "message": "Video processing API is operational"}`

### Video Management
- **GET /videos**
  - List all available videos in Azure Storage
  - Response: Array of video filenames

### Video Processing
- **POST /process/{video_name}**
  - Start processing a video
  - Response: Processing status and message
  - Example response:
    ```json
    {
        "status": "processing",
        "message": "Processing of video.mp4 started"
    }
    ```

### Upload Management
- **POST /upload/request**
  - Get a pre-signed URL for uploading a video file
  - Query parameter: `filename`
  - Response: Upload URL and cleaned filename

- **GET /upload/status/{filename}**
  - Check the status of a video upload and processing
  - Response includes:
    - Status (processing/completed/failed)
    - Message
    - Transcript URL (when completed)

## Usage Example

```python
import requests

# API base URL
BASE_URL = "https://your-api-url"

# Upload a video
filename = "example.mp4"
# 1. Get upload URL
upload_request = requests.post(f"{BASE_URL}/upload/request", params={"filename": filename})
upload_data = upload_request.json()

# 2. Upload the video
with open(filename, 'rb') as video_file:
    upload_response = requests.put(
        upload_data["upload_url"],
        data=video_file,
        headers={"x-ms-blob-type": "BlockBlob", "Content-Type": "video/mp4"}
    )

# 3. Start processing
process_response = requests.post(f"{BASE_URL}/process/{upload_data['cleaned_filename']}")

# 4. Check status
status_response = requests.get(f"{BASE_URL}/upload/status/{upload_data['cleaned_filename']}")
```

## Error Handling

The API uses standard HTTP status codes:
- 200: Success
- 404: Resource not found
- 500: Server error

Error responses include a detail message explaining the issue.

## Security

- CORS is configured to allow specific origins
- File uploads are secured using Azure SAS tokens
- API keys are required for OpenAI integration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

