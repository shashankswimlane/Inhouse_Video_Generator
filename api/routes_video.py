from datetime import datetime
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse

from config import UPLOADS_DIR
from core import VideoProcessorService, GroqVisionService
from schemas import VideoUploadResponse, VideoAnalysisResponse

router = APIRouter(prefix="/api", tags=["Video Services"])

@router.post("/upload-video", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """Uploads an MP4 video file and returns duration, file size, and URL."""
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.webm', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file type. Upload a video (.mp4, .mov, .avi, .webm).")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"video_{timestamp}_{file.filename.replace(' ', '_')}"
    file_path = UPLOADS_DIR / safe_name

    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

    fps, frame_count, duration_sec = VideoProcessorService.get_video_metadata(file_path)

    return VideoUploadResponse(
        status="success",
        filename=safe_name,
        original_name=file.filename,
        file_size_mb=round(len(contents) / (1024 * 1024), 2),
        duration_seconds=duration_sec,
        video_url=f"/uploads/{safe_name}"
    )

@router.post("/analyze-video", response_model=VideoAnalysisResponse)
async def analyze_video(
    filename: str = Form(...),
    custom_prompt: Optional[str] = Form(None),
    num_frames: int = Form(6)
):
    """Extracts keyframes and processes Groq Qwen Multimodal Vision analysis in 2-image batches."""
    video_path = UPLOADS_DIR / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file does not exist.")

    try:
        base64_frames, keyframe_web_paths, duration_sec = VideoProcessorService.extract_keyframes(video_path, num_frames)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Keyframe Extraction Error: {str(e)}")

    try:
        groq_service = GroqVisionService()
        analysis_text, output_filepath = groq_service.analyze_video_keyframes(
            filename=filename,
            base64_frames=base64_frames,
            duration_sec=duration_sec,
            custom_prompt=custom_prompt or ""
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API Error: {str(e)}")

    return VideoAnalysisResponse(
        status="success",
        model_used=groq_service.model,
        keyframes=keyframe_web_paths,
        keyframe_count=len(keyframe_web_paths),
        video_duration=duration_sec,
        analysis=analysis_text,
        saved_report=f"saved_responses/{output_filepath.name}"
    )
