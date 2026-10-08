from pydantic import BaseModel
from typing import List, Optional

class VideoUploadResponse(BaseModel):
    status: str
    filename: str
    original_name: str
    file_size_mb: float
    duration_seconds: float
    video_url: str

class VideoAnalysisResponse(BaseModel):
    status: str
    model_used: str
    keyframes: List[str]
    keyframe_count: int
    video_duration: float
    analysis: str
    saved_report: str

class ChatMessageItem(BaseModel):
    role: str
    content: str

class ChatFollowupRequest(BaseModel):
    filename: str
    question: str
    initial_analysis: str
    history: List[ChatMessageItem] = []

class ChatFollowupResponse(BaseModel):
    status: str
    reply: str
