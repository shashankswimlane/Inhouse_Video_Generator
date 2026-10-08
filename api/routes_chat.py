from fastapi import APIRouter, HTTPException

from core import GroqVisionService
from schemas import ChatFollowupRequest, ChatFollowupResponse

router = APIRouter(prefix="/api", tags=["Chat Services"])

@router.post("/chat-followup", response_model=ChatFollowupResponse)
async def chat_followup(req: ChatFollowupRequest):
    """Multi-turn conversational Q&A endpoint retaining visual memory context of video analysis."""
    try:
        groq_service = GroqVisionService()
        reply_text = groq_service.chat_followup(
            filename=req.filename,
            question=req.question,
            initial_analysis=req.initial_analysis,
            history=req.history
        )
        return ChatFollowupResponse(status="success", reply=reply_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq Chat Error: {str(e)}")
