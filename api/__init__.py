from fastapi import APIRouter
from .routes_video import router as video_router
from .routes_chat import router as chat_router

api_router = APIRouter()
api_router.include_router(video_router)
api_router.include_router(chat_router)

__all__ = ["api_router"]
