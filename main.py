from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from config import BASE_DIR, STATIC_DIR, UPLOADS_DIR
from api import api_router

app = FastAPI(
    title="Groq AI Video Intelligence Studio",
    description="Full-length Multimodal Video Analysis & AI Video Generation API powered by Groq and OpenCV.",
    version="2.0.0"
)

# Register API Routes
app.include_router(api_router)

# Mount Static Files & Video Uploads
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_index():
    """Serves the main application web interface."""
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/uploads/{filename}")
async def serve_uploaded_video(filename: str):
    """Serves uploaded MP4 video files for local player preview."""
    file_path = UPLOADS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found.")
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
