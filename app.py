import os
import sys
import time
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from groq import Groq


# Ensure UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Setup paths
BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR / ".env"
UPLOADS_DIR = BASE_DIR / "uploads"
KEYFRAMES_DIR = BASE_DIR / "static" / "keyframes"
RESPONSES_DIR = BASE_DIR / "saved_responses"

for d in [UPLOADS_DIR, KEYFRAMES_DIR, RESPONSES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(dotenv_path=ENV_PATH, override=True)

app = FastAPI(title="Groq AI Video Analyzer")

# Serve static frontend files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
async def serve_index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Uploads an MP4 video file and saves it locally."""
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.webm', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file (.mp4, .mov, .avi, .webm).")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"video_{timestamp}_{file.filename.replace(' ', '_')}"
    file_path = UPLOADS_DIR / safe_name

    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

    # Get video duration and basic properties using OpenCV
    cap = cv2.VideoCapture(str(file_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = round(frame_count / fps, 2) if frame_count > 0 else 0
    cap.release()

    return {
        "status": "success",
        "filename": safe_name,
        "original_name": file.filename,
        "file_size_mb": round(len(contents) / (1024 * 1024), 2),
        "duration_seconds": duration_sec,
        "video_url": f"/uploads/{safe_name}"
    }


@app.get("/uploads/{filename}")
async def serve_uploaded_video(filename: str):
    file_path = UPLOADS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found.")
    return FileResponse(file_path)


@app.post("/api/analyze-video")
async def analyze_video(
    filename: str = Form(...),
    custom_prompt: Optional[str] = Form(None),
    num_frames: int = Form(6)
):
    """Extracts keyframes from video and calls Groq Vision LLM for analysis."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise HTTPException(status_code=400, detail="GROQ_API_KEY is not configured in .env file.")

    video_path = UPLOADS_DIR / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file does not exist.")

    # Step 1: Keyframe Extraction via OpenCV
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = round(total_frames / fps, 2) if total_frames > 0 else 0

    if total_frames <= 0:
        cap.release()
        raise HTTPException(status_code=400, detail="Could not read video frames.")

    # Determine frame count adaptively or explicitly up to 60 keyframes
    if num_frames == -1 or num_frames == 0:
        # Adaptive mode: 1 frame per second (min 5 frames, max 60 frames)
        num_frames = max(5, min(int(duration_sec), 60))
    else:
        num_frames = max(2, min(num_frames, 60))

    step = max(1, total_frames // num_frames)

    extracted_base64_frames = []
    keyframe_web_paths = []

    video_prefix = video_path.stem

    for i in range(num_frames):
        frame_idx = min(i * step, total_frames - 1)
        timestamp_sec = round(frame_idx / fps, 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Save thumbnail to static/keyframes
        thumb_filename = f"{video_prefix}_frame_{i+1}.jpg"
        thumb_path = KEYFRAMES_DIR / thumb_filename
        cv2.imwrite(str(thumb_path), frame)
        keyframe_web_paths.append(f"/static/keyframes/{thumb_filename}")

        # Resize for API optimization (max 512px width/height for token efficiency)
        h, w = frame.shape[:2]
        max_dim = 512
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Encode to JPEG base64
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        b64_str = base64.b64encode(buffer).decode('utf-8')
        extracted_base64_frames.append((timestamp_sec, b64_str))

    cap.release()

    if not extracted_base64_frames:
        raise HTTPException(status_code=500, detail="Failed to extract any valid keyframes from the video.")

    # Step 2: Groq Multimodal Vision Model Integration (2-Image Batch Looping)
    client = Groq(api_key=api_key)
    target_model = "qwen/qwen3.8-27b"

    default_user_prompt = (
        "Provide an exhaustive, shot-by-shot visual breakdown for these keyframe images. For each frame, provide:\n"
        "1. Exact Timestamp & Subject Expression/Actions\n"
        "2. Detailed Jewelry, Wardrobe & Accessory Analysis (metal types, gemstones, design details)\n"
        "3. Camera Angle, Lens Framing, & Movement (close-up, medium shot, tracking)\n"
        "4. Lighting, Color Palette, & On-screen Text/Graphics"
    )
    user_prompt_text = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else default_user_prompt

    batch_size = 2
    total_frames_count = len(extracted_base64_frames)
    all_batch_reports = []

    print(f"[+] Processing {total_frames_count} keyframes in 2-image batches using {target_model}...")

    # Loop through all keyframes in packs of 2 images
    for i in range(0, total_frames_count, batch_size):
        chunk_frames = extracted_base64_frames[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_frames_count + batch_size - 1) // batch_size

        print(f"[+] Batch {batch_num}/{total_batches} (Keyframes #{i+1} to #{i+len(chunk_frames)})...")

        content_payload = [
            {
                "type": "text",
                "text": (
                    f"Video Title: {filename}\n"
                    f"Batch {batch_num}/{total_batches} (Keyframes #{i+1} to #{i+len(chunk_frames)} out of {total_frames_count}):\n\n"
                    f"Instructions: {user_prompt_text}\n\n"
                    "Analyze the 2 image keyframes below in chronological order:"
                )
            }
        ]

        for idx_in_batch, (t_sec, b64_img) in enumerate(chunk_frames):
            frame_num = i + idx_in_batch + 1
            content_payload.append({
                "type": "text",
                "text": f"\n--- Keyframe #{frame_num} [Timestamp: {t_sec:.1f}s / {duration_sec}s] ---"
            })
            content_payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}"
                }
            })

        system_instruction = (
            "You are an elite Cinematographer and Fashion/Jewelry Vision Analyst. "
            "Inspect these 2 keyframe images directly and output precise, timestamped visual breakdowns, "
            "identifying subjects, jewelry, wardrobe, camera work, lighting, and text overlays."
        )

        try:
            completion = client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": content_payload}
                ],
                max_completion_tokens=1000,
                temperature=0.3
            )
            batch_text = completion.choices[0].message.content
            all_batch_reports.append(f"## 🎬 Shot Segment: Keyframes #{i+1} to #{i+len(chunk_frames)}\n\n{batch_text}")
        except Exception as e:
            print(f"[!] Warning on Batch {batch_num}: {e}")
            all_batch_reports.append(f"## 🎬 Shot Segment: Keyframes #{i+1} to #{i+len(chunk_frames)}\n\n[Error analyzing batch: {e}]")

        # Sleep briefly between batch iterations to remain comfortably under ITPM rate limits
        time.sleep(1.2)


    # Combine all batch visual breakdowns
    combined_batch_analysis = "\n\n---\n\n".join(all_batch_reports)

    # Master Synthesis Pass: Synthesize overall AI Video Generation Prompt based on all batch breakdowns
    try:
        synthesis = client.chat.completions.create(
            model=target_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Master Film Director and AI Video Prompt Engineer. Review the complete multi-shot visual analysis and generate an Executive Summary and a 200-400 word Master Prompt for AI video generation tools (Sora, Runway Gen-3, CogVideoX)."
                },
                {
                    "role": "user",
                    "content": (
                        f"Video Title: {filename} (Duration: {duration_sec}s, Total Keyframes: {total_frames_count})\n\n"
                        f"=== COMPLETE SHOT-BY-SHOT VISUAL ANALYSIS ===\n\n{combined_batch_analysis}\n\n"
                        "Please provide:\n"
                        "1. Executive Commercial Summary\n"
                        "2. Jewelry & Styling Audit\n"
                        "3. Master AI Video Generation Prompt (200-400 words) to recreate a near-identical commercial."
                    )
                }
            ],
            max_completion_tokens=1200,
            temperature=0.3
        )
        master_summary = synthesis.choices[0].message.content
        analysis_text = f"# 🌟 Master Video Commercial Synthesis\n\n{master_summary}\n\n---\n\n# 📸 Shot-by-Shot Visual Breakdown ({total_batches} Batches, {total_frames_count} Keyframes)\n\n{combined_batch_analysis}"
    except Exception as e:
        analysis_text = combined_batch_analysis

    try:
        # Save response text file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filepath = RESPONSES_DIR / f"analysis_{timestamp}.txt"
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(f"=== GROQ QWEN MULTIMODAL BATCHED VISION REPORT ===\n")
            f.write(f"Video File: {filename}\n")
            f.write(f"Video Duration: {duration_sec} seconds\n")
            f.write(f"Model: {target_model}\n")
            f.write(f"Total Keyframes Sampled: {total_frames_count}\n")
            f.write(f"Batches Processed (3 images/batch): {len(all_batch_reports)}\n\n")
            f.write(f"=== USER PROMPT ===\n{user_prompt_text}\n\n")
            f.write(f"=== UNIFIED VISION REPORT ===\n{analysis_text}\n")

        return {
            "status": "success",
            "model_used": target_model,
            "keyframes": keyframe_web_paths,
            "keyframe_count": len(keyframe_web_paths),
            "video_duration": duration_sec,
            "analysis": analysis_text,
            "saved_report": f"saved_responses/analysis_{timestamp}.txt"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API Error: {str(e)}")




from pydantic import BaseModel
from typing import List

class ChatMessageItem(BaseModel):
    role: str
    content: str

class ChatFollowupRequest(BaseModel):
    filename: str
    question: str
    initial_analysis: str
    history: List[ChatMessageItem] = []


@app.post("/api/chat-followup")
async def chat_followup(req: ChatFollowupRequest):
    """Multi-turn conversational Q&A endpoint with memory context of video analysis."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY is not configured.")

    client = Groq(api_key=api_key)
    target_model = "qwen/qwen3.8-27b"

    system_context = (
        "You are an elite Cinematographer and AI Video Analyst Assistant engaged in a multi-turn conversation. "
        "You have complete visual memory context of the video keyframes and its detailed visual report.\n\n"
        f"Video Title: {req.filename}\n\n"
        f"=== BASE VISUAL ANALYSIS REPORT ===\n"
        f"{req.initial_analysis}\n"
        "====================================\n\n"
        "Use this deep visual knowledge base to answer the user's follow-up questions accurately, "
        "providing specific shot breakdowns, jewelry/wardrobe details, camera choices, lighting specs, or 200-400 word master prompts for AI video generators (Sora, Runway Gen-3, CogVideoX) as requested."
    )


    messages = [{"role": "system", "content": system_context}]

    # Reconstruct dialogue history
    for item in req.history:
        messages.append({"role": item.role, "content": item.content})

    # Add latest question
    messages.append({"role": "user", "content": req.question})

    try:
        completion = client.chat.completions.create(
            model=target_model,
            messages=messages,
            max_completion_tokens=2048,
            temperature=0.3
        )

        reply_text = completion.choices[0].message.content
        return {
            "status": "success",
            "reply": reply_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

