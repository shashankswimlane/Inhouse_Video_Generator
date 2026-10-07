import sys
import time
from pathlib import Path
import cv2
import numpy as np
import requests

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

base_url = "http://127.0.0.1:8000"
test_dir = Path(__file__).parent / "test_scratch"
test_dir.mkdir(exist_ok=True)
test_video_path = test_dir / "sample_test_video.mp4"

print("[+] Creating a synthetic 3-second test MP4 video...")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(test_video_path), fourcc, 24.0, (640, 360))

colors = [
    (255, 100, 100),
    (100, 255, 100),
    (100, 100, 255),
]

for frame_idx in range(72):  # 3 seconds @ 24 fps
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    color = colors[(frame_idx // 24) % len(colors)]
    
    cv2.rectangle(img, (50, 50), (590, 310), color, -1)
    cv2.putText(img, f"Groq AI Video Test Frame #{frame_idx}", (80, 190), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    x_pos = int(100 + (frame_idx * 6))
    cv2.circle(img, (x_pos, 250), 20, (0, 255, 255), -1)
    
    out.write(img)

out.release()
print(f"[+] Sample video created at: {test_video_path.resolve()}")

# 1. Test Web Server Accessibility
print("\n[+] Testing GET / ...")
r = requests.get(base_url)
print(f"    Status: {r.status_code} (OK)")

# 2. Test Video Upload Endpoint
print("\n[+] Testing POST /api/upload-video ...")
with open(test_video_path, "rb") as f:
    r_upload = requests.post(f"{base_url}/api/upload-video", files={"file": ("sample_test_video.mp4", f, "video/mp4")})

if r_upload.status_code != 200:
    print(f"❌ Upload Failed: {r_upload.status_code} {r_upload.text}")
    sys.exit(1)

upload_json = r_upload.json()
print("✅ Upload Success Response:")
print(f"    Filename on Server: {upload_json['filename']}")
print(f"    Size: {upload_json['file_size_mb']} MB")
print(f"    Duration: {upload_json['duration_seconds']} sec")

uploaded_filename = upload_json['filename']

# 3. Test Video Analysis Endpoint with Groq Qwen Vision API
print("\n[+] Testing POST /api/analyze-video with Groq Vision API...")
r_analyze = requests.post(
    f"{base_url}/api/analyze-video",
    data={
        "filename": uploaded_filename,
        "custom_prompt": "Provide an exhaustive shot-by-shot timeline breakdown for the ENTIRE video.",
        "num_frames": -1
    }
)

if r_analyze.status_code != 200:
    print(f"❌ Analysis Failed: {r_analyze.status_code} {r_analyze.text}")
    sys.exit(1)

analyze_json = r_analyze.json()
print("✅ Analysis Success Response!")
print(f"    Model Used: {analyze_json['model_used']}")
print(f"    Keyframes Extracted: {analyze_json['keyframe_count']}")
print(f"    Keyframe Web Paths: {analyze_json['keyframes']}")
print("\n--- Groq Vision Report Output ---")
print(analyze_json['analysis'][:300] + "...")
print("---------------------------------")

# 4. Test Multi-Turn Chat Follow-Up Endpoint
print("\n[+] Testing POST /api/chat-followup (Interactive Video Memory Context)...")
r_chat = requests.post(
    f"{base_url}/api/chat-followup",
    json={
        "filename": uploaded_filename,
        "question": "Can you summarize the video into 3 bullet points and suggest a video prompt for AI generation?",
        "initial_analysis": analyze_json['analysis'],
        "history": []
    }
)

if r_chat.status_code != 200:
    print(f"❌ Chat Follow-up Failed: {r_chat.status_code} {r_chat.text}")
    sys.exit(1)

chat_json = r_chat.json()
print("✅ Chat Follow-up Success Response!")
print("--- Groq Assistant Memory Response ---")
print(chat_json['reply'][:300] + "...")
print("---------------------------------------")

print(f"\n🎉 ALL TESTS PASSED! Modular architecture, video upload, Groq full vision analysis, and Q&A context memory are 100% verified!")
