import os
import sys
import glob
import base64
from dotenv import load_dotenv
from groq import Groq

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

keyframes = sorted(glob.glob("static/keyframes/video_*_instagram_*.jpg"))
if not keyframes:
    keyframes = sorted(glob.glob("static/keyframes/*.jpg"))

print(f"Found {len(keyframes)} keyframes to analyze.")

sample_keyframes = keyframes[:6]
batch_size = 3
all_reports = []

print(f"\n[+] Testing 3-image batch looping on {len(sample_keyframes)} keyframes...")

for i in range(0, len(sample_keyframes), batch_size):
    chunk = sample_keyframes[i : i + batch_size]
    batch_num = (i // batch_size) + 1
    
    content_payload = [
        {"type": "text", "text": f"Analyzing Batch #{batch_num} ({len(chunk)} images): Describe subjects, jewelry, attire, camera angle, and on-screen text."}
    ]
    for idx_in_batch, img_path in enumerate(chunk):
        with open(img_path, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode("utf-8")
        content_payload.append({
            "type": "text",
            "text": f"--- Frame #{i + idx_in_batch + 1} ---"
        })
        content_payload.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
        })
    
    res = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": "You are a senior fashion & jewelry visual analyst."},
            {"role": "user", "content": content_payload}
        ],
        max_completion_tokens=600
    )
    all_reports.append(f"### Batch #{batch_num} Output:\n" + res.choices[0].message.content)

print("\n[SUCCESS] Unified Output from all 3-image batches:")
print("=" * 60)
print("\n\n---\n\n".join(all_reports))
print("=" * 60)

