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

keyframes = glob.glob("static/keyframes/*.jpg")
if not keyframes:
    print("No keyframes found.")
    exit(1)

sample_img = keyframes[0]
with open(sample_img, "rb") as f:
    b64_img = base64.b64encode(f.read()).decode("utf-8")

vision_model_candidates = [
    "llama-3.2-11b-vision-instruct",
    "llama-3.2-90b-vision-instruct",
    "llava-v1.5-7b-wrapper",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b"
]

for model in vision_model_candidates:
    print(f"\n--- Testing Model: `{model}` ---")
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the contents of this image in detail (subject, clothing, jewelry, background, colors, camera shot type)."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }
            ],
            max_completion_tokens=300
        )
        print("[SUCCESS] Output:")
        print(res.choices[0].message.content[:300])
        break
    except Exception as e:
        print(f"[FAILED] {e}")
