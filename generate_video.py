import os
import sys
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import replicate

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

base_dir = Path(__file__).parent
env_path = base_dir / ".env"
video_output_dir = base_dir / "output_videos"
video_output_dir.mkdir(parents=True, exist_ok=True)

print(f"[+] Loading environment configuration...")
load_dotenv(dotenv_path=env_path, override=True)

groq_key = os.getenv("GROQ_API_KEY")
replicate_token = os.getenv("REPLICATE_API_TOKEN")

if groq_key:
    groq_key = groq_key.strip("'\" \t\r\n")
if replicate_token:
    replicate_token = replicate_token.strip("'\" \t\r\n")

if not groq_key or groq_key == "your_groq_api_key_here":
    print("[X] Error: GROQ_API_KEY is not set in .env")
    exit(1)

if not replicate_token or replicate_token == "r8_your_replicate_api_token_here":
    print("[X] Error: REPLICATE_API_TOKEN is missing or set to placeholder in .env")
    print("--> Please add your Replicate API token to .env: REPLICATE_API_TOKEN=r8_...")
    print("--> Get a token at: https://replicate.com/account/api-tokens")
    exit(1)

# Set Replicate environment token
os.environ["REPLICATE_API_TOKEN"] = replicate_token

client = Groq(api_key=groq_key)

# Dynamic Scene Prompt from CLI arguments or user input
if len(sys.argv) > 1:
    prompt = " ".join(sys.argv[1:]).strip()
else:
    prompt = input("Enter video scene prompt: ").strip()

if not prompt:
    print("[X] Error: Prompt cannot be empty.")
    sys.exit(1)

print(f"\n[+] Input Scene Prompt:\n{prompt}\n")

print("[+] Step 1: Using Groq LLM to optimize prompt for video generation model...")
try:
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a prompt engineer for AI video generation models (CogVideoX, Minimax, Luma). Distill the scene into a dense, cinematic prompt under 100 words focused strictly on visual motion, subject, lighting, camera angle, and atmosphere."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    optimized_prompt = completion.choices[0].message.content.strip()
    print(f"\n[+] Optimized Video Prompt:\n{optimized_prompt}\n")
except Exception as e:
    print(f"[!] Groq optimization warning: {e}. Using raw prompt.")
    optimized_prompt = prompt

print("[+] Step 2: Generating MP4 Video via Replicate (CogVideoX model)...")

try:
    # Running CogVideoX 5B on Replicate
    model_identifier = "lucataco/cogvideox-5b:50c18080f5d7522d7168d1b11b51e0655d0a6c0b395d9703664d4715f187a414"
    
    output = replicate.run(
        model_identifier,
        input={
            "prompt": optimized_prompt,
            "num_frames": 49,
            "guidance_scale": 6.0,
            "num_inference_steps": 50
        }
    )

    # Output can be a FileOutput object or URL string
    video_url = str(output)
    print(f"\n[SUCCESS] Video generated remotely!")
    print(f"🔗 Video URL: {video_url}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_video_path = video_output_dir / f"video_{timestamp}.mp4"

    print(f"\n[+] Step 3: Downloading generated .mp4 to {local_video_path.resolve()}...")
    res = requests.get(video_url, stream=True)
    res.raise_for_status()

    with open(local_video_path, "wb") as f:
        for chunk in res.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"\n🎉 [COMPLETE] MP4 Video saved to local disk!")
    print(f"📁 Local File Path: file:///{local_video_path.as_posix()}")

except Exception as e:
    print(f"\n[ERROR] Failed during Replicate video generation: {e}")
