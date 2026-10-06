import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Get current script directory and .env file path
base_dir = Path(__file__).parent
env_path = base_dir / ".env"
output_dir = base_dir / "saved_responses"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"[+] Looking for .env file at: {env_path.resolve()}")
print(f"[+] .env file exists: {env_path.exists()}")
print(f"[+] Output directory ready at: {output_dir.resolve()}")

# Load environment variables
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("GROQ_API_KEY")
if api_key:
    api_key = api_key.strip("'\" \t\r\n")

if not api_key:
    print("[X] Error: GROQ_API_KEY environment variable is missing.")
    print("--> Please add your key to .env: GROQ_API_KEY=gsk_your_key_here")
    exit(1)
elif api_key == "your_groq_api_key_here":
    print("[X] Error: GROQ_API_KEY is still set to placeholder 'your_groq_api_key_here'.")
    print("--> Please open the .env file and paste your actual Groq API key (starts with gsk_...).")
    exit(1)

# Mask API key for logging
masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
print(f"[+] Loaded API Key: {masked_key}")

# Initialize Groq client
client = Groq(api_key=api_key)

# Video scene description prompt
video_prompt = """A 13-year-old Jewish girl with dark hair and a yellow Star of David sewn onto her coat walks down a quiet Amsterdam street at dawn. She is overdressed for the weather, wearing multiple layers of clothing that make her movements stiff. Warm rain falls steadily, slicking the cobblestones. She carries a school satchel clutched tight against her chest. Her expression is a mix of fear and forced composure. The camera follows her from a slight distance, keeping the Star of David visible on her back. 1940s Amsterdam architecture, brick buildings with dark windows, a bicycle leaning against a wall. Ambient audio: steady rain, distant church bells, soft footsteps on wet stone. The girl murmurs to herself: "This is the beginning. Everything is about to change." Moody, desaturated color palette with warm amber tones from streetlamps reflecting on wet pavement. Shallow depth of field, slow tracking shot."""

try:
    # Fetch available models for this API key
    models_list = client.models.list()
    available_model_ids = [m.id for m in models_list.data]
    print(f"\n[+] Available Groq models ({len(available_model_ids)} found):")
    for m_id in available_model_ids[:5]:
        print(f"   - {m_id}")

    target_model = "openai/gpt-oss-120b"
    if target_model not in available_model_ids and available_model_ids:
        target_model = available_model_ids[0]

    print(f"\n[+] Sending video prompt request to Groq API (model: `{target_model}`)...")
    completion = client.chat.completions.create(
        model=target_model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert AI Video Production Assistant. Enhance and break down the user's video prompt into structured video generation parameters, keyframe details, camera movement instructions, and lighting notes."
            },
            {
                "role": "user",
                "content": video_prompt
            }
        ]
    )

    response_text = completion.choices[0].message.content

    print("\n[SUCCESS] Groq LLM Response:")
    print("-" * 60)
    print(response_text)
    print("-" * 60)

    # Save response to saved_responses folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filepath = output_dir / f"video_response_{timestamp}.txt"
    
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(f"=== INPUT VIDEO PROMPT ===\n{video_prompt}\n\n")
        f.write(f"=== GROQ LLM RESPONSE ({target_model}) ===\n")
        f.write(response_text)
        f.write("\n")

    print(f"\n[+] Response saved successfully to: [video_response_{timestamp}.txt](file:///{output_filepath.as_posix()})")

except Exception as e:
    print(f"\n[ERROR] Error calling Groq API: {e}")




