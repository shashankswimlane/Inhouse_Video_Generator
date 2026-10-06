import os
import glob
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
replicate_token = os.getenv("REPLICATE_API_TOKEN")

keyframes = glob.glob("static/keyframes/*.jpg")
print(f"Found {len(keyframes)} keyframe images in static/keyframes/")

if replicate_token and not replicate_token.startswith("r8_your"):
    print("[+] Replicate API token found! Testing Moondream2 vision model...")
    import replicate
    try:
        sample_img = keyframes[0]
        with open(sample_img, "rb") as f:
            output = replicate.run(
                "vikhyatk/moondream2:9c7819f474d048d3c5097a81057e5b5c928424a6ef24a138f3ec296f0b4a45a3",
                input={"image": f, "prompt": "Describe this image in detail focusing on the subject, clothing, jewelry, background, lighting, and camera shot type."}
            )
        print("✅ Moondream2 Vision Output:")
        print(output)
    except Exception as e:
        print("❌ Replicate Vision Error:", e)
else:
    print("[!] Replicate token is placeholder. Testing open-source vision / fallback extraction...")
