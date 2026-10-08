import os
from datetime import datetime
from pathlib import Path
import requests
import replicate

from config import REPLICATE_API_TOKEN, RESPONSES_DIR

class VideoGenService:
    """Service for generating MP4 video files via Replicate AI Video API."""

    @staticmethod
    def generate_video_from_prompt(prompt: str, output_dir: Path = None) -> Path:
        if not REPLICATE_API_TOKEN or REPLICATE_API_TOKEN.startswith("r8_your"):
            raise ValueError("REPLICATE_API_TOKEN is missing or set to placeholder in .env")

        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
        output_dir = output_dir or (RESPONSES_DIR.parent / "output_videos")
        output_dir.mkdir(parents=True, exist_ok=True)

        model_identifier = "lucataco/cogvideox-5b:50c18080f5d7522d7168d1b11b51e0655d0a6c0b395d9703664d4715f187a414"

        output = replicate.run(
            model_identifier,
            input={
                "prompt": prompt,
                "num_frames": 49,
                "guidance_scale": 6.0,
                "num_inference_steps": 50
            }
        )

        video_url = str(output)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_video_path = output_dir / f"video_{timestamp}.mp4"

        res = requests.get(video_url, stream=True)
        res.raise_for_status()

        with open(local_video_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)

        return local_video_path
