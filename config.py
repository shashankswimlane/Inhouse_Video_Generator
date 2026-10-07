import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Force UTF-8 encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")

# Directory Paths
UPLOADS_DIR = BASE_DIR / "uploads"
KEYFRAMES_DIR = BASE_DIR / "static" / "keyframes"
RESPONSES_DIR = BASE_DIR / "saved_responses"
STATIC_DIR = BASE_DIR / "static"

for directory in [UPLOADS_DIR, KEYFRAMES_DIR, RESPONSES_DIR, STATIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Vision Model Configuration
GROQ_VISION_MODEL = "qwen/qwen3.8-27b"
VISION_BATCH_SIZE = 2             # 2 images per API request to stay under 7000 ITPM rate limit
KEYFRAME_MAX_DIMENSION = 512      # Rescale max image dimension for token optimization
BATCH_SLEEP_INTERVAL = 1.2        # Delay in seconds between batch calls
