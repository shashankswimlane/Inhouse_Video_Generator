import base64
from pathlib import Path
from typing import List, Tuple
import cv2

from config import KEYFRAMES_DIR, KEYFRAME_MAX_DIMENSION

class VideoProcessorService:
    """Service for OpenCV video file inspection and keyframe image extraction."""

    @staticmethod
    def get_video_metadata(video_path: Path) -> Tuple[float, int, float]:
        """Returns (fps, frame_count, duration_seconds)."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = round(total_frames / fps, 2) if total_frames > 0 else 0.0
        cap.release()
        return fps, total_frames, duration_sec

    @staticmethod
    def extract_keyframes(video_path: Path, num_frames_requested: int) -> Tuple[List[Tuple[float, str]], List[str], float]:
        """
        Extracts keyframes from video, saves thumbnail images, and returns:
        (base64_frames, web_paths, duration_seconds)
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = round(total_frames / fps, 2) if total_frames > 0 else 0.0

        if total_frames <= 0:
            cap.release()
            raise ValueError("Could not read frames from video file.")

        # Determine frame count adaptively or explicitly
        if num_frames_requested <= 0:
            # Adaptive mode: 1 frame per second (min 5, max 60)
            num_frames = max(5, min(int(duration_sec), 60))
        else:
            num_frames = max(2, min(num_frames_requested, 60))

        step = max(1, total_frames // num_frames)

        base64_frames = []
        web_paths = []
        video_prefix = video_path.stem

        for i in range(num_frames):
            frame_idx = min(i * step, total_frames - 1)
            timestamp_sec = round(frame_idx / fps, 1)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Save full thumbnail to disk
            thumb_filename = f"{video_prefix}_frame_{i+1}.jpg"
            thumb_path = KEYFRAMES_DIR / thumb_filename
            cv2.imwrite(str(thumb_path), frame)
            web_paths.append(f"/static/keyframes/{thumb_filename}")

            # Resize frame for API token optimization
            h, w = frame.shape[:2]
            max_dim = KEYFRAME_MAX_DIMENSION
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            # JPEG Base64 encoding
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            b64_str = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append((timestamp_sec, b64_str))

        cap.release()
        return base64_frames, web_paths, duration_sec
