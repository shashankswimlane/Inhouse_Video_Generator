import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from groq import Groq

from config import (
    GROQ_API_KEY,
    GROQ_VISION_MODEL,
    VISION_BATCH_SIZE,
    BATCH_SLEEP_INTERVAL,
    RESPONSES_DIR,
)
from schemas import ChatMessageItem

class GroqVisionService:
    """Service handling Groq Multimodal Vision API batching, report synthesis, and context Q&A."""

    def __init__(self):
        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY is missing or invalid in configuration.")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_VISION_MODEL

    def analyze_video_keyframes(
        self,
        filename: str,
        base64_frames: List[Tuple[float, str]],
        duration_sec: float,
        custom_prompt: str = ""
    ) -> Tuple[str, Path]:
        """
        Processes keyframes in batches of 2 images per request, aggregates chunk outputs,
        synthesizes a master AI video prompt, and saves the report to disk.
        """
        default_prompt = (
            "Provide an exhaustive, shot-by-shot visual breakdown for these keyframe images. For each frame, provide:\n"
            "1. Exact Timestamp & Subject Expression/Actions\n"
            "2. Detailed Jewelry, Wardrobe & Accessory Analysis (metal types, gemstones, design details)\n"
            "3. Camera Angle, Lens Framing, & Movement (close-up, medium shot, tracking)\n"
            "4. Lighting, Color Palette, & On-screen Text/Graphics"
        )
        user_prompt_text = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else default_prompt

        batch_size = VISION_BATCH_SIZE
        total_frames_count = len(base64_frames)
        all_batch_reports = []

        print(f"[+] GroqVisionService: Processing {total_frames_count} keyframes in {batch_size}-image batches...")

        for i in range(0, total_frames_count, batch_size):
            chunk_frames = base64_frames[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_frames_count + batch_size - 1) // batch_size

            content_payload = [
                {
                    "type": "text",
                    "text": (
                        f"Video Title: {filename}\n"
                        f"Batch {batch_num}/{total_batches} (Keyframes #{i+1} to #{i+len(chunk_frames)} out of {total_frames_count}):\n\n"
                        f"Instructions: {user_prompt_text}\n\n"
                        f"Analyze the {len(chunk_frames)} image keyframe(s) below in chronological order:"
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
                "Inspect these keyframe images directly and output precise, timestamped visual breakdowns, "
                "identifying subjects, jewelry, wardrobe, camera work, lighting, and text overlays."
            )

            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
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

            time.sleep(BATCH_SLEEP_INTERVAL)

        combined_batch_analysis = "\n\n---\n\n".join(all_batch_reports)

        # Master Synthesis Pass
        try:
            synthesis = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Master Film Director and AI Video Prompt Engineer. Review the complete multi-shot visual analysis and generate an Executive Summary, Jewelry Audit, and a 200-400 word Master Prompt for AI video generation tools."
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

        # Save analysis report file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filepath = RESPONSES_DIR / f"analysis_{timestamp}.txt"
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(f"=== GROQ QWEN MULTIMODAL BATCHED VISION REPORT ===\n")
            f.write(f"Video File: {filename}\n")
            f.write(f"Video Duration: {duration_sec} seconds\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"Total Keyframes Sampled: {total_frames_count}\n\n")
            f.write(f"=== USER PROMPT ===\n{user_prompt_text}\n\n")
            f.write(f"=== UNIFIED VISION REPORT ===\n{analysis_text}\n")

        return analysis_text, output_filepath

    def chat_followup(self, filename: str, question: str, initial_analysis: str, history: List[ChatMessageItem]) -> str:
        """Processes multi-turn follow-up Q&A retaining memory context of video report."""
        system_context = (
            "You are an elite Cinematographer and AI Video Analyst Assistant engaged in a multi-turn conversation. "
            "You have complete visual memory context of the video keyframes and its detailed visual report.\n\n"
            f"Video Title: {filename}\n\n"
            f"=== BASE VISUAL ANALYSIS REPORT ===\n"
            f"{initial_analysis}\n"
            "====================================\n\n"
            "Use this deep visual knowledge base to answer the user's follow-up questions accurately, "
            "providing specific shot breakdowns, jewelry/wardrobe details, camera choices, lighting specs, or 200-400 word master prompts as requested."
        )

        messages = [{"role": "system", "content": system_context}]

        for item in history:
            messages.append({"role": item.role, "content": item.content})

        messages.append({"role": "user", "content": question})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=2048,
            temperature=0.3
        )
        return completion.choices[0].message.content
