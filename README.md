# ⚡ Groq AI Video Intelligence & Generator Studio

A high-performance web application powered by **FastAPI**, **OpenCV**, and **Groq Multimodal Vision AI (`qwen/qwen3.8-27b`)** for full-length video analysis, keyframe breakdown, multi-turn Q&A context memory, and AI video prompt generation.

---

## 🌟 Key Features

- **⚡ Multimodal Vision Video Intelligence**: Reads and analyzes uploaded `.mp4`, `.mov`, `.avi`, `.webm` videos directly using Groq's `qwen/qwen3.8-27b` Multimodal Vision API.
- **📸 2-Image Batching Engine**: Automatically chunks keyframes into pairs of 2 images per request to prevent token rate limits (ITPM 7000 cap).
- **🎬 Shot-by-Shot Analysis**: Extracts keyframes, computes exact timestamps, and analyzes jewelry, wardrobe, subject expressions, camera angles, lighting, and text overlays.
- **💬 Interactive Memory Context Assistant**: Multi-turn chat assistant with full memory of the video analysis session to answer follow-up questions, draft AI prompts, or generate bullet summaries.
- **🚀 Replicate Video Generation Pipeline**: Includes `generate_video.py` to send optimized scene prompts to Replicate AI video generators (*CogVideoX-5B*, *Minimax*, *Luma*).

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shashankswimlane/Inhouse_Video_Generator.git
   cd Inhouse_Video_Generator
   ```

2. **Create & Activate Virtual Environment**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1   # On Windows PowerShell
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=gsk_your_groq_api_key_here
   REPLICATE_API_TOKEN=r8_your_replicate_token_here
   ```

---

## 🚀 Running the Application

Start the FastAPI Uvicorn web server:
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser to access the web studio interface.

---

## 🧪 Testing

Run end-to-end automated tests:
```bash
python test_upload_and_analysis.py
```
