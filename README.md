#  AI Video Editor — Semantic Clip Search

> Upload a video · Ask in plain English · Get your clips

---

##  What It Does

Type `"snowy mountains at sunset"` → get the exact clips from your video. No manual scrubbing.

---

##  How It Works

```
User Query (natural language)
        ↓
LLM Layer (Claude API) → simplified visual query
        ↓
CLIP Text Encoder → text embedding
        ↓
Cosine Similarity → ranked frame matches
        ↓
Smart Segment Grouping → merged continuous clips
        ↓
ffmpeg → cut + merge → final video
```

---

##  Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** CLIP installs from GitHub. Requires `torch` — GPU optional but faster.

### 2. Set your Anthropic API key (for LLM query optimization)

```bash
export ANTHROPIC_API_KEY=your_key_here
```

> The app works without this — LLM optimization is optional (toggle in UI).

### 3. Ensure ffmpeg is installed

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### 4. Run the app

```bash
streamlit run app.py
```

---

##  Project Structure

```
video_editor_ai/
│
├── app.py                          # Streamlit UI (4 steps)
│
├── pipeline/
│   ├── extract_frames.py           # cv2 frame extraction at N fps
│   ├── embeddings.py               # CLIP image embeddings + NPZ cache
│   ├── search.py                   # Cosine similarity search + segment grouping
│   └── clip_generator.py           # ffmpeg clip cutting + merging
│
├── utils/
│   ├── video_utils.py              # Video metadata helpers
│   ├── similarity.py               # Vectorized cosine similarity
│   └── query_simplifier.py         # Claude API LLM query optimizer
│
├── models/
│   └── clip_model_loader.py        # Singleton CLIP model loader
│
│
└── requirements.txt
```

---

#Features

| Feature | Description |
|---|---|
| CLIP Embeddings | Visual-semantic search — no keywords needed |
|  LLM Query Layer | Claude optimizes your query for CLIP |
|  Smart Segment Grouping | Nearby frames → one clean clip (no choppy cuts) |
|  Top-K Control | Choose how many clips to retrieve |
|  Preview Thumbnails | See best frame per segment before exporting |
|  Download | Individual clips or merged highlight reel |

---



##  Configuration

| Parameter | Default | Notes |
|---|---|---|
| Extraction FPS | 1.0 | Higher = more accurate, slower |
| Top-K segments | 3 | Number of clips to return |
| Segment gap threshold | 8s | Frames within 8s are merged |
| Clip padding | 5s | Seconds added before/after each segment |
| Similarity threshold | 0.20 | Minimum CLIP score to include frame |

---

##  Tech Stack

- **CLIP** (OpenAI) — Vision-language embeddings
- **PyTorch** — Model inference
- **OpenCV** — Frame extraction
- **ffmpeg** — Video cutting & merging
- **Streamlit** — UI
- **Claude API** — Natural language query optimization
- **NumPy** — Vectorized similarity computation
