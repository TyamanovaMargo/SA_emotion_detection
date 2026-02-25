# Slim Voice Feature Extraction Pipeline

Audio feature extraction pipeline — processes audio files and returns a JSON with all extracted voice features. No LLM, no transcription. Just signal processing + ML-based emotion detection.

## What it does

Takes an audio file → returns JSON with:
- **Prosody** — speaking rate (WPM), pitch mean/variance, energy, pauses, rhythm, speech-to-silence ratio
- **Voice Quality** — HNR, jitter, shimmer
- **Spectral** — MFCC mean + std (26 values)
- **eGeMAPS** — 88 acoustic features via OpenSMILE
- **Emotion Timeline** — per-segment emotion probabilities + VAD (valence/arousal/dominance) from MERaLiON-SER-v1
- **Emotion Aggregates** — dominant emotion, valence/arousal stats, arc type, stress index, confidence score
- **Motivation & Engagement** — deterministic score (0–100) based on voice indicators
- **Granular Features** — 43 flat keys for downstream processing
- **L2 Adjustments** — WPM cap + confidence penalty for non-native speakers
- **Paralinguistic Summary** — text description of voice profile

## Models

| Model | Purpose | Size | Cache |
|---|---|---|---|
| **MERaLiON/MERaLiON-SER-v1** | Emotion detection (7 emotions + VAD) | 770M params / 6.1 GB | `~/.cache/huggingface` |

OpenSMILE (eGeMAPS) and Praat (voice quality) are local CPU tools — no download needed.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Set HF_TOKEN=hf_... (required for MERaLiON gated model)

# 3. Download model once
python download_models.py

# 4a. Run as API server (recommended)
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1

# 4b. Or run CLI on a file
python slim_main.py audio.webm --output-dir outputs/
```

## API Server

Start:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

> **workers=1** is required — the MERaLiON model is loaded once into GPU memory and shared across all requests. Multiple workers would duplicate GPU memory usage.

Model loads at startup and stays in memory — no reload per request.

### Endpoints

**`GET /health`**
```json
{
  "status": "ok",
  "pipeline": "slim",
  "model_loaded": true,
  "device": "mps",
  "model": "MERaLiON/MERaLiON-SER-v1"
}
```

**`POST /assess`** — upload one audio file, get full feature JSON
```bash
curl -X POST http://localhost:8000/assess \
  -F "audio=@interview.webm" \
  -F "language_profile=non_native_english"
```

**`POST /assess/batch`** — upload multiple files
```bash
curl -X POST http://localhost:8000/assess/batch \
  -F "files=@audio1.webm" \
  -F "files=@audio2.webm" \
  -F "language_profile=non_native_english"
```

From Python:
```python
import requests

with open("interview.webm", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/assess",
        files={"audio": ("interview.webm", f, "audio/webm")},
        data={"language_profile": "non_native_english"},
    )

result = resp.json()
print(result["emotion_aggregates"]["dominant_emotion"])
print(result["motivation_engagement"]["motivation_score"])
```

### Supported audio formats
`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.webm`, `.aac`

## CLI

```bash
# Single file
python slim_main.py audio.webm

# With options
python slim_main.py audio.webm \
  --output-dir ./outputs \
  --language-profile non_native_english

# Batch (folder)
python slim_main.py ./recordings/ --output-dir ./outputs
```

Output: `outputs/<filename>_<timestamp>_slim.json`

## JSON Output Structure

```json
{
  "pipeline": "slim",
  "version": "1.0",
  "timestamp": "...",
  "audio_file": "interview.webm",
  "audio_duration_seconds": 74.0,
  "language_profile": "non_native_english",
  "processing_time_seconds": 68.0,

  "prosody": {
    "speaking_rate_wpm": 203.3,
    "pitch_mean_hz": 190.1,
    "pitch_variance": 1733.4,
    "energy_mean": 0.05,
    "energy_level": "medium",
    "pauses_per_minute": 13.8,
    "rhythm_regularity": 1.16,
    "speech_to_silence_ratio": 4.78,
    "...": "16 keys total"
  },

  "voice_quality": {
    "HNR": 10.93,
    "jitter": 0.020043,
    "shimmer": 0.133876
  },

  "spectral": { "...": "26 MFCC keys" },

  "egemaps": { "...": "88 acoustic features" },

  "emotion_timeline": [
    {
      "start": 0.0, "end": 7.5,
      "emotion": "sad",
      "confidence": 0.62,
      "valence": -0.355,
      "arousal": 0.032,
      "dominance": 0.353,
      "probabilities": { "neutral": 0.08, "sad": 0.71, "..." : "7 emotions" }
    }
  ],

  "emotion_aggregates": {
    "dominant_emotion": "sad",
    "dominant_ratio": 0.73,
    "valence_mean": -0.355,
    "arousal_mean": 0.032,
    "stress_index": 0.096,
    "confidence_score": 0.62,
    "arc_type": "fluctuating",
    "emotional_shifts": 8,
    "segments_analyzed": 37,
    "...": "17 keys total"
  },

  "motivation_engagement": {
    "motivation_score": 66,
    "motivation_level": "Medium",
    "pattern": "steady pitch, dynamic energy",
    "components": { "energy": 0.0, "pace": 10.0, "..." : "8 components" },
    "voice_indicators": ["speaking_rate_wpm=203.3 (fast)", "..."]
  },

  "l2_adjustments": { "...": "L2 speaker corrections applied" },
  "paralinguistic_summary": "Voice profile: ...",
  "granular_features": { "...": "43 flat keys" }
}
```

## Deployment on GPU Server

### Without Docker
```bash
git clone <repo> && cd SA_emotion_detection_pipeline
pip install -r requirements.txt
cp .env.example .env && nano .env   # set HF_TOKEN

python download_models.py           # download MERaLiON once (~6 GB)

uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

### With Docker (recommended for remote servers)
```bash
cp .env.example .env   # set HF_TOKEN

# Build (downloads model into image at build time)
docker compose up -d --build

# Or build manually
docker build --build-arg HF_TOKEN=hf_xxx -t slim-pipeline .
docker run --gpus all -p 8000:8000 \
  -v slim-pipeline-models:/models \
  -v ./outputs:/app/outputs \
  slim-pipeline
```

Model is stored in Docker volume `slim-pipeline-models` — persists across container restarts, no re-download.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | **Required.** HuggingFace token for MERaLiON gated model |
| `EMOTION_DEVICE` | `auto` | Device: `auto`, `cpu`, `cuda`, `cuda:0`, `mps` |
| `EMOTION_GPU` | `0` | GPU index when using CUDA |
| `EMOTION_BATCH_SIZE` | `0` | Batch size (0 = auto-detect) |
| `LANGUAGE_PROFILE` | `non_native_english` | Scoring profile: `non_native_english`, `native_english`, `sea_english` |
| `HF_HOME` | `~/.cache/huggingface` | Model cache directory |

## Project Structure

```
├── api.py                    # FastAPI server (POST /assess, GET /health)
├── slim_main.py              # CLI entry point
├── slim_pipeline.py          # Core pipeline orchestrator
├── download_models.py        # Pre-download MERaLiON to HF cache
├── Dockerfile                # Production container (CUDA)
├── docker-compose.yml        # GPU server deployment
├── requirements.txt
├── .env.example
└── src/
    ├── config.py             # EmotionConfig, ProsodyConfig, EgemapsConfig
    ├── extractors/
    │   ├── prosody.py        # Pitch, energy, pauses (librosa)
    │   ├── egemaps.py        # eGeMAPS features (openSMILE)
    │   ├── emotion_meralion.py  # MERaLiON-SER-v1 emotion detection
    │   ├── emotion_fusion.py    # SNR filtering, VAD smoothing
    │   └── voice_analyzer.py   # Unified voice analysis + aggregates
    ├── assessment/
    │   └── motivation_scorer.py  # Deterministic motivation/engagement
    ├── models/schemas.py     # Pydantic data models
    └── utils/
        ├── audio.py          # load_audio, normalize, trim_silence
        ├── device.py         # GPU/MPS/CPU detection
        └── scoring.py        # Score-to-label mapping
```

## Performance

Tested on Apple M-series (MPS):

| Audio Duration | Processing Time |
|---|---|
| 74 sec | ~68 sec |

On NVIDIA GPU (CUDA) expected **3–5x faster** (~15–25 sec for 74 sec audio).

Model loads once at server startup (~3 sec from cache). All subsequent requests reuse the loaded model.
