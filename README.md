# HR Personality & Motivation Assessment Pipeline

AI-powered voice analysis pipeline that extracts **Big Five personality traits** and **motivation/engagement levels** from audio recordings using voice features (prosody, emotion, acoustics) and optional transcripts.

## 🎯 Key Features

- **Voice-Only Analysis** — Assess personality and motivation from voice features alone (no transcript needed)
- **Stable Scoring** — Deterministic formulas ensure consistent results for the same speaker
- **Universal Audio Input** — Single file or folder, any audio format (WAV, MP3, M4A, AAC, FLAC, OGG)
- **GPU Acceleration** — CUDA support with automatic GPU selection for shared servers
- **Prosody Analysis** — Speaking rate, pitch variance, energy, pauses, rhythm
- **Emotion Detection** — emotion2vec model with CPU fallback
- **Acoustic Features** — eGeMAPS via OpenSMILE
- **AI Assessment** — Groq LLM (LLaMA 3.3 70B) with deterministic scoring (temperature=0.0)
- **Multiple Outputs** — JSON reports, HTML reports, batch summary tables
- **Beautiful Visualization** — Color-coded progress bars for motivation and engagement

## 📊 Assessment Output

### Big Five Personality Profile (0–100)

| Trait | What it measures | Voice indicators |
|-------|-----------------|------------------|
| **Openness** | Creativity, curiosity | Wide pitch range, expressive tone |
| **Conscientiousness** | Organization, discipline | Steady pace, few fillers, structured speech |
| **Extraversion** | Sociability, assertiveness | Fast rate, high energy, loud volume |
| **Agreeableness** | Cooperation, trust | Warm prosody, smooth pitch |
| **Neuroticism** | Emotional instability | Unstable pitch, many pauses, rough voice |

### Motivation & Engagement Assessment

- **Motivation Score** (0-100) — Computed from voice features using deterministic formulas
- **Engagement Score** (0-100) — Derived from motivation + extraversion
- **Level** — High / Medium / Low (with hysteresis to prevent flickering)
- **Pattern** — Rising / Falling / Consistent / Fluctuating
- **Voice Indicators** — Energy, speaking rate, pauses, pitch dynamics, emotion

**Example Output:**
```
Motivation & Engagement Analysis:
  Overall Motivation [████░░░░░░░░░░░░░░░░] Low (20/100)
  Pattern: Consistent

  Voice-Based Indicators:
    • energy_mean=0.028 (low)
    • speaking_rate_wpm=104 (slow)
    • pauses_per_minute=7.2 (high)
    • pitch_variance=250 (low)

  Engagement Level   [██████░░░░░░░░░░░░░░] Low (30/100)
  Derived from motivation (20) and extraversion (30)
```

## 🚀 Quick Start

### Installation

```bash
cd SA_emotion_detection
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Configure API key:
```bash
cp .env.example .env
# Edit .env → set GROQ_API_KEY
```

### Basic Usage

```bash
# Single audio file (voice-only analysis)
python main.py interview.wav --skip-transcription

# Folder of audio files
python main.py recordings/ --skip-transcription --limit 5

# With transcript
python main.py interview.wav --transcript interview.txt

# Generate HTML report
python main.py interview.wav --html-report
```

## 🐳 Docker Deployment

### Quick Start (Automatic GPU Selection)

```bash
# Build Docker image
docker compose build

# Check available GPUs
./check_gpu.sh

# Run pipeline (automatically selects free GPU)
./run_pipeline.sh "Team Recordings/Digvijay/Audio/" --skip-transcription --limit 2
```

### Manual Docker Commands

```bash
# Build
docker compose build hr-assessment

# Run on specific GPU (e.g., GPU 1)
docker run -d --name hr-assessment-pipeline \
  --gpus '"device=1"' \
  -v "$(pwd)/Team Recordings:/app/Team Recordings:ro" \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/.env:/app/.env:ro" \
  -e GROQ_API_KEY \
  -e CUDA_VISIBLE_DEVICES=1 \
  -e WHISPER_DEVICE=cuda \
  -e EMOTION_DEVICE=cuda \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --shm-size=8g \
  sa_emotion_detection-hr-assessment tail -f /dev/null

# Run pipeline inside container
docker exec hr-assessment-pipeline python main.py "Team Recordings/Digvijay/Audio/" --skip-transcription

# Stop container
docker stop hr-assessment-pipeline && docker rm hr-assessment-pipeline
```

## 🖥️ GPU Usage (Shared Server)

### Automatic GPU Selection

The `run_pipeline.sh` script automatically finds a free GPU and starts the container:

```bash
# Check available GPUs
./check_gpu.sh

# Run pipeline (auto-selects free GPU)
./run_pipeline.sh "Team Recordings/Digvijay/Audio/" --skip-transcription
```

**Example output:**
```
🔍 Searching for available GPU...
✅ Found free GPU: 2 (Quadro RTX 6000, 23.5 GB free)
🚀 Starting container on GPU 2...
✅ Container started successfully on GPU 2
```

### Using Multiple GPUs

Distribute models across 2 GPUs for better performance:

```bash
docker run -d --name hr-assessment-pipeline \
  --gpus '"device=0,1"' \
  -v "$(pwd)/Team Recordings:/app/Team Recordings:ro" \
  -v "$(pwd)/outputs:/app/outputs" \
  -e GROQ_API_KEY \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e WHISPER_DEVICE=cuda:0 \
  -e EMOTION_DEVICE=cuda:1 \
  --shm-size=8g \
  sa_emotion_detection-hr-assessment tail -f /dev/null
```

**Model distribution:**
- **GPU 0**: Whisper, WavLM, prosody
- **GPU 1**: emotion2vec

### Best Practices for Shared Servers

1. **Always check GPU availability** before running: `./check_gpu.sh`
2. **Use automatic GPU selection**: `./run_pipeline.sh`
3. **Stop container when done**: `docker stop hr-assessment-pipeline`
4. **Monitor GPU usage**: `watch -n 1 nvidia-smi`
5. **Be considerate** — don't occupy GPUs unnecessarily

## 📝 CLI Options

```bash
python main.py <input_path> [options]

Positional:
  input_path              Audio file or folder

Options:
  -t, --transcript PATH   Transcript file (.txt/.json)
  -c, --candidate-id ID   Candidate identifier
  -p, --position ROLE     Position / role context
  -o, --output-dir DIR    Output directory (default: ./outputs)
  -l, --limit N           Max files to process
  --skip-transcription    Voice-only analysis (no Whisper)
  --whisper-model SIZE    tiny/base/small/medium/large (default: base)
  --html-report           Generate HTML report per file
  --group-by-folder       Group results by parent folder
  --no-save               Don't write JSON output
  -q, --quiet             Suppress detailed output
```

## 🔬 How It Works

### 1. Voice Feature Extraction

**Prosody Features:**
- Speaking rate (words per minute)
- Pitch variance, range, mean, slope
- Energy mean, std, range
- Pauses per minute, long pauses count
- Speech-to-silence ratio
- Rhythm regularity

**Emotion Detection:**
- emotion2vec model (9 emotions: happy, sad, angry, neutral, fearful, surprised, disgusted, contempt, unknown)
- Primary emotion + confidence score
- Emotion timeline

**Acoustic Features:**
- eGeMAPS (88 features via OpenSMILE)
- Voice quality (HNR, jitter, shimmer)
- Spectral features

### 2. Deterministic Scoring

**Motivation Score (0-100):**
```
Start: 50

Energy:     if energy_mean >= 0.06: +15, if <= 0.03: -15
Pace:       if speaking_rate >= 150: +15, if <= 110: -15
Pauses:     if pauses_per_minute <= 3: +10, if >= 6: -10
Pitch:      if pitch_variance >= 800: +10, if <= 300: -10
Emotion:    if happy/surprised + conf >= 0.5: +10
            if sad/fearful + conf >= 0.5: -10

Clamp to [0, 100]
```

**Engagement Score (0-100):**
```
engagement_score = round(0.6 * motivation_score + 0.4 * extraversion_score)
```

**Hysteresis (Stable Levels):**
- Scores within ±7 points of boundaries (40, 70) → set to "Medium" to prevent flickering
- Ensures same speaker gets consistent level across multiple recordings

### 3. LLM Assessment

- **Model**: Groq LLaMA 3.3 70B
- **Temperature**: 0.0 (deterministic output)
- **Input**: Voice features + optional transcript
- **Output**: Big Five scores, motivation/engagement, strengths, development areas, HR summary

## 📂 Output Files

### JSON Report
```json
{
  "metadata": {
    "audio_file": "interview.wav",
    "candidate_id": "C001",
    "timestamp": "20260215_120000"
  },
  "assessment": {
    "big_five": {
      "openness": {"score": 65, "confidence": 80, "reason": "..."},
      "conscientiousness": {"score": 72, "confidence": 85, "reason": "..."},
      ...
    },
    "motivation": {
      "overall_level": "High",
      "motivation_score": 75,
      "pattern": "Rising",
      "voice_indicators": ["high energy", "fast pace", ...]
    },
    "engagement": {
      "overall_level": "High",
      "engagement_score": 78,
      "reason": "Derived from motivation (75) and extraversion (85)"
    },
    "trait_strengths": ["Conscientiousness", "Extraversion", ...],
    "hr_summary": "..."
  }
}
```

### Console Output
```
============================================================
HR ASSESSMENT SUMMARY
============================================================
Candidate: John_Doe

Big Five Personality Profile:
  Openness           [█████████████░░░░░░░] 65/100 (80% conf)
  Conscientiousness  [██████████████░░░░░░] 72/100 (85% conf)
  Extraversion       [█████████████████░░░] 85/100 (90% conf)
  Agreeableness      [████████████░░░░░░░░] 60/100 (75% conf)
  Neuroticism        [██████░░░░░░░░░░░░░░] 30/100 (70% conf)

Motivation & Engagement Analysis:
  Overall Motivation [███████████████░░░░░] High (75/100)
  Pattern: Rising

  Voice-Based Indicators:
    • energy_mean=0.065 (high)
    • speaking_rate_wpm=165 (fast)
    • pauses_per_minute=2.5 (low)
    • pitch_variance=850 (high)

  Engagement Level   [███████████████░░░░░] High (78/100)
  Derived from motivation (75) and extraversion (85)

Key Strengths:
  • Extraversion
  • Conscientiousness
  • Achievement-Striving

Development Areas:
  • Openness
  • Agreeableness

✓ Processing completed in 125.34s
============================================================
```

## 🐍 Python API

```python
from src.pipeline import HRAssessmentPipeline
from src.config import load_config

# Initialize pipeline
pipeline = HRAssessmentPipeline(load_config())

# Process audio (voice-only)
result = pipeline.process(
    audio_path="interview.wav",
    candidate_id="C001",
    skip_transcription=True
)

# Access results
print(f"Motivation: {result.motivation.motivation_score}/100")
print(f"Engagement: {result.engagement.engagement_score}/100")
print(f"Extraversion: {result.big_five.extraversion.score}/100")

# Print summary
pipeline.print_summary(result)
```

## 🌐 REST API

```bash
# Start API server
uvicorn api:app --reload

# Assess candidate
curl -X POST http://localhost:8000/assess \
  -F "audio=@interview.wav" \
  -F "candidate_id=C001"
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/assess` | POST | JSON assessment |
| `/assess/html` | POST | HTML report |

## ⚙️ Configuration

### Environment Variables

```bash
# .env
GROQ_API_KEY=gsk_...                       # Required
GROQ_MODEL=llama-3.3-70b-versatile         # Optional
WHISPER_MODEL=base                          # tiny/base/small/medium/large
WHISPER_DEVICE=cuda                         # cpu/cuda
EMOTION_DEVICE=cuda                         # cpu/cuda
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Python Configuration

```python
from src.config import PipelineConfig, WhisperConfig, GroqConfig

config = PipelineConfig(
    whisper=WhisperConfig(model_name="medium", device="cuda"),
    groq=GroqConfig(temperature=0.0),  # Deterministic output
)

pipeline = HRAssessmentPipeline(config)
```

## 🛠️ Troubleshooting

### CUDA Out of Memory

**Symptoms:**
```
Model detection failed: CUDA out of memory. Tried to allocate 6.19 GiB...
```

**Solutions:**
1. **Use 2 GPUs** (recommended):
   ```bash
   ./run_pipeline.sh  # Auto-selects free GPU
   ```

2. **CPU fallback** (automatic) — emotion2vec will run on CPU if GPU OOM occurs

3. **Increase shared memory**:
   ```bash
   --shm-size=16g
   ```

4. **Use CPU mode** (slower):
   ```bash
   -e WHISPER_DEVICE=cpu -e EMOTION_DEVICE=cpu
   ```

### No Free GPUs

```bash
# Check GPU availability
./check_gpu.sh

# Wait for GPUs to free up, or use CPU mode
```

### Inconsistent Motivation Scores

The new deterministic formulas ensure consistency. If you see variations:
- Check that voice features are extracted correctly
- Verify temperature=0.0 in config
- Ensure using latest Docker image

## 📁 Project Structure

```
SA_emotion_detection/
├── main.py                    # CLI entry point
├── api.py                     # FastAPI REST server
├── Dockerfile                 # Docker image
├── docker-compose.yml         # Docker orchestration
├── requirements.txt
├── .env.example
├── check_gpu.sh              # GPU availability checker
├── run_pipeline.sh           # Auto GPU selection script
├── src/
│   ├── config.py              # Configuration
│   ├── pipeline.py            # Main orchestrator
│   ├── models/
│   │   └── schemas.py         # Pydantic data models
│   ├── extractors/
│   │   ├── transcription.py   # Whisper speech-to-text
│   │   ├── prosody.py         # Pitch, energy, pauses
│   │   ├── emotion.py         # emotion2vec detection
│   │   └── egemaps.py         # eGeMAPS acoustic features
│   ├── assessment/
│   │   ├── groq_assessor.py   # Groq LLM integration
│   │   └── prompt_templates.py # Deterministic scoring prompts
│   └── utils/
│       ├── audio.py           # Audio utilities
│       └── reporting.py       # HTML report generation
└── outputs/                   # Generated reports
```

## 📋 Requirements

- Python 3.9+
- FFmpeg
- Groq API key → https://console.groq.com
- NVIDIA GPU (optional, for acceleration)
- NVIDIA Container Toolkit (for Docker GPU support)

## 🎵 Supported Audio Formats

WAV, MP3, M4A, AAC, FLAC, OGG, WebM

## 📄 License

MIT License

## 🙏 Acknowledgments

- **Whisper** — OpenAI speech-to-text
- **emotion2vec** — Alibaba DAMO Academy
- **OpenSMILE** — eGeMAPS acoustic features
- **Groq** — Fast LLM inference
- **librosa** — Audio processing

---

**For detailed GPU setup and troubleshooting, see the helper scripts:**
- `./check_gpu.sh` — Check GPU availability
- `./run_pipeline.sh` — Auto-select free GPU and run pipeline
