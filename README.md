# HR Personality & Motivation Assessment Pipeline

AI-powered voice analysis pipeline that takes **any audio file** as input and produces Big Five personality traits + motivation assessment reports.

## Features

- **Universal audio input** — single file or folder, any structure
- **Auto-transcript detection** — finds matching `.txt`/`.json` transcripts automatically
- **Speech-to-Text** — Whisper transcription when no transcript is available
- **Prosody Analysis** — speaking rate, pitch, energy, pause detection
- **Emotion Detection** — emotion2vec with acoustic fallback
- **Acoustic Features** — eGeMAPS via OpenSMILE (librosa fallback)
- **AI Assessment** — Groq LLM (LLaMA 3.3 70B) for Big Five + motivation scoring
- **Multiple outputs** — JSON reports, HTML reports, batch summary tables

## Installation

```bash
cd SA_emotion_detection_pipeline
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Configure API key:
```bash
cp .env.example .env
# Edit .env → set GROQ_API_KEY
```

## Usage

### Single audio file

```bash
python main.py interview.wav
python main.py interview.wav -c "John Doe" -p "Software Engineer"
```

### Audio file + existing transcript

```bash
python main.py interview.wav --transcript interview.txt
python main.py interview.wav -t notes.json
```

If a `.txt` or `.json` file with the same name exists next to the audio, it is picked up automatically — no `--transcript` flag needed.

### Folder of audio files

```bash
# Process all audio files recursively
python main.py recordings/

# Limit to first 5 files
python main.py recordings/ --limit 5

# Generate HTML reports
python main.py recordings/ --html-report

# Group results by subfolder (e.g. one folder per person)
python main.py recordings/ --group-by-folder
```

### Transcript auto-detection

The pipeline searches for transcripts in this order:
1. Same directory, same filename stem (`.txt` or `.json`)
2. Sibling `transcripts/` or `Transcription/` folder
3. Fuzzy match (case-insensitive, ignoring spaces/underscores/dashes)

If no transcript is found, Whisper transcribes the audio automatically.

### Docker (server deployment)

```bash
docker-compose build
docker-compose up -d hr-assessment

# Run inside container
docker-compose exec hr-assessment python main.py recordings/ --limit 10

# Start REST API
docker-compose up -d hr-assessment-api
# → http://localhost:8000
```

See `DOCKER_DEPLOYMENT.md` for full server guide.

## CLI Options

```
python main.py <input_path> [options]

Positional:
  input_path              Audio file or folder

Options:
  -t, --transcript PATH   Transcript file (.txt/.json) for single-file mode
  -c, --candidate-id ID   Candidate identifier
  -p, --position ROLE     Position / role context
  -o, --output-dir DIR    Output directory (default: ./outputs)
  -l, --limit N           Max files to process
  -q, --quiet             Suppress detailed output
  --whisper-model SIZE    tiny/base/small/medium/large (default: base)
  --html-report           Generate HTML report per file
  --group-by-folder       Group results by parent folder name
  --no-save               Don't write JSON output
```

## Python API

```python
from src.pipeline import HRAssessmentPipeline
from src.config import load_config

pipeline = HRAssessmentPipeline(load_config())

result = pipeline.process("interview.wav", candidate_id="C001")
pipeline.print_summary(result)

print(result.big_five.extraversion.score)   # 0-100
print(result.motivation.overall_level)       # High/Medium/Low
```

## REST API

```bash
uvicorn api:app --reload

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

## Output

### Big Five Personality Profile (0–100)

| Trait | What it measures |
|-------|-----------------|
| **Openness** | Creativity, curiosity, openness to experience |
| **Conscientiousness** | Organization, dependability, self-discipline |
| **Extraversion** | Sociability, assertiveness, positive emotions |
| **Agreeableness** | Cooperation, trust, helpfulness |
| **Neuroticism** | Emotional instability, anxiety, moodiness |

### Motivation Assessment
- **Level**: High / Medium / Low
- **Voice indicators**: energy, pitch variation, speaking rate
- **Content indicators**: future-oriented language, proactive statements

### Reports
- JSON assessment file per audio
- HTML visual report (with `--html-report`)
- `SUMMARY.json` per group (with `--group-by-folder`)

## Project Structure

```
SA_emotion_detection_pipeline/
├── main.py                    # Universal CLI entry point
├── api.py                     # FastAPI REST server
├── Dockerfile                 # Docker image
├── docker-compose.yml         # Docker orchestration
├── requirements.txt
├── .env.example
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
│   │   └── prompt_templates.py
│   └── utils/
│       ├── audio.py           # Audio utilities
│       └── reporting.py       # HTML report generation
├── examples/
│   └── demo.py
└── outputs/                   # Generated reports
```

## Voice → Personality Mapping

| Trait | High score indicators | Low score indicators |
|-------|----------------------|---------------------|
| **Openness** | Wide pitch range, expressive tone | Monotone, repetitive speech |
| **Conscientiousness** | Steady pace, few fillers, structured | Erratic pace, many fillers |
| **Extraversion** | Fast rate, high energy, loud | Slow, quiet, withdrawn |
| **Agreeableness** | Warm prosody, smooth pitch | Cold tone, high variance |
| **Neuroticism** | Unstable pitch, many pauses | Stable pitch, calm tone |

## Configuration

```bash
# .env
GROQ_API_KEY=gsk_...                       # Required
GROQ_MODEL=llama-3.3-70b-versatile         # Optional
WHISPER_MODEL=base                          # tiny/base/small/medium/large
WHISPER_DEVICE=cpu                          # cpu/cuda
```

```python
from src.config import PipelineConfig, WhisperConfig, GroqConfig

config = PipelineConfig(
    whisper=WhisperConfig(model_name="medium", device="cuda"),
    groq=GroqConfig(temperature=0.2),
)
```

## Requirements

- Python 3.9+
- FFmpeg
- Groq API key → https://console.groq.com

## Supported Audio Formats

WAV, MP3, M4A, AAC, FLAC, OGG, WebM

## License

MIT License
