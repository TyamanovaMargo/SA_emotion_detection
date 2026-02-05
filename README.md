# HR Personality & Motivation Assessment Pipeline

A comprehensive Python pipeline for analyzing candidate personality traits and motivation levels from voice recordings using AI-powered speech analysis and Groq.

## Features

- **Speech-to-Text**: Whisper-based transcription with filler word detection
- **Prosody Analysis**: Speaking rate, pitch, energy, and pause detection
- **Emotion Detection**: emotion2vec-based emotion recognition with fallback
- **Acoustic Features**: eGeMAPS feature extraction via OpenSMILE
- **AI Assessment**: Groq-powered Big Five personality and motivation analysis (LLaMA 3.3 70B)
- **Multiple Interfaces**: CLI, Python API, and REST API

## Installation

### 1. Clone and Setup

```bash
cd SA_emotion_detection_pipeline
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

## Quick Start

### Docker Deployment (Recommended for Server)

```bash
# 1. Build the Docker image
docker-compose build

# 2. Start the container
docker-compose up -d hr-assessment

# 3. Run commands inside container
docker-compose exec hr-assessment python main.py "audio/folder/" --limit 10

# 4. Process team recordings
docker-compose exec hr-assessment python process_team_recordings.py "Team Recordings" --person Mikhail

# 5. Start API service
docker-compose up -d hr-assessment-api
# API available at http://localhost:8000
```

**Memory Configuration:**
- Default: 8GB RAM, 2GB shared memory, 4 CPUs
- Adjust in `docker-compose.yml` if needed:
  ```yaml
  mem_limit: 16g  # Increase for larger batches
  cpus: "8.0"     # Use more cores
  ```

### Command Line

```bash
# Process a single audio file
python main.py path/to/interview.wav

# Process all audio files in a folder (searches subfolders recursively)
python main.py audio/candidates/

# Process only first 10 files (useful for testing)
python main.py audio/candidates/ --limit 10

# Process folder with position info
python main.py audio/candidates/ --position "Software Engineer"

# Generate HTML reports for all files
python main.py audio/candidates/ --html-report

# Use larger Whisper model for better accuracy
python main.py interview.wav --whisper-model medium
```

### Python API

```python
from src.pipeline import HRAssessmentPipeline
from src.config import load_config

# Initialize pipeline
config = load_config()
pipeline = HRAssessmentPipeline(config)

# Process audio file
result = pipeline.process(
    audio_path="interview.wav",
    candidate_id="C001",
    position="Software Engineer"
)

# Print summary
pipeline.print_summary(result)

# Access specific results
print(f"Extraversion: {result.big_five.extraversion.score}/100")
print(f"Motivation: {result.motivation.overall_level}")
print(f"HR Summary: {result.hr_summary}")
```

### REST API

```bash
# Start the server
uvicorn api:app --reload

# Upload audio for assessment
curl -X POST "http://localhost:8000/assess" \
  -F "audio=@interview.wav" \
  -F "candidate_id=C001" \
  -F "position=Software Engineer"
```

## Output Format

### Big Five Personality Profile (0-100)
- **Openness**: Creativity, curiosity, openness to experience
- **Conscientiousness**: Organization, dependability, self-discipline
- **Extraversion**: Sociability, assertiveness, positive emotions
- **Agreeableness**: Cooperation, trust, helpfulness
- **Neuroticism**: Emotional instability, anxiety, moodiness

### Motivation Assessment
- **Level**: High / Medium / Low
- **Voice Indicators**: Energy, pitch variation, speaking rate
- **Content Indicators**: Future-oriented language, proactive statements

### HR-Relevant Output
- Top trait strengths
- Motivation-related strengths
- Areas for development
- 3-sentence HR summary

## Project Structure

```
SA_emotion_detection_pipeline/
├── main.py                 # CLI entry point
├── api.py                  # FastAPI server
├── requirements.txt        # Dependencies
├── .env.example           # Environment template
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── pipeline.py        # Main pipeline orchestrator
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py     # Pydantic data models
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── transcription.py  # Whisper transcription
│   │   ├── prosody.py        # Prosody extraction
│   │   ├── emotion.py        # Emotion detection
│   │   └── egemaps.py        # Acoustic features
│   ├── assessment/
│   │   ├── __init__.py
│   │   ├── claude_assessor.py    # Claude integration
│   │   └── prompt_templates.py   # HR assessment prompts
│   └── utils/
│       ├── __init__.py
│       ├── audio.py       # Audio utilities
│       └── reporting.py   # HTML/PDF reports
├── examples/
│   └── demo.py            # Demo scripts
└── outputs/               # Generated reports
```

## Voice Feature Interpretation

### Motivation Indicators

| Feature | High Motivation | Low Motivation |
|---------|-----------------|----------------|
| Speaking Rate | Fast (>140 wpm) | Slow (<100 wpm) |
| Energy Level | High | Low |
| Pitch Variance | Moderate-High | Low (monotone) |
| Pauses | Few, purposeful | Many, hesitant |
| Filler Words | Minimal | Frequent |
| Dominant Emotion | Happy, Confident | Neutral, Sad |

### Personality Mapping

| Trait | Voice Indicators |
|-------|------------------|
| Extraversion | High energy, fast rate, pitch variation |
| Conscientiousness | Steady pace, clear structure, low fillers |
| Agreeableness | Warm prosody, smooth pitch, moderate loudness |
| Neuroticism | Unstable pitch, high variance, uncertainty pauses |
| Openness | Varied pitch range, expressive tone, diverse vocabulary |

## Configuration

### Environment Variables

```bash
GROQ_API_KEY=gsk_...          # Required
GROQ_MODEL=llama-3.3-70b-versatile  # Optional (or mixtral-8x7b-32768)
WHISPER_MODEL=base            # tiny/base/small/medium/large
WHISPER_DEVICE=cpu            # cpu/cuda
```

### Programmatic Configuration

```python
from src.config import PipelineConfig, WhisperConfig, GroqConfig

config = PipelineConfig(
    whisper=WhisperConfig(model_name="medium", device="cuda"),
    groq=GroqConfig(temperature=0.2),
)
pipeline = HRAssessmentPipeline(config)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/assess` | POST | JSON assessment |
| `/assess/html` | POST | HTML report |

## Requirements

- Python 3.9+
- FFmpeg (for audio processing)
- Groq API key (get one at https://console.groq.com)

## Supported Audio Formats

- WAV, MP3, M4A, FLAC, OGG, WebM

## License

MIT License
