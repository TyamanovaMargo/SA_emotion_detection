# Docker Guide for HR Assessment Pipeline

## Prerequisites

- Docker installed
- Docker Compose installed
- GROQ API key

## Setup

1. **Set your GROQ API key** in `.env` file:
```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

2. **Build the Docker image**:
```bash
docker-compose build
```

## Running the Pipeline

### Option 1: Interactive Mode (Recommended)

Start the container in background:
```bash
docker-compose up -d hr-assessment
```

Execute commands inside the container:
```bash
# Process a single audio file
docker-compose exec hr-assessment python main.py \
  "Team Recordings/Interview audios/daniel_kapkov_audio_0.webm" \
  --position "Data Scientist" \
  --skip-transcription

# Process with transcript
docker-compose exec hr-assessment python main.py \
  "Team Recordings/Idan/Audio/Idan_Question 1 Astrology.m4a" \
  --position "Data Scientist"

# Batch process all files in a folder
docker-compose exec hr-assessment python main.py \
  "Team Recordings/Interview audios" \
  --batch \
  --skip-transcription
```

Stop the container:
```bash
docker-compose down
```

### Option 2: One-off Command

Run a single command without keeping container running:
```bash
docker-compose run --rm hr-assessment python main.py \
  "Team Recordings/Interview audios/daniel_kapkov_audio_0.webm" \
  --position "Data Scientist" \
  --skip-transcription
```

### Option 3: GPU Mode (if you have NVIDIA GPU)

Use the GPU-enabled compose file:
```bash
# Build
docker-compose -f docker-compose.gpu.yml build

# Run
docker-compose -f docker-compose.gpu.yml up -d hr-assessment-gpu

# Execute
docker-compose -f docker-compose.gpu.yml exec hr-assessment-gpu python main.py \
  "Team Recordings/Interview audios/daniel_kapkov_audio_0.webm" \
  --position "Data Scientist" \
  --skip-transcription
```

## Viewing Results

Results are saved to `./outputs/` folder on your host machine (mounted as volume).

```bash
# List all results
ls -lh outputs/

# View a specific result
cat outputs/daniel_kapkov_audio_0_20260215_093000_assessment.json | jq .
```

## Troubleshooting

### Out of Memory
Increase memory limit in `docker-compose.yml`:
```yaml
mem_limit: 16g  # increase from 8g
```

### Permission Issues
If you get permission errors on `outputs/` folder:
```bash
sudo chown -R $USER:$USER outputs/
```

### Audio Format Issues
The pipeline supports `.webm`, `.m4a`, `.wav`, `.mp3` via ffmpeg.
If you get audio loading errors, check that ffmpeg is working:
```bash
docker-compose exec hr-assessment ffmpeg -version
```

## Environment Variables

You can customize behavior via environment variables in `.env`:

```bash
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
WHISPER_MODEL=base
WHISPER_DEVICE=cpu
EMOTION_DEVICE=cpu
```

## Clean Up

Remove all containers and images:
```bash
docker-compose down --rmi all --volumes
```
