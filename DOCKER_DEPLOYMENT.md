# Docker Deployment Guide

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 8GB RAM available
- GROQ API key

## Setup

### 1. Configure Environment

Create `.env` file:
```bash
cp .env.example .env
```

Edit `.env` and add your API key:
```
GROQ_API_KEY=your_actual_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
WHISPER_MODEL=base
WHISPER_DEVICE=cpu
```

### 2. Build Docker Image

```bash
docker-compose build
```

This will:
- Install Python 3.11
- Install system dependencies (ffmpeg, libsndfile)
- Install all Python packages
- Copy application code

**Build time:** ~5-10 minutes (first time)

### 3. Start Container

```bash
# Start in background
docker-compose up -d hr-assessment

# Check logs
docker-compose logs -f hr-assessment
```

## Usage

### Process Single Audio File

```bash
docker-compose exec hr-assessment python main.py audio/interview.wav \
  --candidate-id "C001" \
  --position "Software Engineer"
```

### Process Folder (Batch)

```bash
docker-compose exec hr-assessment python main.py "audio/Acted Emotional Speech Dynamic Database/" \
  --limit 20 \
  --position "Emotion Analysis"
```

### Process Team Recordings

```bash
# Process one person
docker-compose exec hr-assessment python process_team_recordings.py "Team Recordings" \
  --person Mikhail

# Process all team members (one by one to avoid memory issues)
for person in Mikhail Digvijay Dima Idan Sibi Tzafrir; do
  docker-compose exec hr-assessment python process_team_recordings.py "Team Recordings" \
    --person "$person"
done
```

### Start REST API

```bash
# Start API service
docker-compose up -d hr-assessment-api

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

Test API:
```bash
curl -X POST http://localhost:8000/assess \
  -F "audio=@audio/interview.wav" \
  -F "candidate_id=C001" \
  -F "position=Engineer"
```

## Volume Mounts

Data is shared between host and container:

```yaml
volumes:
  - ./audio:/app/audio:ro                    # Input audio (read-only)
  - ./Team Recordings:/app/Team Recordings:ro # Team data (read-only)
  - ./outputs:/app/outputs                   # Results (read-write)
  - ./team_reports:/app/team_reports         # Team reports (read-write)
```

**Results are saved on your host machine** in `outputs/` and `team_reports/`.

## Resource Configuration

### Memory Issues?

If you get "Killed" errors, increase memory:

```yaml
# docker-compose.yml
services:
  hr-assessment:
    mem_limit: 16g      # Increase from 8g
    shm_size: 4g        # Increase from 2g
    cpus: "8.0"         # Use more cores
```

Then restart:
```bash
docker-compose down
docker-compose up -d hr-assessment
```

### GPU Support (Optional)

For faster processing with CUDA GPU:

```yaml
# docker-compose.yml
services:
  hr-assessment:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - WHISPER_DEVICE=cuda
```

Requires: NVIDIA Docker runtime

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs hr-assessment

# Check if port is in use
lsof -i :8000

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Out of memory

```bash
# Check container stats
docker stats hr-assessment

# Increase memory limit in docker-compose.yml
# Process fewer files at once (use --limit)
```

### Permission errors

```bash
# Fix output directory permissions
chmod -R 777 outputs team_reports
```

### Models not downloading

```bash
# Enter container and check internet
docker-compose exec hr-assessment bash
ping google.com

# Manually download models
python -c "import whisper; whisper.load_model('base')"
```

## Production Deployment

### Server Setup

1. **Install Docker on server:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```

2. **Copy project to server:**
```bash
scp -r SA_emotion_detection_pipeline user@server:/opt/
```

3. **Configure and start:**
```bash
ssh user@server
cd /opt/SA_emotion_detection_pipeline
nano .env  # Add API key
docker-compose up -d
```

### Auto-restart on Failure

```yaml
# docker-compose.yml
services:
  hr-assessment:
    restart: unless-stopped
```

### Run as Background Service

```bash
# Start on boot
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f --tail=100
```

### Monitoring

```bash
# Resource usage
docker stats hr-assessment

# Disk usage
docker system df

# Clean up old images
docker system prune -a
```

## Batch Processing Script

Create `batch_process.sh`:

```bash
#!/bin/bash
set -e

PERSONS=("Mikhail" "Digvijay" "Dima" "Idan" "Sibi" "Tzafrir")

for person in "${PERSONS[@]}"; do
  echo "Processing $person..."
  docker-compose exec -T hr-assessment python process_team_recordings.py \
    "Team Recordings" --person "$person"
  
  # Wait between persons to avoid memory buildup
  sleep 10
done

echo "All team members processed!"
```

Run:
```bash
chmod +x batch_process.sh
./batch_process.sh
```

## Cleanup

```bash
# Stop containers
docker-compose down

# Remove volumes (WARNING: deletes output data)
docker-compose down -v

# Remove images
docker rmi $(docker images -q hr-assessment*)
```

## Performance Tips

1. **Use SSD storage** for audio files
2. **Allocate 8-16GB RAM** minimum
3. **Use `--limit` flag** for large folders
4. **Process one person at a time** for team recordings
5. **Use `base` Whisper model** (faster) or `medium` (more accurate)
6. **Enable GPU** if available (10x faster transcription)

## Support

For issues:
1. Check logs: `docker-compose logs hr-assessment`
2. Check resources: `docker stats`
3. Verify `.env` file has valid API key
4. Ensure audio files are in correct format
