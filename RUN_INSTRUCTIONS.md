# Quick Start Instructions

## Running Locally (Recommended for Mac)

### 1. Single File Processing

```bash
.venv/bin/python main.py \
  "Team Recordings/Interview audios/daniel_kapkov_audio_0.webm" \
  --position "Data Scientist" \
  --skip-transcription
```

### 2. Batch Processing (All Files in Folder)

```bash
.venv/bin/python main.py \
  "Team Recordings/Interview audios" \
  --batch \
  --skip-transcription
```

### 3. With Transcript (if available)

```bash
.venv/bin/python main.py \
  "Team Recordings/Idan/Audio/Idan_Question 1 Astrology.m4a" \
  --position "Data Scientist"
```

## Running with Docker

### 1. Start Docker Desktop
Open Docker Desktop application on your Mac.

### 2. Build Image (First Time Only)

```bash
docker-compose build
```

### 3. Start Container

```bash
docker-compose up -d hr-assessment
```

### 4. Run Processing

```bash
# Single file
docker-compose exec hr-assessment python main.py \
  "Team Recordings/Interview audios/daniel_kapkov_audio_0.webm" \
  --position "Data Scientist" \
  --skip-transcription

# Batch processing
docker-compose exec hr-assessment python main.py \
  "Team Recordings/Interview audios" \
  --batch \
  --skip-transcription
```

### 5. Stop Container

```bash
docker-compose down
```

## Output

Results are saved to `outputs/` folder:
- JSON files with full assessment
- Motivation level: High/Medium/Low
- Big Five personality scores
- Voice indicators

## Troubleshooting

### "Process killed" Error
Your Mac ran out of memory. Solutions:
1. Close other applications
2. Process files one at a time (not batch)
3. Restart your Mac

### "Docker daemon not running"
1. Open Docker Desktop application
2. Wait for it to fully start (green icon in menu bar)
3. Try command again

### Audio Format Issues
Supported formats: `.webm`, `.m4a`, `.aac`, `.wav`, `.mp3`
All formats are automatically converted via ffmpeg.

## Memory Usage

- Single file: ~2-4 GB RAM
- Batch processing: ~4-8 GB RAM
- Docker: 8 GB allocated (configurable in docker-compose.yml)

## Tips

1. **Use `--skip-transcription`** if you don't need text analysis (faster, less memory)
2. **Process in batches of 5-10 files** to avoid memory issues
3. **Check `outputs/` folder** for results after each run
4. **Use Docker** if you have memory issues on Mac (better isolation)
