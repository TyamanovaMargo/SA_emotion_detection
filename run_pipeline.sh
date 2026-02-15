#!/bin/bash
# run_pipeline.sh - Automatically run HR Assessment Pipeline on free GPU
# Usage: ./run_pipeline.sh [audio_path] [options]
#
# Examples:
#   ./run_pipeline.sh "Team Recordings/Digvijay/Audio/" --skip-transcription --limit 2
#   ./run_pipeline.sh "Team Recordings/Digvijay/Audio/file.aac" --skip-transcription

set -e

echo "=== HR Assessment Pipeline - Auto GPU Selection ==="
echo ""

# Find GPU with most free memory (>20GB)
echo "🔍 Searching for available GPU..."
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F', ' '$2 > 20000 {print $1; exit}')

if [ -z "$FREE_GPU" ]; then
  echo "❌ No free GPU found (need >20GB free memory)"
  echo ""
  echo "Current GPU usage:"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  echo ""
  echo "💡 Options:"
  echo "   1. Wait for other jobs to finish"
  echo "   2. Use CPU mode (slower): set EMOTION_DEVICE=cpu"
  echo "   3. Check again: ./check_gpu.sh"
  exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $FREE_GPU)
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $FREE_GPU)

echo "✅ Found free GPU: $FREE_GPU ($GPU_NAME, $(echo "scale=1; $GPU_FREE/1024" | bc) GB free)"
echo ""

# Stop old container if exists
if docker ps -a --format '{{.Names}}' | grep -q '^hr-assessment-pipeline$'; then
  echo "🔄 Stopping existing container..."
  docker stop hr-assessment-pipeline 2>/dev/null || true
  docker rm hr-assessment-pipeline 2>/dev/null || true
fi

# Start container on free GPU
echo "🚀 Starting container on GPU $FREE_GPU..."
docker run -d --name hr-assessment-pipeline \
  --gpus "\"device=$FREE_GPU\"" \
  -v "$(pwd)/Team Recordings:/app/Team Recordings:ro" \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/.env:/app/.env:ro" \
  -e GROQ_API_KEY \
  -e CUDA_VISIBLE_DEVICES=$FREE_GPU \
  -e WHISPER_DEVICE=cuda \
  -e EMOTION_DEVICE=cuda \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --shm-size=8g \
  sa_emotion_detection-hr-assessment tail -f /dev/null

echo ""
echo "✅ Container started successfully on GPU $FREE_GPU"
echo ""

# If arguments provided, run the pipeline
if [ $# -gt 0 ]; then
  echo "▶️  Running pipeline with arguments: $@"
  echo ""
  docker exec hr-assessment-pipeline python main.py "$@"
else
  echo "📝 Container is ready. Run pipeline with:"
  echo ""
  echo "   # Process single file"
  echo "   docker exec hr-assessment-pipeline python main.py \"Team Recordings/Digvijay/Audio/file.aac\" --skip-transcription"
  echo ""
  echo "   # Process folder (limit 2 files)"
  echo "   docker exec hr-assessment-pipeline python main.py \"Team Recordings/Digvijay/Audio/\" --skip-transcription --limit 2"
  echo ""
  echo "   # Process all files in folder"
  echo "   docker exec hr-assessment-pipeline python main.py \"Team Recordings/Digvijay/Audio/\" --skip-transcription"
  echo ""
  echo "   # Check GPU usage"
  echo "   docker exec hr-assessment-pipeline nvidia-smi"
  echo ""
fi
