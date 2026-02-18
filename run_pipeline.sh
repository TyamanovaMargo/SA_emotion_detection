#!/bin/bash
# run_pipeline.sh - Automatically run HR Assessment Pipeline on 2 free GPUs
# Usage: ./run_pipeline.sh [audio_path] [options]
#
# Models are split across GPUs:
#   GPU 0 (inside container) = Whisper (speech-to-text, ~13GB)
#   GPU 1 (inside container) = Emotion2vec (emotion detection, ~6GB)
#
# Examples:
#   ./run_pipeline.sh "Team Recordings/Digvijay/Audio/" --skip-transcription --limit 2
#   ./run_pipeline.sh "Team Recordings/Digvijay/Audio/file.aac" --skip-transcription

set -e

# Persistent model cache — avoids re-downloading on every run
MODEL_CACHE_DIR="$(pwd)/.model_cache"
mkdir -p "$MODEL_CACHE_DIR"

# Load environment variables from .env file
if [ -f .env ]; then
  export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

echo "=== HR Assessment Pipeline - Multi-GPU Selection ==="
echo ""

# Find 2 GPUs with free memory (>20GB each)
echo "🔍 Searching for 2 available GPUs..."
FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F', ' '$2 > 20000 {print $1}' | head -2)

GPU_COUNT=$(echo "$FREE_GPUS" | wc -l)

if [ "$GPU_COUNT" -lt 2 ]; then
  # Try with 1 GPU (models will share + memory cleanup between stages)
  FREE_GPU=$(echo "$FREE_GPUS" | head -1)
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
  echo "⚠️  Only 1 free GPU found. Models will share GPU $FREE_GPU (with memory cleanup between stages)"
  GPU0=$FREE_GPU
  GPU1=$FREE_GPU
  DOCKER_GPUS="\"device=$FREE_GPU\""
  WHISPER_GPU=0
  EMOTION_GPU=0
else
  GPU0=$(echo "$FREE_GPUS" | sed -n '1p')
  GPU1=$(echo "$FREE_GPUS" | sed -n '2p')
  echo "✅ Found 2 free GPUs:"
  GPU0_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU0)
  GPU0_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU0)
  GPU1_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU1)
  GPU1_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU1)
  echo "   GPU $GPU0 ($GPU0_NAME, $(echo "scale=1; $GPU0_FREE/1024" | bc) GB free) → Whisper"
  echo "   GPU $GPU1 ($GPU1_NAME, $(echo "scale=1; $GPU1_FREE/1024" | bc) GB free) → MERaLiON-SER / Emotion"
  DOCKER_GPUS="\"device=$GPU0,$GPU1\""
  # Inside container, GPUs are remapped to 0,1
  WHISPER_GPU=0
  EMOTION_GPU=1
fi

echo ""

# Stop old container if exists
if docker ps -a --format '{{.Names}}' | grep -q '^hr-assessment-pipeline$'; then
  echo "🔄 Stopping existing container..."
  docker stop hr-assessment-pipeline 2>/dev/null || true
  docker rm hr-assessment-pipeline 2>/dev/null || true
fi

# Start container with 2 GPUs
echo "🚀 Starting container with GPUs $GPU0,$GPU1..."
docker run -d --name hr-assessment-pipeline \
  --gpus "$DOCKER_GPUS" \
  -v "$(pwd)/Team Recordings:/app/Team Recordings:ro" \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/.env:/app/.env:ro" \
  -v "$MODEL_CACHE_DIR:/root/.cache" \
  -e GROQ_API_KEY \
  -e HF_TOKEN \
  -e WHISPER_DEVICE=cuda \
  -e WHISPER_GPU=$WHISPER_GPU \
  -e EMOTION_DEVICE=cuda \
  -e EMOTION_GPU=$EMOTION_GPU \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HF_HOME=/root/.cache/huggingface \
  -e MODELSCOPE_CACHE=/root/.cache/modelscope \
  --shm-size=8g \
  hr-assessment-pipeline tail -f /dev/null

echo ""
echo "✅ Container started successfully"
echo "   Whisper → cuda:$WHISPER_GPU (host GPU $GPU0)"
echo "   Emotion → cuda:$EMOTION_GPU (host GPU $GPU1)"
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
  echo "   # Check GPU usage inside container"
  echo "   docker exec hr-assessment-pipeline nvidia-smi"
  echo ""
fi
