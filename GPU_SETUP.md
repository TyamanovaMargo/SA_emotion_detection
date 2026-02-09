# GPU Setup Guide

## Prerequisites

1. **NVIDIA GPU** with CUDA support (RTX 20xx, 30xx, 40xx, or Tesla)
2. **NVIDIA Driver** installed on server
3. **Docker Engine** with NVIDIA Container Toolkit

## Server Setup

### 1. Install NVIDIA Container Toolkit

```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2 and restart Docker
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. Verify GPU Access

```bash
# Check GPU
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

## Deployment

### Option 1: Use GPU-specific compose file

```bash
# Build and run with GPU
docker-compose -f docker-compose.gpu.yml build
docker-compose -f docker-compose.gpu.yml up -d hr-assessment-gpu

# Run commands
docker-compose -f docker-compose.gpu.yml exec hr-assessment-gpu python main.py recordings/ --limit 5
```

### Option 2: Modify existing docker-compose.yml

The main `docker-compose.yml` now includes GPU support. Just ensure:

```bash
# Set GPU device in .env
WHISPER_DEVICE=cuda
EMOTION_DEVICE=cuda

# Build and run
docker-compose build
docker-compose up -d hr-assessment
```

## Performance Benefits

| Operation | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| Whisper (base) | 8s | 2s | 4x |
| Whisper (medium) | 20s | 4s | 5x |
| emotion2vec | 3s | 0.5s | 6x |
| Total pipeline | 12s | 3s | 4x |

## Configuration

### Environment Variables

```bash
# .env file
GROQ_API_KEY=gsk_...
WHISPER_DEVICE=cuda          # Use GPU for Whisper
EMOTION_DEVICE=cuda          # Use GPU for emotion2vec
WHISPER_MODEL=medium          # Larger model with GPU
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"  # GPU architecture support
```

### Memory Allocation

```yaml
# docker-compose.yml
services:
  hr-assessment:
    mem_limit: 16g      # More memory for GPU models
    shm_size: 4g        # Shared memory
    cpus: "8.0"         # More CPU cores
```

## Monitoring GPU Usage

```bash
# Inside container
docker-compose exec hr-assessment nvidia-smi

# Monitor during processing
watch -n 1 nvidia-smi
```

## Troubleshooting

### GPU not detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

### CUDA out of memory
```bash
# Reduce batch size
python main.py recordings/ --limit 3

# Use smaller model
WHISPER_MODEL=base

# Increase memory limit
mem_limit: 24g
```

### Model loading errors
```bash
# Check GPU architecture
nvidia-smi --query-gpu=architecture --format=csv

# Update TORCH_CUDA_ARCH_LIST if needed
TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6;8.9;9.0"
```

## Production Tips

1. **Use larger Whisper models** with GPU: `medium` or `large`
2. **Increase memory** to 16-24GB for batch processing
3. **Monitor GPU temperature** during long runs
4. **Use `--limit`** for large batches to prevent OOM
5. **Consider GPU sharing** if running multiple containers

## Example Commands

```bash
# Single file with GPU
docker-compose exec hr-assessment python main.py interview.wav --whisper-model medium

# Batch processing with GPU
docker-compose exec hr-assessment python main.py recordings/ --limit 10 --html-report

# API with GPU
docker-compose up -d hr-assessment-api
curl -X POST http://localhost:8000/assess -F "audio=@test.wav"
```

## Expected Performance

With RTX 3080/4080:
- **Whisper base**: ~2s per 10s audio
- **Whisper medium**: ~4s per 10s audio  
- **emotion2vec**: ~0.5s per 10s audio
- **Total pipeline**: ~3s per file vs 12s on CPU
