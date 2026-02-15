# GPU Optimization Guide

## Overview

The pipeline now supports automatic GPU detection and optimization for:
- **NVIDIA GPUs** (CUDA)
- **Apple Silicon** (MPS - M1/M2/M3 chips)
- **CPU fallback** (automatic)

## Quick Start

### 1. Test GPU Availability

```bash
.venv/bin/python test_gpu.py
```

This will show:
- Available devices (CUDA/MPS/CPU)
- GPU memory
- Recommended settings
- Model loading times

### 2. Enable GPU (Automatic)

Create or update `.env` file:

```bash
# Auto-detect best device (recommended)
EMOTION_DEVICE=auto
WHISPER_DEVICE=auto
EMOTION_BATCH_SIZE=0  # Auto-detect based on GPU memory

# Or specify explicitly:
# For NVIDIA GPU:
# EMOTION_DEVICE=cuda
# WHISPER_DEVICE=cuda

# For Apple Silicon:
# EMOTION_DEVICE=mps
# WHISPER_DEVICE=mps

# For CPU only:
# EMOTION_DEVICE=cpu
# WHISPER_DEVICE=cpu
```

### 3. Run with GPU

```bash
.venv/bin/python main.py \
  "Team Recordings/Interview audios/daniel_kapkov_audio_0.webm" \
  --position "Data Scientist" \
  --skip-transcription
```

The pipeline will automatically:
- Detect optimal device
- Set batch size based on GPU memory
- Manage GPU memory efficiently
- Fall back to CPU if GPU fails

## Performance Comparison

### CPU vs GPU Processing Times (single file)

| Component | CPU | NVIDIA GPU | Apple M1/M2 |
|-----------|-----|------------|-------------|
| Emotion Detection | 10-15s | 2-3s | 4-6s |
| Whisper (base) | 20-30s | 5-8s | 8-12s |
| Total Pipeline | 40-60s | 15-20s | 20-30s |

### Batch Processing (10 files)

| Device | Time | Speedup |
|--------|------|---------|
| CPU | 6-8 min | 1x |
| NVIDIA GPU | 2-3 min | 3x |
| Apple M1/M2 | 3-4 min | 2x |

## GPU Memory Requirements

### Minimum GPU Memory

- **Emotion Detection**: 2 GB
- **Whisper (base)**: 1 GB
- **Whisper (small)**: 2 GB
- **Whisper (medium)**: 5 GB
- **Total Pipeline**: 4 GB recommended

### Batch Size Auto-Detection

The pipeline automatically sets batch size based on available GPU memory:

| GPU Memory | Batch Size |
|------------|------------|
| < 4 GB | 1-2 |
| 4-8 GB | 4 |
| 8-16 GB | 8 |
| 16+ GB | 16 |

## Advanced Configuration

### Manual Batch Size

If auto-detection doesn't work well, set manually:

```bash
# In .env
EMOTION_BATCH_SIZE=4  # 1, 2, 4, 8, or 16
```

### GPU Memory Fraction

The pipeline uses 70% of GPU memory by default. To change:

Edit `src/extractors/emotion.py`:
```python
setup_gpu_memory(self.config.device, memory_fraction=0.8)  # Use 80%
```

### Mixed Precision (NVIDIA only)

For faster inference on Ampere GPUs (RTX 30xx, A100):

```bash
# In .env
TORCH_ALLOW_TF32=1
```

Already enabled by default in `src/utils/device.py`.

## Troubleshooting

### "CUDA out of memory"

1. Reduce batch size:
   ```bash
   EMOTION_BATCH_SIZE=1
   ```

2. Use smaller Whisper model:
   ```bash
   WHISPER_MODEL=tiny  # or base (default)
   ```

3. Process files one at a time (not batch)

### "MPS backend not available"

Apple Silicon requires:
- macOS 12.3+
- PyTorch 1.12+

Update PyTorch:
```bash
pip install --upgrade torch torchaudio
```

### GPU not detected

Check CUDA/MPS availability:
```bash
.venv/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
```

### Slower than CPU

This can happen if:
- GPU memory is too small (< 2 GB)
- Batch size is 1 (no parallelization benefit)
- Model loading overhead dominates (for single files)

GPU is faster for:
- Batch processing (5+ files)
- Larger Whisper models (medium/large)
- Long audio files (> 2 minutes)

## Docker with GPU

### NVIDIA GPU

Use `docker-compose.gpu.yml`:

```bash
docker-compose -f docker-compose.gpu.yml build
docker-compose -f docker-compose.gpu.yml up -d hr-assessment-gpu
docker-compose -f docker-compose.gpu.yml exec hr-assessment-gpu python main.py \
  "Team Recordings/Interview audios/daniel_kapkov_audio_0.webm" \
  --position "Data Scientist" \
  --skip-transcription
```

Requires:
- NVIDIA Docker runtime
- CUDA 12.1+
- nvidia-docker2

### Apple Silicon

Docker on Mac doesn't support MPS acceleration yet. Use local execution instead.

## Monitoring GPU Usage

### NVIDIA

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or during processing
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 1
```

### Apple Silicon

```bash
# Activity Monitor > Window > GPU History
# Or use powermetrics (requires sudo)
sudo powermetrics --samplers gpu_power -i 1000
```

## Best Practices

1. **Use GPU for batch processing** (5+ files)
2. **Set EMOTION_DEVICE=auto** for automatic detection
3. **Monitor GPU memory** during first run
4. **Use smaller Whisper models** (base/small) for real-time
5. **Process in batches of 10-20 files** for optimal throughput
6. **Clear GPU cache** between large batches (automatic)

## Performance Tips

### For Maximum Speed (NVIDIA GPU)

```bash
# .env
EMOTION_DEVICE=cuda
WHISPER_DEVICE=cuda
WHISPER_MODEL=base
EMOTION_BATCH_SIZE=8
```

### For Maximum Accuracy

```bash
# .env
EMOTION_DEVICE=cuda
WHISPER_DEVICE=cuda
WHISPER_MODEL=medium
EMOTION_BATCH_SIZE=4
```

### For Low Memory (< 4 GB GPU)

```bash
# .env
EMOTION_DEVICE=cuda
WHISPER_DEVICE=cpu  # Keep Whisper on CPU
WHISPER_MODEL=tiny
EMOTION_BATCH_SIZE=1
```
