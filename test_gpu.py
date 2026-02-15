#!/usr/bin/env python3
"""Test GPU availability and performance."""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.device import print_device_info, get_optimal_device
from src.config import load_config


def main():
    """Test GPU setup and performance."""
    print("\n" + "=" * 60)
    print("GPU PERFORMANCE TEST")
    print("=" * 60 + "\n")
    
    # Print device info
    print_device_info()
    
    # Load config
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    config = load_config()
    print(f"Whisper Device: {config.whisper.device}")
    print(f"Emotion Device: {config.emotion.device}")
    print(f"Emotion Batch Size: {config.emotion.batch_size}")
    
    # Test model loading
    print("\n" + "=" * 60)
    print("MODEL LOADING TEST")
    print("=" * 60)
    
    try:
        from src.extractors import EmotionDetector
        print("\n1. Testing Emotion Detector...")
        start = time.time()
        detector = EmotionDetector(config.emotion)
        detector._load_model()
        elapsed = time.time() - start
        print(f"   ✓ Loaded in {elapsed:.2f}s on {detector.config.device}")
        print(f"   Batch size: {detector.config.batch_size}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    try:
        from src.extractors import WhisperTranscriber
        print("\n2. Testing Whisper Transcriber...")
        start = time.time()
        transcriber = WhisperTranscriber(config.whisper)
        _ = transcriber.model  # Trigger lazy loading
        elapsed = time.time() - start
        print(f"   ✓ Loaded in {elapsed:.2f}s on {transcriber.config.device}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60 + "\n")
    
    # Recommendations
    optimal = get_optimal_device()
    print("RECOMMENDATIONS:")
    if optimal == "cuda":
        print("✓ NVIDIA GPU detected - excellent performance expected")
        print("  Set EMOTION_DEVICE=cuda and WHISPER_DEVICE=cuda in .env")
    elif optimal == "mps":
        print("✓ Apple Silicon detected - good performance expected")
        print("  Set EMOTION_DEVICE=mps and WHISPER_DEVICE=mps in .env")
    else:
        print("⚠ No GPU detected - using CPU (slower)")
        print("  Consider using a machine with GPU for faster processing")
    
    print("\nTo enable GPU, add to .env file:")
    print(f"  EMOTION_DEVICE={optimal}")
    print(f"  WHISPER_DEVICE={optimal}")
    print(f"  EMOTION_BATCH_SIZE=0  # Auto-detect based on GPU memory")
    print()


if __name__ == "__main__":
    main()
