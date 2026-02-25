#!/usr/bin/env python3
"""
Pre-download all ML models to local HuggingFace cache.

Run once before deployment so the server never downloads at runtime.

Usage:
    python download_models.py
    HF_HOME=/models python download_models.py   # custom cache dir
"""

import os
import sys


def main():
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set.")
        print("MERaLiON-SER-v1 is a gated model — you need a HuggingFace token.")
        print("  1. Go to https://huggingface.co/MERaLiON/MERaLiON-SER-v1")
        print("  2. Request access")
        print("  3. Set HF_TOKEN=hf_... in your .env or environment")
        sys.exit(1)

    print(f"HF cache dir: {hf_home}")
    print(f"HF token:     {'set (' + hf_token[:8] + '...)' if hf_token else 'NOT SET'}")
    print()

    # ── 1. MERaLiON-SER-v1 (emotion detection) ──
    model_id = "MERaLiON/MERaLiON-SER-v1"
    print(f"Downloading {model_id}...")
    try:
        from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

        processor = AutoFeatureExtractor.from_pretrained(
            model_id, trust_remote_code=True, token=hf_token,
        )
        model = AutoModelForAudioClassification.from_pretrained(
            model_id, trust_remote_code=True, token=hf_token,
        )
        print(f"  ✅ {model_id} downloaded and cached")
        del model, processor
    except Exception as e:
        print(f"  ❌ Failed to download {model_id}: {e}")
        sys.exit(1)

    print()
    print("=" * 50)
    print("All models downloaded successfully!")
    print(f"Cache location: {hf_home}")
    print()
    print("To verify cache contents:")
    print(f"  ls -la {hf_home}/hub/")
    print()
    print("Set HF_HOME in production to point to this cache dir.")
    print("=" * 50)


if __name__ == "__main__":
    main()
