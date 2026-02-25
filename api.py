"""
Slim Voice Feature Extraction API — FastAPI server.

Endpoints:
    POST /assess       — upload audio file, get full feature extraction JSON
    GET  /health       — health check + model status
    POST /assess/batch — upload multiple files

Usage:
    python api.py                        # default: 0.0.0.0:8000
    python api.py --port 8080            # custom port
    python api.py --host 127.0.0.1      # localhost only
"""

import os
import time
import argparse
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from slim_pipeline import SlimPipeline
from src.config import load_config

# ── Singleton pipeline (load model once, reuse across requests) ──
_pipeline: Optional[SlimPipeline] = None


def get_pipeline() -> SlimPipeline:
    global _pipeline
    if _pipeline is None:
        config = load_config()
        _pipeline = SlimPipeline(config)
        # Warm up: load MERaLiON model into GPU memory
        _pipeline.emotion_detector._load_model()
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load models on server start so first request is fast."""
    get_pipeline()
    yield


app = FastAPI(
    title="Slim Voice Feature Extraction API",
    description="Extract acoustic, prosodic, and emotion features from audio. No LLM, no transcription.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Health check ──

@app.get("/health")
async def health():
    pipe = get_pipeline()
    model_loaded = pipe.emotion_detector._model is not None
    device = pipe.emotion_detector.config.device
    return {
        "status": "ok",
        "pipeline": "slim",
        "model_loaded": model_loaded,
        "device": device,
        "model": pipe.emotion_detector.config.meralion_model,
    }


# ── Main assessment endpoint ──

@app.post("/assess")
async def assess(
    audio: UploadFile = File(..., description="Audio file (.wav, .mp3, .m4a, .webm, .ogg, .flac)"),
    language_profile: str = Form(default="non_native_english", description="Language profile for scoring"),
):
    """
    Extract all voice features from uploaded audio file.

    Returns JSON with:
    - prosody (pitch, energy, pauses, rhythm, speaking rate)
    - voice_quality (HNR, jitter, shimmer)
    - spectral (MFCC)
    - egemaps (88 acoustic features)
    - emotion_timeline (MERaLiON-SER per-segment emotions + VAD)
    - emotion_aggregates (dominant emotion, valence/arousal stats, arc type)
    - motivation_engagement (deterministic score + components)
    - l2_adjustments
    - paralinguistic_summary
    """
    allowed_ext = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"}
    suffix = Path(audio.filename).suffix.lower() if audio.filename else ".wav"
    if suffix not in allowed_ext:
        raise HTTPException(400, f"Unsupported format: {suffix}. Allowed: {allowed_ext}")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pipe = get_pipeline()
        t0 = time.time()
        result = pipe.process(
            audio_path=tmp_path,
            language_profile=language_profile,
            output_dir=None,  # don't save to disk
        )
        result["processing_time_seconds"] = round(time.time() - t0, 2)
        result["audio_file"] = audio.filename or "uploaded_audio"
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


# ── Batch endpoint ──

@app.post("/assess/batch")
async def assess_batch(
    files: list[UploadFile] = File(..., description="Multiple audio files"),
    language_profile: str = Form(default="non_native_english"),
):
    """Process multiple audio files and return results array."""
    results = []
    errors = []

    for audio in files:
        suffix = Path(audio.filename).suffix.lower() if audio.filename else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            pipe = get_pipeline()
            result = pipe.process(
                audio_path=tmp_path,
                language_profile=language_profile,
                output_dir=None,
            )
            result["audio_file"] = audio.filename
            results.append(result)
        except Exception as e:
            errors.append({"file": audio.filename, "error": str(e)})
        finally:
            os.unlink(tmp_path)

    return JSONResponse(content={
        "results": results,
        "errors": errors,
        "total": len(files),
        "success": len(results),
        "failed": len(errors),
    })


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Slim Voice Feature Extraction API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    args = parser.parse_args()

    print(f"Starting Slim Pipeline API on http://{args.host}:{args.port}")
    print(f"Docs: http://localhost:{args.port}/docs")
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        workers=1,
        reload=args.reload,
    )
