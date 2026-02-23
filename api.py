"""
FastAPI server for HR Assessment Pipeline.

Run with: uvicorn api:app --reload
"""

import tempfile
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import HRAssessmentPipeline
from src.config import load_config
from src.models.schemas import HRAssessmentResult
from src.utils.reporting import generate_html_report
from src.utils.scoring import (
    score_to_label,
    validate_strengths_and_dev_areas,
    compute_final_assessment,
    generate_hr_summary_from_scores,
    compute_content_voice_alignment,
)


app = FastAPI(
    title="HR Voice Assessment API",
    description="Analyze candidate personality and motivation from voice recordings",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[HRAssessmentPipeline] = None


def get_pipeline() -> HRAssessmentPipeline:
    """Get or create the pipeline instance."""
    global pipeline
    if pipeline is None:
        config = load_config()
        pipeline = HRAssessmentPipeline(config)
    return pipeline


class AssessmentResponse(BaseModel):
    """API response model — mirrors the full JSON report from main.py."""
    success: bool
    candidate_id: Optional[str] = None
    position: Optional[str] = None
    personality_assessment: dict
    final_assessment: dict
    motivation_engagement: dict
    emotion_summary: Optional[dict] = None
    voice_analysis: Optional[dict] = None
    hr_summary: str
    pipeline_metadata: Optional[dict] = None


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "HR Voice Assessment API",
        "version": "1.0.0",
        "endpoints": {
            "POST /assess": "Upload audio file for assessment",
            "GET /health": "Health check",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/assess", response_model=AssessmentResponse)
async def assess_candidate(
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)"),
    candidate_id: Optional[str] = Form(None, description="Candidate identifier"),
    position: Optional[str] = Form(None, description="Position being applied for"),
):
    """
    Assess a candidate from their voice recording.
    
    Upload an audio file containing the candidate's speech (e.g., interview response)
    and receive a comprehensive personality and motivation assessment.
    """
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
    file_ext = Path(audio.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        pipe = get_pipeline()
        result = pipe.process(
            audio_path=Path(tmp_path),
            candidate_id=candidate_id,
            position=position,
            save_output=False,
        )

        # Build full report (same logic as main.py process_single)
        va = result.voice_analysis or {}
        b5 = {}
        for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
            s = getattr(result.big_five, trait)
            b5[trait] = {"score": s.score, "confidence": s.confidence, "reason": s.reason}

        l2_adj = va.get("l2_adjustments")

        approx_b5_raw = None
        if result.approximate_assessment:
            approx_b5_raw = result.approximate_assessment.model_dump().get("big5_approximate", {})
        enriched_b5 = None
        if result.llm_comparison and isinstance(result.llm_comparison, dict):
            enriched_b5 = result.llm_comparison.get("enriched_big5", {})

        final_assessment = compute_final_assessment(b5, approx_b5_raw, enriched_b5)

        # Apply L2 adjustments to final scores
        if l2_adj:
            b5_adj = l2_adj.get("big5_adjustments", {})
            traits_fa = final_assessment.get("traits", {})
            for t_name, delta in b5_adj.items():
                if t_name in traits_fa and isinstance(delta, (int, float)):
                    old = traits_fa[t_name]["score"]
                    traits_fa[t_name]["score"] = max(0, min(100, old + delta))
                    traits_fa[t_name]["label"] = score_to_label(traits_fa[t_name]["score"])
                    traits_fa[t_name]["l2_delta"] = delta

        # Validated strengths from final scores
        fa_b5 = {t: {"score": final_assessment.get("traits", {}).get(t, {}).get("score", 50)}
                 for t in b5}
        strengths, dev_areas = validate_strengths_and_dev_areas(
            fa_b5, result.trait_strengths, result.personality_development_areas)

        emo_summary = result.emotion_summary or {}
        dominant_emo = emo_summary.get("dominant_emotion", "neutral")
        if l2_adj and dominant_emo == "sad":
            dominant_emo = "subdued/neutral"

        mot_score = result.motivation.motivation_score
        eng_score = result.engagement.engagement_score if result.engagement else None
        hr_summary = generate_hr_summary_from_scores(
            final_assessment, mot_score, eng_score, dominant_emo)

        return AssessmentResponse(
            success=True,
            candidate_id=result.candidate_id,
            position=result.position,
            personality_assessment={"big_five": b5, "strengths": strengths, "development_areas": dev_areas},
            final_assessment=final_assessment,
            motivation_engagement={
                "motivation": {
                    "level": result.motivation.overall_level,
                    "score": mot_score,
                    "pattern": result.motivation.pattern,
                    "voice_indicators": result.motivation.voice_indicators,
                },
                "engagement": {
                    "level": result.engagement.overall_level,
                    "score": eng_score,
                    "reason": result.engagement.reason,
                },
            },
            emotion_summary=emo_summary,
            voice_analysis={
                "prosody": va.get("prosody"),
                "voice_quality": va.get("voice_quality"),
                "emotion_aggregates": va.get("emotion_aggregates"),
                "l2_adjustments": l2_adj,
            },
            hr_summary=hr_summary,
            pipeline_metadata={"pipeline_version": "2.0.0", "processing_date": datetime.now().isoformat()},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        os.unlink(tmp_path)


@app.post("/assess/html", response_class=HTMLResponse)
async def assess_candidate_html(
    audio: UploadFile = File(...),
    candidate_id: Optional[str] = Form(None),
    position: Optional[str] = Form(None),
):
    """
    Assess a candidate and return an HTML report.
    """
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
    file_ext = Path(audio.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        pipe = get_pipeline()
        result = pipe.process(
            audio_path=Path(tmp_path),
            candidate_id=candidate_id,
            position=position,
            save_output=False,
        )
        
        html = generate_html_report(result)
        return HTMLResponse(content=html)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
