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
    """API response model."""
    success: bool
    candidate_id: Optional[str]
    position: Optional[str]
    big_five: dict
    motivation: dict
    trait_strengths: list
    motivation_strengths: list
    personality_development_areas: list
    motivation_development_areas: list
    hr_summary: str


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
        
        return AssessmentResponse(
            success=True,
            candidate_id=result.candidate_id,
            position=result.position,
            big_five={
                "openness": result.big_five.openness.model_dump(),
                "conscientiousness": result.big_five.conscientiousness.model_dump(),
                "extraversion": result.big_five.extraversion.model_dump(),
                "agreeableness": result.big_five.agreeableness.model_dump(),
                "neuroticism": result.big_five.neuroticism.model_dump(),
            },
            motivation=result.motivation.model_dump(),
            trait_strengths=result.trait_strengths,
            motivation_strengths=result.motivation_strengths,
            personality_development_areas=result.personality_development_areas,
            motivation_development_areas=result.motivation_development_areas,
            hr_summary=result.hr_summary,
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
