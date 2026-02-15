"""Configuration settings for the HR assessment pipeline."""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class WhisperConfig(BaseModel):
    """Whisper transcription configuration."""
    model_name: str = Field(default="base", description="Whisper model size")
    language: Optional[str] = Field(default=None, description="Language code or None for auto-detect")
    device: str = Field(
        default_factory=lambda: os.getenv("WHISPER_DEVICE", "auto"),
        description="Device to run on (cpu/cuda/mps/auto)"
    )


class EmotionConfig(BaseModel):
    """Emotion detection configuration."""
    model_name: str = Field(
        default="iic/emotion2vec_plus_base",
        description="Emotion2vec model name"
    )
    device: str = Field(
        default_factory=lambda: os.getenv("EMOTION_DEVICE", "auto"),
        description="Device to run on (cpu/cuda/mps/auto)"
    )
    batch_size: int = Field(
        default_factory=lambda: int(os.getenv("EMOTION_BATCH_SIZE", "0")),
        description="Batch size (0=auto based on device)"
    )


class ProsodyConfig(BaseModel):
    """Prosody extraction configuration."""
    frame_length: float = Field(default=0.025, description="Frame length in seconds")
    hop_length: float = Field(default=0.010, description="Hop length in seconds")
    min_pitch: float = Field(default=75.0, description="Minimum pitch in Hz")
    max_pitch: float = Field(default=500.0, description="Maximum pitch in Hz")


class EgemapsConfig(BaseModel):
    """eGeMAPS feature extraction configuration."""
    feature_set: str = Field(default="eGeMAPSv02", description="OpenSMILE feature set")
    feature_level: str = Field(default="Functionals", description="Feature level")


class GroqConfig(BaseModel):
    """Groq API configuration."""
    api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model: str = Field(default="llama-3.3-70b-versatile", description="Groq model to use")
    max_tokens: int = Field(default=4096, description="Maximum tokens in response")
    temperature: float = Field(default=0.0, description="Temperature for generation (0.0 for deterministic)")


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    emotion: EmotionConfig = Field(default_factory=EmotionConfig)
    prosody: ProsodyConfig = Field(default_factory=ProsodyConfig)
    egemaps: EgemapsConfig = Field(default_factory=EgemapsConfig)
    groq: GroqConfig = Field(default_factory=GroqConfig)
    
    output_dir: Path = Field(default=Path("./outputs"), description="Output directory")
    cache_dir: Path = Field(default=Path("./cache"), description="Cache directory")
    
    class Config:
        arbitrary_types_allowed = True


def load_config() -> PipelineConfig:
    """Load configuration from environment and defaults."""
    return PipelineConfig()
