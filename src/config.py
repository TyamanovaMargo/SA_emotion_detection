"""Configuration settings for the HR assessment pipeline."""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class WhisperConfig(BaseModel):
    """Whisper transcription configuration."""
    model_name: str = Field(default="distil-whisper/distil-large-v3", description="Whisper model name (HuggingFace model ID)")
    language: Optional[str] = Field(default=None, description="Language code or None for auto-detect")
    device: str = Field(
        default_factory=lambda: os.getenv("WHISPER_DEVICE", "auto"),
        description="Device to run on (cpu/cuda/cuda:0/cuda:1/mps/auto)"
    )
    gpu_index: int = Field(
        default_factory=lambda: int(os.getenv("WHISPER_GPU", "0")),
        description="GPU index for Whisper model"
    )


class EmotionConfig(BaseModel):
    """Emotion detection configuration (MERaLiON-SER-v1 only)."""
    meralion_model: str = Field(
        default="MERaLiON/MERaLiON-SER-v1",
        description="MERaLiON SER model name from HuggingFace"
    )
    device: str = Field(
        default_factory=lambda: os.getenv("EMOTION_DEVICE", "auto"),
        description="Device to run on (cpu/cuda/cuda:0/cuda:1/mps/auto)"
    )
    gpu_index: int = Field(
        default_factory=lambda: int(os.getenv("EMOTION_GPU", "1")),
        description="GPU index for emotion model"
    )
    batch_size: int = Field(
        default_factory=lambda: int(os.getenv("EMOTION_BATCH_SIZE", "0")),
        description="Batch size (0=auto based on device)"
    )
    segment_duration: float = Field(
        default=7.5,
        description="Segment duration in seconds for emotion timeline (longer = more stable)"
    )
    hop_ratio: float = Field(
        default=0.5,
        description="Overlap ratio for timeline segments (0.5 = 50% overlap)"
    )
    silence_threshold_factor: float = Field(
        default=1.5,
        description="Silence threshold = noise_floor * this factor"
    )
    min_snr_db: float = Field(
        default=6.0,
        description="Minimum SNR in dB for a segment to be processed"
    )


class ProsodyConfig(BaseModel):
    """Prosody extraction configuration."""
    frame_length: float = Field(default=0.064, description="Frame length in seconds (1024 samples at 16kHz)")
    hop_length: float = Field(default=0.010, description="Hop length in seconds")
    min_pitch: float = Field(default=80.0, description="Minimum pitch in Hz")
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


class MotivationConfig(BaseModel):
    """Motivation scoring configuration."""
    language_profile: str = Field(
        default_factory=lambda: os.getenv("LANGUAGE_PROFILE", "non_native_english"),
        description="Language profile: 'native_english' or 'non_native_english'"
    )


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    emotion: EmotionConfig = Field(default_factory=EmotionConfig)
    prosody: ProsodyConfig = Field(default_factory=ProsodyConfig)
    egemaps: EgemapsConfig = Field(default_factory=EgemapsConfig)
    groq: GroqConfig = Field(default_factory=GroqConfig)
    motivation: MotivationConfig = Field(default_factory=MotivationConfig)
    
    output_dir: Path = Field(default=Path("./outputs"), description="Output directory")
    cache_dir: Path = Field(default=Path("./cache"), description="Cache directory")
    
    class Config:
        arbitrary_types_allowed = True


def load_config() -> PipelineConfig:
    """Load configuration from environment and defaults."""
    return PipelineConfig()
