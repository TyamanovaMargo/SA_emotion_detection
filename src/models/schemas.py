"""Pydantic schemas for data models."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class ProsodyFeatures(BaseModel):
    """Prosody features extracted from audio."""
    speaking_rate_wpm: float = Field(description="Speaking rate in words per minute")
    pitch_mean_hz: float = Field(description="Mean pitch in Hz")
    pitch_variance: float = Field(description="Pitch variance")
    pitch_range: float = Field(description="Pitch range (max - min)")
    pitch_slope: float = Field(default=0.0, description="Pitch trend over time: >0 rising, <0 falling")
    energy_level: str = Field(description="Energy level: low/medium/high")
    energy_mean: float = Field(description="Mean energy/RMS value")
    energy_std: float = Field(default=0.0, description="Energy standard deviation")
    energy_range: float = Field(default=0.0, description="Energy range (max - min)")
    pauses_per_minute: float = Field(description="Number of pauses per minute")
    pause_duration_mean: float = Field(description="Mean pause duration in seconds")
    pause_duration_std: float = Field(default=0.0, description="Pause duration standard deviation")
    long_pauses_count: int = Field(default=0, description="Number of pauses > 1 second")
    articulation_rate: float = Field(description="Articulation rate (syllables per second)")
    speech_to_silence_ratio: float = Field(default=0.0, description="Ratio of speech time to silence time")
    rhythm_regularity: float = Field(default=0.0, description="CV of syllable intervals (0=regular, 1=irregular)")


class EmotionResult(BaseModel):
    """Emotion detection results."""
    primary_emotion: str = Field(description="Dominant emotion")
    confidence: float = Field(description="Confidence score for primary emotion")
    emotion_scores: Dict[str, float] = Field(description="Scores for all emotions")
    emotion_timeline: Optional[List[Dict[str, Any]]] = Field(
        default=None, 
        description="Emotion changes over time"
    )


class EgemapsFeatures(BaseModel):
    """eGeMAPS acoustic features summary."""
    spectral_features: Dict[str, float] = Field(description="Spectral features")
    frequency_features: Dict[str, float] = Field(description="Frequency-related features")
    energy_features: Dict[str, float] = Field(description="Energy-related features")
    temporal_features: Dict[str, float] = Field(description="Temporal features")
    voice_quality: Dict[str, float] = Field(description="Voice quality features")
    summary: str = Field(description="Human-readable summary of acoustic profile")


class TranscriptionResult(BaseModel):
    """Speech-to-text transcription result."""
    text: str = Field(description="Full transcription text")
    segments: List[Dict[str, Any]] = Field(description="Timestamped segments")
    language: str = Field(description="Detected language")
    word_count: int = Field(description="Total word count")
    duration_seconds: float = Field(description="Audio duration in seconds")
    filler_words: Dict[str, int] = Field(description="Filler word counts")
    filler_word_rate: float = Field(description="Filler words per minute")


class AudioFeatures(BaseModel):
    """Combined audio features."""
    sample_rate: int = Field(description="Audio sample rate")
    duration_seconds: float = Field(description="Audio duration")
    channels: int = Field(description="Number of audio channels")


class VoiceFeatures(BaseModel):
    """Complete voice features for HR assessment."""
    emotions: EmotionResult
    prosody: ProsodyFeatures
    acoustic_features: EgemapsFeatures
    wavlm_embedding_summary: str = Field(description="Summary of voice embedding profile")
    detected_language: str = Field(default="unknown", description="Detected language code (e.g. 'en', 'id', 'zh')")
    language_confidence: float = Field(default=0.0, description="Confidence of language detection (0-1)")
    language_profile: str = Field(default="non_native_english", description="Language profile: native_english, non_native_english, sea_english")


class HRAssessmentInput(BaseModel):
    """Input data for HR assessment."""
    transcript: str
    voice_features: VoiceFeatures
    audio_duration: float
    candidate_id: Optional[str] = None
    position: Optional[str] = None
    language_profile: str = Field(default="non_native_english", description="Language profile for scoring calibration")


class BigFiveScore(BaseModel):
    """Score for a Big Five personality trait."""
    score: int = Field(ge=0, le=100, description="Score from 0-100")
    confidence: int = Field(ge=0, le=100, description="Confidence percentage")
    reason: str = Field(description="Brief reason for the score")

    @field_validator("score", "confidence", mode="before")
    @classmethod
    def coerce_to_int(cls, v):
        if isinstance(v, float):
            if v <= 1.0:
                v = v * 100
            return int(round(v))
        return v


class BigFiveProfile(BaseModel):
    """Complete Big Five personality profile."""
    openness: BigFiveScore
    conscientiousness: BigFiveScore
    extraversion: BigFiveScore
    agreeableness: BigFiveScore
    neuroticism: BigFiveScore


class MotivationAssessment(BaseModel):
    """Motivation level assessment."""
    overall_level: str = Field(description="High/Medium/Low")
    motivation_score: int = Field(ge=0, le=100, description="Numeric motivation score 0-100")
    pattern: str = Field(description="Pattern description")
    voice_indicators: List[str] = Field(description="Key indicators from voice")
    content_indicators: List[str] = Field(description="Key indicators from content")


class EngagementAssessment(BaseModel):
    """Engagement level assessment."""
    overall_level: str = Field(description="High/Medium/Low")
    engagement_score: int = Field(ge=0, le=100, description="Numeric engagement score 0-100")
    reason: str = Field(description="Explanation of engagement level")


class HRAssessmentResult(BaseModel):
    """Complete HR assessment result."""
    candidate_id: Optional[str] = None
    position: Optional[str] = None
    
    big_five: BigFiveProfile
    motivation: MotivationAssessment
    engagement: EngagementAssessment
    
    trait_strengths: List[str] = Field(description="Top trait strengths")
    motivation_strengths: List[str] = Field(description="Top motivation strengths")
    
    personality_development_areas: List[str]
    motivation_development_areas: List[str]
    
    hr_summary: str = Field(description="3-sentence HR summary")
    
    voice_features: Optional["VoiceFeatures"] = Field(default=None, description="Detailed voice features")
    raw_response: Optional[str] = Field(default=None, description="Raw Claude response")
