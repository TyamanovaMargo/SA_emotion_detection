"""Feature extraction modules (slim — no transcription)."""

from .prosody import ProsodyExtractor
from .emotion_meralion import EmotionDetector
from .egemaps import EgemapsExtractor
from .voice_analyzer import VoiceAnalyzer

__all__ = [
    "ProsodyExtractor",
    "EmotionDetector",
    "EgemapsExtractor",
    "VoiceAnalyzer",
]
