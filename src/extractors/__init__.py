"""Feature extraction modules."""

from .transcription import WhisperTranscriber
from .prosody import ProsodyExtractor
from .emotion import EmotionDetector
from .egemaps import EgemapsExtractor

__all__ = [
    "WhisperTranscriber",
    "ProsodyExtractor",
    "EmotionDetector",
    "EgemapsExtractor",
]
