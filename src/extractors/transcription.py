"""Speech-to-text transcription using Whisper."""

from typing import Optional, Dict, List
import numpy as np
import whisper
import torch

from ..config import WhisperConfig
from ..models.schemas import TranscriptionResult
from ..utils.device import get_optimal_device


FILLER_WORDS = [
    "um", "uh", "er", "ah", "like", "you know", "i mean", 
    "basically", "actually", "literally", "right", "so", 
    "well", "kind of", "sort of", "hmm", "uhh", "umm"
]


class WhisperTranscriber:
    """Transcribe audio using OpenAI Whisper."""
    
    def __init__(self, config: Optional[WhisperConfig] = None):
        self.config = config or WhisperConfig()
        self._model = None
        
        # Auto-detect device if set to "auto"
        if self.config.device == "auto":
            self.config.device = get_optimal_device()
    
    @property
    def model(self):
        """Lazy load the Whisper model."""
        if self._model is None:
            print(f"Loading Whisper model on device: {self.config.device}")
            
            # Clear GPU cache before loading
            if self.config.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._model = whisper.load_model(
                self.config.model_name, 
                device=self.config.device
            )
            
            print(f"Whisper model loaded successfully on {self.config.device}")
        return self._model
    
    def transcribe(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        duration: float
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate of the audio
            duration: Duration in seconds
            
        Returns:
            TranscriptionResult with text and metadata
        """
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        
        result = self.model.transcribe(
            audio,
            language=self.config.language,
            word_timestamps=True,
            verbose=False
        )
        
        text = result["text"].strip()
        segments = self._process_segments(result.get("segments", []))
        language = result.get("language", "unknown")
        
        word_count = len(text.split())
        filler_counts = self._count_filler_words(text)
        total_fillers = sum(filler_counts.values())
        filler_rate = (total_fillers / duration) * 60 if duration > 0 else 0
        
        return TranscriptionResult(
            text=text,
            segments=segments,
            language=language,
            word_count=word_count,
            duration_seconds=duration,
            filler_words=filler_counts,
            filler_word_rate=round(filler_rate, 2)
        )
    
    def _process_segments(self, segments: List[Dict]) -> List[Dict[str, Any]]:
        """Process and clean up segments."""
        processed = []
        for seg in segments:
            processed.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip(),
                "words": seg.get("words", [])
            })
        return processed
    
    def _count_filler_words(self, text: str) -> Dict[str, int]:
        """Count filler words in the transcript."""
        text_lower = text.lower()
        counts = {}
        
        for filler in FILLER_WORDS:
            pattern = r'\b' + re.escape(filler) + r'\b'
            matches = re.findall(pattern, text_lower)
            if matches:
                counts[filler] = len(matches)
        
        return counts
