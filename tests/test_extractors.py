"""Tests for feature extractors."""

import pytest
import numpy as np


class TestProsodyExtractor:
    """Tests for ProsodyExtractor."""
    
    def test_extract_pitch(self):
        """Test pitch extraction from synthetic audio."""
        from src.extractors.prosody import ProsodyExtractor
        
        extractor = ProsodyExtractor()
        
        sample_rate = 16000
        duration = 2.0
        frequency = 200  # Hz
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        prosody = extractor.extract(audio, sample_rate, word_count=20, duration=duration)
        
        assert prosody.pitch_mean_hz > 0
        assert prosody.speaking_rate_wpm == 600.0  # 20 words / 2 sec * 60
    
    def test_energy_levels(self):
        """Test energy level classification."""
        from src.extractors.prosody import ProsodyExtractor
        
        extractor = ProsodyExtractor()
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        quiet_audio = 0.01 * np.sin(2 * np.pi * 200 * t)
        prosody = extractor.extract(quiet_audio, sample_rate, 10, duration)
        assert prosody.energy_level == "low"
        
        loud_audio = 0.5 * np.sin(2 * np.pi * 200 * t)
        prosody = extractor.extract(loud_audio, sample_rate, 10, duration)
        assert prosody.energy_level in ["medium", "high"]


class TestEmotionDetector:
    """Tests for EmotionDetector."""
    
    def test_fallback_detection(self):
        """Test fallback emotion detection."""
        from src.extractors.emotion import EmotionDetector
        
        detector = EmotionDetector()
        detector._use_fallback = True  # Force fallback mode
        
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * 200 * t)
        
        result = detector.detect(audio, sample_rate, duration)
        
        assert result.primary_emotion in [
            "angry", "disgusted", "fearful", "happy",
            "neutral", "sad", "surprised", "other"
        ]
        assert 0 <= result.confidence <= 1
        assert len(result.emotion_scores) > 0


class TestTranscriber:
    """Tests for WhisperTranscriber."""
    
    def test_filler_word_counting(self):
        """Test filler word detection."""
        from src.extractors.transcription import WhisperTranscriber
        
        transcriber = WhisperTranscriber()
        
        text = "Um, I think, like, you know, it's basically, um, a good idea."
        counts = transcriber._count_filler_words(text)
        
        assert counts.get("um", 0) == 2
        assert counts.get("like", 0) == 1
        assert counts.get("you know", 0) == 1
        assert counts.get("basically", 0) == 1


class TestEgemapsExtractor:
    """Tests for EgemapsExtractor."""
    
    def test_fallback_extraction(self):
        """Test fallback feature extraction."""
        from src.extractors.egemaps import EgemapsExtractor
        
        extractor = EgemapsExtractor()
        extractor._use_fallback = True  # Force fallback mode
        
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * 200 * t)
        
        features = extractor.extract(audio, sample_rate)
        
        assert len(features.spectral_features) > 0
        assert len(features.frequency_features) > 0
        assert len(features.energy_features) > 0
        assert features.summary != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
