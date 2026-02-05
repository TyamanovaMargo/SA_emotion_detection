"""Tests for data schemas."""

import pytest
from pydantic import ValidationError


class TestSchemas:
    """Tests for Pydantic schemas."""
    
    def test_big_five_score_validation(self):
        """Test BigFiveScore validation."""
        from src.models.schemas import BigFiveScore
        
        valid = BigFiveScore(score=75, confidence=80, reason="Test reason")
        assert valid.score == 75
        
        with pytest.raises(ValidationError):
            BigFiveScore(score=150, confidence=80, reason="Invalid score")
        
        with pytest.raises(ValidationError):
            BigFiveScore(score=-10, confidence=80, reason="Negative score")
    
    def test_prosody_features(self):
        """Test ProsodyFeatures schema."""
        from src.models.schemas import ProsodyFeatures
        
        prosody = ProsodyFeatures(
            speaking_rate_wpm=145.0,
            pitch_mean_hz=185.5,
            pitch_variance=850.0,
            pitch_range=120.0,
            energy_level="high",
            energy_mean=0.065,
            pauses_per_minute=4.2,
            pause_duration_mean=0.45,
            articulation_rate=4.8,
        )
        
        assert prosody.speaking_rate_wpm == 145.0
        assert prosody.energy_level == "high"
    
    def test_emotion_result(self):
        """Test EmotionResult schema."""
        from src.models.schemas import EmotionResult
        
        emotion = EmotionResult(
            primary_emotion="happy",
            confidence=0.85,
            emotion_scores={"happy": 0.85, "neutral": 0.10, "sad": 0.05},
        )
        
        assert emotion.primary_emotion == "happy"
        assert emotion.confidence == 0.85
        assert sum(emotion.emotion_scores.values()) == pytest.approx(1.0)
    
    def test_hr_assessment_result(self):
        """Test HRAssessmentResult schema."""
        from src.models.schemas import (
            HRAssessmentResult,
            BigFiveProfile,
            BigFiveScore,
            MotivationAssessment,
        )
        
        default_score = BigFiveScore(score=50, confidence=70, reason="Test")
        
        result = HRAssessmentResult(
            big_five=BigFiveProfile(
                openness=default_score,
                conscientiousness=default_score,
                extraversion=default_score,
                agreeableness=default_score,
                neuroticism=default_score,
            ),
            motivation=MotivationAssessment(
                overall_level="High",
                pattern="Consistent",
                voice_indicators=["High energy"],
                content_indicators=["Future-oriented"],
            ),
            trait_strengths=["Leadership"],
            motivation_strengths=["Persistent"],
            personality_development_areas=["Assertiveness"],
            motivation_development_areas=["Sustained focus"],
            hr_summary="Test summary.",
        )
        
        assert result.motivation.overall_level == "High"
        assert len(result.trait_strengths) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
