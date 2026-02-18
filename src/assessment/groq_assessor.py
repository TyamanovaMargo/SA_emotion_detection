"""Groq-based HR assessment module."""

import json
import re
from typing import Optional
from groq import Groq

from ..config import GroqConfig
from ..models.schemas import (
    VoiceFeatures,
    HRAssessmentInput,
    HRAssessmentResult,
    BigFiveProfile,
    BigFiveScore,
    MotivationAssessment,
    EngagementAssessment,
)
from .prompt_templates import HR_ASSESSMENT_PROMPT, STRUCTURED_OUTPUT_PROMPT
from .motivation_scorer import MotivationScorer


class GroqHRAssessor:
    """Perform HR personality and motivation assessment using Groq."""
    
    def __init__(self, config: Optional[GroqConfig] = None, motivation_config = None):
        self.config = config or GroqConfig()
        self._client = None
        
        # Initialize MotivationScorer with language profile
        from ..config import MotivationConfig
        mot_config = motivation_config or MotivationConfig()
        self.motivation_scorer = MotivationScorer(language_profile=mot_config.language_profile)
    
    @property
    def client(self) -> Groq:
        """Lazy load the Groq client."""
        if self._client is None:
            if not self.config.api_key:
                raise ValueError(
                    "GROQ_API_KEY not set. Please set it in your environment "
                    "or .env file."
                )
            self._client = Groq(api_key=self.config.api_key)
        return self._client
    
    def assess(self, input_data: HRAssessmentInput) -> HRAssessmentResult:
        """
        Perform HR assessment on candidate data.
        
        Args:
            input_data: HRAssessmentInput containing transcript and voice features
            
        Returns:
            HRAssessmentResult with personality and motivation analysis
        """
        prompt = self._build_prompt(input_data)
        
        response = self.client.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        raw_response = response.choices[0].message.content
        
        structured_response = self._get_structured_output(raw_response)
        
        result = self._parse_response(structured_response, raw_response, input_data)
        result.candidate_id = input_data.candidate_id
        result.position = input_data.position
        result.voice_features = input_data.voice_features
        
        return result
    
    def _build_prompt(self, input_data: HRAssessmentInput) -> str:
        """Build the assessment prompt from input data."""
        voice = input_data.voice_features
        prosody = voice.prosody
        language_profile = input_data.language_profile

        # Position context
        if input_data.position:
            position_context = (
                f"TARGET POSITION: {input_data.position.upper()}\n"
                f"Evaluate this candidate specifically for the {input_data.position} role."
            )
        else:
            position_context = "General assessment without specific position context."

        # Transcript section
        if input_data.transcript:
            transcript_section = f"TRANSCRIPT:\n{input_data.transcript}"
        else:
            transcript_section = "TRANSCRIPT: [Not available — voice-only assessment]"

        # Filter emotion timeline: remove "undetected" entries
        total_segments = 0
        valid_segments = 0
        mean_confidence = 0.0
        mean_valence = 0.0
        mean_arousal = 0.0
        emotion_distribution = {}
        if voice.emotions.emotion_timeline:
            total_segments = len(voice.emotions.emotion_timeline)
            filtered = [
                entry for entry in voice.emotions.emotion_timeline
                if entry.get("emotion") != "undetected"
            ]
            valid_segments = len(filtered)
            if filtered:
                mean_confidence = sum(e.get("confidence", 0) for e in filtered) / len(filtered)
                mean_valence = sum(e.get("valence", 0) for e in filtered) / len(filtered)
                mean_arousal = sum(e.get("arousal", 0) for e in filtered) / len(filtered)
                for e in filtered:
                    emo = e.get("emotion", "neutral")
                    emotion_distribution[emo] = emotion_distribution.get(emo, 0) + 1
        
        valid_emotion_segments = f"{valid_segments}/{total_segments}" if total_segments > 0 else "0/0"
        valid_ratio = valid_segments / total_segments if total_segments > 0 else 0.0
        
        # Language context for LLM
        lang_context = {
            "native_english": "Native English speaker — use standard voice thresholds.",
            "non_native_english": "Non-native English speaker — slower pace and more pauses are normal cognitive load, NOT low motivation. Reduce emotion weight.",
            "sea_english": "Southeast Asian English speaker — significantly slower pace, more pauses, and lower energy are culturally normal. Emotion model is unreliable for this accent. Rely primarily on prosody dynamics (energy changes, pitch variance, rhythm) for motivation.",
        }.get(language_profile, "Non-native English speaker.")
        
        compact_data = {
            "language": {
                "detected_language": voice.detected_language,
                "language_confidence": voice.language_confidence,
                "language_profile": language_profile,
                "note": lang_context,
            },
            "prosody": {
                "speaking_rate_wpm": prosody.speaking_rate_wpm,
                "articulation_rate": prosody.articulation_rate,
                "pitch_mean_hz": prosody.pitch_mean_hz,
                "pitch_range": prosody.pitch_range,
                "pitch_variance": prosody.pitch_variance,
                "pitch_slope": prosody.pitch_slope,
                "energy_level": prosody.energy_level,
                "energy_mean": prosody.energy_mean,
                "energy_std": prosody.energy_std,
                "energy_range": prosody.energy_range,
                "pauses_per_minute": prosody.pauses_per_minute,
                "pause_duration_mean": prosody.pause_duration_mean,
                "pause_duration_std": prosody.pause_duration_std,
                "long_pauses_count": prosody.long_pauses_count,
                "speech_to_silence_ratio": prosody.speech_to_silence_ratio,
                "rhythm_regularity": prosody.rhythm_regularity,
            },
            "emotion": {
                "emotion_backend": "MERaLiON-SER-v1" if hasattr(self, '_emotion_backend') else "auto",
                "primary_emotion": voice.emotions.primary_emotion,
                "confidence": voice.emotions.confidence,
                "emotion_scores": voice.emotions.emotion_scores,
                "valid_emotion_segments": valid_emotion_segments,
                "emotion_distribution": emotion_distribution,
                "emotion_reliability": {
                    "valid_ratio": round(valid_ratio, 2),
                    "mean_confidence": round(mean_confidence, 3),
                },
                "valence_arousal": {
                    "mean_valence": round(mean_valence, 3),
                    "mean_arousal": round(mean_arousal, 3),
                    "note": "Derived from emotion labels: valence [-1,+1] (negative→positive), arousal [-1,+1] (calm→excited)"
                },
            },
            "voice_quality": voice.acoustic_features.voice_quality,
            "acoustic_summary": voice.acoustic_features.summary,
            "embedding_profile": voice.wavlm_embedding_summary,
        }

        compact_data_json = json.dumps(compact_data, indent=2)

        prompt = HR_ASSESSMENT_PROMPT.format(
            position_context=position_context,
            voice_profile=voice.wavlm_embedding_summary,
            compact_data_json=compact_data_json,
            transcript_section=transcript_section,
        )

        return prompt
    
    def _get_structured_output(self, analysis_response: str) -> str:
        """Get structured JSON output from Groq based on the analysis."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            max_tokens=2048,
            temperature=0.1,
            messages=[
                {"role": "user", "content": f"Here is an HR assessment analysis:\n\n{analysis_response}"},
                {"role": "assistant", "content": "I've reviewed the HR assessment analysis."},
                {"role": "user", "content": STRUCTURED_OUTPUT_PROMPT}
            ]
        )
        
        return response.choices[0].message.content
    
    def _parse_response(
        self, 
        structured_response: str, 
        raw_response: str,
        input_data: HRAssessmentInput
    ) -> HRAssessmentResult:
        """Parse Groq's response into structured result."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', structured_response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
            
            return self._build_result_from_json(data, raw_response, input_data)
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Could not parse structured response: {e}")
            return self._build_fallback_result(raw_response)
    
    def _build_result_from_json(
        self, 
        data: dict, 
        raw_response: str,
        input_data: HRAssessmentInput
    ) -> HRAssessmentResult:
        """Build HRAssessmentResult from parsed JSON."""
        big_five_data = data.get("big_five", {})
        big_five = BigFiveProfile(
            openness=BigFiveScore(**big_five_data.get("openness", {"score": 50, "confidence": 50, "reason": "Unable to assess"})),
            conscientiousness=BigFiveScore(**big_five_data.get("conscientiousness", {"score": 50, "confidence": 50, "reason": "Unable to assess"})),
            extraversion=BigFiveScore(**big_five_data.get("extraversion", {"score": 50, "confidence": 50, "reason": "Unable to assess"})),
            agreeableness=BigFiveScore(**big_five_data.get("agreeableness", {"score": 50, "confidence": 50, "reason": "Unable to assess"})),
            neuroticism=BigFiveScore(**big_five_data.get("neuroticism", {"score": 50, "confidence": 50, "reason": "Unable to assess"})),
        )
        
        # Use deterministic MotivationScorer with per-file language profile
        file_language_profile = input_data.language_profile
        scorer = MotivationScorer(language_profile=file_language_profile)
        motivation_result = scorer.compute_motivation_score(
            voice_features=input_data.voice_features,
            extraversion_score=big_five.extraversion.score
        )
        
        motivation = MotivationAssessment(
            overall_level=motivation_result['motivation_level'],
            motivation_score=motivation_result['motivation_score'],
            pattern=motivation_result['pattern'],
            voice_indicators=motivation_result['voice_indicators'],
            content_indicators=data.get("motivation", {}).get("content_indicators", []),
        )
        
        # Use deterministic engagement score
        engagement_score = motivation_result['engagement_score']
        engagement_level = "High" if engagement_score >= 70 else "Medium" if engagement_score >= 40 else "Low"
        
        engagement = EngagementAssessment(
            overall_level=engagement_level,
            engagement_score=engagement_score,
            reason=f"Computed from voice features: motivation ({motivation_result['motivation_score']}) and extraversion ({big_five.extraversion.score})"
        )
        
        return HRAssessmentResult(
            big_five=big_five,
            motivation=motivation,
            engagement=engagement,
            trait_strengths=data.get("trait_strengths", []),
            motivation_strengths=data.get("motivation_strengths", []),
            personality_development_areas=data.get("personality_development_areas", []),
            motivation_development_areas=data.get("motivation_development_areas", []),
            hr_summary=data.get("hr_summary", "Assessment completed. Please review the detailed analysis."),
            raw_response=raw_response,
        )
    
    def _build_fallback_result(self, raw_response: str) -> HRAssessmentResult:
        """Build a fallback result when parsing fails."""
        default_score = BigFiveScore(score=50, confidence=30, reason="Parsing failed - manual review required")
        
        return HRAssessmentResult(
            big_five=BigFiveProfile(
                openness=default_score,
                conscientiousness=default_score,
                extraversion=default_score,
                agreeableness=default_score,
                neuroticism=default_score,
            ),
            motivation=MotivationAssessment(
                overall_level="Medium",
                motivation_score=50,
                pattern="Unable to parse - see raw response",
                voice_indicators=["See raw response for details"],
                content_indicators=["See raw response for details"],
            ),
            engagement=EngagementAssessment(
                overall_level="Medium",
                engagement_score=50,
                reason="Unable to parse - see raw response"
            ),
            trait_strengths=["See raw response for details"],
            motivation_strengths=["See raw response for details"],
            personality_development_areas=["See raw response for details"],
            motivation_development_areas=["See raw response for details"],
            hr_summary="Assessment completed but structured parsing failed. Please review the raw response for detailed analysis.",
            raw_response=raw_response,
        )
    
    def assess_batch(
        self, 
        inputs: list[HRAssessmentInput]
    ) -> list[HRAssessmentResult]:
        """
        Assess multiple candidates.
        
        Args:
            inputs: List of HRAssessmentInput objects
            
        Returns:
            List of HRAssessmentResult objects
        """
        results = []
        for input_data in inputs:
            try:
                result = self.assess(input_data)
                results.append(result)
            except Exception as e:
                print(f"Error assessing candidate {input_data.candidate_id}: {e}")
                results.append(self._build_fallback_result(str(e)))
        
        return results
