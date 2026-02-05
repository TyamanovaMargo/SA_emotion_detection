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
)
from .prompt_templates import HR_ASSESSMENT_PROMPT, STRUCTURED_OUTPUT_PROMPT


class GroqHRAssessor:
    """Perform HR personality and motivation assessment using Groq."""
    
    def __init__(self, config: Optional[GroqConfig] = None):
        self.config = config or GroqConfig()
        self._client = None
    
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
        
        result = self._parse_response(structured_response, raw_response)
        result.candidate_id = input_data.candidate_id
        result.position = input_data.position
        
        return result
    
    def _build_prompt(self, input_data: HRAssessmentInput) -> str:
        """Build the assessment prompt from input data."""
        voice = input_data.voice_features
        
        emotions_json = json.dumps(voice.emotions.model_dump(), indent=2)
        
        egemaps_summary = json.dumps({
            "spectral": voice.acoustic_features.spectral_features,
            "frequency": voice.acoustic_features.frequency_features,
            "energy": voice.acoustic_features.energy_features,
            "summary": voice.acoustic_features.summary
        }, indent=2)
        
        prompt = HR_ASSESSMENT_PROMPT.format(
            transcript=input_data.transcript,
            emotions_json=emotions_json,
            speaking_rate=voice.prosody.speaking_rate_wpm,
            pitch_mean=voice.prosody.pitch_mean_hz,
            pitch_var=voice.prosody.pitch_variance,
            energy=voice.prosody.energy_level,
            pauses=voice.prosody.pauses_per_minute,
            egemaps_summary=egemaps_summary,
            embedding_profile=voice.wavlm_embedding_summary
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
        raw_response: str
    ) -> HRAssessmentResult:
        """Parse Groq's response into structured result."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', structured_response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
            
            return self._build_result_from_json(data, raw_response)
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Could not parse structured response: {e}")
            return self._build_fallback_result(raw_response)
    
    def _build_result_from_json(
        self, 
        data: dict, 
        raw_response: str
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
        
        motivation_data = data.get("motivation", {})
        motivation = MotivationAssessment(
            overall_level=motivation_data.get("overall_level", "Medium"),
            pattern=motivation_data.get("pattern", "Unable to determine pattern"),
            voice_indicators=motivation_data.get("voice_indicators", []),
            content_indicators=motivation_data.get("content_indicators", []),
        )
        
        return HRAssessmentResult(
            big_five=big_five,
            motivation=motivation,
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
                pattern="Unable to parse - see raw response",
                voice_indicators=["See raw response for details"],
                content_indicators=["See raw response for details"],
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
