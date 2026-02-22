"""Deterministic motivation and engagement scoring from voice features."""

import numpy as np
from typing import Dict, List, Optional
from ..models.schemas import VoiceFeatures, EmotionResult, ProsodyFeatures


class MotivationScorer:
    """
    Compute motivation and engagement scores from voice features.
    
    Uses deterministic formulas based on acoustic, prosodic, and emotional features.
    Provides temporal analysis to detect motivation trends over time.
    
    Supports three language profiles:
    - native_english: standard thresholds, full emotion weight
    - non_native_english: relaxed pace/pause thresholds, reduced emotion weight
    - sea_english: further relaxed thresholds for SEA-accented English,
                   minimal emotion weight, prosody proxies replace emotion
    """
    
    # Profile-specific threshold presets
    PROFILES = {
        "native_english": {
            "energy_high": 0.08, "energy_low": 0.02,
            "rate_high": 160, "rate_low": 100,
            "pauses_low": 4, "pauses_high": 10,
            "pitch_var_high": 700, "pitch_var_low": 250,
            "rhythm_low": 0.3, "rhythm_mid": 0.8, "rhythm_high": 1.3,
            "ratio_high": 12, "ratio_low": 4,
            "emotion_weight": 1.0,  # full
        },
        "non_native_english": {
            "energy_high": 0.06, "energy_low": 0.015,
            "rate_high": 140, "rate_low": 80,
            "pauses_low": 6, "pauses_high": 15,
            "pitch_var_high": 500, "pitch_var_low": 150,
            "rhythm_low": 0.3, "rhythm_mid": 0.8, "rhythm_high": 1.3,
            "ratio_high": 10, "ratio_low": 3,
            "emotion_weight": 0.5,  # reduced
        },
        "sea_english": {
            "energy_high": 0.05, "energy_low": 0.012,
            "rate_high": 130, "rate_low": 70,
            "pauses_low": 8, "pauses_high": 18,
            "pitch_var_high": 400, "pitch_var_low": 100,
            "rhythm_low": 0.3, "rhythm_mid": 0.9, "rhythm_high": 1.5,
            "ratio_high": 8, "ratio_low": 2,
            "emotion_weight": 0.25,  # minimal — rely on prosody proxies
        },
    }
    
    def __init__(self, language_profile: str = "non_native_english"):
        """
        Initialize scorer with thresholds adapted for language profile.
        
        Args:
            language_profile: "native_english", "non_native_english", or "sea_english"
        """
        self.language_profile = language_profile
        profile = self.PROFILES.get(language_profile, self.PROFILES["non_native_english"])
        
        self.energy_high_threshold = profile["energy_high"]
        self.energy_low_threshold = profile["energy_low"]
        self.rate_high_threshold = profile["rate_high"]
        self.rate_low_threshold = profile["rate_low"]
        self.pauses_low_threshold = profile["pauses_low"]
        self.pauses_high_threshold = profile["pauses_high"]
        self.pitch_var_high_threshold = profile["pitch_var_high"]
        self.pitch_var_low_threshold = profile["pitch_var_low"]
        self.rhythm_low_threshold = profile["rhythm_low"]
        self.rhythm_mid_threshold = profile["rhythm_mid"]
        self.rhythm_high_threshold = profile["rhythm_high"]
        self.ratio_high_threshold = profile["ratio_high"]
        self.ratio_low_threshold = profile["ratio_low"]
        self.emotion_weight = profile["emotion_weight"]
        
        # Voice quality thresholds (universal)
        self.hnr_high_threshold = 8
        self.hnr_low_threshold = 3
        self.jitter_low_threshold = 0.04
        self.jitter_high_threshold = 0.08
    
    def compute_motivation_score(
        self, 
        voice_features: VoiceFeatures,
        extraversion_score: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Compute motivation score from voice features.
        
        Args:
            voice_features: Extracted voice features
            extraversion_score: Optional Big Five extraversion score for engagement
            
        Returns:
            Dictionary with motivation_score, engagement_score, components, and indicators
        """
        prosody = voice_features.prosody
        emotions = voice_features.emotions
        acoustic = voice_features.acoustic_features
        
        # Start at baseline
        score = 50.0
        components = {}
        indicators = []
        
        # Energy component (15%)
        energy_score = self._score_energy(prosody)
        score += energy_score
        components['energy'] = energy_score
        if energy_score != 0:
            indicators.append(f"energy_mean={prosody.energy_mean:.3f} ({'high' if energy_score > 0 else 'low'})")
        
        # Pace component (15%)
        pace_score = self._score_pace(prosody)
        score += pace_score
        components['pace'] = pace_score
        if pace_score != 0:
            from ..utils.scoring import wpm_label
            rate_label = wpm_label(prosody.speaking_rate_wpm, self.language_profile)
            indicators.append(f"speaking_rate_wpm={prosody.speaking_rate_wpm:.1f} ({rate_label})")
        
        # Pauses component (10%)
        pause_score = self._score_pauses(prosody)
        score += pause_score
        components['pauses'] = pause_score
        if pause_score != 0:
            indicators.append(f"pauses_per_minute={prosody.pauses_per_minute:.1f} ({'few' if pause_score > 0 else 'many'})")
        
        # Pitch dynamics component (10%)
        pitch_score = self._score_pitch(prosody)
        score += pitch_score
        components['pitch'] = pitch_score
        if pitch_score != 0:
            indicators.append(f"pitch_variance={prosody.pitch_variance:.1f} ({'high' if pitch_score > 0 else 'low'})")
        
        # Voice quality component (15%)
        quality_score = self._score_voice_quality(acoustic)
        score += quality_score
        components['voice_quality'] = quality_score
        if quality_score != 0:
            hnr = acoustic.voice_quality.get('HNRdBACF_sma3nz_amean', 0)
            jitter = acoustic.voice_quality.get('jitterLocal_sma3nz_amean', 0)
            indicators.append(f"HNR={hnr:.1f}dB, jitter={jitter:.3f} ({'clear' if quality_score > 0 else 'strained'})")
        
        # Rhythm component (10%)
        rhythm_score = self._score_rhythm(prosody)
        score += rhythm_score
        components['rhythm'] = rhythm_score
        if rhythm_score != 0:
            indicators.append(f"rhythm_regularity={prosody.rhythm_regularity:.2f} ({'controlled' if rhythm_score > 0 else 'chaotic'})")
        
        # Speech-to-silence ratio component (5%)
        ratio_score = self._score_speech_ratio(prosody)
        score += ratio_score
        components['speech_ratio'] = ratio_score
        if ratio_score != 0:
            indicators.append(f"speech_to_silence_ratio={prosody.speech_to_silence_ratio:.1f} ({'fluent' if ratio_score > 0 else 'hesitant'})")
        
        # Emotion timeline component (20%) — with prosody proxy fallback
        emotion_score = self._score_emotion_timeline(emotions, prosody)
        score += emotion_score
        components['emotion'] = emotion_score
        if emotion_score != 0:
            indicators.append(f"emotional_trend={emotion_score/16:.2f} ({'positive' if emotion_score > 0 else 'negative'})")
        
        # Clamp to valid range
        motivation_score = int(np.clip(score, 0, 100))
        
        # Compute engagement score if extraversion provided
        # HR logic: motivation = internal drive, engagement = motivation × extraversion × context
        # Engagement should NEVER exceed motivation (quiet motivated introvert: mot=85, eng=65)
        engagement_score = None
        if extraversion_score is not None:
            # Extraversion factor: maps 0-100 extraversion to 0.6-1.0 multiplier
            # Low extraversion (20) → 0.68, Mid (50) → 0.80, High (80) → 0.92
            extraversion_factor = 0.6 + 0.4 * (extraversion_score / 100.0)
            
            # Voice engagement bonus: energy + pace dynamics (max +5)
            voice_bonus = 0.0
            if prosody.energy_mean > self.energy_high_threshold:
                voice_bonus += 2.5
            if prosody.speaking_rate_wpm > self.rate_high_threshold:
                voice_bonus += 2.5
            
            engagement_score = int(round(motivation_score * extraversion_factor + voice_bonus))
            engagement_score = min(engagement_score, motivation_score)  # never exceed motivation
            engagement_score = max(0, min(100, engagement_score))
        
        # Determine motivation level with hysteresis
        motivation_level = self._determine_level_with_hysteresis(motivation_score)
        
        # Determine pattern
        pattern = self._determine_pattern(prosody, emotions)
        
        return {
            'motivation_score': motivation_score,
            'engagement_score': engagement_score,
            'motivation_level': motivation_level,
            'pattern': pattern,
            'components': components,
            'voice_indicators': indicators,
        }
    
    def _score_energy(self, prosody: ProsodyFeatures) -> float:
        """Score based on energy level (weight: 15%)."""
        if prosody.energy_mean >= self.energy_high_threshold:
            return 10.0
        elif prosody.energy_mean <= self.energy_low_threshold:
            return -8.0
        return 0.0
    
    def _score_pace(self, prosody: ProsodyFeatures) -> float:
        """Score based on speaking rate (weight: 15%).
        Non-native/SEA: slower pace is cognitive load, not low motivation — reduced penalty.
        """
        if prosody.speaking_rate_wpm >= self.rate_high_threshold:
            return 10.0
        elif prosody.speaking_rate_wpm <= self.rate_low_threshold:
            if self.language_profile == "sea_english":
                return -3.0
            elif self.language_profile == "non_native_english":
                return -4.0
            return -8.0
        return 0.0
    
    def _score_pauses(self, prosody: ProsodyFeatures) -> float:
        """Score based on pause frequency (weight: 10%).
        Non-native/SEA: pauses are cognitive load, not nervousness — reduced penalty.
        """
        if prosody.pauses_per_minute <= self.pauses_low_threshold:
            return 6.0
        elif prosody.pauses_per_minute >= self.pauses_high_threshold:
            # Softer penalty for non-native speakers
            if self.language_profile == "sea_english":
                return -2.0
            elif self.language_profile == "non_native_english":
                return -3.0
            return -6.0
        return 0.0
    
    def _score_pitch(self, prosody: ProsodyFeatures) -> float:
        """Score based on pitch variance (weight: 10%)."""
        if prosody.pitch_variance >= self.pitch_var_high_threshold:
            return 6.0
        elif prosody.pitch_variance <= self.pitch_var_low_threshold:
            return -6.0
        return 0.0
    
    def _score_voice_quality(self, acoustic_features) -> float:
        """Score based on voice quality - HNR and jitter (weight: 15%)."""
        try:
            hnr = acoustic_features.voice_quality.get('HNRdBACF_sma3nz_amean', 0)
            jitter = acoustic_features.voice_quality.get('jitterLocal_sma3nz_amean', 0)
            
            # Clear voice = confidence
            if hnr > self.hnr_high_threshold and jitter < self.jitter_low_threshold:
                return 8.0
            # Strained voice = stress
            elif hnr < self.hnr_low_threshold or jitter > self.jitter_high_threshold:
                return -6.0
        except (AttributeError, KeyError):
            pass
        return 0.0
    
    def _score_rhythm(self, prosody: ProsodyFeatures) -> float:
        """Score based on rhythm regularity (weight: 10%)."""
        rhythm = prosody.rhythm_regularity
        # Controlled rhythm (not too regular, not too chaotic)
        if self.rhythm_low_threshold < rhythm < self.rhythm_mid_threshold:
            return 6.0
        # Chaotic speech
        elif rhythm > self.rhythm_high_threshold:
            return -5.0
        return 0.0
    
    def _score_speech_ratio(self, prosody: ProsodyFeatures) -> float:
        """Score based on speech-to-silence ratio (weight: 5%)."""
        ratio = prosody.speech_to_silence_ratio
        if ratio > self.ratio_high_threshold:
            return 3.0
        elif ratio < self.ratio_low_threshold:
            return -3.0
        return 0.0
    
    def _compute_prosody_proxies(self, prosody: ProsodyFeatures) -> Dict[str, float]:
        """
        Compute valence/arousal proxies from prosody features.
        Used as emotion fallback when emotion2vec is unreliable (non-native, SEA).
        
        Returns:
            Dict with 'arousal' and 'valence' proxy scores (roughly -1 to +1 range)
        """
        # Arousal proxy: high energy + high pitch variance + few pauses + high speech ratio
        # Normalize each feature to roughly [-1, +1] using profile thresholds
        energy_mid = (self.energy_high_threshold + self.energy_low_threshold) / 2
        energy_range = (self.energy_high_threshold - self.energy_low_threshold) / 2
        energy_z = (prosody.energy_mean - energy_mid) / energy_range if energy_range > 0 else 0
        
        pitch_mid = (self.pitch_var_high_threshold + self.pitch_var_low_threshold) / 2
        pitch_range = (self.pitch_var_high_threshold - self.pitch_var_low_threshold) / 2
        pitch_z = (prosody.pitch_variance - pitch_mid) / pitch_range if pitch_range > 0 else 0
        
        pause_mid = (self.pauses_high_threshold + self.pauses_low_threshold) / 2
        pause_range = (self.pauses_high_threshold - self.pauses_low_threshold) / 2
        pause_z = -(prosody.pauses_per_minute - pause_mid) / pause_range if pause_range > 0 else 0
        
        ratio_mid = (self.ratio_high_threshold + self.ratio_low_threshold) / 2
        ratio_range = (self.ratio_high_threshold - self.ratio_low_threshold) / 2
        ratio_z = (prosody.speech_to_silence_ratio - ratio_mid) / ratio_range if ratio_range > 0 else 0
        
        arousal = float(np.clip((energy_z + pitch_z + pause_z + ratio_z) / 4, -1, 1))
        
        # Valence proxy: rising pitch + energy stability + few long pauses
        slope_z = float(np.clip(prosody.pitch_slope / 0.5, -1, 1))
        stability_z = float(np.clip(1.0 - prosody.energy_std / 0.05, -1, 1))
        long_pause_z = float(np.clip(1.0 - prosody.long_pauses_count / 5.0, -1, 1))
        
        valence = float(np.clip((slope_z + stability_z + long_pause_z) / 3, -1, 1))
        
        return {"arousal": round(arousal, 3), "valence": round(valence, 3)}
    
    def _score_emotion_timeline(self, emotions: EmotionResult, prosody: ProsodyFeatures = None) -> float:
        """
        Score based on emotion timeline analysis (weight: 20%).
        
        Uses emotion2vec when reliable, falls back to prosody proxies when not.
        Emotion weight is scaled by language profile's emotion_weight multiplier.
        """
        score = 0.0
        
        # Compute emotion reliability
        valid_ratio = 0.0
        use_emotion = True
        
        if emotions.emotion_timeline:
            valid_segments = [
                seg for seg in emotions.emotion_timeline
                if seg.get('emotion', '').lower() != 'undetected'
            ]
            total_segments = len(emotions.emotion_timeline)
            valid_count = len(valid_segments)
            valid_ratio = valid_count / total_segments if total_segments > 0 else 0.0
        else:
            valid_segments = []
            valid_count = 0
            total_segments = 0
        
        # Skip emotion entirely if primary is undetected
        if emotions.primary_emotion == "undetected":
            use_emotion = False
        
        # Determine effective emotion weight based on reliability + profile
        # Profile weight: 1.0 (native), 0.5 (non-native), 0.25 (SEA)
        # Reliability multiplier: valid_ratio clamped to [0, 1]
        reliability_mult = min(valid_ratio * 2, 1.0)  # 0.5 ratio → 1.0 mult, below 0.25 → <0.5
        effective_emotion_weight = self.emotion_weight * reliability_mult
        
        if use_emotion and valid_count > 0 and effective_emotion_weight > 0.1:
            # Count positive and negative emotions
            positive_count = 0
            negative_count = 0
            
            for segment in valid_segments:
                emotion = segment.get('emotion', '').lower()
                confidence = segment.get('confidence', 0)
                
                if confidence > 0.5:
                    if 'happy' in emotion or 'surprised' in emotion:
                        positive_count += 1
                    elif 'sad' in emotion or 'fearful' in emotion:
                        negative_count += 1
            
            emotional_trend = (positive_count - negative_count) / valid_count
            
            # Base emotion score (max ±12), scaled by effective weight
            if emotional_trend > 0.3:
                score += 12.0 * effective_emotion_weight
            elif emotional_trend < -0.3:
                score -= 8.0 * effective_emotion_weight
            
            # Variability bonus/penalty
            variability = self._calculate_emotional_variability(valid_segments)
            if variability > 0.4:
                score += 3.0 * effective_emotion_weight
            elif variability < 0.2:
                score -= 2.0 * effective_emotion_weight
        
        # Prosody proxy fallback: fills the gap left by reduced emotion weight
        # The less we trust emotion, the more we rely on prosody proxies
        if prosody is not None:
            proxy_weight = 1.0 - effective_emotion_weight  # complement of emotion weight
            if proxy_weight > 0.1:
                proxies = self._compute_prosody_proxies(prosody)
                # Arousal proxy: max ±6 points
                score += proxies["arousal"] * 6.0 * proxy_weight
                # Valence proxy: max ±4 points
                score += proxies["valence"] * 4.0 * proxy_weight
        
        return score
    
    def _calculate_emotional_variability(self, emotion_timeline: List[Dict]) -> float:
        """
        Calculate emotional variability from timeline.
        
        Measures how much emotions change over time (0=monotone, 1=highly variable).
        """
        if not emotion_timeline or len(emotion_timeline) < 2:
            return 0.0
        
        # Count unique emotions with confidence > 0.3
        emotions_present = set()
        for segment in emotion_timeline:
            emotion = segment.get('emotion', '').lower()
            confidence = segment.get('confidence', 0)
            if confidence > 0.3:
                emotions_present.add(emotion)
        
        # Variability = number of distinct emotions / total possible (8 emotions)
        variability = len(emotions_present) / 8.0
        return variability
    
    def _determine_level_with_hysteresis(self, score: int) -> str:
        """
        Determine motivation level with hysteresis to prevent flickering.
        
        Scores near boundaries (33-46, 63-77) are set to Medium.
        """
        if 33 <= score <= 46 or 63 <= score <= 77:
            return "Medium"
        elif score < 40:
            return "Low"
        elif score >= 70:
            return "High"
        else:
            return "Medium"
    
    def _determine_pattern(self, prosody: ProsodyFeatures, emotions: EmotionResult) -> str:
        """Determine motivation pattern from pitch slope and energy variability."""
        pattern_parts = []
        
        # Pitch slope pattern
        if prosody.pitch_slope > 0.3:
            pattern_parts.append("rising")
        elif prosody.pitch_slope < -0.3:
            pattern_parts.append("falling")
        else:
            pattern_parts.append("consistent")
        
        # Energy variability
        if prosody.energy_std > 0.03:
            pattern_parts.append("fluctuating")
        
        return " ".join(pattern_parts) if pattern_parts else "consistent"
    
    def analyze_temporal_dynamics(
        self, 
        voice_features: VoiceFeatures,
        audio_duration: float
    ) -> Dict[str, any]:
        """
        Analyze how motivation changes over time (beginning, middle, end).
        
        Args:
            voice_features: Voice features with emotion timeline
            audio_duration: Total audio duration in seconds
            
        Returns:
            Dictionary with temporal analysis results
        """
        if not voice_features.emotions.emotion_timeline:
            return {
                'temporal_trend': 'unknown',
                'beginning_motivation': None,
                'middle_motivation': None,
                'end_motivation': None,
            }
        
        timeline = voice_features.emotions.emotion_timeline
        
        # Split timeline into thirds
        third = len(timeline) // 3
        beginning = timeline[:third] if third > 0 else timeline[:1]
        middle = timeline[third:2*third] if third > 0 else timeline
        end = timeline[2*third:] if third > 0 else timeline[-1:]
        
        # Calculate average emotional valence for each section
        def calc_valence(segments):
            positive = sum(1 for s in segments if 'happy' in s.get('emotion', '').lower() or 'surprised' in s.get('emotion', '').lower())
            negative = sum(1 for s in segments if 'sad' in s.get('emotion', '').lower() or 'fearful' in s.get('emotion', '').lower())
            return (positive - negative) / len(segments) if segments else 0
        
        beginning_valence = calc_valence(beginning)
        middle_valence = calc_valence(middle)
        end_valence = calc_valence(end)
        
        # Determine trend
        if end_valence > beginning_valence + 0.2:
            trend = "increasing"
        elif end_valence < beginning_valence - 0.2:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            'temporal_trend': trend,
            'beginning_motivation': beginning_valence,
            'middle_motivation': middle_valence,
            'end_motivation': end_valence,
        }
