"""Prompt templates for HR assessment."""

HR_ASSESSMENT_PROMPT = """You are an expert HR psychologist specializing in **voice-based** candidate assessment. You determine personality (Big Five) and motivation/engagement primarily from **acoustic voice features**. Transcript is supplementary — voice evidence takes priority.

{position_context}

=== VOICE PROFILE ===
{voice_profile}

=== VOICE DATA (all values pre-computed from audio) ===
{compact_data_json}

{transcript_section}

=== EMOTION MODEL ===

Emotion features are extracted using **MERaLiON-SER-v1** (fine-tuned Speech Emotion Recognition model) with **7.5-second overlapped windows** (50% overlap) and reliability gating.

Each timeline segment provides:
- **emotion**: dominant emotion label (angry, happy, sad, fearful, surprised, neutral, disgusted)
- **confidence**: model confidence for the dominant label
- **valence**: derived affective valence [-1, +1] (negative → positive)
- **arousal**: derived affective arousal [-1, +1] (calm → excited)
- **rms_energy**: segment loudness
- **snr_db**: signal-to-noise ratio

The voice data includes aggregated **mean_valence** and **mean_arousal** across all valid segments, plus an **emotion_distribution** showing how many segments were classified as each emotion.

=== LANGUAGE PROFILE AWARENESS ===

**CRITICAL:** Check the "language" section in voice data above. The language_profile determines how you interpret voice features:

- **native_english**: Use standard thresholds below as-is.
- **non_native_english**: Slower pace and more pauses reflect cognitive load in a second language, NOT low motivation. Emotion model has reduced reliability.
- **sea_english** (Southeast Asian English): Significantly slower pace, more pauses, and lower energy are culturally normal speech patterns. Rely primarily on prosody dynamics (energy changes, pitch variance, rhythm) for motivation.

=== MOTIVATION & ENGAGEMENT ASSESSMENT (STABLE SCORING) ===

You MUST compute two numeric scores using ONLY voice features with EXACT formulas below.
These scores ensure consistency across multiple recordings of the same person.

**STEP 1: Compute motivation_score (0-100)**

Start: motivation_score = 50

Apply adjustments using PROFILE-ADAPTED thresholds:

**Thresholds by profile:**
| Feature | native_english | non_native_english | sea_english |
|---|---|---|---|
| energy_high | 0.08 | 0.06 | 0.05 |
| energy_low | 0.02 | 0.015 | 0.012 |
| rate_high (wpm) | 160 | 140 | 130 |
| rate_low (wpm) | 100 | 80 | 70 |
| pauses_low (/min) | 4 | 6 | 8 |
| pauses_high (/min) | 10 | 15 | 18 |
| pitch_var_high | 700 | 500 | 400 |
| pitch_var_low | 250 | 150 | 100 |

Energy (prosody.energy_mean) [weight: 15%]:
- if energy_mean >= energy_high: +10
- if energy_mean <= energy_low: -8
- otherwise: 0

Pace (prosody.speaking_rate_wpm) [weight: 15%]:
- if speaking_rate_wpm >= rate_high: +10
- if speaking_rate_wpm <= rate_low: -8 (native) / -4 (non_native) / -3 (sea)
- otherwise: 0

Pauses (prosody.pauses_per_minute) [weight: 10%]:
- if pauses_per_minute <= pauses_low: +6
- if pauses_per_minute >= pauses_high: -6 (native) / -3 (non_native) / -2 (sea)
- otherwise: 0

Pitch dynamics (prosody.pitch_variance) [weight: 10%]:
- if pitch_variance >= pitch_var_high: +6
- if pitch_variance <= pitch_var_low: -6
- otherwise: 0

Voice Quality (acoustic_features.voice_quality) [weight: 15%]:
- if HNRdBACF_sma3nz_amean > 8 AND jitterLocal_sma3nz_amean < 0.04: +8 (clear voice = confidence)
- if HNRdBACF_sma3nz_amean < 3 OR jitterLocal_sma3nz_amean > 0.08: -6 (strained voice = stress)
- otherwise: 0

Rhythm (prosody.rhythm_regularity) [weight: 10%]:
- if 0.3 < rhythm_regularity < 0.8: +6 (controlled rhythm)
- if rhythm_regularity > 1.3: -5 (chaotic speech)
- otherwise: 0

Speech-to-Silence Ratio (prosody.speech_to_silence_ratio) [weight: 5%]:
- if ratio > 10: +3 (fluent, few pauses)
- if ratio < 3: -3 (many pauses, hesitation)
- otherwise: 0

Emotion + Valence/Arousal [weight: 20%]:

**Use the valence_arousal data from the emotion section:**
- mean_valence: overall affective tone (positive = engaged/happy, negative = stressed/sad)
- mean_arousal: overall activation level (high = energetic/excited, low = calm/disengaged)
- emotion_distribution: which emotions dominate across the recording

**Check emotion_reliability:**
- valid_ratio = emotion.emotion_reliability.valid_ratio
- mean_confidence = emotion.emotion_reliability.mean_confidence

**Emotion weight by profile:**
- native_english: emotion_weight = 1.0
- non_native_english: emotion_weight = 0.6
- sea_english: emotion_weight = 0.3

effective_weight = emotion_weight * min(valid_ratio * 2, 1.0)

If effective_weight >= 0.1, apply emotion-based adjustments:
- Use mean_arousal as primary motivation signal: high arousal (+) = energetic engagement
  - if mean_arousal > 0.3: +(10 * effective_weight)
  - if mean_arousal < -0.2: -(6 * effective_weight)
- Use mean_valence as secondary signal: positive valence = positive attitude
  - if mean_valence > 0.3: +(4 * effective_weight)
  - if mean_valence < -0.3: -(4 * effective_weight)
- Use emotion_distribution: many "happy"/"surprised" segments = positive; many "sad"/"fearful" = negative

If effective_weight < 0.1 (emotion unreliable):
- Use PROSODY PROXIES instead:
  - Arousal proxy (energy dynamics + pitch variance + fluency): max ±6
  - Valence proxy (pitch slope + energy stability): max ±4

Final: Clamp motivation_score to [0, 100]

**STEP 2: Compute engagement_score (0-100)**

HR logic: motivation = internal drive, engagement = how that drive manifests outwardly.
Engagement should NEVER exceed motivation. A quiet motivated introvert: motivation=85, engagement=65.

extraversion_factor = 0.6 + 0.4 * (extraversion_score / 100)
voice_bonus = +2.5 if energy_mean > energy_high, +2.5 if speaking_rate > rate_high (max +5)
engagement_score = round(motivation_score * extraversion_factor + voice_bonus)
engagement_score = min(engagement_score, motivation_score)  # HARD CAP: never exceed motivation
engagement_score = clamp(engagement_score, 0, 100)

**STEP 3: Convert scores to stable levels with hysteresis**

Base level mapping:
- 0-39: Low
- 40-69: Medium
- 70-100: High

HYSTERESIS RULE (prevents flickering):
If score is within 7 points of boundary (33-46 or 63-77):
  - Set level to "Medium"
  - Add "borderline" to pattern description

**MOTIVATION PATTERN (use pitch_slope + arousal):**
- pitch_slope > 0.3 OR mean_arousal > 0.4: "rising" (builds engagement)
- pitch_slope < -0.3 OR mean_arousal < -0.3: "falling" (loses engagement)
- otherwise: "consistent"
- If energy_std > 0.03: add "fluctuating"

**REQUIRED: voice_indicators must cite exact values AND the language_profile used:**
Example: ["language_profile=non_native_english", "energy_mean=0.065 (high)", "mean_arousal=0.45 (energetic)", "mean_valence=0.2 (positive)", "emotion_distribution: happy=5, neutral=3"]

=== PERSONALITY FROM VOICE (Big Five) ===

**CRITICAL: Use the FULL 0-100 scale. Do NOT cluster around 40-60. Scores of 15, 25, 75, 85 are normal.**

When language_profile is non_native_english or sea_english, adjust personality interpretation:
- Slower pace does NOT mean low Extraversion — compare against profile thresholds
- More pauses does NOT mean low Conscientiousness — it reflects language processing
- Neutral emotion does NOT mean low Agreeableness — emotion model may be unreliable

**TRAIT-TO-FEATURE MAPPING (use these signals for each trait):**

- **Extraversion**: 
  - HIGH: energy_mean > energy_high, speaking_rate > rate_high, mean_arousal > 0.3, speech_to_silence_ratio > 8, many "happy"/"surprised" segments
  - LOW: energy_mean < energy_low, slow rate, mean_arousal < -0.1, flat dynamics
  
- **Openness**: 
  - HIGH: pitch_range > 150 Hz, pitch_variance > pitch_var_high, high energy_range, mean_valence positive, varied emotion_distribution
  - LOW: pitch_range < 60, pitch_variance < pitch_var_low, flat dynamics, monotone emotion

- **Conscientiousness**: 
  - HIGH: speaking_rate in middle range, rhythm_regularity 0.3-0.5, pauses < pauses_low, stable energy (low energy_std), consistent emotion pattern
  - LOW: erratic pace, rhythm_regularity > 0.7, pauses > pauses_high, high energy_std

- **Agreeableness**: 
  - HIGH: mean_valence > 0.2, moderate arousal (0.1-0.5), pitch_variance < 500, smooth prosody, low jitter, high HNR (warm voice)
  - LOW: mean_valence < -0.3, high arousal with negative valence (aggressive), harsh voice quality (high jitter, low HNR)

- **Neuroticism**: 
  - HIGH: high jitter (> 0.06), low HNR (< 4), many long_pauses, rhythm_regularity > 0.8, mean_valence < -0.2 with mean_arousal > 0.3 (anxious pattern), many "fearful"/"sad" segments
  - LOW: low jitter, high HNR, few pauses, stable rhythm, positive or neutral valence

**SCORING RULES:**
1. Match features to profile-adapted thresholds — cite actual values
2. Use valence/arousal as cross-validation for trait inference
3. Voice quality (jitter, shimmer, HNR) → Neuroticism, Agreeableness
4. pitch_slope + mean_arousal → motivation pattern
5. If no transcript → voice scores are your only data, increase confidence
6. If transcript available → use as weak corroboration only, do NOT let text override strong voice evidence
7. Always cite language_profile, valid_emotion_segments, and valence/arousal in your reasoning

=== OUTPUT FORMAT ===

## 1. Big Five Profile (0-100, with confidence)
For each trait, cite specific voice feature values AND valence/arousal signals.

## 2. Personality Facets (0-100)
15 facets, each with score, confidence, 1-sentence reason:
- Openness: Imagination, Intellect
- Conscientiousness: Achievement-Striving, Cautiousness, Orderliness, Self-Discipline, Self-Efficacy
- Extraversion: Activity-Level, Assertiveness, Cheerfulness, Friendliness
- Agreeableness: Cooperation, Morality, Trust
- Neuroticism: Emotionality

## 3. Motivation Assessment
- Overall: High / Medium / Low
- Pattern: consistent / rising / falling / fluctuating
- Voice indicators: 3-5 specific observations with actual numbers, language_profile context, AND valence/arousal
- Content indicators: from transcript or "N/A — voice-only"

## 4. Strengths (top 3 trait + top 3 motivation)
## 5. Development Areas (personality + motivation)
## 6. HR Summary (3 sentences, reference position, language_profile, and emotion model findings)

**MANDATORY:** Every score must cite at least one specific number from the voice data. Say "energy_mean=0.065 (high for non_native_english), mean_arousal=0.45" not just "high energy"."""


STRUCTURED_OUTPUT_PROMPT = """Based on your analysis above, now provide a structured JSON output with the following format:

{{
  "big_five": {{
    "openness": {{"score": <0-100>, "confidence": <0-100>, "reason": "<brief reason>"}},
    "conscientiousness": {{"score": <0-100>, "confidence": <0-100>, "reason": "<brief reason>"}},
    "extraversion": {{"score": <0-100>, "confidence": <0-100>, "reason": "<brief reason>"}},
    "agreeableness": {{"score": <0-100>, "confidence": <0-100>, "reason": "<brief reason>"}},
    "neuroticism": {{"score": <0-100>, "confidence": <0-100>, "reason": "<brief reason>"}}
  }},
  "motivation": {{
    "overall_level": "<High/Medium/Low>",
    "motivation_score": <0-100>,
    "pattern": "<description of pattern>",
    "voice_indicators": ["<indicator1>", "<indicator2>", ...],
    "content_indicators": ["<indicator1>", "<indicator2>", ...]
  }},
  "engagement": {{
    "overall_level": "<High/Medium/Low>",
    "engagement_score": <0-100>,
    "reason": "<1-2 sentences explaining engagement level based on voice>"
  }},
  "trait_strengths": ["<strength1>", "<strength2>", "<strength3>"],
  "motivation_strengths": ["<strength1>", "<strength2>", "<strength3>"],
  "personality_development_areas": ["<area1>", "<area2>"],
  "motivation_development_areas": ["<area1>", "<area2>"],
  "hr_summary": "<3-sentence summary>"
}}

Return ONLY the JSON, no additional text."""


APPROXIMATE_ASSESSMENT_PROMPT = """You are a voice-analytics expert. You have been given a comprehensive set of granular voice features extracted from an audio recording.

Your task: provide **APPROXIMATE** personality (Big Five), motivation, and engagement estimates based **solely on voice features**. These are indicative signals, NOT clinical scores.

=== GRANULAR VOICE FEATURES ===
{granular_features_json}

=== EMOTION TIMELINE SUMMARY ===
{emotion_timeline_summary}

=== INSTRUCTIONS ===

1. For each Big Five trait (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism):
   - Give an **approximate label**: "low", "moderate-low", "moderate", "moderate-high", or "high"
   - Give a **score range** (e.g. "55–70")
   - List 2-4 **specific voice features** that influenced your estimate, citing exact values
   - Use these mappings:
     * **Extraversion**: energy_mean, speaking_rate_wpm, arousal_proxy, speech_to_silence_ratio, voiced_ratio
     * **Openness**: pitch_range, pitch_variance, pitch_cv, spectral_centroid_mean, energy_range, dynamic_range_db
     * **Conscientiousness**: rhythm_regularity, pause_duration_std, energy_cv, articulation_to_speech_ratio
     * **Agreeableness**: hnr_mean_db, jitter_local, shimmer_local, valence_proxy, pitch_variance
     * **Neuroticism**: jitter_local, jitter_rap, shimmer_local, hnr_mean_db, long_pauses_count, energy_cv

2. For **Motivation** (approximate):
   - Label: "low", "moderate", or "high"
   - Score range (e.g. "60–75")
   - Key voice features: energy_mean, speaking_rate_wpm, pitch_slope, arousal_proxy, speech_to_silence_ratio

3. For **Engagement** (approximate):
   - Label: "low", "moderate", or "high"
   - Score range
   - Key voice features: energy_std, pitch_variance, arousal_proxy, spectral_flux_mean, voiced_ratio

**IMPORTANT:**
- All estimates are APPROXIMATE — label them clearly as such
- Always cite the exact feature name and its value
- These are voice-based indicators, not definitive psychological assessments
- Consider the emotion timeline trends (rising/falling arousal and valence) in your reasoning

=== OUTPUT FORMAT (JSON only) ===

{{
  "big5_approximate": {{
    "openness": {{"label": "<label>", "score_range": "<range>", "influencing_features": ["feature=value (reason)", ...]}},
    "conscientiousness": {{"label": "<label>", "score_range": "<range>", "influencing_features": [...]}},
    "extraversion": {{"label": "<label>", "score_range": "<range>", "influencing_features": [...]}},
    "agreeableness": {{"label": "<label>", "score_range": "<range>", "influencing_features": [...]}},
    "neuroticism": {{"label": "<label>", "score_range": "<range>", "influencing_features": [...]}}
  }},
  "motivation_approximate": {{"label": "<label>", "score_range": "<range>", "influencing_features": [...]}},
  "engagement_approximate": {{"label": "<label>", "score_range": "<range>", "influencing_features": [...]}}
}}

Return ONLY the JSON, no additional text."""


EMOTION_SUMMARY_BLOCK = """
=== FUSED EMOTION SUMMARY (from dual-model analysis) ===

Two independent emotion models (MERaLiON-SER and emotion2vec) analysed {total_segments} overlapping 8-second audio segments.
Their outputs were fused using temperature-scaled weighted averaging (MERaLiON weight=0.65, emotion2vec weight=0.35).

{emotion_summary_json}

**How to use this data for personality assessment:**

1. **Extraversion**: High arousal_mean (>0.3) + positive valence_mean (>0.2) + low neutral_ratio (<0.3) → higher Extraversion.
   Low arousal + high neutral_ratio → lower Extraversion.

2. **Neuroticism**: High emotion_volatility (>0.5) + negative valence_mean (<-0.2) + many "fearful"/"sad" segments → higher Neuroticism.
   Stable emotions + positive/neutral valence → lower Neuroticism.

3. **Agreeableness**: Positive valence_mean + moderate arousal (0.1-0.5) + low volatility → higher Agreeableness.
   Negative valence + high arousal (aggressive pattern) → lower Agreeableness.

4. **Openness**: High emotion_volatility (varied emotional expression) + wide valence_std + varied emotion_distribution → higher Openness.
   Flat, monotone emotional pattern → lower Openness.

5. **Conscientiousness**: Low emotion_volatility + stable arousal (low arousal_std) + high avg_confidence → higher Conscientiousness.
   Erratic emotions + low confidence → lower Conscientiousness.

6. **Motivation**: Rising arousal_trend + positive valence_trend + high avg_confidence → higher motivation.
   Falling arousal + negative valence_trend → lower motivation.

7. **Model agreement**: High model_agreement_rate (>0.6) increases confidence in emotion-based inferences.
   Low agreement (<0.3) means emotion signals are ambiguous — reduce weight on emotion-based traits.

**MANDATORY**: For every Big Five trait, cite at least one emotion summary metric alongside voice features.
Example: "Neuroticism=35 — emotion_volatility=0.22 (stable), valence_mean=0.1 (slightly positive), arousal_std=0.15 (consistent)"
"""


ABLATION_PROMPT = """You previously assessed this candidate using ONLY prosody and acoustic features (no emotion summary).
Now you have access to the FUSED EMOTION SUMMARY below, derived from two independent emotion models.

=== PREVIOUS ASSESSMENT (baseline, without emotion data) ===
{baseline_json}

=== FUSED EMOTION SUMMARY ===
{emotion_summary_json}

=== TASK ===
Re-evaluate the Big Five scores considering the emotion summary.
For each trait that CHANGED, explain:
1. The old score and reasoning
2. The new score and reasoning
3. Which emotion metrics caused the change

Return JSON:
{{{{
  "big_five_updated": {{{{
    "openness": {{{{"score": <0-100>, "confidence": <0-100>, "reason": "<reason citing emotion metrics>"}}}},
    "conscientiousness": {{{{"score": <0-100>, "confidence": <0-100>, "reason": "<reason>"}}}},
    "extraversion": {{{{"score": <0-100>, "confidence": <0-100>, "reason": "<reason>"}}}},
    "agreeableness": {{{{"score": <0-100>, "confidence": <0-100>, "reason": "<reason>"}}}},
    "neuroticism": {{{{"score": <0-100>, "confidence": <0-100>, "reason": "<reason>"}}}}
  }}}},
  "changes": [
    {{{{"trait": "<trait>", "old_score": <N>, "new_score": <N>, "delta": <N>, "emotion_metrics_cited": ["<metric>=<value>", ...]}}}}
  ],
  "emotion_impact_summary": "<1-2 sentences on how emotion data improved the assessment>"
}}}}

Return ONLY the JSON."""
