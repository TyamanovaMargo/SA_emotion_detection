"""Prompt templates for HR assessment."""

HR_ASSESSMENT_PROMPT = """You are an expert HR psychologist specializing in candidate assessment using **voice prosody, speech content, and motivation patterns**.

Analyze this candidate's profile for hiring and personality fit using the **Big Five** (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) and **motivation level**. Use BOTH the transcript and the voice features to infer personality and engagement.

=== INPUT DATA ===

TRANSCRIPT:
{transcript}

VOICE FEATURES:
{{
  "emotions": {emotions_json},
  "prosody": {{
    "speaking_rate_wpm": {speaking_rate},
    "pitch_mean_hz": {pitch_mean},
    "pitch_variance": {pitch_var},
    "energy_level": "{energy}",
    "pauses_per_minute": {pauses}
  }},
  "acoustic_features": {egemaps_summary},
  "wavlm_embedding_summary": "{embedding_profile}"
}}

=== MOTIVATION-RELATED VOICE PATTERNS ===
- **High motivation signs**:
  - Faster speech rate + moderate pauses
  - Higher energy level + stable pitch
  - Clear articulation and reduced filler words
  - Positive emotion (e.g., happy, confident) dominating
- **Low motivation signs**:
  - Slow speech rate + many pauses
  - Low energy and monotone pitch
  - High filler-word usage
  - Emotion dominated by neutral/sad/flat tone

=== PERSONALITY INTERPRETATION GUIDELINES ===

**CRITICAL: Use the FULL 0-100 scale. Avoid clustering scores around 40-60. Differentiate candidates clearly.**

- **Openness (0-100)**:
  - HIGH (70-100): Wide pitch range (variance >800), expressive/dynamic tone, creative vocabulary, exploratory language
  - MEDIUM (40-69): Moderate pitch variation (400-800), standard vocabulary, some expressiveness
  - LOW (0-39): Monotone/flat (variance <400), repetitive speech, conventional language, rigid patterns

- **Conscientiousness (0-100)**:
  - HIGH (70-100): Steady pace (100-140 wpm), <3 pauses/min, <2 fillers/min, structured speech, clear articulation
  - MEDIUM (40-69): Moderate pace/pauses, some fillers (2-5/min), generally organized
  - LOW (0-39): Erratic pace, >6 pauses/min, >5 fillers/min, disorganized, unclear structure

- **Extraversion (0-100)**:
  - HIGH (70-100): Fast speech (>150 wpm), high energy (>0.06 RMS), high pitch (>200 Hz), positive emotion dominant
  - MEDIUM (40-69): Moderate rate (120-150 wpm), medium energy (0.03-0.06), balanced emotion
  - LOW (0-39): Slow (<120 wpm), low energy (<0.03), subdued/withdrawn tone, minimal variation

- **Agreeableness (0-100)**:
  - HIGH (70-100): Warm/smooth prosody, cooperative tone, low pitch variance (<500), positive/neutral emotion, gentle energy
  - MEDIUM (40-69): Balanced warmth, moderate prosody, some cooperative signals
  - LOW (0-39): Cold/harsh tone, high pitch variance (>1000), angry/aggressive emotion, confrontational patterns

- **Neuroticism (0-100)**:
  - HIGH (70-100): Unstable pitch (variance >1000), >6 pauses/min, anxious/fearful/sad emotion, tense voice quality
  - MEDIUM (40-69): Some pitch instability (500-1000), moderate pauses, occasional negative emotion
  - LOW (0-39): Stable pitch (<500), <3 pauses/min, calm/confident tone, emotional stability

**SCORING RULES:**
1. Use specific numeric thresholds above - don't guess
2. Each candidate should have DIFFERENT scores - avoid identical profiles
3. Spread scores across the full 0-100 range
4. If uncertain, use confidence <60% rather than defaulting to 50/100  

=== OUTPUT FORMAT ===

## 1. Big Five Profile (0-100, with confidence)
- **Openness**: XX/100 (confidence: XX%) - brief reason based on voice+text  
- **Conscientiousness**: XX/100 (confidence: XX%) - brief reason  
- **Extraversion**: XX/100 (confidence: XX%) - brief reason  
- **Agreeableness**: XX/100 (confidence: XX%) - brief reason  
- **Neuroticism**: XX/100 (confidence: XX%) - brief reason  

## 2. Motivation Level Assessment
- **Overall motivation pattern**:  
  - High / Medium / Low (choose one)  
  - Duration: consistent across answer / fluctuates / starts low, ends high, etc.  
- **Key motivation indicators from voice**:  
  - 3-5 specific observations (e.g., "high energy with clear pitch variation", "few pauses and minimal fillers")  
- **Key motivation indicators from content**:  
  - 3-5 specific observations (e.g., "future-oriented statements", "proactive suggestions")  

## 3. HR-Relevant Strengths
- **Top 3 traits strengths** for this role, mentioning both voice and text  
- **Top 3 motivation-related strengths** (e.g., "high engagement signals", "persistent, determined tone")  

## 4. Areas for Development
- **Personality-related areas** (e.g., "tone sometimes sounds hesitant, suggesting low assertiveness")  
- **Motivation-related areas** (e.g., "speech energy drops in the second half of answers, indicating possible low sustained engagement")  

## 5. Summary for HR Decision
Write a 3-sentence summary suitable for an HR report:  
- Personality fit for a team-oriented/leadership/customer-facing role.  
- Level of motivation and engagement inferred from voice.  
- One-sentence recommendation insight (e.g., "highly motivated and confident candidate, but may need to moderate emotional expressiveness in a conservative environment").

Be specific about which voice features (energy, pitch, pauses, emotion) support each judgment.  
Use weighted scores: voice-based cues slightly less certain than text-based, but essential for detecting motivation and engagement."""


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
    "pattern": "<description of pattern>",
    "voice_indicators": ["<indicator1>", "<indicator2>", ...],
    "content_indicators": ["<indicator1>", "<indicator2>", ...]
  }},
  "trait_strengths": ["<strength1>", "<strength2>", "<strength3>"],
  "motivation_strengths": ["<strength1>", "<strength2>", "<strength3>"],
  "personality_development_areas": ["<area1>", "<area2>"],
  "motivation_development_areas": ["<area1>", "<area2>"],
  "hr_summary": "<3-sentence summary>"
}}

Return ONLY the JSON, no additional text."""
