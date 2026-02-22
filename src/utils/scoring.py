"""Shared scoring utilities: label mapping, validation, post-processing.

Used across pipeline, groq_assessor, voice_analyzer, and main.py
to ensure consistent score→label mapping and output validation.
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import numpy as np


# ── Canonical score → label mapping (#7, #16) ──────────────────────────

SCORE_LABELS = [
    (0, 30, "low"),
    (31, 45, "moderate-low"),
    (46, 55, "moderate"),
    (56, 70, "moderate-high"),
    (71, 85, "high"),
    (86, 100, "very high"),
]

BIG_FIVE_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


def score_to_label(score: int) -> str:
    """Convert a 0-100 score to a canonical text label."""
    for lo, hi, label in SCORE_LABELS:
        if lo <= score <= hi:
            return label
    return "moderate"


# ── WPM thresholds (#1) ────────────────────────────────────────────────

def wpm_label(wpm: float, language_profile: str = "native_english") -> str:
    """Return speaking rate label adjusted for language profile."""
    if language_profile in ("non_native_english", "sea_english"):
        # Non-native: multiply thresholds by 0.8
        if wpm < 88:
            return "slow"
        elif wpm <= 120:
            return "moderate"
        else:
            return "fast"
    else:
        if wpm < 110:
            return "slow"
        elif wpm <= 150:
            return "moderate"
        else:
            return "fast"


# ── Dominant emotion with tiebreaker (#3) ──────────────────────────────

def dominant_emotion_with_tiebreaker(
    emotion_stats: Dict[str, dict],
    timeline: List[Dict[str, Any]],
) -> str:
    """Pick dominant emotion with tiebreaker: dom_segments → mean → recency → 'mixed'."""
    if not emotion_stats:
        return "neutral"

    emotions_7 = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]

    # 1. Sort by dominant_segments descending
    ranked = sorted(
        emotions_7,
        key=lambda e: emotion_stats.get(e, {}).get("dominant_segments", 0),
        reverse=True,
    )
    top_count = emotion_stats.get(ranked[0], {}).get("dominant_segments", 0)
    tied = [e for e in ranked if emotion_stats.get(e, {}).get("dominant_segments", 0) == top_count]

    if len(tied) == 1:
        return tied[0]

    # 2. Tiebreaker: higher mean probability
    tied.sort(key=lambda e: emotion_stats.get(e, {}).get("mean", 0), reverse=True)
    top_mean = emotion_stats.get(tied[0], {}).get("mean", 0)
    still_tied = [e for e in tied if abs(emotion_stats.get(e, {}).get("mean", 0) - top_mean) < 0.01]

    if len(still_tied) == 1:
        return still_tied[0]

    # 3. Tiebreaker: recency — dominant in last 25% of segments
    if timeline:
        n = len(timeline)
        tail = timeline[-max(n // 4, 1):]
        tail_counts = Counter()
        for seg in tail:
            seg_scores = {e: seg.get(e, 0.0) for e in emotions_7}
            tail_counts[max(seg_scores, key=seg_scores.get)] += 1
        for e in still_tied:
            if tail_counts.get(e, 0) > tail_counts.get(still_tied[0], 0):
                return e
        # If one of the tied wins in tail
        best_tail = max(still_tied, key=lambda e: tail_counts.get(e, 0))
        if tail_counts.get(best_tail, 0) > tail_counts.get(still_tied[0], 0):
            return best_tail

    # 4. Still tied → "mixed"
    return "mixed"


# ── Slope / trend consistency (#4) ─────────────────────────────────────

def compute_slope_and_trend(values: np.ndarray, threshold: float = 0.005) -> Tuple[float, str]:
    """Compute linear slope and derive trend label consistently.
    
    Returns (slope_rounded_4dp, trend_label).
    Trend is always derived from slope, never independently.
    """
    n = len(values)
    if n < 3:
        return 0.0, "stable"
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, values, 1)[0])
    slope = round(slope, 4)
    if slope > threshold:
        trend = "rising"
    elif slope < -threshold:
        trend = "falling"
    else:
        trend = "stable"
    return slope, trend


# ── Strengths / development areas validation (#2, #5) ──────────────────

def validate_strengths_and_dev_areas(
    big_five: Dict[str, dict],
    trait_strengths: List[str],
    dev_areas: List[str],
) -> Tuple[List[str], List[str]]:
    """Validate strengths and development_areas:
    - No trait in both lists
    - Neuroticism is inverted: high → dev area, low → strength
    - Score >= 60 → strength only, <= 40 → dev only, 41-59 → neither
    """
    clean_strengths = []
    clean_dev = []

    for trait in BIG_FIVE_TRAITS:
        score_data = big_five.get(trait, {})
        score = score_data.get("score", 50) if isinstance(score_data, dict) else 50

        # Neuroticism is inverted (#5)
        if trait == "neuroticism":
            if score >= 60:
                clean_dev.append(f"Emotional Stability (Neuroticism {score})")
            elif score <= 40:
                clean_strengths.append(f"Emotional Stability (Neuroticism {score})")
        else:
            if score >= 60:
                clean_strengths.append(f"{trait.title()} ({score})")
            elif score <= 40:
                clean_dev.append(f"{trait.title()} ({score})")

    return clean_strengths, clean_dev


# ── Final assessment with weighted averaging (#6, #17) ──────────────────

def compute_final_assessment(
    basic_b5: Dict[str, dict],
    approximate_b5: Optional[Dict[str, Any]],
    enriched_b5: Optional[Dict[str, dict]],
) -> Dict[str, Any]:
    """Compute final_assessment by weighted-averaging across 3 modules.
    
    Returns per-trait: score, confidence, confidence_interval, calibration_warning.
    """
    result = {}
    calibration_warnings = []

    for trait in BIG_FIVE_TRAITS:
        scores = []
        confidences = []

        # Basic (LLM)
        basic = basic_b5.get(trait, {})
        b_score = basic.get("score", 50)
        b_conf = basic.get("confidence", 50)
        scores.append(b_score)
        confidences.append(b_conf)

        # Approximate (voice-only LLM)
        if approximate_b5:
            approx = approximate_b5.get(trait, {})
            sr = approx.get("score_range", "") if isinstance(approx, dict) else ""
            if isinstance(sr, str) and "-" in sr.replace("–", "-"):
                parts = sr.replace("–", "-").split("-")
                try:
                    a_lo, a_hi = int(parts[0].strip()), int(parts[1].strip())
                    a_score = (a_lo + a_hi) // 2
                    a_conf = max(40, 100 - (a_hi - a_lo))  # wider range → lower confidence
                    scores.append(a_score)
                    confidences.append(a_conf)
                except (ValueError, IndexError):
                    pass
            elif isinstance(approx, dict):
                # Try score_low / score_high
                if "score_low" in approx and "score_high" in approx:
                    a_lo, a_hi = approx["score_low"], approx["score_high"]
                    a_score = (a_lo + a_hi) // 2
                    a_conf = max(40, 100 - (a_hi - a_lo))
                    scores.append(a_score)
                    confidences.append(a_conf)

        # Enriched (ablation)
        if enriched_b5:
            enr = enriched_b5.get(trait, {})
            if isinstance(enr, dict) and "score" in enr:
                e_score = enr["score"]
                e_conf = enr.get("confidence", 60)
                scores.append(e_score)
                confidences.append(e_conf)

        # Weighted average
        if not scores:
            result[trait] = {"score": 50, "confidence": 30, "confidence_interval": [40, 60], "label": "moderate"}
            continue

        weights = [c / 100.0 for c in confidences]
        total_w = sum(weights)
        if total_w > 0:
            final_score = int(round(sum(s * w for s, w in zip(scores, weights)) / total_w))
        else:
            final_score = int(round(np.mean(scores)))

        spread = max(scores) - min(scores)
        avg_conf = int(round(np.mean(confidences)))

        # Confidence interval: ±1 std of the estimates
        if len(scores) > 1:
            std = float(np.std(scores))
            ci_lo = max(0, int(round(final_score - std)))
            ci_hi = min(100, int(round(final_score + std)))
        else:
            ci_lo = max(0, final_score - 10)
            ci_hi = min(100, final_score + 10)

        entry = {
            "score": final_score,
            "confidence": avg_conf,
            "confidence_interval": [ci_lo, ci_hi],
            "label": score_to_label(final_score),
        }

        if spread > 15:
            entry["calibration_warning"] = True
            calibration_warnings.append({
                "trait": trait,
                "spread": spread,
                "scores": {
                    "basic": scores[0],
                    "approximate": scores[1] if len(scores) > 1 else None,
                    "enriched": scores[2] if len(scores) > 2 else None,
                },
            })

        result[trait] = entry

    return {"traits": result, "calibration_warnings": calibration_warnings}


# ── HR summary from final scores (#7) ──────────────────────────────────

def generate_hr_summary_from_scores(
    final_assessment: Dict[str, Any],
    motivation_score: int,
    engagement_score: Optional[int],
    dominant_emotion: str = "neutral",
) -> str:
    """Generate HR summary programmatically from final_assessment scores."""
    traits = final_assessment.get("traits", {})
    parts = []

    # Personality overview
    trait_descriptions = []
    for trait in BIG_FIVE_TRAITS:
        t = traits.get(trait, {})
        label = t.get("label", "moderate")
        score = t.get("score", 50)
        if trait == "neuroticism":
            # Invert for readability
            stability = "emotionally stable" if score <= 45 else "somewhat emotionally reactive" if score <= 65 else "emotionally reactive"
            trait_descriptions.append(stability)
        else:
            if score >= 60 or score <= 40:
                trait_descriptions.append(f"{label} {trait}")

    if trait_descriptions:
        parts.append(f"The candidate shows {', '.join(trait_descriptions[:3])}")

    # Motivation
    mot_label = score_to_label(motivation_score)
    parts.append(f"{mot_label} motivation (score: {motivation_score})")

    # Engagement
    if engagement_score is not None:
        eng_label = score_to_label(engagement_score)
        parts.append(f"{eng_label} engagement (score: {engagement_score})")

    # Emotion
    if dominant_emotion not in ("neutral", "mixed"):
        parts.append(f"dominant vocal emotion: {dominant_emotion}")

    # Calibration warnings
    warnings = final_assessment.get("calibration_warnings", [])
    if warnings:
        flagged = [w["trait"] for w in warnings]
        parts.append(f"Note: {', '.join(flagged)} scores had high cross-module variance — interpret with caution")

    return ". ".join(parts) + "."


# ── Final emotion arc (#10) ────────────────────────────────────────────

def compute_final_emotion_arc(
    timeline: List[Dict[str, Any]],
    window_seconds: float = 15.0,
    segment_step: float = 2.0,
) -> Dict[str, Any]:
    """Compute emotion arc for the last `window_seconds` of the timeline."""
    if not timeline:
        return {}

    emotions_7 = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]

    # Estimate how many segments fit in the window
    n_segs = max(1, int(window_seconds / segment_step))
    tail = timeline[-n_segs:]

    # Dominant emotion in tail
    tail_counts = Counter()
    val_sum = ar_sum = 0.0
    for seg in tail:
        seg_scores = {e: seg.get(e, 0.0) for e in emotions_7}
        tail_counts[max(seg_scores, key=seg_scores.get)] += 1
        val_sum += seg.get("vad", {}).get("valence", 0.0)
        ar_sum += seg.get("vad", {}).get("arousal", 0.0)

    n = len(tail)
    dominant = tail_counts.most_common(1)[0][0] if tail_counts else "neutral"

    return {
        "window_seconds": window_seconds,
        "segments_in_window": n,
        "dominant_emotion": dominant,
        "valence_mean": round(val_sum / max(n, 1), 4),
        "arousal_mean": round(ar_sum / max(n, 1), 4),
    }


# ── Emotion stability on 10s windows (#11) ────────────────────────────

def compute_emotion_stability_10s(
    timeline: List[Dict[str, Any]],
    segment_step: float = 2.0,
    window_sec: float = 10.0,
) -> float:
    """Compute emotion stability on non-overlapping 10s windows.
    
    Returns stability = 1 - (changes / num_windows). Range [0, 1].
    1.0 = perfectly stable, 0.0 = changes every window.
    """
    if not timeline or len(timeline) < 2:
        return 1.0

    emotions_7 = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]

    # Group segments into non-overlapping 10s windows
    segs_per_window = max(1, int(window_sec / segment_step))
    windows = []
    for i in range(0, len(timeline), segs_per_window):
        chunk = timeline[i:i + segs_per_window]
        if chunk:
            counts = Counter()
            for seg in chunk:
                seg_scores = {e: seg.get(e, 0.0) for e in emotions_7}
                counts[max(seg_scores, key=seg_scores.get)] += 1
            windows.append(counts.most_common(1)[0][0])

    if len(windows) < 2:
        return 1.0

    changes = sum(1 for i in range(1, len(windows)) if windows[i] != windows[i - 1])
    return round(1.0 - changes / (len(windows) - 1), 4)


# ── Adjusted emotional shifts (#12) ────────────────────────────────────

def adjusted_emotional_shifts(raw_shifts: int, step_size: float, window_size: float) -> float:
    """Normalize emotional_shifts for overlapping windows."""
    if window_size <= 0:
        return float(raw_shifts)
    return round(raw_shifts * (step_size / window_size), 2)


# ── Content-voice alignment (#15) ──────────────────────────────────────

def compute_content_voice_alignment(
    transcript: str,
    valence_mean: float,
) -> Dict[str, Any]:
    """Basic content-voice alignment: compare transcript sentiment proxies with voice valence."""
    if not transcript:
        return {"available": False}

    words = transcript.lower().split()
    word_count = len(words)

    # Simple lexical sentiment proxies
    positive_words = {"good", "great", "love", "like", "enjoy", "happy", "interesting",
                      "amazing", "wonderful", "excellent", "positive", "beautiful", "best",
                      "fun", "excited", "agree", "yes", "sure", "absolutely", "definitely"}
    negative_words = {"bad", "hate", "dislike", "sad", "terrible", "awful", "worst",
                      "boring", "angry", "frustrated", "difficult", "hard", "no", "never",
                      "problem", "wrong", "fail", "unfortunately", "sorry", "worried"}
    hedge_words = {"maybe", "perhaps", "possibly", "might", "could", "somewhat",
                   "kind of", "sort of", "i guess", "i think", "probably", "not sure"}

    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    hedge_count = sum(1 for w in words if w in hedge_words)
    self_ref = sum(1 for w in words if w in ("i", "me", "my", "myself", "i'm", "i've"))

    text_sentiment = (pos_count - neg_count) / max(word_count, 1)
    voice_valence_label = "positive" if valence_mean > 0.1 else "negative" if valence_mean < -0.1 else "neutral"
    text_sentiment_label = "positive" if text_sentiment > 0.02 else "negative" if text_sentiment < -0.02 else "neutral"

    mismatch = (voice_valence_label == "negative" and text_sentiment_label == "positive") or \
               (voice_valence_label == "positive" and text_sentiment_label == "negative")

    return {
        "available": True,
        "text_sentiment": round(text_sentiment, 4),
        "text_sentiment_label": text_sentiment_label,
        "voice_valence_mean": round(valence_mean, 4),
        "voice_valence_label": voice_valence_label,
        "content_voice_mismatch": mismatch,
        "mismatch_note": "Possible emotional masking: text sentiment diverges from vocal tone" if mismatch else None,
        "linguistic_indicators": {
            "positive_word_count": pos_count,
            "negative_word_count": neg_count,
            "hedge_word_count": hedge_count,
            "self_reference_count": self_ref,
            "assertion_ratio": round((word_count - hedge_count) / max(word_count, 1), 3),
        },
    }
