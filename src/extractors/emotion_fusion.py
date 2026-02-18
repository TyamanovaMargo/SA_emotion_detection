"""Emotion fusion, reliability metrics, smoothing, and summary aggregation.

Provides:
- SNR / entropy / top2-gap computation
- Temperature-scaled weighted probability fusion (MERaLiON + emotion2vec)
- Energy-based VAD with adaptive thresholds
- Sticky-transition temporal smoothing
- Rich emotion summary for LLM consumption
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import math
from collections import Counter


# ---------------------------------------------------------------------------
# Fusion defaults
# ---------------------------------------------------------------------------
DEFAULT_WEIGHT_MERALION = 0.65
DEFAULT_WEIGHT_E2V = 0.35
DEFAULT_TEMP_MERALION = 1.0
DEFAULT_TEMP_E2V = 1.2
STICKY_PENALTY = 0.15          # penalty for changing emotion between segments
MIN_SPEECH_ENERGY = 0.005      # minimum RMS for speech
MIN_SEGMENT_DURATION = 1.5     # seconds
MERGE_GAP = 0.3                # seconds - merge speech regions closer than this
MIN_SNR_DB = 8.0               # minimum SNR to trust a segment


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------

def compute_entropy(probs: Dict[str, float]) -> float:
    """Shannon entropy of probability distribution (bits). Lower = more certain."""
    h = 0.0
    for p in probs.values():
        if p > 1e-12:
            h -= p * math.log2(p)
    return round(h, 4)


def compute_top2_gap(probs: Dict[str, float]) -> float:
    """Gap between highest and second-highest probability."""
    sorted_p = sorted(probs.values(), reverse=True)
    if len(sorted_p) < 2:
        return sorted_p[0] if sorted_p else 0.0
    return round(sorted_p[0] - sorted_p[1], 4)


def compute_snr(segment: np.ndarray, noise_floor: float) -> float:
    """Compute segment SNR in dB relative to estimated noise floor."""
    rms = float(np.sqrt(np.mean(segment ** 2)))
    if noise_floor < 1e-10:
        return 60.0  # effectively infinite SNR
    snr = 20 * np.log10(max(rms, 1e-10) / noise_floor)
    return round(snr, 1)


def estimate_noise_floor(audio: np.ndarray, sample_rate: int) -> float:
    """Estimate noise floor from 10th percentile of short-window RMS."""
    window_size = int(0.025 * sample_rate)  # 25ms
    hop = window_size // 2
    rms_values = []
    for i in range(0, len(audio) - window_size, hop):
        w = audio[i:i + window_size]
        rms_values.append(float(np.sqrt(np.mean(w ** 2))))
    if rms_values:
        return max(float(np.percentile(rms_values, 10)), 1e-6)
    return 0.005


# ---------------------------------------------------------------------------
# Temperature scaling + fusion
# ---------------------------------------------------------------------------

def temperature_scale(probs: Dict[str, float], temperature: float) -> Dict[str, float]:
    """Apply temperature scaling to probability distribution."""
    if temperature <= 0:
        temperature = 1.0
    # Convert to log-probs, scale, re-softmax
    labels = list(probs.keys())
    vals = np.array([max(probs[l], 1e-12) for l in labels])
    logits = np.log(vals) / temperature
    logits -= logits.max()  # numerical stability
    exp_logits = np.exp(logits)
    scaled = exp_logits / exp_logits.sum()
    return {l: round(float(s), 6) for l, s in zip(labels, scaled)}


def fuse_probabilities(
    mer_probs: Optional[Dict[str, float]],
    e2v_probs: Optional[Dict[str, float]],
    w_mer: float = DEFAULT_WEIGHT_MERALION,
    w_e2v: float = DEFAULT_WEIGHT_E2V,
    temp_mer: float = DEFAULT_TEMP_MERALION,
    temp_e2v: float = DEFAULT_TEMP_E2V,
) -> Dict[str, float]:
    """
    Fuse probability distributions from two models with temperature scaling.
    
    Returns fused probability dict over union of labels.
    If one model is None, returns the other (temperature-scaled).
    """
    if mer_probs is None and e2v_probs is None:
        return {}

    if mer_probs is None:
        return temperature_scale(e2v_probs, temp_e2v)
    if e2v_probs is None:
        return temperature_scale(mer_probs, temp_mer)

    # Temperature scale each
    mer_scaled = temperature_scale(mer_probs, temp_mer)
    e2v_scaled = temperature_scale(e2v_probs, temp_e2v)

    # Union of labels
    all_labels = set(mer_scaled.keys()) | set(e2v_scaled.keys())
    
    # Normalize weights
    total_w = w_mer + w_e2v
    wm = w_mer / total_w
    we = w_e2v / total_w

    fused = {}
    for label in all_labels:
        fused[label] = wm * mer_scaled.get(label, 0.0) + we * e2v_scaled.get(label, 0.0)

    # Re-normalize
    total = sum(fused.values())
    if total > 0:
        fused = {k: round(v / total, 6) for k, v in fused.items()}

    return fused


# ---------------------------------------------------------------------------
# Energy-based VAD
# ---------------------------------------------------------------------------

def energy_vad_segments(
    audio: np.ndarray,
    sample_rate: int,
    min_speech_sec: float = MIN_SEGMENT_DURATION,
    merge_gap_sec: float = MERGE_GAP,
    energy_threshold_factor: float = 3.0,
) -> List[Tuple[int, int]]:
    """
    Simple energy-based VAD returning speech regions as (start_sample, end_sample).
    
    Uses adaptive threshold: speech if RMS > noise_floor * energy_threshold_factor.
    Merges nearby regions and discards short ones.
    """
    noise_floor = estimate_noise_floor(audio, sample_rate)
    threshold = max(noise_floor * energy_threshold_factor, MIN_SPEECH_ENERGY)

    frame_size = int(0.03 * sample_rate)  # 30ms frames
    hop = frame_size // 2
    
    # Compute per-frame energy
    speech_frames = []
    for i in range(0, len(audio) - frame_size, hop):
        rms = float(np.sqrt(np.mean(audio[i:i + frame_size] ** 2)))
        speech_frames.append((i, rms >= threshold))

    if not speech_frames:
        return [(0, len(audio))]

    # Extract contiguous speech regions
    regions = []
    in_speech = False
    start = 0
    for pos, is_speech in speech_frames:
        if is_speech and not in_speech:
            start = pos
            in_speech = True
        elif not is_speech and in_speech:
            regions.append((start, pos))
            in_speech = False
    if in_speech:
        regions.append((start, len(audio)))

    if not regions:
        return [(0, len(audio))]

    # Merge regions closer than merge_gap_sec
    merge_samples = int(merge_gap_sec * sample_rate)
    merged = [regions[0]]
    for s, e in regions[1:]:
        if s - merged[-1][1] <= merge_samples:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # Discard short regions
    min_samples = int(min_speech_sec * sample_rate)
    merged = [(s, e) for s, e in merged if (e - s) >= min_samples]

    if not merged:
        # Fall back to full audio
        return [(0, len(audio))]

    return merged


# ---------------------------------------------------------------------------
# Sticky-transition temporal smoothing
# ---------------------------------------------------------------------------

def smooth_timeline_sticky(
    timeline: List[Dict[str, Any]],
    penalty: float = STICKY_PENALTY,
    key: str = "fused_emotion",
) -> List[Dict[str, Any]]:
    """
    Apply sticky-transition smoothing to emotion timeline.
    
    For each segment, if switching emotion costs `penalty` in confidence,
    only switch if the new emotion's confidence exceeds
    current_run_emotion + penalty.
    
    This prevents rapid flipping between emotions.
    """
    if len(timeline) < 2:
        return timeline

    smoothed = [dict(t) for t in timeline]
    current_emotion = smoothed[0].get(key, "neutral")

    for i in range(1, len(smoothed)):
        seg = smoothed[i]
        new_emotion = seg.get(key, "neutral")
        new_conf = seg.get("fused_confidence", 0.0)
        
        # Get fused_scores if available for current_emotion confidence
        fused_scores = seg.get("fused_scores", {})
        current_score = fused_scores.get(current_emotion, 0.0)

        if new_emotion != current_emotion:
            # Only switch if new emotion clearly beats current + penalty
            if new_conf < current_score + penalty:
                # Keep previous emotion
                seg[key] = current_emotion
                seg["fused_confidence"] = current_score
                seg["smoothed"] = True
            else:
                current_emotion = new_emotion
        else:
            current_emotion = new_emotion

    return smoothed


# ---------------------------------------------------------------------------
# Emotion summary aggregation for LLM
# ---------------------------------------------------------------------------

def compute_emotion_summary(timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute a rich emotion summary from the fused timeline for LLM consumption.
    
    Returns dict with: dominant_emotion, ratios, volatility, valence/arousal 
    stats, trends, agreement rate, confidence/entropy stats, top transitions.
    """
    if not timeline:
        return {"error": "no timeline data"}

    n = len(timeline)

    # Collect per-segment data
    emotions = [s.get("fused_emotion", s.get("emotion", "neutral")) for s in timeline]
    valences = [s.get("fused_valence", s.get("valence", 0.0)) for s in timeline]
    arousals = [s.get("fused_arousal", s.get("arousal", 0.0)) for s in timeline]
    confidences = [s.get("fused_confidence", s.get("confidence", 0.0)) for s in timeline]
    entropies = [s.get("entropy", 0.0) for s in timeline]
    agreements = [s.get("models_agree", None) for s in timeline]

    # Emotion distribution
    emo_counts = Counter(emotions)
    dominant_emotion = emo_counts.most_common(1)[0][0]
    dominant_ratio = round(emo_counts[dominant_emotion] / n, 3)

    # Volatility: 1 - (longest consecutive run / total segments)
    max_run = 1
    current_run = 1
    for i in range(1, n):
        if emotions[i] == emotions[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    volatility = round(1.0 - max_run / n, 3)

    # Neutral ratio
    neutral_count = emo_counts.get("neutral", 0)
    neutral_ratio = round(neutral_count / n, 3)

    # Valence/arousal stats
    val_arr = np.array(valences)
    ar_arr = np.array(arousals)

    # Trends (linear regression slope)
    if n >= 3:
        x = np.arange(n, dtype=float)
        val_slope = float(np.polyfit(x, val_arr, 1)[0])
        ar_slope = float(np.polyfit(x, ar_arr, 1)[0])
    else:
        val_slope = 0.0
        ar_slope = 0.0

    def trend_label(slope: float) -> str:
        if slope > 0.02:
            return "rising"
        elif slope < -0.02:
            return "falling"
        return "stable"

    # Model agreement rate
    agree_count = sum(1 for a in agreements if a is True)
    disagree_count = sum(1 for a in agreements if a is False)
    total_compared = agree_count + disagree_count
    agreement_rate = round(agree_count / total_compared, 3) if total_compared > 0 else None

    # Top transitions
    transitions = Counter()
    for i in range(1, n):
        if emotions[i] != emotions[i - 1]:
            transitions[f"{emotions[i-1]}→{emotions[i]}"] += 1
    top_transition = transitions.most_common(1)[0] if transitions else None

    # Low-confidence ratio
    low_conf_count = sum(1 for c in confidences if c < 0.4)

    summary = {
        "total_segments": n,
        "dominant_emotion": dominant_emotion,
        "dominant_emotion_ratio": dominant_ratio,
        "emotion_volatility": volatility,
        "emotion_distribution": dict(emo_counts),
        "neutral_ratio": neutral_ratio,
        "valence_mean": round(float(np.mean(val_arr)), 3),
        "valence_std": round(float(np.std(val_arr)), 3),
        "arousal_mean": round(float(np.mean(ar_arr)), 3),
        "arousal_std": round(float(np.std(ar_arr)), 3),
        "valence_trend": trend_label(val_slope),
        "valence_slope": round(val_slope, 4),
        "arousal_trend": trend_label(ar_slope),
        "arousal_slope": round(ar_slope, 4),
        "avg_confidence": round(float(np.mean(confidences)), 3),
        "avg_entropy": round(float(np.mean(entropies)), 3) if any(e > 0 for e in entropies) else None,
        "low_confidence_ratio": round(low_conf_count / n, 3),
        "model_agreement_rate": agreement_rate,
        "top_transition": f"{top_transition[0]} ({top_transition[1]}x)" if top_transition else None,
    }

    return summary
