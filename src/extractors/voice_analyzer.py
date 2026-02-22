"""Unified voice feature analyzer with emotion dynamics.

Extracts ALL scientifically-backed voice features from audio:
- Prosodic features (F0 stats, energy, speech rate, pauses)
- Voice quality (HNR, jitter, shimmer, H1-H2)
- Spectral features (MFCC mean/std)
- Emotion timeline with overlapping segments + VAD
- Emotion/stress/confidence aggregates and dynamics
- Paralinguistic text summary for LLM

Uses MERaLiON-SER-v1 for emotion detection on 5s overlapping segments
with 2s step and energy-based VAD filtering.
"""

from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import librosa
import math
import gc
from collections import Counter

from ..config import EmotionConfig, ProsodyConfig
from .emotion_meralion import EmotionDetector, normalize_label, EMOTION_LABELS
from .emotion_fusion import (
    estimate_noise_floor, compute_snr, compute_entropy, compute_top2_gap,
    energy_vad_segments, MIN_SNR_DB,
)

# VAD mapping: emotion label -> valence/arousal/dominance
EMOTION_VAD = {
    "happy":     {"valence": 0.80, "arousal": 0.60, "dominance": 0.65},
    "surprised": {"valence": 0.30, "arousal": 0.80, "dominance": 0.40},
    "angry":     {"valence": -0.70, "arousal": 0.90, "dominance": 0.80},
    "fearful":   {"valence": -0.60, "arousal": 0.70, "dominance": 0.20},
    "disgusted": {"valence": -0.80, "arousal": 0.40, "dominance": 0.50},
    "sad":       {"valence": -0.70, "arousal": -0.40, "dominance": 0.20},
    "neutral":   {"valence": 0.00, "arousal": 0.00, "dominance": 0.50},
    "other":     {"valence": 0.00, "arousal": 0.10, "dominance": 0.40},
    "undetected": {"valence": 0.0, "arousal": 0.0, "dominance": 0.3},
    "low_snr":   {"valence": 0.0, "arousal": 0.0, "dominance": 0.3},
}


def _get_vad(emotion: str) -> Dict[str, float]:
    return EMOTION_VAD.get(emotion, EMOTION_VAD["neutral"])


class VoiceAnalyzer:
    """Unified voice feature extractor with emotion dynamics."""

    def __init__(
        self,
        emotion_config: Optional[EmotionConfig] = None,
        prosody_config: Optional[ProsodyConfig] = None,
        segment_duration: float = 5.0,
        segment_step: float = 2.0,
        min_segment_sec: float = 1.5,
        emotion_detector: Optional[EmotionDetector] = None,
    ):
        self.emotion_config = emotion_config or EmotionConfig()
        self.prosody_config = prosody_config or ProsodyConfig()
        self.segment_duration = segment_duration
        self.segment_step = segment_step
        self.min_segment_sec = min_segment_sec

        # Share detector if provided, otherwise create lazily
        self._emotion_detector: Optional[EmotionDetector] = emotion_detector

    @property
    def emotion_detector(self) -> EmotionDetector:
        if self._emotion_detector is None:
            self._emotion_detector = EmotionDetector(self.emotion_config)
        return self._emotion_detector

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int,
        word_count: int = 0,
    ) -> Dict[str, Any]:
        """
        Run full voice analysis and return simplified JSON structure.

        Returns:
            {
                "prosody": {...},
                "voice_quality": {...},
                "spectral": {...},
                "emotion_timeline": [...],
                "emotion_aggregates": {...},
                "paralinguistic_summary": "..."
            }
        """
        duration = len(audio) / sample_rate

        prosody = self._extract_prosody(audio, sample_rate, word_count, duration)
        voice_quality = self._extract_voice_quality(audio, sample_rate)
        spectral = self._extract_spectral(audio, sample_rate)
        emotion_timeline = self._build_emotion_timeline(audio, sample_rate)
        emotion_aggregates = self._compute_emotion_aggregates(emotion_timeline, prosody)
        paralinguistic_summary = self._generate_paralinguistic_summary(
            prosody, voice_quality, spectral, emotion_aggregates, duration,
            emotion_timeline=emotion_timeline,
        )

        return {
            "prosody": prosody,
            "voice_quality": voice_quality,
            "spectral": spectral,
            "emotion_timeline": emotion_timeline,
            "emotion_aggregates": emotion_aggregates,
            "paralinguistic_summary": paralinguistic_summary,
        }

    # ------------------------------------------------------------------
    # 1. PROSODIC FEATURES
    # ------------------------------------------------------------------

    def _extract_prosody(
        self,
        audio: np.ndarray,
        sr: int,
        word_count: int,
        duration: float,
    ) -> Dict[str, Any]:
        cfg = self.prosody_config
        hop = int(cfg.hop_length * sr)
        frame = int(cfg.frame_length * sr)

        # --- F0 ---
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=cfg.min_pitch, fmax=cfg.max_pitch,
            sr=sr, frame_length=frame, hop_length=hop,
        )
        f0v = f0[~np.isnan(f0)]

        if len(f0v) > 0:
            f0_mean = float(np.mean(f0v))
            f0_median = float(np.median(f0v))
            f0_min = float(np.min(f0v))
            f0_max = float(np.max(f0v))
            f0_std = float(np.std(f0v))
            f0_range = f0_max - f0_min

            # F0 delta (frame-to-frame change)
            f0_delta = np.diff(f0v)
            f0_delta_std = float(np.std(f0_delta)) if len(f0_delta) > 0 else 0.0

            # F0 delta entropy
            if len(f0_delta) > 2:
                hist, _ = np.histogram(f0_delta, bins=20, density=True)
                hist = hist[hist > 0]
                f0_delta_entropy = float(-np.sum(hist * np.log2(hist + 1e-12)) * (f0_delta.max() - f0_delta.min()) / 20)
            else:
                f0_delta_entropy = 0.0
        else:
            f0_mean = f0_median = f0_min = f0_max = f0_std = f0_range = 0.0
            f0_delta_std = f0_delta_entropy = 0.0

        # --- Energy / Loudness ---
        rms = librosa.feature.rms(y=audio, frame_length=frame, hop_length=hop)[0]
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))
        energy_range = float(np.max(rms) - np.min(rms))

        # --- Voiced ratio ---
        total_frames = max(len(f0), 1)
        voiced_frames = int(np.sum(~np.isnan(f0)))
        voiced_ratio = voiced_frames / total_frames

        # --- Speech rate ---
        if word_count and word_count > 0 and duration > 0:
            # Prefer transcript-based rate (words per second)
            speech_rate = word_count / duration
        else:
            # Fallback: onset-based syllable estimation
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=False)
            syllable_count = len(onsets)
            speech_rate = syllable_count / duration if duration > 0 else 0.0

        # --- Pauses ---
        pauses = self._detect_pauses(audio, sr, rms, hop)
        pause_durations = [e - s for s, e in pauses]
        pause_count = len(pauses)
        total_pause_time = sum(pause_durations) if pause_durations else 0.0
        pause_mean_duration = float(np.mean(pause_durations)) if pause_durations else 0.0
        pause_max_duration = float(np.max(pause_durations)) if pause_durations else 0.0
        pause_time_ratio = total_pause_time / duration if duration > 0 else 0.0

        return {
            "f0_mean": round(f0_mean, 2),
            "f0_median": round(f0_median, 2),
            "f0_min": round(f0_min, 2),
            "f0_max": round(f0_max, 2),
            "f0_std": round(f0_std, 2),
            "f0_range": round(f0_range, 2),
            "f0_delta_std": round(f0_delta_std, 4),
            "f0_delta_entropy": round(f0_delta_entropy, 4),
            "energy_mean": round(energy_mean, 5),
            "energy_std": round(energy_std, 5),
            "energy_range": round(energy_range, 5),
            "speech_rate": round(speech_rate, 2),
            "voiced_ratio": round(voiced_ratio, 3),
            "pause_count": pause_count,
            "pause_mean_duration": round(pause_mean_duration, 3),
            "pause_max_duration": round(pause_max_duration, 3),
            "pause_time_ratio": round(pause_time_ratio, 3),
        }

    def _detect_pauses(
        self,
        audio: np.ndarray,
        sr: int,
        rms: np.ndarray,
        hop: int,
        min_pause_sec: float = 0.3,
        silence_threshold: float = 0.0,
    ) -> List[Tuple[float, float]]:
        hop_sec = hop / sr
        # Adaptive threshold: 15% of median RMS (handles loud/quiet recordings)
        if silence_threshold <= 0:
            median_rms = float(np.median(rms))
            silence_threshold = max(median_rms * 0.15, 0.002)
        is_silence = rms < silence_threshold
        pauses = []
        pause_start = None

        for i, silent in enumerate(is_silence):
            t = i * hop_sec
            if silent and pause_start is None:
                pause_start = t
            elif not silent and pause_start is not None:
                dur = t - pause_start
                if dur >= min_pause_sec:
                    pauses.append((pause_start, t))
                pause_start = None

        if pause_start is not None:
            final_t = len(is_silence) * hop_sec
            if final_t - pause_start >= min_pause_sec:
                pauses.append((pause_start, final_t))

        return pauses

    # ------------------------------------------------------------------
    # 2. VOICE QUALITY (parselmouth)
    # ------------------------------------------------------------------

    def _extract_voice_quality(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        result = {
            "HNR": 0.0,
            "jitter": 0.0,
            "shimmer": 0.0,
            "H1_H2": 0.0,
        }

        try:
            import parselmouth
            from parselmouth.praat import call
        except ImportError:
            print("  [warn] parselmouth not installed — voice quality zeros")
            return result

        try:
            snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sr))
            f0_min = self.prosody_config.min_pitch
            f0_max = self.prosody_config.max_pitch
            pitch_obj = call(snd, "To Pitch", 0.0, f0_min, f0_max)
            point_proc = call(snd, "To PointProcess (periodic, cc)", f0_min, f0_max)
        except Exception as e:
            print(f"  [warn] parselmouth base objects failed: {e}")
            return result

        # Jitter (local) — called on PointProcess
        try:
            jit = call(point_proc, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            if not math.isnan(jit):
                result["jitter"] = round(float(jit), 6)
        except Exception as e:
            print(f"  [warn] jitter extraction failed: {e}")

        # Shimmer (local) — needs BOTH Sound and PointProcess
        try:
            shim = call([snd, point_proc], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            if not math.isnan(shim):
                result["shimmer"] = round(float(shim), 6)
        except Exception as e:
            print(f"  [warn] shimmer extraction failed: {e}")

        # HNR (Harmonics-to-Noise Ratio)
        try:
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, f0_min, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            if not math.isnan(hnr):
                result["HNR"] = round(float(hnr), 2)
        except Exception as e:
            print(f"  [warn] HNR extraction failed: {e}")

        # H1-H2: spectral tilt proxy
        try:
            spectrum = call(snd, "To Spectrum", "yes")
            ltas = call(spectrum, "To Ltas (1-to-1)")
            f0_est = call(pitch_obj, "Get mean", 0, 0, "Hertz")
            if f0_est > 0 and not math.isnan(f0_est):
                h1_db = float(call(ltas, "Get value at frequency", f0_est, "Nearest"))
                h2_db = float(call(ltas, "Get value at frequency", f0_est * 2, "Nearest"))
                if not (math.isnan(h1_db) or math.isnan(h2_db)):
                    result["H1_H2"] = round(h1_db - h2_db, 2)
        except Exception as e:
            print(f"  [warn] H1-H2 extraction failed: {e}")

        return result

    # ------------------------------------------------------------------
    # 3. SPECTRAL FEATURES (MFCC)
    # ------------------------------------------------------------------

    def _extract_spectral(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        n_mfcc = 13
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        result = {}
        for i in range(n_mfcc):
            result[f"mfcc_mean_{i}"] = round(float(np.mean(mfccs[i])), 4)
            result[f"mfcc_std_{i}"] = round(float(np.std(mfccs[i])), 4)

        return result

    # ------------------------------------------------------------------
    # 4. EMOTION TIMELINE (overlapping segments + VAD)
    # ------------------------------------------------------------------

    def _build_emotion_timeline(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> List[Dict[str, Any]]:
        """
        Build emotion timeline with overlapping 5s segments, 2s step, VAD filtering.
        Each segment: {start_sec, end_sec, emotions: {7 probs}, vad: {v,a,d}}
        """
        duration = len(audio) / sr
        seg_samples = int(self.segment_duration * sr)
        step_samples = int(self.segment_step * sr)
        min_samples = int(self.min_segment_sec * sr)
        total_samples = len(audio)

        # Get VAD speech regions
        vad_regions = energy_vad_segments(audio, sr, min_speech_sec=1.0, merge_gap_sec=0.3)

        # Noise floor for SNR
        noise_floor = estimate_noise_floor(audio, sr)

        # Load emotion model
        self.emotion_detector._load_model()

        timeline = []
        pos = 0

        while pos < total_samples:
            end = min(pos + seg_samples, total_samples)
            segment = audio[pos:end]

            if len(segment) < min_samples:
                break

            start_sec = pos / sr
            end_sec = end / sr

            # Check if segment overlaps with any VAD speech region
            seg_has_speech = any(
                not (end <= vad_s or pos >= vad_e)
                for vad_s, vad_e in vad_regions
            )

            if not seg_has_speech:
                pos += step_samples
                continue

            # SNR check
            rms_energy = float(np.sqrt(np.mean(segment ** 2)))
            snr_db = compute_snr(segment, noise_floor)

            if snr_db < MIN_SNR_DB:
                pos += step_samples
                continue

            # Run emotion detection on segment
            seg_dur = len(segment) / sr
            emo_result = self.emotion_detector._single_detection(segment, sr, seg_dur)

            # Normalize scores to standard 7 emotions
            scores = {}
            for label in ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]:
                scores[label] = round(emo_result.emotion_scores.get(label, 0.0), 4)

            # Temperature scaling (T=1.5) to reduce overconfident predictions / sad bias
            scores = self._temperature_scale(scores, temperature=1.5)

            # Normalize to sum=1
            total = sum(scores.values())
            if total > 0:
                scores = {k: round(v / total, 4) for k, v in scores.items()}

            # Compute VAD from weighted emotion probabilities
            vad = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
            for emo, prob in scores.items():
                emo_vad = _get_vad(emo)
                vad["valence"] += prob * emo_vad["valence"]
                vad["arousal"] += prob * emo_vad["arousal"]
                vad["dominance"] += prob * emo_vad["dominance"]

            vad = {k: round(v, 4) for k, v in vad.items()}

            timeline.append({
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
                **scores,
                "vad": vad,
            })

            pos += step_samples

        # Apply sticky-transition smoothing to reduce noisy emotion flipping
        if len(timeline) >= 3:
            timeline = self._smooth_timeline_sticky(timeline, penalty=0.15)

        return timeline

    @staticmethod
    def _temperature_scale(scores: Dict[str, float], temperature: float = 1.5) -> Dict[str, float]:
        """Apply temperature scaling to soften overconfident predictions."""
        labels = list(scores.keys())
        vals = np.array([max(scores[l], 1e-12) for l in labels])
        logits = np.log(vals) / temperature
        logits -= logits.max()
        exp_logits = np.exp(logits)
        scaled = exp_logits / exp_logits.sum()
        return {l: round(float(s), 4) for l, s in zip(labels, scaled)}

    @staticmethod
    def _smooth_timeline_sticky(
        timeline: List[Dict[str, Any]],
        penalty: float = 0.10,
    ) -> List[Dict[str, Any]]:
        """Sticky-transition smoothing: only switch emotion if new one clearly wins.
        
        When suppressing a transition, boost the current_emotion score so that
        downstream aggregates see the smoothed dominant emotion.
        """
        if len(timeline) < 2:
            return timeline
        emotions_7 = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]
        smoothed = [dict(t) for t in timeline]

        # Determine current run emotion from first segment
        first_scores = {e: smoothed[0].get(e, 0.0) for e in emotions_7}
        current_emotion = max(first_scores, key=first_scores.get)

        for i in range(1, len(smoothed)):
            seg = smoothed[i]
            seg_scores = {e: seg.get(e, 0.0) for e in emotions_7}
            new_emotion = max(seg_scores, key=seg_scores.get)

            if new_emotion != current_emotion:
                new_score = seg_scores[new_emotion]
                old_score = seg_scores.get(current_emotion, 0.0)
                # Only switch if new emotion exceeds old + penalty
                if new_score < old_score + penalty:
                    # Suppress transition: boost current_emotion to be dominant
                    boost = new_score + 0.01  # just above the new winner
                    seg[current_emotion] = boost
                    # Re-normalize to sum=1
                    total = sum(seg.get(e, 0.0) for e in emotions_7)
                    if total > 0:
                        for e in emotions_7:
                            seg[e] = round(seg.get(e, 0.0) / total, 4)
                    # Recompute VAD for smoothed segment
                    vad = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
                    for e in emotions_7:
                        prob = seg.get(e, 0.0)
                        ev = EMOTION_VAD.get(e, EMOTION_VAD["neutral"])
                        vad["valence"] += prob * ev["valence"]
                        vad["arousal"] += prob * ev["arousal"]
                        vad["dominance"] += prob * ev["dominance"]
                    seg["vad"] = {k: round(v, 4) for k, v in vad.items()}
                else:
                    current_emotion = new_emotion
            # else: same emotion, keep running

        return smoothed

    # ------------------------------------------------------------------
    # 5. EMOTION AGGREGATES & DYNAMICS
    # ------------------------------------------------------------------

    def _compute_emotion_aggregates(
        self,
        timeline: List[Dict[str, Any]],
        prosody: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not timeline:
            return {"error": "no_timeline_data"}

        n = len(timeline)
        emotions_7 = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]

        # --- Per-emotion stats ---
        emotion_stats = {}
        for emo in emotions_7:
            vals = [seg.get(emo, 0.0) for seg in timeline]
            arr = np.array(vals)
            dominant_count = sum(1 for seg in timeline if max(
                (seg.get(e, 0.0) for e in emotions_7), default=0
            ) == seg.get(emo, 0.0) and seg.get(emo, 0.0) > 0)

            emotion_stats[emo] = {
                "mean": round(float(np.mean(arr)), 4),
                "max": round(float(np.max(arr)), 4),
                "volatility": round(float(np.std(arr)), 4),
                "dominant_segments": dominant_count,
            }

        # --- VAD stats ---
        valences = np.array([seg["vad"]["valence"] for seg in timeline])
        arousals = np.array([seg["vad"]["arousal"] for seg in timeline])
        dominances = np.array([seg["vad"]["dominance"] for seg in timeline])

        vad_stats = {
            "valence": {
                "mean": round(float(np.mean(valences)), 4),
                "max": round(float(np.max(valences)), 4),
                "min": round(float(np.min(valences)), 4),
                "std": round(float(np.std(valences)), 4),
            },
            "arousal": {
                "mean": round(float(np.mean(arousals)), 4),
                "max": round(float(np.max(arousals)), 4),
                "min": round(float(np.min(arousals)), 4),
                "std": round(float(np.std(arousals)), 4),
            },
            "dominance": {
                "mean": round(float(np.mean(dominances)), 4),
                "max": round(float(np.max(dominances)), 4),
                "min": round(float(np.min(dominances)), 4),
                "std": round(float(np.std(dominances)), 4),
            },
        }

        # --- Dynamics ---
        # Arousal volatility
        arousal_volatility = round(float(np.std(arousals)), 4)

        # Stress segments: arousal > 0.7 AND valence < 0.4
        stress_mask = (arousals > 0.7) & (valences < 0.4)
        stress_segments = int(np.sum(stress_mask))

        # Emotional shifts: count transitions where dominant emotion changes
        dominant_per_seg = []
        for seg in timeline:
            seg_scores = {e: seg.get(e, 0.0) for e in emotions_7}
            dominant_per_seg.append(max(seg_scores, key=seg_scores.get))

        emotional_shifts = sum(
            1 for i in range(1, len(dominant_per_seg))
            if dominant_per_seg[i] != dominant_per_seg[i - 1]
        )

        # Peak arousal time
        peak_idx = int(np.argmax(arousals))
        peak_arousal_time = round(timeline[peak_idx]["start_sec"], 2) if timeline else 0.0

        # Emotional arc type
        arc_type = self._classify_arc(arousals)

        # --- Derived scores ---
        pause_ratio = prosody.get("pause_time_ratio", 0.0)
        mean_dominance = float(np.mean(dominances))
        mean_arousal = float(np.mean(arousals))
        mean_valence = float(np.mean(valences))
        speech_rate = prosody.get("speech_rate", 0.0)
        # Normalize speech rate to [0,1]: 2 wps=0.3, 4 wps=0.7, 6+ wps=1.0
        rate_norm = min(max((speech_rate - 1.0) / 5.0, 0.0), 1.0)

        # Confidence: dominance + low pause ratio + speech fluency (NOT raw arousal)
        confidence_score = round(
            0.35 * mean_dominance + 0.35 * (1 - pause_ratio) + 0.30 * rate_norm,
            4,
        )
        # Stress: use |arousal| * (1 - valence)/2, clamped to [0, 1]
        abs_arousals = np.abs(arousals)
        neg_valences = (1.0 - valences) / 2.0  # map [-1,1] -> [1, 0]
        stress_raw = float(np.mean(abs_arousals * neg_valences)) if n > 0 else 0.0
        stress_index = round(min(max(stress_raw, 0.0), 1.0), 4)

        return {
            "emotion_stats": emotion_stats,
            "vad_stats": vad_stats,
            "dynamics": {
                "arousal_volatility": arousal_volatility,
                "stress_segments": stress_segments,
                "emotional_shifts": emotional_shifts,
                "peak_arousal_time": peak_arousal_time,
                "arc_type": arc_type,
            },
            "derived": {
                "confidence_score": confidence_score,
                "stress_index": stress_index,
            },
        }

    def _classify_arc(self, arousals: np.ndarray) -> str:
        """Classify the emotional arc from arousal trajectory."""
        n = len(arousals)
        if n < 3:
            return "flat"

        # Split into thirds
        third = n // 3
        first = float(np.mean(arousals[:third]))
        middle = float(np.mean(arousals[third:2 * third]))
        last = float(np.mean(arousals[2 * third:]))

        # Linear slope
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, arousals, 1)[0])

        if slope > 0.03:
            return "building_tension"
        elif slope < -0.03:
            return "releasing_tension"
        elif middle > first + 0.1 and middle > last + 0.1:
            return "peak_middle"
        elif first > middle + 0.1 and last > middle + 0.1:
            return "valley_middle"
        elif abs(slope) < 0.01 and float(np.std(arousals)) < 0.1:
            return "flat"
        else:
            return "fluctuating"

    # ------------------------------------------------------------------
    # 6. PARALINGUISTIC SUMMARY FOR LLM
    # ------------------------------------------------------------------

    def _generate_paralinguistic_summary(
        self,
        prosody: Dict[str, Any],
        voice_quality: Dict[str, float],
        spectral: Dict[str, float],
        emotion_agg: Dict[str, Any],
        duration: float,
        emotion_timeline: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a detailed text summary of voice profile for LLM consumption."""
        parts = []

        # --- Pitch profile ---
        f0 = prosody.get("f0_mean", 0)
        f0_std = prosody.get("f0_std", 0)
        if f0 > 0:
            pitch_level = "high" if f0 > 200 else "low" if f0 < 130 else "medium"
            variability = "high" if f0_std > 30 else "low" if f0_std < 15 else "moderate"
            trait_hint = "(extraverted)" if variability == "high" else "(introverted)" if variability == "low" else ""
            parts.append(f"{pitch_level} pitch with {variability} variability {trait_hint}".strip())

        # --- Speech rate ---
        rate = prosody.get("speech_rate", 0)
        pause_ratio = prosody.get("pause_time_ratio", 0)
        if rate > 0:
            speed = "fast" if rate > 5.0 else "slow" if rate < 3.0 else "moderate"
            pauses = "few pauses" if pause_ratio < 0.1 else "many pauses" if pause_ratio > 0.25 else "moderate pauses"
            conf_hint = "(confident)" if speed in ["fast", "moderate"] and pause_ratio < 0.15 else "(hesitant)" if pause_ratio > 0.25 else ""
            parts.append(f"speaks {speed} with {pauses} {conf_hint}".strip())

        # --- Stress peaks ---
        dynamics = emotion_agg.get("dynamics", {})
        stress_segs = dynamics.get("stress_segments", 0)
        peak_time = dynamics.get("peak_arousal_time", 0)
        if stress_segs > 0:
            parts.append(f"{stress_segs} stress peak(s) at ~{peak_time:.0f}s")

        # --- Emotional arc ---
        arc = dynamics.get("arc_type", "flat")
        arc_desc = {
            "building_tension": "arousal builds over time",
            "releasing_tension": "starts tense, relaxes",
            "peak_middle": "peaks mid-recording",
            "valley_middle": "dips mid-recording",
            "flat": "stable emotional tone",
            "fluctuating": "fluctuating emotional state",
        }
        parts.append(arc_desc.get(arc, arc))

        # --- Dominant / ending emotion (#8: use last 20% of timeline) ---
        emo_stats = emotion_agg.get("emotion_stats", {})
        emotions_7 = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]
        if emo_stats:
            sorted_by_dom = sorted(emo_stats, key=lambda e: emo_stats[e].get("dominant_segments", 0), reverse=True)
            dominant = sorted_by_dom[0]
            others = [e for e in sorted_by_dom if e != dominant]
            second = others[0] if others else dominant

            # Ending emotion from last 20% of actual timeline
            ending_emo = dominant
            if emotion_timeline and len(emotion_timeline) >= 5:
                tail = emotion_timeline[-max(len(emotion_timeline) // 5, 1):]
                tail_counts: Dict[str, int] = {}
                for seg in tail:
                    seg_scores = {e: seg.get(e, 0.0) for e in emotions_7}
                    winner = max(seg_scores, key=seg_scores.get)
                    tail_counts[winner] = tail_counts.get(winner, 0) + 1
                ending_emo = max(tail_counts, key=tail_counts.get) if tail_counts else dominant

            if arc == "fluctuating":
                parts.append(f"fluctuates between {dominant}/{second}, ending {ending_emo}")
            else:
                parts.append(f"stabilizes {dominant}, ending {ending_emo}")

        # --- Voice quality ---
        hnr = voice_quality.get("HNR", 0)
        jitter = voice_quality.get("jitter", 0)
        if hnr > 0:
            quality = "clear" if hnr > 15 else "breathy/rough"
            parts.append(f"{quality} voice (HNR={hnr:.1f}dB)")

        # --- Acoustic confidence ---
        derived = emotion_agg.get("derived", {})
        conf = derived.get("confidence_score", 0)
        conf_label = "high" if conf > 0.6 else "low" if conf < 0.35 else "moderate"
        parts.append(f"Overall acoustic confidence: {conf_label}")

        return "Voice profile: " + ", ".join(parts) + "."

    # ------------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------------

    def unload(self):
        """Free GPU memory."""
        if self._emotion_detector is not None:
            self._emotion_detector.unload_model()
            self._emotion_detector = None
        gc.collect()
