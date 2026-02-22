"""Emotion detection from audio using MERaLiON-SER-v1 only.

Single-model emotion detector using MERaLiON/MERaLiON-SER-v1 from HuggingFace.
Requires HF_TOKEN environment variable for gated model access.

Model: https://huggingface.co/MERaLiON/MERaLiON-SER-v1
"""

from typing import Optional, Dict, List, Any
import os
import numpy as np
import torch
import gc

from ..config import EmotionConfig
from ..models.schemas import EmotionResult
from ..utils.device import get_optimal_device, get_batch_size_for_device, setup_gpu_memory, clear_gpu_memory


EMOTION_LABELS = [
    "angry", "disgusted", "fearful", "happy",
    "neutral", "sad", "surprised", "other"
]

# MERaLiON label normalization
MERALION_LABEL_MAP = {
    "angry": "angry",
    "disgusted": "disgusted",
    "fearful": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprised": "surprised",
    "others": "other",
    "other": "other",
}

# Valence / Arousal / Dominance mapping
EMOTION_VAD = {
    "happy":     {"valence":  0.80, "arousal":  0.60, "dominance": 0.65},
    "surprised": {"valence":  0.30, "arousal":  0.80, "dominance": 0.40},
    "angry":     {"valence": -0.70, "arousal":  0.90, "dominance": 0.80},
    "fearful":   {"valence": -0.60, "arousal":  0.70, "dominance": 0.20},
    "disgusted": {"valence": -0.80, "arousal":  0.40, "dominance": 0.50},
    "sad":       {"valence": -0.70, "arousal": -0.40, "dominance": 0.20},
    "neutral":   {"valence":  0.00, "arousal":  0.00, "dominance": 0.50},
    "other":     {"valence":  0.00, "arousal":  0.10, "dominance": 0.40},
    "undetected": {"valence": 0.0, "arousal": 0.0, "dominance": 0.3},
}


def normalize_label(label: str) -> str:
    """Normalize emotion label to standard English."""
    low = label.lower().strip()
    return MERALION_LABEL_MAP.get(low, low)


def get_valence_arousal(emotion: str) -> Dict[str, float]:
    """Get valence/arousal values for an emotion label."""
    vad = EMOTION_VAD.get(emotion, EMOTION_VAD["neutral"])
    return {"valence": vad["valence"], "arousal": vad["arousal"]}


def get_vad(emotion: str) -> Dict[str, float]:
    """Get full valence/arousal/dominance for an emotion label."""
    return EMOTION_VAD.get(emotion, EMOTION_VAD["neutral"])


class EmotionDetector:
    """Detect emotions from audio using MERaLiON-SER-v1."""

    def __init__(self, config: Optional[EmotionConfig] = None):
        self.config = config or EmotionConfig()
        self._model = None
        self._processor = None
        self._use_fallback = False

        # Resolve device
        if self.config.device == "auto":
            self.config.device = get_optimal_device(gpu_index=self.config.gpu_index)
        elif self.config.device == "cuda":
            self.config.device = f"cuda:{self.config.gpu_index}"

        # Auto-detect batch size
        if self.config.batch_size == 0:
            self.config.batch_size = get_batch_size_for_device(self.config.device, base_batch_size=4)

        # Setup GPU memory
        if self.config.device.startswith("cuda") or self.config.device == "mps":
            setup_gpu_memory(self.config.device, memory_fraction=0.9)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load MERaLiON-SER-v1 model."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

            device = self.config.device
            model_id = self.config.meralion_model

            hf_token = os.getenv("HF_TOKEN", None)
            if not hf_token:
                raise RuntimeError(
                    "HF_TOKEN environment variable not set. "
                    "MERaLiON-SER-v1 is a gated model — set HF_TOKEN in your .env file."
                )

            # Clear GPU cache before loading
            if device.startswith("cuda") and torch.cuda.is_available():
                clear_gpu_memory(device)

            print(f"Loading MERaLiON-SER-v1 on device: {device}")

            # Patch torch.logspace to force CPU during model init (avoids meta tensor bug)
            _orig_logspace = torch.logspace
            def _cpu_logspace(*args, **kwargs):
                kwargs.pop("device", None)
                return _orig_logspace(*args, **kwargs, device="cpu")
            torch.logspace = _cpu_logspace

            try:
                self._processor = AutoFeatureExtractor.from_pretrained(
                    model_id, trust_remote_code=True, token=hf_token,
                )
                self._model = AutoModelForAudioClassification.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    token=hf_token,
                    low_cpu_mem_usage=False,
                    device_map=None,
                )
            finally:
                torch.logspace = _orig_logspace

            self._model = self._model.to(device).eval()
            print(f"MERaLiON-SER-v1 loaded successfully on {device}")

        except Exception as e:
            print(f"Warning: Could not load MERaLiON-SER-v1: {e}")
            print("Using fallback acoustic-based emotion detection.")
            self._use_fallback = True

    def unload_model(self):
        """Unload model and free GPU memory."""
        device = self.config.device
        if self._model is not None:
            del self._model
            self._model = None
            self._processor = None
        if device.startswith("cuda"):
            clear_gpu_memory(device)
        print(f"MERaLiON-SER model unloaded, GPU memory freed on {device}")

    # ------------------------------------------------------------------
    # Detection API
    # ------------------------------------------------------------------

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        duration: float,
    ) -> EmotionResult:
        """
        Detect emotions from audio.
        For long audio (>10s), processes in overlapping chunks and aggregates.
        """
        self._load_model()

        if self._use_fallback:
            return self._fallback_detection(audio, sample_rate, duration)

        if duration <= 10.0:
            return self._meralion_detection(audio, sample_rate, duration)

        return self._chunked_detection(audio, sample_rate, duration)

    def _meralion_detection(
        self,
        audio: np.ndarray,
        sample_rate: int,
        duration: float,
    ) -> EmotionResult:
        """Run MERaLiON-SER-v1 inference on a single segment."""
        try:
            device = self.config.device
            if device.startswith("cuda:"):
                gpu_idx = int(device.split(":")[1])
                torch.cuda.set_device(gpu_idx)

            # MERaLiON expects 16kHz
            if sample_rate != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000

            # Process with feature extractor
            inputs = self._processor(
                audio.astype(np.float32),
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                if isinstance(outputs, dict):
                    logits = outputs.get("logits", outputs.get("output"))
                else:
                    logits = outputs.logits
                if logits is None:
                    raise ValueError(f"Unexpected model output: {type(outputs)}")
                logits = logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Map id2label
            id2label = self._model.config.id2label
            emotion_scores = {}
            for idx, prob in enumerate(probs):
                raw_label = id2label.get(idx, id2label.get(str(idx), f"label_{idx}"))
                label = normalize_label(raw_label)
                emotion_scores[label] = emotion_scores.get(label, 0.0) + float(prob)

            # Ensure all standard labels exist
            for label in EMOTION_LABELS:
                if label not in emotion_scores:
                    emotion_scores[label] = 0.0

            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[primary_emotion]

            return EmotionResult(
                primary_emotion=primary_emotion,
                confidence=round(confidence, 4),
                emotion_scores={k: round(v, 4) for k, v in emotion_scores.items()},
                emotion_timeline=None,
            )

        except Exception as e:
            print(f"MERaLiON detection failed: {e}, using fallback")
            if self.config.device.startswith("cuda") and torch.cuda.is_available():
                clear_gpu_memory(self.config.device)
            return self._fallback_detection(audio, sample_rate, duration)

    def _chunked_detection(
        self,
        audio: np.ndarray,
        sample_rate: int,
        duration: float,
        chunk_duration: float = 7.5,
        hop_ratio: float = 0.5,
    ) -> EmotionResult:
        """Process long audio in overlapping chunks and aggregate."""
        chunk_samples = int(chunk_duration * sample_rate)
        hop_samples = int(chunk_samples * hop_ratio)
        total_samples = len(audio)

        aggregated_scores: Dict[str, float] = {}
        chunk_count = 0

        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            segment = audio[start:end]

            if len(segment) < sample_rate * 0.5:
                break

            seg_duration = len(segment) / sample_rate
            result = self._meralion_detection(segment, sample_rate, seg_duration)

            for label, score in result.emotion_scores.items():
                aggregated_scores[label] = aggregated_scores.get(label, 0.0) + score
            chunk_count += 1

            start += hop_samples

        if chunk_count == 0:
            return self._fallback_detection(audio, sample_rate, duration)

        aggregated_scores = {k: v / chunk_count for k, v in aggregated_scores.items()}
        primary_emotion = max(aggregated_scores, key=aggregated_scores.get)
        confidence = aggregated_scores[primary_emotion]

        return EmotionResult(
            primary_emotion=primary_emotion,
            confidence=round(confidence, 4),
            emotion_scores={k: round(v, 4) for k, v in aggregated_scores.items()},
            emotion_timeline=None,
        )

    # ------------------------------------------------------------------
    # Timeline detection (for VoiceAnalyzer)
    # ------------------------------------------------------------------

    def detect_timeline(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segment_duration: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect emotions over time using overlapped windows.
        Returns list of dicts with timestamps, emotion, confidence, valence, arousal.
        """
        if segment_duration is None:
            segment_duration = self.config.segment_duration

        hop_ratio = self.config.hop_ratio
        timeline = []
        segment_samples = int(segment_duration * sample_rate)
        hop_samples = int(segment_samples * hop_ratio)
        total_samples = len(audio)

        self._load_model()

        start_sample = 0
        while start_sample < total_samples:
            end_sample = min(start_sample + segment_samples, total_samples)
            segment = audio[start_sample:end_sample]

            if len(segment) < sample_rate * 0.5:
                break

            seg_duration = len(segment) / sample_rate
            result = self._meralion_detection(segment, sample_rate, seg_duration) \
                if not self._use_fallback else self._fallback_detection(segment, sample_rate, seg_duration)

            label = normalize_label(result.primary_emotion)
            va = get_valence_arousal(label)
            rms_energy = float(np.sqrt(np.mean(segment ** 2)))

            timeline.append({
                "start_time": round(start_sample / sample_rate, 2),
                "end_time": round(end_sample / sample_rate, 2),
                "emotion": label,
                "confidence": result.confidence,
                "valence": va["valence"],
                "arousal": va["arousal"],
                "rms_energy": round(rms_energy, 4),
                "scores": {normalize_label(k): v for k, v in result.emotion_scores.items()},
            })

            start_sample += hop_samples

        # Median filter smoothing
        if len(timeline) >= 3:
            timeline = self._median_filter_timeline(timeline)

        return timeline

    def _single_detection(
        self,
        audio: np.ndarray,
        sample_rate: int,
        duration: float,
    ) -> EmotionResult:
        """Single segment detection — used by VoiceAnalyzer."""
        self._load_model()
        if self._use_fallback:
            return self._fallback_detection(audio, sample_rate, duration)
        return self._meralion_detection(audio, sample_rate, duration)

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def _median_filter_timeline(self, timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suppress one-off emotion spikes via majority vote (window=3)."""
        if len(timeline) < 3:
            return timeline

        smoothed = list(timeline)
        for i in range(1, len(timeline) - 1):
            prev_emo = timeline[i - 1]["emotion"]
            curr_emo = timeline[i]["emotion"]
            next_emo = timeline[i + 1]["emotion"]

            if prev_emo == next_emo and curr_emo != prev_emo:
                smoothed[i] = dict(timeline[i])
                smoothed[i]["emotion"] = prev_emo
                avg_conf = (timeline[i - 1]["confidence"] + timeline[i + 1]["confidence"]) / 2
                smoothed[i]["confidence"] = round(avg_conf, 3)
                va = get_valence_arousal(prev_emo)
                smoothed[i]["valence"] = va["valence"]
                smoothed[i]["arousal"] = va["arousal"]

        return smoothed

    # ------------------------------------------------------------------
    # Fallback (acoustic heuristic)
    # ------------------------------------------------------------------

    def _fallback_detection(
        self,
        audio: np.ndarray,
        sample_rate: int,
        duration: float,
    ) -> EmotionResult:
        """Fallback emotion detection using acoustic features."""
        import librosa

        rms = librosa.feature.rms(y=audio)[0]
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)

        f0, voiced_flag, _ = librosa.pyin(audio, fmin=75, fmax=500, sr=sample_rate)
        f0_voiced = f0[~np.isnan(f0)]

        if len(f0_voiced) > 0:
            pitch_mean = np.mean(f0_voiced)
            pitch_std = np.std(f0_voiced)
        else:
            pitch_mean = 150
            pitch_std = 20

        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))

        scores = {emotion: 0.1 for emotion in EMOTION_LABELS}

        high_energy = energy_mean > 0.05
        high_pitch = pitch_mean > 200
        high_variation = pitch_std > 40

        if high_energy and high_pitch and high_variation:
            scores["happy"] = 0.6
            scores["surprised"] = 0.3
        elif high_energy and not high_pitch:
            scores["angry"] = 0.5
            scores["disgusted"] = 0.2
        elif not high_energy and not high_variation:
            scores["sad"] = 0.4
            scores["neutral"] = 0.4
        elif not high_energy and high_variation:
            scores["fearful"] = 0.4
            scores["surprised"] = 0.3
        else:
            scores["neutral"] = 0.6

        total = sum(scores.values())
        scores = {k: v / total for k, v in scores.items()}

        primary_emotion = max(scores, key=scores.get)
        confidence = scores[primary_emotion]

        return EmotionResult(
            primary_emotion=primary_emotion,
            confidence=round(confidence, 3),
            emotion_scores={k: round(v, 3) for k, v in scores.items()},
            emotion_timeline=None,
        )
