"""Emotion detection from audio using MERaLiON-SER or emotion2vec."""

from typing import Optional, Dict, List, Any
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

# Mapping from emotion2vec Chinese/bilingual labels to English
LABEL_NORMALIZE = {
    "生气/angry": "angry",
    "厌恶/disgusted": "disgusted",
    "恐惧/fearful": "fearful",
    "开心/happy": "happy",
    "中立/neutral": "neutral",
    "难过/sad": "sad",
    "吃惊/surprised": "surprised",
    "其他/other": "other",
    "<unk>": "other",
}

# MERaLiON label normalization (model outputs these labels)
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

# Valence/arousal mapping from emotion labels (for LLM trait inference)
EMOTION_VALENCE_AROUSAL = {
    "happy":     {"valence":  0.8, "arousal":  0.6},
    "surprised": {"valence":  0.3, "arousal":  0.8},
    "angry":     {"valence": -0.7, "arousal":  0.9},
    "fearful":   {"valence": -0.6, "arousal":  0.7},
    "disgusted": {"valence": -0.8, "arousal":  0.4},
    "sad":       {"valence": -0.7, "arousal": -0.4},
    "neutral":   {"valence":  0.0, "arousal":  0.0},
    "other":     {"valence":  0.0, "arousal":  0.1},
    "undetected":{"valence":  0.0, "arousal":  0.0},
}


def normalize_label(label: str) -> str:
    """Normalize emotion label to English."""
    low = label.lower().strip()
    if low in MERALION_LABEL_MAP:
        return MERALION_LABEL_MAP[low]
    if label in LABEL_NORMALIZE:
        return LABEL_NORMALIZE[label]
    # Try extracting English part after '/'
    if "/" in label:
        return label.split("/")[-1].lower()
    return low


def get_valence_arousal(emotion: str) -> Dict[str, float]:
    """Get valence/arousal values for an emotion label."""
    return EMOTION_VALENCE_AROUSAL.get(emotion, {"valence": 0.0, "arousal": 0.0})


class EmotionDetector:
    """Detect emotions from audio using MERaLiON-SER or emotion2vec."""
    
    def __init__(self, config: Optional[EmotionConfig] = None):
        self.config = config or EmotionConfig()
        self._model = None
        self._meralion_pipe = None
        self._use_fallback = False
        self._backend = self.config.backend  # "meralion" or "emotion2vec"
        
        # Resolve device: "auto" -> "cuda:N", "cuda" -> "cuda:N"
        if self.config.device == "auto":
            self.config.device = get_optimal_device(gpu_index=self.config.gpu_index)
        elif self.config.device == "cuda":
            self.config.device = f"cuda:{self.config.gpu_index}"
        
        # Auto-detect batch size if set to 0
        if self.config.batch_size == 0:
            self.config.batch_size = get_batch_size_for_device(self.config.device, base_batch_size=4)
        
        # Setup GPU memory if using GPU
        if self.config.device.startswith("cuda") or self.config.device == "mps":
            setup_gpu_memory(self.config.device, memory_fraction=0.9)
    
    def _load_model(self):
        """Load the emotion detection model (MERaLiON or emotion2vec)."""
        if self._backend == "meralion":
            self._load_meralion()
        else:
            self._load_emotion2vec()
    
    def _load_meralion(self):
        """Load MERaLiON-SER-v1 via HuggingFace transformers pipeline."""
        if self._meralion_pipe is not None:
            return
        
        try:
            from transformers import pipeline as hf_pipeline
            
            device = self.config.device
            model_id = self.config.meralion_model
            
            # Clear GPU cache before loading
            if device.startswith("cuda") and torch.cuda.is_available():
                clear_gpu_memory(device)
            
            print(f"Loading MERaLiON-SER on device: {device}")
            
            torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
            
            self._meralion_pipe = hf_pipeline(
                "audio-classification",
                model=model_id,
                top_k=None,
                device=device,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            
            print(f"MERaLiON-SER loaded successfully on {device}")
            
        except Exception as e:
            print(f"Warning: Could not load MERaLiON-SER: {e}")
            print("Falling back to emotion2vec...")
            self._backend = "emotion2vec"
            self._load_emotion2vec()
    
    def _load_emotion2vec(self):
        """Load emotion2vec via funasr."""
        if self._model is not None:
            return
        
        try:
            from funasr import AutoModel
            
            device = self.config.device
            
            if device.startswith("cuda") and torch.cuda.is_available():
                clear_gpu_memory(device)
            
            funasr_device = device
            if device.startswith("cuda:"):
                gpu_idx = int(device.split(":")[1])
                torch.cuda.set_device(gpu_idx)
                funasr_device = "cuda"
            
            print(f"Loading emotion2vec on device: {device}")
            self._model = AutoModel(
                model=self.config.model_name,
                device=funasr_device,
                disable_update=True
            )
            
            if device.startswith("cuda") and hasattr(self._model, "model"):
                gpu_idx = int(device.split(":")[1]) if ":" in device else 0
                self._model.model = self._model.model.to(f"cuda:{gpu_idx}")
            
            print(f"Emotion2vec loaded successfully on {device}")
            
        except Exception as e:
            print(f"Warning: Could not load emotion2vec model: {e}")
            print("Using fallback acoustic-based emotion detection.")
            self._use_fallback = True
    
    def unload_model(self):
        """Unload emotion model and free GPU memory."""
        device = self.config.device
        if self._meralion_pipe is not None:
            del self._meralion_pipe
            self._meralion_pipe = None
        if self._model is not None:
            del self._model
            self._model = None
        if device.startswith("cuda"):
            clear_gpu_memory(device)
        print(f"Emotion model unloaded, GPU memory freed on {device}")
    
    def detect(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        duration: float
    ) -> EmotionResult:
        """
        Detect emotions from audio.
        
        For long audio (>10s), processes in chunks to avoid GPU OOM,
        then aggregates results.
        """
        self._load_model()
        
        if self._use_fallback:
            return self._fallback_detection(audio, sample_rate, duration)
        
        if duration <= 10.0:
            return self._single_detection(audio, sample_rate, duration)
        
        return self._chunked_detection(audio, sample_rate, duration)
    
    def _single_detection(
        self,
        audio: np.ndarray,
        sample_rate: int,
        duration: float
    ) -> EmotionResult:
        """Detect emotion on a single segment."""
        if self._backend == "meralion":
            return self._meralion_detection(audio, sample_rate, duration)
        else:
            return self._emotion2vec_detection(audio, sample_rate, duration)
    
    def _chunked_detection(
        self,
        audio: np.ndarray,
        sample_rate: int,
        duration: float,
        chunk_duration: float = 7.5
    ) -> EmotionResult:
        """Detect emotions by processing audio in chunks and aggregating."""
        chunk_samples = int(chunk_duration * sample_rate)
        hop_samples = int(chunk_samples * self.config.hop_ratio)
        total_samples = len(audio)
        
        aggregated_scores = {}
        chunk_count = 0
        
        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            segment = audio[start:end]
            
            if len(segment) < sample_rate * 0.5:
                break
            
            seg_duration = len(segment) / sample_rate
            result = self._single_detection(segment, sample_rate, seg_duration)
            
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
            confidence=round(confidence, 3),
            emotion_scores={k: round(v, 3) for k, v in aggregated_scores.items()},
            emotion_timeline=None
        )
    
    def _meralion_detection(
        self,
        audio: np.ndarray,
        sample_rate: int,
        duration: float
    ) -> EmotionResult:
        """Detect emotions using MERaLiON-SER pipeline."""
        try:
            device = self.config.device
            if device.startswith("cuda:"):
                gpu_idx = int(device.split(":")[1])
                torch.cuda.set_device(gpu_idx)
            
            # MERaLiON expects 16kHz audio
            if sample_rate != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            # Run inference
            results = self._meralion_pipe(
                {"raw": audio.astype(np.float32), "sampling_rate": sample_rate}
            )
            
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Parse results: list of {"label": ..., "score": ...}
            emotion_scores = {}
            for item in results:
                label = normalize_label(item["label"])
                emotion_scores[label] = emotion_scores.get(label, 0.0) + float(item["score"])
            
            # Ensure all standard labels exist
            for label in EMOTION_LABELS:
                if label not in emotion_scores:
                    emotion_scores[label] = 0.0
            
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[primary_emotion]
            
            # Apply reliability gates
            primary_emotion, confidence = self._apply_reliability_gates(
                primary_emotion, confidence, emotion_scores
            )
            
            return EmotionResult(
                primary_emotion=primary_emotion,
                confidence=round(confidence, 3),
                emotion_scores={k: round(v, 4) for k, v in emotion_scores.items()},
                emotion_timeline=None
            )
            
        except Exception as e:
            print(f"MERaLiON detection failed: {e}, using fallback")
            if self.config.device.startswith("cuda") and torch.cuda.is_available():
                clear_gpu_memory(self.config.device)
            return self._fallback_detection(audio, sample_rate, duration)
    
    def _emotion2vec_detection(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        duration: float
    ) -> EmotionResult:
        """Detect emotions using emotion2vec model."""
        try:
            device = self.config.device
            if device.startswith("cuda:"):
                gpu_idx = int(device.split(":")[1])
                torch.cuda.set_device(gpu_idx)
            
            result = self._model.generate(
                input=audio,
                granularity="utterance",
                extract_embedding=False,
                batch_size=self.config.batch_size
            )
            
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if result and len(result) > 0:
                scores = result[0].get("scores", [])
                labels = result[0].get("labels", EMOTION_LABELS)
                
                emotion_scores = {}
                for label, score in zip(labels, scores):
                    eng_label = normalize_label(label)
                    if eng_label in emotion_scores:
                        emotion_scores[eng_label] += float(score)
                    else:
                        emotion_scores[eng_label] = float(score)
                
                primary_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = emotion_scores[primary_emotion]
                
                primary_emotion, confidence = self._apply_reliability_gates(
                    primary_emotion, confidence, emotion_scores
                )
                
                return EmotionResult(
                    primary_emotion=primary_emotion,
                    confidence=round(confidence, 3),
                    emotion_scores=emotion_scores,
                    emotion_timeline=None
                )
        except Exception as e:
            print(f"Emotion2vec detection failed: {e}, using fallback")
            if self.config.device.startswith("cuda") and torch.cuda.is_available():
                clear_gpu_memory(self.config.device)
        
        return self._fallback_detection(audio, sample_rate, duration)
    
    def _apply_reliability_gates(
        self,
        primary_emotion: str,
        confidence: float,
        emotion_scores: Dict[str, float]
    ) -> tuple:
        """Apply reliability gates to filter fake detections."""
        neutral_score = emotion_scores.get("neutral", 0.0)
        other_scores = [v for k, v in emotion_scores.items() if k != "neutral"]
        
        # Gate 1: Fake-confident neutral (true silence/noise only)
        if neutral_score > 0.98 and all(s < 0.002 for s in other_scores):
            return "undetected", 0.0
        
        # Gate 2: Low separation — neutral barely beats runner-up
        if primary_emotion == "neutral":
            sorted_scores = sorted(emotion_scores.values(), reverse=True)
            if len(sorted_scores) >= 2:
                separation = sorted_scores[0] - sorted_scores[1]
                if separation < 0.02:
                    return "undetected", 0.0
        
        return primary_emotion, confidence
    
    def _fallback_detection(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        duration: float
    ) -> EmotionResult:
        """
        Fallback emotion detection using acoustic features.
        This is a simplified heuristic-based approach.
        """
        import librosa
        
        rms = librosa.feature.rms(y=audio)[0]
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=75, fmax=500, sr=sample_rate
        )
        f0_voiced = f0[~np.isnan(f0)]
        
        if len(f0_voiced) > 0:
            pitch_mean = np.mean(f0_voiced)
            pitch_std = np.std(f0_voiced)
        else:
            pitch_mean = 150
            pitch_std = 20
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
        
        emotion_scores = self._heuristic_emotion_scores(
            energy_mean, energy_std, pitch_mean, pitch_std, spectral_centroid
        )
        
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[primary_emotion]
        
        return EmotionResult(
            primary_emotion=primary_emotion,
            confidence=round(confidence, 3),
            emotion_scores={k: round(v, 3) for k, v in emotion_scores.items()},
            emotion_timeline=None
        )
    
    def _heuristic_emotion_scores(
        self,
        energy_mean: float,
        energy_std: float,
        pitch_mean: float,
        pitch_std: float,
        spectral_centroid: float
    ) -> Dict[str, float]:
        """
        Compute heuristic emotion scores based on acoustic features.
        This is a simplified approach - real emotion detection requires trained models.
        """
        scores = {emotion: 0.1 for emotion in EMOTION_LABELS}
        
        high_energy = energy_mean > 0.05
        high_pitch = pitch_mean > 200
        high_variation = pitch_std > 40
        bright_timbre = spectral_centroid > 2000
        
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
        
        return scores
    
    def detect_timeline(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        segment_duration: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect emotions over time using overlapped windows.
        
        Uses config-driven segment_duration (default 7.5s) with 50% overlap
        for more stable emotion detection across the recording.
        
        Returns:
            List of emotion results with timestamps, valence, and arousal
        """
        if segment_duration is None:
            segment_duration = self.config.segment_duration
        
        hop_ratio = self.config.hop_ratio
        silence_factor = self.config.silence_threshold_factor
        min_snr = self.config.min_snr_db
        
        timeline = []
        segment_samples = int(segment_duration * sample_rate)
        hop_samples = int(segment_samples * hop_ratio)
        total_samples = len(audio)
        
        # Estimate noise floor from 10th percentile of short-window RMS
        window_size = int(0.025 * sample_rate)  # 25ms windows
        hop = window_size // 2
        rms_values = []
        for i in range(0, total_samples - window_size, hop):
            w = audio[i:i + window_size]
            rms_values.append(float(np.sqrt(np.mean(w ** 2))))
        
        if rms_values:
            noise_floor = float(np.percentile(rms_values, 10))
        else:
            noise_floor = 0.005
        
        noise_floor = max(noise_floor, 1e-6)
        
        # Adaptive silence threshold
        silence_threshold = max(noise_floor * silence_factor, 0.001)
        
        start_sample = 0
        while start_sample < total_samples:
            end_sample = min(start_sample + segment_samples, total_samples)
            segment = audio[start_sample:end_sample]
            
            if len(segment) < sample_rate * 0.5:
                break
            
            rms_energy = float(np.sqrt(np.mean(segment ** 2)))
            
            # Skip silence/noise: adaptive threshold per recording
            if rms_energy < silence_threshold:
                start_sample += hop_samples
                continue
            
            # Skip low-SNR segments
            snr_db = 20 * np.log10(rms_energy / noise_floor) if noise_floor > 0 else 0
            if snr_db < min_snr:
                start_sample += hop_samples
                continue
            
            # Detect emotion on this segment
            seg_duration = len(segment) / sample_rate
            result = self._single_detection(segment, sample_rate, seg_duration)
            
            normalized_scores = {normalize_label(k): v for k, v in result.emotion_scores.items()}
            emotion_label = normalize_label(result.primary_emotion)
            va = get_valence_arousal(emotion_label)
            
            timeline.append({
                "start_time": round(start_sample / sample_rate, 2),
                "end_time": round(end_sample / sample_rate, 2),
                "emotion": emotion_label,
                "confidence": result.confidence,
                "valence": va["valence"],
                "arousal": va["arousal"],
                "rms_energy": round(rms_energy, 4),
                "snr_db": round(snr_db, 1),
                "scores": normalized_scores
            })
            
            start_sample += hop_samples
        
        # Median filter: smooth out one-off emotion spikes (kernel=3)
        if len(timeline) >= 3:
            timeline = self._median_filter_timeline(timeline)
        
        return timeline
    
    def _median_filter_timeline(self, timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply median filter to emotion timeline to suppress one-off spikes.
        For each segment, if its neighbors both have a different emotion,
        replace it with the neighbor emotion (majority vote of window=3).
        """
        if len(timeline) < 3:
            return timeline
        
        smoothed = list(timeline)  # shallow copy
        for i in range(1, len(timeline) - 1):
            prev_emo = timeline[i - 1]["emotion"]
            curr_emo = timeline[i]["emotion"]
            next_emo = timeline[i + 1]["emotion"]
            
            # If current is an outlier (both neighbors agree on something else)
            if prev_emo == next_emo and curr_emo != prev_emo and curr_emo != "undetected":
                smoothed[i] = dict(timeline[i])
                smoothed[i]["emotion"] = prev_emo
                # Use average confidence of neighbors
                avg_conf = (timeline[i - 1]["confidence"] + timeline[i + 1]["confidence"]) / 2
                smoothed[i]["confidence"] = round(avg_conf, 3)
                # Update valence/arousal to match new emotion
                va = get_valence_arousal(prev_emo)
                smoothed[i]["valence"] = va["valence"]
                smoothed[i]["arousal"] = va["arousal"]
        
        return smoothed
