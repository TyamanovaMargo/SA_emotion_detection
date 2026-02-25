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
        self._model = None              # emotion2vec
        self._meralion_pipe = None       # kept for compat
        self._meralion_model = None      # MERaLiON model object
        self._meralion_processor = None  # MERaLiON feature extractor
        self._use_fallback = False
        self._backend = self.config.backend  # "meralion" or "emotion2vec"
        self._meralion_available = False  # True if MERaLiON loaded OK
        
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
        """Load MERaLiON-SER-v1 via direct model loading (avoids pipeline meta-tensor bug)."""
        if self._meralion_model is not None:
            return
        
        try:
            import os
            from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
            
            device = self.config.device
            model_id = self.config.meralion_model
            
            # HuggingFace token for gated model access
            hf_token = os.getenv("HF_TOKEN", None)
            if not hf_token:
                raise RuntimeError(
                    "HF_TOKEN environment variable not set. "
                    "MERaLiON-SER-v1 is a gated model — you need a HuggingFace token. "
                    "Set HF_TOKEN in your .env file or pass -e HF_TOKEN=... to Docker."
                )
            
            # Clear GPU cache before loading
            if device.startswith("cuda") and torch.cuda.is_available():
                clear_gpu_memory(device)
            
            print(f"Loading MERaLiON-SER on device: {device}")
            
            # Patch torch.logspace to force CPU during model init (avoids meta tensor bug)
            _orig_logspace = torch.logspace
            def _cpu_logspace(*args, **kwargs):
                kwargs.pop('device', None)
                return _orig_logspace(*args, **kwargs, device='cpu')
            torch.logspace = _cpu_logspace
            try:
                self._meralion_processor = AutoFeatureExtractor.from_pretrained(
                    model_id, trust_remote_code=True, token=hf_token,
                )
                self._meralion_model = AutoModelForAudioClassification.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    token=hf_token,
                    low_cpu_mem_usage=False,
                    device_map=None,
                )
            finally:
                torch.logspace = _orig_logspace
            self._meralion_model = self._meralion_model.to(device).eval()
            
            self._meralion_available = True
            print(f"MERaLiON-SER loaded successfully on {device}")
            
        except Exception as e:
            print(f"Warning: Could not load MERaLiON-SER: {e}")
            print("Falling back to emotion2vec...")
            self._meralion_available = False
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
    
    def _load_both_models(self):
        """Load both MERaLiON-SER and emotion2vec for dual comparison."""
        # Try MERaLiON first (don't let it fallback to emotion2vec)
        if self._meralion_model is None:
            try:
                import os
                from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
                device = self.config.device
                model_id = self.config.meralion_model
                hf_token = os.getenv("HF_TOKEN", None)
                if hf_token:
                    if device.startswith("cuda") and torch.cuda.is_available():
                        clear_gpu_memory(device)
                    print(f"Loading MERaLiON-SER on device: {device}")
                    _orig_logspace = torch.logspace
                    def _cpu_logspace(*args, **kwargs):
                        kwargs.pop('device', None)
                        return _orig_logspace(*args, **kwargs, device='cpu')
                    torch.logspace = _cpu_logspace
                    try:
                        self._meralion_processor = AutoFeatureExtractor.from_pretrained(
                            model_id, trust_remote_code=True, token=hf_token,
                        )
                        self._meralion_model = AutoModelForAudioClassification.from_pretrained(
                            model_id, trust_remote_code=True, token=hf_token,
                            low_cpu_mem_usage=False, device_map=None,
                        )
                    finally:
                        torch.logspace = _orig_logspace
                    self._meralion_model = self._meralion_model.to(device).eval()
                    self._meralion_available = True
                    print(f"MERaLiON-SER loaded successfully on {device}")
                else:
                    print("HF_TOKEN not set — skipping MERaLiON-SER")
            except Exception as e:
                print(f"Warning: Could not load MERaLiON-SER: {e}")
                self._meralion_available = False
        # Always load emotion2vec
        if self._model is None:
            try:
                from funasr import AutoModel as FunASRAutoModel
                device = self.config.device
                funasr_device = device
                if device.startswith("cuda:"):
                    gpu_idx = int(device.split(":")[1])
                    torch.cuda.set_device(gpu_idx)
                    funasr_device = "cuda"
                print(f"Loading emotion2vec on device: {device}")
                self._model = FunASRAutoModel(
                    model=self.config.model_name, device=funasr_device, disable_update=True
                )
                if device.startswith("cuda") and hasattr(self._model, "model"):
                    gpu_idx = int(device.split(":")[1]) if ":" in device else 0
                    self._model.model = self._model.model.to(f"cuda:{gpu_idx}")
                print(f"Emotion2vec loaded successfully on {device}")
            except Exception as e:
                print(f"Warning: Could not load emotion2vec: {e}")

    def unload_model(self):
        """Unload emotion model and free GPU memory."""
        device = self.config.device
        if self._meralion_pipe is not None:
            del self._meralion_pipe
            self._meralion_pipe = None
        if self._meralion_model is not None:
            del self._meralion_model
            self._meralion_model = None
            self._meralion_processor = None
            self._meralion_available = False
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
        if self._backend == "meralion" and self._meralion_available:
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
        """Detect emotions using MERaLiON-SER (direct model inference)."""
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
            
            # Process with feature extractor
            inputs = self._meralion_processor(
                audio.astype(np.float32),
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._meralion_model(**inputs)
                # Model may return dict or ModelOutput — handle both
                if isinstance(outputs, dict):
                    logits = outputs.get("logits", outputs.get("output"))
                else:
                    logits = outputs.logits
                if logits is None:
                    raise ValueError(f"Unexpected model output keys: {outputs.keys() if isinstance(outputs, dict) else type(outputs)}")
                logits = logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Map id2label (keys can be int or str depending on config)
            id2label = self._meralion_model.config.id2label
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
        duration: float,
        skip_gates: bool = False,
    ) -> EmotionResult:
        """Detect emotions using emotion2vec model.
        
        Args:
            skip_gates: If True, skip legacy reliability gates (used by fused
                        pipeline which has its own SNR/entropy/low_confidence signals).
        """
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
                
                if not skip_gates:
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
        
        # Gate 1: Only filter as undetected when ALL non-neutral scores are
        # essentially zero (< 1e-5) — i.e. pure silence/noise, not real speech.
        # emotion2vec legitimately outputs high-confidence neutral for calm speech.
        if neutral_score > 0.9999 and all(s < 1e-5 for s in other_scores):
            return "undetected", 0.0
        
        # Gate 2: Low separation — neutral barely beats runner-up
        if primary_emotion == "neutral":
            sorted_scores = sorted(emotion_scores.values(), reverse=True)
            if len(sorted_scores) >= 2:
                separation = sorted_scores[0] - sorted_scores[1]
                if separation < 0.005:
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
    
    def detect_emotion_timeline_rich(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segment_duration: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """
        Per-segment emotion tracking with acoustic features for dashboard.

        Each segment returns: time_start, time_end, emotion, confidence,
        valence, arousal, rms_energy, pitch_mean.

        This is separate from detect_timeline() — it uses fixed non-overlapped
        segments and includes per-segment pitch for richer visualisation.
        """
        import librosa as _lr

        self._load_model()

        total_samples = len(audio)
        seg_samples = int(segment_duration * sample_rate)
        timeline: List[Dict[str, Any]] = []

        pos = 0
        while pos < total_samples:
            end = min(pos + seg_samples, total_samples)
            seg = audio[pos:end]
            if len(seg) < sample_rate * 0.5:
                break

            seg_dur = len(seg) / sample_rate

            # emotion
            if not self._use_fallback and (self._meralion_model is not None or self._model is not None):
                emo = self._single_detection(seg, sample_rate, seg_dur)
            else:
                emo = self._fallback_detection(seg, sample_rate, seg_dur)

            label = normalize_label(emo.primary_emotion)
            va = get_valence_arousal(label)

            # energy
            rms_energy = float(np.sqrt(np.mean(seg ** 2)))

            # pitch mean
            try:
                f0, _, _ = _lr.pyin(seg, fmin=80, fmax=500, sr=sample_rate)
                f0v = f0[~np.isnan(f0)]
                pitch_mean = float(np.mean(f0v)) if len(f0v) > 0 else 0.0
            except Exception:
                pitch_mean = 0.0

            timeline.append({
                "time_start": round(pos / sample_rate, 2),
                "time_end": round(end / sample_rate, 2),
                "emotion": label,
                "confidence": emo.confidence,
                "valence": va["valence"],
                "arousal": va["arousal"],
                "rms_energy": round(rms_energy, 4),
                "pitch_mean": round(pitch_mean, 1),
            })

            pos = end

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

    # ------------------------------------------------------------------
    # Dual-model comparison methods (with fusion, VAD, SNR, smoothing)
    # ------------------------------------------------------------------

    def detect_dual(
        self,
        audio: np.ndarray,
        sample_rate: int,
        duration: float,
    ) -> Dict[str, Any]:
        """
        Run BOTH MERaLiON-SER and emotion2vec on the same audio.
        Returns per-model results + fused result with entropy/top2_gap.
        """
        from .emotion_fusion import (
            fuse_probabilities, compute_entropy, compute_top2_gap,
        )

        self._load_both_models()

        results: Dict[str, Any] = {}
        mer_probs = None
        e2v_probs = None

        # emotion2vec
        if self._model is not None:
            try:
                e2v = self._emotion2vec_detection(audio, sample_rate, duration, skip_gates=True)
                e2v_probs = e2v.emotion_scores
                results["emotion2vec"] = {
                    "primary_emotion": e2v.primary_emotion,
                    "confidence": e2v.confidence,
                    "scores": e2v.emotion_scores,
                    "entropy": compute_entropy(e2v.emotion_scores),
                    "top2_gap": compute_top2_gap(e2v.emotion_scores),
                }
            except Exception as e:
                results["emotion2vec"] = {"error": str(e)}
        else:
            results["emotion2vec"] = {"error": "model not loaded"}

        # MERaLiON-SER
        if self._meralion_available and self._meralion_model is not None:
            try:
                mer = self._meralion_detection(audio, sample_rate, duration)
                mer_probs = mer.emotion_scores
                results["meralion_ser"] = {
                    "primary_emotion": mer.primary_emotion,
                    "confidence": mer.confidence,
                    "scores": mer.emotion_scores,
                    "entropy": compute_entropy(mer.emotion_scores),
                    "top2_gap": compute_top2_gap(mer.emotion_scores),
                }
            except Exception as e:
                results["meralion_ser"] = {"error": str(e)}
        else:
            results["meralion_ser"] = {"error": "model not loaded (need HF_TOKEN + access)"}

        # Fused result
        fused_probs = fuse_probabilities(mer_probs, e2v_probs)
        if fused_probs:
            fused_emo = max(fused_probs, key=fused_probs.get)
            results["fused"] = {
                "primary_emotion": fused_emo,
                "confidence": round(fused_probs[fused_emo], 4),
                "scores": fused_probs,
                "entropy": compute_entropy(fused_probs),
                "top2_gap": compute_top2_gap(fused_probs),
            }

        # Agreement flag
        e2v_emo = results.get("emotion2vec", {}).get("primary_emotion")
        mer_emo = results.get("meralion_ser", {}).get("primary_emotion")
        results["models_agree"] = (e2v_emo == mer_emo) if (e2v_emo and mer_emo) else None

        return results

    def detect_emotion_timeline_dual(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segment_duration: float = 8.0,
        hop_duration: float = 4.0,
    ) -> List[Dict[str, Any]]:
        """
        Enhanced per-segment emotion timeline from BOTH models with:
        - Overlapping windows (segment_duration with hop_duration)
        - SNR filtering (skip low-SNR segments)
        - Probability fusion (temperature-scaled weighted average)
        - Entropy & top2_gap per segment
        - Sticky-transition smoothing
        - Per-segment reliability signals (low_confidence flag)

        Each segment has: time_start, time_end, rms_energy, snr_db, pitch_mean,
        per-model results, fused results, entropy, top2_gap, models_agree.
        """
        import librosa as _lr
        from .emotion_fusion import (
            fuse_probabilities, compute_entropy, compute_top2_gap,
            estimate_noise_floor, compute_snr,
            smooth_timeline_sticky, MIN_SNR_DB,
        )

        self._load_both_models()

        total_samples = len(audio)
        seg_samples = int(segment_duration * sample_rate)
        hop_samples = int(hop_duration * sample_rate)
        noise_floor = estimate_noise_floor(audio, sample_rate)
        timeline: List[Dict[str, Any]] = []

        pos = 0
        while pos < total_samples:
            end = min(pos + seg_samples, total_samples)
            seg = audio[pos:end]
            if len(seg) < sample_rate * 1.0:  # skip < 1s
                break

            seg_dur = len(seg) / sample_rate
            rms_energy = float(np.sqrt(np.mean(seg ** 2)))
            snr_db = compute_snr(seg, noise_floor)

            # pitch mean
            try:
                f0, _, _ = _lr.pyin(seg, fmin=80, fmax=500, sr=sample_rate)
                f0v = f0[~np.isnan(f0)]
                pitch_mean = float(np.mean(f0v)) if len(f0v) > 0 else 0.0
            except Exception:
                pitch_mean = 0.0

            entry: Dict[str, Any] = {
                "time_start": round(pos / sample_rate, 2),
                "time_end": round(end / sample_rate, 2),
                "rms_energy": round(rms_energy, 4),
                "snr_db": round(snr_db, 1),
                "pitch_mean": round(pitch_mean, 1),
            }

            # Skip very low-SNR segments — mark as low confidence
            if snr_db < MIN_SNR_DB:
                entry.update({
                    "emotion2vec_emotion": "low_snr", "emotion2vec_confidence": 0.0,
                    "emotion2vec_valence": 0.0, "emotion2vec_arousal": 0.0,
                    "meralion_emotion": "low_snr", "meralion_confidence": 0.0,
                    "meralion_valence": 0.0, "meralion_arousal": 0.0,
                    "fused_emotion": "low_snr", "fused_confidence": 0.0,
                    "fused_scores": {}, "fused_valence": 0.0, "fused_arousal": 0.0,
                    "entropy": 0.0, "top2_gap": 0.0,
                    "low_confidence": True, "models_agree": None,
                })
                timeline.append(entry)
                pos += hop_samples
                continue

            e2v_probs = None
            mer_probs = None

            # emotion2vec
            if self._model is not None:
                try:
                    e2v = self._emotion2vec_detection(seg, sample_rate, seg_dur, skip_gates=True)
                    label = normalize_label(e2v.primary_emotion)
                    va = get_valence_arousal(label)
                    e2v_probs = e2v.emotion_scores
                    entry["emotion2vec_emotion"] = label
                    entry["emotion2vec_confidence"] = e2v.confidence
                    entry["emotion2vec_valence"] = va["valence"]
                    entry["emotion2vec_arousal"] = va["arousal"]
                except Exception:
                    entry["emotion2vec_emotion"] = "error"
                    entry["emotion2vec_confidence"] = 0.0
                    entry["emotion2vec_valence"] = 0.0
                    entry["emotion2vec_arousal"] = 0.0
            else:
                entry["emotion2vec_emotion"] = "unavailable"
                entry["emotion2vec_confidence"] = 0.0
                entry["emotion2vec_valence"] = 0.0
                entry["emotion2vec_arousal"] = 0.0

            # MERaLiON-SER
            if self._meralion_available and self._meralion_model is not None:
                try:
                    mer = self._meralion_detection(seg, sample_rate, seg_dur)
                    label = normalize_label(mer.primary_emotion)
                    va = get_valence_arousal(label)
                    mer_probs = mer.emotion_scores
                    entry["meralion_emotion"] = label
                    entry["meralion_confidence"] = mer.confidence
                    entry["meralion_valence"] = va["valence"]
                    entry["meralion_arousal"] = va["arousal"]
                except Exception:
                    entry["meralion_emotion"] = "error"
                    entry["meralion_confidence"] = 0.0
                    entry["meralion_valence"] = 0.0
                    entry["meralion_arousal"] = 0.0
            else:
                entry["meralion_emotion"] = "unavailable"
                entry["meralion_confidence"] = 0.0
                entry["meralion_valence"] = 0.0
                entry["meralion_arousal"] = 0.0

            # Fused result
            fused_probs = fuse_probabilities(mer_probs, e2v_probs)
            if fused_probs:
                fused_emo = max(fused_probs, key=fused_probs.get)
                fused_conf = fused_probs[fused_emo]
                fused_va = get_valence_arousal(fused_emo)
                entry["fused_emotion"] = fused_emo
                entry["fused_confidence"] = round(fused_conf, 4)
                entry["fused_scores"] = fused_probs
                entry["fused_valence"] = fused_va["valence"]
                entry["fused_arousal"] = fused_va["arousal"]
            else:
                entry["fused_emotion"] = entry.get("meralion_emotion", entry.get("emotion2vec_emotion", "neutral"))
                entry["fused_confidence"] = 0.0
                entry["fused_scores"] = {}
                entry["fused_valence"] = 0.0
                entry["fused_arousal"] = 0.0

            entry["entropy"] = compute_entropy(fused_probs) if fused_probs else 0.0
            entry["top2_gap"] = compute_top2_gap(fused_probs) if fused_probs else 0.0
            entry["low_confidence"] = entry["fused_confidence"] < 0.4
            entry["models_agree"] = entry.get("emotion2vec_emotion") == entry.get("meralion_emotion")

            timeline.append(entry)
            pos += hop_samples

        # Apply sticky-transition smoothing
        if len(timeline) >= 2:
            timeline = smooth_timeline_sticky(timeline)

        return timeline
