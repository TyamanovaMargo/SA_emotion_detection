"""Emotion detection from audio using emotion2vec."""

from typing import Optional, Dict, List, Any
import numpy as np
import torch

from ..config import EmotionConfig
from ..models.schemas import EmotionResult


EMOTION_LABELS = [
    "angry", "disgusted", "fearful", "happy", 
    "neutral", "sad", "surprised", "other"
]


class EmotionDetector:
    """Detect emotions from audio using emotion2vec or fallback methods."""
    
    def __init__(self, config: Optional[EmotionConfig] = None):
        self.config = config or EmotionConfig()
        self._model = None
        self._use_fallback = False
    
    def _load_model(self):
        """Load the emotion detection model."""
        if self._model is not None:
            return
        
        try:
            from funasr import AutoModel
            self._model = AutoModel(
                model=self.config.model_name,
                device=self.config.device
            )
        except Exception as e:
            print(f"Warning: Could not load emotion2vec model: {e}")
            print("Using fallback acoustic-based emotion detection.")
            self._use_fallback = True
    
    def detect(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        duration: float
    ) -> EmotionResult:
        """
        Detect emotions from audio.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            duration: Audio duration
            
        Returns:
            EmotionResult with emotion scores
        """
        self._load_model()
        
        if self._use_fallback:
            return self._fallback_detection(audio, sample_rate, duration)
        
        return self._model_detection(audio, sample_rate, duration)
    
    def _model_detection(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        duration: float
    ) -> EmotionResult:
        """Detect emotions using the loaded model."""
        try:
            result = self._model.generate(
                input=audio,
                granularity="utterance",
                extract_embedding=False
            )
            
            if result and len(result) > 0:
                scores = result[0].get("scores", [])
                labels = result[0].get("labels", EMOTION_LABELS)
                
                emotion_scores = {}
                for label, score in zip(labels, scores):
                    emotion_scores[label] = float(score)
                
                primary_idx = np.argmax(scores)
                primary_emotion = labels[primary_idx]
                confidence = float(scores[primary_idx])
                
                return EmotionResult(
                    primary_emotion=primary_emotion,
                    confidence=round(confidence, 3),
                    emotion_scores=emotion_scores,
                    emotion_timeline=None
                )
        except Exception as e:
            print(f"Model detection failed: {e}, using fallback")
        
        return self._fallback_detection(audio, sample_rate, duration)
    
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
        segment_duration: float = 3.0
    ) -> List[Dict[str, Any]]:
        """
        Detect emotions over time by segmenting the audio.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            segment_duration: Duration of each segment in seconds
            
        Returns:
            List of emotion results with timestamps
        """
        timeline = []
        segment_samples = int(segment_duration * sample_rate)
        total_samples = len(audio)
        
        for start_sample in range(0, total_samples, segment_samples):
            end_sample = min(start_sample + segment_samples, total_samples)
            segment = audio[start_sample:end_sample]
            
            if len(segment) < sample_rate * 0.5:
                continue
            
            result = self.detect(segment, sample_rate, len(segment) / sample_rate)
            
            timeline.append({
                "start_time": start_sample / sample_rate,
                "end_time": end_sample / sample_rate,
                "emotion": result.primary_emotion,
                "confidence": result.confidence,
                "scores": result.emotion_scores
            })
        
        return timeline
