"""Prosody feature extraction from audio."""

from typing import Optional, List, Tuple
import numpy as np
import librosa

from ..config import ProsodyConfig
from ..models.schemas import ProsodyFeatures


class ProsodyExtractor:
    """Extract prosodic features from audio."""
    
    def __init__(self, config: Optional[ProsodyConfig] = None):
        self.config = config or ProsodyConfig()
    
    def extract(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        word_count: int,
        duration: float
    ) -> ProsodyFeatures:
        """
        Extract prosodic features from audio.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            word_count: Number of words in transcript
            duration: Audio duration in seconds
            
        Returns:
            ProsodyFeatures object
        """
        pitch_mean, pitch_var, pitch_range = self._extract_pitch(audio, sample_rate)
        
        energy_mean, energy_level = self._extract_energy(audio, sample_rate)
        
        pauses, pause_duration_mean = self._detect_pauses(audio, sample_rate)
        pauses_per_minute = (len(pauses) / duration) * 60 if duration > 0 else 0
        
        speaking_rate = (word_count / duration) * 60 if duration > 0 else 0
        
        syllable_count = self._estimate_syllables(audio, sample_rate)
        speech_time = duration - (len(pauses) * pause_duration_mean) if pauses else duration
        articulation_rate = syllable_count / speech_time if speech_time > 0 else 0
        
        return ProsodyFeatures(
            speaking_rate_wpm=round(speaking_rate, 1),
            pitch_mean_hz=round(pitch_mean, 1),
            pitch_variance=round(pitch_var, 2),
            pitch_range=round(pitch_range, 1),
            energy_level=energy_level,
            energy_mean=round(energy_mean, 4),
            pauses_per_minute=round(pauses_per_minute, 1),
            pause_duration_mean=round(pause_duration_mean, 3),
            articulation_rate=round(articulation_rate, 2)
        )
    
    def _extract_pitch(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> Tuple[float, float, float]:
        """Extract pitch (F0) features using librosa."""
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=self.config.min_pitch,
            fmax=self.config.max_pitch,
            sr=sample_rate,
            frame_length=int(self.config.frame_length * sample_rate),
            hop_length=int(self.config.hop_length * sample_rate)
        )
        
        f0_voiced = f0[~np.isnan(f0)]
        
        if len(f0_voiced) == 0:
            return 0.0, 0.0, 0.0
        
        pitch_mean = np.mean(f0_voiced)
        pitch_var = np.var(f0_voiced)
        pitch_range = np.max(f0_voiced) - np.min(f0_voiced)
        
        return pitch_mean, pitch_var, pitch_range
    
    def _extract_energy(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> Tuple[float, str]:
        """Extract energy/RMS features."""
        hop_length = int(self.config.hop_length * sample_rate)
        frame_length = int(self.config.frame_length * sample_rate)
        
        rms = librosa.feature.rms(
            y=audio, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        energy_mean = np.mean(rms)
        
        if energy_mean < 0.02:
            energy_level = "low"
        elif energy_mean < 0.08:
            energy_level = "medium"
        else:
            energy_level = "high"
        
        return energy_mean, energy_level
    
    def _detect_pauses(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        min_pause_duration: float = 0.3,
        silence_threshold: float = 0.01
    ) -> Tuple[List[Tuple[float, float]], float]:
        """Detect pauses in speech."""
        hop_length = int(self.config.hop_length * sample_rate)
        frame_length = int(self.config.frame_length * sample_rate)
        
        rms = librosa.feature.rms(
            y=audio, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        is_silence = rms < silence_threshold
        
        pauses = []
        pause_start = None
        
        for i, silent in enumerate(is_silence):
            time = i * self.config.hop_length
            
            if silent and pause_start is None:
                pause_start = time
            elif not silent and pause_start is not None:
                pause_duration = time - pause_start
                if pause_duration >= min_pause_duration:
                    pauses.append((pause_start, time))
                pause_start = None
        
        if pause_start is not None:
            final_time = len(is_silence) * self.config.hop_length
            if final_time - pause_start >= min_pause_duration:
                pauses.append((pause_start, final_time))
        
        if pauses:
            pause_durations = [end - start for start, end in pauses]
            pause_duration_mean = np.mean(pause_durations)
        else:
            pause_duration_mean = 0.0
        
        return pauses, pause_duration_mean
    
    def _estimate_syllables(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> int:
        """Estimate syllable count from audio using onset detection."""
        onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, 
            sr=sample_rate,
            backtrack=False
        )
        return len(onsets)
