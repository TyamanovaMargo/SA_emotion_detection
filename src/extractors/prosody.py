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
        pitch_mean, pitch_var, pitch_range, pitch_slope = self._extract_pitch(audio, sample_rate)
        
        energy_mean, energy_std, energy_range, energy_level = self._extract_energy(audio, sample_rate)
        
        pauses, pause_duration_mean, pause_duration_std, long_pauses_count = self._detect_pauses(audio, sample_rate)
        pauses_per_minute = (len(pauses) / duration) * 60 if duration > 0 else 0
        
        # Speaking rate from word count (if available)
        speaking_rate = (word_count / duration) * 60 if duration > 0 and word_count > 0 else 0
        
        # Fallback: estimate from syllables when word_count unavailable
        syllable_count = self._estimate_syllables(audio, sample_rate)
        total_pause_time = sum(end - start for start, end in pauses) if pauses else 0.0
        speech_time = duration - total_pause_time
        articulation_rate = syllable_count / speech_time if speech_time > 0 else 0
        
        # If no word count, estimate speaking rate from syllables
        # Typical: 1 word ≈ 1.5 syllables, so syllables/1.5 = approx words
        if speaking_rate == 0 and articulation_rate > 0:
            # Convert syllables/sec to words/min: (syl/sec) * 60 / 1.5
            speaking_rate = (articulation_rate * 60) / 1.5
        
        # Cap WPM to human-plausible range [0, 250]
        # Onset-based syllable estimation can wildly overestimate for some formats (.webm)
        if speaking_rate > 250:
            speaking_rate = min(speaking_rate, 250.0)
        
        speech_to_silence_ratio = speech_time / total_pause_time if total_pause_time > 0 else 99.0
        
        rhythm_regularity = self._compute_rhythm_regularity(audio, sample_rate)
        
        return ProsodyFeatures(
            speaking_rate_wpm=round(speaking_rate, 1),
            pitch_mean_hz=round(pitch_mean, 1),
            pitch_variance=round(pitch_var, 2),
            pitch_range=round(pitch_range, 1),
            pitch_slope=round(pitch_slope, 4),
            energy_level=energy_level,
            energy_mean=round(energy_mean, 4),
            energy_std=round(energy_std, 4),
            energy_range=round(energy_range, 4),
            pauses_per_minute=round(pauses_per_minute, 1),
            pause_duration_mean=round(pause_duration_mean, 3),
            pause_duration_std=round(pause_duration_std, 3),
            long_pauses_count=long_pauses_count,
            articulation_rate=round(articulation_rate, 2),
            speech_to_silence_ratio=round(speech_to_silence_ratio, 2),
            rhythm_regularity=round(rhythm_regularity, 3),
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
            return 0.0, 0.0, 0.0, 0.0
        
        pitch_mean = np.mean(f0_voiced)
        pitch_var = np.var(f0_voiced)
        pitch_range = np.max(f0_voiced) - np.min(f0_voiced)
        
        # Pitch slope: linear regression over time (positive = rising engagement)
        if len(f0_voiced) > 2:
            x = np.arange(len(f0_voiced))
            coeffs = np.polyfit(x, f0_voiced, 1)
            pitch_slope = float(coeffs[0])
        else:
            pitch_slope = 0.0
        
        return pitch_mean, pitch_var, pitch_range, pitch_slope
    
    def _extract_energy(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> Tuple[float, float, float, str]:
        """Extract energy/RMS features."""
        hop_length = int(self.config.hop_length * sample_rate)
        frame_length = int(self.config.frame_length * sample_rate)
        
        rms = librosa.feature.rms(
            y=audio, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        energy_range = float(np.max(rms) - np.min(rms))
        
        if energy_mean < 0.02:
            energy_level = "low"
        elif energy_mean < 0.08:
            energy_level = "medium"
        else:
            energy_level = "high"
        
        return energy_mean, energy_std, energy_range, energy_level
    
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
            pause_duration_std = float(np.std(pause_durations)) if len(pause_durations) > 1 else 0.0
            long_pauses_count = sum(1 for d in pause_durations if d > 1.0)
        else:
            pause_duration_mean = 0.0
            pause_duration_std = 0.0
            long_pauses_count = 0
        
        return pauses, pause_duration_mean, pause_duration_std, long_pauses_count
    
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
    
    def _compute_rhythm_regularity(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Compute rhythm regularity as CV of inter-onset intervals.
        0 = perfectly regular, >0.8 = very irregular."""
        onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sample_rate,
            backtrack=False,
        )
        if len(onsets) < 3:
            return 0.0
        hop = int(self.config.hop_length * sample_rate)
        onset_times = onsets * hop / sample_rate
        intervals = np.diff(onset_times)
        if np.mean(intervals) == 0:
            return 0.0
        return float(np.std(intervals) / np.mean(intervals))

    # ------------------------------------------------------------------
    # Granular feature extraction (extends base extract())
    # ------------------------------------------------------------------

    def extract_granular(
        self,
        audio: np.ndarray,
        sample_rate: int,
        word_count: int,
        duration: float,
    ) -> dict:
        """
        Extract granular voice features as a flat dict for LLM prompts and dashboard.

        Includes everything from the base extract() plus:
        - Pause frequency, total pause time, articulation/speech rate ratio
        - Pitch contour slope, high/low-pitch ratios
        - Energy variability, dynamic range dB
        - Voiced/unvoiced ratio
        - Jitter, Shimmer, HNR (via parselmouth)
        - Spectral centroid mean, spectral flux mean (via librosa)
        - Derived arousal and valence proxy scores
        """
        # --- reuse base features ---------------------------------------------------
        base = self.extract(audio, sample_rate, word_count, duration)
        feat: dict = {}

        # copy all base prosody fields
        for field_name in base.model_fields:
            feat[field_name] = getattr(base, field_name)

        # --- pause details ---------------------------------------------------------
        pauses, pause_dur_mean, pause_dur_std, long_pauses = self._detect_pauses(audio, sample_rate)
        pause_durations = [e - s for s, e in pauses] if pauses else []
        total_pause_time = sum(pause_durations) if pause_durations else 0.0
        speech_time = max(duration - total_pause_time, 0.01)

        feat["pause_count"] = len(pauses)
        feat["total_pause_time_s"] = round(total_pause_time, 3)
        feat["pause_frequency_per_min"] = round((len(pauses) / duration) * 60, 2) if duration > 0 else 0.0
        feat["articulation_to_speech_ratio"] = round(
            base.articulation_rate / (base.speaking_rate_wpm / 60) if base.speaking_rate_wpm > 0 else 0.0, 3
        )

        # --- pitch detail ----------------------------------------------------------
        hop_length = int(self.config.hop_length * sample_rate)
        frame_length = int(self.config.frame_length * sample_rate)

        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=self.config.min_pitch,
            fmax=self.config.max_pitch,
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        f0_voiced = f0[~np.isnan(f0)]

        if len(f0_voiced) > 0:
            pitch_p10 = float(np.percentile(f0_voiced, 10))
            pitch_p90 = float(np.percentile(f0_voiced, 90))
            high_pitch_thresh = pitch_p90
            low_pitch_thresh = pitch_p10
            feat["pitch_p10_hz"] = round(pitch_p10, 1)
            feat["pitch_p90_hz"] = round(pitch_p90, 1)
            feat["high_pitch_ratio"] = round(float(np.mean(f0_voiced > high_pitch_thresh)), 3)
            feat["low_pitch_ratio"] = round(float(np.mean(f0_voiced < low_pitch_thresh)), 3)
            feat["pitch_cv"] = round(float(np.std(f0_voiced) / np.mean(f0_voiced)) if np.mean(f0_voiced) > 0 else 0.0, 4)
        else:
            feat["pitch_p10_hz"] = 0.0
            feat["pitch_p90_hz"] = 0.0
            feat["high_pitch_ratio"] = 0.0
            feat["low_pitch_ratio"] = 0.0
            feat["pitch_cv"] = 0.0

        # --- voiced / unvoiced ratio -----------------------------------------------
        total_frames = len(f0) if len(f0) > 0 else 1
        voiced_frames = int(np.sum(~np.isnan(f0)))
        feat["voiced_ratio"] = round(voiced_frames / total_frames, 3)
        feat["unvoiced_ratio"] = round(1.0 - feat["voiced_ratio"], 3)

        # --- energy detail ---------------------------------------------------------
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        rms_nonzero = rms[rms > 0]

        feat["energy_p10"] = round(float(np.percentile(rms, 10)), 5)
        feat["energy_p90"] = round(float(np.percentile(rms, 90)), 5)
        feat["energy_cv"] = round(float(np.std(rms) / np.mean(rms)) if np.mean(rms) > 0 else 0.0, 4)
        if len(rms_nonzero) >= 2:
            feat["dynamic_range_db"] = round(float(20 * np.log10(np.max(rms_nonzero) / np.min(rms_nonzero))), 2)
        else:
            feat["dynamic_range_db"] = 0.0

        # energy slope (global trend)
        if len(rms) > 2:
            x = np.arange(len(rms))
            coeffs = np.polyfit(x, rms, 1)
            feat["energy_slope"] = round(float(coeffs[0]), 6)
        else:
            feat["energy_slope"] = 0.0

        # --- jitter, shimmer, HNR via parselmouth ----------------------------------
        try:
            import parselmouth
            from parselmouth.praat import call

            snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sample_rate))
            pitch_obj = call(snd, "To Pitch", 0.0, self.config.min_pitch, self.config.max_pitch)
            point_proc = call(snd, "To PointProcess (periodic, cc)", self.config.min_pitch, self.config.max_pitch)

            feat["jitter_local"] = round(float(call(point_proc, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)), 6)
            feat["jitter_rap"] = round(float(call(point_proc, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)), 6)
            feat["shimmer_local"] = round(float(call([snd, point_proc], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)), 6)
            feat["shimmer_apq3"] = round(float(call([snd, point_proc], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)), 6)

            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, self.config.min_pitch, 0.1, 1.0)
            feat["hnr_mean_db"] = round(float(call(harmonicity, "Get mean", 0, 0)), 2)
        except Exception as e:
            feat["jitter_local"] = 0.0
            feat["jitter_rap"] = 0.0
            feat["shimmer_local"] = 0.0
            feat["shimmer_apq3"] = 0.0
            feat["hnr_mean_db"] = 0.0
            print(f"  [warn] parselmouth features unavailable: {e}")

        # --- spectral features via librosa -----------------------------------------
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        feat["spectral_centroid_mean"] = round(float(np.mean(spectral_centroid)), 2)
        feat["spectral_centroid_std"] = round(float(np.std(spectral_centroid)), 2)

        spec = np.abs(librosa.stft(audio, hop_length=hop_length))
        flux = np.sqrt(np.sum(np.diff(spec, axis=1) ** 2, axis=0))
        feat["spectral_flux_mean"] = round(float(np.mean(flux)), 4)
        feat["spectral_flux_std"] = round(float(np.std(flux)), 4)

        # --- derived arousal & valence proxy scores --------------------------------
        # Arousal proxy: high energy + fast rate + high pitch variance → high arousal
        energy_norm = min(base.energy_mean / 0.1, 1.0)
        rate_norm = min(base.speaking_rate_wpm / 200.0, 1.0)
        pitch_var_norm = min(base.pitch_variance / 1000.0, 1.0)
        feat["arousal_proxy"] = round(0.4 * energy_norm + 0.3 * rate_norm + 0.3 * pitch_var_norm, 3)

        # Valence proxy: high HNR + higher pitch + positive spectral brightness → positive
        hnr_norm = min(max(feat["hnr_mean_db"], 0) / 20.0, 1.0)
        pitch_norm = min(base.pitch_mean_hz / 300.0, 1.0) if base.pitch_mean_hz > 0 else 0.0
        centroid_norm = min(feat["spectral_centroid_mean"] / 4000.0, 1.0)
        feat["valence_proxy"] = round(0.4 * hnr_norm + 0.3 * pitch_norm + 0.3 * centroid_norm, 3)

        return feat
