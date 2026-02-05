"""eGeMAPS acoustic feature extraction using OpenSMILE."""

from typing import Optional, Dict
import numpy as np
import tempfile
import os

from ..config import EgemapsConfig
from ..models.schemas import EgemapsFeatures


class EgemapsExtractor:
    """Extract eGeMAPS features using OpenSMILE."""
    
    def __init__(self, config: Optional[EgemapsConfig] = None):
        self.config = config or EgemapsConfig()
        self._smile = None
        self._use_fallback = False
    
    def _load_smile(self):
        """Load OpenSMILE extractor."""
        if self._smile is not None:
            return
        
        try:
            import opensmile
            self._smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        except Exception as e:
            print(f"Warning: Could not load OpenSMILE: {e}")
            print("Using fallback librosa-based feature extraction.")
            self._use_fallback = True
    
    def extract(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> EgemapsFeatures:
        """
        Extract eGeMAPS features from audio.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            EgemapsFeatures object
        """
        self._load_smile()
        
        if self._use_fallback:
            return self._fallback_extraction(audio, sample_rate)
        
        return self._opensmile_extraction(audio, sample_rate)
    
    def _opensmile_extraction(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> EgemapsFeatures:
        """Extract features using OpenSMILE."""
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio, sample_rate)
        
        try:
            features_df = self._smile.process_file(temp_path)
            features = features_df.iloc[0].to_dict()
        finally:
            os.unlink(temp_path)
        
        spectral = self._extract_spectral_features(features)
        frequency = self._extract_frequency_features(features)
        energy = self._extract_energy_features(features)
        temporal = self._extract_temporal_features(features)
        voice_quality = self._extract_voice_quality_features(features)
        
        summary = self._generate_summary(spectral, frequency, energy, temporal, voice_quality)
        
        return EgemapsFeatures(
            spectral_features=spectral,
            frequency_features=frequency,
            energy_features=energy,
            temporal_features=temporal,
            voice_quality=voice_quality,
            summary=summary
        )
    
    def _extract_spectral_features(self, features: Dict) -> Dict[str, float]:
        """Extract spectral features from OpenSMILE output."""
        keys = [
            "spectralFlux_sma3_amean", "spectralFlux_sma3_stddevNorm",
            "mfcc1_sma3_amean", "mfcc2_sma3_amean", "mfcc3_sma3_amean", "mfcc4_sma3_amean"
        ]
        return {k: round(features.get(k, 0.0), 4) for k in keys if k in features}
    
    def _extract_frequency_features(self, features: Dict) -> Dict[str, float]:
        """Extract frequency features from OpenSMILE output."""
        keys = [
            "F0semitoneFrom27.5Hz_sma3nz_amean", "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
            "F1frequency_sma3nz_amean", "F2frequency_sma3nz_amean", "F3frequency_sma3nz_amean",
            "F1bandwidth_sma3nz_amean"
        ]
        return {k: round(features.get(k, 0.0), 4) for k in keys if k in features}
    
    def _extract_energy_features(self, features: Dict) -> Dict[str, float]:
        """Extract energy features from OpenSMILE output."""
        keys = [
            "loudness_sma3_amean", "loudness_sma3_stddevNorm",
            "shimmerLocaldB_sma3nz_amean", "shimmerLocaldB_sma3nz_stddevNorm"
        ]
        return {k: round(features.get(k, 0.0), 4) for k in keys if k in features}
    
    def _extract_temporal_features(self, features: Dict) -> Dict[str, float]:
        """Extract temporal features from OpenSMILE output."""
        keys = [
            "loudnessPeaksPerSec", "VoicedSegmentsPerSec",
            "MeanVoicedSegmentLengthSec", "StddevVoicedSegmentLengthSec"
        ]
        return {k: round(features.get(k, 0.0), 4) for k in keys if k in features}
    
    def _extract_voice_quality_features(self, features: Dict) -> Dict[str, float]:
        """Extract voice quality features from OpenSMILE output."""
        keys = [
            "jitterLocal_sma3nz_amean", "jitterLocal_sma3nz_stddevNorm",
            "HNRdBACF_sma3nz_amean", "HNRdBACF_sma3nz_stddevNorm"
        ]
        return {k: round(features.get(k, 0.0), 4) for k in keys if k in features}
    
    def _fallback_extraction(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> EgemapsFeatures:
        """Fallback feature extraction using librosa."""
        import librosa
        
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
        spectral_flux = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        
        spectral = {
            "spectral_centroid_mean": round(float(np.mean(spectral_centroid)), 2),
            "spectral_centroid_std": round(float(np.std(spectral_centroid)), 2),
            "spectral_bandwidth_mean": round(float(np.mean(spectral_bandwidth)), 2),
            "spectral_rolloff_mean": round(float(np.mean(spectral_rolloff)), 2),
            "spectral_flux_mean": round(float(np.mean(spectral_flux)), 4),
        }
        
        f0, voiced_flag, _ = librosa.pyin(audio, fmin=75, fmax=500, sr=sample_rate)
        f0_voiced = f0[~np.isnan(f0)]
        
        if len(f0_voiced) > 0:
            frequency = {
                "f0_mean_hz": round(float(np.mean(f0_voiced)), 2),
                "f0_std_hz": round(float(np.std(f0_voiced)), 2),
                "f0_range_hz": round(float(np.max(f0_voiced) - np.min(f0_voiced)), 2),
            }
        else:
            frequency = {"f0_mean_hz": 0.0, "f0_std_hz": 0.0, "f0_range_hz": 0.0}
        
        rms = librosa.feature.rms(y=audio)[0]
        energy = {
            "rms_mean": round(float(np.mean(rms)), 4),
            "rms_std": round(float(np.std(rms)), 4),
            "rms_max": round(float(np.max(rms)), 4),
        }
        
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        temporal = {
            "zero_crossing_rate_mean": round(float(np.mean(zcr)), 4),
            "zero_crossing_rate_std": round(float(np.std(zcr)), 4),
        }
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        voice_quality = {
            f"mfcc_{i}_mean": round(float(np.mean(mfccs[i])), 4)
            for i in range(min(4, mfccs.shape[0]))
        }
        
        summary = self._generate_summary(spectral, frequency, energy, temporal, voice_quality)
        
        return EgemapsFeatures(
            spectral_features=spectral,
            frequency_features=frequency,
            energy_features=energy,
            temporal_features=temporal,
            voice_quality=voice_quality,
            summary=summary
        )
    
    def _generate_summary(
        self,
        spectral: Dict[str, float],
        frequency: Dict[str, float],
        energy: Dict[str, float],
        temporal: Dict[str, float],
        voice_quality: Dict[str, float]
    ) -> str:
        """Generate a human-readable summary of acoustic features."""
        parts = []
        
        f0_key = next((k for k in frequency if "mean" in k.lower() and "f0" in k.lower()), None)
        if f0_key and frequency.get(f0_key, 0) > 0:
            f0_val = frequency[f0_key]
            if f0_val < 120:
                parts.append("low-pitched voice")
            elif f0_val < 180:
                parts.append("medium-pitched voice")
            else:
                parts.append("high-pitched voice")
        
        energy_key = next((k for k in energy if "mean" in k.lower()), None)
        if energy_key:
            energy_val = energy[energy_key]
            if energy_val < 0.02:
                parts.append("soft/quiet speaking")
            elif energy_val < 0.08:
                parts.append("moderate loudness")
            else:
                parts.append("loud/energetic speaking")
        
        spectral_key = next((k for k in spectral if "centroid" in k.lower() and "mean" in k.lower()), None)
        if spectral_key:
            centroid = spectral[spectral_key]
            if centroid > 2500:
                parts.append("bright timbre")
            elif centroid < 1500:
                parts.append("dark/warm timbre")
            else:
                parts.append("balanced timbre")
        
        if not parts:
            return "Standard acoustic profile"
        
        return "; ".join(parts)
