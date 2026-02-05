"""Audio utility functions."""

from pathlib import Path
from typing import Union, Tuple
import numpy as np
import librosa


def load_audio(
    audio_path: Union[str, Path],
    target_sr: int = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        mono: Convert to mono
        
    Returns:
        Tuple of (audio array, sample rate)
    """
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=mono)
    return audio, sr


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range.
    
    Args:
        audio: Audio waveform
        
    Returns:
        Normalized audio
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    top_db: int = 20,
) -> np.ndarray:
    """
    Trim silence from beginning and end of audio.
    
    Args:
        audio: Audio waveform
        sample_rate: Sample rate
        top_db: Threshold in dB below reference to consider as silence
        
    Returns:
        Trimmed audio
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def get_audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """Get audio duration in seconds."""
    return len(audio) / sample_rate


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
