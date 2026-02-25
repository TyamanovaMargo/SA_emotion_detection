"""
Lightweight audio preprocessor for HR assessment pipeline.

Goals:
- Improve SNR for emotion detection and eGeMAPS
- Preserve prosody (pitch, rhythm, pauses) — no aggressive compression
- Standardise sample rate, channels, bit depth

Steps applied (in order):
1. Convert to mono 16 kHz float32
2. DC offset removal
3. High-pass filter 80 Hz (removes rumble, HVAC hum)
4. Optional 50/60 Hz notch filter (power-line hum)
5. Gentle spectral noise reduction (RMS-based spectral subtraction, max 6 dB)
6. Peak normalisation to -1 dBFS  (no dynamic compression — preserves energy dynamics)
7. Trim leading/trailing silence (>500 ms blocks only)
"""

from __future__ import annotations

import numpy as np
import scipy.signal as signal
from typing import Tuple


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(
    audio: np.ndarray,
    sample_rate: int,
    notch_hz: float | None = None,
    noise_reduce_db: float = 4.0,
    trim_silence: bool = True,
    silence_threshold_db: float = -45.0,
    silence_min_ms: float = 500.0,
) -> Tuple[np.ndarray, int]:
    """
    Apply lightweight preprocessing to audio.

    Args:
        audio:               Mono float32 array (any sample rate)
        sample_rate:         Input sample rate
        notch_hz:            Frequency to notch-filter (50 or 60). None = skip.
        noise_reduce_db:     Max spectral subtraction gain (dB). 0 = skip.
        trim_silence:        Trim leading/trailing silence blocks.
        silence_threshold_db: RMS threshold for silence detection (dBFS).
        silence_min_ms:      Minimum silence block length to trim (ms).

    Returns:
        (processed_audio, 16000)  — always 16 kHz mono float32
    """
    # 1. Ensure mono float32
    audio = _to_mono_float(audio)

    # 2. Resample to 16 kHz
    if sample_rate != 16000:
        audio = _resample(audio, sample_rate, 16000)
        sample_rate = 16000

    # 3. DC offset removal
    audio = audio - np.mean(audio)

    # 4. High-pass filter at 80 Hz (preserves fundamental pitch, removes rumble)
    audio = _highpass(audio, sample_rate, cutoff_hz=80.0, order=4)

    # 5. Notch filter (power-line hum)
    if notch_hz is not None:
        audio = _notch(audio, sample_rate, freq_hz=notch_hz, q=30.0)
        # Also notch 2nd harmonic
        audio = _notch(audio, sample_rate, freq_hz=notch_hz * 2, q=30.0)

    # 6. Gentle spectral noise reduction
    if noise_reduce_db > 0:
        audio = _spectral_subtract(audio, sample_rate, max_reduce_db=noise_reduce_db)

    # 7. Peak normalise to -1 dBFS (no compression — preserves dynamics)
    audio = _peak_normalise(audio, target_dbfs=-1.0)

    # 8. Trim leading/trailing silence
    if trim_silence:
        audio = _trim_silence(
            audio, sample_rate,
            threshold_db=silence_threshold_db,
            min_silence_ms=silence_min_ms,
        )

    return audio, sample_rate


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_mono_float(audio: np.ndarray) -> np.ndarray:
    audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def _highpass(audio: np.ndarray, sr: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    nyq = sr / 2.0
    b, a = signal.butter(order, cutoff_hz / nyq, btype="high")
    return signal.filtfilt(b, a, audio).astype(np.float32)


def _notch(audio: np.ndarray, sr: int, freq_hz: float, q: float = 30.0) -> np.ndarray:
    nyq = sr / 2.0
    if freq_hz >= nyq:
        return audio
    b, a = signal.iirnotch(freq_hz / nyq, q)
    return signal.filtfilt(b, a, audio).astype(np.float32)


def _spectral_subtract(
    audio: np.ndarray,
    sr: int,
    max_reduce_db: float = 4.0,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    noise_percentile: float = 10.0,
) -> np.ndarray:
    """
    Gentle spectral subtraction using noise floor estimated from quiet frames.
    Capped at max_reduce_db to avoid musical noise / prosody distortion.
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    window = np.hanning(frame_len)

    # Pad audio
    pad = frame_len // 2
    audio_pad = np.pad(audio, pad, mode="reflect")

    # Compute STFT magnitudes
    frames = []
    positions = []
    for start in range(0, len(audio_pad) - frame_len + 1, hop_len):
        frame = audio_pad[start:start + frame_len] * window
        spectrum = np.fft.rfft(frame)
        frames.append(spectrum)
        positions.append(start)

    if not frames:
        return audio

    magnitudes = np.abs(np.array(frames))  # (n_frames, n_bins)
    phases = np.angle(np.array(frames))

    # Estimate noise floor from quietest percentile of frames
    frame_rms = np.sqrt(np.mean(magnitudes ** 2, axis=1))
    noise_mask = frame_rms <= np.percentile(frame_rms, noise_percentile)
    if noise_mask.sum() == 0:
        return audio
    noise_spectrum = np.mean(magnitudes[noise_mask], axis=0)

    # Spectral subtraction with floor
    max_gain = 10 ** (max_reduce_db / 20.0)
    subtracted = magnitudes - noise_spectrum[np.newaxis, :] * (max_gain - 1.0) / max_gain
    subtracted = np.maximum(subtracted, magnitudes / max_gain)  # floor = original / max_gain

    # Reconstruct
    reconstructed = subtracted * np.exp(1j * phases)

    # Overlap-add
    output = np.zeros(len(audio_pad), dtype=np.float32)
    weight = np.zeros(len(audio_pad), dtype=np.float32)
    for i, start in enumerate(positions):
        frame_out = np.fft.irfft(reconstructed[i], n=frame_len).real * window
        output[start:start + frame_len] += frame_out.astype(np.float32)
        weight[start:start + frame_len] += window ** 2

    weight = np.where(weight > 1e-8, weight, 1.0)
    output /= weight
    output = output[pad:pad + len(audio)]

    return output.astype(np.float32)


def _peak_normalise(audio: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    target_linear = 10 ** (target_dbfs / 20.0)
    return (audio * target_linear / peak).astype(np.float32)


def _trim_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -45.0,
    min_silence_ms: float = 500.0,
) -> np.ndarray:
    """Trim only leading and trailing silence blocks >= min_silence_ms."""
    threshold_linear = 10 ** (threshold_db / 20.0)
    frame_ms = 20.0
    frame_len = int(sr * frame_ms / 1000)
    min_frames = int(min_silence_ms / frame_ms)

    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return audio

    rms = np.array([
        np.sqrt(np.mean(audio[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])
    is_silent = rms < threshold_linear

    # Find first non-silent frame
    start_frame = 0
    silent_count = 0
    for i, s in enumerate(is_silent):
        if s:
            silent_count += 1
        else:
            if silent_count >= min_frames:
                start_frame = i
            else:
                start_frame = max(0, i - silent_count)
            break

    # Find last non-silent frame
    end_frame = n_frames
    silent_count = 0
    for i in range(n_frames - 1, -1, -1):
        if is_silent[i]:
            silent_count += 1
        else:
            if silent_count >= min_frames:
                end_frame = i + 1
            else:
                end_frame = min(n_frames, i + 1 + silent_count)
            break

    start_sample = start_frame * frame_len
    end_sample = end_frame * frame_len
    return audio[start_sample:end_sample]
