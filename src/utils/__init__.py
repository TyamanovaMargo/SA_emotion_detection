"""Utility functions (slim)."""

from .audio import load_audio, normalize_audio, trim_silence

__all__ = [
    "load_audio",
    "normalize_audio",
    "trim_silence",
]
