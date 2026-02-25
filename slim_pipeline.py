"""
Slim Voice Feature Extraction Pipeline.

No LLM, no transcription — only acoustic/prosodic/emotion feature extraction
and deterministic motivation/engagement scoring.

Output: JSON with all extracted features + motivation & engagement scores.
"""

import json
import numpy as np
import librosa
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import PipelineConfig, load_config
from src.extractors.prosody import ProsodyExtractor
from src.extractors.emotion_meralion import EmotionDetector
from src.extractors.egemaps import EgemapsExtractor
from src.extractors.voice_analyzer import VoiceAnalyzer
from src.assessment.motivation_scorer import MotivationScorer
from src.models.schemas import VoiceFeatures, ProsodyFeatures, EmotionResult, EgemapsFeatures

console = Console()


class SlimPipeline:
    """
    Lightweight voice feature extraction pipeline.

    Steps:
      1. Load audio (resample to 16 kHz)
      2. Extract prosody features (pitch, energy, pauses, rhythm)
      3. Extract voice quality (HNR, jitter, shimmer)
      4. Extract spectral features (MFCC)
      5. Extract eGeMAPS 88 acoustic features
      6. Run MERaLiON-SER emotion timeline
      7. Compute deterministic motivation & engagement scores
      8. Write JSON report
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or load_config()

        self._prosody_extractor: Optional[ProsodyExtractor] = None
        self._emotion_detector: Optional[EmotionDetector] = None
        self._egemaps_extractor: Optional[EgemapsExtractor] = None
        self._voice_analyzer: Optional[VoiceAnalyzer] = None

    # -- lazy loaders -------------------------------------------------------

    @property
    def prosody_extractor(self) -> ProsodyExtractor:
        if self._prosody_extractor is None:
            self._prosody_extractor = ProsodyExtractor(self.config.prosody)
        return self._prosody_extractor

    @property
    def emotion_detector(self) -> EmotionDetector:
        if self._emotion_detector is None:
            self._emotion_detector = EmotionDetector(self.config.emotion)
        return self._emotion_detector

    @property
    def egemaps_extractor(self) -> EgemapsExtractor:
        if self._egemaps_extractor is None:
            self._egemaps_extractor = EgemapsExtractor(self.config.egemaps)
        return self._egemaps_extractor

    @property
    def voice_analyzer(self) -> VoiceAnalyzer:
        if self._voice_analyzer is None:
            self._voice_analyzer = VoiceAnalyzer(
                emotion_config=self.config.emotion,
                prosody_config=self.config.prosody,
                segment_duration=5.0,
                segment_step=2.0,
                emotion_detector=self.emotion_detector,
            )
        return self._voice_analyzer

    # -- audio loading ------------------------------------------------------

    @staticmethod
    def _load_audio(path: Path) -> tuple:
        audio, sr = librosa.load(path, sr=16000, mono=True)
        return audio, sr

    # -- main entry point ---------------------------------------------------

    def process(
        self,
        audio_path: Union[str, Path],
        language_profile: str = "non_native_english",
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Extract all voice features from an audio file.

        Args:
            audio_path: path to audio file (.wav/.mp3/.m4a/.webm/…)
            language_profile: scoring profile
            output_dir: where to save JSON (None = don't save)

        Returns:
            dict with all extracted features + motivation/engagement
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        console.print(f"\n[bold blue]Processing:[/bold blue] {audio_path.name}")
        start = datetime.now()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading audio…", total=None)
            audio, sr = self._load_audio(audio_path)
            duration = len(audio) / sr

            # 1. Prosody (ProsodyExtractor — librosa)
            progress.update(task, description="Extracting prosody features…")
            prosody = self.prosody_extractor.extract(audio, sr, word_count=0, duration=duration)
            if prosody.speaking_rate_wpm > 220:
                prosody.speaking_rate_wpm = 220.0

            # 2. Emotions (MERaLiON-SER — emotion detection)
            progress.update(task, description="Detecting emotions (MERaLiON-SER)…")
            emotions = self.emotion_detector.detect(audio, sr, duration)
            emotion_timeline = self.emotion_detector.detect_timeline(audio, sr)
            emotions.emotion_timeline = emotion_timeline

            # 3. eGeMAPS (openSMILE — 88 acoustic features)
            progress.update(task, description="Extracting eGeMAPS acoustic features…")
            egemaps = self.egemaps_extractor.extract(audio, sr)

            # 4. Unified voice analysis (prosody + voice quality + spectral + emotion timeline + aggregates)
            progress.update(task, description="Running unified voice analysis…")
            voice_analysis = self.voice_analyzer.analyze(
                audio, sr, word_count=0, language_profile=language_profile,
            )

            # 5. Granular features
            progress.update(task, description="Extracting granular features…")
            granular = self.prosody_extractor.extract_granular(audio, sr, word_count=0, duration=duration)

            # 6. Motivation & engagement (deterministic scorer)
            progress.update(task, description="Computing motivation & engagement…")
            voice_features = VoiceFeatures(
                emotions=emotions,
                prosody=prosody,
                acoustic_features=egemaps,
                wavlm_embedding_summary="",
                detected_language="unknown",
                language_confidence=0.0,
                language_profile=language_profile,
            )
            scorer = MotivationScorer(language_profile=language_profile)
            motivation_result = scorer.compute_motivation_score(voice_features)

            progress.update(task, description="Complete!")

        elapsed = (datetime.now() - start).total_seconds()

        # -- assemble JSON --------------------------------------------------
        result = self._build_result(
            audio_path=audio_path,
            duration=duration,
            language_profile=language_profile,
            prosody=prosody,
            egemaps=egemaps,
            voice_analysis=voice_analysis,
            granular=granular,
            motivation_result=motivation_result,
            elapsed=elapsed,
        )

        # -- save -----------------------------------------------------------
        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"{audio_path.stem}_{ts}_slim.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            console.print(f"  [bold green]JSON saved:[/bold green] {out_path}")

        # -- summary --------------------------------------------------------
        self._print_summary(result, elapsed)

        return result

    # -- result assembly ----------------------------------------------------

    @staticmethod
    def _build_result(
        audio_path: Path,
        duration: float,
        language_profile: str,
        prosody: ProsodyFeatures,
        egemaps: EgemapsFeatures,
        voice_analysis: Dict[str, Any],
        granular: Dict[str, Any],
        motivation_result: Dict[str, Any],
        elapsed: float,
    ) -> Dict[str, Any]:

        agg = voice_analysis.get("emotion_aggregates", {})
        emotion_stats = agg.get("emotion_stats", {})
        vad_stats = agg.get("vad_stats", {})
        dynamics = agg.get("dynamics", {})
        derived = agg.get("derived", {})

        # Find dominant emotion from emotion_stats
        dominant_emotion = "unknown"
        dominant_ratio = 0.0
        timeline_segs = voice_analysis.get("emotion_timeline", [])
        n_segments = len(timeline_segs)
        if emotion_stats:
            best_emo = max(emotion_stats, key=lambda e: emotion_stats[e].get("dominant_segments", 0))
            dominant_emotion = best_emo
            dominant_ratio = emotion_stats[best_emo]["dominant_segments"] / n_segments if n_segments else 0

        # Neutral ratio
        neutral_ratio = 0.0
        if emotion_stats and n_segments:
            neutral_ratio = emotion_stats.get("neutral", {}).get("dominant_segments", 0) / n_segments

        return {
            "pipeline": "slim",
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "audio_file": str(audio_path),
            "audio_duration_seconds": round(duration, 2),
            "language_profile": language_profile,
            "processing_time_seconds": round(elapsed, 2),

            # ── Prosody (from ProsodyExtractor) ──
            "prosody": {
                "speaking_rate_wpm": round(prosody.speaking_rate_wpm, 1),
                "pitch_mean_hz": round(prosody.pitch_mean_hz, 1),
                "pitch_variance": round(prosody.pitch_variance, 1),
                "pitch_range": round(prosody.pitch_range, 1),
                "pitch_slope": round(prosody.pitch_slope, 4),
                "energy_mean": round(prosody.energy_mean, 4),
                "energy_std": round(prosody.energy_std, 4),
                "energy_range": round(prosody.energy_range, 4),
                "energy_level": prosody.energy_level,
                "pauses_per_minute": round(prosody.pauses_per_minute, 1),
                "pause_duration_mean": round(prosody.pause_duration_mean, 3),
                "pause_duration_std": round(prosody.pause_duration_std, 3),
                "long_pauses_count": prosody.long_pauses_count,
                "articulation_rate": round(prosody.articulation_rate, 2),
                "speech_to_silence_ratio": round(prosody.speech_to_silence_ratio, 2),
                "rhythm_regularity": round(prosody.rhythm_regularity, 3),
            },

            # ── Voice Quality (from VoiceAnalyzer) ──
            "voice_quality": voice_analysis.get("voice_quality", {}),

            # ── Spectral (from VoiceAnalyzer) ──
            "spectral": voice_analysis.get("spectral", {}),

            # ── eGeMAPS (from openSMILE — 88 features) ──
            "egemaps": {
                "spectral_features": egemaps.spectral_features,
                "frequency_features": egemaps.frequency_features,
                "energy_features": egemaps.energy_features,
                "temporal_features": egemaps.temporal_features,
                "voice_quality": egemaps.voice_quality,
                "summary": egemaps.summary,
            },

            # ── Emotion Timeline (MERaLiON-SER per-segment) ──
            "emotion_timeline": timeline_segs,

            # ── Emotion Aggregates ──
            "emotion_aggregates": {
                "dominant_emotion": dominant_emotion,
                "dominant_ratio": round(dominant_ratio, 2),
                "emotion_stats": emotion_stats,
                "valence_mean": round(vad_stats.get("valence", {}).get("mean", 0), 3),
                "valence_std": round(vad_stats.get("valence", {}).get("std", 0), 3),
                "arousal_mean": round(vad_stats.get("arousal", {}).get("mean", 0), 3),
                "arousal_std": round(vad_stats.get("arousal", {}).get("std", 0), 3),
                "dominance_mean": round(vad_stats.get("dominance", {}).get("mean", 0), 3),
                "emotion_volatility": round(dynamics.get("arousal_volatility", 0), 3),
                "stress_index": round(derived.get("stress_index", 0), 3),
                "confidence_score": round(derived.get("confidence_score", 0), 3),
                "arc_type": dynamics.get("arc_type", "unknown"),
                "emotional_shifts": dynamics.get("emotional_shifts", 0),
                "stress_segments": dynamics.get("stress_segments", 0),
                "peak_arousal_time": dynamics.get("peak_arousal_time", 0),
                "neutral_ratio": round(neutral_ratio, 2),
                "segments_analyzed": n_segments,
            },

            # ── Motivation & Engagement (deterministic) ──
            "motivation_engagement": {
                "motivation_score": motivation_result["motivation_score"],
                "motivation_level": motivation_result["motivation_level"],
                "engagement_score": motivation_result.get("engagement_score"),
                "pattern": motivation_result["pattern"],
                "components": motivation_result["components"],
                "voice_indicators": motivation_result["voice_indicators"],
            },

            # ── L2 Adjustments ──
            "l2_adjustments": voice_analysis.get("l2_adjustments"),

            # ── Paralinguistic Summary ──
            "paralinguistic_summary": voice_analysis.get("paralinguistic_summary", ""),

            # ── Granular Features (flat dict for dashboards) ──
            "granular_features": granular,
        }

    # -- console summary ----------------------------------------------------

    @staticmethod
    def _print_summary(result: Dict[str, Any], elapsed: float):
        dur = result["audio_duration_seconds"]
        mot = result["motivation_engagement"]
        agg = result["emotion_aggregates"]

        console.print(f"\n[bold]{'=' * 60}[/bold]")
        console.print(f"[bold blue]SLIM PIPELINE — FEATURE EXTRACTION RESULTS[/bold blue]")
        console.print(f"[bold]{'=' * 60}[/bold]")
        console.print(f"Audio: {result['audio_file']}")
        console.print(f"Duration: {dur:.1f}s  |  Profile: {result['language_profile']}")
        console.print(f"Processing time: {elapsed:.1f}s")

        # Prosody
        p = result["prosody"]
        console.print(f"\n[bold cyan]Prosody:[/bold cyan]")
        console.print(f"  WPM: {p['speaking_rate_wpm']}  |  Pitch: {p['pitch_mean_hz']} Hz (var={p['pitch_variance']})")
        console.print(f"  Energy: {p['energy_mean']} ({p['energy_level']})  |  Pauses: {p['pauses_per_minute']}/min")
        console.print(f"  Rhythm: {p['rhythm_regularity']}  |  Speech/silence: {p['speech_to_silence_ratio']}")

        # Emotion
        console.print(f"\n[bold magenta]Emotion (MERaLiON-SER):[/bold magenta]")
        console.print(f"  Dominant: {agg['dominant_emotion']} ({agg['dominant_ratio']:.0%})")
        console.print(f"  Valence: mean={agg['valence_mean']}  |  Arousal: mean={agg['arousal_mean']}")
        console.print(f"  Arc: {agg['arc_type']}  |  Shifts: {agg['emotional_shifts']}  |  Stress: {agg['stress_index']}")
        console.print(f"  Segments: {agg['segments_analyzed']}  |  Confidence: {agg['confidence_score']}")

        # Voice quality
        vq = result.get("voice_quality", {})
        if vq:
            console.print(f"\n[bold yellow]Voice Quality:[/bold yellow]")
            console.print(f"  HNR: {vq.get('HNR', 'N/A')} dB  |  Jitter: {vq.get('jitter', 'N/A')}  |  Shimmer: {vq.get('shimmer', 'N/A')}")

        # Motivation
        def _bar(score, width=20):
            filled = int(score / 100 * width)
            return "█" * filled + "░" * (width - filled)

        console.print(f"\n[bold green]Motivation & Engagement:[/bold green]")
        console.print(f"  Motivation [{_bar(mot['motivation_score'])}] {mot['motivation_level']} ({mot['motivation_score']}/100)")
        if mot.get("engagement_score") is not None:
            console.print(f"  Engagement [{_bar(mot['engagement_score'])}] ({mot['engagement_score']}/100)")
        console.print(f"  Pattern: {mot['pattern']}")
        for ind in mot["voice_indicators"]:
            console.print(f"    • {ind}")

        console.print(f"\n[bold]{'=' * 60}[/bold]\n")
