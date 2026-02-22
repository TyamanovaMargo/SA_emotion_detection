"""Main HR Assessment Pipeline orchestrator."""

import json
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
import numpy as np
import librosa
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import PipelineConfig, load_config
from .models.schemas import (
    VoiceFeatures,
    HRAssessmentInput,
    HRAssessmentResult,
)
from .extractors import (
    WhisperTranscriber,
    ProsodyExtractor,
    EmotionDetector,
    EgemapsExtractor,
    VoiceAnalyzer,
)
from .assessment import GroqHRAssessor
from .utils.scoring import (
    dominant_emotion_with_tiebreaker,
    compute_slope_and_trend,
    compute_final_emotion_arc,
    compute_emotion_stability_10s,
    adjusted_emotional_shifts,
)


console = Console()


class HRAssessmentPipeline:
    """
    Complete HR personality and motivation assessment pipeline.
    
    Processes audio files through:
    1. Speech-to-text transcription (Whisper)
    2. Prosody extraction (pitch, energy, pauses)
    3. Emotion detection (emotion2vec)
    4. Acoustic feature extraction (eGeMAPS)
    5. Claude-based HR assessment
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or load_config()
        
        self._transcriber: Optional[WhisperTranscriber] = None
        self._prosody_extractor: Optional[ProsodyExtractor] = None
        self._emotion_detector: Optional[EmotionDetector] = None
        self._egemaps_extractor: Optional[EgemapsExtractor] = None
        self._voice_analyzer: Optional[VoiceAnalyzer] = None
        self._assessor: Optional[GroqHRAssessor] = None
    
    @property
    def transcriber(self) -> WhisperTranscriber:
        if self._transcriber is None:
            self._transcriber = WhisperTranscriber(self.config.whisper)
        return self._transcriber
    
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

    @property
    def assessor(self) -> GroqHRAssessor:
        if self._assessor is None:
            self._assessor = GroqHRAssessor(self.config.groq, self.config.motivation)
        return self._assessor
    
    def process(
        self,
        audio_path: Union[str, Path],
        candidate_id: Optional[str] = None,
        position: Optional[str] = None,
        save_output: bool = True,
        skip_transcription: bool = False,
        transcript_text: Optional[str] = None,
    ) -> HRAssessmentResult:
        """
        Process an audio file through the complete pipeline.
        
        Args:
            audio_path: Path to the audio file
            candidate_id: Optional candidate identifier
            position: Optional position being applied for
            save_output: Whether to save results to file
            skip_transcription: Skip Whisper transcription (voice-only mode)
            transcript_text: Pre-existing transcript text to use instead of Whisper
            
        Returns:
            HRAssessmentResult with complete analysis
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        console.print(f"\n[bold blue]Processing:[/bold blue] {audio_path.name}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading audio...", total=None)
            audio, sample_rate = self._load_audio(audio_path)
            duration = len(audio) / sample_rate
            
            # Detect language/accent (works even in skip-transcription mode)
            progress.update(task, description="Detecting language...")
            try:
                detected_lang, lang_confidence, language_profile = self.transcriber.detect_language(audio, sample_rate)
                console.print(f"  [dim]Language: {detected_lang} (conf={lang_confidence:.2f}), profile: {language_profile}[/dim]")
            except Exception as e:
                console.print(f"  [dim]Language detection failed: {e}, defaulting to non_native_english[/dim]")
                detected_lang = "unknown"
                lang_confidence = 0.0
                language_profile = "non_native_english"
            
            if transcript_text:
                from .models.schemas import TranscriptionResult
                transcription = TranscriptionResult(
                    text=transcript_text,
                    segments=[],
                    language="unknown",
                    word_count=len(transcript_text.split()),
                    duration_seconds=duration,
                    filler_words={},
                    filler_word_rate=0.0,
                )
                console.print("  [dim]Using provided transcript[/dim]")
            elif skip_transcription:
                from .models.schemas import TranscriptionResult
                transcription = TranscriptionResult(
                    text="",
                    segments=[],
                    language="unknown",
                    word_count=0,
                    duration_seconds=duration,
                    filler_words={},
                    filler_word_rate=0.0,
                )
                console.print("  [dim]Skipping transcription (voice-only mode)[/dim]")
            else:
                progress.update(task, description="Transcribing speech...")
                transcription = self.transcriber.transcribe(audio, sample_rate, duration)
            
            progress.update(task, description="Extracting prosody features...")
            prosody = self.prosody_extractor.extract(
                audio, sample_rate, transcription.word_count, duration
            )
            # B2: Cap legacy WPM for L2 speakers (matches voice_analyzer L2 cap)
            if language_profile in ("non_native_english", "sea_english") and prosody.speaking_rate_wpm > 220:
                prosody.speaking_rate_wpm = 220.0
            
            # Free Whisper GPU memory before loading emotion model
            if self._transcriber is not None:
                whisper_dev = self.config.whisper.device
                emotion_dev = self.config.emotion.device
                # Always unload to free memory, especially if sharing GPU
                if whisper_dev.startswith("cuda"):
                    console.print(f"  [dim]Unloading Whisper from {whisper_dev} to free GPU memory[/dim]")
                    self._transcriber.unload_model()
            
            progress.update(task, description="Detecting emotions...")
            emotions = self.emotion_detector.detect(audio, sample_rate, duration)
            
            # Build emotion timeline for detailed emotion analysis
            progress.update(task, description="Building emotion timeline...")
            emotion_timeline = self.emotion_detector.detect_timeline(audio, sample_rate)
            emotions.emotion_timeline = emotion_timeline
            
            progress.update(task, description="Extracting acoustic features...")
            egemaps = self.egemaps_extractor.extract(audio, sample_rate)
            
            progress.update(task, description="Generating voice profile summary...")
            embedding_summary = self._generate_embedding_summary(
                prosody, emotions, egemaps
            )
            
            voice_features = VoiceFeatures(
                emotions=emotions,
                prosody=prosody,
                acoustic_features=egemaps,
                wavlm_embedding_summary=embedding_summary,
                detected_language=detected_lang,
                language_confidence=round(lang_confidence, 3),
                language_profile=language_profile,
            )
            
            # Extract granular voice features for dashboard & approximate assessment
            progress.update(task, description="Extracting granular voice features...")
            granular_features = self.prosody_extractor.extract_granular(
                audio, sample_rate, transcription.word_count, duration
            )

            # --- Unified Voice Analysis (MERaLiON-SER: overlapping 5s segments + VAD + emotion dynamics) ---
            progress.update(task, description="Running unified voice analysis (MERaLiON-SER emotion dynamics)...")
            voice_analysis = self.voice_analyzer.analyze(
                audio, sample_rate, word_count=transcription.word_count,
                language_profile=language_profile,
            )

            # Generate emotion timeline visualization
            progress.update(task, description="Generating emotion visualization...")
            try:
                from .utils.emotion_visualizer import plot_emotion_timeline, plot_emotion_summary
                vis_dir = self.config.output_dir / "visualizations"
                vis_dir.mkdir(parents=True, exist_ok=True)
                plot_name = audio_path.stem
                plot_emotion_timeline(
                    voice_analysis["emotion_timeline"],
                    output_path=str(vis_dir / f"{plot_name}_emotion_timeline.png"),
                    title=f"Emotion Dynamics: {audio_path.stem}",
                )
                plot_emotion_summary(
                    voice_analysis["emotion_aggregates"],
                    output_path=str(vis_dir / f"{plot_name}_emotion_summary.png"),
                    title=f"Emotion Summary: {audio_path.stem}",
                )
            except Exception as e:
                console.print(f"  [dim]Visualization skipped: {e}[/dim]")

            # Build emotion summary from unified timeline for LLM
            progress.update(task, description="Computing emotion summary for LLM...")
            emotion_summary = self._build_emotion_summary_from_voice_analysis(voice_analysis)

            progress.update(task, description="Running HR assessment with Groq LLM...")
            assessment_input = HRAssessmentInput(
                transcript=transcription.text,
                voice_features=voice_features,
                audio_duration=duration,
                candidate_id=candidate_id,
                position=position,
                language_profile=language_profile,
            )
            
            result = self.assessor.assess(assessment_input, emotion_summary=emotion_summary)

            # Get approximate assessment from granular features
            progress.update(task, description="Getting approximate voice-based assessment...")
            emotion_timeline_rich = voice_analysis.get("emotion_timeline", [])
            approx = self.assessor.assess_approximate(granular_features, emotion_timeline_rich)

            # Run ablation: baseline (no emotion summary) vs enriched
            progress.update(task, description="Running emotion ablation comparison...")
            try:
                llm_comparison = self.assessor.assess_with_ablation(
                    assessment_input, emotion_summary
                )
            except Exception as e:
                print(f"  [warn] ablation failed: {e}")
                llm_comparison = None

            # Attach data to result
            result.granular_voice_features = granular_features
            result.emotion_timeline_rich = emotion_timeline_rich
            result.approximate_assessment = approx
            result.dual_emotions = None
            result.emotion_summary = emotion_summary
            result.llm_comparison = llm_comparison
            result.voice_analysis = voice_analysis
            
            progress.update(task, description="Complete!")
        
        if save_output:
            self._save_result(result, audio_path, transcription.text)
        
        return result
    
    def _build_emotion_summary_from_voice_analysis(self, voice_analysis: dict) -> dict:
        """Convert unified voice_analysis into emotion_summary dict for LLM.
        
        Fixes applied:
        - #3: dominant emotion tiebreaker (count → mean → recency → 'mixed')
        - #4: slope/trend always consistent (trend derived from slope)
        - #10: final_emotion_arc (last 15s)
        - #11: emotion_stability_10s
        - #12: adjusted_emotional_shifts for overlapping windows
        """
        agg = voice_analysis.get("emotion_aggregates", {})
        timeline = voice_analysis.get("emotion_timeline", [])
        dynamics = agg.get("dynamics", {})
        vad_stats = agg.get("vad_stats", {})
        derived = agg.get("derived", {})
        emo_stats = agg.get("emotion_stats", {})

        if not timeline:
            return {"error": "no timeline data"}

        n = len(timeline)

        # Dominant emotion with tiebreaker (#3)
        dominant = dominant_emotion_with_tiebreaker(emo_stats, timeline)
        dominant_ratio = round(emo_stats.get(dominant, {}).get("dominant_segments", 0) / max(n, 1), 3)

        # Emotion distribution
        emotions_7 = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]
        emo_distribution = {e: emo_stats.get(e, {}).get("dominant_segments", 0) for e in emotions_7}

        # Neutral ratio
        neutral_ratio = round(emo_distribution.get("neutral", 0) / max(n, 1), 3)

        # Volatility from dynamics
        volatility = dynamics.get("arousal_volatility", 0.0)

        # Compute actual slopes from timeline VAD data (#4)
        valences = np.array([seg.get("vad", {}).get("valence", 0.0) for seg in timeline])
        arousals = np.array([seg.get("vad", {}).get("arousal", 0.0) for seg in timeline])
        val_slope, val_trend = compute_slope_and_trend(valences)
        ar_slope, ar_trend = compute_slope_and_trend(arousals)

        val_stats = vad_stats.get("valence", {})
        ar_stats = vad_stats.get("arousal", {})

        # Raw and adjusted emotional shifts (#12)
        raw_shifts = dynamics.get("emotional_shifts", 0)
        adj_shifts = adjusted_emotional_shifts(raw_shifts, step_size=2.0, window_size=5.0)

        # Emotion stability on 10s windows (#11)
        stability_10s = compute_emotion_stability_10s(timeline, segment_step=2.0)

        # Final emotion arc — last 15s (#10)
        final_arc = compute_final_emotion_arc(timeline, window_seconds=15.0, segment_step=2.0)

        return {
            "total_segments": n,
            "dominant_emotion": dominant,
            "dominant_emotion_ratio": dominant_ratio,
            "emotion_volatility": volatility,
            "emotion_distribution": emo_distribution,
            "neutral_ratio": neutral_ratio,
            "valence_mean": val_stats.get("mean", 0),
            "valence_std": val_stats.get("std", 0),
            "arousal_mean": ar_stats.get("mean", 0),
            "arousal_std": ar_stats.get("std", 0),
            "valence_trend": val_trend,
            "valence_slope": val_slope,
            "arousal_trend": ar_trend,
            "arousal_slope": ar_slope,
            "avg_confidence": derived.get("confidence_score", 0),
            "avg_entropy": None,
            "low_confidence_ratio": 0.0,
            "model_agreement_rate": None,
            "top_transition": None,
            "stress_segments": dynamics.get("stress_segments", 0),
            "raw_emotional_shifts": raw_shifts,
            "adjusted_emotional_shifts": adj_shifts,
            "emotional_shifts": raw_shifts,
            "arc_type": dynamics.get("arc_type", "flat"),
            "peak_arousal_time": dynamics.get("peak_arousal_time", 0),
            "confidence_score": derived.get("confidence_score", 0),
            "stress_index": derived.get("stress_index", 0),
            "emotion_stability_10s": stability_10s,
            "final_emotion_arc": final_arc,
            "paralinguistic_summary": voice_analysis.get("paralinguistic_summary", ""),
        }

    def process_transcript_only(
        self,
        transcript: str,
        voice_features: VoiceFeatures,
        candidate_id: Optional[str] = None,
        position: Optional[str] = None,
    ) -> HRAssessmentResult:
        """
        Process pre-extracted features (skip audio processing).
        
        Useful when you already have transcript and voice features.
        """
        assessment_input = HRAssessmentInput(
            transcript=transcript,
            voice_features=voice_features,
            audio_duration=0,
            candidate_id=candidate_id,
            position=position,
        )
        
        return self.assessor.assess(assessment_input)
    
    def _load_audio(self, audio_path: Path) -> tuple[np.ndarray, int]:
        """Load and preprocess audio file."""
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        return audio, sr
    
    def _generate_embedding_summary(
        self,
        prosody,
        emotions,
        egemaps,
    ) -> str:
        """Generate a comprehensive voice profile summary."""
        parts = []
        
        # Energy
        if prosody.energy_level == "high":
            parts.append(f"HIGH energy speaker (mean={prosody.energy_mean:.3f}, std={prosody.energy_std:.3f})")
        elif prosody.energy_level == "low":
            parts.append(f"LOW energy speaker (mean={prosody.energy_mean:.3f})")
        else:
            parts.append(f"moderate energy (mean={prosody.energy_mean:.3f})")
        
        if prosody.energy_std > 0.03:
            parts.append("dynamic volume (large energy swings)")
        elif prosody.energy_std < 0.01:
            parts.append("flat volume (minimal variation)")
        
        # Pace (#1: corrected thresholds — 108 WPM is slow, not fast)
        wpm = prosody.speaking_rate_wpm
        if wpm > 180:
            parts.append(f"VERY fast speech ({wpm:.0f} wpm)")
        elif wpm > 150:
            parts.append(f"fast speech ({wpm:.0f} wpm)")
        elif wpm < 90:
            parts.append(f"VERY slow speech ({wpm:.0f} wpm)")
        elif wpm < 110:
            parts.append(f"slow speech ({wpm:.0f} wpm)")
        else:
            parts.append(f"moderate pace ({wpm:.0f} wpm)")
        
        # Pitch
        if prosody.pitch_variance > 800:
            parts.append("highly expressive intonation")
        elif prosody.pitch_variance < 200:
            parts.append("monotone/flat intonation")
        else:
            parts.append("moderate pitch variation")
        
        if prosody.pitch_slope > 0.3:
            parts.append("rising pitch trend (increasing engagement)")
        elif prosody.pitch_slope < -0.3:
            parts.append("falling pitch trend (declining energy)")
        
        # Fluency
        if prosody.pauses_per_minute > 8:
            parts.append(f"many pauses ({prosody.pauses_per_minute:.0f}/min)")
        elif prosody.pauses_per_minute < 2:
            parts.append("very few pauses (fluent)")
        if prosody.long_pauses_count > 3:
            parts.append(f"{prosody.long_pauses_count} long pauses (>1s)")
        
        # Emotion
        if emotions.primary_emotion in ["happy", "surprised"]:
            parts.append("positive emotional tone")
        elif emotions.primary_emotion in ["sad", "fearful"]:
            parts.append("subdued emotional tone")
        elif emotions.primary_emotion == "angry":
            parts.append("intense emotional tone")
        else:
            parts.append("neutral emotional baseline")
        
        parts.append(egemaps.summary)
        
        return "; ".join(parts)
    
    def _save_result(
        self,
        result: HRAssessmentResult,
        audio_path: Path,
        transcript: str,
    ) -> Path:
        """Save assessment result to JSON file."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{audio_path.stem}_{timestamp}_assessment.json"
        output_path = self.config.output_dir / filename
        
        output_data = {
            "metadata": {
                "audio_file": str(audio_path),
                "candidate_id": result.candidate_id,
                "position": result.position,
                "timestamp": timestamp,
            },
            "transcript": transcript,
            "assessment": result.model_dump(exclude={"raw_response"}),
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]Results saved to:[/green] {output_path}")
        
        return output_path
    
    def print_summary(self, result: HRAssessmentResult):
        """Print a formatted summary of the assessment."""
        console.print("\n" + "=" * 60)
        console.print("[bold cyan]HR ASSESSMENT SUMMARY[/bold cyan]")
        console.print("=" * 60)
        
        if result.candidate_id:
            console.print(f"[bold]Candidate:[/bold] {result.candidate_id}")
        if result.position:
            console.print(f"[bold]Position:[/bold] {result.position}")
        
        # --- MERaLiON-SER Emotion Analysis ---
        va = result.voice_analysis
        if va and va.get("emotion_aggregates"):
            agg = va["emotion_aggregates"]
            dynamics = agg.get("dynamics", {})
            derived = agg.get("derived", {})
            vad = agg.get("vad_stats", {})
            tl = va.get("emotion_timeline", [])

            console.print(f"\n[bold yellow]Emotion Detection (MERaLiON-SER-v1):[/bold yellow]")
            console.print(f"  Segments analyzed: {len(tl)}")

            # Dominant emotion from stats
            emo_stats = agg.get("emotion_stats", {})
            if emo_stats:
                dominant = max(emo_stats, key=lambda e: emo_stats[e].get("dominant_segments", 0))
                dom_segs = emo_stats[dominant].get("dominant_segments", 0)
                console.print(f"  Dominant: {dominant} ({dom_segs}/{len(tl)} segments)")

            # VAD
            ar = vad.get("arousal", {})
            val = vad.get("valence", {})
            console.print(f"  Valence:  mean={val.get('mean', 0):.2f}  std={val.get('std', 0):.2f}")
            console.print(f"  Arousal:  mean={ar.get('mean', 0):.2f}  std={ar.get('std', 0):.2f}")

            # Dynamics
            console.print(f"  Arc: {dynamics.get('arc_type', '?')}  |  Shifts: {dynamics.get('emotional_shifts', 0)}  |  Stress peaks: {dynamics.get('stress_segments', 0)}")
            console.print(f"  Confidence: {derived.get('confidence_score', 0):.2f}  |  Stress index: {derived.get('stress_index', 0):.2f}")

            # Paralinguistic summary
            para = va.get("paralinguistic_summary", "")
            if para:
                console.print(f"  [dim]{para}[/dim]")

        # --- Emotion summary (for LLM) ---
        emo_sum = result.emotion_summary
        if emo_sum and not emo_sum.get("error"):
            console.print("\n[bold yellow]Emotion Summary:[/bold yellow]")
            console.print(f"  Dominant: {emo_sum.get('dominant_emotion', '?')} ({emo_sum.get('dominant_emotion_ratio', 0):.0%})")
            console.print(f"  Volatility: {emo_sum.get('emotion_volatility', 0):.2f}  |  Neutral ratio: {emo_sum.get('neutral_ratio', 0):.2f}")

        # --- Ablation deltas ---
        ablation = result.llm_comparison
        if ablation and ablation.get("changes"):
            console.print("\n[bold yellow]Emotion Impact on Big Five (ablation):[/bold yellow]")
            for ch in ablation["changes"]:
                delta = ch.get("delta", 0)
                arrow = "[green]↑[/green]" if delta > 0 else "[red]↓[/red]"
                console.print(f"  {ch['trait'].capitalize():18} {ch.get('old_score','?')} → {ch.get('new_score','?')} ({arrow}{abs(delta)})")
            impact = ablation.get("emotion_impact_summary", "")
            if impact:
                console.print(f"  [dim]{impact}[/dim]")

        console.print("\n[bold yellow]Big Five Personality Profile:[/bold yellow]")
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            score = getattr(result.big_five, trait)
            bar = "█" * (score.score // 5) + "░" * (20 - score.score // 5)
            console.print(f"  {trait.capitalize():18} [{bar}] {score.score}/100 ({score.confidence}% conf)")
        
        # Motivation & Engagement visualization
        console.print("\n[bold yellow]Motivation & Engagement Analysis:[/bold yellow]")
        
        # Use motivation_score from result
        motivation_score = result.motivation.motivation_score
        
        # Color-coded motivation bar
        motivation_color = {
            "High": "green",
            "Medium": "yellow", 
            "Low": "red"
        }.get(result.motivation.overall_level, "white")
        
        motivation_bar = "█" * (motivation_score // 5) + "░" * (20 - motivation_score // 5)
        console.print(f"  Overall Motivation [{motivation_bar}] [{motivation_color}]{result.motivation.overall_level}[/{motivation_color}] ({motivation_score}/100)")
        console.print(f"  Pattern: {result.motivation.pattern}")
        
        if result.motivation.voice_indicators:
            console.print("\n  [bold cyan]Voice-Based Indicators:[/bold cyan]")
            for ind in result.motivation.voice_indicators[:5]:
                console.print(f"    • {ind}")
        
        # Use engagement_score from result
        engagement_score = result.engagement.engagement_score
        engagement_color = {
            "High": "green",
            "Medium": "yellow",
            "Low": "red"
        }.get(result.engagement.overall_level, "white")
        engagement_bar = "█" * (engagement_score // 5) + "░" * (20 - engagement_score // 5)
        
        console.print(f"\n  Engagement Level   [{engagement_bar}] [{engagement_color}]{result.engagement.overall_level}[/{engagement_color}] ({engagement_score}/100)")
        console.print(f"  [dim]{result.engagement.reason}[/dim]")
        
        console.print("\n[bold green]Key Strengths:[/bold green]")
        for strength in result.trait_strengths[:3]:
            console.print(f"  • {strength}")
        
        console.print("\n[bold red]Development Areas:[/bold red]")
        for area in result.personality_development_areas[:2]:
            console.print(f"  • {area}")
        
        console.print("\n[bold cyan]HR Summary:[/bold cyan]")
        console.print(f"  {result.hr_summary}")
        console.print("=" * 60 + "\n")
