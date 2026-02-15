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
)
from .assessment import GroqHRAssessor


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
    def assessor(self) -> GroqHRAssessor:
        if self._assessor is None:
            self._assessor = GroqHRAssessor(self.config.groq)
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
            
            progress.update(task, description="Detecting emotions...")
            emotions = self.emotion_detector.detect(audio, sample_rate, duration)
            
            # Skip emotion timeline to reduce memory usage
            # Uncomment if you have enough RAM (16GB+)
            # progress.update(task, description="Building emotion timeline...")
            # emotion_timeline = self.emotion_detector.detect_timeline(audio, sample_rate)
            # emotions.emotion_timeline = emotion_timeline
            
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
            )
            
            progress.update(task, description="Running HR assessment with Groq LLM...")
            assessment_input = HRAssessmentInput(
                transcript=transcription.text,
                voice_features=voice_features,
                audio_duration=duration,
                candidate_id=candidate_id,
                position=position,
            )
            
            result = self.assessor.assess(assessment_input)
            
            progress.update(task, description="Complete!")
        
        if save_output:
            self._save_result(result, audio_path, transcription.text)
        
        return result
    
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
        
        # Pace
        wpm = prosody.speaking_rate_wpm
        if wpm > 170:
            parts.append(f"VERY fast speech ({wpm:.0f} wpm)")
        elif wpm > 140:
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
        
        console.print("\n[bold yellow]Big Five Personality Profile:[/bold yellow]")
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            score = getattr(result.big_five, trait)
            bar = "█" * (score.score // 5) + "░" * (20 - score.score // 5)
            console.print(f"  {trait.capitalize():18} [{bar}] {score.score}/100 ({score.confidence}% conf)")
        
        console.print(f"\n[bold yellow]Motivation Level:[/bold yellow] {result.motivation.overall_level}")
        console.print(f"  Pattern: {result.motivation.pattern}")
        if result.motivation.voice_indicators:
            console.print("  Voice indicators:")
            for ind in result.motivation.voice_indicators[:5]:
                console.print(f"    • {ind}")
        
        console.print("\n[bold green]Key Strengths:[/bold green]")
        for strength in result.trait_strengths[:3]:
            console.print(f"  • {strength}")
        
        console.print("\n[bold red]Development Areas:[/bold red]")
        for area in result.personality_development_areas[:2]:
            console.print(f"  • {area}")
        
        console.print("\n[bold cyan]HR Summary:[/bold cyan]")
        console.print(f"  {result.hr_summary}")
        console.print("=" * 60 + "\n")
