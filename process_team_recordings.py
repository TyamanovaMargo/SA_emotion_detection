#!/usr/bin/env python3
"""
Process Team Recordings with paired audio + transcript files.

Generates individual HR assessment reports for each team member.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console
from rich.table import Table

from src.pipeline import HRAssessmentPipeline
from src.config import load_config
from src.models.schemas import HRAssessmentResult, VoiceFeatures
from src.utils.reporting import generate_html_report


console = Console()


def find_audio_transcript_pairs(person_folder: Path, auto_transcribe: bool = True) -> List[Tuple[Path, Path | None]]:
    """
    Find matching audio and transcript files for a person.
    
    Args:
        person_folder: Path to person's folder
        auto_transcribe: If True, return audio without transcript (will auto-transcribe)
    
    Returns:
        List of (audio_path, transcript_path or None) tuples
    """
    audio_folder = person_folder / "Audio"
    transcript_folder = person_folder / "Transcription"
    
    if not audio_folder.exists():
        return []
    
    pairs = []
    
    # Find all audio files
    audio_files = list(audio_folder.glob("*.aac")) + list(audio_folder.glob("*.wav")) + \
                  list(audio_folder.glob("*.mp3")) + list(audio_folder.glob("*.m4a"))
    
    # Get all transcript files for fuzzy matching
    transcript_files = []
    if transcript_folder.exists():
        transcript_files = list(transcript_folder.glob("*.json"))
    
    for audio_path in audio_files:
        transcript_path = None
        
        # Try exact match first
        exact_match = transcript_folder / f"{audio_path.stem}.json"
        if exact_match.exists():
            transcript_path = exact_match
        else:
            # Try fuzzy matching (case-insensitive, ignore spaces/underscores)
            audio_normalized = audio_path.stem.lower().replace('_', '').replace(' ', '')
            
            for trans_file in transcript_files:
                trans_normalized = trans_file.stem.lower().replace('_', '').replace(' ', '')
                if audio_normalized == trans_normalized:
                    transcript_path = trans_file
                    break
        
        if transcript_path:
            pairs.append((audio_path, transcript_path))
        elif auto_transcribe:
            console.print(f"[yellow]No transcript found for {audio_path.name} - will auto-transcribe[/yellow]")
            pairs.append((audio_path, None))
        else:
            console.print(f"[yellow]Warning:[/yellow] No transcript found for {audio_path.name}")
    
    return sorted(pairs, key=lambda x: x[0])


def load_transcript(transcript_path: Path) -> str:
    """Load transcript text from JSON file."""
    with open(transcript_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract text field
    if isinstance(data, dict) and 'text' in data:
        return data['text']
    elif isinstance(data, str):
        return data
    else:
        raise ValueError(f"Unexpected transcript format in {transcript_path}")


def process_person(
    person_name: str,
    person_folder: Path,
    pipeline: HRAssessmentPipeline,
    output_dir: Path,
    position: str = "Team Member"
) -> List[HRAssessmentResult]:
    """
    Process all recordings for one person.
    
    Args:
        person_name: Name of the person
        person_folder: Path to person's folder
        pipeline: HRAssessmentPipeline instance
        output_dir: Output directory
        position: Position/role
        
    Returns:
        List of assessment results
    """
    console.print(f"\n[bold cyan]Processing {person_name}[/bold cyan]")
    
    pairs = find_audio_transcript_pairs(person_folder)
    
    if not pairs:
        console.print(f"[red]No audio-transcript pairs found for {person_name}[/red]")
        return []
    
    console.print(f"Found {len(pairs)} audio-transcript pairs")
    
    results = []
    
    for i, (audio_path, transcript_path) in enumerate(pairs, 1):
        console.print(f"\n  [{i}/{len(pairs)}] {audio_path.stem}")
        
        try:
            # Check if we need to auto-transcribe
            if transcript_path is None:
                console.print(f"    [cyan]Auto-transcribing...[/cyan]")
                # Use pipeline's full process method (includes transcription)
                result = pipeline.process(
                    audio_path=audio_path,
                    candidate_id=f"{person_name}_{audio_path.stem}",
                    position=position,
                )
                transcript = result.transcript
            else:
                # Load pre-existing transcript
                transcript = load_transcript(transcript_path)
                
                # Process audio to extract voice features
                import librosa
                audio, sr = librosa.load(audio_path, sr=16000, mono=True)
                duration = len(audio) / sr
                
                # Extract voice features
                from src.extractors import ProsodyExtractor, EmotionDetector, EgemapsExtractor
                
                prosody = pipeline.prosody_extractor.extract(audio, sr, len(transcript.split()), duration)
                emotions = pipeline.emotion_detector.detect(audio, sr, duration)
                egemaps = pipeline.egemaps_extractor.extract(audio, sr)
                
                embedding_summary = pipeline._generate_embedding_summary(prosody, emotions, egemaps)
                
                voice_features = VoiceFeatures(
                    emotions=emotions,
                    prosody=prosody,
                    acoustic_features=egemaps,
                    wavlm_embedding_summary=embedding_summary,
                )
                
                # Run assessment with pre-loaded transcript
                result = pipeline.process_transcript_only(
                    transcript=transcript,
                    voice_features=voice_features,
                    candidate_id=f"{person_name}_{audio_path.stem}",
                    position=position,
                )
            
            results.append(result)
            
            # Save individual result
            person_output_dir = output_dir / person_name
            person_output_dir.mkdir(parents=True, exist_ok=True)
            
            result_path = person_output_dir / f"{audio_path.stem}_assessment.json"
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "candidate_id": result.candidate_id,
                    "position": result.position,
                    "transcript": transcript,
                    "assessment": result.model_dump(exclude={"raw_response"}),
                }, f, indent=2, ensure_ascii=False)
            
            # Generate HTML report
            html_path = person_output_dir / f"{audio_path.stem}_report.html"
            generate_html_report(result, output_path=html_path, transcript=transcript)
            
            console.print(f"    [green]✓[/green] Saved to {person_output_dir}/")
            
        except Exception as e:
            console.print(f"    [red]✗ Error:[/red] {e}")
    
    return results


def generate_person_summary(
    person_name: str,
    results: List[HRAssessmentResult],
    output_dir: Path
):
    """Generate aggregate summary for one person across all their recordings."""
    if not results:
        return
    
    # Calculate average Big Five scores
    avg_scores = {
        "openness": sum(r.big_five.openness.score for r in results) / len(results),
        "conscientiousness": sum(r.big_five.conscientiousness.score for r in results) / len(results),
        "extraversion": sum(r.big_five.extraversion.score for r in results) / len(results),
        "agreeableness": sum(r.big_five.agreeableness.score for r in results) / len(results),
        "neuroticism": sum(r.big_five.neuroticism.score for r in results) / len(results),
    }
    
    # Count motivation levels
    motivation_counts = {}
    for r in results:
        level = r.motivation.overall_level
        motivation_counts[level] = motivation_counts.get(level, 0) + 1
    
    dominant_motivation = max(motivation_counts, key=motivation_counts.get)
    
    # Collect all strengths
    all_strengths = []
    for r in results:
        all_strengths.extend(r.trait_strengths)
    
    # Save summary
    summary = {
        "person": person_name,
        "recordings_analyzed": len(results),
        "average_big_five": {k: round(v, 1) for k, v in avg_scores.items()},
        "dominant_motivation": dominant_motivation,
        "motivation_distribution": motivation_counts,
        "common_strengths": list(set(all_strengths))[:5],
    }
    
    person_output_dir = output_dir / person_name
    summary_path = person_output_dir / "SUMMARY.json"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[bold green]Summary saved:[/bold green] {summary_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process Team Recordings with audio + transcript pairs"
    )
    
    parser.add_argument(
        "team_folder",
        type=Path,
        default=Path("Team Recordings"),
        nargs='?',
        help="Path to Team Recordings folder (default: Team Recordings)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./team_reports"),
        help="Output directory (default: ./team_reports)"
    )
    
    parser.add_argument(
        "--position",
        type=str,
        default="Team Member",
        help="Position/role for assessment"
    )
    
    parser.add_argument(
        "--person",
        type=str,
        default=None,
        help="Process only specific person (folder name)"
    )
    
    args = parser.parse_args()
    
    if not args.team_folder.exists():
        console.print(f"[red]Error:[/red] Folder not found: {args.team_folder}")
        sys.exit(1)
    
    # Initialize pipeline
    config = load_config()
    config.output_dir = args.output_dir
    pipeline = HRAssessmentPipeline(config)
    
    # Find all person folders
    person_folders = [f for f in args.team_folder.iterdir() if f.is_dir() and not f.name.startswith('.')]
    
    if args.person:
        person_folders = [f for f in person_folders if f.name == args.person]
        if not person_folders:
            console.print(f"[red]Error:[/red] Person '{args.person}' not found")
            sys.exit(1)
    
    console.print(f"\n[bold blue]Found {len(person_folders)} team members[/bold blue]")
    
    all_results = {}
    
    for person_folder in person_folders:
        person_name = person_folder.name
        results = process_person(
            person_name,
            person_folder,
            pipeline,
            args.output_dir,
            args.position
        )
        
        if results:
            all_results[person_name] = results
            generate_person_summary(person_name, results, args.output_dir)
    
    # Print team comparison table
    if all_results:
        print_team_comparison(all_results)
    
    console.print(f"\n[bold green]✓ Complete![/bold green] Reports saved to: {args.output_dir}")


def print_team_comparison(all_results: Dict[str, List[HRAssessmentResult]]):
    """Print comparison table of all team members."""
    console.print("\n")
    
    table = Table(title="Team Assessment Comparison", show_header=True, header_style="bold cyan")
    table.add_column("Person", style="bold")
    table.add_column("Recordings", justify="center")
    table.add_column("Avg O", justify="center")
    table.add_column("Avg C", justify="center")
    table.add_column("Avg E", justify="center")
    table.add_column("Avg A", justify="center")
    table.add_column("Avg N", justify="center")
    table.add_column("Motivation")
    
    for person_name, results in sorted(all_results.items()):
        avg_o = sum(r.big_five.openness.score for r in results) / len(results)
        avg_c = sum(r.big_five.conscientiousness.score for r in results) / len(results)
        avg_e = sum(r.big_five.extraversion.score for r in results) / len(results)
        avg_a = sum(r.big_five.agreeableness.score for r in results) / len(results)
        avg_n = sum(r.big_five.neuroticism.score for r in results) / len(results)
        
        motivation_counts = {}
        for r in results:
            level = r.motivation.overall_level
            motivation_counts[level] = motivation_counts.get(level, 0) + 1
        
        dominant_motivation = max(motivation_counts, key=motivation_counts.get)
        
        motivation_color = {
            "High": "green",
            "Medium": "yellow",
            "Low": "red"
        }.get(dominant_motivation, "white")
        
        table.add_row(
            person_name,
            str(len(results)),
            str(round(avg_o)),
            str(round(avg_c)),
            str(round(avg_e)),
            str(round(avg_a)),
            str(round(avg_n)),
            f"[{motivation_color}]{dominant_motivation}[/{motivation_color}]",
        )
    
    console.print(table)
    console.print("\n[dim]O=Openness, C=Conscientiousness, E=Extraversion, A=Agreeableness, N=Neuroticism[/dim]")


if __name__ == "__main__":
    main()
