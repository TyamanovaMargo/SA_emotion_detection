#!/usr/bin/env python3
"""
HR Personality & Motivation Assessment Pipeline

Main entry point for processing audio files and generating HR assessments.
Supports both single files and folders of audio files.
"""

import argparse
from pathlib import Path
import sys
from typing import List

from rich.console import Console
from rich.table import Table

from src.pipeline import HRAssessmentPipeline
from src.config import load_config
from src.utils.reporting import generate_html_report
from src.models.schemas import HRAssessmentResult


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"}
console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="HR Personality & Motivation Assessment from Voice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single audio file
  python main.py audio/interview.wav
  
  # Process all audio files in a folder
  python main.py audio/candidates/
  
  # Process folder with position info
  python main.py audio/candidates/ --position "Software Engineer"
  
  # Generate HTML reports for all files
  python main.py audio/candidates/ --html-report
  
  # Use a specific Whisper model
  python main.py audio/interview.wav --whisper-model medium
        """
    )
    
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to audio file or folder containing audio files"
    )
    
    parser.add_argument(
        "--candidate-id", "-c",
        type=str,
        default=None,
        help="Candidate identifier"
    )
    
    parser.add_argument(
        "--position", "-p",
        type=str,
        default=None,
        help="Position being applied for"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./outputs"),
        help="Output directory for results (default: ./outputs)"
    )
    
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML report"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save JSON output"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress detailed output"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Maximum number of audio files to process (useful for testing)"
    )
    
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=True,
        help="Search for audio files recursively in subfolders (default: True)"
    )
    
    args = parser.parse_args()
    
    if not args.input_path.exists():
        print(f"Error: Path not found: {args.input_path}")
        sys.exit(1)
    
    config = load_config()
    config.output_dir = args.output_dir
    config.whisper.model_name = args.whisper_model
    
    pipeline = HRAssessmentPipeline(config)
    
    # Determine if input is file or folder
    if args.input_path.is_dir():
        return process_folder(pipeline, args)
    else:
        return process_single_file(pipeline, args)


def find_audio_files(folder: Path, recursive: bool = True) -> List[Path]:
    """Find all audio files in a folder."""
    audio_files = []
    pattern = "**/*" if recursive else "*"
    
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(folder.glob(f"{pattern}{ext}"))
        audio_files.extend(folder.glob(f"{pattern}{ext.upper()}"))
    
    return sorted(audio_files)


def process_single_file(pipeline: HRAssessmentPipeline, args) -> int:
    """Process a single audio file."""
    try:
        result = pipeline.process(
            audio_path=args.input_path,
            candidate_id=args.candidate_id,
            position=args.position,
            save_output=not args.no_save,
        )
        
        if not args.quiet:
            pipeline.print_summary(result)
        
        if args.html_report:
            html_path = args.output_dir / f"{args.input_path.stem}_report.html"
            generate_html_report(result, output_path=html_path)
            console.print(f"[green]HTML report saved to:[/green] {html_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


def process_folder(pipeline: HRAssessmentPipeline, args) -> int:
    """Process all audio files in a folder."""
    audio_files = find_audio_files(args.input_path, recursive=args.recursive)
    
    # Apply limit if specified
    if args.limit and args.limit > 0:
        console.print(f"[yellow]Limiting to first {args.limit} files[/yellow]")
        audio_files = audio_files[:args.limit]
    
    if not audio_files:
        console.print(f"[red]No audio files found in:[/red] {args.input_path}")
        console.print(f"Supported formats: {', '.join(AUDIO_EXTENSIONS)}")
        return 1
    
    console.print(f"\n[bold blue]Found {len(audio_files)} audio files to process[/bold blue]")
    
    results: List[tuple[Path, HRAssessmentResult]] = []
    errors: List[tuple[Path, str]] = []
    
    for i, audio_path in enumerate(audio_files, 1):
        console.print(f"\n[bold]({i}/{len(audio_files)})[/bold] Processing: {audio_path.name}")
        
        # Use filename as candidate_id if not provided
        candidate_id = args.candidate_id or audio_path.stem
        
        try:
            result = pipeline.process(
                audio_path=audio_path,
                candidate_id=candidate_id,
                position=args.position,
                save_output=not args.no_save,
            )
            results.append((audio_path, result))
            
            if args.html_report:
                html_path = args.output_dir / f"{audio_path.stem}_report.html"
                generate_html_report(result, output_path=html_path)
            
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            errors.append((audio_path, str(e)))
    
    # Print summary table
    if not args.quiet and results:
        print_batch_summary(results)
    
    # Print errors if any
    if errors:
        console.print(f"\n[red]Failed to process {len(errors)} file(s):[/red]")
        for path, error in errors:
            console.print(f"  • {path.name}: {error}")
    
    console.print(f"\n[bold green]Completed:[/bold green] {len(results)}/{len(audio_files)} files processed")
    console.print(f"[bold]Output directory:[/bold] {args.output_dir}")
    
    return 0 if not errors else 1


def print_batch_summary(results: List[tuple[Path, HRAssessmentResult]]):
    """Print a summary table of all processed candidates."""
    console.print("\n")
    
    table = Table(title="Batch Assessment Summary", show_header=True, header_style="bold cyan")
    table.add_column("Candidate", style="bold")
    table.add_column("Motivation", justify="center")
    table.add_column("O", justify="center")  # Openness
    table.add_column("C", justify="center")  # Conscientiousness
    table.add_column("E", justify="center")  # Extraversion
    table.add_column("A", justify="center")  # Agreeableness
    table.add_column("N", justify="center")  # Neuroticism
    table.add_column("Top Strength")
    
    for path, result in results:
        motivation_color = {
            "High": "green",
            "Medium": "yellow",
            "Low": "red"
        }.get(result.motivation.overall_level, "white")
        
        table.add_row(
            result.candidate_id or path.stem,
            f"[{motivation_color}]{result.motivation.overall_level}[/{motivation_color}]",
            str(result.big_five.openness.score),
            str(result.big_five.conscientiousness.score),
            str(result.big_five.extraversion.score),
            str(result.big_five.agreeableness.score),
            str(result.big_five.neuroticism.score),
            result.trait_strengths[0] if result.trait_strengths else "-",
        )
    
    console.print(table)
    console.print("\n[dim]O=Openness, C=Conscientiousness, E=Extraversion, A=Agreeableness, N=Neuroticism[/dim]")


if __name__ == "__main__":
    sys.exit(main())
