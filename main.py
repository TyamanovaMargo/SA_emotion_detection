#!/usr/bin/env python3
"""
HR Personality & Motivation Assessment Pipeline

Universal audio processor: takes any audio file(s) as input,
extracts voice features, and generates Big Five + motivation reports.

Supports:
  - Single audio file
  - Folder of audio files (recursive)
  - Optional transcript file (txt/json) paired with audio
  - Batch processing with summary table
  - JSON + HTML report output
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table

from src.pipeline import HRAssessmentPipeline
from src.config import load_config
from src.models.schemas import HRAssessmentResult
from src.utils.reporting import generate_html_report


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"}
TRANSCRIPT_EXTENSIONS = {".txt", ".json"}
console = Console()


# ---------------------------------------------------------------------------
# Audio & transcript discovery
# ---------------------------------------------------------------------------

def find_audio_files(folder: Path, recursive: bool = True) -> List[Path]:
    """Find all audio files in a folder."""
    audio_files = []
    pattern = "**/*" if recursive else "*"
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(folder.glob(f"{pattern}{ext}"))
        audio_files.extend(folder.glob(f"{pattern}{ext.upper()}"))
    return sorted(set(audio_files))


def find_transcript_for_audio(audio_path: Path) -> Optional[Path]:
    """
    Try to find a matching transcript file for an audio file.

    Search order:
      1. Same directory, same stem, .txt or .json
      2. Sibling 'transcripts/' or 'Transcription/' folder, same stem
      3. Fuzzy match (case-insensitive, ignore spaces/underscores)
    """
    stem = audio_path.stem
    parent = audio_path.parent

    # 1. Same directory — exact stem match
    for ext in TRANSCRIPT_EXTENSIONS:
        candidate = parent / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    # 2. Sibling transcript folders
    for folder_name in ("transcripts", "Transcripts", "Transcription", "transcription", "text", "Text"):
        trans_dir = parent / folder_name
        if not trans_dir.is_dir():
            # Also check one level up
            trans_dir = parent.parent / folder_name
        if trans_dir.is_dir():
            for ext in TRANSCRIPT_EXTENSIONS:
                candidate = trans_dir / f"{stem}{ext}"
                if candidate.exists():
                    return candidate

    # 3. Fuzzy match in same directory + sibling folders
    norm = lambda s: s.lower().replace("_", "").replace(" ", "").replace("-", "")
    audio_norm = norm(stem)

    search_dirs = [parent]
    for folder_name in ("transcripts", "Transcripts", "Transcription", "transcription"):
        for base in (parent, parent.parent):
            d = base / folder_name
            if d.is_dir():
                search_dirs.append(d)

    for d in search_dirs:
        for ext in TRANSCRIPT_EXTENSIONS:
            for f in d.glob(f"*{ext}"):
                if norm(f.stem) == audio_norm:
                    return f

    return None


def load_transcript(path: Path) -> str:
    """Load transcript text from .txt or .json file."""
    text = path.read_text(encoding="utf-8").strip()

    if path.suffix == ".json":
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                # Support {"text": "..."} or {"transcript": "..."}
                for key in ("text", "transcript", "content"):
                    if key in data:
                        return str(data[key])
            if isinstance(data, str):
                return data
        except json.JSONDecodeError:
            pass  # treat as plain text

    return text


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------

def process_single(
    pipeline: HRAssessmentPipeline,
    audio_path: Path,
    transcript_path: Optional[Path],
    candidate_id: Optional[str],
    position: Optional[str],
    output_dir: Path,
    html_report: bool = False,
    save: bool = True,
    quiet: bool = False,
    skip_transcription: bool = False,
) -> HRAssessmentResult:
    """
    Process one audio file through the full pipeline.

    If a transcript is provided, it is passed to the pipeline directly.
    If skip_transcription is True and no transcript exists, the pipeline
    runs in voice-only mode (no Whisper).
    """
    transcript_text = None
    if transcript_path and transcript_path.exists():
        transcript_text = load_transcript(transcript_path)
        console.print(f"  [dim]Transcript:[/dim] {transcript_path.name}")

    result = pipeline.process(
        audio_path=audio_path,
        candidate_id=candidate_id,
        position=position,
        save_output=False,  # we handle saving ourselves
        skip_transcription=skip_transcription and transcript_text is None,
        transcript_text=transcript_text,
    )
    transcript = transcript_text or ""

    # Save JSON report
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"{audio_path.stem}_{timestamp}_assessment.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "audio_file": str(audio_path),
                        "transcript_file": str(transcript_path) if transcript_path else None,
                        "candidate_id": result.candidate_id,
                        "position": result.position,
                        "timestamp": timestamp,
                    },
                    "transcript": transcript or None,
                    "voice_features": result.model_dump(include={"voice_features"}).get("voice_features") if hasattr(result, "voice_features") and result.voice_features else None,
                    "assessment": result.model_dump(exclude={"raw_response", "voice_features"} if hasattr(result, "voice_features") else {"raw_response"}),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        if not quiet:
            console.print(f"  [green]JSON saved:[/green] {json_path}")

    # HTML report
    if html_report:
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / f"{audio_path.stem}_report.html"
        generate_html_report(result, output_path=html_path, transcript=transcript or None)
        if not quiet:
            console.print(f"  [green]HTML saved:[/green] {html_path}")

    return result


def generate_summary(
    results: Dict[str, List[HRAssessmentResult]],
    output_dir: Path,
):
    """Generate aggregate summary JSON when multiple files are grouped by person/folder."""
    for group_name, group_results in results.items():
        if not group_results:
            continue

        avg_scores = {}
        for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
            avg_scores[trait] = round(
                sum(getattr(r.big_five, trait).score for r in group_results) / len(group_results), 1
            )

        motivation_counts: Dict[str, int] = {}
        for r in group_results:
            level = r.motivation.overall_level
            motivation_counts[level] = motivation_counts.get(level, 0) + 1

        all_strengths = []
        for r in group_results:
            all_strengths.extend(r.trait_strengths)

        summary = {
            "group": group_name,
            "recordings_analyzed": len(group_results),
            "average_big_five": avg_scores,
            "dominant_motivation": max(motivation_counts, key=motivation_counts.get),
            "motivation_distribution": motivation_counts,
            "common_strengths": list(set(all_strengths))[:5],
        }

        group_dir = output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        summary_path = group_dir / "SUMMARY.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        console.print(f"[bold green]Summary saved:[/bold green] {summary_path}")


# ---------------------------------------------------------------------------
# Batch summary table
# ---------------------------------------------------------------------------

def print_batch_summary(results: List[Tuple[Path, HRAssessmentResult]]):
    """Print a summary table of all processed files."""
    console.print("\n")

    table = Table(title="Assessment Summary", show_header=True, header_style="bold cyan")
    table.add_column("File / ID", style="bold")
    table.add_column("Motivation", justify="center")
    table.add_column("O", justify="center")
    table.add_column("C", justify="center")
    table.add_column("E", justify="center")
    table.add_column("A", justify="center")
    table.add_column("N", justify="center")
    table.add_column("Top Strength")

    for path, result in results:
        motivation_color = {
            "High": "green",
            "Medium": "yellow",
            "Low": "red",
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
    console.print("[dim]O=Openness  C=Conscientiousness  E=Extraversion  A=Agreeableness  N=Neuroticism[/dim]\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HR Personality & Motivation Assessment from Voice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single audio file
  python main.py interview.wav

  # Single audio + existing transcript
  python main.py interview.wav --transcript interview.txt

  # Folder of audio files (recursive)
  python main.py recordings/

  # Limit number of files
  python main.py recordings/ --limit 5

  # Group results by subfolder name (e.g. per person)
  python main.py recordings/ --group-by-folder

  # Custom output directory + HTML reports
  python main.py recordings/ -o reports/ --html-report

  # Specify candidate info
  python main.py interview.wav -c "John Doe" -p "Software Engineer"
        """,
    )

    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to an audio file or a folder containing audio files",
    )
    parser.add_argument(
        "--transcript", "-t",
        type=Path,
        default=None,
        help="Path to transcript file (.txt or .json) for single-file mode",
    )
    parser.add_argument(
        "--candidate-id", "-c",
        type=str,
        default=None,
        help="Candidate identifier",
    )
    parser.add_argument(
        "--position", "-p",
        type=str,
        default=None,
        help="Position / role for assessment context",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./outputs"),
        help="Output directory (default: ./outputs)",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML report for each file",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save JSON output files",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress detailed output",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Max number of audio files to process",
    )
    parser.add_argument(
        "--group-by-folder",
        action="store_true",
        help="Group results by immediate parent folder (useful for per-person analysis)",
    )
    parser.add_argument(
        "--auto-transcript",
        action="store_true",
        default=True,
        help="Auto-detect transcript files next to audio files (default: True)",
    )
    parser.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Skip Whisper transcription — use voice features only (faster, no transcript needed)",
    )

    args = parser.parse_args()

    if not args.input_path.exists():
        console.print(f"[red]Error:[/red] Path not found: {args.input_path}")
        sys.exit(1)

    # Build pipeline config
    config = load_config()
    config.output_dir = args.output_dir
    config.whisper.model_name = args.whisper_model
    pipeline = HRAssessmentPipeline(config)

    # --- Single file mode ---
    if args.input_path.is_file():
        if args.input_path.suffix.lower() not in AUDIO_EXTENSIONS:
            console.print(f"[red]Error:[/red] Not a supported audio format: {args.input_path.suffix}")
            console.print(f"Supported: {', '.join(sorted(AUDIO_EXTENSIONS))}")
            sys.exit(1)

        transcript_path = args.transcript
        if transcript_path is None and args.auto_transcript:
            transcript_path = find_transcript_for_audio(args.input_path)

        console.print(f"\n[bold blue]Processing:[/bold blue] {args.input_path.name}")
        try:
            result = process_single(
                pipeline=pipeline,
                audio_path=args.input_path,
                transcript_path=transcript_path,
                candidate_id=args.candidate_id or args.input_path.stem,
                position=args.position,
                output_dir=args.output_dir,
                html_report=args.html_report,
                save=not args.no_save,
                quiet=args.quiet,
                skip_transcription=args.skip_transcription,
            )
            if not args.quiet:
                pipeline.print_summary(result)
            sys.exit(0)

        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            if not args.quiet:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    # --- Folder mode ---
    audio_files = find_audio_files(args.input_path, recursive=True)

    if args.limit and args.limit > 0:
        console.print(f"[yellow]Limiting to first {args.limit} files[/yellow]")
        audio_files = audio_files[: args.limit]

    if not audio_files:
        console.print(f"[red]No audio files found in:[/red] {args.input_path}")
        console.print(f"Supported: {', '.join(sorted(AUDIO_EXTENSIONS))}")
        sys.exit(1)

    console.print(f"\n[bold blue]Found {len(audio_files)} audio file(s)[/bold blue]")

    all_results: List[Tuple[Path, HRAssessmentResult]] = []
    grouped_results: Dict[str, List[HRAssessmentResult]] = {}
    errors: List[Tuple[Path, str]] = []

    for i, audio_path in enumerate(audio_files, 1):
        console.print(f"\n[bold]({i}/{len(audio_files)})[/bold] {audio_path.name}")

        transcript_path = None
        if args.auto_transcript:
            transcript_path = find_transcript_for_audio(audio_path)

        candidate_id = args.candidate_id or audio_path.stem

        try:
            result = process_single(
                pipeline=pipeline,
                audio_path=audio_path,
                transcript_path=transcript_path,
                candidate_id=candidate_id,
                position=args.position,
                output_dir=args.output_dir,
                html_report=args.html_report,
                save=not args.no_save,
                quiet=args.quiet,
                skip_transcription=args.skip_transcription,
            )
            all_results.append((audio_path, result))

            if args.group_by_folder:
                group = audio_path.parent.name
                grouped_results.setdefault(group, []).append(result)

        except Exception as e:
            console.print(f"  [red]Error:[/red] {e}")
            errors.append((audio_path, str(e)))

    # Print batch summary table
    if not args.quiet and all_results:
        print_batch_summary(all_results)

    # Generate per-group summaries
    if args.group_by_folder and grouped_results:
        generate_summary(grouped_results, args.output_dir)

    # Print errors
    if errors:
        console.print(f"\n[red]Failed: {len(errors)} file(s)[/red]")
        for path, error in errors:
            console.print(f"  - {path.name}: {error}")

    console.print(
        f"\n[bold green]Done:[/bold green] {len(all_results)}/{len(audio_files)} processed"
    )
    console.print(f"[bold]Output:[/bold] {args.output_dir}")

    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
