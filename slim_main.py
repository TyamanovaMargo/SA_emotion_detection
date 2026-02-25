#!/usr/bin/env python3
"""
Slim Voice Feature Extraction — CLI entry point.

No LLM, no transcription. Extracts all acoustic/prosodic/emotion features
and outputs a JSON report with deterministic motivation & engagement scores.

Usage:
    python slim_main.py audio.webm
    python slim_main.py audio.webm --output-dir results/
    python slim_main.py folder/ --output-dir results/
    python slim_main.py audio.webm --language-profile non_native_english
"""

import argparse
import sys
import time
from pathlib import Path
from rich.console import Console

from slim_pipeline import SlimPipeline
from src.config import load_config

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"}
console = Console()


def find_audio_files(folder: Path) -> list:
    """Recursively find all audio files in a folder."""
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(folder.rglob(f"*{ext}"))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Slim Voice Feature Extraction Pipeline (no LLM, no transcription)"
    )
    parser.add_argument(
        "input_path",
        help="Audio file or folder of audio files",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="outputs",
        help="Output directory for JSON reports (default: outputs/)",
    )
    parser.add_argument(
        "--language-profile",
        default="non_native_english",
        choices=["native_english", "non_native_english", "sea_english"],
        help="Language profile for scoring thresholds (default: non_native_english)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save JSON output (just print to console)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal console output",
    )

    args = parser.parse_args()
    input_path = Path(args.input_path)

    if not input_path.exists():
        console.print(f"[red]Error:[/red] Path not found: {input_path}")
        sys.exit(1)

    # Discover audio files
    if input_path.is_dir():
        audio_files = find_audio_files(input_path)
        if not audio_files:
            console.print(f"[red]Error:[/red] No audio files found in {input_path}")
            sys.exit(1)
        console.print(f"Found [bold]{len(audio_files)}[/bold] audio files")
    else:
        if input_path.suffix.lower() not in AUDIO_EXTENSIONS:
            console.print(f"[yellow]Warning:[/yellow] Unexpected extension: {input_path.suffix}")
        audio_files = [input_path]

    # Initialize pipeline
    config = load_config()
    pipeline = SlimPipeline(config)

    output_dir = None if args.no_save else args.output_dir
    results = []
    t0 = time.time()

    for audio_path in audio_files:
        try:
            result = pipeline.process(
                audio_path=audio_path,
                language_profile=args.language_profile,
                output_dir=output_dir,
            )
            results.append(result)
        except Exception as e:
            console.print(f"[red]Error processing {audio_path.name}:[/red] {e}")
            if not args.quiet:
                import traceback
                traceback.print_exc()

    # Batch summary
    if len(results) > 1:
        elapsed = time.time() - t0
        console.print(f"\n[bold]Batch complete:[/bold] {len(results)}/{len(audio_files)} files in {elapsed:.1f}s")

        from rich.table import Table
        table = Table(title="Slim Pipeline — Batch Results")
        table.add_column("File", style="cyan")
        table.add_column("Duration", justify="right")
        table.add_column("Dominant Emotion", style="magenta")
        table.add_column("Motivation", justify="right", style="green")
        table.add_column("Engagement", justify="right", style="blue")

        for r in results:
            mot = r["motivation_engagement"]
            agg = r["emotion_aggregates"]
            table.add_row(
                Path(r["audio_file"]).name,
                f"{r['audio_duration_seconds']:.1f}s",
                f"{agg['dominant_emotion']} ({agg['dominant_ratio']:.0%})",
                f"{mot['motivation_score']} ({mot['motivation_level']})",
                str(mot.get("engagement_score", "—")),
            )
        console.print(table)


if __name__ == "__main__":
    main()
