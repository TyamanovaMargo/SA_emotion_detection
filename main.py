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
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table

from src.pipeline import HRAssessmentPipeline
from src.config import load_config
from src.models.schemas import HRAssessmentResult
from src.utils.reporting import generate_html_report
from src.utils.scoring import (
    score_to_label,
    wpm_label,
    validate_strengths_and_dev_areas,
    compute_final_assessment,
    generate_hr_summary_from_scores,
    compute_content_voice_alignment,
)
from src.utils.comparison_report import (
    extract_person_name,
    analyze_person,
    generate_comparison_html,
    generate_comparison_json,
)
from src.utils.person_report import generate_person_aggregated_json
from src.utils.feature_impact import generate_feature_impact


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

def _enrich_content_indicators(
    llm_indicators: list,
    content_voice: dict,
) -> list:
    """Enrich LLM content_indicators with transcript-derived metrics (#18)."""
    enriched = list(llm_indicators) if llm_indicators else []
    if not content_voice or not content_voice.get("available"):
        return enriched

    li = content_voice.get("linguistic_indicators", {})
    hedge = li.get("hedge_word_count", 0)
    self_ref = li.get("self_reference_count", 0)
    assertion_ratio = li.get("assertion_ratio", 1.0)
    pos = li.get("positive_word_count", 0)
    neg = li.get("negative_word_count", 0)

    if hedge > 3:
        enriched.append(f"frequent hedging language ({hedge} hedge words)")
    elif hedge == 0:
        enriched.append("no hedging language (direct/assertive)")
    if self_ref > 10:
        enriched.append(f"high self-reference ({self_ref} instances — strong personal engagement)")
    if assertion_ratio < 0.95:
        enriched.append(f"assertion ratio: {assertion_ratio:.2f} (lower = more tentative)")
    if pos > neg + 3:
        enriched.append(f"positive lexicon dominant ({pos} pos vs {neg} neg words)")
    elif neg > pos + 3:
        enriched.append(f"negative lexicon dominant ({neg} neg vs {pos} pos words)")

    if content_voice.get("content_voice_mismatch"):
        enriched.append("content-voice mismatch: text sentiment diverges from vocal tone")

    return enriched


def _parse_score_range(sr: str) -> tuple:
    """Parse '80-90' or '80–90' string into (80, 90) ints. Returns (50, 50) on failure."""
    if not sr or not isinstance(sr, str):
        return (50, 50)
    cleaned = sr.replace("–", "-").replace("—", "-")
    parts = cleaned.split("-")
    try:
        return (int(parts[0].strip()), int(parts[1].strip()))
    except (ValueError, IndexError):
        return (50, 50)


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

        # --- Build structured output with all validation & post-processing ---
        vf = result.voice_features.model_dump() if result.voice_features else {}
        va = result.voice_analysis or {}

        # Big Five as clean dict (basic LLM scores)
        b5 = {}
        for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
            score_obj = getattr(result.big_five, trait)
            b5[trait] = {"score": score_obj.score, "confidence": score_obj.confidence, "reason": score_obj.reason}

        # (#2, #5) Validate strengths / development_areas — no overlap, neuroticism inverted
        validated_strengths, validated_dev = validate_strengths_and_dev_areas(
            b5, result.trait_strengths, result.personality_development_areas,
        )

        # (#6, #17) Final assessment: weighted average across basic, approximate, enriched
        approx_b5_raw = None
        if result.approximate_assessment:
            ad = result.approximate_assessment.model_dump()
            approx_b5_raw = ad.get("big5_approximate", {})
        enriched_b5 = None
        if result.llm_comparison and isinstance(result.llm_comparison, dict):
            enriched_b5 = result.llm_comparison.get("enriched_big5", {})
        final_assessment = compute_final_assessment(b5, approx_b5_raw, enriched_b5)

        # (#7) HR summary from final scores, not LLM template
        emo_summary = result.emotion_summary or {}
        dominant_emo = emo_summary.get("dominant_emotion", "neutral")
        mot_score = result.motivation.motivation_score
        eng_score = result.engagement.engagement_score if result.engagement else None
        hr_summary_generated = generate_hr_summary_from_scores(
            final_assessment, mot_score, eng_score, dominant_emo,
        )

        # (#14) Language detection fallback
        lang_detected = vf.get("detected_language", "unknown")
        lang_confidence = vf.get("language_confidence", 0.0)
        lang_source = "whisper"
        if (lang_confidence or 0) < 0.3 and transcript:
            # Fallback: assume English if transcript exists and has English words
            common_en = {"the", "is", "and", "to", "a", "of", "in", "that", "it", "for"}
            words_lower = set(transcript.lower().split()[:100])
            if len(words_lower & common_en) >= 3:
                lang_detected = "en"
                lang_confidence = 0.5
                lang_source = "transcript_fallback"

        # (#13) Non-native speaker notes
        lang_profile = vf.get("language_profile", "non_native_english")
        non_native_notes = None
        if lang_profile in ("non_native_english", "sea_english"):
            non_native_notes = {
                "jitter_shimmer_note": "Elevated jitter/shimmer may reflect L2 articulatory effort, not emotional instability",
                "pause_note": "Higher pause frequency is expected for non-native speakers (cognitive load)",
                "wpm_thresholds_adjusted": True,
            }

        # (#15) Content-voice alignment
        valence_mean = emo_summary.get("valence_mean", 0.0)
        content_voice = compute_content_voice_alignment(transcript, valence_mean)

        # (#20) Convert approximate_assessment score_range to numeric
        approx_dump = None
        if result.approximate_assessment:
            approx_dump = result.approximate_assessment.model_dump()
            for section_key in ("big5_approximate", "motivation_approximate", "engagement_approximate"):
                section = approx_dump.get(section_key, {})
                if isinstance(section, dict) and "score_range" not in section:
                    # big5_approximate is a dict of traits
                    for trait_key, trait_val in section.items():
                        if isinstance(trait_val, dict) and "score_range" in trait_val:
                            sr = trait_val["score_range"]
                            lo, hi = _parse_score_range(sr)
                            trait_val["score_low"] = lo
                            trait_val["score_high"] = hi
                elif isinstance(section, dict) and "score_range" in section:
                    sr = section["score_range"]
                    lo, hi = _parse_score_range(sr)
                    section["score_low"] = lo
                    section["score_high"] = hi

        # (#19) Pipeline metadata
        audio_dur = va.get("prosody", {}).get("f0_mean", 0)  # duration from result
        processing_metadata = {
            "pipeline_version": "2.0.0",
            "processing_date": datetime.now().isoformat(),
            "modules_used": [
                "whisper_transcription",
                "prosody_extraction_v2",
                "emotion_meralion_ser_v1",
                "egemaps_v02",
                "groq_llm_assessment",
                "voice_analysis_unified",
            ],
            "language_profile": lang_profile,
            "warnings": [],
        }
        # Add calibration warnings to metadata
        for w in final_assessment.get("calibration_warnings", []):
            processing_metadata["warnings"].append(
                f"High cross-module variance for {w['trait']}: spread={w['spread']}"
            )

        report = {
            # ── 0. Pipeline metadata (#19) ──
            "pipeline_metadata": processing_metadata,

            # ── 1. Candidate info ──
            "candidate": {
                "id": result.candidate_id,
                "position": result.position,
                "audio_file": str(audio_path),
                "transcript_file": str(transcript_path) if transcript_path else None,
                "timestamp": timestamp,
            },

            # ── 2. Transcript ──
            "transcript": transcript or None,

            # ── 3. Unified voice analysis ──
            "voice_analysis": {
                "prosody": va.get("prosody"),
                "voice_quality": va.get("voice_quality"),
                "spectral": va.get("spectral"),
                "emotion_timeline": va.get("emotion_timeline"),
                "emotion_aggregates": va.get("emotion_aggregates"),
                "paralinguistic_summary": va.get("paralinguistic_summary"),
            },

            # ── 4. Legacy voice features ──
            "legacy_voice_features": {
                "prosody": vf.get("prosody"),
                "acoustic_features": vf.get("acoustic_features"),
                "embedding_summary": vf.get("wavlm_embedding_summary"),
                "language": {
                    "detected": lang_detected,
                    "confidence": lang_confidence,
                    "profile": lang_profile,
                    "source": lang_source,
                },
                "non_native_notes": non_native_notes,
                "granular_features": result.granular_voice_features,
            },

            # ── 5. Emotion summary ──
            "emotion_summary": emo_summary,

            # ── 6. Personality assessment (LLM basic) ──
            "personality_assessment": {
                "big_five": b5,
                "strengths": validated_strengths,
                "development_areas": validated_dev,
            },

            # ── 6b. Final assessment (#6, #9, #17) — weighted average ──
            "final_assessment": final_assessment,

            # ── 7. Motivation & engagement (#18: enriched content_indicators) ──
            "motivation_engagement": {
                "motivation": {
                    "level": result.motivation.overall_level,
                    "score": result.motivation.motivation_score,
                    "pattern": result.motivation.pattern,
                    "voice_indicators": result.motivation.voice_indicators,
                    "content_indicators": _enrich_content_indicators(
                        result.motivation.content_indicators, content_voice,
                    ),
                },
                "engagement": {
                    "level": result.engagement.overall_level,
                    "score": result.engagement.engagement_score,
                    "reason": result.engagement.reason,
                },
                "strengths": result.motivation_strengths,
                "development_areas": result.motivation_development_areas,
            },

            # ── 8. Approximate voice-only assessment (#20: numeric score_range) ──
            "approximate_assessment": approx_dump,

            # ── 9. LLM ablation: baseline vs enriched ──
            "llm_comparison": result.llm_comparison,

            # ── 10. Content-voice alignment (#15) ──
            "content_voice_alignment": content_voice,

            # ── 11. HR summary (#7: from final scores) ──
            "hr_summary": hr_summary_generated,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
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
        "--group-by-person",
        action="store_true",
        help="Group results by person name extracted from filename (Name_Name_audio_N.ext)",
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
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch Streamlit dashboard after processing to visualise results",
    )
    parser.add_argument(
        "--feature-impact-report",
        action="store_true",
        help="Generate feature impact report showing how emotion features change Big Five scores",
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
        start_time = time.time()
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
            elapsed_time = time.time() - start_time
            console.print(f"\n[green]✓ Processing completed in {elapsed_time:.2f}s[/green]")
            if not args.quiet:
                pipeline.print_summary(result)
            sys.exit(0)

        except Exception as e:
            elapsed_time = time.time() - start_time
            console.print(f"\n[red]✗ Processing failed after {elapsed_time:.2f}s[/red]")
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
    person_results: Dict[str, List[Tuple[Path, HRAssessmentResult]]] = {}
    errors: List[Tuple[Path, str]] = []

    for i, audio_path in enumerate(audio_files, 1):
        console.print(f"\n[bold]({i}/{len(audio_files)})[/bold] {audio_path.name}")

        transcript_path = None
        if args.auto_transcript:
            transcript_path = find_transcript_for_audio(audio_path)

        candidate_id = args.candidate_id or audio_path.stem

        start_time = time.time()
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
            elapsed_time = time.time() - start_time
            console.print(f"  [green]✓ Completed in {elapsed_time:.2f}s[/green]")
            
            all_results.append((audio_path, result))

            if args.group_by_folder:
                group = audio_path.parent.name
                grouped_results.setdefault(group, []).append(result)

            # Always group by person for aggregated JSON generation
            person = extract_person_name(audio_path.name)
            person_results.setdefault(person, []).append((audio_path, result))

        except Exception as e:
            elapsed_time = time.time() - start_time
            console.print(f"  [red]✗ Failed after {elapsed_time:.2f}s - Error:[/red] {e}")
            errors.append((audio_path, str(e)))

    # Print batch summary table
    if not args.quiet and all_results:
        print_batch_summary(all_results)

    # Generate per-group summaries
    if args.group_by_folder and grouped_results:
        generate_summary(grouped_results, args.output_dir)

    # Always generate aggregated JSON per person (all recordings in one file)
    if person_results:
        console.print("\n[bold cyan]Generating aggregated JSON per person...[/bold cyan]")
        generate_person_aggregated_json(person_results, args.output_dir)
        console.print(f"[green]Aggregated JSONs saved to:[/green] {args.output_dir}")
    
    # Generate per-person comparison report (only if flag is set)
    if args.group_by_person and person_results:
        console.print("\n[bold cyan]Generating cross-recording comparison report...[/bold cyan]")
        person_analyses = {}
        for person_name, results_list in person_results.items():
            person_analyses[person_name] = analyze_person(person_name, results_list)
            console.print(f"  {person_name}: {len(results_list)} recordings, consistency={person_analyses[person_name]['overall_consistency']}")

        # Save JSON
        json_path = args.output_dir / "comparison_report.json"
        generate_comparison_json(person_analyses, json_path)
        console.print(f"[green]Comparison JSON:[/green] {json_path}")

        # Save HTML
        html_path = args.output_dir / "comparison_report.html"
        generate_comparison_html(person_analyses, html_path)
        console.print(f"[green]Comparison HTML:[/green] {html_path}")

    # Generate feature impact report (how emotion features change Big Five)
    if args.feature_impact_report or args.group_by_person:
        console.print("\n[bold cyan]Generating feature impact report...[/bold cyan]")
        try:
            impact_report = generate_feature_impact(args.output_dir)
            n_recs = impact_report.get("records_count", 0)
            if n_recs > 0:
                console.print(f"[green]Feature impact report:[/green] {n_recs} recordings analyzed")
                console.print(f"  JSON: {args.output_dir / 'feature_impact_report.json'}")
                console.print(f"  CSV:  {args.output_dir / 'feature_impact_summary.csv'}")
                console.print(f"  HTML: {args.output_dir / 'feature_impact_report.html'}")
            else:
                console.print("[yellow]No ablation data found — run with emotion features enabled[/yellow]")
        except Exception as e:
            console.print(f"[red]Feature impact report failed:[/red] {e}")

    # Print errors
    if errors:
        console.print(f"\n[red]Failed: {len(errors)} file(s)[/red]")
        for path, error in errors:
            console.print(f"  - {path.name}: {error}")

    console.print(
        f"\n[bold green]Done:[/bold green] {len(all_results)}/{len(audio_files)} processed"
    )
    console.print(f"[bold]Output:[/bold] {args.output_dir}")

    # Launch Streamlit dashboard if requested
    if args.dashboard:
        import subprocess
        dashboard_script = Path(__file__).parent / "src" / "utils" / "dashboard.py"
        if dashboard_script.exists():
            console.print("\n[bold cyan]Launching dashboard...[/bold cyan]")
            subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", str(dashboard_script), "--", str(args.output_dir)],
            )
        else:
            console.print(f"[red]Dashboard script not found:[/red] {dashboard_script}")

    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
