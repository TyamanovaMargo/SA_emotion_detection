#!/usr/bin/env python3
"""
Lightweight Team Recordings processor using ONLY librosa features.
No heavy models (emotion2vec, Whisper) - just acoustic analysis.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import librosa
from rich.console import Console
from rich.table import Table
from groq import Groq

from src.config import load_config


console = Console()


def extract_personality_features(audio_path: Path) -> Dict:
    """
    Extract comprehensive acoustic features for personality inference.
    Based on research correlations with Big Five traits.
    """
    console.print(f"    [dim]Extracting acoustic features...[/dim]")
    
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Core prosody features
    intervals = librosa.effects.split(y, top_db=30)
    num_segments = len(intervals)
    
    # Speaking rate (segments per minute)
    speaking_rate = num_segments / duration * 60 if duration > 0 else 0
    pauses_per_min = max(0, num_segments - 1) / duration * 60 if duration > 0 else 0
    
    # Pitch (F0) analysis
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=75, fmax=300, sr=sr)
    f0_voiced = f0[voiced_flag] if voiced_flag.any() else np.array([0])
    
    pitch_mean = float(np.nanmedian(f0_voiced)) if len(f0_voiced) > 0 else 0
    pitch_std = float(np.nanstd(f0_voiced)) if len(f0_voiced) > 0 else 0
    pitch_range = float(np.nanmax(f0_voiced) - np.nanmin(f0_voiced)) if len(f0_voiced) > 0 else 0
    
    # Energy/loudness
    rms = librosa.feature.rms(y=y)
    energy_mean = float(rms.mean())
    energy_std = float(rms.std())
    energy_max = float(rms.max())
    
    # Spectral features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    
    # Spectral contrast (voice clarity)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    # Chroma (harmonic content)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Mel spectrogram energy
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_energy = float(librosa.power_to_db(mel_spec).mean())
    
    return {
        "duration_seconds": round(duration, 2),
        
        # Core prosody (HIGH personality correlation)
        "speaking_rate_segments_per_min": round(speaking_rate, 1),
        "pauses_per_minute": round(pauses_per_min, 1),
        "pitch_mean_hz": round(pitch_mean, 1),
        "pitch_std_hz": round(pitch_std, 1),
        "pitch_range_hz": round(pitch_range, 1),
        "energy_mean_rms": round(energy_mean, 4),
        "energy_std_rms": round(energy_std, 4),
        "energy_max_rms": round(energy_max, 4),
        
        # Spectral features (timbre/voice quality)
        "mfcc_coefficients": [round(float(x), 2) for x in mfcc_mean[:5]],
        "spectral_centroid_hz": round(float(spectral_centroid.mean()), 1),
        "spectral_rolloff_hz": round(float(spectral_rolloff.mean()), 1),
        "spectral_bandwidth_hz": round(float(spectral_bandwidth.mean()), 1),
        "zero_crossing_rate": round(float(zcr.mean()), 4),
        
        # Advanced features
        "spectral_contrast_mean": [round(float(x), 2) for x in spectral_contrast.mean(axis=1)[:4]],
        "chroma_mean": [round(float(x), 2) for x in chroma.mean(axis=1)[:6]],
        "mel_energy_db": round(mel_energy, 1),
        
        # Derived personality indicators
        "voice_brightness": "bright" if spectral_centroid.mean() > 2000 else "moderate" if spectral_centroid.mean() > 1500 else "dark",
        "energy_level": "high" if energy_mean > 0.06 else "medium" if energy_mean > 0.03 else "low",
        "pitch_stability": "stable" if pitch_std < 30 else "moderate" if pitch_std < 50 else "unstable",
    }


def assess_personality_from_features(
    features: Dict,
    transcript: str,
    candidate_id: str,
    position: str,
    groq_client: Groq,
    model: str = "llama-3.3-70b-versatile"
) -> Dict:
    """
    Use Groq LLM to assess personality from acoustic features + transcript.
    """
    console.print(f"    [dim]Running personality assessment...[/dim]")
    
    prompt = f"""You are an expert psychologist analyzing personality traits from voice acoustic features and speech content.

=== CANDIDATE INFO ===
Candidate ID: {candidate_id}
Position: {position}

=== TRANSCRIPT ===
{transcript}

=== ACOUSTIC FEATURES (from librosa analysis) ===
{json.dumps(features, indent=2)}

=== PERSONALITY INTERPRETATION GUIDELINES ===

**Use FULL 0-100 scale. Differentiate candidates clearly.**

**Openness (0-100)**:
- HIGH (70-100): Wide pitch range (>100 Hz), bright voice (centroid >2000 Hz), expressive content
- MEDIUM (40-69): Moderate pitch variation (50-100 Hz), standard vocabulary
- LOW (0-39): Narrow pitch range (<50 Hz), monotone, conventional language

**Conscientiousness (0-100)**:
- HIGH (70-100): Steady speaking rate (100-150 seg/min), <3 pauses/min, organized speech
- MEDIUM (40-69): Moderate pace/pauses, generally structured
- LOW (0-39): Erratic pace, >6 pauses/min, disorganized

**Extraversion (0-100)**:
- HIGH (70-100): Fast rate (>150 seg/min), high energy (>0.06 RMS), high pitch (>200 Hz)
- MEDIUM (40-69): Moderate rate (100-150), medium energy (0.03-0.06)
- LOW (0-39): Slow (<100), low energy (<0.03), subdued tone

**Agreeableness (0-100)**:
- HIGH (70-100): Warm timbre (MFCC patterns), low pitch variance (<30 Hz), cooperative tone
- MEDIUM (40-69): Balanced warmth, moderate prosody
- LOW (0-39): Cold/harsh tone, high pitch variance (>60 Hz), confrontational

**Neuroticism (0-100)**:
- HIGH (70-100): Unstable pitch (std >50 Hz), >6 pauses/min, tense voice
- MEDIUM (40-69): Some instability (std 30-50 Hz), moderate pauses
- LOW (0-39): Stable pitch (<30 Hz), <3 pauses/min, calm tone

**Motivation Level**:
- HIGH: Fast consistent pace, high energy, stable pitch, few pauses
- MEDIUM: Moderate pace/energy, some variation
- LOW: Slow pace, low energy, many pauses, unstable

=== OUTPUT FORMAT (JSON ONLY) ===

{{
  "big_five": {{
    "openness": {{"score": <0-100>, "confidence": <0-100>, "reason": "<based on pitch_range, spectral_centroid, content>"}},
    "conscientiousness": {{"score": <0-100>, "confidence": <0-100>, "reason": "<based on speaking_rate, pauses, structure>"}},
    "extraversion": {{"score": <0-100>, "confidence": <0-100>, "reason": "<based on speaking_rate, energy, pitch_mean>"}},
    "agreeableness": {{"score": <0-100>, "confidence": <0-100>, "reason": "<based on MFCC, pitch_std, tone>"}},
    "neuroticism": {{"score": <0-100>, "confidence": <0-100>, "reason": "<based on pitch_std, pauses, voice_quality>"}}
  }},
  "motivation": {{
    "overall_level": "<High/Medium/Low>",
    "voice_indicators": ["<indicator from features>", "..."],
    "content_indicators": ["<indicator from transcript>", "..."]
  }},
  "trait_strengths": ["<strength1>", "<strength2>", "<strength3>"],
  "hr_summary": "<2-3 sentence summary>"
}}

Return ONLY valid JSON, no markdown, no extra text."""

    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Clean markdown if present
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        
        return json.loads(result_text)
        
    except Exception as e:
        console.print(f"    [red]Assessment error: {e}[/red]")
        return None


def process_person_lightweight(
    person_name: str,
    person_folder: Path,
    groq_client: Groq,
    output_dir: Path,
    position: str = "Team Member"
) -> List[Dict]:
    """Process all recordings for one person using lightweight analysis."""
    console.print(f"\n[bold cyan]Processing {person_name}[/bold cyan]")
    
    audio_folder = person_folder / "Audio"
    transcript_folder = person_folder / "Transcription"
    
    if not audio_folder.exists():
        console.print(f"[red]No Audio folder found for {person_name}[/red]")
        return []
    
    # Find audio files
    audio_files = list(audio_folder.glob("*.aac")) + list(audio_folder.glob("*.wav")) + \
                  list(audio_folder.glob("*.mp3")) + list(audio_folder.glob("*.m4a"))
    
    if not audio_files:
        console.print(f"[red]No audio files found for {person_name}[/red]")
        return []
    
    console.print(f"Found {len(audio_files)} audio files")
    
    results = []
    
    for i, audio_path in enumerate(sorted(audio_files), 1):
        console.print(f"\n  [{i}/{len(audio_files)}] {audio_path.stem}")
        
        try:
            # Extract acoustic features
            features = extract_personality_features(audio_path)
            
            # Load transcript if available
            transcript = ""
            if transcript_folder.exists():
                # Try fuzzy matching
                audio_normalized = audio_path.stem.lower().replace('_', '').replace(' ', '')
                
                for trans_file in transcript_folder.glob("*.json"):
                    trans_normalized = trans_file.stem.lower().replace('_', '').replace(' ', '')
                    if audio_normalized == trans_normalized:
                        with open(trans_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            transcript = data.get('text', '') if isinstance(data, dict) else str(data)
                        break
            
            if not transcript:
                transcript = "[No transcript available - analysis based on acoustic features only]"
            
            # Run personality assessment
            assessment = assess_personality_from_features(
                features=features,
                transcript=transcript,
                candidate_id=f"{person_name}_{audio_path.stem}",
                position=position,
                groq_client=groq_client
            )
            
            if assessment:
                result = {
                    "candidate_id": f"{person_name}_{audio_path.stem}",
                    "audio_file": audio_path.name,
                    "acoustic_features": features,
                    "transcript": transcript,
                    "assessment": assessment
                }
                
                results.append(result)
                
                # Save individual result
                person_output_dir = output_dir / person_name
                person_output_dir.mkdir(parents=True, exist_ok=True)
                
                result_path = person_output_dir / f"{audio_path.stem}_assessment.json"
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                console.print(f"    [green]✓[/green] Saved to {person_output_dir}/")
            
        except Exception as e:
            console.print(f"    [red]✗ Error:[/red] {e}")
            import traceback
            traceback.print_exc()
    
    return results


def generate_person_summary(person_name: str, results: List[Dict], output_dir: Path):
    """Generate aggregate summary for one person."""
    if not results:
        return
    
    # Calculate average Big Five scores
    scores = []
    for r in results:
        if r.get('assessment') and r['assessment'].get('big_five'):
            scores.append(r['assessment']['big_five'])
    
    if not scores:
        return
    
    avg_scores = {
        "openness": sum(s['openness']['score'] for s in scores) / len(scores),
        "conscientiousness": sum(s['conscientiousness']['score'] for s in scores) / len(scores),
        "extraversion": sum(s['extraversion']['score'] for s in scores) / len(scores),
        "agreeableness": sum(s['agreeableness']['score'] for s in scores) / len(scores),
        "neuroticism": sum(s['neuroticism']['score'] for s in scores) / len(scores),
    }
    
    # Count motivation levels
    motivation_counts = {}
    for r in results:
        if r.get('assessment') and r['assessment'].get('motivation'):
            level = r['assessment']['motivation']['overall_level']
            motivation_counts[level] = motivation_counts.get(level, 0) + 1
    
    dominant_motivation = max(motivation_counts, key=motivation_counts.get) if motivation_counts else "Unknown"
    
    summary = {
        "person": person_name,
        "recordings_analyzed": len(results),
        "average_big_five": {k: round(v, 1) for k, v in avg_scores.items()},
        "dominant_motivation": dominant_motivation,
        "motivation_distribution": motivation_counts,
    }
    
    person_output_dir = output_dir / person_name
    summary_path = person_output_dir / "SUMMARY.json"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[bold green]Summary saved:[/bold green] {summary_path}")


def print_team_comparison(all_results: Dict[str, List[Dict]]):
    """Print comparison table of all team members."""
    console.print("\n")
    
    table = Table(title="Team Assessment Comparison", show_header=True, header_style="bold cyan")
    table.add_column("Person", style="bold")
    table.add_column("Files", justify="center")
    table.add_column("Avg O", justify="center")
    table.add_column("Avg C", justify="center")
    table.add_column("Avg E", justify="center")
    table.add_column("Avg A", justify="center")
    table.add_column("Avg N", justify="center")
    table.add_column("Motivation")
    
    for person_name, results in sorted(all_results.items()):
        scores = [r['assessment']['big_five'] for r in results if r.get('assessment') and r['assessment'].get('big_five')]
        
        if not scores:
            continue
        
        avg_o = sum(s['openness']['score'] for s in scores) / len(scores)
        avg_c = sum(s['conscientiousness']['score'] for s in scores) / len(scores)
        avg_e = sum(s['extraversion']['score'] for s in scores) / len(scores)
        avg_a = sum(s['agreeableness']['score'] for s in scores) / len(scores)
        avg_n = sum(s['neuroticism']['score'] for s in scores) / len(scores)
        
        motivation_counts = {}
        for r in results:
            if r.get('assessment') and r['assessment'].get('motivation'):
                level = r['assessment']['motivation']['overall_level']
                motivation_counts[level] = motivation_counts.get(level, 0) + 1
        
        dominant_motivation = max(motivation_counts, key=motivation_counts.get) if motivation_counts else "Unknown"
        
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


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Lightweight Team Recordings processor (librosa only, no heavy models)"
    )
    
    parser.add_argument(
        "team_folder",
        type=Path,
        default=Path("Team Recordings"),
        nargs='?',
        help="Path to Team Recordings folder"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./team_reports_lightweight"),
        help="Output directory"
    )
    
    parser.add_argument(
        "--position",
        type=str,
        default="Team Member",
        help="Position/role"
    )
    
    parser.add_argument(
        "--person",
        type=str,
        default=None,
        help="Process only specific person"
    )
    
    args = parser.parse_args()
    
    if not args.team_folder.exists():
        console.print(f"[red]Error:[/red] Folder not found: {args.team_folder}")
        sys.exit(1)
    
    # Initialize Groq client
    config = load_config()
    groq_client = Groq(api_key=config.groq.api_key)
    
    # Find person folders
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
        results = process_person_lightweight(
            person_name,
            person_folder,
            groq_client,
            args.output_dir,
            args.position
        )
        
        if results:
            all_results[person_name] = results
            generate_person_summary(person_name, results, args.output_dir)
    
    # Print team comparison
    if all_results:
        print_team_comparison(all_results)
    
    console.print(f"\n[bold green]✓ Complete![/bold green] Reports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
