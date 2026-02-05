#!/usr/bin/env python3
"""
Demo script showing how to use the HR Assessment Pipeline programmatically.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import HRAssessmentPipeline
from src.config import load_config
from src.models.schemas import (
    VoiceFeatures,
    ProsodyFeatures,
    EmotionResult,
    EgemapsFeatures,
)
from src.utils.reporting import generate_html_report


def demo_with_audio_file():
    """Demo: Process an actual audio file."""
    print("=" * 60)
    print("Demo: Processing Audio File")
    print("=" * 60)
    
    config = load_config()
    pipeline = HRAssessmentPipeline(config)
    
    audio_path = Path("examples/sample_interview.wav")
    
    if not audio_path.exists():
        print(f"Note: Sample audio file not found at {audio_path}")
        print("To test with real audio, place a WAV file there.")
        return None
    
    result = pipeline.process(
        audio_path=audio_path,
        candidate_id="DEMO-001",
        position="Software Engineer",
    )
    
    pipeline.print_summary(result)
    
    return result


def demo_with_mock_data():
    """Demo: Process with pre-extracted features (no audio needed)."""
    print("=" * 60)
    print("Demo: Processing with Mock Data")
    print("=" * 60)
    
    mock_transcript = """
    I'm really excited about this opportunity. In my previous role, I led a team of 
    five developers and we successfully delivered three major projects ahead of schedule. 
    I believe in proactive communication and always try to anticipate potential issues 
    before they become problems. What drives me is the challenge of solving complex 
    technical problems while also mentoring junior team members. I'm particularly 
    interested in your company's focus on innovation and would love to contribute 
    to your AI initiatives.
    """
    
    mock_voice_features = VoiceFeatures(
        emotions=EmotionResult(
            primary_emotion="happy",
            confidence=0.72,
            emotion_scores={
                "happy": 0.72,
                "neutral": 0.18,
                "surprised": 0.05,
                "sad": 0.02,
                "angry": 0.01,
                "fearful": 0.01,
                "disgusted": 0.01,
            },
            emotion_timeline=None,
        ),
        prosody=ProsodyFeatures(
            speaking_rate_wpm=145.0,
            pitch_mean_hz=185.5,
            pitch_variance=850.0,
            pitch_range=120.0,
            energy_level="high",
            energy_mean=0.065,
            pauses_per_minute=4.2,
            pause_duration_mean=0.45,
            articulation_rate=4.8,
        ),
        acoustic_features=EgemapsFeatures(
            spectral_features={
                "spectral_centroid_mean": 2150.5,
                "spectral_flux_mean": 0.0234,
            },
            frequency_features={
                "f0_mean_hz": 185.5,
                "f0_std_hz": 28.3,
            },
            energy_features={
                "rms_mean": 0.065,
                "rms_std": 0.018,
            },
            temporal_features={
                "zero_crossing_rate_mean": 0.085,
            },
            voice_quality={
                "mfcc_0_mean": -12.5,
                "mfcc_1_mean": 45.2,
            },
            summary="medium-pitched voice; moderate loudness; balanced timbre",
        ),
        wavlm_embedding_summary="energetic speaker; expressive intonation; positive emotional tone",
    )
    
    config = load_config()
    pipeline = HRAssessmentPipeline(config)
    
    result = pipeline.process_transcript_only(
        transcript=mock_transcript,
        voice_features=mock_voice_features,
        candidate_id="MOCK-001",
        position="Team Lead",
    )
    
    pipeline.print_summary(result)
    
    html = generate_html_report(result, transcript=mock_transcript)
    output_path = Path("outputs/mock_demo_report.html")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"\nHTML report saved to: {output_path}")
    
    return result


def demo_batch_processing():
    """Demo: Process multiple candidates."""
    print("=" * 60)
    print("Demo: Batch Processing")
    print("=" * 60)
    
    candidates = [
        {
            "audio_path": "examples/candidate_1.wav",
            "candidate_id": "C001",
            "position": "Frontend Developer",
        },
        {
            "audio_path": "examples/candidate_2.wav",
            "candidate_id": "C002",
            "position": "Frontend Developer",
        },
    ]
    
    config = load_config()
    pipeline = HRAssessmentPipeline(config)
    
    results = []
    for candidate in candidates:
        audio_path = Path(candidate["audio_path"])
        if not audio_path.exists():
            print(f"Skipping {candidate['candidate_id']}: audio file not found")
            continue
        
        result = pipeline.process(
            audio_path=audio_path,
            candidate_id=candidate["candidate_id"],
            position=candidate["position"],
        )
        results.append(result)
    
    if results:
        print("\n" + "=" * 60)
        print("Batch Results Summary")
        print("=" * 60)
        for result in results:
            print(f"\n{result.candidate_id}:")
            print(f"  Motivation: {result.motivation.overall_level}")
            print(f"  Extraversion: {result.big_five.extraversion.score}/100")
    else:
        print("No audio files found for batch processing demo.")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HR Assessment Pipeline Demo")
    parser.add_argument(
        "--mode",
        choices=["audio", "mock", "batch"],
        default="mock",
        help="Demo mode: audio (real file), mock (simulated data), batch (multiple files)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "audio":
        demo_with_audio_file()
    elif args.mode == "mock":
        demo_with_mock_data()
    elif args.mode == "batch":
        demo_batch_processing()
