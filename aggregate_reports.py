#!/usr/bin/env python3
"""
Aggregate existing per-file assessment JSONs into per-person reports
+ feature impact report.

NO model loading — reads only JSON files from outputs/.

Usage:
    python aggregate_reports.py --outputs outputs
    python aggregate_reports.py --outputs outputs --person "Anastasiya Pavliukevich"
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Person name extraction (same logic as comparison_report.py)
# ---------------------------------------------------------------------------

def extract_person_name(filename: str) -> str:
    """Extract person name from filename like Name_Name_audio_N.ext"""
    stem = Path(filename).stem
    # Remove timestamp suffix like _20260218_142203_assessment
    stem = re.sub(r'_\d{8}_\d{6}_assessment$', '', stem)
    m = re.match(r'^(.+?)_audio_\d+', stem, re.IGNORECASE)
    if m:
        return m.group(1).replace('_', ' ').title()
    m2 = re.match(r'^(.+?)_[Qq]uestion', stem)
    if m2:
        return m2.group(1).replace('_', ' ').title()
    return stem.replace('_', ' ').title()


# ---------------------------------------------------------------------------
# Safe math
# ---------------------------------------------------------------------------

def _safe_stdev(vals: List[float]) -> float:
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0


# ---------------------------------------------------------------------------
# Aggregate emotion summaries across recordings
# ---------------------------------------------------------------------------

def _aggregate_emotion_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not summaries:
        return {}

    def _avg(key):
        vals = [s.get(key, 0) for s in summaries if s.get(key) is not None]
        return round(statistics.mean(vals), 3) if vals else 0

    def _stdev(key):
        vals = [s.get(key, 0) for s in summaries if s.get(key) is not None]
        return round(_safe_stdev(vals), 3)

    emo_counts: Counter = Counter()
    for s in summaries:
        emo = s.get("dominant_emotion")
        if emo:
            emo_counts[emo] += 1

    merged_dist: Dict[str, int] = defaultdict(int)
    for s in summaries:
        for emo, cnt in s.get("emotion_distribution", {}).items():
            merged_dist[emo] += cnt

    return {
        "recordings_count": len(summaries),
        "dominant_emotion": emo_counts.most_common(1)[0][0] if emo_counts else "neutral",
        "dominant_emotion_votes": dict(emo_counts.most_common()),
        "emotion_distribution_total": dict(merged_dist),
        "emotion_volatility_mean": _avg("emotion_volatility"),
        "emotion_volatility_std": _stdev("emotion_volatility"),
        "valence_mean": _avg("valence_mean"),
        "valence_std_across": _stdev("valence_mean"),
        "arousal_mean": _avg("arousal_mean"),
        "arousal_std_across": _stdev("arousal_mean"),
        "neutral_ratio_mean": _avg("neutral_ratio"),
        "avg_confidence_mean": _avg("avg_confidence"),
        "avg_entropy_mean": _avg("avg_entropy"),
        "model_agreement_rate_mean": _avg("model_agreement_rate"),
        "low_confidence_ratio_mean": _avg("low_confidence_ratio"),
    }


def _aggregate_ablation(comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not comparisons:
        return {}

    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    deltas: Dict[str, List[float]] = {t: [] for t in traits}

    for cmp in comparisons:
        for ch in cmp.get("changes", []):
            t = (ch.get("trait") or "").lower()
            d = ch.get("delta", 0)
            if t in deltas:
                deltas[t].append(d)

    agg = {}
    for t in traits:
        vals = deltas[t]
        if vals:
            agg[t] = {
                "mean_delta": round(statistics.mean(vals), 1),
                "max_delta": max(vals, key=abs),
                "n_changed": len(vals),
                "n_total": len(comparisons),
            }
    return agg


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def aggregate(outputs_dir: Path, filter_person: str = None):
    print(f"Scanning {outputs_dir} for *_assessment.json ...")

    files = sorted(outputs_dir.glob("*_assessment.json"))
    if not files:
        print(f"ERROR: No *_assessment.json files found in {outputs_dir}")
        sys.exit(1)

    print(f"Found {len(files)} assessment files\n")

    # Group by person
    person_files: Dict[str, List[Tuple[Path, dict]]] = defaultdict(list)
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")
            continue
        person = extract_person_name(f.name)
        if filter_person and person.lower() != filter_person.lower():
            continue
        person_files[person].append((f, data))

    if not person_files:
        print("No matching files found.")
        sys.exit(1)

    print(f"Persons: {len(person_files)}")
    for name, items in sorted(person_files.items()):
        print(f"  {name}: {len(items)} recordings")
    print()

    TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

    # Generate per-person aggregated JSON
    for person_name, items in sorted(person_files.items()):
        safe_name = person_name.replace(' ', '_').replace('/', '_')
        json_path = outputs_dir / f"{safe_name}_aggregated.json"

        recordings = []
        motivation_scores = []
        engagement_scores = []
        speaking_rates = []
        emotion_summaries = []
        ablation_results = []
        big5_scores: Dict[str, List[float]] = {t: [] for t in TRAITS}

        for f_path, data in items:
            recording_data = {
                "filename": f_path.name,
                "assessment": {
                    "big_five": data.get("personality_assessment", {}).get("big_five"),
                    "motivation": data.get("motivation_engagement", {}).get("motivation"),
                    "engagement": data.get("motivation_engagement", {}).get("engagement"),
                    "hr_summary": data.get("hr_summary"),
                },
                "voice_features": data.get("voice_analysis"),
                "emotion_summary": data.get("emotion_summary"),
                "llm_comparison": data.get("llm_comparison"),
                "emotion_analysis": data.get("emotion_analysis"),
            }
            recordings.append(recording_data)

            # Collect scores
            mot = data.get("motivation_engagement", {}).get("motivation", {})
            eng = data.get("motivation_engagement", {}).get("engagement", {})
            if isinstance(mot, dict) and mot.get("motivation_score"):
                motivation_scores.append(mot["motivation_score"])
            if isinstance(eng, dict) and eng.get("engagement_score"):
                engagement_scores.append(eng["engagement_score"])

            prosody = data.get("voice_analysis", {}).get("prosody", {})
            rate = prosody.get("speaking_rate_wpm", 0)
            if isinstance(rate, (int, float)) and rate > 0:
                speaking_rates.append(rate)

            emo = data.get("emotion_summary")
            if emo:
                emotion_summaries.append(emo)

            cmp = data.get("llm_comparison")
            if cmp:
                ablation_results.append(cmp)

            big5 = data.get("personality_assessment", {}).get("big_five", {})
            for t in TRAITS:
                score = (big5.get(t) or {}).get("score")
                if isinstance(score, (int, float)):
                    big5_scores[t].append(score)

        # Build statistics
        stats: Dict[str, Any] = {
            "total_recordings": len(recordings),
            "big_five": {},
            "motivation": {
                "mean": round(statistics.mean(motivation_scores), 1) if motivation_scores else 0,
                "stdev": round(_safe_stdev(motivation_scores), 1),
                "min": round(min(motivation_scores), 1) if motivation_scores else 0,
                "max": round(max(motivation_scores), 1) if motivation_scores else 0,
            },
            "engagement": {
                "mean": round(statistics.mean(engagement_scores), 1) if engagement_scores else 0,
                "stdev": round(_safe_stdev(engagement_scores), 1),
                "min": round(min(engagement_scores), 1) if engagement_scores else 0,
                "max": round(max(engagement_scores), 1) if engagement_scores else 0,
            },
            "speaking_rate": {
                "mean": round(statistics.mean(speaking_rates), 1) if speaking_rates else 0,
                "stdev": round(_safe_stdev(speaking_rates), 1),
                "min": round(min(speaking_rates), 1) if speaking_rates else 0,
                "max": round(max(speaking_rates), 1) if speaking_rates else 0,
            },
        }

        for t, vals in big5_scores.items():
            if vals:
                stats["big_five"][t] = {
                    "mean": round(statistics.mean(vals), 1),
                    "stdev": round(_safe_stdev(vals), 1),
                    "min": min(vals),
                    "max": max(vals),
                    "values": vals,
                }

        all_stdevs = [stats["big_five"][t]["stdev"] for t in TRAITS if t in stats["big_five"]]
        all_stdevs.extend([stats["motivation"]["stdev"], stats["engagement"]["stdev"]])
        avg_stdev = statistics.mean(all_stdevs) if all_stdevs else 0

        if avg_stdev < 5:
            consistency = "Very Consistent"
        elif avg_stdev < 10:
            consistency = "Consistent"
        elif avg_stdev < 20:
            consistency = "Moderate Variance"
        else:
            consistency = "High Variance"

        stats["consistency"] = {"overall_stdev": round(avg_stdev, 1), "level": consistency}

        agg_emotion = _aggregate_emotion_summaries(emotion_summaries)
        agg_ablation = _aggregate_ablation(ablation_results)

        aggregated = {
            "person": person_name,
            "generated": datetime.now().isoformat(),
            "statistics": stats,
            "emotion_aggregate": agg_emotion,
            "ablation_aggregate": agg_ablation,
            "recordings": recordings,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
        print(f"✅ {person_name}: {json_path}")

    # Feature impact report
    print("\nGenerating feature impact report...")
    try:
        # Import standalone (no heavy deps)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "feature_impact",
            str(Path(__file__).parent / "src" / "utils" / "feature_impact.py")
        )
        fi = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fi)
        report = fi.generate_feature_impact(outputs_dir)
        n = report.get("records_count", 0)
        if n > 0:
            print(f"✅ Feature impact: {n} records")
            print(f"   JSON: {outputs_dir / 'feature_impact_report.json'}")
            print(f"   CSV:  {outputs_dir / 'feature_impact_summary.csv'}")
            print(f"   HTML: {outputs_dir / 'feature_impact_report.html'}")
        else:
            print("⚠️  No ablation data found in assessment files")
    except Exception as e:
        print(f"⚠️  Feature impact generation failed: {e}")

    print(f"\nDone! Output: {outputs_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate per-file assessment JSONs into per-person reports (NO model loading)"
    )
    parser.add_argument(
        "--outputs", "-o",
        type=Path,
        default=Path("outputs"),
        help="Directory with *_assessment.json files (default: outputs)",
    )
    parser.add_argument(
        "--person",
        type=str,
        default=None,
        help="Filter: aggregate only this person (e.g. 'Anastasiya Pavliukevich')",
    )
    args = parser.parse_args()

    if not args.outputs.exists():
        print(f"ERROR: {args.outputs} does not exist")
        sys.exit(1)

    aggregate(args.outputs, filter_person=args.person)
