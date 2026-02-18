"""
Generate aggregated JSON reports per person.

Each person gets a single JSON file containing all their recordings,
including emotion analysis (fused timeline, emotion summary, ablation).
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import statistics

from ..models import HRAssessmentResult


def _safe_stdev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _aggregate_emotion_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate emotion summaries across multiple recordings."""
    if not summaries:
        return {}

    def _avg(key):
        vals = [s.get(key, 0) for s in summaries if s.get(key) is not None]
        return round(statistics.mean(vals), 3) if vals else 0

    def _stdev(key):
        vals = [s.get(key, 0) for s in summaries if s.get(key) is not None]
        return round(_safe_stdev(vals), 3)

    # Dominant emotion across all recordings (majority vote)
    emo_counts: Counter = Counter()
    for s in summaries:
        emo = s.get("dominant_emotion")
        if emo:
            emo_counts[emo] += 1

    # Merge emotion distributions
    merged_dist: Dict[str, int] = defaultdict(int)
    for s in summaries:
        for emo, cnt in s.get("emotion_distribution", {}).items():
            merged_dist[emo] += cnt

    total_segs = sum(merged_dist.values()) or 1

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
    """Aggregate ablation deltas across multiple recordings."""
    if not comparisons:
        return {}

    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    deltas: Dict[str, List[float]] = {t: [] for t in traits}

    for cmp in comparisons:
        for ch in cmp.get("changes", []):
            t = ch.get("trait", "").lower()
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


def generate_person_aggregated_json(
    person_results: Dict[str, List[Tuple[Path, HRAssessmentResult]]],
    output_dir: Path
) -> None:
    """
    Generate one JSON file per person containing all their recordings,
    including emotion analysis, fused timeline, ablation, and aggregate stats.
    """
    for person_name, results_list in person_results.items():
        safe_name = person_name.replace(' ', '_').replace('/', '_')
        json_path = output_dir / f"{safe_name}_aggregated.json"

        recordings = []
        motivation_scores = []
        engagement_scores = []
        speaking_rates = []
        emotion_summaries = []
        ablation_results = []
        big5_scores: Dict[str, List[int]] = {t: [] for t in
            ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")}

        for audio_path, result in results_list:
            result_data = result.model_dump(exclude={"raw_response"})

            recording_data: Dict[str, Any] = {
                'filename': audio_path.name,
                'timestamp': datetime.now().isoformat(),
                'assessment': {
                    'big_five': result_data.get('big_five'),
                    'motivation': result_data.get('motivation'),
                    'engagement': result_data.get('engagement'),
                    'trait_strengths': result_data.get('trait_strengths'),
                    'motivation_strengths': result_data.get('motivation_strengths'),
                    'hr_summary': result_data.get('hr_summary'),
                },
                'voice_features': result_data.get('voice_features', {}),
                'emotion_summary': result_data.get('emotion_summary'),
                'llm_comparison': result_data.get('llm_comparison'),
                'dual_emotions': result_data.get('dual_emotions'),
                'emotion_timeline_rich': result_data.get('emotion_timeline_rich'),
            }
            recordings.append(recording_data)

            # Collect for aggregation
            if result.motivation:
                motivation_scores.append(result.motivation.motivation_score)
            if result.engagement:
                engagement_scores.append(result.engagement.engagement_score)
            if result.voice_features:
                rate = result.voice_features.prosody.speaking_rate_wpm
                if isinstance(rate, (int, float)) and rate > 0:
                    speaking_rates.append(rate)
            if result.emotion_summary:
                emotion_summaries.append(result.emotion_summary)
            if result.llm_comparison:
                ablation_results.append(result.llm_comparison)

            for t in big5_scores:
                score_obj = getattr(result.big_five, t, None)
                if score_obj:
                    big5_scores[t].append(score_obj.score)

        # Build statistics
        stats: Dict[str, Any] = {
            'total_recordings': len(recordings),
            'big_five': {},
            'motivation': {
                'mean': round(statistics.mean(motivation_scores), 1) if motivation_scores else 0,
                'stdev': round(_safe_stdev(motivation_scores), 1),
                'min': round(min(motivation_scores), 1) if motivation_scores else 0,
                'max': round(max(motivation_scores), 1) if motivation_scores else 0,
            },
            'engagement': {
                'mean': round(statistics.mean(engagement_scores), 1) if engagement_scores else 0,
                'stdev': round(_safe_stdev(engagement_scores), 1),
                'min': round(min(engagement_scores), 1) if engagement_scores else 0,
                'max': round(max(engagement_scores), 1) if engagement_scores else 0,
            },
            'speaking_rate': {
                'mean': round(statistics.mean(speaking_rates), 1) if speaking_rates else 0,
                'stdev': round(_safe_stdev(speaking_rates), 1),
                'min': round(min(speaking_rates), 1) if speaking_rates else 0,
                'max': round(max(speaking_rates), 1) if speaking_rates else 0,
            },
        }

        # Big Five per-trait stats
        for t, vals in big5_scores.items():
            if vals:
                stats['big_five'][t] = {
                    'mean': round(statistics.mean(vals), 1),
                    'stdev': round(_safe_stdev(vals), 1),
                    'min': min(vals),
                    'max': max(vals),
                    'values': vals,
                }

        # Consistency
        all_stdevs = [stats['big_five'][t]['stdev'] for t in big5_scores if t in stats['big_five']]
        all_stdevs.extend([stats['motivation']['stdev'], stats['engagement']['stdev']])
        avg_stdev = statistics.mean(all_stdevs) if all_stdevs else 0

        if avg_stdev < 5:
            consistency = 'Very Consistent'
        elif avg_stdev < 10:
            consistency = 'Consistent'
        elif avg_stdev < 20:
            consistency = 'Moderate Variance'
        else:
            consistency = 'High Variance'

        stats['consistency'] = {
            'overall_stdev': round(avg_stdev, 1),
            'level': consistency,
        }

        # Aggregated emotion summary
        agg_emotion = _aggregate_emotion_summaries(emotion_summaries)
        agg_ablation = _aggregate_ablation(ablation_results)

        aggregated_data = {
            'person': person_name,
            'generated': datetime.now().isoformat(),
            'statistics': stats,
            'emotion_aggregate': agg_emotion,
            'ablation_aggregate': agg_ablation,
            'recordings': recordings,
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated_data, f, indent=2, ensure_ascii=False)

        print(f"  Saved aggregated JSON for {person_name}: {json_path}")
