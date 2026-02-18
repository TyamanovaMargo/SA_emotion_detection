"""
Generate aggregated JSON reports per person.

Each person gets a single JSON file containing all their recordings.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import statistics

from ..models import HRAssessmentResult


def generate_person_aggregated_json(
    person_results: Dict[str, List[Tuple[Path, HRAssessmentResult]]],
    output_dir: Path
) -> None:
    """
    Generate one JSON file per person containing all their recordings.
    
    Args:
        person_results: Dict mapping person name to list of (audio_path, result) tuples
        output_dir: Directory to save JSON files
    """
    for person_name, results_list in person_results.items():
        # Sanitize person name for filename
        safe_name = person_name.replace(' ', '_').replace('/', '_')
        json_path = output_dir / f"{safe_name}_aggregated.json"
        
        # Build aggregated data structure
        recordings = []
        for audio_path, result in results_list:
            # Serialize full result (excluding raw_response)
            result_data = result.model_dump(exclude={"raw_response"})
            
            recording_data = {
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
            }
            
            recordings.append(recording_data)
        
        # Calculate aggregate statistics
        motivation_scores = []
        engagement_scores = []
        speaking_rates = []
        
        for _, result in results_list:
            if hasattr(result, 'motivation') and result.motivation:
                motivation_scores.append(result.motivation.motivation_score)
            if hasattr(result, 'engagement') and result.engagement:
                engagement_scores.append(result.engagement.engagement_score)
            if hasattr(result, 'voice_features') and result.voice_features:
                rate = result.voice_features.prosody.speaking_rate_wpm
                if isinstance(rate, (int, float)) and rate > 0:
                    speaking_rates.append(rate)
        
        # Build statistics
        statistics_data = {
            'total_recordings': len(recordings),
            'motivation': {
                'mean': round(statistics.mean(motivation_scores), 1) if motivation_scores else 0,
                'stdev': round(statistics.stdev(motivation_scores), 1) if len(motivation_scores) > 1 else 0,
                'min': round(min(motivation_scores), 1) if motivation_scores else 0,
                'max': round(max(motivation_scores), 1) if motivation_scores else 0,
            },
            'engagement': {
                'mean': round(statistics.mean(engagement_scores), 1) if engagement_scores else 0,
                'stdev': round(statistics.stdev(engagement_scores), 1) if len(engagement_scores) > 1 else 0,
                'min': round(min(engagement_scores), 1) if engagement_scores else 0,
                'max': round(max(engagement_scores), 1) if engagement_scores else 0,
            },
            'speaking_rate': {
                'mean': round(statistics.mean(speaking_rates), 1) if speaking_rates else 0,
                'stdev': round(statistics.stdev(speaking_rates), 1) if len(speaking_rates) > 1 else 0,
                'min': round(min(speaking_rates), 1) if speaking_rates else 0,
                'max': round(max(speaking_rates), 1) if speaking_rates else 0,
            }
        }
        
        # Consistency assessment
        avg_stdev = 0
        if motivation_scores and engagement_scores and len(motivation_scores) > 1:
            avg_stdev = (statistics.stdev(motivation_scores) + statistics.stdev(engagement_scores)) / 2
        
        if avg_stdev < 5:
            consistency = 'Very Consistent'
        elif avg_stdev < 10:
            consistency = 'Consistent'
        elif avg_stdev < 20:
            consistency = 'Moderate Variance'
        else:
            consistency = 'High Variance'
        
        statistics_data['consistency'] = {
            'overall_stdev': round(avg_stdev, 1),
            'level': consistency
        }
        
        # Build final JSON structure
        aggregated_data = {
            'person': person_name,
            'generated': datetime.now().isoformat(),
            'statistics': statistics_data,
            'recordings': recordings
        }
        
        # Save to file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved aggregated JSON for {person_name}: {json_path}")
