"""Cross-recording comparison report generator.

Groups assessment results by person (extracted from filename),
computes per-person statistics, variance across recordings,
and generates an HTML comparison report.
"""

import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..models.schemas import HRAssessmentResult


# ---------------------------------------------------------------------------
# Person name extraction
# ---------------------------------------------------------------------------

def extract_person_name(filename: str) -> str:
    """
    Extract person name from filename.

    Patterns handled:
    - Firstname_Lastname_audio_N.ext -> Firstname Lastname
    - Firstname_Question N Topic.ext -> Firstname
    - firstname_lastname_audio_N.ext -> Firstname Lastname
    """
    stem = Path(filename).stem

    # Pattern: Name_audio_N or Name_Name_audio_N
    match = re.match(r'^(.+?)_audio_\d+', stem, re.IGNORECASE)
    if match:
        raw = match.group(1)
        return raw.replace('_', ' ').title()

    # Pattern: Name_Question ... (e.g. Idan_Question 1 Astrology)
    match = re.match(r'^(.+?)_[Qq]uestion', stem)
    if match:
        raw = match.group(1)
        return raw.replace('_', ' ').title()

    # Fallback: everything before last underscore+digit
    match = re.match(r'^(.+?)_\d+$', stem)
    if match:
        return match.group(1).replace('_', ' ').title()

    return stem.replace('_', ' ').title()


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _safe_stdev(values: List[float]) -> float:
    """Standard deviation, returns 0 for < 2 values."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _consistency_label(stdev: float, scale: float = 100.0) -> str:
    """Convert stdev to a consistency label."""
    ratio = stdev / scale if scale > 0 else 0
    if ratio < 0.05:
        return "Very Consistent"
    elif ratio < 0.10:
        return "Consistent"
    elif ratio < 0.20:
        return "Moderate Variance"
    else:
        return "High Variance"


def _color_for_consistency(label: str) -> str:
    return {
        "Very Consistent": "#27ae60",
        "Consistent": "#2ecc71",
        "Moderate Variance": "#f39c12",
        "High Variance": "#e74c3c",
    }.get(label, "#95a5a6")


# ---------------------------------------------------------------------------
# Per-person analysis
# ---------------------------------------------------------------------------

def analyze_person(
    name: str,
    results: List[Tuple[Path, HRAssessmentResult]],
) -> Dict:
    """Compute per-person aggregate statistics."""
    n = len(results)

    # Big Five
    traits = {}
    for trait_name in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
        scores = [getattr(r.big_five, trait_name).score for _, r in results]
        traits[trait_name] = {
            "mean": round(statistics.mean(scores), 1),
            "stdev": round(_safe_stdev(scores), 1),
            "min": min(scores),
            "max": max(scores),
            "values": scores,
            "consistency": _consistency_label(_safe_stdev(scores)),
        }

    # Motivation
    mot_scores = [r.motivation.motivation_score for _, r in results]
    motivation = {
        "mean": round(statistics.mean(mot_scores), 1),
        "stdev": round(_safe_stdev(mot_scores), 1),
        "min": min(mot_scores),
        "max": max(mot_scores),
        "values": mot_scores,
        "consistency": _consistency_label(_safe_stdev(mot_scores)),
        "levels": [r.motivation.overall_level for _, r in results],
    }

    # Engagement
    eng_scores = [r.engagement.engagement_score for _, r in results]
    engagement = {
        "mean": round(statistics.mean(eng_scores), 1),
        "stdev": round(_safe_stdev(eng_scores), 1),
        "min": min(eng_scores),
        "max": max(eng_scores),
        "values": eng_scores,
        "consistency": _consistency_label(_safe_stdev(eng_scores)),
    }

    # Emotions (from voice_features if available)
    emotion_counts = defaultdict(int)
    for _, r in results:
        if r.voice_features and r.voice_features.emotions:
            emotion_counts[r.voice_features.emotions.primary_emotion] += 1

    # Per-recording details
    recordings = []
    for path, r in results:
        rec = {
            "filename": path.name,
            "big_five": {t: getattr(r.big_five, t).score for t in traits},
            "motivation_score": r.motivation.motivation_score,
            "motivation_level": r.motivation.overall_level,
            "engagement_score": r.engagement.engagement_score,
            "primary_emotion": r.voice_features.emotions.primary_emotion if r.voice_features and r.voice_features.emotions else "N/A",
        }
        recordings.append(rec)

    # Overall consistency score (average of all trait consistencies)
    all_stdevs = [traits[t]["stdev"] for t in traits]
    all_stdevs.append(motivation["stdev"])
    all_stdevs.append(engagement["stdev"])
    avg_stdev = statistics.mean(all_stdevs)
    overall_consistency = _consistency_label(avg_stdev)

    return {
        "name": name,
        "recording_count": n,
        "big_five": traits,
        "motivation": motivation,
        "engagement": engagement,
        "emotion_distribution": dict(emotion_counts),
        "recordings": recordings,
        "overall_consistency": overall_consistency,
        "avg_stdev": round(avg_stdev, 1),
    }


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def generate_comparison_json(
    person_analyses: Dict[str, Dict],
    output_path: Path,
):
    """Save comparison data as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated": datetime.now().isoformat(),
        "total_persons": len(person_analyses),
        "total_recordings": sum(p["recording_count"] for p in person_analyses.values()),
        "persons": person_analyses,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return output_path


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_comparison_html(
    person_analyses: Dict[str, Dict],
    output_path: Path,
) -> Path:
    """Generate a comprehensive HTML comparison report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    persons = list(person_analyses.values())
    persons.sort(key=lambda p: p["name"])

    # Build person cards
    person_cards = ""
    for p in persons:
        person_cards += _build_person_card(p)

    # Build cross-person comparison table
    comparison_table = _build_comparison_table(persons)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HR Assessment - Cross-Recording Comparison Report</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f0f2f5;
            color: #1a1a2e;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white;
            padding: 40px;
            border-radius: 16px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{ font-size: 2em; margin-bottom: 10px; }}
        .header .subtitle {{ opacity: 0.8; font-size: 1.1em; }}
        .stats-row {{
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .stat-card {{
            flex: 1;
            min-width: 200px;
            background: white;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .stat-card .number {{ font-size: 2.5em; font-weight: bold; color: #0f3460; }}
        .stat-card .label {{ color: #666; margin-top: 5px; }}
        .section-title {{
            font-size: 1.5em;
            color: #1a1a2e;
            margin: 30px 0 15px;
            padding-bottom: 8px;
            border-bottom: 3px solid #0f3460;
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 30px;
        }}
        .comparison-table th {{
            background: #1a1a2e;
            color: white;
            padding: 12px 16px;
            text-align: center;
            font-size: 0.85em;
        }}
        .comparison-table td {{
            padding: 10px 16px;
            text-align: center;
            border-bottom: 1px solid #eee;
        }}
        .comparison-table tr:hover {{ background: #f8f9fa; }}
        .comparison-table td:first-child {{ text-align: left; font-weight: bold; }}
        .person-card {{
            background: white;
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .person-card h2 {{
            color: #0f3460;
            margin-bottom: 5px;
        }}
        .person-card .meta {{ color: #888; margin-bottom: 15px; font-size: 0.9em; }}
        .consistency-badge {{
            display: inline-block;
            padding: 3px 12px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            color: white;
        }}
        .trait-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 12px;
            margin: 15px 0;
        }}
        .trait-item {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 12px;
        }}
        .trait-item .name {{ font-weight: bold; font-size: 0.9em; color: #333; }}
        .trait-item .bar-bg {{
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            margin: 6px 0;
            position: relative;
        }}
        .trait-item .bar-fill {{
            height: 100%;
            border-radius: 4px;
            background: linear-gradient(90deg, #3498db, #2ecc71);
        }}
        .trait-item .bar-range {{
            position: absolute;
            top: -3px;
            height: 14px;
            background: rgba(231, 76, 60, 0.3);
            border-radius: 4px;
        }}
        .trait-item .stats {{ font-size: 0.8em; color: #666; }}
        .recordings-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 0.85em;
        }}
        .recordings-table th {{
            background: #f0f2f5;
            padding: 8px 12px;
            text-align: center;
            font-weight: 600;
        }}
        .recordings-table td {{
            padding: 8px 12px;
            text-align: center;
            border-bottom: 1px solid #eee;
        }}
        .recordings-table td:first-child {{ text-align: left; }}
        .level-high {{ color: #27ae60; font-weight: bold; }}
        .level-medium {{ color: #f39c12; font-weight: bold; }}
        .level-low {{ color: #e74c3c; font-weight: bold; }}
        .emotion-tags {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }}
        .emotion-tag {{
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            background: #e8f4fd;
            color: #0f3460;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Cross-Recording Comparison Report</h1>
        <div class="subtitle">
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} |
            {len(persons)} persons | {sum(p['recording_count'] for p in persons)} recordings
        </div>
    </div>

    <div class="stats-row">
        <div class="stat-card">
            <div class="number">{len(persons)}</div>
            <div class="label">Persons Analyzed</div>
        </div>
        <div class="stat-card">
            <div class="number">{sum(p['recording_count'] for p in persons)}</div>
            <div class="label">Total Recordings</div>
        </div>
        <div class="stat-card">
            <div class="number">{sum(1 for p in persons if p['overall_consistency'] in ('Very Consistent', 'Consistent'))}</div>
            <div class="label">Consistent Profiles</div>
        </div>
        <div class="stat-card">
            <div class="number">{sum(1 for p in persons if p['motivation']['mean'] >= 50)}</div>
            <div class="label">Motivated (avg >= 50)</div>
        </div>
    </div>

    <h2 class="section-title">Overview: All Persons</h2>
    {comparison_table}

    <h2 class="section-title">Detailed Per-Person Analysis</h2>
    {person_cards}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


# ---------------------------------------------------------------------------
# HTML building helpers
# ---------------------------------------------------------------------------

def _build_comparison_table(persons: List[Dict]) -> str:
    """Build the cross-person comparison table."""
    rows = ""
    for p in persons:
        bf = p["big_five"]
        mot = p["motivation"]
        eng = p["engagement"]
        consistency_color = _color_for_consistency(p["overall_consistency"])

        rows += f"""<tr>
            <td>{p['name']}</td>
            <td>{p['recording_count']}</td>
            <td>{bf['openness']['mean']}</td>
            <td>{bf['conscientiousness']['mean']}</td>
            <td>{bf['extraversion']['mean']}</td>
            <td>{bf['agreeableness']['mean']}</td>
            <td>{bf['neuroticism']['mean']}</td>
            <td>{mot['mean']}</td>
            <td>{eng['mean']}</td>
            <td><span class="consistency-badge" style="background:{consistency_color}">{p['overall_consistency']}</span></td>
            <td>{p['avg_stdev']}</td>
        </tr>"""

    return f"""<table class="comparison-table">
        <thead>
            <tr>
                <th>Person</th>
                <th>Recs</th>
                <th>O</th>
                <th>C</th>
                <th>E</th>
                <th>A</th>
                <th>N</th>
                <th>Motiv.</th>
                <th>Engag.</th>
                <th>Consistency</th>
                <th>Avg StDev</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>"""


def _build_person_card(p: Dict) -> str:
    """Build a detailed card for one person."""
    consistency_color = _color_for_consistency(p["overall_consistency"])

    # Trait grid
    trait_items = ""
    for trait_name, data in p["big_five"].items():
        tc = _color_for_consistency(data["consistency"])
        range_left = data["min"]
        range_width = max(data["max"] - data["min"], 1)
        trait_items += f"""<div class="trait-item">
            <div class="name">{trait_name.capitalize()}
                <span class="consistency-badge" style="background:{tc}; font-size:0.75em;">{data['consistency']}</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill" style="width:{data['mean']}%"></div>
                <div class="bar-range" style="left:{range_left}%; width:{range_width}%"></div>
            </div>
            <div class="stats">
                Mean: {data['mean']} | StDev: {data['stdev']} | Range: {data['min']}-{data['max']}
            </div>
        </div>"""

    # Motivation + Engagement
    mot = p["motivation"]
    eng = p["engagement"]
    mot_tc = _color_for_consistency(mot["consistency"])
    eng_tc = _color_for_consistency(eng["consistency"])

    trait_items += f"""<div class="trait-item">
        <div class="name">Motivation
            <span class="consistency-badge" style="background:{mot_tc}; font-size:0.75em;">{mot['consistency']}</span>
        </div>
        <div class="bar-bg">
            <div class="bar-fill" style="width:{mot['mean']}%"></div>
            <div class="bar-range" style="left:{mot['min']}%; width:{max(mot['max']-mot['min'],1)}%"></div>
        </div>
        <div class="stats">
            Mean: {mot['mean']} | StDev: {mot['stdev']} | Range: {mot['min']}-{mot['max']}
        </div>
    </div>"""

    trait_items += f"""<div class="trait-item">
        <div class="name">Engagement
            <span class="consistency-badge" style="background:{eng_tc}; font-size:0.75em;">{eng['consistency']}</span>
        </div>
        <div class="bar-bg">
            <div class="bar-fill" style="width:{eng['mean']}%"></div>
            <div class="bar-range" style="left:{eng['min']}%; width:{max(eng['max']-eng['min'],1)}%"></div>
        </div>
        <div class="stats">
            Mean: {eng['mean']} | StDev: {eng['stdev']} | Range: {eng['min']}-{eng['max']}
        </div>
    </div>"""

    # Emotion tags
    emotion_tags = ""
    for emotion, count in sorted(p["emotion_distribution"].items(), key=lambda x: -x[1]):
        emotion_tags += f'<span class="emotion-tag">{emotion} ({count})</span>'

    # Per-recording table
    rec_rows = ""
    for rec in p["recordings"]:
        level_class = f"level-{rec['motivation_level'].lower()}"
        rec_rows += f"""<tr>
            <td>{rec['filename']}</td>
            <td>{rec['big_five']['openness']}</td>
            <td>{rec['big_five']['conscientiousness']}</td>
            <td>{rec['big_five']['extraversion']}</td>
            <td>{rec['big_five']['agreeableness']}</td>
            <td>{rec['big_five']['neuroticism']}</td>
            <td class="{level_class}">{rec['motivation_score']} ({rec['motivation_level']})</td>
            <td>{rec['engagement_score']}</td>
            <td>{rec['primary_emotion']}</td>
        </tr>"""

    return f"""<div class="person-card">
        <h2>{p['name']}
            <span class="consistency-badge" style="background:{consistency_color}; font-size:0.5em; vertical-align:middle;">
                {p['overall_consistency']}
            </span>
        </h2>
        <div class="meta">{p['recording_count']} recordings | Avg StDev: {p['avg_stdev']}</div>

        <div class="trait-grid">{trait_items}</div>

        <div class="emotion-tags">{emotion_tags}</div>

        <h3 style="margin-top:15px; font-size:0.95em;">Per-Recording Breakdown</h3>
        <table class="recordings-table">
            <thead>
                <tr>
                    <th>File</th><th>O</th><th>C</th><th>E</th><th>A</th><th>N</th>
                    <th>Motivation</th><th>Engagement</th><th>Emotion</th>
                </tr>
            </thead>
            <tbody>{rec_rows}</tbody>
        </table>
    </div>"""
