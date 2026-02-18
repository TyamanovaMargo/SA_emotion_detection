"""
Feature Impact Report — how emotion features improve Big Five trait assessment.

Collects baseline vs enriched (ablation) results from per-file assessment JSONs,
computes per-trait deltas, top cited emotion metrics, and correlations between
emotion summary signals and trait score changes.

Outputs: JSON, CSV, and a styled HTML report.
"""

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
EMO_NUMERIC_KEYS = [
    "emotion_volatility", "valence_mean", "valence_std", "arousal_mean", "arousal_std",
    "neutral_ratio", "model_agreement_rate", "avg_confidence", "avg_entropy",
    "low_confidence_ratio", "dominant_emotion_ratio",
]


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _pearson(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Pearson r with approximate p-value (t-distribution approximation)."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx == 0 or dy == 0:
        return 0.0, 1.0
    r = num / (dx * dy)
    r = max(-1.0, min(1.0, r))
    if abs(r) >= 0.9999:
        return round(r, 4), 0.0
    t_stat = abs(r) * math.sqrt((n - 2) / (1 - r * r))
    # Approximate two-tailed p from t using Abramowitz & Stegun 26.2.17
    df = n - 2
    p = (1 + t_stat / math.sqrt(df)) ** (-df) if df > 0 else 1.0
    p = min(max(p * 2, 0.0), 1.0)  # two-tailed
    return round(r, 4), round(p, 4)


def _safe_stdev(vals: List[float]) -> float:
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0


def _parse_metric_key(token: str) -> str:
    """Extract metric name from 'metric=value' token."""
    if "=" in token:
        return token.split("=", 1)[0].strip()
    return token.strip()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_ablation_records(outputs_dir: Path) -> List[Dict[str, Any]]:
    """
    Scan all *_assessment.json files and extract ablation + emotion_summary data.
    Returns list of per-file records.
    """
    records = []
    for f in sorted(outputs_dir.glob("*_assessment.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        cmp = data.get("llm_comparison")
        if not cmp:
            continue

        emo = data.get("emotion_summary") or {}
        baseline = cmp.get("baseline_big5", {})
        enriched = cmp.get("enriched_big5", {})
        changes = cmp.get("changes", [])

        # Extract person name from filename (Name_Name_audio_N pattern)
        import re as _re
        stem = f.stem
        m = _re.match(r'^(.+?)_audio_\d+', stem, _re.IGNORECASE)
        if m:
            person = m.group(1).replace('_', ' ').title()
        else:
            # Fallback: Name_Question pattern or raw stem
            m2 = _re.match(r'^(.+?)_[Qq]uestion', stem)
            person = m2.group(1).replace('_', ' ').title() if m2 else stem.rsplit("_", 2)[0].replace('_', ' ').title()

        records.append({
            "file": f.name,
            "person": person,
            "emotion_summary": emo,
            "baseline": baseline,
            "enriched": enriched,
            "changes": changes,
            "impact_summary": cmp.get("emotion_impact_summary", ""),
        })

    return records


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_feature_impact_report(outputs_dir: Path) -> Dict[str, Any]:
    """Build the full feature impact report from assessment JSONs."""
    records = collect_ablation_records(outputs_dir)

    if not records:
        return {"error": "No assessment files with ablation data found", "records_count": 0}

    # --- 1. Per-trait delta statistics ---
    trait_deltas: Dict[str, List[float]] = {t: [] for t in TRAITS}
    trait_baseline: Dict[str, List[float]] = {t: [] for t in TRAITS}
    trait_enriched: Dict[str, List[float]] = {t: [] for t in TRAITS}
    cited_metrics: Dict[str, List[str]] = {t: [] for t in TRAITS}

    for rec in records:
        base = rec["baseline"]
        enr = rec["enriched"]
        for t in TRAITS:
            b = (base.get(t) or {}).get("score")
            e = (enr.get(t) or {}).get("score")
            if isinstance(b, (int, float)) and isinstance(e, (int, float)):
                trait_deltas[t].append(e - b)
                trait_baseline[t].append(b)
                trait_enriched[t].append(e)

        for ch in rec.get("changes", []):
            t_name = (ch.get("trait") or "").lower()
            if t_name in cited_metrics:
                for tok in ch.get("emotion_metrics_cited", []):
                    cited_metrics[t_name].append(tok)

    trait_stats = {}
    for t in TRAITS:
        arr = trait_deltas[t]
        if not arr:
            trait_stats[t] = {"n": 0}
            continue
        abs_arr = [abs(v) for v in arr]
        trait_stats[t] = {
            "n": len(arr),
            "mean_delta": round(statistics.mean(arr), 2),
            "median_delta": round(statistics.median(arr), 2),
            "mean_abs_delta": round(statistics.mean(abs_arr), 2),
            "stdev_delta": round(_safe_stdev(arr), 2),
            "changed_ratio": round(sum(1 for v in arr if v != 0) / len(arr), 3),
            "positive_ratio": round(sum(1 for v in arr if v > 0) / len(arr), 3),
            "negative_ratio": round(sum(1 for v in arr if v < 0) / len(arr), 3),
            "min_delta": min(arr),
            "max_delta": max(arr),
            "baseline_mean": round(statistics.mean(trait_baseline[t]), 1),
            "enriched_mean": round(statistics.mean(trait_enriched[t]), 1),
        }

    # --- 2. Top cited emotion metrics per trait ---
    top_cited = {}
    for t in TRAITS:
        freq: Counter = Counter()
        for tok in cited_metrics[t]:
            key = _parse_metric_key(tok)
            if key:
                freq[key] += 1
        top_cited[t] = [{"metric": k, "count": c} for k, c in freq.most_common(10)]

    # Global top cited
    all_cited: Counter = Counter()
    for t in TRAITS:
        for tok in cited_metrics[t]:
            key = _parse_metric_key(tok)
            if key:
                all_cited[key] += 1
    global_top_cited = [{"metric": k, "count": c} for k, c in all_cited.most_common(15)]

    # --- 3. Correlations: delta_trait vs emotion_summary metrics ---
    correlations = {}
    for t in TRAITS:
        correlations[t] = {}
        for mk in EMO_NUMERIC_KEYS:
            xs, ys = [], []
            for rec in records:
                base = rec["baseline"]
                enr = rec["enriched"]
                emo = rec["emotion_summary"] or {}
                b = (base.get(t) or {}).get("score")
                e = (enr.get(t) or {}).get("score")
                mv = emo.get(mk)
                if isinstance(b, (int, float)) and isinstance(e, (int, float)) and isinstance(mv, (int, float)):
                    xs.append(e - b)
                    ys.append(mv)
            if len(xs) >= 3:
                r_val, p_val = _pearson(xs, ys)
                correlations[t][mk] = {"r": r_val, "p": p_val, "n": len(xs)}

    # --- 4. Per-person summary ---
    person_data: Dict[str, List[Dict]] = defaultdict(list)
    for rec in records:
        person_data[rec["person"]].append(rec)

    person_summaries = {}
    for person, recs in sorted(person_data.items()):
        p_deltas = {t: [] for t in TRAITS}
        for rec in recs:
            for t in TRAITS:
                b = (rec["baseline"].get(t) or {}).get("score")
                e = (rec["enriched"].get(t) or {}).get("score")
                if isinstance(b, (int, float)) and isinstance(e, (int, float)):
                    p_deltas[t].append(e - b)
        p_emo = [r["emotion_summary"] for r in recs if r.get("emotion_summary")]
        person_summaries[person] = {
            "recordings": len(recs),
            "trait_deltas": {t: round(statistics.mean(v), 1) if v else 0 for t, v in p_deltas.items()},
            "dominant_emotion": Counter(
                e.get("dominant_emotion") for e in p_emo if e.get("dominant_emotion")
            ).most_common(1)[0][0] if p_emo else "N/A",
            "avg_volatility": round(statistics.mean([e.get("emotion_volatility", 0) for e in p_emo]), 2) if p_emo else 0,
            "avg_valence": round(statistics.mean([e.get("valence_mean", 0) for e in p_emo]), 2) if p_emo else 0,
        }

    return {
        "generated": datetime.now().isoformat(),
        "records_count": len(records),
        "persons_count": len(person_summaries),
        "trait_stats": trait_stats,
        "top_cited_metrics_global": global_top_cited,
        "top_cited_metrics_per_trait": top_cited,
        "correlations": correlations,
        "person_summaries": person_summaries,
    }


# ---------------------------------------------------------------------------
# Output generators
# ---------------------------------------------------------------------------

def save_report_json(report: Dict[str, Any], path: Path):
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def save_report_csv(report: Dict[str, Any], path: Path):
    """Save per-person summary as CSV."""
    persons = report.get("person_summaries", {})
    if not persons:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "person", "recordings", "dominant_emotion", "avg_volatility", "avg_valence",
            "delta_O", "delta_C", "delta_E", "delta_A", "delta_N",
        ])
        for name, p in sorted(persons.items()):
            d = p["trait_deltas"]
            w.writerow([
                name, p["recordings"], p["dominant_emotion"],
                p["avg_volatility"], p["avg_valence"],
                d.get("openness", 0), d.get("conscientiousness", 0),
                d.get("extraversion", 0), d.get("agreeableness", 0),
                d.get("neuroticism", 0),
            ])


def save_report_html(report: Dict[str, Any], path: Path):
    """Generate a styled HTML feature impact report."""
    ts = report.get("trait_stats", {})
    persons = report.get("person_summaries", {})
    corr = report.get("correlations", {})
    global_cited = report.get("top_cited_metrics_global", [])
    per_trait_cited = report.get("top_cited_metrics_per_trait", {})

    # --- Trait delta summary table rows ---
    trait_rows = ""
    for t in TRAITS:
        s = ts.get(t, {})
        if not s.get("n"):
            continue
        delta_color = "#27ae60" if s["mean_delta"] >= 0 else "#e74c3c"
        bar_width = min(abs(s["mean_abs_delta"]) * 3, 100)
        trait_rows += f"""<tr>
            <td style="font-weight:bold">{t.capitalize()}</td>
            <td>{s['n']}</td>
            <td>{s['baseline_mean']}</td>
            <td>{s['enriched_mean']}</td>
            <td style="color:{delta_color};font-weight:bold">{s['mean_delta']:+.1f}</td>
            <td>{s['median_delta']:+.1f}</td>
            <td>{s['mean_abs_delta']:.1f}</td>
            <td>{int(s['changed_ratio']*100)}%</td>
            <td>
                <div style="background:#e0e0e0;border-radius:4px;height:12px;width:100px;display:inline-block">
                    <div style="background:{delta_color};height:100%;border-radius:4px;width:{bar_width}px"></div>
                </div>
            </td>
        </tr>"""

    # --- Person summary table rows ---
    person_rows = ""
    for name, p in sorted(persons.items()):
        d = p["trait_deltas"]
        cells = ""
        for t in TRAITS:
            v = d.get(t, 0)
            c = "#27ae60" if v > 0 else "#e74c3c" if v < 0 else "#666"
            cells += f'<td style="color:{c};font-weight:bold">{v:+.0f}</td>'
        person_rows += f"""<tr>
            <td>{name}</td>
            <td>{p['recordings']}</td>
            <td>{p['dominant_emotion']}</td>
            <td>{p['avg_volatility']:.2f}</td>
            <td>{p['avg_valence']:.2f}</td>
            {cells}
        </tr>"""

    # --- Top cited metrics ---
    cited_html = ""
    for item in global_cited[:10]:
        w = min(item["count"] * 15, 200)
        cited_html += f"""<div style="margin:4px 0">
            <span style="display:inline-block;width:200px;font-size:0.9em">{item['metric']}</span>
            <div style="display:inline-block;background:#3b82f6;height:16px;border-radius:3px;width:{w}px;vertical-align:middle"></div>
            <span style="font-size:0.85em;color:#666"> {item['count']}</span>
        </div>"""

    # --- Correlation highlights (|r| > 0.3) ---
    corr_rows = ""
    highlights = []
    for t in TRAITS:
        for mk, d in corr.get(t, {}).items():
            if abs(d["r"]) >= 0.25:
                highlights.append((t, mk, d["r"], d["p"], d["n"]))
    highlights.sort(key=lambda x: -abs(x[2]))
    for t, mk, r, p, n in highlights[:20]:
        r_color = "#27ae60" if r > 0 else "#e74c3c"
        corr_rows += f"""<tr>
            <td>{t.capitalize()}</td>
            <td>{mk}</td>
            <td style="color:{r_color};font-weight:bold">{r:+.3f}</td>
            <td>{p:.4f}</td>
            <td>{n}</td>
        </tr>"""

    # Pre-compute stats for the template
    traits_changed_50 = sum(1 for t in TRAITS if ts.get(t, {}).get("changed_ratio", 0) > 0.5)
    n_sig_corr = len(highlights)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Feature Impact Report — Emotion Features vs Big Five</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f0f2f5; color: #1a1a2e; padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white; padding: 40px; border-radius: 16px; margin-bottom: 30px; text-align: center;
        }}
        .header h1 {{ font-size: 2em; margin-bottom: 10px; }}
        .header .sub {{ opacity: 0.8; font-size: 1.1em; }}
        .stats-row {{ display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap; }}
        .stat-card {{
            flex: 1; min-width: 180px; background: white; border-radius: 12px;
            padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .stat-card .num {{ font-size: 2.2em; font-weight: bold; color: #0f3460; }}
        .stat-card .lbl {{ color: #666; margin-top: 5px; }}
        .section {{ background: white; border-radius: 12px; padding: 25px; margin-bottom: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .section h2 {{ color: #0f3460; margin-bottom: 15px; font-size: 1.3em; border-bottom: 3px solid #0f3460; padding-bottom: 8px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
        th {{ background: #1a1a2e; color: white; padding: 10px 12px; text-align: center; }}
        td {{ padding: 8px 12px; text-align: center; border-bottom: 1px solid #eee; }}
        td:first-child {{ text-align: left; }}
        tr:hover {{ background: #f8f9fa; }}
        .note {{ font-size: 0.85em; color: #888; margin-top: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Feature Impact Report</h1>
        <div class="sub">How Emotion Features Improve Big Five Personality Assessment</div>
        <div class="sub" style="margin-top:8px;font-size:0.9em">
            Generated: {report.get('generated', '')} |
            {report['records_count']} recordings | {report['persons_count']} persons
        </div>
    </div>

    <div class="stats-row">
        <div class="stat-card">
            <div class="num">{report['records_count']}</div>
            <div class="lbl">Recordings Analyzed</div>
        </div>
        <div class="stat-card">
            <div class="num">{report['persons_count']}</div>
            <div class="lbl">Persons</div>
        </div>
        <div class="stat-card">
            <div class="num">{traits_changed_50}</div>
            <div class="lbl">Traits Changed &gt;50%</div>
        </div>
        <div class="stat-card">
            <div class="num">{n_sig_corr}</div>
            <div class="lbl">Significant Correlations</div>
        </div>
    </div>

    <div class="section">
        <h2>1. Trait Delta Summary (Baseline vs Enriched)</h2>
        <p class="note">Baseline = LLM assessment without emotion data. Enriched = with fused emotion summary injected into prompt.</p>
        <table>
            <tr><th>Trait</th><th>N</th><th>Baseline Mean</th><th>Enriched Mean</th><th>Mean Delta</th><th>Median</th><th>Mean |Delta|</th><th>Changed %</th><th>Impact</th></tr>
            {trait_rows}
        </table>
    </div>

    <div class="section">
        <h2>2. Most Cited Emotion Metrics (by LLM)</h2>
        <p class="note">When the LLM changed a trait score, it cited these emotion metrics as reasons.</p>
        {cited_html}
    </div>

    <div class="section">
        <h2>3. Correlations: Trait Deltas vs Emotion Signals</h2>
        <p class="note">Pearson r between (enriched−baseline) delta and emotion summary metric. Only |r| &ge; 0.25 shown.</p>
        <table>
            <tr><th>Trait</th><th>Emotion Metric</th><th>r</th><th>p (approx)</th><th>N</th></tr>
            {corr_rows if corr_rows else '<tr><td colspan="5">Not enough data for correlations (need &ge;3 records)</td></tr>'}
        </table>
    </div>

    <div class="section">
        <h2>4. Per-Person Summary</h2>
        <p class="note">Average trait deltas per person. Positive = enriched score higher than baseline.</p>
        <table>
            <tr><th>Person</th><th>Recs</th><th>Emotion</th><th>Volatility</th><th>Valence</th>
                <th>&Delta;O</th><th>&Delta;C</th><th>&Delta;E</th><th>&Delta;A</th><th>&Delta;N</th></tr>
            {person_rows}
        </table>
    </div>

    <div class="section">
        <h2>5. Interpretation Guide</h2>
        <ul style="padding-left:20px;line-height:1.8">
            <li><strong>Mean Delta &ne; 0</strong> &rarr; emotion features systematically shift trait scores.</li>
            <li><strong>Changed % &gt; 50%</strong> &rarr; emotion data frequently influences that trait.</li>
            <li><strong>High |r| correlation</strong> &rarr; specific emotion metrics reliably predict score changes.</li>
            <li><strong>Neuroticism typically &uarr;</strong> when volatility is high and valence is negative.</li>
            <li><strong>Extraversion typically &darr;</strong> when arousal and valence are low.</li>
            <li><strong>Top cited metrics</strong> show which emotion signals the LLM finds most diagnostic.</li>
        </ul>
    </div>
</body>
</html>"""

    path.write_text(html, encoding="utf-8")


def generate_feature_impact(outputs_dir: Path) -> Dict[str, Any]:
    """Main entry: build report and save all formats."""
    report = build_feature_impact_report(outputs_dir)
    if report.get("error"):
        print(f"  [warn] {report['error']}")
        return report

    save_report_json(report, outputs_dir / "feature_impact_report.json")
    save_report_csv(report, outputs_dir / "feature_impact_summary.csv")
    save_report_html(report, outputs_dir / "feature_impact_report.html")
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate feature impact report from assessment JSONs")
    p.add_argument("--outputs", type=str, default="outputs", help="Directory with *_assessment.json files")
    args = p.parse_args()
    out = Path(args.outputs)
    rep = generate_feature_impact(out)
    n = rep.get("records_count", 0)
    print(f"Feature impact report: {n} records -> {out}/feature_impact_report.*")
