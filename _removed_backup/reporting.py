"""Report generation utilities."""

from pathlib import Path
from typing import Optional
from datetime import datetime

from ..models.schemas import HRAssessmentResult


def generate_html_report(
    result: HRAssessmentResult,
    output_path: Optional[Path] = None,
    transcript: Optional[str] = None,
) -> str:
    """
    Generate an HTML report from assessment results.
    
    Args:
        result: HRAssessmentResult object
        output_path: Optional path to save the HTML file
        transcript: Optional transcript to include
        
    Returns:
        HTML string
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HR Assessment Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        .container {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .metadata {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .trait {{
            display: flex;
            align-items: center;
            margin: 10px 0;
        }}
        .trait-name {{
            width: 150px;
            font-weight: bold;
        }}
        .progress-bar {{
            flex: 1;
            height: 20px;
            background: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin: 0 10px;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            border-radius: 10px;
        }}
        .score {{
            width: 80px;
            text-align: right;
        }}
        .motivation-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
        }}
        .motivation-high {{ background: #27ae60; }}
        .motivation-medium {{ background: #f39c12; }}
        .motivation-low {{ background: #e74c3c; }}
        .strengths {{ color: #27ae60; }}
        .development {{ color: #e74c3c; }}
        .summary-box {{
            background: #3498db;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        ul {{ padding-left: 20px; }}
        li {{ margin: 8px 0; }}
        .transcript {{
            background: #fafafa;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin-top: 20px;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 HR Personality & Motivation Assessment</h1>
        
        <div class="metadata">
            <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
            {f'<strong>Candidate ID:</strong> {result.candidate_id}<br>' if result.candidate_id else ''}
            {f'<strong>Position:</strong> {result.position}<br>' if result.position else ''}
        </div>
        
        <h2>📊 Big Five Personality Profile</h2>
        {_generate_trait_html(result.big_five.openness, "Openness")}
        {_generate_trait_html(result.big_five.conscientiousness, "Conscientiousness")}
        {_generate_trait_html(result.big_five.extraversion, "Extraversion")}
        {_generate_trait_html(result.big_five.agreeableness, "Agreeableness")}
        {_generate_trait_html(result.big_five.neuroticism, "Neuroticism")}
        
        <h2>🔥 Motivation Assessment</h2>
        <p>
            <span class="motivation-badge motivation-{result.motivation.overall_level.lower()}">
                {result.motivation.overall_level}
            </span>
        </p>
        <p><strong>Pattern:</strong> {result.motivation.pattern}</p>
        
        <h3>Voice Indicators:</h3>
        <ul>
            {''.join(f'<li>{ind}</li>' for ind in result.motivation.voice_indicators)}
        </ul>
        
        <h3>Content Indicators:</h3>
        <ul>
            {''.join(f'<li>{ind}</li>' for ind in result.motivation.content_indicators)}
        </ul>
        
        <h2 class="strengths">✅ Key Strengths</h2>
        <h3>Trait Strengths:</h3>
        <ul>
            {''.join(f'<li>{s}</li>' for s in result.trait_strengths)}
        </ul>
        <h3>Motivation Strengths:</h3>
        <ul>
            {''.join(f'<li>{s}</li>' for s in result.motivation_strengths)}
        </ul>
        
        <h2 class="development">📈 Areas for Development</h2>
        <h3>Personality-Related:</h3>
        <ul>
            {''.join(f'<li>{a}</li>' for a in result.personality_development_areas)}
        </ul>
        <h3>Motivation-Related:</h3>
        <ul>
            {''.join(f'<li>{a}</li>' for a in result.motivation_development_areas)}
        </ul>
        
        <div class="summary-box">
            <h2 style="margin-top: 0; color: white;">📋 HR Summary</h2>
            <p>{result.hr_summary}</p>
        </div>
        
        {f'<h2>📝 Transcript</h2><div class="transcript">{transcript}</div>' if transcript else ''}
    </div>
</body>
</html>"""
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
    
    return html


def _generate_trait_html(score, name: str) -> str:
    """Generate HTML for a single trait."""
    return f"""
        <div class="trait">
            <span class="trait-name">{name}</span>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {score.score}%"></div>
            </div>
            <span class="score">{score.score}/100</span>
        </div>
        <p style="margin-left: 160px; font-size: 0.9em; color: #666;">
            {score.reason} (Confidence: {score.confidence}%)
        </p>
    """


def generate_pdf_report(
    result: HRAssessmentResult,
    output_path: Path,
    transcript: Optional[str] = None,
) -> Path:
    """
    Generate a PDF report from assessment results.
    
    Note: Requires weasyprint or similar library.
    Falls back to HTML if PDF generation is not available.
    """
    try:
        from weasyprint import HTML
        
        html_content = generate_html_report(result, transcript=transcript)
        pdf_path = output_path.with_suffix(".pdf")
        HTML(string=html_content).write_pdf(pdf_path)
        return pdf_path
        
    except ImportError:
        print("Warning: weasyprint not installed. Generating HTML instead.")
        html_path = output_path.with_suffix(".html")
        generate_html_report(result, output_path=html_path, transcript=transcript)
        return html_path
