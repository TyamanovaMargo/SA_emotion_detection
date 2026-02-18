"""
Streamlit dashboard for voice-based personality & motivation analysis.

Usage:
    streamlit run src/utils/dashboard.py -- ./outputs
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_json_files(output_dir: Path) -> List[Path]:
    """Find all assessment JSON files in output directory."""
    files = sorted(output_dir.glob("*_assessment.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def load_assessment(path: Path) -> Dict[str, Any]:
    """Load a single assessment JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dashboard sections
# ---------------------------------------------------------------------------

def render_voice_features_table(granular: Dict[str, Any]):
    """Section a) — Voice Features Table."""
    st.subheader("Voice Features (Granular)")

    # Split into categories for readability
    categories = {
        "Prosody — Pitch": [k for k in granular if k.startswith("pitch")],
        "Prosody — Energy": [k for k in granular if k.startswith("energy") or k.startswith("dynamic")],
        "Prosody — Pauses & Fluency": [k for k in granular if "pause" in k or "speech" in k or "articulation" in k or "rhythm" in k or "speaking" in k],
        "Voice Quality (Parselmouth)": [k for k in granular if k.startswith(("jitter", "shimmer", "hnr"))],
        "Spectral Features": [k for k in granular if k.startswith("spectral")],
        "Voiced/Unvoiced": [k for k in granular if "voiced" in k],
        "Derived Scores": [k for k in granular if "proxy" in k],
    }

    # Collect keys already shown
    shown = set()
    for cat_name, keys in categories.items():
        if not keys:
            continue
        with st.expander(cat_name, expanded=True):
            rows = []
            for k in sorted(keys):
                rows.append({"Feature": k, "Value": granular[k]})
                shown.add(k)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Remaining features
    remaining = {k: v for k, v in granular.items() if k not in shown}
    if remaining:
        with st.expander("Other Features"):
            rows = [{"Feature": k, "Value": v} for k, v in sorted(remaining.items())]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_big5_approximate(approx: Dict[str, Any]):
    """Section b) — Big Five Approximate Indicators."""
    st.subheader("Big Five Personality — Approximate Indicators")
    st.caption("⚠️ Approximate indicators based on voice analysis. These are NOT clinical scores.")

    big5 = approx.get("big5_approximate", {})
    label_to_val = {"low": 0.2, "moderate-low": 0.35, "moderate": 0.5, "moderate-high": 0.65, "high": 0.8}

    trait_names = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

    for trait in trait_names:
        td = big5.get(trait, {})
        label = td.get("label", "moderate")
        score_range = td.get("score_range", "40–60")
        features = td.get("influencing_features", [])
        bar_val = label_to_val.get(label, 0.5)

        col1, col2 = st.columns([3, 2])
        with col1:
            fig = go.Figure(go.Bar(
                x=[bar_val], y=[trait.capitalize()],
                orientation="h",
                marker_color=px.colors.qualitative.Set2[trait_names.index(trait) % 8],
                text=[f"~{score_range}  ({label})"],
                textposition="inside",
                width=0.5,
            ))
            fig.update_layout(
                xaxis=dict(range=[0, 1], showticklabels=False),
                yaxis=dict(showticklabels=True),
                height=70, margin=dict(l=0, r=10, t=0, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"big5_{trait}")
        with col2:
            if features:
                st.markdown("**Influencing features:**")
                for f in features[:4]:
                    st.markdown(f"- `{f}`")


def render_motivation_engagement(approx: Dict[str, Any]):
    """Section c) — Motivation & Engagement Approximate Scores."""
    st.subheader("Motivation & Engagement — Approximate")
    st.caption("⚠️ Approximate indicators based on voice analysis.")

    label_to_color = {"low": "#ef4444", "moderate": "#f59e0b", "high": "#22c55e"}

    for key, title in [("motivation_approximate", "Motivation"), ("engagement_approximate", "Engagement")]:
        td = approx.get(key, {})
        label = td.get("label", "moderate")
        score_range = td.get("score_range", "40–60")
        features = td.get("influencing_features", [])
        color = label_to_color.get(label, "#6b7280")

        col1, col2 = st.columns([2, 3])
        with col1:
            st.metric(label=title, value=f"~{score_range}", delta=label.upper())
        with col2:
            if features:
                st.markdown("**Contributing features:**")
                for f in features[:4]:
                    st.markdown(f"- `{f}`")


def render_emotion_timeline(timeline: List[Dict[str, Any]], dual_emotions: Optional[Dict[str, Any]] = None):
    """Section d) — Emotion Trend Timeline with dual-model comparison."""
    st.subheader("Emotional Dynamics Over Time — Dual Model Comparison")

    if not timeline:
        st.info("No emotion timeline data available.")
        return

    df = pd.DataFrame(timeline)
    df["mid_time"] = (df["time_start"] + df["time_end"]) / 2

    # Detect if dual-model format (has emotion2vec_* and meralion_* columns)
    is_dual = "emotion2vec_emotion" in df.columns and "meralion_emotion" in df.columns

    # --- Overall comparison summary ---
    if dual_emotions:
        st.markdown("### Overall Emotion (full audio)")
        col1, col2, col3 = st.columns(3)
        e2v = dual_emotions.get("emotion2vec", {})
        mer = dual_emotions.get("meralion_ser", {})
        agree = dual_emotions.get("models_agree")

        with col1:
            st.metric("emotion2vec", e2v.get("primary_emotion", "N/A").capitalize(),
                       f"{e2v.get('confidence', 0):.1%}")
        with col2:
            st.metric("MERaLiON-SER", mer.get("primary_emotion", "N/A").capitalize(),
                       f"{mer.get('confidence', 0):.1%}")
        with col3:
            if agree is True:
                st.success("✅ Models AGREE")
            elif agree is False:
                st.warning("⚠️ Models DISAGREE")
            else:
                st.info("One model unavailable")

        # Score comparison bars
        if "scores" in e2v and "scores" in mer:
            st.markdown("### Score Distribution Comparison")
            e2v_scores = e2v["scores"]
            mer_scores = mer["scores"]
            all_labels = sorted(set(list(e2v_scores.keys()) + list(mer_scores.keys())))

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                name="emotion2vec",
                x=all_labels, y=[e2v_scores.get(l, 0) for l in all_labels],
                marker_color="#3b82f6",
            ))
            fig_cmp.add_trace(go.Bar(
                name="MERaLiON-SER",
                x=all_labels, y=[mer_scores.get(l, 0) for l in all_labels],
                marker_color="#f59e0b",
            ))
            fig_cmp.update_layout(
                barmode="group", height=300,
                yaxis_title="Probability",
                margin=dict(l=40, r=20, t=10, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("---")
    st.markdown("### Per-Segment Timeline")

    emotion_colors = {
        "happy": "#22c55e", "surprised": "#f59e0b", "angry": "#ef4444",
        "fearful": "#a855f7", "sad": "#3b82f6", "disgusted": "#64748b",
        "neutral": "#94a3b8", "other": "#d1d5db", "undetected": "#e5e7eb",
        "unavailable": "#e5e7eb", "error": "#dc2626",
    }

    # View toggle
    has_fused = "fused_emotion" in df.columns if is_dual else False
    view_mode = st.radio("View", ["Dual models", "Fused"], horizontal=True) if has_fused else "Dual models"

    if is_dual:
        # --- Arousal/Valence chart ---
        fig = go.Figure()
        if view_mode == "Fused" and "fused_arousal" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["mid_time"], y=df["fused_arousal"],
                mode="lines+markers", name="Fused Arousal",
                line=dict(color="#ef4444", width=3), marker=dict(size=6),
            ))
            fig.add_trace(go.Scatter(
                x=df["mid_time"], y=df["fused_valence"],
                mode="lines+markers", name="Fused Valence",
                line=dict(color="#3b82f6", width=3), marker=dict(size=6),
            ))
        else:
            if "emotion2vec_arousal" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["mid_time"], y=df["emotion2vec_arousal"],
                    mode="lines+markers", name="emotion2vec Arousal",
                    line=dict(color="#ef4444", width=2), marker=dict(size=5),
                ))
                fig.add_trace(go.Scatter(
                    x=df["mid_time"], y=df["emotion2vec_valence"],
                    mode="lines+markers", name="emotion2vec Valence",
                    line=dict(color="#3b82f6", width=2), marker=dict(size=5),
                ))
            if "meralion_arousal" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["mid_time"], y=df["meralion_arousal"],
                    mode="lines+markers", name="MERaLiON Arousal",
                    line=dict(color="#ef4444", width=2, dash="dash"), marker=dict(size=5, symbol="diamond"),
                ))
                fig.add_trace(go.Scatter(
                    x=df["mid_time"], y=df["meralion_valence"],
                    mode="lines+markers", name="MERaLiON Valence",
                    line=dict(color="#3b82f6", width=2, dash="dash"), marker=dict(size=5, symbol="diamond"),
                ))
            # Add fused as thin overlay
            if "fused_arousal" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["mid_time"], y=df["fused_arousal"],
                    mode="lines", name="Fused Arousal",
                    line=dict(color="#22c55e", width=2, dash="dot"),
                ))
                fig.add_trace(go.Scatter(
                    x=df["mid_time"], y=df["fused_valence"],
                    mode="lines", name="Fused Valence",
                    line=dict(color="#a855f7", width=2, dash="dot"),
                ))
        fig.update_layout(
            xaxis_title="Time (seconds)", yaxis_title="Score (−1 to +1)",
            yaxis=dict(range=[-1.1, 1.1]), height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Entropy chart (if available) ---
        if "entropy" in df.columns and df["entropy"].sum() > 0:
            fig_ent = go.Figure()
            fig_ent.add_trace(go.Bar(
                x=df["mid_time"], y=df["entropy"], name="Entropy",
                marker_color=["#ef4444" if e > 2.0 else "#f59e0b" if e > 1.0 else "#22c55e" for e in df["entropy"]],
            ))
            fig_ent.update_layout(
                xaxis_title="Time (seconds)", yaxis_title="Entropy (bits)",
                height=200, margin=dict(l=40, r=20, t=10, b=40),
            )
            st.plotly_chart(fig_ent, use_container_width=True)
            st.caption("Lower entropy = more certain prediction. Red bars indicate high uncertainty.")

        # --- Emotion labels chart ---
        fig2 = go.Figure()
        if view_mode == "Fused" and "fused_emotion" in df.columns:
            for emo in df["fused_emotion"].unique():
                sub = df[df["fused_emotion"] == emo]
                fig2.add_trace(go.Scatter(
                    x=sub["mid_time"], y=sub["rms_energy"],
                    mode="markers", name=f"Fused: {emo}",
                    marker=dict(color=emotion_colors.get(emo, "#6b7280"), size=12, symbol="star"),
                ))
        else:
            for emo in df["emotion2vec_emotion"].unique():
                sub = df[df["emotion2vec_emotion"] == emo]
                fig2.add_trace(go.Scatter(
                    x=sub["mid_time"], y=sub["rms_energy"],
                    mode="markers", name=f"e2v: {emo}",
                    marker=dict(color=emotion_colors.get(emo, "#6b7280"), size=10, symbol="circle"),
                ))
            for emo in df["meralion_emotion"].unique():
                sub = df[df["meralion_emotion"] == emo]
                fig2.add_trace(go.Scatter(
                    x=sub["mid_time"] + 0.5, y=sub["rms_energy"],
                    mode="markers", name=f"MER: {emo}",
                    marker=dict(color=emotion_colors.get(emo, "#6b7280"), size=10, symbol="diamond"),
                ))
        fig2.update_layout(
            xaxis_title="Time (seconds)", yaxis_title="RMS Energy",
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # --- Agreement table ---
        st.markdown("### Segment-by-Segment Comparison")
        table_rows = []
        for _, row in df.iterrows():
            agree_icon = "✅" if row.get("models_agree") else "⚠️"
            low_conf = "⚡" if row.get("low_confidence") else ""
            table_rows.append({
                "Time": f"{row['time_start']:.0f}–{row['time_end']:.0f}s",
                "emotion2vec": row.get("emotion2vec_emotion", "?"),
                "e2v conf": f"{row.get('emotion2vec_confidence', 0):.1%}",
                "MERaLiON": row.get("meralion_emotion", "?"),
                "MER conf": f"{row.get('meralion_confidence', 0):.1%}",
                "Fused": row.get("fused_emotion", "?"),
                "F conf": f"{row.get('fused_confidence', 0):.1%}",
                "Agree": agree_icon,
                "SNR": f"{row.get('snr_db', 0):.0f}dB",
                "Entropy": f"{row.get('entropy', 0):.2f}",
                "Flag": low_conf,
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    else:
        # Legacy single-model format
        if "arousal" in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["mid_time"], y=df["arousal"],
                mode="lines+markers", name="Arousal",
                line=dict(color="#ef4444", width=2), marker=dict(size=6),
            ))
            fig.add_trace(go.Scatter(
                x=df["mid_time"], y=df["valence"],
                mode="lines+markers", name="Valence",
                line=dict(color="#3b82f6", width=2), marker=dict(size=6),
            ))
            fig.update_layout(
                xaxis_title="Time (seconds)", yaxis_title="Score (−1 to +1)",
                yaxis=dict(range=[-1.1, 1.1]), height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        if "emotion" in df.columns:
            fig2 = go.Figure()
            for emo in df["emotion"].unique():
                sub = df[df["emotion"] == emo]
                fig2.add_trace(go.Scatter(
                    x=sub["mid_time"], y=sub["rms_energy"],
                    mode="markers", name=emo.capitalize(),
                    marker=dict(color=emotion_colors.get(emo, "#6b7280"), size=10),
                ))
            fig2.update_layout(
                xaxis_title="Time (seconds)", yaxis_title="RMS Energy",
                height=250,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)


def render_llm_panel(data: Dict[str, Any]):
    """Section e) — LLM Assessment Panel."""
    st.subheader("LLM Assessment")

    # Support both old flat ("assessment") and new structured format
    assessment = data.get("assessment", {})
    personality = data.get("personality_assessment", {})
    mot_eng = data.get("motivation_engagement", {})

    # Show which features were passed
    with st.expander("Voice features passed to LLM", expanded=False):
        vf = data.get("voice_features") or (data.get("voice_analysis", {}).get("prosody"))
        if vf:
            st.json(vf)
        else:
            st.info("Voice features not available in this file.")

    # Big Five from LLM — try new format first, then old
    big5 = personality.get("big_five") or assessment.get("big_five", {})
    if big5:
        st.markdown("### LLM Big Five Scores")
        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        scores = [big5.get(t, {}).get("score", 50) for t in traits]
        reasons = [big5.get(t, {}).get("reason", "") for t in traits]

        fig = go.Figure(go.Bar(
            y=[t.capitalize() for t in traits],
            x=scores,
            orientation="h",
            marker_color=px.colors.qualitative.Set2[:5],
            text=[f"{s}/100" for s in scores],
            textposition="inside",
        ))
        fig.update_layout(
            xaxis=dict(range=[0, 100]),
            height=250, margin=dict(l=0, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        for t, r in zip(traits, reasons):
            if r:
                st.caption(f"**{t.capitalize()}**: {r}")

    # Motivation & Engagement from LLM — try new format first
    mot = mot_eng.get("motivation") or assessment.get("motivation", {})
    eng = mot_eng.get("engagement") or assessment.get("engagement", {})

    col1, col2 = st.columns(2)
    with col1:
        score_val = mot.get("score", mot.get("motivation_score", "?"))
        st.metric("Motivation", f"{score_val}/100", mot.get("level", mot.get("overall_level", "")))
        st.caption(mot.get("pattern", ""))
    with col2:
        score_val = eng.get("score", eng.get("engagement_score", "?"))
        st.metric("Engagement", f"{score_val}/100", eng.get("level", eng.get("overall_level", "")))
        st.caption(eng.get("reason", ""))

    # HR Summary — try new location first
    summary = data.get("hr_summary") or assessment.get("hr_summary", "")
    if summary:
        st.markdown("### HR Summary")
        st.info(summary)

    # Strengths & Development Areas
    strengths = personality.get("strengths", assessment.get("trait_strengths", []))
    dev_areas = personality.get("development_areas", assessment.get("personality_development_areas", []))
    if strengths or dev_areas:
        col1, col2 = st.columns(2)
        with col1:
            if strengths:
                st.markdown("### Key Strengths")
                for s in strengths:
                    st.markdown(f"- {s}")
        with col2:
            if dev_areas:
                st.markdown("### Development Areas")
                for a in dev_areas:
                    st.markdown(f"- {a}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Voice Assessment Dashboard", layout="wide", page_icon="🎙️")

    st.title("🎙️ Voice-Based Personality & Motivation Dashboard")
    st.caption("Approximate indicators derived from audio voice features")

    # Determine output dir
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[-1])
    else:
        output_dir = Path("./outputs")

    if not output_dir.exists():
        st.error(f"Output directory not found: {output_dir}")
        return

    json_files = find_json_files(output_dir)
    if not json_files:
        st.warning(f"No assessment JSON files found in {output_dir}")
        return

    # File selector
    selected_file = st.sidebar.selectbox(
        "Select assessment file",
        json_files,
        format_func=lambda p: p.stem,
    )

    data = load_assessment(selected_file)

    # Metadata — support both old ("metadata") and new ("candidate") formats
    meta = data.get("candidate", data.get("metadata", {}))
    audio_name = Path(meta.get("audio_file", "")).name
    st.sidebar.markdown(f"**Audio:** {audio_name}")
    st.sidebar.markdown(f"**Candidate:** {meta.get('id', meta.get('candidate_id', 'N/A'))}")
    st.sidebar.markdown(f"**Timestamp:** {meta.get('timestamp', 'N/A')}")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Voice Features",
        "Big Five (Approx)",
        "Motivation & Engagement",
        "Emotion Timeline",
        "LLM Assessment",
        "Emotion Impact (Ablation)",
    ])

    # Extract data — support both old flat and new structured formats
    voice_analysis = data.get("voice_analysis", {})
    granular = voice_analysis.get("granular_features", data.get("granular_voice_features", {}))
    approx = data.get("approximate_assessment", {})

    # Emotion: rebuild dual timeline from new structured format if available
    emo_analysis = data.get("emotion_analysis", {})
    if emo_analysis:
        # New format — rebuild flat dual-timeline that render_emotion_timeline expects
        e2v_tl = emo_analysis.get("emotion2vec", {}).get("timeline", [])
        mer_tl = emo_analysis.get("meralion_ser", {}).get("timeline", [])
        fused_tl = emo_analysis.get("fused", {}).get("timeline", [])
        n_segs = max(len(e2v_tl), len(mer_tl), len(fused_tl))
        timeline = []
        for i in range(n_segs):
            e = e2v_tl[i] if i < len(e2v_tl) else {}
            m = mer_tl[i] if i < len(mer_tl) else {}
            fu = fused_tl[i] if i < len(fused_tl) else {}
            timeline.append({
                "time_start": e.get("time_start", m.get("time_start", 0)),
                "time_end": e.get("time_end", m.get("time_end", 0)),
                "rms_energy": e.get("rms_energy", m.get("rms_energy", 0)),
                "snr_db": e.get("snr_db", m.get("snr_db", 0)),
                "pitch_mean": e.get("pitch_mean", m.get("pitch_mean", 0)),
                "emotion2vec_emotion": e.get("emotion", "N/A"),
                "emotion2vec_confidence": e.get("confidence", 0),
                "emotion2vec_valence": e.get("valence", 0),
                "emotion2vec_arousal": e.get("arousal", 0),
                "meralion_emotion": m.get("emotion", "N/A"),
                "meralion_confidence": m.get("confidence", 0),
                "meralion_valence": m.get("valence", 0),
                "meralion_arousal": m.get("arousal", 0),
                "fused_emotion": fu.get("emotion", "N/A"),
                "fused_confidence": fu.get("confidence", 0),
                "fused_valence": fu.get("valence", 0),
                "fused_arousal": fu.get("arousal", 0),
                "entropy": fu.get("entropy", 0),
                "top2_gap": fu.get("top2_gap", 0),
                "low_confidence": fu.get("low_confidence", False),
                "models_agree": e.get("emotion", "?") == m.get("emotion", "??"),
            })
        dual_emotions = {
            "emotion2vec": emo_analysis.get("emotion2vec", {}).get("overall", {}),
            "meralion_ser": emo_analysis.get("meralion_ser", {}).get("overall", {}),
            "fused": emo_analysis.get("fused", {}).get("overall", {}),
            "models_agree": emo_analysis.get("comparison", {}).get("overall_agree"),
        }
    else:
        # Old format fallback
        timeline = data.get("emotion_timeline_rich", [])
        dual_emotions = data.get("dual_emotions", {})

    with tab1:
        if granular:
            render_voice_features_table(granular)
        else:
            st.info("Granular voice features not available. Re-run pipeline to generate.")

    with tab2:
        if approx:
            render_big5_approximate(approx)
        else:
            st.info("Approximate assessment not available. Re-run pipeline to generate.")

    with tab3:
        if approx:
            render_motivation_engagement(approx)
        else:
            st.info("Approximate assessment not available.")

    with tab4:
        render_emotion_timeline(timeline, dual_emotions=dual_emotions)

    with tab5:
        render_llm_panel(data)

    with tab6:
        render_ablation(data)


def render_ablation(data: Dict[str, Any]):
    """Section f) — Emotion Impact Ablation: baseline vs enriched Big Five."""
    st.subheader("Emotion Impact on Personality Assessment")

    llm_cmp = data.get("llm_comparison", {})
    emo_sum = data.get("emotion_summary", {})

    if not llm_cmp:
        st.info("No ablation data available. Re-run pipeline to generate.")
        return

    # Emotion summary card
    if emo_sum and not emo_sum.get("error"):
        st.markdown("### Fused Emotion Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Dominant", emo_sum.get("dominant_emotion", "?"),
                       f"{emo_sum.get('dominant_emotion_ratio', 0):.0%}")
        with col2:
            st.metric("Volatility", f"{emo_sum.get('emotion_volatility', 0):.2f}")
        with col3:
            st.metric("Valence", f"{emo_sum.get('valence_mean', 0):.2f}",
                       emo_sum.get("valence_trend", ""))
        with col4:
            st.metric("Arousal", f"{emo_sum.get('arousal_mean', 0):.2f}",
                       emo_sum.get("arousal_trend", ""))

        with st.expander("Full emotion summary JSON"):
            st.json(emo_sum)

    # Ablation deltas
    changes = llm_cmp.get("changes", [])
    baseline = llm_cmp.get("baseline_big5", {})
    enriched = llm_cmp.get("enriched_big5", {})

    if changes:
        st.markdown("### Big Five Score Changes (baseline → enriched)")
        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        old_scores = [baseline.get(t, {}).get("score", 50) for t in traits]
        new_scores = [enriched.get(t, {}).get("score", old_scores[i]) for i, t in enumerate(traits)]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Baseline (no emotion)",
            x=[t.capitalize() for t in traits], y=old_scores,
            marker_color="#94a3b8", text=[str(s) for s in old_scores], textposition="inside",
        ))
        fig.add_trace(go.Bar(
            name="Enriched (with emotion)",
            x=[t.capitalize() for t in traits], y=new_scores,
            marker_color="#3b82f6", text=[str(s) for s in new_scores], textposition="inside",
        ))
        fig.update_layout(
            barmode="group", height=350, yaxis=dict(range=[0, 100]),
            margin=dict(l=40, r=20, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Delta table
        delta_rows = []
        for ch in changes:
            delta = ch.get("delta", 0)
            arrow = "↑" if delta > 0 else "↓"
            delta_rows.append({
                "Trait": ch.get("trait", "?").capitalize(),
                "Old": ch.get("old_score", "?"),
                "New": ch.get("new_score", "?"),
                "Delta": f"{arrow}{abs(delta)}",
                "Cited Metrics": ", ".join(ch.get("emotion_metrics_cited", [])),
            })
        st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No score changes detected between baseline and enriched assessment.")

    # Impact summary
    impact = llm_cmp.get("emotion_impact_summary", "")
    if impact:
        st.markdown("### Impact Summary")
        st.info(impact)

    # Raw comparison data
    with st.expander("Raw ablation data"):
        st.json(llm_cmp)


if __name__ == "__main__":
    main()
