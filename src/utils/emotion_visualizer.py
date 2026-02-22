"""Emotion timeline visualization using matplotlib.

Generates arousal/valence/dominance timeline plots and emotion heatmaps
from the voice analyzer output.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np


def plot_emotion_timeline(
    timeline: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Emotion Dynamics Timeline",
    figsize: tuple = (14, 10),
) -> Optional[Any]:
    """
    Plot arousal/valence/dominance timeline + emotion probability heatmap.

    Args:
        timeline: List of segment dicts from VoiceAnalyzer.analyze()
        output_path: If provided, save plot to this path (PNG/PDF/SVG)
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object (or None if matplotlib unavailable)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("[warn] matplotlib not installed, skipping visualization")
        return None

    if not timeline:
        print("[warn] Empty timeline, skipping visualization")
        return None

    emotions_7 = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]

    # Extract data
    times = [(seg["start_sec"] + seg["end_sec"]) / 2 for seg in timeline]
    valences = [seg["vad"]["valence"] for seg in timeline]
    arousals = [seg["vad"]["arousal"] for seg in timeline]
    dominances = [seg["vad"]["dominance"] for seg in timeline]

    # Emotion probability matrix
    emo_matrix = np.array([[seg.get(e, 0.0) for seg in timeline] for e in emotions_7])

    # Stress segments
    stress_times = [
        t for t, seg in zip(times, timeline)
        if seg["vad"]["arousal"] > 0.7 and seg["vad"]["valence"] < 0.4
    ]

    # --- Create figure ---
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1.5, 0.8], hspace=0.35)

    # --- Panel 1: VAD timeline ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, arousals, "r-o", markersize=3, linewidth=1.5, label="Arousal", alpha=0.9)
    ax1.plot(times, valences, "b-s", markersize=3, linewidth=1.5, label="Valence", alpha=0.9)
    ax1.plot(times, dominances, "g-^", markersize=3, linewidth=1.5, label="Dominance", alpha=0.7)

    # Mark stress segments
    if stress_times:
        ax1.scatter(stress_times, [0.85] * len(stress_times), marker="v", c="red",
                    s=80, zorder=5, label=f"Stress peaks ({len(stress_times)})")

    # Trend lines
    if len(times) >= 3:
        x = np.array(times)
        for vals, color in [(arousals, "r"), (valences, "b")]:
            coeffs = np.polyfit(x, vals, 1)
            trend = np.polyval(coeffs, x)
            ax1.plot(x, trend, f"{color}--", alpha=0.4, linewidth=1)

    ax1.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax1.set_ylabel("Score")
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylim(-1.0, 1.0)
    ax1.grid(True, alpha=0.2)

    # --- Panel 2: Emotion probability heatmap ---
    ax2 = fig.add_subplot(gs[1])
    cmap = LinearSegmentedColormap.from_list("emotion", ["#f0f0f0", "#ff6b35", "#d62828"])
    im = ax2.imshow(
        emo_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1,
        extent=[times[0], times[-1], len(emotions_7) - 0.5, -0.5],
        interpolation="nearest",
    )
    ax2.set_yticks(range(len(emotions_7)))
    ax2.set_yticklabels(emotions_7, fontsize=9)
    ax2.set_ylabel("Emotion")
    ax2.set_title("Emotion Probabilities Over Time", fontsize=11)
    fig.colorbar(im, ax=ax2, shrink=0.8, label="Probability")

    # --- Panel 3: Dominant emotion bar ---
    ax3 = fig.add_subplot(gs[2])
    dominant_per_seg = []
    for seg in timeline:
        seg_scores = {e: seg.get(e, 0.0) for e in emotions_7}
        dominant_per_seg.append(max(seg_scores, key=seg_scores.get))

    emo_colors = {
        "neutral": "#808080", "happy": "#FFD700", "sad": "#4169E1",
        "angry": "#DC143C", "surprised": "#FF8C00", "fearful": "#9370DB",
        "disgusted": "#2E8B57",
    }

    for i, (t, emo) in enumerate(zip(times, dominant_per_seg)):
        width = timeline[i]["end_sec"] - timeline[i]["start_sec"]
        ax3.barh(0, width, left=timeline[i]["start_sec"],
                 color=emo_colors.get(emo, "#808080"), edgecolor="white", linewidth=0.5)

    ax3.set_yticks([])
    ax3.set_xlabel("Time (seconds)")
    ax3.set_title("Dominant Emotion", fontsize=11)

    # Legend for emotion colors
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, label=e) for e, c in emo_colors.items()]
    ax3.legend(handles=legend_patches, loc="upper right", ncol=7, fontsize=7)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Emotion timeline plot saved: {output_path}")

    return fig


def plot_emotion_summary(
    emotion_aggregates: Dict[str, Any],
    output_path: Optional[str] = None,
    title: str = "Emotion Summary",
    figsize: tuple = (12, 5),
) -> Optional[Any]:
    """
    Plot emotion summary: radar chart of mean emotions + bar chart of dynamics.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    emo_stats = emotion_aggregates.get("emotion_stats", {})
    dynamics = emotion_aggregates.get("dynamics", {})
    derived = emotion_aggregates.get("derived", {})

    if not emo_stats:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Left: Emotion means bar chart ---
    emotions = list(emo_stats.keys())
    means = [emo_stats[e]["mean"] for e in emotions]
    maxes = [emo_stats[e]["max"] for e in emotions]

    x = np.arange(len(emotions))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, means, width, label="Mean", color="#4169E1", alpha=0.8)
    bars2 = ax1.bar(x + width / 2, maxes, width, label="Max", color="#DC143C", alpha=0.6)

    ax1.set_xticks(x)
    ax1.set_xticklabels(emotions, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Probability")
    ax1.set_title("Emotion Distribution", fontsize=11, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.2, axis="y")

    # --- Right: Dynamics metrics ---
    metrics = {
        "Confidence": derived.get("confidence_score", 0),
        "Stress Index": derived.get("stress_index", 0),
        "Arousal Vol.": dynamics.get("arousal_volatility", 0),
        "Emo. Shifts": dynamics.get("emotional_shifts", 0) / max(1, len(emotions)),
        "Stress Segs": dynamics.get("stress_segments", 0) / 10,  # normalize
    }

    names = list(metrics.keys())
    values = list(metrics.values())
    colors = ["#2E8B57", "#DC143C", "#FF8C00", "#4169E1", "#9370DB"]

    bars = ax2.barh(names, values, color=colors, alpha=0.8)
    ax2.set_xlim(0, 1)
    ax2.set_title("Dynamics & Derived Scores", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.2, axis="x")

    # Add value labels
    for bar, val in zip(bars, values):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Emotion summary plot saved: {output_path}")

    return fig
