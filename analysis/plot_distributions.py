"""Signal distribution plots: overall histograms and per-subject KDE curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.data_io import ALL_COLS, ALL_CHANNELS, SUBJECT_PALETTE, SUBJECTS

_CHANNEL_UNITS = {
    "Ax": "Ax (g)",
    "Ay": "Ay (g)",
    "Az": "Az (g)",
    "Gx": "Gx (°/s)",
    "Gy": "Gy (°/s)",
    "Gz": "Gz (°/s)",
    "KneeAngle": "Knee Angle (°)",
}


def plot_overall_histograms(df_all: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()
    for i, col in enumerate(ALL_COLS):
        ax = axes[i]
        sns.histplot(df_all[col], bins=120, kde=True, ax=ax, color="steelblue", alpha=0.6)
        ax.set_xlabel(_CHANNEL_UNITS.get(col, col))
        ax.set_ylabel("Count")
        ax.set_title(col)
    axes[-1].set_visible(False)  # 7 channels → 1 empty panel in 2×4 grid
    fig.suptitle("Signal Distributions — All Subjects Combined", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "dist_overall_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_subject_kde(df_all: pd.DataFrame, channel: str, out_dir: Path) -> None:
    # Sub-sample per subject to keep KDE estimation fast
    sample_per_subject = min(50_000, df_all.groupby("subject").size().min())
    frames = []
    for s in SUBJECTS:
        subj = df_all[df_all["subject"] == s]
        if len(subj) > 0:
            frames.append(subj.sample(n=min(sample_per_subject, len(subj)), random_state=42))
    sampled = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for subject in SUBJECTS:
        data = sampled.loc[sampled["subject"] == subject, channel]
        if data.empty:
            continue
        sns.kdeplot(data, ax=ax, color=SUBJECT_PALETTE[subject], label=subject, linewidth=1.8)
    ax.set_xlabel(_CHANNEL_UNITS.get(channel, channel))
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {channel} by Subject")
    ax.legend(title="Subject", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"dist_per_subject_kde_{channel}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_per_subject_kdes(df_all: pd.DataFrame, out_dir: Path) -> None:
    for col in ALL_COLS:
        plot_per_subject_kde(df_all, col, out_dir)
