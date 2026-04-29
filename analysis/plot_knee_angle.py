"""Knee angle deep-dive: overall histogram, per-subject violin, range, left vs right."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.data_io import SUBJECT_PALETTE, SUBJECTS


def plot_knee_overall_hist(df_all: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df_all["KneeAngle"], bins=150, kde=True, ax=ax, color="steelblue", alpha=0.6)
    mean = df_all["KneeAngle"].mean()
    std = df_all["KneeAngle"].std()
    ax.axvline(mean, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {mean:.1f}°")
    ax.axvline(mean - std, color="orange", linestyle=":", linewidth=1.2, label=f"±1 SD")
    ax.axvline(mean + std, color="orange", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Knee Angle (°)")
    ax.set_ylabel("Count")
    ax.set_title("Knee Angle Distribution — All Subjects Combined")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "knee_overall_hist.png", dpi=150)
    plt.close(fig)


def plot_knee_per_subject_violin(df_all: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(
        data=df_all,
        x="subject",
        y="KneeAngle",
        hue="subject",
        order=SUBJECTS,
        palette=SUBJECT_PALETTE,
        inner="box",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Subject")
    ax.set_ylabel("Knee Angle (°)")
    ax.set_title("Knee Angle Distribution per Subject")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_dir / "knee_per_subject_violin.png", dpi=150)
    plt.close(fig)


def plot_knee_range_per_subject(df_all: pd.DataFrame, out_dir: Path) -> None:
    stats = df_all.groupby("subject")["KneeAngle"].agg(["min", "max"]).reindex(SUBJECTS)
    stats["range"] = stats["max"] - stats["min"]

    x = np.arange(len(SUBJECTS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    bars_min = ax.bar(x - width / 2, stats["min"], width, label="Min (flexion)", color="steelblue")
    bars_max = ax.bar(x + width / 2, stats["max"], width, label="Max (extension)", color="salmon")
    ax.bar_label(bars_min, fmt="%.1f°", padding=3, fontsize=8)
    ax.bar_label(bars_max, fmt="%.1f°", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(SUBJECTS, rotation=15)
    ax.set_ylabel("Knee Angle (°)")
    ax.set_title("Knee Angle Range per Subject")
    ax.legend()
    # Annotate total range above each pair
    for i, subj in enumerate(SUBJECTS):
        ax.text(i, stats.loc[subj, "max"] + 2, f"Δ{stats.loc[subj, 'range']:.1f}°",
                ha="center", fontsize=8, color="dimgray")
    fig.tight_layout()
    fig.savefig(out_dir / "knee_range_per_subject.png", dpi=150)
    plt.close(fig)


def plot_knee_left_vs_right(df_all: pd.DataFrame, out_dir: Path) -> None:
    sample_n = min(200_000, len(df_all) // 2)
    fig, ax = plt.subplots(figsize=(10, 5))
    for leg, color in [("left", "royalblue"), ("right", "tomato")]:
        subleg = df_all[df_all["leg"] == leg]
        if subleg.empty:
            continue
        data = subleg["KneeAngle"].sample(n=min(sample_n, len(subleg)), random_state=42)
        sns.kdeplot(data, ax=ax, color=color, label=f"{leg.capitalize()} leg", linewidth=2)
    ax.set_xlabel("Knee Angle (°)")
    ax.set_ylabel("Density")
    ax.set_title("Knee Angle Distribution: Left vs Right Leg")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "knee_left_vs_right_kde.png", dpi=150)
    plt.close(fig)
