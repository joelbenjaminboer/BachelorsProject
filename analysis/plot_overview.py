"""Dataset overview plots: file counts, row counts, recording length distributions."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.data_io import SUBJECT_PALETTE, SUBJECTS


def plot_file_count(catalog: pd.DataFrame, out_dir: Path) -> None:
    counts = catalog.groupby("subject").size().reindex(SUBJECTS)
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(counts.index, counts.values, color=[SUBJECT_PALETTE[s] for s in counts.index])
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Number of Recordings")
    ax.set_title("Recording Count per Subject")
    fig.tight_layout()
    fig.savefig(out_dir / "overview_file_count_per_subject.png", dpi=150)
    plt.close(fig)


def plot_row_count(file_stats: pd.DataFrame, out_dir: Path) -> None:
    totals = file_stats.groupby("subject")["n_rows"].sum().reindex(SUBJECTS) / 1e6
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(totals.index, totals.values, color=[SUBJECT_PALETTE[s] for s in totals.index])
    ax.bar_label(bars, fmt="%.1fM", padding=3)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Total Samples (millions)")
    ax.set_title("Total Data Points per Subject")
    fig.tight_layout()
    fig.savefig(out_dir / "overview_row_count_per_subject.png", dpi=150)
    plt.close(fig)


def plot_recording_length_dist(file_stats: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(
        data=file_stats,
        x="subject",
        y="n_rows",
        order=SUBJECTS,
        palette=SUBJECT_PALETTE,
        inner="box",
        ax=ax,
    )
    ax.set_xlabel("Subject")
    ax.set_ylabel("Samples per Recording")
    ax.set_title("Recording Length Distribution per Subject")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_dir / "overview_recording_length_dist.png", dpi=150)
    plt.close(fig)
