"""Signal statistics: mean/std/min/max per channel per subject, saved as CSV and table plot."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.data_io import ALL_COLS, SUBJECT_PALETTE, SUBJECTS


def compute_channel_stats(df_all: pd.DataFrame) -> pd.DataFrame:
    """Returns MultiIndex DataFrame: (subject, channel) × (mean, std, min, max)."""
    records = []
    for subject in SUBJECTS:
        sub = df_all.loc[df_all["subject"] == subject, ALL_COLS]
        for col in ALL_COLS:
            s = sub[col]
            records.append({
                "subject": subject,
                "channel": col,
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "max": s.max(),
            })
    return pd.DataFrame(records).set_index(["subject", "channel"])


def save_stats_csv(stats_df: pd.DataFrame, analysis_dir: Path) -> None:
    path = analysis_dir / "signal_statistics.csv"
    stats_df.to_csv(path)
    print(f"  Saved {path}")


def plot_stats_table(stats_df: pd.DataFrame, out_dir: Path) -> None:
    df = stats_df.reset_index()
    df[["mean", "std", "min", "max"]] = df[["mean", "std", "min", "max"]].map(
        lambda v: f"{v:.3g}"
    )
    col_labels = ["Subject", "Channel", "Mean", "Std", "Min", "Max"]
    cell_text = df.values.tolist()

    # Build row colors: alternate shading per subject
    row_colors = []
    for subject in df["subject"]:
        base = SUBJECT_PALETTE[subject]
        row_colors.append([(*base, 0.25)] * len(col_labels))

    fig_height = max(8, len(df) * 0.28 + 1)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=row_colors,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax.set_title("Signal Statistics per Subject and Channel", pad=12, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "stats_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
