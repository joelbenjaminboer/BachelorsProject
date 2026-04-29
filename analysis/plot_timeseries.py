"""Time-series sample traces: one figure per subject showing all 7 channels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.data_io import ALL_COLS, SAMPLE_RATE_HZ, SUBJECT_PALETTE

_CHANNEL_UNITS = {
    "Ax": "Ax (g)",
    "Ay": "Ay (g)",
    "Az": "Az (g)",
    "Gx": "Gx (°/s)",
    "Gy": "Gy (°/s)",
    "Gz": "Gz (°/s)",
    "KneeAngle": "Knee Angle (°)",
}


def plot_subject_timeseries(
    subject: str,
    df: pd.DataFrame,
    out_dir: Path,
    window: int = 2000,
) -> None:
    n = len(df)
    start = n // 4
    end = min(start + window, n)
    segment = df.iloc[start:end].reset_index(drop=True)
    t = segment.index / SAMPLE_RATE_HZ  # x-axis in seconds

    color = SUBJECT_PALETTE[subject]
    fig, axes = plt.subplots(len(ALL_COLS), 1, figsize=(16, 14), sharex=True)
    for ax, col in zip(axes, ALL_COLS):
        ax.plot(t, segment[col], color=color, linewidth=0.8)
        ax.set_ylabel(_CHANNEL_UNITS.get(col, col), fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Sample Time-Series — Subject {subject}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"timeseries_{subject}.png", dpi=150)
    plt.close(fig)


def plot_all_timeseries(
    sample_files: dict[str, pd.DataFrame],
    out_dir: Path,
    window: int = 2000,
) -> None:
    for subject, df in sample_files.items():
        plot_subject_timeseries(subject, df, out_dir, window=window)
