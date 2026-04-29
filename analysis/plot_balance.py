"""Recording balance plots: left/right leg counts and pre/post counts per subject."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.data_io import SUBJECTS


def plot_leg_side_balance(catalog: pd.DataFrame, out_dir: Path) -> None:
    counts = (
        catalog.groupby(["subject", "leg"])
        .size()
        .unstack(fill_value=0)
        .reindex(SUBJECTS, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    counts.plot(kind="bar", ax=ax, color=["royalblue", "tomato"], width=0.7)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Number of Recordings")
    ax.set_title("Left vs Right Leg Recording Count per Subject")
    ax.legend(title="Leg")
    ax.tick_params(axis="x", rotation=15)
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "balance_leg_side.png", dpi=150)
    plt.close(fig)


def plot_pre_post_balance(catalog: pd.DataFrame, out_dir: Path) -> None:
    counts = (
        catalog.groupby(["subject", "pre_post"])
        .size()
        .unstack(fill_value=0)
        .reindex(SUBJECTS, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    counts.plot(kind="bar", ax=ax, width=0.7)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Number of Recordings")
    ax.set_title("Pre vs Post Recording Count per Subject")
    ax.legend(title="Session")
    ax.tick_params(axis="x", rotation=15)
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=2, fontsize=9)
    # Note that all processed files are post-session only
    if "pre" not in counts.columns or counts.get("pre", pd.Series(0)).sum() == 0:
        ax.text(
            0.5, 0.92,
            "All processed files are post-session recordings (by design in preprocessing.py)",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            color="dimgray",
            style="italic",
        )
    fig.tight_layout()
    fig.savefig(out_dir / "balance_pre_post.png", dpi=150)
    plt.close(fig)
