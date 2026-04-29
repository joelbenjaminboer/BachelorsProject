"""Pearson correlation heatmaps: global and per-subject grid."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.data_io import ALL_COLS, SUBJECTS


def plot_correlation_heatmap(df_all: pd.DataFrame, out_dir: Path) -> None:
    corr = df_all[ALL_COLS].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 10},
    )
    ax.set_title("Pearson Correlation Matrix — All Subjects Combined")
    fig.tight_layout()
    fig.savefig(out_dir / "corr_global_heatmap.png", dpi=150)
    plt.close(fig)


def plot_per_subject_correlations(df_all: pd.DataFrame, out_dir: Path) -> None:
    n_subjects = len(SUBJECTS)
    ncols = 5
    nrows = (n_subjects + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()

    for i, subject in enumerate(SUBJECTS):
        subj_df = df_all.loc[df_all["subject"] == subject, ALL_COLS]
        corr = subj_df.corr()
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.3,
            ax=axes[i],
            annot_kws={"size": 7},
            cbar=False,
        )
        axes[i].set_title(subject, fontsize=10)
        axes[i].tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Pearson Correlation Matrix per Subject", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "corr_per_subject_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
