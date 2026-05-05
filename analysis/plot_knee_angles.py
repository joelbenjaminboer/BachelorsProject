#!/usr/bin/env python3
"""
Plot reconstructed knee angle from N random trials per subject.
Uses the test split of each LOSO fold (= that subject's held-out data).
y[:,0] from each window gives the instantaneous angle signal.

Usage:
    python plot_knee_angles.py
    python plot_knee_angles.py --n_trials 20 --seed 7
    python plot_knee_angles.py --data_dir data/processed/ENABL3S --output knee_angles.png
"""

import argparse
import glob
import os

import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FS = 100  # Hz after downsampling


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/processed/ENABL3S")
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--output",   default="knee_angles.png")
    return p.parse_args()


def load_subject_trials(h5_path: str, n_trials: int, rng: np.random.Generator):
    """
    Return a list of 1-D angle arrays (degrees) from random trials in the test split.
    y[:,0] from each window = instantaneous knee angle at that timestep.
    """
    with h5py.File(h5_path, "r") as f:
        grp    = f["test"]
        y_keys = sorted(k for k in grp.keys() if k.startswith("y_"))
        chosen = sorted(rng.choice(len(y_keys), min(n_trials, len(y_keys)), replace=False))
        trials = [grp[y_keys[i]][:, 0].astype(np.float32) for i in chosen]
    return trials


def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    h5_files = sorted(glob.glob(os.path.join(args.data_dir, "fold_*/data.h5")))
    if not h5_files:
        print(f"No HDF5 files found under {args.data_dir}")
        return

    n_subj = len(h5_files)
    n_cols = 5
    n_rows = (n_subj + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5.2, n_rows * 3.8),
        facecolor="white",
    )
    fig.suptitle(
        f"Knee Angle Signal — {args.n_trials} Random Trials per Subject  "
        f"(test / hold-out split, y[:,0] reconstructed)",
        fontsize=12, fontweight="bold", color="#2c3e50", y=0.998,
    )
    axes_flat = axes.ravel() if n_subj > 1 else [axes]

    # Collect global y-limits for a shared axis scale
    all_mins, all_maxs = [], []

    # ── First pass: load data and collect range ────────────────────────────────
    subject_data = []
    for h5_path in h5_files:
        subject = os.path.basename(os.path.dirname(h5_path))[5:]  # strip "fold_"
        trials  = load_subject_trials(h5_path, args.n_trials, rng)
        subject_data.append((subject, trials))
        for a in trials:
            all_mins.append(float(a.min()))
            all_maxs.append(float(a.max()))
        print(f"  {subject}: {len(trials)} trials loaded")

    y_lo = min(all_mins) - 5
    y_hi = max(all_maxs) + 5

    # ── Second pass: plot ──────────────────────────────────────────────────────
    cmap = plt.cm.plasma

    for ax_idx, (subject, trials) in enumerate(subject_data):
        ax = axes_flat[ax_idx]

        # Shaded angle-range bands (convention: negative = flexion)
        ax.axhspan(-130, -90, color="#8e44ad", alpha=0.07, zorder=0)
        ax.axhspan( -90, -60, color="#e74c3c", alpha=0.08, zorder=0)
        ax.axhspan( -60, -30, color="#f39c12", alpha=0.09, zorder=0)
        ax.axhspan( -30,   0, color="#2ecc71", alpha=0.09, zorder=0)
        ax.axhspan(   0,  20, color="#3498db", alpha=0.07, zorder=0)
        ax.axhline(0, color="#aaa", lw=0.8, ls="--", zorder=2)

        # Individual trial lines (colour = trial index via plasma colormap)
        n = len(trials)
        for ti, angles in enumerate(trials):
            t = np.arange(len(angles)) / FS
            ax.plot(t, angles,
                    lw=0.7, alpha=0.50,
                    color=cmap(ti / max(n - 1, 1)),
                    zorder=3)

        # Mean trace (aligned to shortest trial)
        if n > 1:
            min_len  = min(len(a) for a in trials)
            stacked  = np.stack([a[:min_len] for a in trials])
            mean_t   = np.arange(min_len) / FS
            ax.plot(mean_t, stacked.mean(axis=0),
                    lw=2.0, color="#1a1a2e", alpha=0.92, zorder=5)
        elif n == 1:
            t = np.arange(len(trials[0])) / FS
            ax.plot(t, trials[0], lw=2.0, color="#1a1a2e", alpha=0.92, zorder=5)

        # Per-subject statistics annotation
        all_vals = np.concatenate(trials)
        stats_txt = (
            f"mean {all_vals.mean():.1f}°\n"
            f"std  {all_vals.std():.1f}°\n"
            f"min  {all_vals.min():.1f}°\n"
            f"max  {all_vals.max():.1f}°"
        )
        ax.text(0.98, 0.97, stats_txt,
                transform=ax.transAxes, fontsize=7,
                va="top", ha="right", color="#444",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#ccc", alpha=0.85))

        ax.set_title(subject, fontsize=10.5, fontweight="bold", color="#2c3e50", pad=5)
        ax.set_xlabel("Time (s)", fontsize=8, labelpad=3)
        ax.set_ylabel("Knee angle (°)", fontsize=8, labelpad=3)
        ax.set_ylim(y_lo, y_hi)
        ax.tick_params(labelsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide any unused axes
    for ax in axes_flat[n_subj:]:
        ax.set_visible(False)

    # Shared legend: angle ranges + mean line
    range_patches = [
        mpatches.Patch(color="#8e44ad", alpha=0.55, label="Deep flex  >90°"),
        mpatches.Patch(color="#e74c3c", alpha=0.55, label="High flex  60–90°"),
        mpatches.Patch(color="#f39c12", alpha=0.55, label="Moderate   30–60°"),
        mpatches.Patch(color="#2ecc71", alpha=0.55, label="Slight flex 0–30°"),
        mpatches.Patch(color="#3498db", alpha=0.55, label="Extension  <0°"),
        plt.Line2D([0], [0], color="#1a1a2e", lw=2.0, label="Mean across trials"),
    ]
    fig.legend(
        handles=range_patches,
        loc="lower center", ncol=6, fontsize=8.5,
        framealpha=0.92, edgecolor="#ccc",
        bbox_to_anchor=(0.5, -0.012),
    )

    # Colorbar for trial index
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, args.n_trials))
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.55])
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label("Trial index", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0.04, 0.91, 0.995])
    plt.savefig(args.output, dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved → {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
