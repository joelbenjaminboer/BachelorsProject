"""Visualise the Leave-One-Subject-Out (LOSO) split.

Each fold holds out one subject as 100% test; the remaining subjects' trials are
pooled, shuffled, and split 90/10 into train/val (a trial-level slice, not a whole
subject). Panel 1 shows the fold × subject assignment matrix; panel 2 shows the
train/val/test window counts for one example fold.

Usage:
    python plot_loso_split.py
    python plot_loso_split.py --fold AB194 --out loso_split.png
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


def split_counts(h5_path: Path):
    """Return {split: (n_windows, n_trials)} for one fold's data.h5."""
    out = {}
    with h5py.File(h5_path, "r") as f:
        for g in ("train", "val", "test"):
            xs = [k for k in f[g] if k.startswith("X_")]
            out[g] = (sum(f[g][k].shape[0] for k in xs), len(xs))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed/ENABL3S/100hz")
    p.add_argument("--fold", default="AB194", help="fold to detail in panel 2")
    p.add_argument("--out", default="loso_split.png")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    subjects = sorted(d.name.replace("fold_", "") for d in data_dir.glob("fold_*"))
    if not subjects:
        raise FileNotFoundError(f"No fold_* dirs in {data_dir} — run preprocessing first.")
    if args.fold not in subjects:
        raise ValueError(f"--fold {args.fold} not in {subjects}")

    n = len(subjects)
    # Assignment matrix: 0 = train/val pool, 1 = held-out test (the diagonal).
    M = np.zeros((n, n), dtype=int)
    np.fill_diagonal(M, 1)

    fig, (ax_m, ax_b) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.4, 1]}
    )

    # --- Panel 1: LOSO assignment matrix ---
    cmap = ListedColormap(["#9ecae1", "#e34a33"])  # pool (blue), test (red)
    ax_m.imshow(M, cmap=cmap, aspect="auto")
    ax_m.set_xticks(range(n))
    ax_m.set_xticklabels(subjects, rotation=45, ha="right", fontsize=8)
    ax_m.set_yticks(range(n))
    ax_m.set_yticklabels([f"fold {s}" for s in subjects], fontsize=8)
    ax_m.set_xlabel("Subject")
    ax_m.set_title(f"LOSO split — {n} subjects, {n} folds\n(one held out per fold)")
    # Gridlines between cells.
    ax_m.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax_m.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax_m.grid(which="minor", color="white", linewidth=1.5)
    ax_m.tick_params(which="minor", length=0)
    ax_m.legend(
        handles=[
            Patch(facecolor="#9ecae1", label="train + val pool (90/10 over trials)"),
            Patch(facecolor="#e34a33", label="test (held-out subject)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9,
    )

    # --- Panel 2: window/trial counts for the example fold ---
    counts = split_counts(data_dir / f"fold_{args.fold}" / "data.h5")
    splits = ["train", "val", "test"]
    windows = [counts[s][0] for s in splits]
    trials = [counts[s][1] for s in splits]
    colors = ["#3182bd", "#9ecae1", "#e34a33"]

    bars = ax_b.bar(splits, windows, color=colors)
    ax_b.set_ylabel("Windows (samples)")
    ax_b.set_title(f"Fold {args.fold}: split sizes\n(test = subject {args.fold})")
    ax_b.grid(True, axis="y", alpha=0.3)
    total = sum(windows)
    for bar, wv, tr in zip(bars, windows, trials):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{wv:,}\n{wv / total:.0%} · {tr} trials",
            ha="center", va="bottom", fontsize=8,
        )
    ax_b.set_ylim(0, max(windows) * 1.18)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
