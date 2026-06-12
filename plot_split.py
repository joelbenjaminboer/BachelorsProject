"""One graph: the model's input (6 IMU features over the input window) and its
target (GONIO knee angle only, over the prediction horizon), shown z-normalized
exactly as the model sees them.

Z-normalization mirrors the dataloader: per-channel IMU stats and scalar knee
stats are computed from the TRAIN split of the fold, then applied to the window.
After standardization every signal is ~unit variance, so they share one y-axis.

Stats are streamed per trial (one window's first timestep = the continuous
signal) to keep memory bounded — numerically equivalent to the dataloader's
per-window stats.

Usage:
    python plot_split.py                       # fold_AB194, test split
    python plot_split.py --fold AB185 --split train --window 800
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

IMU_NAMES = ["Accel X", "Accel Y", "Accel Z", "Gyro X", "Gyro Y", "Gyro Z"]
GREEN = "#2ca25f"
BLUE = "#3aa0ff"


def train_stats(train_grp):
    """Per-channel IMU mean/std (6,) and scalar knee mean/std from the train split."""

    def welford_stream(prefix, pick):
        n, mean, M2 = 0, None, None
        for k in sorted(train_grp):
            if not k.startswith(prefix):
                continue
            chunk = pick(train_grp[k][:]).astype(np.float64)  # [m, C]
            m = chunk.shape[0]
            if mean is None:
                mean = np.zeros(chunk.shape[1])
                M2 = np.zeros(chunk.shape[1])
            c_mean = chunk.mean(0)
            c_M2 = chunk.var(0) * m
            delta = c_mean - mean
            new_n = n + m
            mean = mean + delta * (m / new_n)
            M2 = M2 + c_M2 + delta**2 * (n * m / new_n)
            n = new_n
        std = np.clip(np.sqrt(M2 / max(n, 1)), 1e-8, None)
        return mean, std

    imu_mean, imu_std = welford_stream("X_", lambda a: a[:, 0, :])   # continuous IMU
    y_mean, y_std = welford_stream("y_", lambda a: a[:, :1])         # continuous knee
    return imu_mean, imu_std, y_mean[0], y_std[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed/ENABL3S/100hz")
    p.add_argument("--fold", default="AB194")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--window", type=int, default=-1,
                   help="window index (default: middle window)")
    p.add_argument("--out", default="split_plot.png")
    args = p.parse_args()

    h5_path = Path(args.data_dir) / f"fold_{args.fold}" / "data.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"{h5_path} not found — run preprocessing first.")

    with h5py.File(h5_path, "r") as f:
        grp = f[args.split]
        X = grp["X_0"][:]   # [windows, context_length, 6]
        y = grp["y_0"][:]   # [windows, forecast_horizon]
        print("Computing train-split z-norm stats...")
        imu_mean, imu_std, y_mean, y_std = train_stats(f["train"])

    n_windows, context_len, _ = X.shape
    horizon = y.shape[1]
    w = args.window if args.window >= 0 else n_windows // 2
    if not (0 <= w < n_windows):
        raise ValueError(f"window {w} out of range [0, {n_windows}).")

    # Apply the model's z-normalization (train-split stats).
    x_win = (X[w] - imu_mean) / imu_std        # [context_len, 6]
    y_win = (y[w] - y_mean) / y_std            # [horizon]
    t_in = np.arange(context_len)
    t_tg = np.arange(context_len, context_len + horizon)

    fig, ax = plt.subplots(figsize=(13, 6))

    # Input region: 6 z-normalized IMU features.
    for i, name in enumerate(IMU_NAMES):
        ax.plot(t_in, x_win[:, i], linewidth=1.0, label=name)
    # Target region: z-normalized GONIO only.
    ax.plot(t_tg, y_win, color="#e34a33", linewidth=2.2, label="Knee angle (GONIO)")

    # Shade + label the two regions.
    ax.axvspan(0, context_len, color=GREEN, alpha=0.08)
    ax.axvspan(context_len, context_len + horizon, color=BLUE, alpha=0.10)
    ax.axvline(context_len, color="0.4", linestyle="--", linewidth=1)
    ax.axhline(0, color="0.6", linewidth=0.7, zorder=0)
    top = ax.get_ylim()[1]
    ax.text(context_len / 2, top, "INPUT — 6 IMU features", ha="center", va="top",
            color=GREEN, fontweight="bold", fontsize=11)
    ax.text(context_len + horizon / 2, top, "TARGET — GONIO", ha="center", va="top",
            color=BLUE, fontweight="bold", fontsize=11)

    ax.set_xlabel("Time (samples @ 100 Hz)")
    ax.set_ylabel("Standardized value (z-score, train stats)")
    ax.set_xlim(0, context_len + horizon)
    ax.grid(True, alpha=0.25)

    ax.legend(ncol=7, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), frameon=False)
    ax.set_title(
        f"Input → target (z-normalized) — fold {args.fold} ({args.split}):  "
        f"6 IMU features ({context_len} samples)  →  knee GONIO ({horizon} samples)"
    )
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
