#!/usr/bin/env python3
"""
Data analysis for preprocessed ENABL3S HDF5 folds.

Checks for: NaN/Inf, constant windows, Gyro/Accel scale mismatch,
cross-fold distribution drift, outliers, and y-horizon consistency.

Usage:
    python analyze_data.py
    python analyze_data.py --data_dir data/processed/ENABL3S
    python analyze_data.py --fold AB192          # single fold only
    python analyze_data.py --max_trials 20       # faster scan (default: 50)
"""

import argparse
import glob
import os

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

CHANNELS   = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
CH_COLORS  = ["#e74c3c","#e67e22","#2ecc71","#3498db","#9b59b6","#1abc9c"]
ACCEL_IDX  = [0, 1, 2]
GYRO_IDX   = [3, 4, 5]

# Thresholds for flagging issues
RATIO_WARN = 20    # gyro/accel std ratio worth noting
RATIO_BAD  = 80    # ratio that strongly destabilises training
OUTLIER_Z  = 5.0   # z-score threshold for outlier detection


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="data/processed/ENABL3S")
    p.add_argument("--fold",        default=None,  help="Single fold ID, e.g. AB192")
    p.add_argument("--max_trials",  default=50, type=int,
                   help="Max trials per split to scan (default 50; use 0 for all)")
    p.add_argument("--output",      default="data_analysis.png")
    return p.parse_args()


# ── Per-fold stats (memory-efficient: one trial at a time) ─────────────────────

def analyse_fold(h5_path: str, max_trials: int) -> dict:
    """
    Returns stats dict keyed by split ('train', 'val', 'test').
    Loads each HDF5 trial once, accumulates statistics, then discards it.
    """
    rng = np.random.default_rng(42)
    result = {}

    with h5py.File(h5_path, "r") as f:
        for split in ("train", "val", "test"):
            grp     = f[split]
            x_keys  = sorted(k for k in grp.keys() if k.startswith("X_"))
            n_avail = len(x_keys)
            keys    = x_keys if max_trials == 0 else x_keys[:max_trials]

            # Running channel stats (Welford-style: accumulate sum / sum-of-squares)
            ch_n    = 0                        # total scalar values per channel
            ch_sum  = np.zeros(6)
            ch_sum2 = np.zeros(6)
            ch_min  = np.full(6,  np.inf)
            ch_max  = np.full(6, -np.inf)

            # Quality counters (over windows)
            n_windows   = 0
            n_nan_x     = 0
            n_inf_x     = 0
            n_const_win = 0   # windows where any channel has zero variance

            # Outlier counters — computed per-trial using trial's own stats
            n_outlier_x = 0

            # y stats
            y_sum  = 0.0
            y_sum2 = 0.0
            y_n    = 0
            y_min  = np.inf
            y_max  = -np.inf
            y_horizon = None

            # Sample buffers for visualisation (kept small)
            x_vis_list = []
            y_vis_list = []

            for kx in keys:
                ky  = kx.replace("X_", "y_")
                X   = grp[kx][:]         # (W, T, 6)
                Y   = grp[ky][:]         # (W, horizon)
                W, _, C = X.shape

                if y_horizon is None:
                    y_horizon = Y.shape[-1]

                n_windows += W

                # Flatten to (W*T, 6) for per-channel stats
                Xf = X.reshape(-1, C).astype(np.float64)
                ch_sum  += Xf.sum(axis=0)
                ch_sum2 += (Xf ** 2).sum(axis=0)
                ch_min   = np.minimum(ch_min, Xf.min(axis=0))
                ch_max   = np.maximum(ch_max, Xf.max(axis=0))
                ch_n    += Xf.shape[0]

                n_nan_x += int(np.isnan(X).sum())
                n_inf_x += int(np.isinf(X).sum())

                # Constant windows: any channel with std=0 across the time axis
                win_std      = X.std(axis=1)               # (W, 6)
                n_const_win += int((win_std < 1e-7).any(axis=1).sum())

                # Outliers: per-trial z-score for each channel × timestep
                trial_mean = Xf.mean(axis=0)
                trial_std  = Xf.std(axis=0) + 1e-8
                z = np.abs((Xf - trial_mean) / trial_std)
                n_outlier_x += int((z > OUTLIER_Z).any(axis=1).sum())

                # y stats
                Yf      = Y.ravel().astype(np.float64)
                y_sum  += Yf.sum()
                y_sum2 += (Yf ** 2).sum()
                y_n    += len(Yf)
                y_min   = min(y_min, Yf.min())
                y_max   = max(y_max, Yf.max())

                # Collect visualisation samples
                if len(x_vis_list) < 8:
                    idx = rng.integers(0, W, min(4, W))
                    x_vis_list.append(X[idx])
                    y_vis_list.append(Y[idx])

            ch_mean = ch_sum / ch_n
            ch_var  = np.maximum(ch_sum2 / ch_n - ch_mean ** 2, 0)
            ch_std  = np.sqrt(ch_var)

            y_mean = y_sum / y_n
            y_std  = np.sqrt(max(y_sum2 / y_n - y_mean ** 2, 0))

            result[split] = dict(
                n_windows    = n_windows,
                n_trials     = len(keys),
                n_avail      = n_avail,
                ch_mean      = ch_mean,
                ch_std       = ch_std,
                ch_min       = ch_min,
                ch_max       = ch_max,
                n_nan        = n_nan_x,
                n_inf        = n_inf_x,
                n_const      = n_const_win,
                n_outlier    = n_outlier_x,
                y_mean       = y_mean,
                y_std        = y_std,
                y_min        = y_min,
                y_max        = y_max,
                y_horizon    = y_horizon,
                x_vis        = np.concatenate(x_vis_list, axis=0) if x_vis_list else np.array([]),
                y_vis        = np.concatenate(y_vis_list, axis=0) if y_vis_list else np.array([]),
            )

    return result


# ── Console report ─────────────────────────────────────────────────────────────

def print_report(all_stats: dict):
    RED  = "\033[91m"; YLW = "\033[93m"; GRN = "\033[92m"
    BOLD = "\033[1m";  RST = "\033[0m"

    def c(val, warn, bad):
        if val >= bad:  return f"{RED}{val}{RST}"
        if val >= warn: return f"{YLW}{val}{RST}"
        return f"{GRN}{val}{RST}"

    print(f"\n{BOLD}══════════════════════ DATA ANALYSIS REPORT ══════════════════════{RST}")

    for fold_name, fold_stats in sorted(all_stats.items()):
        tr = fold_stats["train"]
        partial = f"  (of {tr['n_avail']})" if tr["n_trials"] < tr["n_avail"] else ""
        print(f"\n{BOLD}{fold_name}{RST}  —  y_horizon = {tr['y_horizon']}")
        print(f"  Scanned trials: train={tr['n_trials']}{partial}  "
              f"val={fold_stats['val']['n_trials']}  test={fold_stats['test']['n_trials']}")
        print(f"  Windows:        train={tr['n_windows']:,}  "
              f"val={fold_stats['val']['n_windows']:,}  test={fold_stats['test']['n_windows']:,}")

        print(f"\n  {'Chan':<5} {'mean':>9} {'std':>9} {'min':>9} {'max':>9}")
        for i, ch in enumerate(CHANNELS):
            print(f"  {ch:<5} {tr['ch_mean'][i]:>9.3f} {tr['ch_std'][i]:>9.3f}"
                  f" {tr['ch_min'][i]:>9.3f} {tr['ch_max'][i]:>9.3f}")

        accel_std = tr["ch_std"][ACCEL_IDX].mean()
        gyro_std  = tr["ch_std"][GYRO_IDX].mean()
        ratio     = gyro_std / (accel_std + 1e-8)
        ratio_col = (RED if ratio >= RATIO_BAD else YLW if ratio >= RATIO_WARN else GRN)
        print(f"\n  Gyro / Accel std ratio : {ratio_col}{ratio:.1f}×{RST}  "
              f"({RED}≥{RATIO_BAD}× = unstable{RST}, {YLW}≥{RATIO_WARN}× = caution{RST})")

        print(f"  y (knee angle)         : mean={tr['y_mean']:.2f}°  std={tr['y_std']:.2f}°  "
              f"[{tr['y_min']:.1f}, {tr['y_max']:.1f}]")
        print(f"  NaN={c(tr['n_nan'],1,1)}  Inf={c(tr['n_inf'],1,1)}  "
              f"Const windows={c(tr['n_const'],10,100)}  "
              f"Outlier windows (|z|>{OUTLIER_Z:.0f})={c(tr['n_outlier'],100,1000)}")

    # Cross-fold summary
    print(f"\n{BOLD}Cross-fold channel summary (train, scanned trials):{RST}")
    print(f"  {'Chan':<5} {'μ of means':>12} {'σ of means':>12} {'μ of stds':>12}")
    for i, ch in enumerate(CHANNELS):
        fold_means = [s["train"]["ch_mean"][i] for s in all_stats.values()]
        fold_stds  = [s["train"]["ch_std"][i]  for s in all_stats.values()]
        print(f"  {ch:<5} {np.mean(fold_means):>12.3f} {np.std(fold_means):>12.3f} "
              f"{np.mean(fold_stds):>12.3f}")


# ── Figure ─────────────────────────────────────────────────────────────────────

def make_figure(all_stats: dict, h5_paths: dict, output: str):
    fold_names = sorted(all_stats.keys())
    n_folds    = len(fold_names)
    labels     = [f[5:] for f in fold_names]   # strip "fold_"

    # Pre-compute per-fold channel stats matrices (n_folds × 6)
    means_mat = np.array([[all_stats[f]["train"]["ch_mean"][c] for c in range(6)]
                          for f in fold_names])
    stds_mat  = np.array([[all_stats[f]["train"]["ch_std"][c]  for c in range(6)]
                          for f in fold_names])

    fig = plt.figure(figsize=(26, 30), facecolor="white")
    fig.suptitle("ENABL3S Preprocessed Data Analysis",
                 fontsize=14, fontweight="bold", color="#2c3e50", y=0.990)

    gs = gridspec.GridSpec(4, 3, figure=fig,
                           height_ratios=[1, 1, 1, 1.1],
                           hspace=0.50, wspace=0.34,
                           left=0.06, right=0.97, top=0.963, bottom=0.03)

    # ── [0,0] Sample counts per fold ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    w = 0.26
    x = np.arange(n_folds)
    tr_n  = [all_stats[f]["train"]["n_windows"] / 1e6 for f in fold_names]
    val_n = [all_stats[f]["val"]["n_windows"]   / 1e6 for f in fold_names]
    te_n  = [all_stats[f]["test"]["n_windows"]  / 1e6 for f in fold_names]
    ax.bar(x - w,  tr_n,  w, label="Train", color="#4A90D9", alpha=0.85, edgecolor="white")
    ax.bar(x,      val_n, w, label="Val",   color="#5BA55B", alpha=0.85, edgecolor="white")
    ax.bar(x + w,  te_n,  w, label="Test",  color="#E8A838", alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, fontsize=7.5, ha="right")
    ax.set_ylabel("Windows (millions)"); ax.set_xlabel("Holdout subject")
    ax.set_title("Window Counts per Fold", fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── [0,1] Channel mean heatmap ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    vmax = np.abs(means_mat).max()
    im = ax.imshow(means_mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(6)); ax.set_xticklabels(CHANNELS, fontsize=9)
    ax.set_yticks(range(n_folds)); ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.030, pad=0.03)
    ax.set_title("Channel Means  (train split, raw units)", fontweight="bold")
    for i in range(n_folds):
        for j in range(6):
            ax.text(j, i, f"{means_mat[i,j]:.1f}", ha="center", va="center",
                    fontsize=6.5, color="white" if abs(means_mat[i,j]) > vmax * 0.5 else "#222")

    # ── [0,2] Channel std heatmap ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    im2 = ax.imshow(stds_mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(6)); ax.set_xticklabels(CHANNELS, fontsize=9)
    ax.set_yticks(range(n_folds)); ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im2, ax=ax, fraction=0.030, pad=0.03)
    ax.set_title("Channel Stds  (train split, raw units)\n"
                 "Gyro cols >> Accel cols → scale mismatch", fontweight="bold", fontsize=9)
    for i in range(n_folds):
        for j in range(6):
            ax.text(j, i, f"{stds_mat[i,j]:.1f}", ha="center", va="center",
                    fontsize=6.5, color="white" if stds_mat[i,j] > stds_mat.mean() else "#222")

    # ── [1,0] y (knee angle) distribution per fold ────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    y_means = [all_stats[f]["train"]["y_mean"]  for f in fold_names]
    y_stds  = [all_stats[f]["train"]["y_std"]   for f in fold_names]
    y_mins  = [all_stats[f]["train"]["y_min"]   for f in fold_names]
    y_maxs  = [all_stats[f]["train"]["y_max"]   for f in fold_names]
    fold_colors = plt.cm.tab10(np.linspace(0, 0.9, n_folds))
    for i in range(n_folds):
        ax.errorbar(i, y_means[i], yerr=y_stds[i], fmt="o", color=fold_colors[i],
                    capsize=4, capthick=1.5, markersize=6)
        ax.plot([i, i], [y_mins[i], y_maxs[i]], color=fold_colors[i], lw=1, alpha=0.4)
    ax.set_xticks(range(n_folds)); ax.set_xticklabels(labels, rotation=40, fontsize=7.5, ha="right")
    ax.set_ylabel("Knee angle (°)"); ax.set_xlabel("Holdout subject")
    ax.set_title("Target (y) Distribution per Fold\n(dot=mean, bar=±1σ, thin line=full range)",
                 fontweight="bold", fontsize=9)
    ax.axhline(0, color="#aaa", lw=1, ls="--", label="0°")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── [1,1] Gyro / Accel std ratio per fold ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ratios = []
    for f in fold_names:
        a = all_stats[f]["train"]["ch_std"][ACCEL_IDX].mean()
        g = all_stats[f]["train"]["ch_std"][GYRO_IDX].mean()
        ratios.append(g / (a + 1e-8))
    bar_colors = ["#e74c3c" if r >= RATIO_BAD else "#f39c12" if r >= RATIO_WARN
                  else "#2ecc71" for r in ratios]
    ax.bar(range(n_folds), ratios, color=bar_colors, alpha=0.88, edgecolor="white")
    ax.set_xticks(range(n_folds)); ax.set_xticklabels(labels, rotation=40, fontsize=7.5, ha="right")
    ax.set_ylabel("Gyro std / Accel std"); ax.set_xlabel("Holdout subject")
    ax.set_title("Gyro / Accel Scale Ratio\n(large ratio → unstable training without normalisation)",
                 fontweight="bold", fontsize=9)
    ax.axhline(RATIO_BAD,  color="#e74c3c", lw=1.5, ls="--", label=f"danger  ({RATIO_BAD}×)")
    ax.axhline(RATIO_WARN, color="#f39c12", lw=1.2, ls="--", label=f"caution ({RATIO_WARN}×)")
    ax.legend(fontsize=8)
    for i, r in enumerate(ratios):
        ax.text(i, r + max(ratios) * 0.02, f"{r:.0f}×", ha="center", fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── [1,2] Data quality heatmap (NaN / Inf / Const / Outlier) ──────────────
    ax = fig.add_subplot(gs[1, 2])
    cols = ["tr NaN", "tr Inf", "tr Const", "tr Outlier",
            "val NaN", "val Const", "te NaN", "te Const"]
    quality = np.array([
        [all_stats[f]["train"]["n_nan"],
         all_stats[f]["train"]["n_inf"],
         all_stats[f]["train"]["n_const"],
         all_stats[f]["train"]["n_outlier"],
         all_stats[f]["val"]["n_nan"],
         all_stats[f]["val"]["n_const"],
         all_stats[f]["test"]["n_nan"],
         all_stats[f]["test"]["n_const"]]
        for f in fold_names
    ], dtype=float)
    # Log scale so small nonzeros are visible but the scale isn't dominated by outlier counts
    quality_log = np.log1p(quality)
    im3 = ax.imshow(quality_log, aspect="auto", cmap="Reds")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=38, fontsize=7.5, ha="right")
    ax.set_yticks(range(n_folds)); ax.set_yticklabels(labels, fontsize=8)
    cb3 = plt.colorbar(im3, ax=ax, fraction=0.020, pad=0.03)
    cb3.set_label("log(1 + count)", fontsize=7)
    ax.set_title("Data Quality Flags  (NaN / Inf / Constant / Outlier windows)",
                 fontweight="bold", fontsize=9)
    for i in range(n_folds):
        for j in range(len(cols)):
            ax.text(j, i, f"{int(quality[i,j])}", ha="center", va="center",
                    fontsize=6, color="white" if quality_log[i,j] > quality_log.max()*0.6 else "#333")

    # ── [2,0] Sample raw IMU windows ──────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    fold0 = fold_names[0]
    xs = all_stats[fold0]["train"]["x_vis"]
    if len(xs) >= 3:
        T    = xs.shape[1]
        t    = np.arange(T)
        # Offset channels so they don't overlap: use consistent spacing
        offsets  = np.array([0, 1, 2, 4, 5, 6]) * 50
        for ci, ch in enumerate(CHANNELS):
            for wi in range(min(3, len(xs))):
                ax.plot(t, xs[wi, :, ci] + offsets[ci],
                        color=CH_COLORS[ci], lw=0.9, alpha=0.65)
            ax.text(T + 1, offsets[ci] + xs[:3, :, ci].mean(),
                    ch, va="center", fontsize=8.5, color=CH_COLORS[ci], fontweight="bold")
        ax.set_xlabel("Time step (context window)"); ax.set_xlim(0, T + 12)
        ax.set_yticks([]); ax.set_title(
            f"Sample IMU Windows  ({fold0[5:]})\n"
            "3 random windows per channel, vertically offset (raw, unnormalised)",
            fontweight="bold", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── [2,1] Cross-fold channel mean drift ───────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    for ci, ch in enumerate(CHANNELS):
        ax.plot(range(n_folds), means_mat[:, ci], "o-", label=ch,
                color=CH_COLORS[ci], lw=1.6, markersize=5)
    ax.fill_between(range(n_folds),
                    means_mat[:, :3].min(axis=1), means_mat[:, :3].max(axis=1),
                    alpha=0.10, color="#e74c3c", label="Accel range")
    ax.fill_between(range(n_folds),
                    means_mat[:, 3:].min(axis=1), means_mat[:, 3:].max(axis=1),
                    alpha=0.08, color="#3498db", label="Gyro range")
    ax.set_xticks(range(n_folds)); ax.set_xticklabels(labels, rotation=40, fontsize=7.5, ha="right")
    ax.set_ylabel("Channel mean (raw)"); ax.set_xlabel("Holdout subject (LOSO fold)")
    ax.set_title("Channel Mean Drift Across Folds\n(large spread = subject-specific bias)",
                 fontweight="bold", fontsize=9)
    ax.axhline(0, color="#ddd", lw=1)
    ax.legend(fontsize=7.5, ncol=2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── [2,2] Channel correlation matrix ──────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    xs_sample = all_stats[fold_names[0]]["train"]["x_vis"]
    if len(xs_sample) > 0:
        flat = xs_sample.reshape(-1, 6)
        corr = np.corrcoef(flat.T)
        im4  = ax.imshow(corr, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im4, ax=ax, fraction=0.035, pad=0.03)
        ax.set_xticks(range(6)); ax.set_xticklabels(CHANNELS, fontsize=9)
        ax.set_yticks(range(6)); ax.set_yticklabels(CHANNELS, fontsize=9)
        ax.set_title(f"Channel Correlation Matrix  ({fold_names[0][5:]})\n"
                     "high off-diagonal correlation = redundant channels",
                     fontweight="bold", fontsize=9)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if abs(corr[i,j]) > 0.6 else "#222")
        # Mark Accel / Gyro block boundaries
        for offset in (2.5,):
            ax.axhline(offset, color="white", lw=2, ls="--", alpha=0.6)
            ax.axvline(offset, color="white", lw=2, ls="--", alpha=0.6)
        ax.text(1, -0.7, "Accel", ha="center", va="center", fontsize=8, color="#e74c3c")
        ax.text(4, -0.7, "Gyro",  ha="center", va="center", fontsize=8, color="#3498db")

    # ── [3, 0:2] Knee angle time series (goniometer signal) ───────────────────
    ax_ts = fig.add_subplot(gs[3, 0:2])

    fold0 = fold_names[0]
    with h5py.File(h5_paths[fold0], "r") as f:
        # Pick the test trial with the most windows for the richest signal
        test_keys = sorted(k for k in f["test"].keys() if k.startswith("X_"))
        kx = max(test_keys, key=lambda k: f["test"][k].shape[0])
        ky = kx.replace("X_", "y_")
        Y_seq = f["test"][ky][:3000].astype(np.float32)   # (W, H)

    H        = Y_seq.shape[1]
    angle_t  = Y_seq[:, 0]                    # instantaneous angle at each timestep
    t_s      = np.arange(len(angle_t)) / 100.0   # seconds at 100 Hz

    # Shaded background ranges (convention: negative = flexion)
    y_lo, y_hi = angle_t.min() - 5, angle_t.max() + 5
    ax_ts.axhspan(-130,  -90, color="#8e44ad", alpha=0.07, label="Deep flex  >90°")
    ax_ts.axhspan( -90,  -60, color="#e74c3c", alpha=0.08, label="High flex  60–90°")
    ax_ts.axhspan( -60,  -30, color="#f39c12", alpha=0.09, label="Moderate   30–60°")
    ax_ts.axhspan( -30,    0, color="#2ecc71", alpha=0.09, label="Slight     0–30°")
    ax_ts.axhspan(   0,   20, color="#3498db", alpha=0.07, label="Extension  <0°")

    # Main angle trace
    ax_ts.plot(t_s, angle_t, lw=0.9, color="#2c3e50", alpha=0.9, zorder=3, label="Angle signal")

    # Overlay a few forecast windows as coloured fans
    rng = np.random.default_rng(7)
    fan_idx = rng.integers(50, max(51, len(Y_seq) - H - 1), min(8, len(Y_seq) // 200))
    for idx in fan_idx:
        t_fan = np.arange(H) / 100.0 + t_s[idx]
        ax_ts.plot(t_fan, Y_seq[idx], lw=1.6, color="#e74c3c", alpha=0.50, zorder=4)
        ax_ts.axvline(t_s[idx], color="#e74c3c", lw=0.6, ls="--", alpha=0.25, zorder=2)

    ax_ts.axhline(0, color="#888", lw=0.9, ls="-", zorder=3)
    ax_ts.set_xlim(t_s[0], t_s[-1])
    ax_ts.set_ylim(y_lo, y_hi)
    ax_ts.set_xlabel("Time (s)  [100 Hz after downsampling]", fontsize=9)
    ax_ts.set_ylabel("Knee angle (°)", fontsize=9)
    ax_ts.set_title(
        f"Reconstructed Knee Angle Signal — holdout subject {fold0[5:]}  (test split)\n"
        f"Black: y[:,0] stitched across consecutive windows  ·  "
        f"Red: {H}-step forecast horizon samples",
        fontweight="bold", fontsize=9,
    )
    ax_ts.legend(fontsize=7.5, loc="lower right", ncol=3, framealpha=0.85)
    ax_ts.spines["top"].set_visible(False)
    ax_ts.spines["right"].set_visible(False)

    # ── [3, 2] Goniometer polar distribution ──────────────────────────────────
    ax_g = fig.add_subplot(gs[3, 2], projection="polar")

    # Convert to positive flexion degrees (data convention: negative = flexion)
    MAX_FLEX  = 130
    flexion   = np.clip(-Y_seq[:, 0], 0, MAX_FLEX)

    n_bins    = 26
    bins      = np.linspace(0, MAX_FLEX, n_bins + 1)
    hist, edges = np.histogram(flexion, bins=bins, density=True)
    ctrs_deg  = (edges[:-1] + edges[1:]) / 2
    # Map 0–MAX_FLEX degrees → 0–π radians on the semicircle
    ctrs_rad  = ctrs_deg * np.pi / MAX_FLEX
    bar_w     = (np.pi / MAX_FLEX) * (MAX_FLEX / n_bins) * 0.90

    cmap_g = plt.cm.RdYlGn_r
    for theta, h, deg in zip(ctrs_rad, hist, ctrs_deg):
        ax_g.bar(theta, h, width=bar_w, bottom=0,
                 color=cmap_g(deg / MAX_FLEX), alpha=0.88,
                 edgecolor="white", linewidth=0.4)

    # Reference lines at key clinical angles
    for ref_deg, lbl in [(30, "30°"), (60, "60°"), (90, "90°"), (120, "120°")]:
        ref_rad = ref_deg * np.pi / MAX_FLEX
        ax_g.plot([ref_rad, ref_rad], [0, hist.max() * 0.95],
                  color="#ccc", lw=0.9, ls="--", zorder=0)
        ax_g.text(ref_rad, hist.max() * 1.08, lbl,
                  ha="center", va="center", fontsize=7.5, color="#666")

    # Mean angle needle
    mean_flex = float(flexion.mean())
    mean_rad  = mean_flex * np.pi / MAX_FLEX
    ax_g.plot([mean_rad, mean_rad], [0, hist.max() * 0.88],
              color="#2c3e50", lw=2.5, zorder=5)
    ax_g.plot(mean_rad, hist.max() * 0.88, "v",
              color="#2c3e50", markersize=7, zorder=6)
    ax_g.text(mean_rad, hist.max() * 1.22, f"mean\n{mean_flex:.1f}°",
              ha="center", va="center", fontsize=8, fontweight="bold", color="#2c3e50")

    # Semicircle limits: show only 0–180° (upper arc)
    ax_g.set_thetamin(0)
    ax_g.set_thetamax(180)
    ax_g.set_theta_zero_location("E")   # extension (0°) at the right
    ax_g.set_theta_direction(1)         # increase counter-clockwise (flex goes left)
    ax_g.set_rticks([])

    # Axis labels at 0°, 30°, 60°, 90°, 120° flex
    tick_degs = [0, 30, 60, 90, 120]
    ax_g.set_xticks([d * np.pi / MAX_FLEX for d in tick_degs])
    ax_g.set_xticklabels(["0°\n(ext)", "30°", "60°", "90°", "120°"], fontsize=8)
    ax_g.set_title(
        f"Goniometer View\nKnee Flexion Distribution  ({fold0[5:]})",
        fontweight="bold", fontsize=9, pad=22,
    )

    plt.savefig(output, dpi=130, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Figure saved → {output}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    h5_files = sorted(glob.glob(os.path.join(args.data_dir, "fold_*/data.h5")))
    if args.fold:
        h5_files = [f for f in h5_files if f"fold_{args.fold}" in f]
    if not h5_files:
        print(f"No HDF5 files found under {args.data_dir}")
        return

    max_t = args.max_trials
    note  = f"(first {max_t} trials per split)" if max_t > 0 else "(all trials)"
    print(f"Analysing {len(h5_files)} fold(s) {note} ...")

    all_stats = {}
    for path in h5_files:
        fold_name = os.path.basename(os.path.dirname(path))
        print(f"  {fold_name} ...", end=" ", flush=True)
        all_stats[fold_name] = analyse_fold(path, max_t)
        print("done")

    h5_paths  = {os.path.basename(os.path.dirname(p)): p for p in h5_files}
    print_report(all_stats)
    make_figure(all_stats, h5_paths, args.output)


if __name__ == "__main__":
    main()
