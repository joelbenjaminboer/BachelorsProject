"""
ENABL3S dataset analysis orchestrator.

Usage (from project root):
    python -m analysis.analyze_dataset                        # full run
    python -m analysis.analyze_dataset --sample-frac 0.1     # fast dev run (10% data)
    python -m analysis.analyze_dataset --data-dir /path/to/ENABL3S --out-dir /path/to/plots
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts and SSH

import seaborn as sns

from analysis.data_io import build_file_catalog, compute_file_stats, load_all_data, load_subject_sample_files
from analysis import (
    plot_balance,
    plot_correlations,
    plot_distributions,
    plot_knee_angle,
    plot_overview,
    plot_stats_table,
    plot_timeseries,
)

_PROJECT_ROOT = Path(__file__).parent.parent


def main(data_dir: Path, out_dir: Path, sample_frac: float = 1.0) -> None:
    sns.set_theme(style="whitegrid", palette="deep", context="talk")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building file catalog...")
    catalog = build_file_catalog(data_dir)
    print(f"  Found {len(catalog)} recordings across {catalog['subject'].nunique()} subjects")

    print("Computing file statistics from parquet metadata...")
    file_stats = compute_file_stats(catalog)
    total_rows = file_stats["n_rows"].sum()
    print(f"  Total samples: {total_rows:,} ({total_rows / 1e6:.1f}M)")

    print(f"Loading dataset ({sample_frac * 100:.0f}% sample)...")
    df_all = load_all_data(catalog, sample_frac=sample_frac)
    print(f"  Loaded {len(df_all):,} rows")

    print("Loading per-subject sample files for time-series plots...")
    sample_files = load_subject_sample_files(catalog, file_stats)

    print("\n[1/7] Dataset overview plots...")
    plot_overview.plot_file_count(catalog, out_dir)
    plot_overview.plot_row_count(file_stats, out_dir)
    plot_overview.plot_recording_length_dist(file_stats, out_dir)

    print("[2/7] Signal distribution plots...")
    plot_distributions.plot_overall_histograms(df_all, out_dir)
    plot_distributions.plot_all_per_subject_kdes(df_all, out_dir)

    print("[3/7] Statistics table...")
    stats_df = plot_stats_table.compute_channel_stats(df_all)
    plot_stats_table.save_stats_csv(stats_df, out_dir.parent)
    plot_stats_table.plot_stats_table(stats_df, out_dir)

    print("[4/7] Time-series sample traces...")
    plot_timeseries.plot_all_timeseries(sample_files, out_dir)

    print("[5/7] Knee angle analysis...")
    plot_knee_angle.plot_knee_overall_hist(df_all, out_dir)
    plot_knee_angle.plot_knee_per_subject_violin(df_all, out_dir)
    plot_knee_angle.plot_knee_range_per_subject(df_all, out_dir)
    plot_knee_angle.plot_knee_left_vs_right(df_all, out_dir)

    print("[6/7] Correlation heatmaps...")
    plot_correlations.plot_correlation_heatmap(df_all, out_dir)
    plot_correlations.plot_per_subject_correlations(df_all, out_dir)

    print("[7/7] Recording balance plots...")
    plot_balance.plot_leg_side_balance(catalog, out_dir)
    plot_balance.plot_pre_post_balance(catalog, out_dir)

    plots = sorted(out_dir.glob("*.png"))
    print(f"\nDone. {len(plots)} plots saved to {out_dir}/")
    for p in plots:
        print(f"  {p.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ENABL3S dataset analysis")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "processed" / "ENABL3S",
        help="Path to processed ENABL3S directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default= _PROJECT_ROOT / "reports" / "figures" / "analysis_plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Fraction of data to load per file (0-1). Use 0.1 for fast dev runs.",
    )
    args = parser.parse_args()
    main(args.data_dir, args.out_dir, args.sample_frac)
