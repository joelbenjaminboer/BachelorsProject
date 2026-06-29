"""Average evaluation metrics across LOSO folds.

Reads N metrics_final.json files (the schema saved by
src/training/eval.py's save_eval_artifacts, e.g. eval_encoder_final.json)
and reports mean +/- SD across folds for the overall metrics, per-activity
metrics, residual bias, and per-step RMSE curve.

Usage:
    python aggregate_final_metrics.py outputs/*/eval/metrics_final.json
    python aggregate_final_metrics.py fold_AB156.json fold_AB185.json ...
"""

import json
import sys
from pathlib import Path

import numpy as np

ACTIVITY_KEYS = ["mse", "mae", "rmse"]


def load_runs(paths):
    runs = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        d["_source"] = p
        runs.append(d)
    return runs


def mean_std(values):
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit("Usage: python aggregate_final_metrics.py <metrics_final.json> [...]")

    runs = load_runs(paths)
    n = len(runs)
    print(f"Aggregating {n} fold(s):")
    for r in runs:
        subject = next(iter(r["subject_metrics"]))
        print(f"  {r['_source']}  (subject={subject})")

    print("\nOverall (mean +/- SD across folds):")
    overall = {}
    for key in ACTIVITY_KEYS:
        mean, std = mean_std([r["overall"][key] for r in runs])
        overall[key] = {"mean": mean, "std": std}
        print(f"  {key:5s}: {mean:.4f} +/- {std:.4f}")

    activities = sorted({a for r in runs for a in r.get("activity_metrics", {})})
    activity_metrics = {}
    if activities:
        print("\nPer-activity (mean +/- SD across folds):")
    for act in activities:
        activity_metrics[act] = {}
        parts = []
        for key in ACTIVITY_KEYS:
            vals = [r["activity_metrics"][act][key] for r in runs if act in r.get("activity_metrics", {})]
            mean, std = mean_std(vals)
            activity_metrics[act][key] = {"mean": mean, "std": std}
            parts.append(f"{key}={mean:.3f}+/-{std:.3f}")
        print(f"  {act:12s} " + "  ".join(parts))

    res_mean, res_std = mean_std([r["residual_summary"]["mean"] for r in runs])
    print(f"\nResidual bias: {res_mean:.4f} +/- {res_std:.4f}")

    per_step_mean = {}
    per_step_std = {}
    sample_keys = set(runs[0].get("per_step", {}))
    for key in sample_keys:
        if all(key in r.get("per_step", {}) for r in runs):
            curves = np.array([r["per_step"][key] for r in runs], dtype=float)
            per_step_mean[key] = curves.mean(axis=0).tolist()
            per_step_std[key] = (curves.std(axis=0, ddof=1) if n > 1 else np.zeros(curves.shape[1])).tolist()

    subject_metrics = {}
    for r in runs:
        subject_metrics.update(r["subject_metrics"])

    out = {
        "n_folds": n,
        "sources": [r["_source"] for r in runs],
        "overall": overall,
        "activity_metrics": activity_metrics,
        "residual_bias_mean": {"mean": res_mean, "std": res_std},
        "per_step_mean": per_step_mean,
        "per_step_std": per_step_std,
        "subject_metrics": subject_metrics,
    }

    out_path = Path("aggregate_final_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved aggregate to {out_path}")


if __name__ == "__main__":
    main()
