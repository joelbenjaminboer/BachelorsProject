"""Calculate average gait cycle duration and step count from ENABL3S raw data.

A gait cycle (stride) is defined by consecutive peaks of the knee flexion angle
during level-ground walking (Mode 1) — one peak per swing phase.

A step is defined by each stance-phase trough (local minimum). Two steps occur
per gait cycle: one for each leg passing through stance. Step count is computed
per trial and averaged per subject.
"""

import io
import zipfile
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import decimate, find_peaks


RAW_DIR = "data/raw/ENABL3S"
ORIGINAL_FREQ = 500
TARGET_FREQ = 50
DOWNSAMPLE_FACTOR = ORIGINAL_FREQ // TARGET_FREQ

# Minimum peak distance: 0.5 s (a stride is never shorter than this)
MIN_PEAK_DISTANCE_S = 0.5
MIN_PEAK_DISTANCE = int(MIN_PEAK_DISTANCE_S * TARGET_FREQ)

# Minimum peak prominence to ignore noise (degrees)
MIN_PEAK_PROMINENCE = 5.0


def _load_knee_angle(zip_path: str, member: str) -> np.ndarray | None:
    """Return the right knee angle array (500 Hz) for one trial, walking only."""
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as handle:
            df = pd.read_csv(io.TextIOWrapper(handle, encoding="utf-8"))

    if "Right_Knee" not in df.columns or "Mode" not in df.columns:
        return None

    df = df[df["Mode"] == 1]  # level ground walking only
    if len(df) < ORIGINAL_FREQ:  # skip very short segments
        return None

    return df["Right_Knee"].values.astype(np.float32)


def _downsample(arr: np.ndarray) -> np.ndarray:
    return decimate(arr, q=DOWNSAMPLE_FACTOR, ftype="iir", zero_phase=True).copy()


def _cycle_durations_s(knee_angle_100hz: np.ndarray) -> np.ndarray:
    """Return per-cycle durations in seconds from a 100 Hz knee angle signal."""
    peaks, _ = find_peaks(
        knee_angle_100hz,
        distance=MIN_PEAK_DISTANCE,
        prominence=MIN_PEAK_PROMINENCE,
    )
    if len(peaks) < 2:
        return np.array([])
    intervals_samples = np.diff(peaks)
    return intervals_samples / TARGET_FREQ


def compute_avg_gait_cycle_duration(raw_dir: str = RAW_DIR) -> dict:
    zip_files = sorted(glob(str(Path(raw_dir) / "AB*.zip")))
    if not zip_files:
        raise FileNotFoundError(f"No AB*.zip files found in {raw_dir}")

    all_durations: list[float] = []
    per_subject: dict[str, float] = {}

    for zip_path in zip_files:
        subject_id = Path(zip_path).stem
        subject_durations: list[float] = []

        with zipfile.ZipFile(zip_path) as archive:
            trial_members = sorted(
                m
                for m in archive.namelist()
                if m.lower().startswith(f"{subject_id.lower()}/processed/")
                and m.lower().endswith(".csv")
            )[:40]

        for member in trial_members:
            raw_angle = _load_knee_angle(zip_path, member)
            if raw_angle is None:
                continue
            angle_100hz = _downsample(raw_angle)
            durations = _cycle_durations_s(angle_100hz)
            subject_durations.extend(durations.tolist())

        if subject_durations:
            avg = float(np.mean(subject_durations))
            per_subject[subject_id] = avg
            all_durations.extend(subject_durations)
            print(f"{subject_id}: {avg:.3f} s  (n={len(subject_durations)} cycles)")

    if not all_durations:
        raise RuntimeError("No gait cycles detected — check data paths and column names.")

    overall_avg = float(np.mean(all_durations))
    overall_std = float(np.std(all_durations))
    avg_samples = overall_avg * TARGET_FREQ

    print(f"\nOverall average gait cycle duration: {overall_avg:.3f} ± {overall_std:.3f} s")
    print(f"Average gait cycle length: {avg_samples:.1f} samples @ {TARGET_FREQ} Hz")
    print(f"Across {len(all_durations)} cycles, {len(per_subject)} subjects")

    return {
        "mean_s": overall_avg,
        "std_s": overall_std,
        "mean_samples": avg_samples,
        "n_cycles": len(all_durations),
        "n_subjects": len(per_subject),
        "per_subject": per_subject,
    }


if __name__ == "__main__":
    compute_avg_gait_cycle_duration()
