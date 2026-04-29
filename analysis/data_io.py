"""Shared data loading utilities and constants for ENABL3S analysis."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

SUBJECTS = [
    "AB156", "AB185", "AB186", "AB188", "AB189",
    "AB190", "AB191", "AB192", "AB193", "AB194",
]

# Column order matches preprocessing.py exactly (not alphabetical)
ALL_CHANNELS = ["Ax", "Ay", "Az", "Gy", "Gz", "Gx"]
ALL_COLS = ALL_CHANNELS + ["KneeAngle"]

SAMPLE_RATE_HZ = 100  # ENABL3S records at 100 Hz

SUBJECT_PALETTE: dict[str, tuple] = dict(
    zip(SUBJECTS, sns.color_palette("tab10", n_colors=10))
)


def build_file_catalog(data_dir: Path) -> pd.DataFrame:
    """Scan data_dir for parquet files, parse metadata from filenames only."""
    records = []
    for fp in sorted(data_dir.glob("**/*.parquet")):
        # Filename format: {SubjectID}_Circuit_{NNN}_{pre/post}_{left/right}.parquet
        parts = fp.stem.split("_")
        subject = parts[0]
        circuit_num = parts[2] if len(parts) > 2 else ""
        pre_post = parts[3] if len(parts) > 3 else ""
        leg = parts[4] if len(parts) > 4 else ""
        records.append({
            "subject": subject,
            "filepath": fp,
            "circuit_num": circuit_num,
            "pre_post": pre_post,
            "leg": leg,
        })
    return pd.DataFrame(records)


def compute_file_stats(catalog: pd.DataFrame) -> pd.DataFrame:
    """Read row counts from parquet footer metadata (fast, no data loaded)."""
    rows = []
    for _, row in catalog.iterrows():
        meta = pq.read_metadata(row["filepath"])
        rows.append({
            "subject": row["subject"],
            "filepath": row["filepath"],
            "n_rows": meta.num_rows,
            "leg": row["leg"],
            "pre_post": row["pre_post"],
        })
    return pd.DataFrame(rows)


def load_all_data(
    catalog: pd.DataFrame,
    sample_frac: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Load all parquet files, add subject/leg/pre_post columns, return concatenated DataFrame."""
    frames = []
    for _, row in catalog.iterrows():
        df = pd.read_parquet(row["filepath"], columns=ALL_COLS)
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=seed)
        df["subject"] = row["subject"]
        df["leg"] = row["leg"]
        df["pre_post"] = row["pre_post"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_subject_sample_files(
    catalog: pd.DataFrame,
    file_stats: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """For each subject, load the median-length recording file."""
    merged = catalog.merge(file_stats[["filepath", "n_rows"]], on="filepath")
    result = {}
    for subject, grp in merged.groupby("subject"):
        median_rows = grp["n_rows"].median()
        chosen = grp.iloc[(grp["n_rows"] - median_rows).abs().argsort().iloc[0]]
        result[subject] = pd.read_parquet(chosen["filepath"], columns=ALL_COLS)
    return result
