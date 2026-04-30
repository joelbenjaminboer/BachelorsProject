"""Shared data loading utilities and constants for ENABL3S analysis."""
from pathlib import Path
import h5py
import pandas as pd
import numpy as np

SUBJECTS = [
    "AB156", "AB185", "AB186", "AB188", "AB189",
    "AB190", "AB191", "AB192", "AB193", "AB194",
]

ALL_CHANNELS = ["Ax", "Ay", "Az", "Gy", "Gz", "Gx"]
ALL_COLS = ALL_CHANNELS + ["KneeAngle"]
SAMPLE_RATE_HZ = 100

def _load_h5_group_as_df(filepath, group_name):
    frames = []
    with h5py.File(filepath, 'r') as f:
        grp = f[group_name]
        for key in sorted(grp.keys()):
            if key.startswith('X_'):
                idx = key.split('_')[1]
                X = grp[f'X_{idx}'][...]
                y = grp[f'y_{idx}'][...]
                # Concatenate along feature dimension
                data = np.concatenate([X, y], axis=1)
                df = pd.DataFrame(data, columns=ALL_COLS)
                # No detailed metadata exists in HDF5 besides test/train split
                df["subject"] = "unknown"
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=ALL_COLS + ["subject"])
    return pd.concat(frames, ignore_index=True)

def load_all_data(data_dir: Path, sample_frac: float = 1.0, seed: int = 42) -> pd.DataFrame:
    # Load raw data from fold 1
    raw_h5 = Path(data_dir) / "fold_1" / "raw_data.h5"
    if not raw_h5.exists():
        return pd.DataFrame()
    train_df = _load_h5_group_as_df(raw_h5, 'train')
    val_df = _load_h5_group_as_df(raw_h5, 'val')
    test_df = _load_h5_group_as_df(raw_h5, 'test')
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)
    return df

def load_subject_sample_files(data_dir: Path) -> dict[str, pd.DataFrame]:
    raw_h5 = Path(data_dir) / "fold_1" / "raw_data.h5"
    if not raw_h5.exists():
        return {}
    test_df = _load_h5_group_as_df(raw_h5, 'test')
    return {"Test_Subject_Fold_1": test_df}
