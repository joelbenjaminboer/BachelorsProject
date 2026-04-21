import os
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

IMU_COLS = ("Ax", "Ay", "Az", "Gy", "Gz", "Gx")


def _normalize_subjects(subjects: Optional[Sequence[str]]) -> list[str]:
    if subjects is None:
        return []
    if isinstance(subjects, str):
        subjects = [subjects]

    normalized = []
    for subject in subjects:
        if subject is None:
            continue
        subject_name = str(subject).strip()
        if subject_name:
            normalized.append(subject_name)

    return sorted(set(normalized))


def _subjects_from_dataset(dataset) -> list[str]:
    if dataset is None:
        return []

    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        if len(dataset.indices) == 0:
            return []
        return sorted({base_dataset.window_subject_ids[idx] for idx in dataset.indices})

    if hasattr(dataset, "window_subject_ids"):
        return sorted(set(dataset.window_subject_ids))

    return []


class IMUKneeDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        seq_length: int,
        forecast_horizon: int,
        include_subjects: Optional[Sequence[str]] = None,
        exclude_subjects: Optional[Sequence[str]] = None,
    ):
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.window_size = seq_length + forecast_horizon

        self.include_subjects = set(_normalize_subjects(include_subjects))
        self.exclude_subjects = set(_normalize_subjects(exclude_subjects))

        self.imu_sequences: list[torch.Tensor] = []
        self.knee_sequences: list[torch.Tensor] = []
        self.valid_indices: list[tuple[int, int]] = []
        self.window_subject_ids: list[str] = []
        self.window_start_indices: list[int] = []

        self.load_data(data_dir)

    def _keep_subject(self, subject_id: str) -> bool:
        if self.include_subjects and subject_id not in self.include_subjects:
            return False
        if self.exclude_subjects and subject_id in self.exclude_subjects:
            return False
        return True

    def load_data(self, data_dir: str):
        path = Path(data_dir)
        files = sorted(path.rglob("*.parquet"))

        for file_path in files:
            subject_id = file_path.parent.name
            if not self._keep_subject(subject_id):
                continue

            df = pd.read_parquet(file_path)
            if "KneeAngle" not in df.columns or not all(col in df.columns for col in IMU_COLS):
                continue

            imu_values = torch.tensor(df[list(IMU_COLS)].values, dtype=torch.float32)
            knee_values = torch.tensor(df["KneeAngle"].values, dtype=torch.float32)

            num_windows = len(imu_values) - self.window_size + 1
            if num_windows <= 0:
                continue

            seq_idx = len(self.imu_sequences)
            self.imu_sequences.append(imu_values)
            self.knee_sequences.append(knee_values)

            self.valid_indices.extend((seq_idx, start_idx) for start_idx in range(num_windows))
            self.window_subject_ids.extend([subject_id] * num_windows)
            self.window_start_indices.extend(range(num_windows))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        seq_idx, start_idx = self.valid_indices[idx]

        past_imu = self.imu_sequences[seq_idx][start_idx : start_idx + self.seq_length]
        target_idx = start_idx + self.seq_length
        future_knee = self.knee_sequences[seq_idx][target_idx : target_idx + self.forecast_horizon]

        return {
            "past_imu": past_imu,
            "future_knee": future_knee,
            "subject_id": self.window_subject_ids[idx],
            "window_start_idx": self.window_start_indices[idx],
        }


def build_pretrain_dataloaders(
    data_dir: str,
    seq_length: int,
    forecast_horizon: int,
    batch_size: int,
    split_ratio: float = 0.9,
    split_strategy: str = "loso",
    holdout_subjects: Optional[Sequence[str]] = None,
    include_test_loader: bool = True,
    seed: int = 42,
    num_workers: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
):
    normalized_strategy = str(split_strategy).strip().lower()
    if normalized_strategy not in {"loso", "leave_one_subject_out", "random"}:
        raise ValueError(
            "split_strategy must be one of {'loso', 'leave_one_subject_out', 'random'}"
        )

    holdout_subjects = _normalize_subjects(holdout_subjects)
    if normalized_strategy in {"loso", "leave_one_subject_out"} and not holdout_subjects:
        raise ValueError("LOSO split requires at least one holdout subject in config")
    if normalized_strategy == "random":
        holdout_subjects = []

    train_val_dataset = IMUKneeDataset(
        data_dir=data_dir,
        seq_length=seq_length,
        forecast_horizon=forecast_horizon,
        exclude_subjects=holdout_subjects if holdout_subjects else None,
    )
    if len(train_val_dataset) == 0:
        raise ValueError("No train/val samples available after applying subject filters")

    if len(train_val_dataset) == 1:
        train_dataset = train_val_dataset
        val_dataset = Subset(train_val_dataset, [])
    else:
        train_ratio = min(max(float(split_ratio), 0.0), 1.0)
        train_size = int(train_ratio * len(train_val_dataset))
        train_size = max(1, min(train_size, len(train_val_dataset) - 1))
        val_size = len(train_val_dataset) - train_size

        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(
            train_val_dataset,
            [train_size, val_size],
            generator=generator,
        )

    test_dataset = None
    if include_test_loader and holdout_subjects:
        test_dataset = IMUKneeDataset(
            data_dir=data_dir,
            seq_length=seq_length,
            forecast_horizon=forecast_horizon,
            include_subjects=holdout_subjects,
        )
        if len(test_dataset) == 0:
            raise ValueError(
                "No test samples found for holdout subjects: " + ", ".join(holdout_subjects)
            )

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)

    if num_workers == 0:
        persistent_workers = False
    elif persistent_workers is None:
        persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    split_info = {
        "split_strategy": normalized_strategy,
        "holdout_subjects": holdout_subjects,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset) if test_dataset is not None else 0,
        "train_subjects": _subjects_from_dataset(train_dataset),
        "val_subjects": _subjects_from_dataset(val_dataset),
        "test_subjects": _subjects_from_dataset(test_dataset),
    }

    return train_loader, val_loader, test_loader, split_info
