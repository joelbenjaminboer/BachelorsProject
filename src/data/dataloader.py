"""Per-fold HDF5 loading and DataLoader construction.

Reads the LOSO fold tensors written by ``preprocessing.py``, computes z-score (or
min-max) normalisation statistics from the *training* subset only, and wraps them
in train/val/test ``DataLoader``s. Normalisation is applied lazily in
``HDF5KneeDataset.__getitem__`` — the stored tensors stay in raw physical units."""

import bisect
from dataclasses import dataclass
import os
from pathlib import Path
import random
from typing import Optional

import h5py
from loguru import logger
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


class HDF5KneeDataset(Dataset):
    def __init__(
        self,
        X: list[torch.Tensor],
        y: list[torch.Tensor],
        imu_mean=None,
        imu_std=None,
        y_mean=None,
        y_std=None,
        activity: list[torch.Tensor] | None = None,
        subject_id: str | None = None,
        is_train: bool = False,
        aug_cfg: dict | None = None,
        multitask: bool = False,
        y_vel: list[torch.Tensor] | None = None,
        y_vel_mean=None,
        y_vel_std=None,
        norm_type: str = "zscore",
        imu_min=None,
        imu_max=None,
    ):
        self.X = X
        self.y = y
        self.imu_mean = imu_mean
        self.imu_std = imu_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.activity = activity
        self.subject_id = subject_id
        self.is_train = is_train
        self.aug_cfg = aug_cfg or {}
        self.multitask = multitask
        self.y_vel = y_vel
        self.y_vel_mean = y_vel_mean
        self.y_vel_std = y_vel_std
        self.norm_type = norm_type
        self.imu_min = imu_min
        self.imu_max = imu_max

        self.total_samples = 0
        self.cumulative_sizes = []
        for x_arr in self.X:
            self.total_samples += x_arr.shape[0]
            self.cumulative_sizes.append(self.total_samples)

    def __len__(self):
        return self.total_samples

    def normalize_X(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured IMU normalisation to a tensor (for external callers).

        For ``minmax`` the per-subject min-max scaling to [-1, 1] is already baked
        into the stored data at preprocessing time, so this is an identity.
        """
        if self.norm_type == "minmax":
            return x
        if self.imu_mean is not None and self.imu_std is not None:
            return (x - self.imu_mean) / self.imu_std
        return x

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.aug_cfg

        # 1. Gaussian jitter
        sigma = float(cfg.get("jitter_sigma", 0.02))
        if sigma > 0 and self.imu_std is not None:
            x = x + torch.randn_like(x) * (sigma * self.imu_std)

        # 2. Magnitude scaling
        scale_lo = float(cfg.get("scale_lo", 0.9))
        scale_hi = float(cfg.get("scale_hi", 1.1))
        scale = torch.empty(1).uniform_(scale_lo, scale_hi).item()
        x = x * scale

        # 3. Time warping (linear warp ±time_warp_sigma fraction)
        warp_sigma = float(cfg.get("time_warp_sigma", 0.05))
        if warp_sigma > 0:
            T, C = x.shape
            # Generate a slightly warped monotone index sequence
            noise = torch.randn(T) * warp_sigma * T
            warped_idx = torch.linspace(0, T - 1, T) + noise
            warped_idx = warped_idx.clamp(0, T - 1)
            # Interpolate: use F.grid_sample on [1, C, T] tensor
            grid = (warped_idx / (T - 1)) * 2 - 1  # normalise to [-1, 1]
            grid_4d = grid.view(1, 1, T, 1).expand(1, 1, T, 1)
            x_4d = x.T.unsqueeze(0).unsqueeze(-1)  # [1, C, T, 1]
            x_warped = F.grid_sample(
                x_4d, grid_4d, mode="bilinear", padding_mode="border", align_corners=True
            )
            x = x_warped.squeeze(0).squeeze(-1).T  # [T, C]

        # 4. Sensor rotation (separate 3×3 rotation for accel and gyro)
        rot_sigma = float(cfg.get("rotation_sigma", 0.1))
        if rot_sigma > 0:
            for ch_start in (0, 3):  # accel channels 0-2, gyro channels 3-5
                R, _ = torch.linalg.qr(torch.eye(3) + rot_sigma * torch.randn(3, 3))
                x[:, ch_start : ch_start + 3] = x[:, ch_start : ch_start + 3] @ R.T

        # 5. Channel dropout
        drop_p = float(cfg.get("channel_drop_p", 0.1))
        if drop_p > 0:
            keep = (torch.rand(x.shape[-1]) > drop_p).float()
            x = x * keep

        return x

    def __getitem__(self, idx):
        trial_idx = bisect.bisect_left(self.cumulative_sizes, idx + 1)

        if trial_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[trial_idx - 1]

        past_imu = self.X[trial_idx][local_idx].float()
        future_knee = self.y[trial_idx][local_idx].float()

        # Extract velocity before any normalisation (raw physical units)
        if self.multitask and self.y_vel is not None:
            future_knee_vel = self.y_vel[trial_idx][local_idx].float()

        past_imu = self.normalize_X(past_imu)

        if self.is_train and self.aug_cfg.get("enabled", False):
            past_imu = self._augment(past_imu)

        if self.y_mean is not None and self.y_std is not None:
            future_knee = (future_knee - self.y_mean) / self.y_std

        out = {"past_imu": past_imu, "future_knee": future_knee}

        if self.multitask and self.y_vel is not None:
            if self.y_vel_mean is not None and self.y_vel_std is not None:
                future_knee_vel = (future_knee_vel - self.y_vel_mean) / self.y_vel_std
            out["future_knee_vel"] = future_knee_vel

        if self.activity is not None:
            out["activity_id"] = int(self.activity[trial_idx][local_idx].item())

        if self.subject_id is not None:
            out["subject_id"] = self.subject_id

        return out


def load_trials_from_hdf5(filepath: str):
    with h5py.File(filepath, "r") as f:

        def load_group(group):
            X_list, y_list, yv_list, a_list = [], [], [], []
            for key in sorted(f[group].keys()):
                if key.startswith("X_"):
                    idx = key.split("_")[1]
                    # Cast to float32 on load: the source arrays are float64
                    # (pandas default), which doubles resident RAM. The model
                    # trains in float32, so storing doubles wastes ~2x memory.
                    X = torch.tensor(f[f"{group}/X_{idx}"][:]).float()
                    y = torch.tensor(f[f"{group}/y_{idx}"][:]).float()
                    X_list.append(X)
                    y_list.append(y)

                    yv_key = f"{group}/yv_{idx}"
                    yv_list.append(torch.tensor(f[yv_key][:]).float() if yv_key in f else None)

                    a_key = f"{group}/a_{idx}"
                    if a_key in f:
                        a_list.append(torch.tensor(f[a_key][:]))
                    else:
                        a_list.append(None)
            return X_list, y_list, yv_list, a_list

        X_train, y_train, yv_train, act_train = load_group("train")
        X_val, y_val, yv_val, act_val = load_group("val")
        X_test, y_test, yv_test, act_test = load_group("test")
        return (
            X_train,
            y_train,
            yv_train,
            act_train,
            X_val,
            y_val,
            yv_val,
            act_val,
            X_test,
            y_test,
            yv_test,
            act_test,
        )


def _welford_update(
    n: int,
    mean: torch.Tensor,
    M2: torch.Tensor,
    chunk: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """One step of Chan's parallel update (Welford variant) for running mean/variance."""
    m = chunk.shape[0]
    chunk_mean = chunk.mean(dim=0)
    chunk_M2 = chunk.var(dim=0, unbiased=False) * m
    delta = chunk_mean - mean
    new_n = n + m
    new_mean = mean + delta * (m / new_n)
    new_M2 = M2 + chunk_M2 + delta**2 * (n * m / new_n)
    return new_n, new_mean, new_M2


def _compute_train_y_stats(y_train: list[torch.Tensor]):
    """Calculates Z-score statistics for the target over the training subset only."""
    n, mean, M2 = 0, torch.zeros(()), torch.zeros(())
    for t in y_train:
        flat = t.float().reshape(-1)
        n, mean, M2 = _welford_update(n, mean, M2, flat.unsqueeze(-1))
    mean = mean.squeeze()
    std = torch.clamp(torch.sqrt(M2.squeeze() / max(n, 1)), min=1e-8)
    logger.debug(
        "y_train stats — mean: {:.2f}°, std: {:.2f}° (physiological range: mean 15–30°, std 20–35°)",
        mean.item(),
        std.item(),
    )
    return mean, std


def _compute_train_yvel_stats(yv_train: list[torch.Tensor]):
    """Calculates Z-score statistics for knee velocity over the training subset only."""
    n, mean, M2 = 0, torch.zeros(()), torch.zeros(())
    for t in yv_train:
        flat = t.float().reshape(-1)
        n, mean, M2 = _welford_update(n, mean, M2, flat.unsqueeze(-1))
    mean = mean.squeeze()
    std = torch.clamp(torch.sqrt(M2.squeeze() / max(n, 1)), min=1e-8)
    logger.debug(
        "y_vel stats — mean: {:.4f} deg/s, std: {:.4f} deg/s",
        mean.item(),
        std.item(),
    )
    return mean, std


def _compute_train_imu_stats(X_train: list[torch.Tensor]):
    """Calculates Z-score statistics over the training subset only."""
    if not X_train:
        return torch.zeros(6), torch.ones(6)

    n_features = X_train[0].shape[-1]
    n, mean, M2 = 0, torch.zeros(n_features), torch.zeros(n_features)
    for x in X_train:
        # x: [windows, seq_len, features] → flatten to [windows*seq_len, features]
        chunk = x.float().reshape(-1, n_features)
        n, mean, M2 = _welford_update(n, mean, M2, chunk)

    if n == 0:
        return torch.zeros(n_features), torch.ones(n_features)

    std = torch.clamp(torch.sqrt(M2 / n), min=1e-8)
    return mean.float(), std.float()


def _compute_train_imu_minmax(X_train: list[torch.Tensor]):
    """Calculates per-feature min/max over the training subset for min-max normalisation."""
    if not X_train:
        return torch.zeros(6), torch.ones(6)

    n_features = X_train[0].shape[-1]
    global_min = torch.full((n_features,), float("inf"))
    global_max = torch.full((n_features,), float("-inf"))
    for x in X_train:
        chunk = x.float().reshape(-1, n_features)
        global_min = torch.minimum(global_min, chunk.min(dim=0).values)
        global_max = torch.maximum(global_max, chunk.max(dim=0).values)

    return global_min.float(), global_max.float()


def _none_free(lst: list) -> list | None:
    """Return list if all elements are not None, else None."""
    return lst if all(a is not None for a in lst) else None


@dataclass
class FoldData:
    """Raw per-fold tensors + train-subset normalisation stats, loaded once from HDF5.

    Decouples the expensive disk read / stats computation from the cheap
    Dataset/DataLoader wrapping so callers (e.g. the Optuna search) can load a
    fold once and rebuild loaders per trial without holding multiple copies of
    the dataset in RAM. Velocity stats are computed whenever velocity targets are
    present, so the same FoldData serves both multitask=True and =False trials.
    """

    X_train: list
    y_train: list
    yv_train: Optional[list]
    act_train: Optional[list]
    X_val: list
    y_val: list
    yv_val: Optional[list]
    act_val: Optional[list]
    X_test: list
    y_test: list
    yv_test: Optional[list]
    act_test: Optional[list]
    imu_mean: torch.Tensor
    imu_std: torch.Tensor
    imu_min: Optional[torch.Tensor]
    imu_max: Optional[torch.Tensor]
    norm_type: str
    y_mean: torch.Tensor
    y_std: torch.Tensor
    y_vel_mean: Optional[torch.Tensor]
    y_vel_std: Optional[torch.Tensor]
    subject_id: str
    data_dir: str


def load_fold_data(
    data_dir: str, normalization: str = "zscore", normalize_target: bool = True
) -> FoldData:
    """Read a fold's tensors from HDF5 and compute train-subset normalisation stats."""
    data_file = Path(data_dir) / "data.h5"
    if not data_file.exists():
        raise FileNotFoundError(f"data.h5 not found in {data_dir}. Run preprocessing first.")

    (
        X_train,
        y_train,
        yv_train,
        act_train,
        X_val,
        y_val,
        yv_val,
        act_val,
        X_test,
        y_test,
        yv_test,
        act_test,
    ) = load_trials_from_hdf5(str(data_file))

    norm_type = "minmax" if normalization == "height_minmax" else "zscore"
    logger.info("Computing IMU normalisation stats from training subset (mode={})", normalization)
    # For height_minmax the per-subject min-max scaling to [-1, 1] is baked into
    # the HDF5 at preprocessing; the dataloader only needs z-score stats (used in
    # zscore mode) and applies no further IMU scaling in minmax mode.
    imu_mean, imu_std = _compute_train_imu_stats(X_train)
    imu_min = imu_max = None

    if normalize_target:
        y_mean, y_std = _compute_train_y_stats(y_train)
    else:
        # Target stays in raw physical units (degrees). Storing 0/1 makes every
        # downstream de-norm (eval, baselines, val RMSE) an identity automatically.
        logger.info("Target normalisation disabled — knee angle kept in raw degrees")
        y_mean, y_std = torch.tensor(0.0), torch.tensor(1.0)

    yv_train_clean = _none_free(yv_train)
    yv_val_clean = _none_free(yv_val)
    yv_test_clean = _none_free(yv_test)

    # Always compute velocity stats when present so one FoldData serves both
    # multitask=True and =False trials (the loaders ignore them when not needed).
    y_vel_mean = y_vel_std = None
    if yv_train_clean:
        y_vel_mean, y_vel_std = _compute_train_yvel_stats(yv_train_clean)

    act_train_clean = _none_free(act_train)
    act_val_clean = _none_free(act_val)
    act_test_clean = _none_free(act_test)

    if act_train_clean:
        from collections import Counter

        counts = Counter(int(v) for a in act_train_clean for v in a.tolist())
        logger.info("Train activity distribution: {}", dict(sorted(counts.items())))

    return FoldData(
        X_train=X_train,
        y_train=y_train,
        yv_train=yv_train_clean,
        act_train=act_train_clean,
        X_val=X_val,
        y_val=y_val,
        yv_val=yv_val_clean,
        act_val=act_val_clean,
        X_test=X_test,
        y_test=y_test,
        yv_test=yv_test_clean,
        act_test=act_test_clean,
        imu_mean=imu_mean,
        imu_std=imu_std,
        imu_min=imu_min,
        imu_max=imu_max,
        norm_type=norm_type,
        y_mean=y_mean,
        y_std=y_std,
        y_vel_mean=y_vel_mean,
        y_vel_std=y_vel_std,
        subject_id=Path(data_dir).name.replace("fold_", ""),
        data_dir=str(data_dir),
    )


def build_loaders_from_fold(
    fold: FoldData,
    batch_size: int,
    seed: int = 42,
    num_workers: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
    pin_memory: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    aug_cfg: Optional[dict] = None,
    multitask: bool = False,
):
    """Wrap a pre-loaded FoldData in Datasets + DataLoaders. Cheap; no disk read.

    Only ``batch_size``, ``multitask`` and loader settings vary per call — the
    underlying tensors are shared by reference, so calling this repeatedly (e.g.
    once per Optuna trial) does not duplicate the dataset in memory.
    """
    yv_train = fold.yv_train if multitask else None
    yv_val = fold.yv_val if multitask else None
    yv_test = fold.yv_test if multitask else None
    y_vel_mean = fold.y_vel_mean if multitask else None
    y_vel_std = fold.y_vel_std if multitask else None

    _norm_kwargs = dict(norm_type=fold.norm_type, imu_min=fold.imu_min, imu_max=fold.imu_max)

    train_dataset = HDF5KneeDataset(
        fold.X_train,
        fold.y_train,
        fold.imu_mean,
        fold.imu_std,
        fold.y_mean,
        fold.y_std,
        activity=fold.act_train,
        is_train=True,
        aug_cfg=aug_cfg,
        multitask=multitask,
        y_vel=yv_train,
        y_vel_mean=y_vel_mean,
        y_vel_std=y_vel_std,
        **_norm_kwargs,
    )
    val_dataset = HDF5KneeDataset(
        fold.X_val,
        fold.y_val,
        fold.imu_mean,
        fold.imu_std,
        fold.y_mean,
        fold.y_std,
        activity=fold.act_val,
        multitask=multitask,
        y_vel=yv_val,
        y_vel_mean=y_vel_mean,
        y_vel_std=y_vel_std,
        **_norm_kwargs,
    )
    test_dataset = HDF5KneeDataset(
        fold.X_test,
        fold.y_test,
        fold.imu_mean,
        fold.imu_std,
        fold.y_mean,
        fold.y_std,
        activity=fold.act_test,
        subject_id=fold.subject_id,
        multitask=multitask,
        y_vel=yv_test,
        y_vel_mean=y_vel_mean,
        y_vel_std=y_vel_std,
        **_norm_kwargs,
    )

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if num_workers == 0:
        persistent_workers = False
    elif persistent_workers is None:
        persistent_workers = num_workers > 0

    common_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "pin_memory": bool(pin_memory),
        "worker_init_fn": _seed_worker if num_workers > 0 else None,
    }

    if num_workers > 0 and prefetch_factor is not None:
        common_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=train_loader_generator,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_kwargs)

    split_info = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "fold_dir": fold.data_dir,
        "y_mean": fold.y_mean.item(),
        "y_std": fold.y_std.item(),
        "n_features": int(fold.X_train[0].shape[-1]) if fold.X_train else None,
    }

    if y_vel_mean is not None:
        split_info["y_vel_mean"] = y_vel_mean.item()
        split_info["y_vel_std"] = y_vel_std.item()

    return train_loader, val_loader, test_loader, split_info


def build_dataloaders(
    data_dir: str,
    batch_size: int,
    seed: int = 42,
    num_workers: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
    pin_memory: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    aug_cfg: Optional[dict] = None,
    multitask: bool = False,
    normalization: str = "zscore",
    normalize_target: bool = True,
):
    """Load a fold from disk and build its train/val/test DataLoaders."""
    fold = load_fold_data(data_dir, normalization=normalization, normalize_target=normalize_target)
    return build_loaders_from_fold(
        fold,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        aug_cfg=aug_cfg,
        multitask=multitask,
    )
