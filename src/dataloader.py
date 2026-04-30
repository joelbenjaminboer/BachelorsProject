import os
import random
from pathlib import Path
from typing import Optional, Sequence, Tuple

import h5py
import torch
from torch.utils.data import DataLoader, Dataset

def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)

class HDF5KneeDataset(Dataset):
    def __init__(self, X: list[torch.Tensor], y: list[torch.Tensor], imu_mean=None, imu_std=None):
        self.X = X
        self.y = y
        self.imu_mean = imu_mean
        self.imu_std = imu_std
        
        # Flattens sizes since we get lists of trial tensors
        self.total_samples = 0
        self.cumulative_sizes = []
        for x_arr in self.X:
            self.total_samples += x_arr.shape[0]
            self.cumulative_sizes.append(self.total_samples)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Binary search or simple linear scan for the right trial
        # For simplicity, a small loop (max 40 * subjects)
        trial_idx = 0
        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                trial_idx = i
                break
        
        # Calculate local index within the trial
        if trial_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[trial_idx - 1]
            
        past_imu = self.X[trial_idx][local_idx].float()
        future_knee = self.y[trial_idx][local_idx].float()
        
        if self.imu_mean is not None and self.imu_std is not None:
            past_imu = (past_imu - self.imu_mean) / self.imu_std

        return {
            "past_imu": past_imu,
            "future_knee": future_knee
        }

def load_trials_from_hdf5(filepath: str):
    with h5py.File(filepath, 'r') as f:
        def load_group(group):
            X_list, y_list = [], []
            for key in sorted(f[group].keys()):
                if key.startswith('X_'):
                    idx = key.split('_')[1]
                    X = torch.tensor(f[f'{group}/X_{idx}'][:])
                    y = torch.tensor(f[f'{group}/y_{idx}'][:])
                    X_list.append(X)
                    y_list.append(y)
            return X_list, y_list

        X_train, y_train = load_group('train')
        X_val, y_val = load_group('val')
        X_test, y_test = load_group('test')
        return X_train, y_train, X_val, y_val, X_test, y_test

def _compute_train_imu_stats(X_train: list[torch.Tensor]):
    """Calculates Z-score statistics over the training subset only."""
    count = sum(x.numel() for x in X_train)
    if count == 0:
        return torch.zeros(6), torch.ones(6)

    # Flatten list of 3D tensors (trials, seq_len, 6) or (samples, seq_len, 6)
    # Actually, X_train is list of (num_windows, seq_len, 6) arrays
    # Concat along dimension 0 and compute stats
    all_x = torch.cat(X_train, dim=0) # [total_windows, seq_len, 6]
    mean = all_x.mean(dim=(0, 1))
    std = all_x.std(dim=(0, 1))
    std = torch.clamp(std, min=1e-8)
    return mean.float(), std.float()

def build_pretrain_dataloaders(
    data_dir: str, # This should now point to a specific fold directory, e.g. processed_dir/fold_1
    batch_size: int,
    seed: int = 42,
    num_workers: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
    pin_memory: Optional[bool] = None,
    **kwargs # Accept unused args for compatibility
):
    data_file = Path(data_dir) / "data.h5"
    if not data_file.exists():
        raise FileNotFoundError(f"data.h5 not found in {data_dir}. Run preprocessing first.")

    X_train, y_train, X_val, y_val, X_test, y_test = load_trials_from_hdf5(str(data_file))

    print("Computing IMU standard scaling from the training subset...")
    imu_mean, imu_std = _compute_train_imu_stats(X_train)

    train_dataset = HDF5KneeDataset(X_train, y_train, imu_mean, imu_std)
    val_dataset = HDF5KneeDataset(X_val, y_val, imu_mean, imu_std)
    test_dataset = HDF5KneeDataset(X_test, y_test, imu_mean, imu_std)

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

    train_loader_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_dataset, shuffle=True, generator=train_loader_generator, **common_loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_kwargs)

    split_info = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "fold_dir": str(data_dir)
    }

    return train_loader, val_loader, test_loader, split_info
