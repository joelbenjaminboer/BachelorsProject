"""ENABL3S preprocessing: read raw IMU CSVs from per-subject ZIPs, filter
activities, downsample 500 Hz → 100 Hz, build sliding windows, and write one
HDF5 file per Leave-One-Subject-Out fold under ``data/processed/``."""

from glob import glob
import io
import os
import random
import zipfile

import h5py
import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm

# Heights (cm) from "Subject Trigger Channel Feature Information.xlsx".
ENABL3S_SUBJECT_HEIGHTS_CM: dict[str, float] = {
    "AB156": 193,
    "AB185": 181,
    "AB186": 163,
    "AB188": 185,
    "AB189": 178,
    "AB190": 160,
    "AB191": 163,
    "AB192": 170,
    "AB193": 185,
    "AB194": 160,
}

_DEFAULT_HEIGHT_REF_CM = float(np.mean(list(ENABL3S_SUBJECT_HEIGHTS_CM.values())))  # ≈ 173.8 cm


def _append_handcrafted_features(X_raw: np.ndarray) -> np.ndarray:
    """Append the paper's 4 handcrafted features (eqs 1–4) to the 6 raw IMU
    channels, yielding 10: [Ax,Ay,Az,Gx,Gy,Gz, Accel_L2,Gyro_L2,Accel_M,Gyro_M].

    Computed on the (already height-scaled) accel/gyro so the derived features
    inherit the same body-size compensation.
    """
    accel = X_raw[:, :3]
    gyro = X_raw[:, 3:6]
    accel_l2 = np.sqrt(np.sum(accel**2, axis=1, keepdims=True))
    gyro_l2 = np.sqrt(np.sum(gyro**2, axis=1, keepdims=True))
    accel_m = np.mean(accel, axis=1, keepdims=True)
    gyro_m = np.mean(gyro, axis=1, keepdims=True)
    return np.concatenate([X_raw, accel_l2, gyro_l2, accel_m, gyro_m], axis=1)


def _per_subject_minmax(Xw_trials: list[np.ndarray]) -> list[np.ndarray]:
    """Scale every window of a single subject to [-1, 1] per feature, using that
    subject's own min/max computed across all its windows.

    Memory-frugal: min/max are accumulated by streaming over trials (no big
    concatenated copy), and the scaling is done in-place in float32, replacing
    each trial array as it goes so the float64 originals are freed immediately.
    """
    n_feat = Xw_trials[0].shape[-1]
    smin = np.full(n_feat, np.inf, dtype=np.float64)
    smax = np.full(n_feat, -np.inf, dtype=np.float64)
    for w in Xw_trials:
        flat = w.reshape(-1, n_feat)
        smin = np.minimum(smin, flat.min(axis=0))
        smax = np.maximum(smax, flat.max(axis=0))

    smin = smin.astype(np.float32)
    inv = (2.0 / np.clip(smax - smin, 1e-8, None)).astype(np.float32)
    for i, w in enumerate(Xw_trials):
        wf = w.astype(np.float32, copy=False)
        wf -= smin
        wf *= inv
        wf -= 1.0
        Xw_trials[i] = wf  # drop the float64 original now
    return Xw_trials


def _save_subject_temp(path, Xw_trials, yw_trials, yw_vel_trials, aw_trials):
    with h5py.File(path, "w") as f:
        for i, (x, y, yv, a) in enumerate(zip(Xw_trials, yw_trials, yw_vel_trials, aw_trials)):
            f.create_dataset(f"X_{i}", data=x, compression="gzip")
            f.create_dataset(f"y_{i}", data=y, compression="gzip")
            f.create_dataset(f"yv_{i}", data=yv, compression="gzip")
            f.create_dataset(f"a_{i}", data=a, compression="gzip")


def _load_subject_temp(path):
    Xw, yw, yw_vel, aw = [], [], [], []
    with h5py.File(path, "r") as f:
        i = 0
        while f"X_{i}" in f:
            Xw.append(f[f"X_{i}"][:])
            yw.append(f[f"y_{i}"][:])
            yw_vel.append(f[f"yv_{i}"][:])
            aw.append(f[f"a_{i}"][:])
            i += 1
    return Xw, yw, yw_vel, aw


def save_fold_to_hdf5(
    file_path,
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
):
    with h5py.File(file_path, "w") as f:

        def save_group(group_name, X_list, y_list, yv_list, a_list):
            grp = f.create_group(group_name)
            for i, (x, y, yv, a) in enumerate(zip(X_list, y_list, yv_list, a_list)):
                grp.create_dataset(f"X_{i}", data=x, compression="gzip")
                grp.create_dataset(f"y_{i}", data=y, compression="gzip")
                grp.create_dataset(f"yv_{i}", data=yv, compression="gzip")
                grp.create_dataset(f"a_{i}", data=a, compression="gzip")

        save_group("train", X_train, y_train, yv_train, act_train)
        save_group("val", X_val, y_val, yv_val, act_val)
        save_group("test", X_test, y_test, yv_test, act_test)


class IMUPreprocessor:
    def __init__(
        self,
        root_dir,
        window_size=15,
        horizon=10,
        original_freq=500,
        target_freq=100,
        use_both_legs=False,
        anti_alias=True,
        max_trials_per_subject=None,
        normalization: str = "zscore",
        height_ref_cm: float | None = None,
        subject_heights: dict[str, float] | None = None,
        handcrafted_features: bool = False,
        val_ratio: float = 0.1,
        split_seed: int = 42,
        split_strategy: str = "loso",
        per_subject_train_ratio: float = 0.7,
        per_subject_val_ratio: float = 0.1,
        per_subject_test_ratio: float = 0.2,
    ):
        self.root_dir = root_dir
        self.window_size = window_size
        self.horizon = horizon
        self.downsample_factor = original_freq // target_freq
        self.use_both_legs = use_both_legs
        self.anti_alias = anti_alias
        self.max_trials_per_subject = max_trials_per_subject
        self.normalization = normalization
        self.height_ref_cm = height_ref_cm if height_ref_cm is not None else _DEFAULT_HEIGHT_REF_CM
        self.subject_heights = (
            subject_heights if subject_heights is not None else ENABL3S_SUBJECT_HEIGHTS_CM
        )
        self.handcrafted_features = handcrafted_features
        self.val_ratio = val_ratio
        self.split_seed = split_seed
        self.split_strategy = split_strategy
        self.per_subject_train_ratio = per_subject_train_ratio
        self.per_subject_val_ratio = per_subject_val_ratio
        self.per_subject_test_ratio = per_subject_test_ratio
        self.leg_configs = {
            "right": {
                "predictors": [
                    "Right_Thigh_Ax",
                    "Right_Thigh_Ay",
                    "Right_Thigh_Az",
                    "Right_Thigh_Gx",
                    "Right_Thigh_Gy",
                    "Right_Thigh_Gz",
                ],
                "target": "Right_Knee",
                "velocity": "Right_Knee_Velocity",
            },
            "left": {
                "predictors": [
                    "Left_Thigh_Ax",
                    "Left_Thigh_Ay",
                    "Left_Thigh_Az",
                    "Left_Thigh_Gx",
                    "Left_Thigh_Gy",
                    "Left_Thigh_Gz",
                ],
                "target": "Left_Knee",
                "velocity": "Left_Knee_Velocity",
            },
        }
        self.predictors = self.leg_configs["right"]["predictors"]
        self.target = self.leg_configs["right"]["target"]

    def _downsample(self, arr: np.ndarray, axis: int = 0) -> np.ndarray:
        if self.downsample_factor <= 1:
            return arr
        if self.anti_alias:
            from scipy.signal import decimate

            return decimate(
                arr, q=self.downsample_factor, ftype="iir", zero_phase=True, axis=axis
            ).copy()
        else:
            from scipy.signal import resample

            n = arr.shape[axis] // self.downsample_factor
            return resample(arr, num=n, axis=axis)

    def _load_and_process_csv_from_zip(self, zip_file, member_name, leg="right"):
        with zip_file.open(member_name) as handle:
            df = pd.read_csv(io.TextIOWrapper(handle, encoding="utf-8"))

        mode_label_map = {
            0: "Sitting",
            1: "Level ground walking",
            2: "Ramp ascent",
            3: "Ramp descent",
            4: "Stair ascent",
            5: "Stair descent",
            6: "Standing",
        }
        df["Mode_Label"] = df["Mode"].map(mode_label_map)
        df = df[~df["Mode_Label"].isin(["Sitting", "Standing"])]

        config = self.leg_configs[leg]
        predictors = config["predictors"]
        target = config["target"]
        velocity_col = config["velocity"]

        required_cols = predictors + [target, velocity_col]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns for {leg} leg in {member_name}")
        if len(df) == 0:
            raise ValueError(f"All data filtered out in {member_name}")

        X = df[predictors].values
        y = df[target].values.reshape(-1, 1)
        y_vel = df[velocity_col].values.reshape(-1, 1)
        mode_ids = df["Mode"].values.astype(np.int8)

        if self.downsample_factor > 1:
            X = self._downsample(X, axis=0)
            y = self._downsample(y, axis=0)
            y_vel = self._downsample(y_vel, axis=0)
            # For mode labels, take every Nth sample (nearest-neighbor, no filtering needed)
            mode_ids = mode_ids[:: self.downsample_factor]

        # Align lengths after downsampling (decimate may differ by 1 sample)
        min_len = min(len(X), len(y), len(y_vel), len(mode_ids))
        X = X[:min_len]
        y = y[:min_len]
        y_vel = y_vel[:min_len]
        mode_ids = mode_ids[:min_len]

        return X, y, y_vel, mode_ids

    def _get_subject_sources(self):
        subject_sources = {}
        for subject_zip in sorted(glob(os.path.join(self.root_dir, "AB*.zip"))):
            subject_id = os.path.basename(subject_zip)[:-4]
            subject_sources.setdefault(subject_id, (subject_id, subject_zip))
        return sorted(subject_sources.values(), key=lambda item: item[0])

    def create_sliding_windows(self, X, y, y_vel, activity_ids):
        Xw, yw, yw_vel, aw = [], [], [], []
        for i in range(len(X) - self.window_size - self.horizon):
            Xw.append(X[i : i + self.window_size])
            yw.append(y[i + self.window_size : i + self.window_size + self.horizon, 0])
            yw_vel.append(y_vel[i + self.window_size : i + self.window_size + self.horizon, 0])
            aw.append(int(activity_ids[i]))
        return (
            np.array(Xw),
            np.array(yw),
            np.array(yw_vel),
            np.array(aw, dtype=np.int8),
        )

    def run(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        subject_sources = self._get_subject_sources()
        if not subject_sources:
            raise FileNotFoundError(
                f"No AB*.zip subject files found in {self.root_dir}. "
                "Run download first or verify the extract_dir path."
            )

        subject_temp_files: dict[str, str] = {}
        for subj_id, subject_zip in tqdm(subject_sources, desc="Loading subjects"):
            with zipfile.ZipFile(subject_zip, "r") as archive:
                trial_files = sorted(
                    [
                        name
                        for name in archive.namelist()
                        if name.lower().startswith(f"{subj_id.lower()}/processed/")
                        and name.lower().endswith(".csv")
                    ]
                )

                if self.max_trials_per_subject is not None:
                    trial_files = trial_files[: self.max_trials_per_subject]

                legs_to_process = ["right", "left"] if self.use_both_legs else ["right"]
                h_i = self.subject_heights.get(subj_id, self.height_ref_cm)
                if self.normalization == "height_minmax":
                    accel_scale = self.height_ref_cm / h_i
                    gyro_scale = float(np.sqrt(h_i / self.height_ref_cm))
                else:
                    accel_scale = gyro_scale = None

                tmp_path = os.path.join(output_dir, f"_tmp_{subj_id}.h5")
                n_skipped = 0
                trial_idx = 0
                # Running min/max for per-subject normalisation (no full copy in RAM).
                smin = smax = None
                with h5py.File(tmp_path, "w") as htmp:
                    for f in trial_files:
                        try:
                            for leg in legs_to_process:
                                X_raw, y_raw, y_vel_raw, mode_ids = (
                                    self._load_and_process_csv_from_zip(archive, f, leg=leg)
                                )
                                if accel_scale is not None:
                                    X_raw[:, :3] *= accel_scale
                                    X_raw[:, 3:6] *= gyro_scale
                                if self.handcrafted_features:
                                    X_raw = _append_handcrafted_features(X_raw)
                                Xw, yw, yw_vel, aw = self.create_sliding_windows(
                                    X_raw, y_raw, y_vel_raw, mode_ids
                                )
                                del X_raw, y_raw, y_vel_raw, mode_ids
                                if len(Xw) > 0:
                                    # Accumulate min/max without keeping all windows.
                                    if self.normalization == "height_minmax":
                                        flat = Xw.reshape(-1, Xw.shape[-1])
                                        if smin is None:
                                            n_feat = Xw.shape[-1]
                                            smin = np.full(n_feat, np.inf, dtype=np.float64)
                                            smax = np.full(n_feat, -np.inf, dtype=np.float64)
                                        smin = np.minimum(smin, flat.min(axis=0))
                                        smax = np.maximum(smax, flat.max(axis=0))
                                    htmp.create_dataset(f"X_{trial_idx}", data=Xw, compression="gzip")
                                    htmp.create_dataset(f"y_{trial_idx}", data=yw, compression="gzip")
                                    htmp.create_dataset(f"yv_{trial_idx}", data=yw_vel, compression="gzip")
                                    htmp.create_dataset(f"a_{trial_idx}", data=aw, compression="gzip")
                                    trial_idx += 1
                                    del Xw, yw, yw_vel, aw
                        except Exception as e:
                            logger.warning(f"Skipped {f}: {e}")
                            n_skipped += 1

                if n_skipped:
                    logger.warning(
                        "{}: skipped {}/{} trial files", subj_id, n_skipped, len(trial_files)
                    )
                if n_skipped == len(trial_files):
                    logger.error(
                        "{}: ALL trials skipped — check velocity column names or CSV format",
                        subj_id,
                    )

                if trial_idx > 0:
                    # Stage 2 (paper §2.2.2): per-subject min-max to [-1, 1]. Apply
                    # in-place directly in the temp HDF5 — one trial in RAM at a time.
                    if self.normalization == "height_minmax" and smin is not None:
                        smin_f = smin.astype(np.float32)
                        inv = (2.0 / np.clip(smax - smin, 1e-8, None)).astype(np.float32)
                        with h5py.File(tmp_path, "a") as htmp:
                            for idx in range(trial_idx):
                                key = f"X_{idx}"
                                w = htmp[key][:].astype(np.float32)
                                w -= smin_f
                                w *= inv
                                w -= 1.0
                                del htmp[key]
                                htmp.create_dataset(key, data=w, compression="gzip")
                                del w
                    subject_temp_files[subj_id] = tmp_path
                else:
                    os.remove(tmp_path)

        valid_subject_ids = list(subject_temp_files.keys())
        if not valid_subject_ids:
            raise RuntimeError(
                "No valid subjects loaded. All subject ZIPs failed — "
                "check the data format and column names."
            )

        try:
            if self.split_strategy == "per_subject":
                self._build_per_subject_fold(output_dir, subject_temp_files, valid_subject_ids)
            else:
                self._build_loso_folds(output_dir, subject_temp_files, valid_subject_ids)
        finally:
            for tmp_path in subject_temp_files.values():
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        logger.success(f"Successfully generated HDF5 folds in {output_dir}")

    def _build_per_subject_fold(self, output_dir, subject_temp_files, valid_subject_ids):
        """Pool every subject's own trials into one fold, splitting each subject's
        trials (not windows, to avoid leakage between overlapping windows) into
        train/val/test by the configured ratios."""
        fold_dir = os.path.join(output_dir, "fold_persubject")
        os.makedirs(fold_dir, exist_ok=True)
        hdf5_path = os.path.join(fold_dir, "data.h5")
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)

        train_r = self.per_subject_train_ratio
        val_r = self.per_subject_val_ratio
        test_r = self.per_subject_test_ratio
        logger.info(
            "Creating per-subject fold ({} subjects), ratios train={:.2f} val={:.2f} test={:.2f}...",
            len(valid_subject_ids),
            train_r,
            val_r,
            test_r,
        )

        counts = {"train": 0, "val": 0, "test": 0}
        with h5py.File(hdf5_path, "w") as hfold:
            groups = {name: hfold.create_group(name) for name in ("train", "val", "test")}

            for i, subj in enumerate(valid_subject_ids):
                X_trials, y_trials, yv_trials, a_trials = _load_subject_temp(
                    subject_temp_files[subj]
                )
                n_trials = len(X_trials)
                rng = random.Random(self.split_seed + i)
                order = list(range(n_trials))
                rng.shuffle(order)

                if n_trials <= 1:
                    n_test = n_val = 0
                else:
                    n_test = max(1, round(test_r * n_trials)) if test_r > 0 else 0
                    n_val = max(1, round(val_r * n_trials)) if val_r > 0 else 0
                    # Always leave at least one trial for train.
                    while n_test + n_val >= n_trials and (n_val > 0 or n_test > 0):
                        if n_val > 0:
                            n_val -= 1
                        else:
                            n_test -= 1

                test_idx = set(order[:n_test])
                val_idx = set(order[n_test : n_test + n_val])

                for t in range(n_trials):
                    name = "test" if t in test_idx else "val" if t in val_idx else "train"
                    grp = groups[name]
                    idx = counts[name]
                    grp.create_dataset(f"X_{idx}", data=X_trials[t], compression="gzip")
                    grp.create_dataset(f"y_{idx}", data=y_trials[t], compression="gzip")
                    grp.create_dataset(f"yv_{idx}", data=yv_trials[t], compression="gzip")
                    grp.create_dataset(f"a_{idx}", data=a_trials[t], compression="gzip")
                    counts[name] += 1
                del X_trials, y_trials, yv_trials, a_trials

        logger.info(
            "  fold_persubject: train_trials={}, val_trials={}, test_trials={}",
            counts["train"],
            counts["val"],
            counts["test"],
        )

    def _build_loso_folds(self, output_dir, subject_temp_files, valid_subject_ids):
        logger.info(f"Creating LOSO folds for {len(valid_subject_ids)} subjects...")
        for i, test_subj in enumerate(valid_subject_ids):
            train_subjs = [s for s in valid_subject_ids if s != test_subj]
            rng = random.Random(self.split_seed + i)

            fold_dir = os.path.join(output_dir, f"fold_{test_subj}")
            os.makedirs(fold_dir, exist_ok=True)
            hdf5_path = os.path.join(fold_dir, "data.h5")
            if os.path.exists(hdf5_path):
                os.remove(hdf5_path)

            train_count = val_count = 0
            with h5py.File(hdf5_path, "w") as hfold:
                train_grp = hfold.create_group("train")
                val_grp = hfold.create_group("val")
                test_grp = hfold.create_group("test")

                # Write test subject directly — one subject in RAM at a time.
                X_test, y_test, yv_test, act_test = _load_subject_temp(
                    subject_temp_files[test_subj]
                )
                for t, (x, y, yv, a) in enumerate(
                    zip(X_test, y_test, yv_test, act_test)
                ):
                    test_grp.create_dataset(f"X_{t}", data=x, compression="gzip")
                    test_grp.create_dataset(f"y_{t}", data=y, compression="gzip")
                    test_grp.create_dataset(f"yv_{t}", data=yv, compression="gzip")
                    test_grp.create_dataset(f"a_{t}", data=a, compression="gzip")
                del X_test, y_test, yv_test, act_test

                # Write each train subject immediately; never accumulate across subjects.
                # Test = held-out subject (LOSO). Validation = a per-subject slice of
                # the remaining subjects' trials (val_ratio of EACH subject's trials),
                # so val is drawn from the same distribution as train. Trial-level
                # (not window-level) to avoid leakage between overlapping windows.
                for subj in train_subjs:
                    X_trials, y_trials, yv_trials, a_trials = _load_subject_temp(
                        subject_temp_files[subj]
                    )
                    n_trials = len(X_trials)
                    n_val = max(1, round(self.val_ratio * n_trials)) if n_trials > 1 else 0
                    order = list(range(n_trials))
                    rng.shuffle(order)
                    val_idx = set(order[:n_val])
                    for t in range(n_trials):
                        if t in val_idx:
                            val_grp.create_dataset(f"X_{val_count}", data=X_trials[t], compression="gzip")
                            val_grp.create_dataset(f"y_{val_count}", data=y_trials[t], compression="gzip")
                            val_grp.create_dataset(f"yv_{val_count}", data=yv_trials[t], compression="gzip")
                            val_grp.create_dataset(f"a_{val_count}", data=a_trials[t], compression="gzip")
                            val_count += 1
                        else:
                            train_grp.create_dataset(f"X_{train_count}", data=X_trials[t], compression="gzip")
                            train_grp.create_dataset(f"y_{train_count}", data=y_trials[t], compression="gzip")
                            train_grp.create_dataset(f"yv_{train_count}", data=yv_trials[t], compression="gzip")
                            train_grp.create_dataset(f"a_{train_count}", data=a_trials[t], compression="gzip")
                            train_count += 1
                    del X_trials, y_trials, yv_trials, a_trials

            logger.info(
                f"  fold_{test_subj}: test={test_subj}, "
                f"train_trials={train_count}, val_trials={val_count}, "
                f"train_subjects={train_subjs}"
            )


def run_preprocessing(cfg: DictConfig, version: str = "0.1.0"):
    extract_dir = hydra.utils.to_absolute_path(cfg.dataset.extract_dir)
    processed_dir = hydra.utils.to_absolute_path(cfg.dataset.processed_dir)

    seq_length = cfg.training.context_length
    horizon = cfg.training.forecast_horizon

    original_freq = cfg.dataset.get("original_freq", 500)
    target_freq = cfg.dataset.get("target_freq", 100)
    use_both_legs = cfg.dataset.get("use_both_legs", False)
    anti_alias = cfg.dataset.get("anti_alias", True)
    max_trials_per_subject = cfg.dataset.get("max_trials_per_subject", None)

    normalization = cfg.dataset.get("normalization", "zscore")
    height_ref_cm = cfg.dataset.get("height_ref_cm", None)
    if height_ref_cm is not None:
        height_ref_cm = float(height_ref_cm)

    handcrafted_features = bool(cfg.dataset.get("handcrafted_features", False))
    split_seed = int(cfg.training.get("split_seed", 42))
    val_ratio = 1.0 - float(cfg.training.get("split_ratio", 0.9))

    split_strategy = str(cfg.dataset.get("split_strategy", "loso")).lower()
    per_subject_train_ratio = float(cfg.training.get("per_subject_train_ratio", 0.7))
    per_subject_val_ratio = float(cfg.training.get("per_subject_val_ratio", 0.1))
    per_subject_test_ratio = float(cfg.training.get("per_subject_test_ratio", 0.2))

    logger.info(
        "Preprocessing: anti_alias={}, normalization={}, handcrafted_features={}, "
        "split_strategy={}, val_ratio={:.2f}",
        anti_alias,
        normalization,
        handcrafted_features,
        split_strategy,
        val_ratio,
    )
    if normalization == "height_minmax":
        ref = height_ref_cm if height_ref_cm is not None else _DEFAULT_HEIGHT_REF_CM
        logger.info("Height normalization: h_ref={:.1f} cm", ref)

    preprocessor = IMUPreprocessor(
        root_dir=extract_dir,
        window_size=seq_length,
        horizon=horizon,
        original_freq=original_freq,
        target_freq=target_freq,
        use_both_legs=use_both_legs,
        anti_alias=anti_alias,
        max_trials_per_subject=max_trials_per_subject,
        normalization=normalization,
        height_ref_cm=height_ref_cm,
        handcrafted_features=handcrafted_features,
        val_ratio=val_ratio,
        split_seed=split_seed,
        split_strategy=split_strategy,
        per_subject_train_ratio=per_subject_train_ratio,
        per_subject_val_ratio=per_subject_val_ratio,
        per_subject_test_ratio=per_subject_test_ratio,
    )
    preprocessor.run(processed_dir)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_preprocessing(cfg)


if __name__ == "__main__":
    main()
