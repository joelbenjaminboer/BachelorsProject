from glob import glob
import io
import os
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

_DEFAULT_HEIGHT_REF_CM = float(
    np.mean(list(ENABL3S_SUBJECT_HEIGHTS_CM.values()))
)  # ≈ 173.8 cm


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
        val_split_seed=42,
        normalization: str = "zscore",
        height_ref_cm: float | None = None,
        subject_heights: dict[str, float] | None = None,
    ):
        self.root_dir = root_dir
        self.window_size = window_size
        self.horizon = horizon
        self.downsample_factor = original_freq // target_freq
        self.use_both_legs = use_both_legs
        self.anti_alias = anti_alias
        self.max_trials_per_subject = max_trials_per_subject
        self.val_split_seed = val_split_seed
        self.normalization = normalization
        self.height_ref_cm = height_ref_cm if height_ref_cm is not None else _DEFAULT_HEIGHT_REF_CM
        self.subject_heights = subject_heights if subject_heights is not None else ENABL3S_SUBJECT_HEIGHTS_CM
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

        subject_data = {}
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
                Xw_trials, yw_trials, yw_vel_trials, aw_trials = [], [], [], []
                h_i = self.subject_heights.get(subj_id, self.height_ref_cm)
                if self.normalization == "height_minmax":
                    accel_scale = self.height_ref_cm / h_i
                    gyro_scale = float(np.sqrt(h_i / self.height_ref_cm))
                else:
                    accel_scale = gyro_scale = None

                n_skipped = 0
                for f in trial_files:
                    try:
                        for leg in legs_to_process:
                            X_raw, y_raw, y_vel_raw, mode_ids = (
                                self._load_and_process_csv_from_zip(archive, f, leg=leg)
                            )
                            if accel_scale is not None:
                                X_raw[:, :3] *= accel_scale
                                X_raw[:, 3:] *= gyro_scale
                            Xw, yw, yw_vel, aw = self.create_sliding_windows(
                                X_raw, y_raw, y_vel_raw, mode_ids
                            )
                            if len(Xw) > 0:
                                Xw_trials.append(Xw)
                                yw_trials.append(yw)
                                yw_vel_trials.append(yw_vel)
                                aw_trials.append(aw)
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

                if Xw_trials:
                    subject_data[subj_id] = (Xw_trials, yw_trials, yw_vel_trials, aw_trials)

        valid_subject_ids = list(subject_data.keys())
        if not valid_subject_ids:
            raise RuntimeError(
                "No valid subjects loaded. All subject ZIPs failed — "
                "check the data format and column names."
            )

        logger.info(f"Creating LOSO folds for {len(valid_subject_ids)} subjects...")
        for i, test_subj in enumerate(valid_subject_ids):
            X_test, y_test, yv_test, act_test = subject_data[test_subj]

            # Val subject rotates: next subject in list (wraps around), mirroring
            # how the test subject rotates across folds.
            val_subj = valid_subject_ids[(i + 1) % len(valid_subject_ids)]
            X_val, y_val, yv_val, act_val = subject_data[val_subj]

            train_subjs = [s for s in valid_subject_ids if s != test_subj and s != val_subj]

            X_train, y_train, yv_train, act_train = [], [], [], []
            for subj in train_subjs:
                X_trials, y_trials, yv_trials, a_trials = subject_data[subj]
                X_train.extend(X_trials)
                y_train.extend(y_trials)
                yv_train.extend(yv_trials)
                act_train.extend(a_trials)

            logger.info(f"  fold_{test_subj}: test={test_subj}, val={val_subj}, train={train_subjs}")

            fold_dir = os.path.join(output_dir, f"fold_{test_subj}")
            os.makedirs(fold_dir, exist_ok=True)

            hdf5_path = os.path.join(fold_dir, "data.h5")
            if os.path.exists(hdf5_path):
                os.remove(hdf5_path)

            save_fold_to_hdf5(
                hdf5_path,
                X_train, y_train, yv_train, act_train,
                X_val,   y_val,   yv_val,   act_val,
                X_test,  y_test,  yv_test,  act_test,
            )
        logger.success(f"Successfully generated HDF5 folds in {output_dir}")


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
    val_split_seed = int(cfg.training.get("split_seed", 42))

    normalization = cfg.dataset.get("normalization", "zscore")
    height_ref_cm = cfg.dataset.get("height_ref_cm", None)
    if height_ref_cm is not None:
        height_ref_cm = float(height_ref_cm)

    logger.info("Preprocessing: anti_alias={}, normalization={}", anti_alias, normalization)
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
        val_split_seed=val_split_seed,
        normalization=normalization,
        height_ref_cm=height_ref_cm,
    )
    preprocessor.run(processed_dir)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_preprocessing(cfg)


if __name__ == "__main__":
    main()
