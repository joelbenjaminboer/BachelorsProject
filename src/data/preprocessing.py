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
from scipy.signal import resample
from tqdm import tqdm


def save_fold_to_hdf5(file_path, X_train, y_train, X_val, y_val, X_test, y_test):
    with h5py.File(file_path, "w") as f:

        def save_group(group_name, X_list, y_list):
            grp = f.create_group(group_name)
            for i, (x, y) in enumerate(zip(X_list, y_list)):
                grp.create_dataset(f"X_{i}", data=x, compression="gzip")
                grp.create_dataset(f"y_{i}", data=y, compression="gzip")

        save_group("train", X_train, y_train)
        save_group("val", X_val, y_val)
        save_group("test", X_test, y_test)


class IMUPreprocessor:
    def __init__(
        self,
        root_dir,
        window_size=15,
        horizon=10,
        original_freq=500,
        target_freq=100,
        use_both_legs=False,
    ):
        self.root_dir = root_dir
        self.window_size = window_size
        self.horizon = horizon
        self.downsample_factor = original_freq // target_freq
        self.use_both_legs = use_both_legs
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
            },
        }
        self.predictors = self.leg_configs["right"]["predictors"]
        self.target = self.leg_configs["right"]["target"]

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

        if not all(col in df.columns for col in predictors + [target]):
            raise ValueError(f"Missing required columns for {leg} leg in {member_name}")
        if len(df) == 0:
            raise ValueError(f"All data filtered out in {member_name}")

        X = df[predictors].values
        y = df[target].values.reshape(-1, 1)

        if self.downsample_factor > 1:
            X = resample(X, num=X.shape[0] // self.downsample_factor, axis=0)
            y = resample(y, num=y.shape[0] // self.downsample_factor, axis=0)

        return X, y

    def _get_subject_sources(self):
        subject_sources = {}
        for subject_zip in sorted(glob(os.path.join(self.root_dir, "AB*.zip"))):
            subject_id = os.path.basename(subject_zip)[:-4]
            subject_sources.setdefault(subject_id, (subject_id, subject_zip))
        return sorted(subject_sources.values(), key=lambda item: item[0])

    def create_sliding_windows(self, X, y):
        Xw, yw = [], []
        for i in range(len(X) - self.window_size - self.horizon):
            Xw.append(X[i : i + self.window_size])
            yw.append(y[i + self.window_size : i + self.window_size + self.horizon, 0])
        return np.array(Xw), np.array(yw)

    def run(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        subject_sources = self._get_subject_sources()
        if not subject_sources:
            logger.error(f"No zip files found in {self.root_dir}.")
            return

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

                trial_files = trial_files[:40]

                legs_to_process = ["right", "left"] if self.use_both_legs else ["right"]
                Xw_trials, yw_trials = [], []
                for f in trial_files:
                    try:
                        for leg in legs_to_process:
                            X_raw, y_raw = self._load_and_process_csv_from_zip(archive, f, leg=leg)
                            Xw, yw = self.create_sliding_windows(X_raw, y_raw)
                            if len(Xw) > 0:
                                Xw_trials.append(Xw)
                                yw_trials.append(yw)
                    except Exception as e:
                        logger.warning(f"Skipped {f}: {e}")

                if Xw_trials:
                    subject_data[subj_id] = (Xw_trials, yw_trials)

        valid_subject_ids = list(subject_data.keys())
        if not valid_subject_ids:
            logger.error("No valid subjects loaded.")
            return

        logger.info(f"Creating LOSO folds for {len(valid_subject_ids)} subjects...")
        for i, test_subj in enumerate(valid_subject_ids):
            X_test, y_test = subject_data[test_subj]
            train_subjs = [s for s in valid_subject_ids if s != test_subj]

            X_train_trials, y_train_trials = [], []
            for subj in train_subjs:
                X_trials, y_trials = subject_data[subj]
                X_train_trials.extend(X_trials)
                y_train_trials.extend(y_trials)

            n_total = len(X_train_trials)
            n_val = max(1, int(0.1 * n_total))

            X_val = X_train_trials[:n_val]
            y_val = y_train_trials[:n_val]
            X_train = X_train_trials[n_val:]
            y_train = y_train_trials[n_val:]

            fold_dir = os.path.join(output_dir, f"fold_{test_subj}")
            os.makedirs(fold_dir, exist_ok=True)

            save_fold_to_hdf5(
                os.path.join(fold_dir, "data.h5"), X_train, y_train, X_val, y_val, X_test, y_test
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

    preprocessor = IMUPreprocessor(
        root_dir=extract_dir,
        window_size=seq_length,
        horizon=horizon,
        original_freq=original_freq,
        target_freq=target_freq,
        use_both_legs=use_both_legs,
    )
    preprocessor.run(processed_dir)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_preprocessing(cfg)


if __name__ == "__main__":
    main()
