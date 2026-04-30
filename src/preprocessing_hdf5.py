import os
import io
import zipfile
import numpy as np
import pandas as pd
from glob import glob
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample
import h5py


def save_fold_to_hdf5(file_path, X_train, y_train, X_val, y_val, X_test, y_test):
    with h5py.File(file_path, "w") as f:
        def save_group(group_name, X_list, y_list):
            grp = f.create_group(group_name)
            for i, (x, y) in enumerate(zip(X_list, y_list)):
                grp.create_dataset(f'X_{i}', data=x, compression='gzip')
                grp.create_dataset(f'y_{i}', data=y, compression='gzip')

        save_group('train', X_train, y_train)
        save_group('val', X_val, y_val)
        save_group('test', X_test, y_test)

def save_raw_fold_to_hdf5(file_path, X_train, y_train, X_val, y_val, X_test, y_test):
    with h5py.File(file_path, "w") as f:
        def save_group(group_name, X_list, y_list):
            grp = f.create_group(group_name)
            for i, (x, y) in enumerate(zip(X_list, y_list)):
                grp.create_dataset(f'X_{i}', data=x, compression='gzip')
                grp.create_dataset(f'y_{i}', data=y, compression='gzip')

        save_group('train', X_train, y_train)
        save_group('val', X_val, y_val)
        save_group('test', X_test, y_test)


def load_trials_from_hdf5(filepath):
    with h5py.File(filepath, 'r') as f:
        def load_group(group):
            X_list, y_list = [], []
            for key in sorted(f[group].keys()):
                if key.startswith('X_'):
                    idx = key.split('_')[1]
                    X = f[f'{group}/X_{idx}'][:]
                    y = f[f'{group}/y_{idx}'][:]
                    X_list.append(X)
                    y_list.append(y)
            return X_list, y_list

        X_train, y_train = load_group('train')
        X_val, y_val = load_group('val')
        X_test, y_test = load_group('test')
        return X_train, y_train, X_val, y_val, X_test, y_test
    
def load_raw_trials_from_hdf5(filepath):
    with h5py.File(filepath, 'r') as f:
        def load_group(group):
            X_list, y_list = [], []
            for key in sorted(f[group].keys()):
                if key.startswith('X_'):
                    idx = key.split('_')[1]
                    X = f[f'{group}/X_{idx}'][:]
                    y = f[f'{group}/y_{idx}'][:]
                    X_list.append(X)
                    y_list.append(y)
            return X_list, y_list

        X_train, y_train = load_group('train')
        X_val, y_val = load_group('val')
        X_test, y_test = load_group('test')
        return X_train, y_train, X_val, y_val, X_test, y_test


class IMUPreprocessor:
    def __init__(self, root_dir, target='Right_Knee', predictors=None,
                 window_size=15, original_freq=500, target_freq=50, horizon=10):
        self.root_dir = root_dir
        self.predictors = predictors or [
            'Right_Thigh_Ax', 'Right_Thigh_Ay', 'Right_Thigh_Az',
            'Right_Thigh_Gx', 'Right_Thigh_Gy', 'Right_Thigh_Gz',
            'Right_Knee'
        ]
        self.target = target
        self.window_size = window_size
        self.horizon = horizon
        self.downsample_factor = original_freq // target_freq
        self.target_freq = target_freq

    def _load_and_process_csv(self, file):
        df = pd.read_csv(file)

        # Map numerical mode codes to labels
        mode_label_map = {
            0: "Sitting", 1: "Level ground walking", 2: "Ramp ascent",
            3: "Ramp descent", 4: "Stair ascent", 5: "Stair descent", 6: "Standing"
        }
        df['Mode_Label'] = df['Mode'].map(mode_label_map)

        df = df[~df['Mode_Label'].isin(['Sitting', 'Standing'])]

        if not all(col in df.columns for col in self.predictors + [self.target]):
            raise ValueError(f"Missing required columns in {file}")
        if len(df) == 0:
            raise ValueError(f"All data filtered out in {file}")

        X = df[self.predictors].values
        y = df[self.target].values.reshape(-1, 1)

        X_ds = resample(X, num=X.shape[0] // self.downsample_factor, axis=0)
        y_ds = resample(y, num=y.shape[0] // self.downsample_factor, axis=0)

        return X_ds, y_ds

    def _load_and_process_csv_from_zip(self, zip_file, member_name):
        with zip_file.open(member_name) as handle:
            df = pd.read_csv(io.TextIOWrapper(handle, encoding='utf-8'))

        mode_label_map = {
            0: "Sitting", 1: "Level ground walking", 2: "Ramp ascent",
            3: "Ramp descent", 4: "Stair ascent", 5: "Stair descent", 6: "Standing"
        }
        df['Mode_Label'] = df['Mode'].map(mode_label_map)

        df = df[~df['Mode_Label'].isin(['Sitting', 'Standing'])]

        if not all(col in df.columns for col in self.predictors + [self.target]):
            raise ValueError(f"Missing required columns in {member_name}")
        if len(df) == 0:
            raise ValueError(f"All data filtered out in {member_name}")

        X = df[self.predictors].values
        y = df[self.target].values.reshape(-1, 1)

        X_ds = resample(X, num=X.shape[0] // self.downsample_factor, axis=0)
        y_ds = resample(y, num=y.shape[0] // self.downsample_factor, axis=0)

        return X_ds, y_ds

    def _get_subject_sources(self):
        subject_sources = {}

        for subject_dir in sorted(glob(os.path.join(self.root_dir, 'AB*'))):
            if os.path.isdir(subject_dir):
                subject_id = os.path.basename(subject_dir)
                subject_sources[subject_id] = (subject_id, subject_dir, None)
            elif subject_dir.endswith('.zip'):
                subject_id = os.path.basename(subject_dir)[:-4]
                subject_sources.setdefault(subject_id, (subject_id, None, subject_dir))

        for subject_zip in sorted(glob(os.path.join(self.root_dir, 'AB*.zip'))):
            subject_id = os.path.basename(subject_zip)[:-4]
            subject_sources.setdefault(subject_id, (subject_id, None, subject_zip))

        return sorted(subject_sources.values(), key=lambda item: item[0])

    def _collect_trial_files(self, subj_id, subject_dir=None, subject_zip=None):
        if subject_dir is not None:
            for processed_name in ('Processed', 'processed'):
                subj_path = os.path.join(subject_dir, processed_name)
                trial_files = sorted(glob(os.path.join(subj_path, '*.csv')))
                if trial_files:
                    return trial_files[:40]
            return []

        if subject_zip is not None:
            with zipfile.ZipFile(subject_zip, 'r') as archive:
                trial_files = sorted([
                    name for name in archive.namelist()
                    if name.lower().startswith(f'{subj_id.lower()}/processed/') and name.lower().endswith('.csv')
                ])
            return trial_files[:40]

        return []

    def create_sliding_windows(self, X, y):
        Xw, yw = [], []
        for i in range(len(X) - self.window_size - self.horizon):
            Xw.append(X[i:i + self.window_size])
            yw.append(y[i + self.window_size + self.horizon - 1])
        return np.array(Xw), np.array(yw)


    def main_leave_one_out(self, output_dir='processed_npz_LOSO'):
        os.makedirs(output_dir, exist_ok=True)

        subject_sources = self._get_subject_sources()
        subject_ids = [subject_id for subject_id, _, _ in subject_sources]
        subject_data = {}
        raw_subject_data = {}  # store raw trials here

        # Load and process 40 trials per subject
        for subj_id, subject_dir, subject_zip in subject_sources:
            trial_files = self._collect_trial_files(subj_id, subject_dir=subject_dir, subject_zip=subject_zip)

            Xw_trials, yw_trials = [], []
            raw_X_trials, raw_y_trials = [], []
            if subject_dir is not None:
                for f in trial_files:
                    try:
                        X_raw, y_raw = self._load_and_process_csv(f)
                        raw_X_trials.append(X_raw)
                        raw_y_trials.append(y_raw)

                        Xw, yw = self.create_sliding_windows(X_raw, y_raw)
                        if len(Xw) > 0:
                            Xw_trials.append(Xw)
                            yw_trials.append(yw)

                    except Exception as e:
                        print(f"Skipped {f}: {e}")
            else:
                with zipfile.ZipFile(subject_zip, 'r') as archive:
                    for f in trial_files:
                        try:
                            X_raw, y_raw = self._load_and_process_csv_from_zip(archive, f)
                            raw_X_trials.append(X_raw)
                            raw_y_trials.append(y_raw)

                            Xw, yw = self.create_sliding_windows(X_raw, y_raw)
                            if len(Xw) > 0:
                                Xw_trials.append(Xw)
                                yw_trials.append(yw)

                        except Exception as e:
                            print(f"Skipped {f}: {e}")

            if not Xw_trials:
                print(f"No valid data for subject {subj_id}, skipping.")
                continue

            subject_data[subj_id] = (Xw_trials, yw_trials)
            raw_subject_data[subj_id] = (raw_X_trials, raw_y_trials)

        valid_subject_ids = list(subject_data.keys())

        if not valid_subject_ids:
            raise ValueError(f"No valid subjects were loaded from {self.root_dir}")

        # Leave-One-Subject-Out
        for i, test_subj in enumerate(valid_subject_ids):
            print(f"\nFold {i+1}/{len(valid_subject_ids)} — Testing on {test_subj}")
            X_test, y_test = subject_data[test_subj]
            raw_X_test, raw_y_test = raw_subject_data[test_subj]

            train_subjs = [s for s in valid_subject_ids if s != test_subj]

            # Windowed data
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

            fold_dir = os.path.join(output_dir, f"fold_{i+1}")
            os.makedirs(fold_dir, exist_ok=True)

            save_fold_to_hdf5(
                os.path.join(fold_dir, "data.h5"),
                X_train, y_train,
                X_val, y_val,
                X_test, y_test
            )

            # Raw data, not resampled
            raw_X_train_trials, raw_y_train_trials = [], []
            for subj in train_subjs:
                Xr, yr = raw_subject_data[subj]
                raw_X_train_trials.extend(Xr)
                raw_y_train_trials.extend(yr)

            n_total_raw = len(raw_X_train_trials)
            n_val_raw = max(1, int(0.1 * n_total_raw)) #validation
            raw_X_val = raw_X_train_trials[:n_val_raw]
            raw_y_val = raw_y_train_trials[:n_val_raw]
            raw_X_train = raw_X_train_trials[n_val_raw:]
            raw_y_train = raw_y_train_trials[n_val_raw:]

            save_raw_fold_to_hdf5(
                os.path.join(fold_dir, "raw_data.h5"),
                raw_X_train, raw_y_train,
                raw_X_val, raw_y_val,
                raw_X_test, raw_y_test
            )

            print(f"Fold {i+1} saved to {fold_dir}")


if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pre = IMUPreprocessor(
        root_dir="/Users/joelboer/VSC/BachelorsProject/data/raw/ENABL3S",
        window_size=25,
        original_freq=500,
        target_freq=50,
        horizon=10
    )
    pre.main_leave_one_out(output_dir=os.path.join(repo_root, 'processed_LOSO_25'))

    # # Load windowed data
    # X_train, y_train, X_val, y_val, X_test, y_test = load_trials_from_hdf5(os.path.join(repo_root, 'processed_LOSO_25', 'fold_1', 'data.h5'))

    # # Load raw data
    # raw_X_train, raw_y_train, *_ = load_raw_trials_from_hdf5(os.path.join(repo_root, 'processed_LOSO_25', 'fold_1', 'raw_data.h5'))


    # print("\n--- Dataset Shapes ---")
    # print("X_train:", len(X_train), "trials;", "First shape:", X_train[0].shape)
    # print("y_train:", len(y_train), "trials;", "First shape:", y_train[0].shape)

    # print("X_val:", len(X_val), "trials;", "First shape:", X_val[0].shape)
    # print("y_val:", len(y_val), "trials;", "First shape:", y_val[0].shape)

    # print("X_test:", len(X_test), "trials;", "First shape:", X_test[0].shape)
    # print("y_test:", len(y_test), "trials;", "First shape:", y_test[0].shape)