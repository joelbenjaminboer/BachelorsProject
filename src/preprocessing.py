import zipfile
from pathlib import Path

import hydra
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    extract_dir = Path(hydra.utils.to_absolute_path(cfg.dataset.extract_dir))
    processed_dir = Path(hydra.utils.to_absolute_path(cfg.dataset.processed_dir))

    # Columns mapping (0-indexed base out of the 49 channels)
    # Right: IMU 7-12 -> 6-11, GONIO Knee 46 -> 45
    # Left:  IMU 19-24 -> 18-23, GONIO Knee 48 -> 47
    right_cols = [6, 7, 8, 9, 10, 11, 45]
    left_cols = [18, 19, 20, 21, 22, 23, 47]
    col_names = ["Ax", "Ay", "Az", "Gy", "Gz", "Gx", "KneeAngle"]

    zip_files = list(extract_dir.glob("*.zip"))
    if not zip_files:
        logger.error(f"No zip files found in {extract_dir}. Please run dataset.py first.")
        return

    logger.info(f"Found {len(zip_files)} raw zip files. Starting extraction...")

    for zip_path in tqdm(zip_files, desc="Subjects"):
        subject_id = zip_path.stem  # e.g., 'AB156'
        subject_out_dir = processed_dir / subject_id
        subject_out_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as z:
            data_files = [f for f in z.namelist() if f.endswith("_post.csv") and "Processed" in f]

            for file_name in data_files:
                base_name = Path(file_name).stem

                with z.open(file_name) as f:
                    df_full = pd.read_csv(f)

                # Extract Right Leg
                df_right = df_full.iloc[:, right_cols].copy()
                df_right.columns = col_names
                out_right = subject_out_dir / f"{base_name}_right.parquet"
                df_right.to_parquet(out_right, index=False)

                # Extract Left Leg
                df_left = df_full.iloc[:, left_cols].copy()
                df_left.columns = col_names
                out_left = subject_out_dir / f"{base_name}_left.parquet"
                df_left.to_parquet(out_left, index=False)

    logger.success(f"Preprocessing complete. Files saved to {processed_dir}")


if __name__ == "__main__":
    main()
