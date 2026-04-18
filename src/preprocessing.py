import zipfile
from pathlib import Path

import hydra
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm


class DatasetPreprocessor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.extract_dir = Path(hydra.utils.to_absolute_path(cfg.dataset.extract_dir))
        self.processed_dir = Path(hydra.utils.to_absolute_path(cfg.dataset.processed_dir))

        self.right_cols = [6, 7, 8, 9, 10, 11, 45]
        self.left_cols = [18, 19, 20, 21, 22, 23, 47]
        self.col_names = ["Ax", "Ay", "Az", "Gy", "Gz", "Gx", "KneeAngle"]

    def run(self):
        zip_files = list(self.extract_dir.glob("*.zip"))
        if not zip_files:
            logger.error(
                f"No zip files found in {self.extract_dir}. Please run dataset download first."
            )
            return

        logger.info(f"Found {len(zip_files)} raw zip files. Starting extraction...")

        for zip_path in tqdm(zip_files, desc="Subjects"):
            subject_id = zip_path.stem
            subject_out_dir = self.processed_dir / subject_id
            subject_out_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as z:
                data_files = [
                    f for f in z.namelist() if f.endswith("_post.csv") and "Processed" in f
                ]

                for file_name in data_files:
                    base_name = Path(file_name).stem

                    with z.open(file_name) as f:
                        df_full = pd.read_csv(f)

                    df_right = df_full.iloc[:, self.right_cols].copy()
                    df_right.columns = self.col_names
                    out_right = subject_out_dir / f"{base_name}_right.parquet"
                    df_right.to_parquet(out_right, index=False)

                    df_left = df_full.iloc[:, self.left_cols].copy()
                    df_left.columns = self.col_names
                    out_left = subject_out_dir / f"{base_name}_left.parquet"
                    df_left.to_parquet(out_left, index=False)

        logger.success(f"Preprocessing complete. Files saved to {self.processed_dir}")


def run_preprocessing(cfg: DictConfig):
    preprocessor = DatasetPreprocessor(cfg)
    preprocessor.run()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_preprocessing(cfg)


if __name__ == "__main__":
    main()
