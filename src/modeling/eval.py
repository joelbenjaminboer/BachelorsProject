import os
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import torch.nn as nn

from src.dataloader import IMUKneeDataset
from src.modeling.encoder import IMU_Intent_Encoder


# Setup device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Evaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = device

        self.seq_length = cfg.training.context_length
        self.forecast_horizon = cfg.training.forecast_horizon
        self.batch_size = cfg.training.batch_size
        self.processed_dir = hydra.utils.to_absolute_path(cfg.dataset.processed_dir)
        self.holdout_subjects = cfg.training.get("holdout_subjects", [])

        if not self.holdout_subjects:
            raise ValueError("Eval requires held-out subjects in training.holdout_subjects")

        self.dataset = IMUKneeDataset(
            self.processed_dir,
            self.seq_length,
            self.forecast_horizon,
            include_subjects=self.holdout_subjects,
        )

        if len(self.dataset) == 0:
            raise ValueError("No samples found for held-out subjects")

        num_workers = cfg.training.get("num_workers", min(os.cpu_count() or 1, 8))
        persistent_workers = cfg.training.get("persistent_workers", num_workers > 0)
        if num_workers == 0:
            persistent_workers = False
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

        self.model = IMU_Intent_Encoder(
            input_features=6,
            seq_length=self.seq_length,
            forecast_horizon=self.forecast_horizon,
        ).to(self.device)

        self.criterion = nn.MSELoss(reduction="mean")

    def _find_best_checkpoint(self) -> Path:
        checkpoints_dir = Path(hydra.utils.get_original_cwd()) / "checkpoints"
        candidates = sorted(checkpoints_dir.glob("best_model_epoch_*.pth"))

        if not candidates:
            raise FileNotFoundError(
                f"No checkpoints found in {checkpoints_dir}. Expected best_model_epoch_*.pth"
            )

        def epoch_number(path: Path) -> int:
            name = path.stem
            try:
                return int(name.split("best_model_epoch_")[-1])
            except ValueError:
                return -1

        best = max(candidates, key=lambda p: (epoch_number(p), p.stat().st_mtime))
        return best

    def _load_checkpoint(self, checkpoint_path: Path):
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            logger.warning(
                "Loaded weights with missing keys: {} and unexpected keys: {}",
                incompatible.missing_keys,
                incompatible.unexpected_keys,
            )

    def run(self):
        checkpoint_path = self._find_best_checkpoint()
        self._load_checkpoint(checkpoint_path)

        self.model.eval()
        total_squared_error = 0.0
        total_absolute_error = 0.0
        total_count = 0

        per_step_squared = torch.zeros(self.forecast_horizon, device=self.device)
        per_step_absolute = torch.zeros(self.forecast_horizon, device=self.device)
        per_step_count = 0

        with torch.no_grad():
            for batch in self.loader:
                past_imu = batch["past_imu"].to(self.device)
                future_knee = batch["future_knee"].to(self.device)

                predictions = self.model(past_imu, task="predict")
                errors = predictions - future_knee

                total_squared_error += torch.sum(errors**2).item()
                total_absolute_error += torch.sum(errors.abs()).item()
                total_count += errors.numel()

                per_step_squared += torch.sum(errors**2, dim=0)
                per_step_absolute += torch.sum(errors.abs(), dim=0)
                per_step_count += errors.size(0)

        mse = total_squared_error / total_count
        mae = total_absolute_error / total_count
        rmse = mse**0.5

        per_step_mse = (per_step_squared / per_step_count).detach().cpu()
        per_step_mae = (per_step_absolute / per_step_count).detach().cpu()
        per_step_rmse = torch.sqrt(per_step_mse)

        logger.info(f"Eval - MSE: {mse:.6f} | MAE: {mae:.6f} | RMSE: {rmse:.6f}")
        for step in range(self.forecast_horizon):
            logger.info(
                "Step {:02d} - MSE: {:.6f} | MAE: {:.6f} | RMSE: {:.6f}",
                step + 1,
                per_step_mse[step].item(),
                per_step_mae[step].item(),
                per_step_rmse[step].item(),
            )


def run_eval(cfg: DictConfig):
    evaluator = Evaluator(cfg)
    evaluator.run()


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_eval(cfg)


if __name__ == "__main__":
    main()
