from collections import defaultdict
import os
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from src.dataloader import build_pretrain_dataloaders
from src.modeling.factory import build_encoder
from src.modeling.plotting import save_eval_artifacts
from src.modeling.runtime import (
    autocast_context,
    configure_runtime,
    maybe_compile_model,
    maybe_wrap_parallel,
    resolve_autocast_kwargs,
    resolve_dataloader_kwargs,
    resolve_device,
    unwrap_model,
    use_non_blocking_transfer,
)


class Evaluator:
    def __init__(self, cfg: DictConfig, version=None):
        self.cfg = cfg
        self.device = resolve_device(cfg)
        self.version = version or cfg.get("version", "default")
        configure_runtime(cfg, self.device)
        self.autocast_kwargs = resolve_autocast_kwargs(cfg, self.device)
        logger.info(f"Using device: {self.device} for Evaluation")
        logger.info(f"Evaluation version: {self.version}")
        self.seq_length = cfg.training.context_length
        self.forecast_horizon = cfg.training.forecast_horizon
        self.batch_size = cfg.training.batch_size
        self.processed_dir = hydra.utils.to_absolute_path(cfg.dataset.processed_dir)
        self.holdout_subjects = cfg.training.get("holdout_subjects", [])

        # Use the holdout subject's fold directory
        if not self.holdout_subjects:
            raise ValueError("Eval requires held-out subjects in training.holdout_subjects")

        holdout = self.holdout_subjects[0]
        fold_dir = os.path.join(self.processed_dir, f"fold_{holdout}")

        loader_kwargs = resolve_dataloader_kwargs(cfg, self.device)
        self.non_blocking_transfer = use_non_blocking_transfer(
            cfg,
            self.device,
            pin_memory=loader_kwargs.get("pin_memory", False),
        )
        
        _, _, self.loader, self.split_info = build_pretrain_dataloaders(
            data_dir=fold_dir,
            batch_size=self.batch_size,
            seed=cfg.training.get("split_seed", 42),
            **loader_kwargs
        )
        
        if len(self.loader.dataset) == 0:
            raise ValueError("No samples found for held-out subjects")

        self.model = build_encoder(
            cfg=cfg,
            seq_length=self.seq_length,
            forecast_horizon=self.forecast_horizon,
        ).to(self.device)
        self.model = maybe_compile_model(self.model, cfg)
        self.model = maybe_wrap_parallel(self.model, cfg, self.device)

    def _find_best_checkpoint(self) -> Path:
        base_checkpoints_dir = Path(hydra.utils.get_original_cwd()) / "checkpoints"

        checkpoint_dirs = []
        if self.version:
            checkpoint_dirs.append(base_checkpoints_dir / str(self.version))
            checkpoint_dirs.append(base_checkpoints_dir / f"{self.version}")
        checkpoint_dirs.append(base_checkpoints_dir)

        candidates = []
        seen = set()
        for checkpoints_dir in checkpoint_dirs:
            if checkpoints_dir in seen:
                continue
            seen.add(checkpoints_dir)
            candidates.extend(sorted(checkpoints_dir.glob("best_model_epoch_*.pth")))

        if not candidates:
            searched_dirs = ", ".join(str(path) for path in checkpoint_dirs)
            raise FileNotFoundError(
                "No checkpoints found. Expected best_model_epoch_*.pth in: "
                f"{searched_dirs}"
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
        incompatible = unwrap_model(self.model).load_state_dict(state, strict=False)
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

        residual_chunks = []
        eval_cfg = self.cfg.get("plotting", {}).get("eval", {})
        max_examples = max(0, int(eval_cfg.get("prediction_examples", 6)))
        example_predictions = []
        example_targets = []
        example_subject_ids = []

        subject_squared_error = defaultdict(float)
        subject_absolute_error = defaultdict(float)
        subject_count = defaultdict(int)

        with torch.inference_mode():
            for batch in self.loader:
                past_imu = batch["past_imu"].to(
                    self.device,
                    non_blocking=self.non_blocking_transfer,
                )
                future_knee = batch["future_knee"].to(
                    self.device,
                    non_blocking=self.non_blocking_transfer,
                )

                with autocast_context(self.autocast_kwargs):
                    predictions = self.model(past_imu, task="predict")
                    errors = predictions - future_knee

                total_squared_error += torch.sum(errors**2).item()
                total_absolute_error += torch.sum(errors.abs()).item()
                total_count += errors.numel()

                residual_chunks.append(errors.reshape(-1).detach().cpu())

                per_step_squared += torch.sum(errors**2, dim=0)
                per_step_absolute += torch.sum(errors.abs(), dim=0)
                per_step_count += errors.size(0)

                batch_subject_ids = batch.get("subject_id")
                if batch_subject_ids is None:
                    batch_subject_ids = ["unknown"] * errors.size(0)
                elif isinstance(batch_subject_ids, torch.Tensor):
                    batch_subject_ids = [str(item) for item in batch_subject_ids.tolist()]
                else:
                    batch_subject_ids = [str(item) for item in batch_subject_ids]

                for sample_idx, subject_id in enumerate(batch_subject_ids):
                    sample_errors = errors[sample_idx]
                    subject_squared_error[subject_id] += torch.sum(sample_errors**2).item()
                    subject_absolute_error[subject_id] += torch.sum(sample_errors.abs()).item()
                    subject_count[subject_id] += sample_errors.numel()

                if len(example_subject_ids) < max_examples:
                    take = min(max_examples - len(example_subject_ids), predictions.size(0))
                    if take > 0:
                        example_predictions.extend(predictions[:take].detach().cpu().tolist())
                        example_targets.extend(future_knee[:take].detach().cpu().tolist())
                        example_subject_ids.extend(batch_subject_ids[:take])

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

        subject_metrics = {}
        for subject_id in sorted(subject_count):
            count = subject_count[subject_id]
            if count <= 0:
                continue

            subject_mse = subject_squared_error[subject_id] / count
            subject_mae = subject_absolute_error[subject_id] / count
            subject_rmse = subject_mse**0.5
            subject_metrics[subject_id] = {
                "mse": subject_mse,
                "mae": subject_mae,
                "rmse": subject_rmse,
                "count": count,
            }

        residuals = torch.cat(residual_chunks).tolist() if residual_chunks else []
        overall_metrics = {"mse": mse, "mae": mae, "rmse": rmse}
        per_step_metrics = {
            "mse": per_step_mse.tolist(),
            "mae": per_step_mae.tolist(),
            "rmse": per_step_rmse.tolist(),
        }

        save_eval_artifacts(
            cfg=self.cfg,
            overall_metrics=overall_metrics,
            per_step_metrics=per_step_metrics,
            residuals=residuals,
            prediction_examples=example_predictions,
            target_examples=example_targets,
            example_subject_ids=example_subject_ids,
            subject_metrics=subject_metrics,
            checkpoint_path=str(checkpoint_path),
            tag="final",
        )

        return {
            "overall": overall_metrics,
            "per_step": per_step_metrics,
            "subject": subject_metrics,
            "checkpoint_path": str(checkpoint_path),
        }


def run_eval(cfg: DictConfig, version=None):
    evaluator = Evaluator(cfg, version=version)
    return evaluator.run()


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_eval(cfg, version=cfg.get("version", "0.1.0"))


if __name__ == "__main__":
    main()
