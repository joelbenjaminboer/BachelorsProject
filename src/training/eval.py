from collections import defaultdict
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from src.runtime import (
    RunContext,
    autocast_context,
    load_state_into_model,
)
from src.training.plotting import save_eval_artifacts


class Evaluator:
    def __init__(self, cfg: DictConfig, model, ctx: RunContext, checkpoint_path=None):
        self.cfg = cfg
        self.ctx = ctx
        self.model = model
        self.device = ctx.device
        self.autocast_kwargs = ctx.autocast_kwargs
        self.explicit_checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        self.forecast_horizon = cfg.training.forecast_horizon

        if len(ctx.test_loader.dataset) == 0:
            raise ValueError("No samples found for held-out subjects")

    def _find_best_checkpoint(self) -> Path:
        base_checkpoints_dir = Path(hydra.utils.get_original_cwd()) / "checkpoints"

        candidates = []
        seen = set()
        for checkpoints_dir in [base_checkpoints_dir / self.ctx.version, base_checkpoints_dir]:
            if checkpoints_dir in seen:
                continue
            seen.add(checkpoints_dir)
            candidates.extend(sorted(checkpoints_dir.glob("best_model_epoch_*.pth")))

        if not candidates:
            raise FileNotFoundError(
                f"No best_model_epoch_*.pth checkpoints found in {base_checkpoints_dir}"
            )

        def epoch_number(path: Path) -> int:
            try:
                return int(path.stem.split("best_model_epoch_")[-1])
            except ValueError:
                return -1

        return max(candidates, key=lambda p: (epoch_number(p), p.stat().st_mtime))

    def _build_complete_trials(self, y_mean, y_std) -> list[dict]:
        eval_cfg = self.cfg.get("plotting", {}).get("eval", {})
        if not bool(eval_cfg.get("plot_complete_trials", True)):
            return []

        test_dataset = self.ctx.test_loader.dataset
        num_trials = len(test_dataset.X)
        max_trials = max(1, int(eval_cfg.get("complete_trials", 5)))
        selected = min(max_trials, num_trials)

        trials = []
        for trial_idx in range(selected):
            X_trial = test_dataset.X[trial_idx].float()
            y_trial = test_dataset.y[trial_idx].float()

            if test_dataset.imu_mean is not None:
                X_norm = (X_trial - test_dataset.imu_mean) / test_dataset.imu_std
            else:
                X_norm = X_trial

            preds_parts = []
            targets_parts = []
            with torch.inference_mode():
                for i in range(0, len(X_norm), self.forecast_horizon):
                    x = (
                        X_norm[i]
                        .unsqueeze(0)
                        .to(self.device, non_blocking=self.ctx.non_blocking_transfer)
                    )
                    with autocast_context(self.autocast_kwargs):
                        pred = self.model(x, task="predict")
                    if y_mean is not None and y_std is not None:
                        pred = pred * y_std + y_mean
                    preds_parts.append(pred.squeeze(0).cpu())
                    targets_parts.append(y_trial[i])

            trials.append(
                {
                    "id": f"trial_{trial_idx}",
                    "predictions": torch.cat(preds_parts).tolist(),
                    "targets": torch.cat(targets_parts).tolist(),
                }
            )

        return trials

    def run(self):
        checkpoint_path = self.explicit_checkpoint_path or self._find_best_checkpoint()
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        load_state_into_model(self.model, state, source=str(checkpoint_path))

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

        y_mean = self.ctx.split_info.get("y_mean")
        y_std = self.ctx.split_info.get("y_std")

        with torch.inference_mode():
            for batch in tqdm(self.ctx.test_loader, desc="Eval"):
                past_imu = batch["past_imu"].to(
                    self.device, non_blocking=self.ctx.non_blocking_transfer
                )
                future_knee = batch["future_knee"].to(
                    self.device, non_blocking=self.ctx.non_blocking_transfer
                )

                with autocast_context(self.autocast_kwargs):
                    predictions = self.model(past_imu, task="predict")

                if y_mean is not None and y_std is not None:
                    predictions = predictions * y_std + y_mean
                    future_knee = future_knee * y_std + y_mean

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
            subject_metrics[subject_id] = {
                "mse": subject_mse,
                "mae": subject_mae,
                "rmse": subject_mse**0.5,
                "count": count,
            }

        residuals = torch.cat(residual_chunks).tolist() if residual_chunks else []
        overall_metrics = {"mse": mse, "mae": mae, "rmse": rmse}
        per_step_metrics = {
            "mse": per_step_mse.tolist(),
            "mae": per_step_mae.tolist(),
            "rmse": per_step_rmse.tolist(),
        }

        complete_trials = self._build_complete_trials(y_mean, y_std)

        save_eval_artifacts(
            cfg=self.cfg,
            overall_metrics=overall_metrics,
            per_step_metrics=per_step_metrics,
            residuals=residuals,
            prediction_examples=example_predictions,
            target_examples=example_targets,
            example_subject_ids=example_subject_ids,
            subject_metrics=subject_metrics,
            complete_trials=complete_trials,
            checkpoint_path=str(checkpoint_path),
            tag="final",
        )

        return {
            "overall": overall_metrics,
            "per_step": per_step_metrics,
            "subject": subject_metrics,
            "checkpoint_path": str(checkpoint_path),
        }


def run_eval(cfg: DictConfig, model, ctx: RunContext, checkpoint_path=None):
    return Evaluator(cfg, model=model, ctx=ctx, checkpoint_path=checkpoint_path).run()
