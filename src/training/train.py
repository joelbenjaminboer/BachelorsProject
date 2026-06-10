import os

import hydra
from loguru import logger
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from src.models.factory import build_loss, build_optimizer, build_scheduler
from src.runtime import (
    RunContext,
    autocast_context,
    backward_and_step,
    build_grad_scaler,
    unwrap_model,
)
from src.training.plotting import save_train_artifacts, should_save_intermediate_epoch


class Trainer:
    def __init__(self, cfg: DictConfig, model, ctx: RunContext):
        self.cfg = cfg
        self.ctx = ctx
        self.model = model
        self.device = ctx.device
        self.autocast_kwargs = ctx.autocast_kwargs
        init_scale = cfg.gpu.autocast.get("init_scale", None)
        self.scaler = build_grad_scaler(self.autocast_kwargs, self.device, init_scale=init_scale)

        self.epochs = cfg.training.epochs

        es_cfg = cfg.training.get("early_stopping", {})
        self.early_stop_patience = int(es_cfg.get("patience", 0))
        self.early_stop_min_delta = float(es_cfg.get("min_delta", 0.0))

        self.optimizer = build_optimizer(cfg, self.model.parameters(), device=self.device)
        self.scheduler = build_scheduler(cfg, self.optimizer, total_epochs=self.epochs)
        self.criterion = build_loss(cfg)

        self.vel_loss_weight = float(
            cfg.model.get("multitask", {}).get("velocity_loss_weight", 0.1)
        )

        # Denormalisation scale for reporting/selecting on real-unit (degree) RMSE.
        # Both prediction and target share the same affine z-score, so the mean
        # cancels in the difference: rmse_deg = y_std * sqrt(mse_normalised).
        self.y_std = self.ctx.split_info.get("y_std")

        # Selection metric is real-unit validation RMSE (lower is better), not the
        # normalised loss — the latter hides mean-collapse (see README applied learning).
        self.best_val_metric = float("inf")
        self.best_epoch = None
        self.best_checkpoint_path = None

    def _set_encoder_grad(self, requires_grad: bool):
        _HEADS = ("regression_head", "velocity_head")
        for name, param in unwrap_model(self.model).named_parameters():
            if not any(name.startswith(h) for h in _HEADS):
                param.requires_grad_(requires_grad)

    def _resolve_freeze_epochs(self) -> int:
        freeze_epochs = int(self.cfg.training.get("freeze_encoder_epochs", 0))
        model_type = str(self.cfg.model.get("model_type", "encoder"))
        if freeze_epochs > 0 and model_type != "encoder":
            logger.warning(
                "freeze_encoder_epochs={} is not supported for model_type='{}', skipping freeze phase",
                freeze_epochs,
                model_type,
            )
            return 0
        return freeze_epochs

    def _begin_freeze_phase(self, freeze_epochs: int):
        self._set_encoder_grad(False)
        _HEADS = ("regression_head", "velocity_head")
        head_params = [
            p
            for name, p in unwrap_model(self.model).named_parameters()
            if any(name.startswith(h) for h in _HEADS)
        ]
        self.optimizer = build_optimizer(self.cfg, head_params, device=self.device)
        self.scheduler = build_scheduler(self.cfg, self.optimizer, total_epochs=freeze_epochs)
        logger.info(
            f"Phase 1: encoder frozen, training regression head for {freeze_epochs} epoch(s)"
        )

    def _begin_finetune_phase(self, freeze_epochs: int):
        self._set_encoder_grad(True)
        self.optimizer = build_optimizer(self.cfg, self.model.parameters(), device=self.device)
        finetune_epochs = self.epochs - freeze_epochs
        self.scheduler = build_scheduler(self.cfg, self.optimizer, total_epochs=finetune_epochs)
        logger.info("Phase 2: full model fine-tuning")

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = torch.zeros((), device=self.device)
        total_angle_loss = torch.zeros((), device=self.device)
        self.optimizer.zero_grad(set_to_none=True)
        loader = self.ctx.train_loader

        for batch_idx, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Train]")
        ):
            past_imu = batch["past_imu"].to(
                self.device, non_blocking=self.ctx.non_blocking_transfer
            )
            future_knee = batch["future_knee"].to(
                self.device, non_blocking=self.ctx.non_blocking_transfer
            )

            with autocast_context(self.autocast_kwargs):
                output = self.model(past_imu, task="predict")

            # Loss in fp32 outside autocast to avoid fp16 overflow → NaN.
            if isinstance(output, tuple):
                angle_pred, vel_pred = output
                angle_loss = self.criterion(angle_pred.float(), future_knee)
                fkv = batch["future_knee_vel"].to(
                    self.device, non_blocking=self.ctx.non_blocking_transfer
                )
                vel_loss = self.criterion(vel_pred.float(), fkv)
                loss = angle_loss + self.vel_loss_weight * vel_loss
            else:
                angle_loss = self.criterion(output.float(), future_knee)
                loss = angle_loss

            backward_and_step(
                loss=loss,
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                accumulation_steps=self.ctx.accumulation_steps,
                clip_grad_norm=self.ctx.clip_grad_norm,
                batch_idx=batch_idx,
                total_batches=len(loader),
            )

            total_loss += loss.detach()
            total_angle_loss += angle_loss.detach()

        # Report angle-only loss so train/val curves are comparable (val omits
        # the velocity term); optimisation still uses the full multitask loss.
        return (total_angle_loss / len(loader)).item() if len(loader) > 0 else 0.0

    def _validate_epoch(self, epoch: int) -> tuple[float, float]:
        """Returns (angle-only val loss, real-unit val RMSE in degrees)."""
        self.model.eval()
        scale = float(self.y_std) if self.y_std is not None else 1.0
        total_loss = torch.zeros((), device=self.device)
        sq_err = torch.zeros((), device=self.device)
        count = 0
        loader = self.ctx.val_loader

        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Val]"):
                past_imu = batch["past_imu"].to(
                    self.device, non_blocking=self.ctx.non_blocking_transfer
                )
                future_knee = batch["future_knee"].to(
                    self.device, non_blocking=self.ctx.non_blocking_transfer
                )
                with autocast_context(self.autocast_kwargs):
                    output = self.model(past_imu, task="predict")
                angle_pred = (output[0] if isinstance(output, tuple) else output).float()
                total_loss += self.criterion(angle_pred, future_knee).detach()
                diff = angle_pred - future_knee
                sq_err += torch.sum(diff * diff)
                count += diff.numel()

        n = max(1, len(loader))
        val_loss = (total_loss / n).item()
        val_rmse = scale * (sq_err / max(1, count)).sqrt().item()
        return val_loss, val_rmse

    def _save_if_best(self, val_metric: float, epoch: int):
        if val_metric >= self.best_val_metric:
            return
        self.best_val_metric = val_metric
        self.best_epoch = epoch + 1
        checkpoint_path = os.path.join(
            hydra.utils.get_original_cwd(),
            "checkpoints",
            self.ctx.version,
            f"best_model_epoch_{epoch + 1}.pth",
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(unwrap_model(self.model).state_dict(), checkpoint_path)
        logger.info(f"Saved new best model checkpoint to {checkpoint_path}")
        self.best_checkpoint_path = checkpoint_path

    def _step_scheduler(self, val_loss: float):
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def run(self):
        freeze_epochs = self._resolve_freeze_epochs()
        if freeze_epochs > 0:
            self._begin_freeze_phase(freeze_epochs)

        train_loss_history = []
        val_loss_history = []
        val_rmse_history = []
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            if freeze_epochs > 0 and epoch == freeze_epochs:
                self._begin_finetune_phase(freeze_epochs)
                epochs_no_improve = 0  # reset counter when fine-tuning starts

            train_loss = self._train_epoch(epoch)
            val_loss, val_rmse = self._validate_epoch(epoch)

            # Select / schedule / early-stop on real-unit RMSE, not the loss.
            self._step_scheduler(val_rmse)

            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - Val RMSE: {val_rmse:.4f}° - LR: {current_lr:.2e}"
            )

            train_loss_history.append(float(train_loss))
            val_loss_history.append(float(val_loss))
            val_rmse_history.append(float(val_rmse))

            improved = val_rmse < self.best_val_metric - self.early_stop_min_delta
            self._save_if_best(val_rmse, epoch)

            if improved:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if self.early_stop_patience > 0 and epochs_no_improve >= self.early_stop_patience:
                    logger.info(
                        f"Early stopping: no improvement for {self.early_stop_patience} epochs "
                        f"(best val RMSE {self.best_val_metric:.4f}° at epoch {self.best_epoch})"
                    )
                    break

            if should_save_intermediate_epoch(self.cfg, epoch):
                save_train_artifacts(
                    cfg=self.cfg,
                    train_losses=train_loss_history,
                    val_losses=val_loss_history,
                    best_epoch=self.best_epoch,
                    best_checkpoint_path=self.best_checkpoint_path,
                    tag=f"epoch_{epoch + 1:03d}",
                    val_rmse=val_rmse_history,
                )

        save_train_artifacts(
            cfg=self.cfg,
            train_losses=train_loss_history,
            val_losses=val_loss_history,
            best_epoch=self.best_epoch,
            best_checkpoint_path=self.best_checkpoint_path,
            tag="final",
            val_rmse=val_rmse_history,
        )

        return self.best_checkpoint_path


def run_train(cfg: DictConfig, model, ctx: RunContext):
    return Trainer(cfg, model=model, ctx=ctx).run()
