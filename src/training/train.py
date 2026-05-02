import os

import hydra
from loguru import logger
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from src.models.factory import build_loss, build_optimizer, build_scheduler
from src.training.plotting import save_train_artifacts, should_save_intermediate_epoch
from src.runtime import (
    RunContext,
    autocast_context,
    backward_and_step,
    build_grad_scaler,
    unwrap_model,
)


class Trainer:
    def __init__(self, cfg: DictConfig, model, ctx: RunContext):
        self.cfg = cfg
        self.ctx = ctx
        self.model = model
        self.device = ctx.device
        self.autocast_kwargs = ctx.autocast_kwargs
        self.scaler = build_grad_scaler(self.autocast_kwargs, self.device)

        self.epochs = cfg.training.epochs

        self.optimizer = build_optimizer(cfg, self.model.parameters(), device=self.device)
        self.scheduler = build_scheduler(cfg, self.optimizer)
        self.criterion = build_loss(cfg)

        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.best_checkpoint_path = None

    def _set_encoder_grad(self, requires_grad: bool):
        for name, param in unwrap_model(self.model).named_parameters():
            if not name.startswith("regression_head"):
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
        head_params = [
            p for name, p in unwrap_model(self.model).named_parameters()
            if name.startswith("regression_head")
        ]
        self.optimizer = build_optimizer(self.cfg, head_params, device=self.device)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        logger.info(
            f"Phase 1: encoder frozen, training regression head for {freeze_epochs} epoch(s)"
        )

    def _begin_finetune_phase(self):
        self._set_encoder_grad(True)
        self.optimizer = build_optimizer(self.cfg, self.model.parameters(), device=self.device)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        logger.info("Phase 2: full model fine-tuning")

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
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
                predictions = self.model(past_imu, task="predict")
                loss = self.criterion(predictions, future_knee)

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

            total_loss += loss.item()

        return total_loss / len(loader) if len(loader) > 0 else 0.0

    def _validate_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
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
                    predictions = self.model(past_imu, task="predict")
                    loss = self.criterion(predictions, future_knee)
                total_loss += loss.item()

        return total_loss / len(loader) if len(loader) > 0 else 0.0

    def _save_if_best(self, val_loss: float, epoch: int):
        if val_loss >= self.best_val_loss:
            return
        self.best_val_loss = val_loss
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

    def run(self):
        freeze_epochs = self._resolve_freeze_epochs()
        if freeze_epochs > 0:
            self._begin_freeze_phase(freeze_epochs)

        train_loss_history = []
        val_loss_history = []

        for epoch in range(self.epochs):
            if freeze_epochs > 0 and epoch == freeze_epochs:
                self._begin_finetune_phase()

            train_loss = self._train_epoch(epoch)
            val_loss = self._validate_epoch(epoch)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - LR: {current_lr:.2e}"
            )

            train_loss_history.append(float(train_loss))
            val_loss_history.append(float(val_loss))

            self._save_if_best(val_loss, epoch)

            if should_save_intermediate_epoch(self.cfg, epoch):
                save_train_artifacts(
                    cfg=self.cfg,
                    train_losses=train_loss_history,
                    val_losses=val_loss_history,
                    best_epoch=self.best_epoch,
                    best_checkpoint_path=self.best_checkpoint_path,
                    tag=f"epoch_{epoch + 1:03d}",
                )

        save_train_artifacts(
            cfg=self.cfg,
            train_losses=train_loss_history,
            val_losses=val_loss_history,
            best_epoch=self.best_epoch,
            best_checkpoint_path=self.best_checkpoint_path,
            tag="final",
        )

        return self.best_checkpoint_path


def run_train(cfg: DictConfig, model, ctx: RunContext):
    return Trainer(cfg, model=model, ctx=ctx).run()
