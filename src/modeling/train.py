import os

import hydra
from loguru import logger
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from src.dataloader import build_pretrain_dataloaders
from src.modeling.factory import build_encoder, build_loss, build_optimizer, build_scheduler
from src.modeling.plotting import save_train_artifacts, should_save_intermediate_epoch
from src.modeling.runtime import (
    autocast_context,
    build_grad_scaler,
    configure_runtime,
    maybe_compile_model,
    maybe_wrap_parallel,
    resolve_autocast_kwargs,
    resolve_dataloader_kwargs,
    resolve_device,
    resolve_gradient_settings,
    unwrap_model,
    use_non_blocking_transfer,
)


class Trainer:
    def __init__(self, cfg: DictConfig, pretrained_state_dict=None, version=None):
        self.cfg = cfg
        self.device = resolve_device(cfg)
        self.version = version or cfg.get("version", "default")
        configure_runtime(cfg, self.device)

        logger.info(f"Using device: {self.device}")

        self.seq_length = cfg.training.context_length
        self.forecast_horizon = cfg.training.forecast_horizon
        self.batch_size = cfg.training.batch_size
        self.processed_dir = hydra.utils.to_absolute_path(cfg.dataset.processed_dir)

        loader_kwargs = resolve_dataloader_kwargs(cfg, self.device)
        self.non_blocking_transfer = use_non_blocking_transfer(
            cfg,
            self.device,
            pin_memory=loader_kwargs.get("pin_memory", False),
        )
        
        # Use the holdout subject's fold directory
        holdout = cfg.training.get("holdout_subjects", ["AB156"])[0] 
        fold_dir = os.path.join(self.processed_dir, f"fold_{holdout}")
        self.train_loader, self.val_loader, _, self.split_info = build_pretrain_dataloaders(
            data_dir=fold_dir,
            batch_size=self.batch_size,
            seed=cfg.training.get("split_seed", 42),
            **loader_kwargs
        )

        logger.info("Initializing IMU_Intent_Encoder from cfg.model")
        self.model = build_encoder(
            cfg=cfg,
            seq_length=self.seq_length,
            forecast_horizon=self.forecast_horizon,
        )
        self.model.to(self.device)

        if pretrained_state_dict is not None:
            cleaned_state_dict = dict(pretrained_state_dict)
            cleaned_state_dict.pop("positional_layer.pe", None)
            incompatible = self.model.load_state_dict(cleaned_state_dict, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                logger.warning(
                    "Loaded pretrained weights with missing keys: {} and unexpected keys: {}",
                    incompatible.missing_keys,
                    incompatible.unexpected_keys,
                )
            logger.info("Loaded pretrained weights into training model")

        self.model = maybe_compile_model(self.model, cfg)
        self.model = maybe_wrap_parallel(self.model, cfg, self.device)

        logger.info(f"version: {self.version}")

        self.optimizer = build_optimizer(cfg, self.model.parameters(), device=self.device)
        self.scheduler = build_scheduler(cfg, self.optimizer)
        self.criterion = build_loss(cfg)
        self.autocast_kwargs = resolve_autocast_kwargs(cfg, self.device)
        self.scaler = build_grad_scaler(self.autocast_kwargs, self.device)
        self.accumulation_steps, self.clip_grad_norm = resolve_gradient_settings(cfg)

        self.epochs = cfg.training.epochs
        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.best_checkpoint_path = None

    def _set_encoder_grad(self, requires_grad: bool):
        for name, param in unwrap_model(self.model).named_parameters():
            if not name.startswith("regression_head"):
                param.requires_grad_(requires_grad)

    def run(self):
        freeze_epochs = self.cfg.training.get("freeze_encoder_epochs", 0)

        if freeze_epochs > 0:
            self._set_encoder_grad(False)
            self.optimizer = build_optimizer(
                self.cfg,
                [p for p in self.model.parameters() if p.requires_grad],
                device=self.device,
            )
            self.scheduler = build_scheduler(self.cfg, self.optimizer)
            logger.info(
                f"Phase 1: encoder frozen, training regression head for {freeze_epochs} epoch(s)"
            )

        train_loss_history = []
        val_loss_history = []

        for epoch in range(self.epochs):
            if freeze_epochs > 0 and epoch == freeze_epochs:
                self._set_encoder_grad(True)
                self.optimizer = build_optimizer(
                    self.cfg, self.model.parameters(), device=self.device
                )
                self.scheduler = build_scheduler(self.cfg, self.optimizer)
                logger.info("Phase 2: full model fine-tuning")
            self.model.train()
            train_loss = 0.0
            self.optimizer.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(
                tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Train]")
            ):
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
                    loss = self.criterion(predictions, future_knee)
                    loss_for_backward = loss / self.accumulation_steps

                if self.scaler is not None:
                    self.scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

                should_step = ((batch_idx + 1) % self.accumulation_steps == 0) or (
                    batch_idx + 1 == len(self.train_loader)
                )
                if should_step:
                    if self.clip_grad_norm is not None:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)

                train_loss += loss.item()

            if len(self.train_loader) > 0:
                train_loss /= len(self.train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.inference_mode():
                for batch in tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Val]"):
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
                        loss = self.criterion(predictions, future_knee)
                    val_loss += loss.item()

            if len(self.val_loader) > 0:
                val_loss /= len(self.val_loader)
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - LR: {current_lr:.2e}"
            )

            train_loss_history.append(float(train_loss))
            val_loss_history.append(float(val_loss))

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = os.path.join(
                    hydra.utils.get_original_cwd(),
                    "checkpoints", self.version or "default",
                    f"best_model_epoch_{epoch + 1}.pth",
                )
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(unwrap_model(self.model).state_dict(), checkpoint_path)
                logger.info(f"Saved new best model checkpoint to {checkpoint_path}")
                self.best_epoch = epoch + 1
                self.best_checkpoint_path = checkpoint_path

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


def run_train(cfg: DictConfig, pretrained_state_dict=None, version=None):
    trainer = Trainer(cfg, pretrained_state_dict=pretrained_state_dict, version=version)
    return trainer.run()


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_train(cfg, version=cfg.get("version", "0.1.0"))


if __name__ == "__main__":
    main()
