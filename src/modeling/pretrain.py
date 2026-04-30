import os

import hydra
from loguru import logger
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from src.dataloader import build_pretrain_dataloaders
from src.modeling.factory import build_encoder, build_loss, build_optimizer
from src.modeling.plotting import save_pretrain_artifacts, should_save_intermediate_epoch
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

CHANNEL_NAMES = ("Ax", "Ay", "Az", "Gy", "Gz", "Gx")


def masked_channel_sse(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    masked_predictions = predictions[mask]
    masked_targets = targets[mask]

    if masked_predictions.numel() == 0:
        return None, 0

    squared_error = (masked_predictions - masked_targets).pow(2)
    return squared_error.sum(dim=0), masked_predictions.size(0)


def format_channel_metrics(channel_names, values):
    return " | ".join(f"{channel}={value:.4f}" for channel, value in zip(channel_names, values))


def build_contiguous_block_mask(
    batch_size: int,
    seq_length: int,
    min_block_len: int,
    max_block_len: int,
    device: torch.device,
):
    if batch_size <= 0 or seq_length <= 0:
        return torch.zeros(batch_size, max(seq_length, 0), dtype=torch.bool, device=device)

    min_len = max(1, min(int(min_block_len), seq_length))
    max_len = max(min_len, min(int(max_block_len), seq_length))

    block_lengths = torch.randint(min_len, max_len + 1, (batch_size,), device=device)
    max_start_positions = seq_length - block_lengths

    # torch.randint does not support per-element upper bounds; use scaled random samples.
    start_positions = (
        torch.rand(batch_size, device=device) * (max_start_positions + 1).float()
    ).floor().long()

    time_indices = torch.arange(seq_length, device=device).unsqueeze(0)
    return (time_indices >= start_positions.unsqueeze(1)) & (
        time_indices < (start_positions + block_lengths).unsqueeze(1)
    )


def evaluate_masked_reconstruction(
    model,
    data_loader,
    criterion,
    device,
    autocast_kwargs,
    non_blocking_transfer,
    seq_length,
    mask_block_min_len,
    mask_block_max_len,
    epoch,
    epochs,
    phase_label,
):
    if data_loader is None or len(data_loader) == 0:
        nan_metrics = torch.full((len(CHANNEL_NAMES),), float("nan"))
        return float("nan"), nan_metrics, nan_metrics

    model.eval()
    phase_loss = 0.0
    phase_sq_error_sum = torch.zeros(len(CHANNEL_NAMES), device=device)
    phase_masked_count = 0

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs} [{phase_label}]"):
            past_imu = batch["past_imu"].to(device, non_blocking=non_blocking_transfer)

            batch_size_current = past_imu.size(0)
            mask = build_contiguous_block_mask(
                batch_size=batch_size_current,
                seq_length=seq_length,
                min_block_len=mask_block_min_len,
                max_block_len=mask_block_max_len,
                device=device,
            )

            with autocast_context(autocast_kwargs):
                reconstructed_imu = model(past_imu, mask=mask, task="reconstruct")
                loss = criterion(reconstructed_imu[mask], past_imu[mask])

            batch_sq_error_sum, batch_masked_count = masked_channel_sse(
                reconstructed_imu, past_imu, mask
            )
            if batch_sq_error_sum is not None:
                phase_sq_error_sum += batch_sq_error_sum
                phase_masked_count += batch_masked_count

            phase_loss += loss.item()

    phase_loss /= len(data_loader)

    if phase_masked_count > 0:
        phase_channel_mse = (phase_sq_error_sum / phase_masked_count).detach().cpu()
        phase_channel_rmse = torch.sqrt(phase_channel_mse)
    else:
        phase_channel_mse = torch.full((len(CHANNEL_NAMES),), float("nan"))
        phase_channel_rmse = torch.full((len(CHANNEL_NAMES),), float("nan"))

    return phase_loss, phase_channel_mse, phase_channel_rmse


class Pretrainer:
    def __init__(self, cfg: DictConfig, version=None):
        self.cfg = cfg
        self.device = resolve_device(cfg)
        self.version = version or cfg.get("version", "default")
        configure_runtime(cfg, self.device)
        self.autocast_kwargs = resolve_autocast_kwargs(cfg, self.device)
        self.scaler = build_grad_scaler(self.autocast_kwargs, self.device)
        self.accumulation_steps, self.clip_grad_norm = resolve_gradient_settings(cfg)

        logger.info(f"Using device: {self.device} for Pretraining")
        logger.info(f"Pretraining version: {self.version}")

        self.seq_length = cfg.training.context_length
        self.forecast_horizon = cfg.training.forecast_horizon
        self.batch_size = cfg.training.batch_size
        self.epochs = cfg.training.epochs
        self.split_ratio = cfg.training.get("split_ratio", 0.9)
        self.split_strategy = cfg.training.get("split_strategy", "loso")
        self.holdout_subjects = cfg.training.get("holdout_subjects", [])
        self.split_seed = cfg.training.get("split_seed", 42)
        base_dir = hydra.utils.to_absolute_path(cfg.dataset.processed_dir)
        self.processed_dir = os.path.join(base_dir, f"fold_{self.holdout_subjects[0]}") if self.holdout_subjects else base_dir

        loader_kwargs = resolve_dataloader_kwargs(cfg, self.device)
        self.non_blocking_transfer = use_non_blocking_transfer(
            cfg,
            self.device,
            pin_memory=loader_kwargs.get("pin_memory", False),
        )
        (
            self.train_loader,
            self.val_loader,
            _,
            self.split_info,
        ) = build_pretrain_dataloaders(
            data_dir=self.processed_dir,
            seq_length=self.seq_length,
            forecast_horizon=self.forecast_horizon,
            batch_size=self.batch_size,
            split_ratio=self.split_ratio,
            split_strategy=self.split_strategy,
            holdout_subjects=self.holdout_subjects,
            include_test_loader=False,
            seed=self.split_seed,
            num_workers=loader_kwargs.get("num_workers"),
            persistent_workers=loader_kwargs.get("persistent_workers"),
            prefetch_factor=loader_kwargs.get("prefetch_factor"),
            pin_memory=loader_kwargs.get("pin_memory"),
        )

        logger.info(
            f"Split strategy: {self.split_strategy} | "
            f"held-out subjects: {', '.join(self.holdout_subjects) or 'None'}"
        )
        logger.info(
            f"Samples - train: {self.split_info['train_samples']}, "
            f"val: {self.split_info['val_samples']}"
        )
        train_subjects = self.split_info.get('train_subjects', [])
        val_subjects = self.split_info.get('val_subjects', [])
        logger.info(
            f"Train subjects ({len(train_subjects)}): "
            f"{', '.join(train_subjects) or 'None'}"
        )
        logger.info(
            f"Val subjects ({len(val_subjects)}): "
            f"{', '.join(val_subjects) or 'None'}"
        )

        self.model = build_encoder(
            cfg=cfg,
            seq_length=self.seq_length,
            forecast_horizon=self.forecast_horizon,
        ).to(self.device)

        self.model = maybe_compile_model(self.model, cfg)
        self.model = maybe_wrap_parallel(self.model, cfg, self.device)

        self.criterion = build_loss(cfg)
        self.optimizer = build_optimizer(cfg, self.model.parameters(), device=self.device)

        raw_mask_block_min_len = int(cfg.training.get("mask_block_min_len", 10))
        raw_mask_block_max_len = int(cfg.training.get("mask_block_max_len", 20))
        self.mask_block_min_len = min(raw_mask_block_min_len, raw_mask_block_max_len)
        self.mask_block_max_len = max(raw_mask_block_min_len, raw_mask_block_max_len)
        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.best_checkpoint_path = None

    def run(self):
        logger.info(
            "Starting MAE Pretraining with contiguous block masking "
            f"(length range: {self.mask_block_min_len}-{self.mask_block_max_len})"
        )

        train_loss_history = []
        val_loss_history = []
        train_channel_mse_history = []
        val_channel_mse_history = []
        train_channel_rmse_history = []
        val_channel_rmse_history = []

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            train_sq_error_sum = torch.zeros(len(CHANNEL_NAMES), device=self.device)
            train_masked_count = 0
            self.optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(
                tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Train]")
            ):
                past_imu = batch["past_imu"].to(
                    self.device,
                    non_blocking=self.non_blocking_transfer,
                )
                batch_size_current = past_imu.size(0)
                mask = build_contiguous_block_mask(
                    batch_size=batch_size_current,
                    seq_length=self.seq_length,
                    min_block_len=self.mask_block_min_len,
                    max_block_len=self.mask_block_max_len,
                    device=self.device,
                )

                with autocast_context(self.autocast_kwargs):
                    reconstructed_imu = self.model(past_imu, mask=mask, task="reconstruct")
                    loss = self.criterion(reconstructed_imu[mask], past_imu[mask])
                    loss_for_backward = loss / self.accumulation_steps

                batch_sq_error_sum, batch_masked_count = masked_channel_sse(
                    reconstructed_imu, past_imu, mask
                )
                if batch_sq_error_sum is not None:
                    train_sq_error_sum += batch_sq_error_sum
                    train_masked_count += batch_masked_count

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

            if train_masked_count > 0:
                train_channel_mse = (train_sq_error_sum / train_masked_count).detach().cpu()
                train_channel_rmse = torch.sqrt(train_channel_mse)
            else:
                train_channel_mse = torch.full((len(CHANNEL_NAMES),), float("nan"))
                train_channel_rmse = torch.full((len(CHANNEL_NAMES),), float("nan"))

            val_loss, val_channel_mse, val_channel_rmse = evaluate_masked_reconstruction(
                model=self.model,
                data_loader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
                autocast_kwargs=self.autocast_kwargs,
                non_blocking_transfer=self.non_blocking_transfer,
                seq_length=self.seq_length,
                mask_block_min_len=self.mask_block_min_len,
                mask_block_max_len=self.mask_block_max_len,
                epoch=epoch,
                epochs=self.epochs,
                phase_label="Val",
            )

            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f}"
            )
            logger.info(
                "Train Channel MSE - "
                f"{format_channel_metrics(CHANNEL_NAMES, train_channel_mse.tolist())}"
            )
            logger.info(
                "Train Channel RMSE - "
                f"{format_channel_metrics(CHANNEL_NAMES, train_channel_rmse.tolist())}"
            )
            logger.info(
                "Val Channel MSE - "
                f"{format_channel_metrics(CHANNEL_NAMES, val_channel_mse.tolist())}"
            )
            logger.info(
                "Val Channel RMSE - "
                f"{format_channel_metrics(CHANNEL_NAMES, val_channel_rmse.tolist())}"
            )

            train_loss_history.append(float(train_loss))
            val_loss_history.append(float(val_loss))
            train_channel_mse_history.append(train_channel_mse.tolist())
            val_channel_mse_history.append(val_channel_mse.tolist())
            train_channel_rmse_history.append(train_channel_rmse.tolist())
            val_channel_rmse_history.append(val_channel_rmse.tolist())

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                save_dir = os.path.join(hydra.utils.get_original_cwd(), "checkpoints", "pretrain", self.version or "default")
                os.makedirs(save_dir, exist_ok=True)
                checkpoint_path = os.path.join(save_dir, f"best_pretrained_epoch_{epoch + 1}.pth")

                torch.save(unwrap_model(self.model).state_dict(), checkpoint_path)
                logger.info(f"Saved new best pretrained checkpoint to {checkpoint_path}")

                self.best_checkpoint_path = checkpoint_path

            if should_save_intermediate_epoch(self.cfg, epoch):
                save_pretrain_artifacts(
                    cfg=self.cfg,
                    channel_names=CHANNEL_NAMES,
                    train_losses=train_loss_history,
                    val_losses=val_loss_history,
                    train_channel_mse=train_channel_mse_history,
                    val_channel_mse=val_channel_mse_history,
                    train_channel_rmse=train_channel_rmse_history,
                    val_channel_rmse=val_channel_rmse_history,
                    best_epoch=self.best_epoch,
                    best_checkpoint_path=self.best_checkpoint_path,
                    tag=f"epoch_{epoch + 1:03d}",
                )

        save_pretrain_artifacts(
            cfg=self.cfg,
            channel_names=CHANNEL_NAMES,
            train_losses=train_loss_history,
            val_losses=val_loss_history,
            train_channel_mse=train_channel_mse_history,
            val_channel_mse=val_channel_mse_history,
            train_channel_rmse=train_channel_rmse_history,
            val_channel_rmse=val_channel_rmse_history,
            best_epoch=self.best_epoch,
            best_checkpoint_path=self.best_checkpoint_path,
            tag="final",
        )

        return self.best_checkpoint_path


def run_pretrain(cfg: DictConfig, version=None):
    trainer = Pretrainer(cfg, version=version)
    return trainer.run()


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_pretrain(cfg, version=cfg.get("version", "0.1.0"))


if __name__ == "__main__":
    main()
