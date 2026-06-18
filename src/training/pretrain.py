import math
import os

from loguru import logger
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from src.models.factory import build_loss, build_masked_loss, build_optimizer
from src.runtime import (
    RunContext,
    autocast_context,
    backward_and_step,
    build_grad_scaler,
    checkpoint_dir,
    unwrap_model,
)
from src.training.plotting import save_pretrain_artifacts, should_save_intermediate_epoch

# First 6 are the raw IMU channels; the last 4 are the optional handcrafted
# features (dataset.handcrafted_features). Per-channel logging indexes this tuple,
# so it must cover the configured input_features count (6 or 10).
CHANNEL_NAMES = ("Ax", "Ay", "Az", "Gy", "Gz", "Gx", "AccL2", "GyrL2", "AccM", "GyrM")


def _patchify_target(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[B, T, C] -> [B, ceil(T/P), P*C], matching encoder._patchify (zero-pads if T % P != 0)."""
    import torch.nn.functional as F
    B, T, C = x.shape
    n = math.ceil(T / patch_size)
    pad = n * patch_size - T
    if pad > 0:
        x = F.pad(x, (0, 0, 0, pad))
    return x.reshape(B, n, patch_size * C)


def masked_channel_sse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int = 1,
    n_channels: int = len(CHANNEL_NAMES),
):
    """Compute per-channel sum-of-squared-errors over masked positions.

    Uses element-wise multiplication instead of boolean fancy-indexing so no
    CPU–GPU sync is forced per batch (boolean indexing + .size(0) would require
    mask.sum() to be materialized as a Python int every step).

    Returns (sse_per_channel [n_channels], count [scalar]) as CUDA tensors.
    """
    # Detach: this is a metrics-only path. Without detach, the returned sse
    # carries the autograd graph, and accumulating it across the epoch
    # (train_sq_error_sum += ...) pins every batch's graph in GPU memory → OOM.
    # diff_sq: [B, n_patches, P*C]
    diff_sq = (predictions.detach() - targets.detach()).pow(2)

    # mask: [B, n_patches] bool → float weight [B, n_patches, 1]
    mask_f = mask.to(dtype=diff_sq.dtype).unsqueeze(-1)

    if patch_size > 1:
        B, n_patches, PC = predictions.shape
        # Expand patches: [B, n_patches, P*C] -> [B, n_patches, P, C]
        diff_sq = diff_sq.reshape(B, n_patches, patch_size, n_channels)
        # weight each patch position: [B, n_patches, 1, 1]
        mask_f = mask_f.unsqueeze(-1)
        sse = (diff_sq * mask_f).sum(dim=(0, 1, 2))  # [n_channels]
        count = mask_f.sum() * patch_size  # scalar CUDA tensor
    else:
        sse = (diff_sq * mask_f).sum(dim=(0, 1))  # [n_channels]
        count = mask_f.sum()  # scalar CUDA tensor

    return sse, count


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
        (torch.rand(batch_size, device=device) * (max_start_positions + 1).float()).floor().long()
    )

    time_indices = torch.arange(seq_length, device=device).unsqueeze(0)
    return (time_indices >= start_positions.unsqueeze(1)) & (
        time_indices < (start_positions + block_lengths).unsqueeze(1)
    )


def build_random_mask(
    batch_size: int,
    seq_length: int,
    mask_ratio: float,
    device: torch.device,
):
    return torch.rand(batch_size, seq_length, device=device) < mask_ratio


def evaluate_masked_reconstruction(
    model,
    data_loader,
    masked_criterion,
    device,
    autocast_kwargs,
    non_blocking_transfer,
    seq_length,
    mask_block_min_len,
    mask_block_max_len,
    mask_type,
    mask_ratio,
    epoch,
    epochs,
    phase_label,
    patch_size: int = 1,
    n_channels: int = len(CHANNEL_NAMES),
):
    if data_loader is None or len(data_loader) == 0:
        nan_metrics = torch.full((len(CHANNEL_NAMES),), float("nan"))
        return float("nan"), nan_metrics, nan_metrics

    model.eval()
    phase_loss = torch.zeros((), device=device)
    phase_sq_error_sum = torch.zeros(len(CHANNEL_NAMES), device=device)
    phase_masked_count = torch.zeros((), device=device)

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs} [{phase_label}]"):
            past_imu = batch["past_imu"].to(device, non_blocking=non_blocking_transfer)
            B = past_imu.size(0)

            if mask_type == "random":
                mask = build_random_mask(B, seq_length, mask_ratio, device)
            elif mask_type == "mixed":
                block = build_contiguous_block_mask(
                    B, seq_length, mask_block_min_len, mask_block_max_len, device
                )
                rand = build_random_mask(B, seq_length, mask_ratio / 2, device)
                mask = block | rand
            else:
                mask = build_contiguous_block_mask(
                    B, seq_length, mask_block_min_len, mask_block_max_len, device
                )

            with autocast_context(autocast_kwargs):
                reconstructed_imu = model(past_imu, mask=mask, task="reconstruct")

            # fp32 loss/target (see training loop note): avoids fp16 overflow.
            target_imu = _patchify_target(past_imu, patch_size) if patch_size > 1 else past_imu
            reconstructed_imu = reconstructed_imu.float()
            loss = masked_criterion(reconstructed_imu, target_imu, mask)

            batch_sq_error_sum, batch_masked_count = masked_channel_sse(
                reconstructed_imu, target_imu, mask, patch_size=patch_size, n_channels=n_channels
            )
            phase_sq_error_sum += batch_sq_error_sum
            phase_masked_count += batch_masked_count

            phase_loss += loss.detach()

    # Single sync per epoch to read accumulated scalars back to CPU
    phase_loss = (phase_loss / len(data_loader)).item()
    phase_masked_count_val = phase_masked_count.item()

    if phase_masked_count_val > 0:
        phase_channel_mse = (phase_sq_error_sum / phase_masked_count_val).detach().cpu()
        phase_channel_rmse = torch.sqrt(phase_channel_mse)
    else:
        phase_channel_mse = torch.full((len(CHANNEL_NAMES),), float("nan"))
        phase_channel_rmse = torch.full((len(CHANNEL_NAMES),), float("nan"))

    return phase_loss, phase_channel_mse, phase_channel_rmse


class Pretrainer:
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

        self.criterion = build_loss(cfg)
        # Sync-free masked loss: element-wise weighted by float mask
        # avoids boolean fancy-indexing which forces CPU-GPU sync for shape inference
        self.masked_criterion = build_masked_loss(cfg)
        self.optimizer = build_optimizer(cfg, self.model.parameters(), device=self.device)

        # Effective sequence length: reduced when using patch embedding
        raw_seq = cfg.training.context_length
        patch_size = None
        if hasattr(cfg.model, "encoder"):
            raw_patch = cfg.model.encoder.get("patch_size", None)
            patch_size = int(raw_patch) if raw_patch is not None else None
        self.patch_size = patch_size or 1
        self.n_channels = int(cfg.model.encoder.get("input_features", len(CHANNEL_NAMES)))
        self.seq_length = math.ceil(raw_seq / patch_size) if patch_size else raw_seq

        # Mask configuration: percentage-based or absolute block lengths
        self.mask_ratio = float(cfg.training.get("mask_ratio", 0.0))
        self.mask_type = str(cfg.training.get("mask_type", "block")).lower()

        if self.mask_ratio > 0.0:
            target = int(self.mask_ratio * self.seq_length)
            self.mask_block_min_len = max(1, int(target * 0.8))
            self.mask_block_max_len = max(self.mask_block_min_len, int(target * 1.2))
        else:
            raw_min = int(cfg.training.get("mask_block_min_len", 10))
            raw_max = int(cfg.training.get("mask_block_max_len", 20))
            self.mask_block_min_len = min(raw_min, raw_max)
            self.mask_block_max_len = max(raw_min, raw_max)

        logger.info(
            "MAE mask config: type={} ratio={:.0%} block_len={}-{} seq_len={}",
            self.mask_type,
            self.mask_ratio if self.mask_ratio > 0 else self.mask_block_max_len / self.seq_length,
            self.mask_block_min_len,
            self.mask_block_max_len,
            self.seq_length,
        )

        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.best_checkpoint_path = None

    def _build_mask(self, batch_size: int) -> torch.Tensor:
        if self.mask_type == "random":
            return build_random_mask(batch_size, self.seq_length, self.mask_ratio, self.device)
        if self.mask_type == "mixed":
            block = build_contiguous_block_mask(
                batch_size,
                self.seq_length,
                self.mask_block_min_len,
                self.mask_block_max_len,
                self.device,
            )
            rand = build_random_mask(batch_size, self.seq_length, self.mask_ratio / 2, self.device)
            return block | rand
        return build_contiguous_block_mask(
            batch_size,
            self.seq_length,
            self.mask_block_min_len,
            self.mask_block_max_len,
            self.device,
        )

    def run(self):
        logger.info(
            "Starting MAE Pretraining with {} masking "
            f"(length range: {self.mask_block_min_len}-{self.mask_block_max_len})",
            self.mask_type,
        )

        train_loss_history = []
        val_loss_history = []
        train_channel_mse_history = []
        val_channel_mse_history = []
        train_channel_rmse_history = []
        val_channel_rmse_history = []
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = torch.zeros((), device=self.device)
            train_sq_error_sum = torch.zeros(len(CHANNEL_NAMES), device=self.device)
            train_masked_count = torch.zeros((), device=self.device)
            self.optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(
                tqdm(self.ctx.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Train]")
            ):
                past_imu = batch["past_imu"].to(
                    self.device,
                    non_blocking=self.ctx.non_blocking_transfer,
                )
                mask = self._build_mask(past_imu.size(0))

                with autocast_context(self.autocast_kwargs):
                    reconstructed_imu = self.model(past_imu, mask=mask, task="reconstruct")

                # Loss/target in fp32 outside autocast: the masked SSE sums many
                # squared errors and can overflow fp16 (>65504) → NaN. The heavy
                # transformer matmuls already ran in fp16 inside autocast, so the
                # speedup is kept while the reduction stays numerically safe.
                target_imu = (
                    _patchify_target(past_imu, self.patch_size)
                    if self.patch_size > 1
                    else past_imu
                )
                reconstructed_imu = reconstructed_imu.float()
                loss = self.masked_criterion(reconstructed_imu, target_imu, mask)

                batch_sq_error_sum, batch_masked_count = masked_channel_sse(
                    reconstructed_imu,
                    target_imu,
                    mask,
                    patch_size=self.patch_size,
                    n_channels=self.n_channels,
                )
                train_sq_error_sum += batch_sq_error_sum
                train_masked_count += batch_masked_count

                backward_and_step(
                    loss=loss,
                    model=self.model,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    accumulation_steps=self.ctx.accumulation_steps,
                    clip_grad_norm=self.ctx.clip_grad_norm,
                    batch_idx=batch_idx,
                    total_batches=len(self.ctx.train_loader),
                )

                train_loss += loss.detach()

            train_loss = (
                (train_loss / len(self.ctx.train_loader)).item()
                if len(self.ctx.train_loader) > 0
                else 0.0
            )

            # Single sync per epoch to read count back to CPU
            train_masked_count_val = train_masked_count.item()
            if train_masked_count_val > 0:
                train_channel_mse = (train_sq_error_sum / train_masked_count_val).detach().cpu()
                train_channel_rmse = torch.sqrt(train_channel_mse)
            else:
                train_channel_mse = torch.full((len(CHANNEL_NAMES),), float("nan"))
                train_channel_rmse = torch.full((len(CHANNEL_NAMES),), float("nan"))

            val_loss, val_channel_mse, val_channel_rmse = evaluate_masked_reconstruction(
                model=self.model,
                data_loader=self.ctx.val_loader,
                masked_criterion=self.masked_criterion,
                device=self.device,
                autocast_kwargs=self.autocast_kwargs,
                non_blocking_transfer=self.ctx.non_blocking_transfer,
                seq_length=self.seq_length,
                mask_block_min_len=self.mask_block_min_len,
                mask_block_max_len=self.mask_block_max_len,
                mask_type=self.mask_type,
                mask_ratio=self.mask_ratio,
                epoch=epoch,
                epochs=self.epochs,
                phase_label="Val",
                patch_size=self.patch_size,
                n_channels=self.n_channels,
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

            improved = val_loss < self.best_val_loss - self.early_stop_min_delta

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                save_dir = checkpoint_dir(self.cfg, self.ctx.version, pretrain=True)
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

            if improved:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if self.early_stop_patience > 0 and epochs_no_improve >= self.early_stop_patience:
                    logger.info(
                        f"Early stopping pretraining: no improvement for "
                        f"{self.early_stop_patience} epochs (best val loss "
                        f"{self.best_val_loss:.4f} at epoch {self.best_epoch})"
                    )
                    break

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


def run_pretrain(cfg: DictConfig, model, ctx: RunContext):
    return Pretrainer(cfg, model=model, ctx=ctx).run()
