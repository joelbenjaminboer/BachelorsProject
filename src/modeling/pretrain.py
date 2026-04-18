import os
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from loguru import logger
from tqdm import tqdm

from src.dataloader import build_pretrain_dataloaders
from src.modeling.encoder import IMU_Intent_Encoder

# Setup device correctly
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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


def evaluate_masked_reconstruction(
    model,
    data_loader,
    criterion,
    seq_length,
    mask_ratio,
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

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs} [{phase_label}]"):
            past_imu = batch["past_imu"].to(device)

            batch_size_current = past_imu.size(0)
            mask = torch.rand(batch_size_current, seq_length, device=device) < mask_ratio

            reconstructed_imu = model(past_imu, mask=mask)
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


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger.info(f"Using device: {device} for Pretraining")

    seq_length = cfg.training.context_length
    forecast_horizon = cfg.training.forecast_horizon
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs
    learning_rate = cfg.training.learning_rate
    split_ratio = cfg.training.get("split_ratio", 0.9)
    split_strategy = cfg.training.get("split_strategy", "loso")
    holdout_subjects = cfg.training.get("holdout_subjects", [])
    split_seed = cfg.training.get("split_seed", 42)
    processed_dir = hydra.utils.to_absolute_path(cfg.dataset.processed_dir)

    num_workers = min(os.cpu_count() or 1, 8)
    train_loader, val_loader, _, split_info = build_pretrain_dataloaders(
        data_dir=processed_dir,
        seq_length=seq_length,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
        split_ratio=split_ratio,
        split_strategy=split_strategy,
        holdout_subjects=holdout_subjects,
        include_test_loader=False,
        seed=split_seed,
        num_workers=num_workers,
    )

    logger.info(
        f"Split strategy: {split_info['split_strategy']} | "
        f"held-out subjects: {', '.join(holdout_subjects) or 'None'}"
    )
    logger.info(
        f"Samples - train: {split_info['train_samples']}, val: {split_info['val_samples']}"
    )
    logger.info(
        f"Train subjects ({len(split_info['train_subjects'])}): "
        f"{', '.join(split_info['train_subjects']) or 'None'}"
    )
    logger.info(
        f"Val subjects ({len(split_info['val_subjects'])}): "
        f"{', '.join(split_info['val_subjects']) or 'None'}"
    )

    # Initialize model
    model = IMU_Intent_Encoder(
        input_features=6,
        seq_length=125,
        d_model=64,
        num_heads=4,
        num_layers=3,
        dim_feedforward=128,
    ).to(device)

    # Setup Optimization
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Pretraining specifics
    mask_ratio = cfg.training.get("mask_ratio", 0.2)  # Default 20% masking if not in config
    best_val_loss = float("inf")

    logger.info(f"Starting MAE Pretraining with mask ratio {mask_ratio * 100}%")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_sq_error_sum = torch.zeros(len(CHANNEL_NAMES), device=device)
        train_masked_count = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
            past_imu = batch["past_imu"].to(device)
            # past_imu shape: (Batch, seq_length, 6)

            # Generate random mask based on mask_ratio
            batch_size_current = past_imu.size(0)

            # Create boolean mask (True = Masked token)
            # Shape: (Batch, seq_length)
            mask = torch.rand(batch_size_current, seq_length, device=device) < mask_ratio

            optimizer.zero_grad()

            # Pass original data and mask to model
            reconstructed_imu = model(past_imu, mask=mask)

            # Calculate MSE only on the masked timesteps to force the model to infer them
            # OR over all tokens. Standard MAE usually calculates loss only over masked patches.
            loss = criterion(reconstructed_imu[mask], past_imu[mask])

            batch_sq_error_sum, batch_masked_count = masked_channel_sse(
                reconstructed_imu, past_imu, mask
            )
            if batch_sq_error_sum is not None:
                train_sq_error_sum += batch_sq_error_sum
                train_masked_count += batch_masked_count

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if len(train_loader) > 0:
            train_loss /= len(train_loader)

        if train_masked_count > 0:
            train_channel_mse = (train_sq_error_sum / train_masked_count).detach().cpu()
            train_channel_rmse = torch.sqrt(train_channel_mse)
        else:
            train_channel_mse = torch.full((len(CHANNEL_NAMES),), float("nan"))
            train_channel_rmse = torch.full((len(CHANNEL_NAMES),), float("nan"))

        val_loss, val_channel_mse, val_channel_rmse = evaluate_masked_reconstruction(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            seq_length=seq_length,
            mask_ratio=mask_ratio,
            epoch=epoch,
            epochs=epochs,
            phase_label="Val",
        )

        logger.info(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
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
            f"Val Channel MSE - {format_channel_metrics(CHANNEL_NAMES, val_channel_mse.tolist())}"
        )
        logger.info(
            "Val Channel RMSE - "
            f"{format_channel_metrics(CHANNEL_NAMES, val_channel_rmse.tolist())}"
        )

        # Save model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Create a dedicated directory for pretrain checkpoints
            save_dir = os.path.join(hydra.utils.get_original_cwd(), "checkpoints", "pretrain")
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = os.path.join(save_dir, f"best_pretrained_epoch_{epoch + 1}.pth")

            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved new best pretrained checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
