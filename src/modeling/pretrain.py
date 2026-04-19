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
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = device

        logger.info(f"Using device: {self.device} for Pretraining")

        self.seq_length = cfg.training.context_length
        self.forecast_horizon = cfg.training.forecast_horizon
        self.batch_size = cfg.training.batch_size
        self.epochs = cfg.training.epochs
        self.learning_rate = cfg.training.learning_rate
        self.split_ratio = cfg.training.get("split_ratio", 0.9)
        self.split_strategy = cfg.training.get("split_strategy", "loso")
        self.holdout_subjects = cfg.training.get("holdout_subjects", [])
        self.split_seed = cfg.training.get("split_seed", 42)
        self.processed_dir = hydra.utils.to_absolute_path(cfg.dataset.processed_dir)

        num_workers = cfg.training.get("num_workers", min(os.cpu_count() or 1, 8))
        persistent_workers = cfg.training.get("persistent_workers")
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
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

        logger.info(
            f"Split strategy: {self.split_info['split_strategy']} | "
            f"held-out subjects: {', '.join(self.holdout_subjects) or 'None'}"
        )
        logger.info(
            f"Samples - train: {self.split_info['train_samples']}, "
            f"val: {self.split_info['val_samples']}"
        )
        logger.info(
            f"Train subjects ({len(self.split_info['train_subjects'])}): "
            f"{', '.join(self.split_info['train_subjects']) or 'None'}"
        )
        logger.info(
            f"Val subjects ({len(self.split_info['val_subjects'])}): "
            f"{', '.join(self.split_info['val_subjects']) or 'None'}"
        )

        self.model = IMU_Intent_Encoder(
            input_features=6,
            seq_length=self.seq_length,
            forecast_horizon=self.forecast_horizon,
            d_model=64,
            num_heads=4,
            num_layers=3,
            dim_feedforward=128,
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )

        self.mask_ratio = cfg.training.get("mask_ratio", 0.2)
        self.best_val_loss = float("inf")
        self.best_checkpoint_path = None

    def run(self):
        logger.info(f"Starting MAE Pretraining with mask ratio {self.mask_ratio * 100}%")

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            train_sq_error_sum = torch.zeros(len(CHANNEL_NAMES), device=self.device)
            train_masked_count = 0

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Train]"):
                past_imu = batch["past_imu"].to(self.device)
                batch_size_current = past_imu.size(0)
                mask = (
                    torch.rand(batch_size_current, self.seq_length, device=self.device)
                    < self.mask_ratio
                )

                self.optimizer.zero_grad()

                reconstructed_imu = self.model(past_imu, mask=mask, task="reconstruct")
                loss = self.criterion(reconstructed_imu[mask], past_imu[mask])

                batch_sq_error_sum, batch_masked_count = masked_channel_sse(
                    reconstructed_imu, past_imu, mask
                )
                if batch_sq_error_sum is not None:
                    train_sq_error_sum += batch_sq_error_sum
                    train_masked_count += batch_masked_count

                loss.backward()
                self.optimizer.step()
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
                seq_length=self.seq_length,
                mask_ratio=self.mask_ratio,
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

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_dir = os.path.join(hydra.utils.get_original_cwd(), "checkpoints", "pretrain")
                os.makedirs(save_dir, exist_ok=True)
                checkpoint_path = os.path.join(save_dir, f"best_pretrained_epoch_{epoch + 1}.pth")

                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(f"Saved new best pretrained checkpoint to {checkpoint_path}")

                self.best_checkpoint_path = checkpoint_path

        return self.best_checkpoint_path


def run_pretrain(cfg: DictConfig):
    trainer = Pretrainer(cfg)
    return trainer.run()


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_pretrain(cfg)


if __name__ == "__main__":
    main()
