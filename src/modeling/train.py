import os
import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.modeling.encoder import IMU_Intent_Encoder
from loguru import logger
from tqdm import tqdm
import torch.nn as nn

# Setup device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class IMUKneeDataset(Dataset):
    def __init__(self, data_dir: str, seq_length: int, forecast_horizon: int):
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.window_size = seq_length + forecast_horizon

        self.imu_sequences = []
        self.knee_sequences = []
        self.valid_indices = []
        self.load_data(data_dir)

    def load_data(self, data_dir: str):
        path = Path(data_dir)
        files = list(path.rglob("*.parquet"))
        logger.info(f"Found {len(files)} parquet files in {data_dir}. Loading sequences...")

        imu_cols = ["Ax", "Ay", "Az", "Gy", "Gz", "Gx"]

        for seq_idx, f in enumerate(files):
            df = pd.read_parquet(f)
            if "KneeAngle" not in df.columns or not all(c in df.columns for c in imu_cols):
                continue

            imu_values = torch.tensor(df[imu_cols].values, dtype=torch.float32)
            knee_values = torch.tensor(df["KneeAngle"].values, dtype=torch.float32)

            self.imu_sequences.append(imu_values)
            self.knee_sequences.append(knee_values)

            num_windows = len(imu_values) - self.window_size + 1
            if num_windows > 0:
                self.valid_indices.extend([(seq_idx, i) for i in range(num_windows)])

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        seq_idx, start_idx = self.valid_indices[idx]

        # Past IMU data
        past_imu = self.imu_sequences[seq_idx][start_idx : start_idx + self.seq_length]

        # Future Knee Angle
        target_idx = start_idx + self.seq_length
        future_knee = self.knee_sequences[seq_idx][target_idx : target_idx + self.forecast_horizon]

        return {"past_imu": past_imu, "future_knee": future_knee}


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger.info(f"Using device: {device}")

    # Use seq_length mapping to your config's context_length
    seq_length = cfg.training.context_length
    forecast_horizon = cfg.training.forecast_horizon
    batch_size = cfg.training.batch_size
    processed_dir = hydra.utils.to_absolute_path(cfg.dataset.processed_dir)

    dataset = IMUKneeDataset(processed_dir, seq_length, forecast_horizon)
    logger.info(f"Total samples: {len(dataset)}")

    # Train/val split
    train_size = int(cfg.training.split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Optimize dataloaders for Apple Silicon / multi-core processors
    num_workers = min(os.cpu_count() or 1, 8)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    logger.info("Initializing IMU_Intent_Encoder from encoder.py")
    model = IMU_Intent_Encoder(
        input_features=6, seq_length=seq_length, forecast_steps=forecast_horizon
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )
    criterion = nn.MSELoss()

    epochs = cfg.training.epochs
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
            past_imu = batch["past_imu"].to(device)
            future_knee = batch["future_knee"].to(device)

            optimizer.zero_grad()
            predictions = model(past_imu)
            loss = criterion(predictions, future_knee)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if len(train_loader) > 0:
            train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                past_imu = batch["past_imu"].to(device)
                future_knee = batch["future_knee"].to(device)

                predictions = model(past_imu)
                loss = criterion(predictions, future_knee)
                val_loss += loss.item()

        if len(val_loader) > 0:
            val_loss /= len(val_loader)
        logger.info(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
        )

        # save model checkpoint if weights are improving
        if epoch == 0 or val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(
                hydra.utils.get_original_cwd(), "checkpoints", f"best_model_epoch_{epoch + 1}.pth"
            )
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved new best model checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
