import os
import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.modeling.timesfm_wrapper import TimesFMWrapper
from loguru import logger
from tqdm import tqdm

class KneeAngleDataset(Dataset):
    def __init__(self, data_dir: str, context_length: int, forecast_horizon: int):
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.window_size = context_length + forecast_horizon
        
        self.sequences = []
        self.valid_indices = []
        self.load_data(data_dir)

    def load_data(self, data_dir: str):
        path = Path(data_dir)
        files = list(path.rglob("*.parquet"))
        logger.info(f"Found {len(files)} parquet files in {data_dir}. Loading sequences...")
        
        for seq_idx, f in enumerate(files):
            df = pd.read_parquet(f)
            if "KneeAngle" not in df.columns:
                continue
            
            # Keep sequences as 1D tensors
            values = torch.tensor(df["KneeAngle"].values, dtype=torch.float32)
            self.sequences.append(values)
            
            # Map valid windows (sequence_index, window_start_index)
            num_windows = len(values) - self.window_size + 1
            if num_windows > 0:
                self.valid_indices.extend([(seq_idx, i) for i in range(num_windows)])

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        seq_idx, start_idx = self.valid_indices[idx]
        window = self.sequences[seq_idx][start_idx : start_idx + self.window_size]
        
        past_values = window[:self.context_length]
        future_values = window[self.context_length:]
        return {"past_values": past_values, "future_values": future_values}

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    context_length = cfg.training.context_length
    forecast_horizon = cfg.training.forecast_horizon
    batch_size = cfg.training.batch_size
    processed_dir = hydra.utils.to_absolute_path(cfg.dataset.processed_dir)

    dataset = KneeAngleDataset(processed_dir, context_length, forecast_horizon)
    logger.info(f"Total samples: {len(dataset)}")

    # Train/val split
    train_size = int(cfg.training.split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model_id = cfg.models.timesfm.model_id
    logger.info(f"Initializing TimesFMWrapper with {model_id}")
    model = TimesFMWrapper(model_id=model_id)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

    epochs = cfg.training.epochs
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model.train_step(batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if len(train_loader) > 0:
            train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model.train_step(batch)
                val_loss += loss.item()

        if len(val_loader) > 0:
            val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()
