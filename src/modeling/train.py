import os
import glob
import yaml
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import List
from datetime import datetime

from src.modeling.chronos_wrapper import Chronos2Wrapper
from src.modeling.timesfm_wrapper import TimesFMWrapper

class ENABL3SDataset(Dataset):
    def __init__(self, subject_dirs: List[str], context_length: int, forecast_horizon: int):
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.file_index_map = [] # stores (file_path, start_idx_in_file)
        
        total_window_size = context_length + forecast_horizon
        
        for subj_dir in subject_dirs:
            parquet_files = glob.glob(os.path.join(subj_dir, "*.parquet"))
            for p_file in parquet_files:
                # To be optimal, just read the metadata/length instead of loading full df 
                # but pandas read_parquet is still somewhat fast if we just map pointers
                df_len = pd.read_parquet(p_file, columns=["KneeAngle"]).shape[0]
                
                if df_len >= total_window_size:
                    for i in range(df_len - total_window_size + 1):
                        self.file_index_map.append((p_file, i))
                        
    def __len__(self):
        return len(self.file_index_map)

    def __getitem__(self, idx):
        p_file, start_idx = self.file_index_map[idx]
        total_window_size = self.context_length + self.forecast_horizon
        
        # Only read the specific rows you need for this single window
        # (Though optimizing pandas I/O here might require PyArrow datasets directly)
        series = pd.read_parquet(p_file, columns=["KneeAngle"]).values[start_idx:start_idx+total_window_size].astype(np.float32)
        series = series.flatten()
        
        past_values = torch.tensor(series[:self.context_length])
        future_values = torch.tensor(series[self.context_length:])
        
        return {
            "past_values": past_values,
            "future_values": future_values,
            "input_ids": past_values,
            "labels": future_values
        }

    
def load_config(config_path: str = "conf/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_subjects(processed_dir: str) -> List[str]:
    # Extract subjects from data/processed/ENABL3S directory (e.g. AB156, AB185...)
    subjects = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    return sorted(subjects)

def train_loso(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    processed_dir = config["dataset"]["processed_dir"]
    subjects = get_subjects(processed_dir)
    
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    split_ratio = config["training"]["split_ratio"]
    context_length = config["training"].get("context_length", 512)
    forecast_horizon = config["training"].get("forecast_horizon", 64)
    
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = os.path.join("outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Found subjects: {subjects}")
    
    for holdout_subject in subjects:
        print(f"\n{'='*40}\nStarting Fold for Holdout Subject: {holdout_subject}\n{'='*40}")
        
        train_subjects = [s for s in subjects if s != holdout_subject]
        
        # Collect paths for training subjects
        train_paths = [os.path.join(processed_dir, s) for s in train_subjects]
        test_paths = [os.path.join(processed_dir, holdout_subject)]
        
        # 90/10 split for train/val
        split_idx = int(len(train_paths) * split_ratio)
        train_split = train_paths[:split_idx]
        val_split = train_paths[split_idx:]
        
        # Datasets & Loaders
        train_dataset = ENABL3SDataset(train_split, context_length, forecast_horizon)
        val_dataset = ENABL3SDataset(val_split, context_length, forecast_horizon)
        test_dataset = ENABL3SDataset(test_paths, context_length, forecast_horizon)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Model initialization
        active_model = config["training"]["active_model"]
        if active_model == "chronos2":
            model_id = config["models"]["chronos2"]["model_id"]
            model = Chronos2Wrapper(model_id=model_id)
        elif active_model == "timesfm":
            model_id = config["models"]["timesfm"]["model_id"]
            model = TimesFMWrapper(model_id=model_id)
        else:
            raise ValueError(f"Model {active_model} not implemented.")
            
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Training Loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                loss = model.train_step(batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            if len(train_loader) > 0:
                train_loss /= len(train_loader)
            
            # Validation Loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    loss = model.train_step(batch)
                    val_loss += loss.item()
            
            if len(val_loader) > 0:
                val_loss /= len(val_loader)
                
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
        # Save model for this fold
        fold_dir = os.path.join(output_dir, f"fold_{holdout_subject}")
        os.makedirs(fold_dir, exist_ok=True)
        model.save_model(fold_dir)
        print(f"Saved model for fold {holdout_subject} to {fold_dir}")

if __name__ == "__main__":
    conf = load_config()
    train_loso(conf)
