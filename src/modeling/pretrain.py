import os
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from loguru import logger
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

# Import the dataset and our new Masked Autoencoder from your existing setup
from src.modeling.train import IMUKneeDataset
from src.modeling.encoder import IMU_Intent_Encoder

# Setup device correctly
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger.info(f"Using device: {device} for Pretraining")

    seq_length = cfg.training.context_length
    forecast_horizon = cfg.training.forecast_horizon
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs
    learning_rate = cfg.training.learning_rate
    processed_dir = hydra.utils.to_absolute_path(cfg.dataset.processed_dir)

    # Note: We still load the same Dataset, but we only extract the IMU sequence (past_imu) 
    # and ignore the forecasting (future_knee) target.
    dataset = IMUKneeDataset(processed_dir, seq_length, forecast_horizon)

    # Validation Split
    split_ratio = cfg.training.get("split_ratio", 0.9)
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    # Optional seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # Dataloaders
    num_workers = min(os.cpu_count() or 1, 8)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        persistent_workers=(num_workers > 0)
    )

    # Initialize model
    model = IMU_Intent_Encoder(
        input_features=6, 
        seq_length=seq_length,
        # Ensure these default match your config or `train.py` architecture overrides
        d_model=64, 
        num_heads=4, 
        num_layers=3, 
        dim_feedforward=128
    ).to(device)

    # Setup Optimization
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Pretraining specifics
    mask_ratio = cfg.training.get("mask_ratio", 0.2)  # Default 20% masking if not in config
    best_val_loss = float('inf')

    logger.info(f"Starting MAE Pretraining with mask ratio {mask_ratio*100}%")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
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
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if len(train_loader) > 0:
            train_loss /= len(train_loader)

        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                past_imu = batch["past_imu"].to(device)
                
                # Same masking strategy during validation
                batch_size_current = past_imu.size(0)
                mask = torch.rand(batch_size_current, seq_length, device=device) < mask_ratio
                
                reconstructed_imu = model(past_imu, mask=mask)
                loss = criterion(reconstructed_imu[mask], past_imu[mask])
                val_loss += loss.item()

        if len(val_loader) > 0:
            val_loss /= len(val_loader)
            
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Save model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Create a dedicated directory for pretrain checkpoints
            save_dir = os.path.join(hydra.utils.get_original_cwd(), "checkpoints", "pretrain")
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = os.path.join(save_dir, f"best_pretrained_epoch_{epoch+1}.pth")
            
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved new best pretrained checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()