import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.modeling.timesfm_wrapper import TimesFMWrapper
from loguru import logger
from tqdm import tqdm

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
    processed_dir = Path(hydra.utils.to_absolute_path(cfg.dataset.processed_dir))
    
    # Load a single file for zero-shot testing
    files = list(processed_dir.rglob("*.parquet"))
    if not files:
        logger.error("No parquet files found!")
        return
        
    sample_file = files[0]
    logger.info(f"Loading sample data from {sample_file.name}")
    df = pd.read_parquet(sample_file)
    
    if "KneeAngle" not in df.columns:
        logger.error("KneeAngle not in dataset")
        return
        
    values = torch.tensor(df["KneeAngle"].values, dtype=torch.float32)
    
    num_eval_points = 1000 # Number of points to evaluate on
    start_idx = len(values) // 2 
    end_idx = start_idx + num_eval_points
    
    logger.info(f"Evaluating {num_eval_points} points starting from index {start_idx}...")

    # Load model for Zero-Shot
    model_id = cfg.models.timesfm.model_id
    logger.info(f"Initializing TimesFMWrapper with {model_id} for zero-shot inference...")
    model = TimesFMWrapper(model_id=model_id)
    model.to(device)
    model.eval()
    
    predictions = []
    ground_truth = []
    
    # Rolling prediction loop
    logger.info("Generating predictions in rolling windows...")
    for i in tqdm(range(start_idx, end_idx, forecast_horizon)):
        # Ensure we have enough context length before current point
        if i - context_length < 0:
            continue
            
        current_horizon = min(forecast_horizon, end_idx - i)
        
        # Shape: (1, context_length)
        past_values = values[i - context_length : i].unsqueeze(0).to(device)
        true_future = values[i : i + current_horizon].numpy()
        
        with torch.no_grad():
            preds = model.predict_step({"past_values": past_values})
            
        preds = preds[0, :current_horizon].cpu().numpy()
        
        predictions.extend(preds)
        ground_truth.extend(true_future)

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Calculate Metrics
    mse = np.mean((predictions - ground_truth) ** 2)
    mae = np.mean(np.abs(predictions - ground_truth))
    rmse = np.sqrt(mse)
    
    logger.info("====================================")
    logger.info(f"Zero-Shot Evaluation Metrics:")
    logger.info(f"MSE:  {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE:  {mae:.4f}")
    logger.info("====================================")
    
    # Plotting
    plt.figure(figsize=(15, 6))
    
    x_axis = np.arange(len(ground_truth))
    
    plt.plot(x_axis, ground_truth, label="Ground Truth (KneeAngle)", color="black", alpha=0.7)
    plt.plot(x_axis, predictions, label="Zero-Shot Forecasts", color="red", linestyle="--", alpha=0.9)
    
    # Add vertical lines to show window boundaries
    for v in range(0, len(ground_truth), forecast_horizon):
         plt.axvline(x=v, color='gray', linestyle=':', alpha=0.3)
    
    plt.title(f"TimesFM Rolling Zero-Shot Forecast ({len(ground_truth)} points)\nModel: {model_id} | File: {sample_file.name}\nMSE: {mse:.2f} | MAE: {mae:.2f}")
    plt.xlabel("Time Steps (Relative)")
    plt.ylabel("KneeAngle")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    out_dir = Path(hydra.utils.to_absolute_path("reports/figures"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "timesfm_rolling_zeroshot.png"
    plt.savefig(out_path)
    logger.info(f"Saved rolling zero-shot plot to {out_path}")

if __name__ == "__main__":
    main()
