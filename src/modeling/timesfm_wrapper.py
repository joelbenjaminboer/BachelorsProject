import torch
from transformers import TimesFmModelForPrediction
from src.modeling.base import BaseForecaster
from typing import Dict

class TimesFMWrapper(BaseForecaster):
    """
    Wrapper for Google's TimesFM models using HuggingFace Transformers.
    """
    def __init__(self, model_id: str = "google/timesfm-2.0-500m-pytorch"):
        super().__init__()
        self.model_id = model_id
        
        self.model = TimesFmModelForPrediction.from_pretrained(model_id)
        self.model = self.model.to(torch.float32)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate loss for a training step. 
        Expects 'past_values' and 'future_values' in the batch.
        """
        past_values = batch.get("past_values")
        future_values = batch.get("future_values")
        
        outputs = self.model(past_values=past_values)
        predictions = outputs.mean_predictions
        
        # Slicing predictions to match the forecast horizon if necessary
        horizon = future_values.shape[1]
        loss = torch.nn.functional.mse_loss(predictions[:, :horizon], future_values)
        
        return loss

    def predict_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate predictions.
        """
        with torch.no_grad():
            outputs = self.model(
                past_values=batch["past_values"]
            )
        return outputs.mean_predictions

    def save_model(self, path: str):
        self.model.save_pretrained(path)

    def load_model(self, path: str):
        self.model = TimesFmModelForPrediction.from_pretrained(path)
        self.model = self.model.to(torch.float32)
