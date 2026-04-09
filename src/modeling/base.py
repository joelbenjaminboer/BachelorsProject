import torch
import torch.nn as nn
from typing import Any, Dict

class BaseForecaster(nn.Module):
    """
    Base class for all forecasting models to ensure a unified API.
    """
    def __init__(self):
        super().__init__()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a single training step.
        Returns the loss tensor.
        """
        raise NotImplementedError

    def predict_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a prediction step.
        Returns the output predictions.
        """
        raise NotImplementedError

    def save_model(self, path: str):
        """
        Save model to the given path.
        """
        raise NotImplementedError

    def load_model(self, path: str):
        """
        Load model from the given path.
        """
        raise NotImplementedError
