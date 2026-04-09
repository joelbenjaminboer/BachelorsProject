import torch
from transformers import AutoModelForSeq2SeqLM
from src.modeling.base import BaseForecaster
from typing import Dict

class Chronos2Wrapper(BaseForecaster):
    """
    Wrapper for Amazon's Chronos models using HuggingFace Transformers.
    """
    def __init__(self, model_id: str = "amazon/chronos-t5-small"):
        super().__init__()
        self.model_id = model_id
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate loss for a training step. 
        Expects 'input_ids', 'labels', and optionally 'attention_mask' in batch.
        """
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch["labels"]
        )
        return outputs.loss

    def predict_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate predictions.
        """
        return self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask")
        )

    def save_model(self, path: str):
        self.model.save_pretrained(path)

    def load_model(self, path: str):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
