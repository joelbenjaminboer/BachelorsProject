import torch
import torch.nn as nn

class IMU_Intent_Encoder(nn.Module):
    def __init__(self, input_features=6, seq_length=125, d_model=64, num_heads=4, num_layers=3, forecast_steps=50):
        super(IMU_Intent_Encoder, self).__init__()
        
        # 1. Project the 6 IMU features into a larger dimension
        self.input_projection = nn.Linear(input_features, d_model)
        
        # 2. Setup the Transformer Encoder layers
        # batch_first=True makes our shapes (Batch, Seq, Features)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. The Regression Head
        # After flattening, the shape will be (Batch, seq_length * d_model)
        self.flatten = nn.Flatten()
        self.regression_head = nn.Linear(seq_length * d_model, forecast_steps)

    def forward(self, x):
        # x shape: (Batch, 125, 6)
        
        # Project: (Batch, 125, 6) -> (Batch, 125, 64)
        x = self.input_projection(x)
        
        # Add Positional Encoding here (skipping the math for brevity, 
        # but you would add a positional matrix to x here)
        
        # Pass through the Encoder
        # Output shape is still (Batch, 125, 64)
        encoded_x = self.transformer_encoder(x)
        
        # Flatten all 125 time steps into one long vector per batch
        # Shape: (Batch, 125 * 64) -> (Batch, 8000)
        flat_x = self.flatten(encoded_x)
        
        # Predict the next 50 ms of knee angles
        # Shape: (Batch, 8000) -> (Batch, 50)
        predictions = self.regression_head(flat_x)
        
        return predictions