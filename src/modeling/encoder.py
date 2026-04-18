import torch
import torch.nn as nn


class IMU_Intent_Encoder(nn.Module):
    def __init__(
        self,
        input_features=6,
        seq_length=125,
        d_model=64,
        num_heads=4,
        num_layers=3,
        dim_feedforward=128,
    ):
        super(IMU_Intent_Encoder, self).__init__()
        # Project the 6 IMU features into a larger dimension
        self.input_projection = nn.Linear(input_features, d_model)

        # Positional encoding layer
        self.positional_layer = PositionalEncoding(d_model=d_model, max_len=seq_length)

        # Setup the Transformer Encoder layers
        # batch_first=True makes our shapes (Batch, Seq, Features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pretraining: Mask Token and Reconstruction Head
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.reconstruction_head = nn.Linear(d_model, input_features)

    def forward(self, x, mask=None):
        # x shape: (Batch, 125, 6)

        # Project: (Batch, 125, 6) -> (Batch, 125, 64)
        x = self.input_projection(x)

        # Apply Masking
        if mask is not None:
            # mask shape: (Batch, 125)
            # Expand mask token to the masked positions
            bool_mask = mask.unsqueeze(-1).bool()
            x = torch.where(bool_mask, self.mask_token, x)

        # Add positional encoding: (Batch, 125, 64) -> (Batch, 125, 64)
        x = self.positional_layer(x)

        # Pass through the Encoder
        # Output shape is still (Batch, 125, 64)
        encoded_x = self.transformer_encoder(x)

        # Reconstruct the original 6 IMU features
        # Shape: (Batch, 125, 64) -> (Batch, 125, 6)
        predictions = self.reconstruction_head(encoded_x)

        return predictions


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (Batch, Seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return x
