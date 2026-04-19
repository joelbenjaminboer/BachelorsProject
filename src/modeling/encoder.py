import torch
import torch.nn as nn


class IMU_Intent_Encoder(nn.Module):
    def __init__(
        self,
        input_features=6,
        seq_length=125,
        forecast_horizon=50,
        d_model=64,
        num_heads=4,
        num_layers=3,
        dim_feedforward=128,
    ):
        super(IMU_Intent_Encoder, self).__init__()
        # Project the 6 IMU features into a larger dimension
        self.input_projection = nn.Linear(input_features, d_model)
        
        # Learnable CLS token for pooling
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Positional encoding layer
        self.positional_layer = PositionalEncoding(d_model=d_model, max_len=seq_length + 1)

        # Setup the Transformer Encoder layers
        # batch_first=True makes our shapes (Batch, Seq, Features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pretraining: Mask Token and Reconstruction Head
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.reconstruction_head = nn.Linear(d_model, input_features)

        # Downstream prediction: Regression Head
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, forecast_horizon * input_features),
        )

    def forward(self, x, mask=None, task="reconstruct"):
            # 1. Project: (Batch, 125, 6) -> (Batch, 125, 64)
            x = self.input_projection(x)

            # 2. Apply Masking FIRST (while sequence is still 125)
            if mask is not None:
                bool_mask = mask.unsqueeze(-1).bool()
                x = torch.where(bool_mask, self.mask_token, x)

            # 3. Prepend CLS token (125 -> 126)
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1) 

            # 4. Positional Encoding
            x = self.positional_layer(x)

            # 5. Transformer Encoder
            encoded_x = self.transformer_encoder(x)

            if task == "reconstruct":
                # Remove the CLS token (index 0) so output is 125 again
                # Shape: (Batch, 1:126, 64) -> (Batch, 125, 6)
                return self.reconstruction_head(encoded_x[:, 1:, :])
                
            if task == "predict":
                # Use only the CLS token for the regression head
                pooled = encoded_x[:, 0, :] 
                return self.regression_head(pooled)

            raise ValueError(f"Unknown task: {task}")


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
        x = x + self.pe[:, : x.size(1), :]
        return x
