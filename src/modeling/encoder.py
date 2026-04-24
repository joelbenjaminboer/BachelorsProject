import torch
import torch.nn as nn


class IMU_Intent_Encoder(nn.Module):
    def __init__(
        self,
        input_features,
        forecast_horizon,
        d_model,
        num_heads,
        num_layers,
        dim_feedforward,
        positional_encoding_max_len,
        positional_encoding_base,
        dropout=0.1
    ):
        super(IMU_Intent_Encoder, self).__init__()
        self.input_projection = nn.Linear(input_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.positional_layer = PositionalEncoding(
            d_model=d_model,
            max_len=positional_encoding_max_len,
            base=positional_encoding_base,
        )
        
        # Added dropout after positional encoding
        self.pos_drop = nn.Dropout(p=dropout) 

        # FIX 1: Set norm_first=True for Pre-Norm stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            batch_first=True, 
            dim_feedforward=dim_feedforward,
            norm_first=True, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # FIX 2: Final LayerNorm before the heads
        self.norm = nn.LayerNorm(d_model)

        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.reconstruction_head = nn.Linear(d_model, input_features)

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout), # Good practice in the MLP head
            nn.Linear(dim_feedforward, forecast_horizon),
        )

    def forward(self, x, mask=None, task="reconstruct"):
        x = self.input_projection(x)

        if mask is not None:
            expanded_mask = mask.unsqueeze(-1)
            x = torch.where(expanded_mask, self.mask_token.to(dtype=x.dtype), x)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 

        x = self.positional_layer(x)
        x = self.pos_drop(x) # Apply dropout

        encoded_x = self.transformer_encoder(x)
        
        # Apply final LayerNorm
        encoded_x = self.norm(encoded_x)
        
        if task == "predict":
            pooled = encoded_x[:, 0, :] 
            return self.regression_head(pooled)

        if task == "reconstruct":
            return self.reconstruction_head(encoded_x[:, 1:, :])

        raise ValueError(f"Unknown task: {task}")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, base):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(base)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x
