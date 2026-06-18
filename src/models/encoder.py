import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        dropout=0.1,
        head_dropout=0.1,
        pooling="cls",
        patch_size=None,
        multitask=False,
        head_type="linear",
        tcn_head_num_blocks=2,
        tcn_head_kernel_size=3,
    ):
        super(IMU_Intent_Encoder, self).__init__()
        self.patch_size = patch_size
        self.pooling = pooling
        self.multitask = multitask
        self.head_type = head_type

        proj_input_dim = input_features * patch_size if patch_size else input_features
        self.input_projection = nn.Linear(proj_input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.positional_layer = PositionalEncoding(
            d_model=d_model,
            max_len=positional_encoding_max_len,
            base=positional_encoding_base,
        )

        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=dim_feedforward,
            norm_first=True,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)

        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

        recon_out_dim = input_features * patch_size if patch_size else input_features
        self.reconstruction_head = nn.Linear(d_model, recon_out_dim)

        if head_type == "tcn":
            from src.models.tcn import TCNHead
            self.regression_head = TCNHead(d_model, tcn_head_num_blocks, tcn_head_kernel_size,
                                           forecast_horizon, head_dropout)
        else:
            self.regression_head = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(dim_feedforward, forecast_horizon),
            )

        if multitask:
            self.velocity_head = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(dim_feedforward, forecast_horizon),
            )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, C] -> [B, ceil(T/P), P*C], zero-padding if T % P != 0."""
        B, T, C = x.shape
        P = self.patch_size
        n_patches = math.ceil(T / P)
        pad = n_patches * P - T
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        return x.reshape(B, n_patches, P * C)

    def forward(self, x, mask=None, task="reconstruct"):
        if self.patch_size:
            x = self._patchify(x)

        x = self.input_projection(x)

        if mask is not None:
            expanded_mask = mask.unsqueeze(-1)
            x = torch.where(expanded_mask, self.mask_token.to(dtype=x.dtype), x)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.positional_layer(x)
        x = self.pos_drop(x)

        encoded_x = self.transformer_encoder(x)
        encoded_x = self.norm(encoded_x)

        if task == "predict":
            if self.head_type == "tcn":
                seq = encoded_x[:, 1:, :]  # drop CLS token
                angle_pred = self.regression_head(seq)
                if self.multitask:
                    if self.pooling == "cls":
                        pooled = encoded_x[:, 0, :]
                    elif self.pooling == "mean":
                        pooled = seq.mean(dim=1)
                    elif self.pooling.startswith("last_"):
                        k = int(self.pooling.split("_")[-1])
                        pooled = encoded_x[:, -k:, :].mean(dim=1)
                    else:
                        raise ValueError(f"Unknown pooling mode: {self.pooling!r}")
                    return angle_pred, self.velocity_head(pooled)
                return angle_pred
            if self.pooling == "cls":
                pooled = encoded_x[:, 0, :]
            elif self.pooling == "mean":
                pooled = encoded_x[:, 1:, :].mean(dim=1)
            elif self.pooling.startswith("last_"):
                k = int(self.pooling.split("_")[-1])
                pooled = encoded_x[:, -k:, :].mean(dim=1)
            else:
                raise ValueError(f"Unknown pooling mode: {self.pooling!r}")
            angle_pred = self.regression_head(pooled)
            if self.multitask:
                return angle_pred, self.velocity_head(pooled)
            return angle_pred

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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x
