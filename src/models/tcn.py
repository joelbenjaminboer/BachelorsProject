import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalConv1d(nn.Module):
    """Conv1d with left-only padding to prevent future information leakage."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) — pad only on the left
        return self.conv(F.pad(x, (self.padding, 0)))


class _TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.3):
        super().__init__()
        self.conv = _CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv(x)
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.act(out)
        out = self.dropout(out)
        return self.act(out + self.residual(x))


class TCN(nn.Module):
    """Temporal Convolutional Network for multi-step knee-angle forecasting from 6-channel IMU input.

    Rewritten from TensorFlow/Keras to pure PyTorch.
    Causal dilated convolutions with exponential dilation (2^i per block).
    Multi-step output via a linear head over the last timestep.
    """

    def __init__(
        self,
        input_features: int,
        d_model: int,
        num_blocks: int,
        kernel_size: int,
        forecast_horizon: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_features, d_model, kernel_size=1)
        self.blocks = nn.ModuleList([
            _TCNBlock(d_model, d_model, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(num_blocks)
        ])
        self.head = nn.Linear(d_model, forecast_horizon)

    def forward(self, x: torch.Tensor, task: str = "predict", **kwargs) -> torch.Tensor:
        if task == "reconstruct":
            raise NotImplementedError("TCN does not support MAE pretraining")

        # x: (B, T, input_features)
        x = self.input_proj(x.transpose(1, 2))  # (B, d_model, T)
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, :, -1])  # (B, forecast_horizon)
