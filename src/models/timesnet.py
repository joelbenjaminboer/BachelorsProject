import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class _TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → conv expects (B, C, T)
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)


class _DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = _TokenEmbedding(c_in, d_model)
        self.position_embedding = _PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


class _InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 2):
        super().__init__()
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            for i in range(num_kernels)
        ])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([k(x) for k in self.kernels], dim=-1).mean(-1)


class _TimesBlock(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, d_model: int, d_ff: int, num_kernels: int, top_k: int):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k
        self.conv = nn.Sequential(
            _InceptionBlock(d_model, d_ff, num_kernels),
            nn.GELU(),
            _InceptionBlock(d_ff, d_model, num_kernels),
        )

    def _fft_period(self, x: torch.Tensor):
        xf = torch.fft.rfft(x.float(), dim=1)
        amplitude = abs(xf).mean(-1).to(x.dtype)
        amplitude[:, 0] = 0  # remove DC component
        top_indices = torch.topk(amplitude, self.top_k, dim=1).indices
        periods = x.shape[1] // top_indices.clamp(min=1)
        return periods, amplitude.gather(1, top_indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.shape
        total_len = self.seq_len + self.pred_len
        period_list, period_weight = self._fft_period(x)

        res = []
        for i in range(self.top_k):
            period = max(1, int(torch.max(period_list[:, i]).item()))

            if total_len % period != 0:
                pad_len = (total_len // period + 1) * period - total_len
                out = torch.cat([x, torch.zeros(B, pad_len, N, device=x.device)], dim=1)
            else:
                out = x

            out = out.reshape(B, -1, period, N).permute(0, 3, 1, 2)  # (B, N, num_periods, period)
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)  # (B, T', N)
            res.append(out[:, :total_len, :])

        res = torch.stack(res, dim=-1)  # (B, total_len, N, top_k)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).expand(-1, T, N, -1)
        return torch.sum(res * period_weight, dim=-1) + x


class TimesNet(nn.Module):
    """TimesNet for multi-step knee-angle forecasting from 6-channel IMU input.

    Adapted from https://github.com/thuml/TimesNet.
    Internal normalization removed — the dataloader handles Z-score normalisation.
    """

    def __init__(
        self,
        input_features: int,
        d_model: int,
        num_blocks: int,
        seq_len: int,
        pred_len: int,
        dropout: float = 0.2,
        top_k: int = 3,
        d_ff: int = 128,
        num_kernels: int = 2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.embedding = _DataEmbedding(input_features, d_model, dropout)
        self.predict_linear = nn.Linear(seq_len, seq_len + pred_len)
        self.blocks = nn.ModuleList([
            _TimesBlock(seq_len, pred_len, d_model, d_ff, num_kernels, top_k)
            for _ in range(num_blocks)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, task: str = "predict", **kwargs) -> torch.Tensor:
        if task == "reconstruct":
            raise NotImplementedError("TimesNet does not support MAE pretraining")

        # x: (B, seq_len, input_features)
        enc = self.embedding(x)  # (B, seq_len, d_model)
        enc = self.predict_linear(enc.permute(0, 2, 1)).permute(0, 2, 1)  # (B, seq_len+pred_len, d_model)

        for block in self.blocks:
            enc = self.layer_norm(block(enc))

        out = self.head(enc[:, -self.pred_len :, :].contiguous())  # (B, pred_len, 1)
        return out.squeeze(-1)  # (B, pred_len)
