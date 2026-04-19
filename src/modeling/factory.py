import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.modeling.encoder import IMU_Intent_Encoder


def build_encoder(cfg: DictConfig, seq_length: int, forecast_horizon: int) -> IMU_Intent_Encoder:
    encoder_cfg = cfg.model.encoder
    positional_encoding_max_len = encoder_cfg.positional_encoding_max_len
    if positional_encoding_max_len is None:
        positional_encoding_max_len = seq_length + int(encoder_cfg.positional_encoding_extra_tokens)

    return IMU_Intent_Encoder(
        input_features=int(encoder_cfg.input_features),
        seq_length=seq_length,
        forecast_horizon=forecast_horizon,
        d_model=int(encoder_cfg.d_model),
        num_heads=int(encoder_cfg.num_heads),
        num_layers=int(encoder_cfg.num_layers),
        dim_feedforward=int(encoder_cfg.dim_feedforward),
        positional_encoding_max_len=int(positional_encoding_max_len),
        positional_encoding_base=float(encoder_cfg.positional_encoding_base),
    )


def build_optimizer(cfg: DictConfig, parameters):
    optimizer_cfg = cfg.model.optimizer
    optimizer_name = str(optimizer_cfg.name).lower()

    if optimizer_name == "adamw":
        betas = tuple(float(beta) for beta in optimizer_cfg.betas)
        eps = float(optimizer_cfg.eps)
        return torch.optim.AdamW(
            parameters,
            lr=float(optimizer_cfg.lr),
            weight_decay=float(optimizer_cfg.weight_decay),
            betas=betas,
            eps=eps,
        )

    raise ValueError(f"Unsupported optimizer '{optimizer_cfg.name}'")


def build_loss(cfg: DictConfig):
    loss_cfg = cfg.model.loss
    loss_name = str(loss_cfg.name).lower()
    reduction = str(loss_cfg.reduction)

    if loss_name == "mse":
        return nn.MSELoss(reduction=reduction)

    if loss_name == "l1":
        return nn.L1Loss(reduction=reduction)

    raise ValueError(f"Unsupported loss '{loss_cfg.name}'")
