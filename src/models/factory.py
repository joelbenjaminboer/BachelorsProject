from omegaconf import DictConfig
import torch
import torch.nn as nn

from src.models.encoder import IMU_Intent_Encoder
from src.runtime import RunContext, maybe_compile_model, maybe_wrap_parallel


def build_encoder(cfg: DictConfig, seq_length: int, forecast_horizon: int) -> IMU_Intent_Encoder:
    encoder_cfg = cfg.model.encoder
    positional_encoding_max_len = encoder_cfg.positional_encoding_max_len
    if positional_encoding_max_len is None:
        positional_encoding_max_len = seq_length + int(encoder_cfg.positional_encoding_extra_tokens)

    return IMU_Intent_Encoder(
        input_features=int(encoder_cfg.input_features),
        forecast_horizon=forecast_horizon,
        d_model=int(encoder_cfg.d_model),
        num_heads=int(encoder_cfg.num_heads),
        num_layers=int(encoder_cfg.num_layers),
        dim_feedforward=int(encoder_cfg.dim_feedforward),
        positional_encoding_max_len=int(positional_encoding_max_len),
        positional_encoding_base=float(encoder_cfg.positional_encoding_base),
    )


def build_timesnet(cfg: DictConfig, seq_length: int, forecast_horizon: int) -> nn.Module:
    from src.models.timesnet import TimesNet

    tcfg = cfg.model.timesnet
    return TimesNet(
        input_features=int(tcfg.input_features),
        d_model=int(tcfg.d_model),
        num_blocks=int(tcfg.num_blocks),
        seq_len=seq_length,
        pred_len=forecast_horizon,
        dropout=float(tcfg.dropout),
        top_k=int(tcfg.get("top_k", 3)),
        d_ff=int(tcfg.get("d_ff", 128)),
        num_kernels=int(tcfg.get("num_kernels", 2)),
    )


def build_tcn(cfg: DictConfig, seq_length: int, forecast_horizon: int) -> nn.Module:
    from src.models.tcn import TCN

    tcfg = cfg.model.tcn
    return TCN(
        input_features=int(tcfg.input_features),
        d_model=int(tcfg.d_model),
        num_blocks=int(tcfg.num_blocks),
        kernel_size=int(tcfg.kernel_size),
        forecast_horizon=forecast_horizon,
        dropout=float(tcfg.dropout),
    )


def build_model(cfg: DictConfig, seq_length: int, forecast_horizon: int) -> nn.Module:
    model_type = str(cfg.model.get("model_type", "encoder")).lower()
    if model_type == "encoder":
        return build_encoder(cfg, seq_length, forecast_horizon)
    if model_type == "timesnet":
        return build_timesnet(cfg, seq_length, forecast_horizon)
    if model_type == "tcn":
        return build_tcn(cfg, seq_length, forecast_horizon)
    raise ValueError(f"Unknown model_type: {model_type!r}. Choose from: encoder, timesnet, tcn")


def build_and_prepare_model(cfg: DictConfig, ctx: RunContext) -> nn.Module:
    model = build_model(
        cfg,
        seq_length=cfg.training.context_length,
        forecast_horizon=cfg.training.forecast_horizon,
    ).to(ctx.device)
    model = maybe_compile_model(model, cfg)
    model = maybe_wrap_parallel(model, cfg, ctx.device)
    return model


def build_optimizer(cfg: DictConfig, parameters, device: torch.device | None = None):
    optimizer_cfg = cfg.model.optimizer
    optimizer_name = str(optimizer_cfg.name).lower()

    if optimizer_name == "adamw":
        betas = tuple(float(beta) for beta in optimizer_cfg.betas)
        eps = float(optimizer_cfg.eps)
        optimizer_kwargs = {
            "params": parameters,
            "lr": float(optimizer_cfg.lr),
            "weight_decay": float(optimizer_cfg.weight_decay),
            "betas": betas,
            "eps": eps,
        }

        allow_fused_adamw = bool(
            cfg.get("gpu", {}).get("cuda", {}).get("allow_fused_adamw", True)
        )
        if device is not None and device.type == "cuda" and allow_fused_adamw:
            optimizer_kwargs["fused"] = True

        try:
            return torch.optim.AdamW(**optimizer_kwargs)
        except TypeError:
            optimizer_kwargs.pop("fused", None)
            return torch.optim.AdamW(**optimizer_kwargs)

    raise ValueError(f"Unsupported optimizer '{optimizer_cfg.name}'")


def build_scheduler(cfg: DictConfig, optimizer):
    sched_cfg = cfg.model.get("scheduler", None)
    if sched_cfg is None:
        return None
    name = str(sched_cfg.name).lower()
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(sched_cfg.factor),
            patience=int(sched_cfg.patience),
            min_lr=float(sched_cfg.min_lr),
        )
    raise ValueError(f"Unsupported scheduler '{sched_cfg.name}'")


def build_loss(cfg: DictConfig):
    loss_cfg = cfg.model.loss
    loss_name = str(loss_cfg.name).lower()
    reduction = str(loss_cfg.reduction)

    if loss_name == "mse":
        return nn.MSELoss(reduction=reduction)

    if loss_name == "l1":
        return nn.L1Loss(reduction=reduction)

    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss(reduction=reduction)

    if loss_name == "huber":
        delta = float(loss_cfg.get("delta", 1.0))
        return nn.HuberLoss(reduction=reduction, delta=delta)

    raise ValueError(f"Unsupported loss '{loss_cfg.name}'")
