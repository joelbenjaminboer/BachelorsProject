"""Trivial baselines that define the predictability floor/ceiling.

Knee angle is not part of the model input (input = thigh IMU only), so a
"persistence / last-value" baseline is not available. The meaningful references
are therefore:

  * mean predictor  — always emit the training mean. Its RMSE equals the target
    std; a model that scores here has collapsed to the mean.
  * linear map      — closed-form ridge least squares from the flattened input
    window to the full horizon. If the model cannot beat this, training is broken.

Both are reported in real units (degrees) on the held-out test set.
"""

from loguru import logger
import torch

from src.runtime import RunContext


def _flatten_with_bias(x: torch.Tensor) -> torch.Tensor:
    """[B, T, C] -> [B, T*C + 1] with a trailing bias column."""
    b = x.shape[0]
    flat = x.reshape(b, -1).float()
    ones = torch.ones(b, 1, device=flat.device, dtype=flat.dtype)
    return torch.cat([flat, ones], dim=1)


def evaluate_baselines(
    ctx: RunContext,
    *,
    max_fit_batches: int | None = 300,
    ridge: float = 1.0e-3,
) -> dict[str, float]:
    """Fit a linear baseline on train and evaluate it (and the mean predictor)
    on the test set. Returns RMSE values in degrees. The target arrives from the
    dataloader either z-scored or in raw degrees; we report in degrees by scaling
    with y_std (which is 1.0 when the target is unnormalised, so this is then a
    no-op). The affine mean cancels in the error either way."""
    device = ctx.device
    y_std = ctx.split_info.get("y_std")
    scale = float(y_std) if y_std is not None else 1.0

    # --- Fit linear map via accumulated normal equations (normalised space) ---
    xtx: torch.Tensor | None = None
    xty: torch.Tensor | None = None
    dim = 0
    with torch.inference_mode():
        for i, batch in enumerate(ctx.train_loader):
            if max_fit_batches is not None and i >= max_fit_batches:
                break
            x = _flatten_with_bias(batch["past_imu"].to(device))
            y = batch["future_knee"].to(device).float()
            if xtx is None:
                dim = x.shape[1]
                xtx = torch.zeros(dim, dim, device=device)
                xty = torch.zeros(dim, y.shape[1], device=device)
            xtx += x.t() @ x
            xty += x.t() @ y

    if xtx is None:
        logger.warning("Baseline fit skipped: empty train loader")
        return {}

    xtx = xtx + ridge * torch.eye(dim, device=device)
    weight = torch.linalg.solve(xtx, xty)  # [D, H]

    # --- Evaluate on test set ---
    lin_sq = 0.0
    mean_sq = 0.0  # mean predictor = 0 in normalised space, so error = -target
    count = 0
    with torch.inference_mode():
        for batch in ctx.test_loader:
            x = _flatten_with_bias(batch["past_imu"].to(device))
            y = batch["future_knee"].to(device).float()
            pred = x @ weight
            lin_sq += torch.sum((pred - y) ** 2).item()
            mean_sq += torch.sum(y**2).item()
            count += y.numel()

    if count == 0:
        return {}

    mean_rmse = scale * (mean_sq / count) ** 0.5
    linear_rmse = scale * (lin_sq / count) ** 0.5
    logger.info(
        "Baselines (test, real units) — mean-predictor RMSE: {:.4f}° | "
        "linear-map RMSE: {:.4f}°  (the model must beat the linear baseline)",
        mean_rmse,
        linear_rmse,
    )
    return {"mean_baseline_rmse": mean_rmse, "linear_baseline_rmse": linear_rmse}
