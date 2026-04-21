import json
import math
from pathlib import Path
from typing import Any, Sequence

import hydra
from hydra.core.hydra_config import HydraConfig
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import torch

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency fallback
    sns = None


def _plot_cfg(cfg: DictConfig):
    return cfg.get("plotting", {})


def stage_enabled(cfg: DictConfig, stage: str) -> bool:
    plot_cfg = _plot_cfg(cfg)
    if not bool(plot_cfg.get("enabled", False)):
        return False

    stages_cfg = plot_cfg.get("stages", {})
    return bool(stages_cfg.get(stage, False))


def should_save_intermediate_epoch(cfg: DictConfig, epoch_idx: int) -> bool:
    plot_cfg = _plot_cfg(cfg)
    cadence_cfg = plot_cfg.get("cadence", {})

    if not bool(cadence_cfg.get("save_intermediate", True)):
        return False

    every_n_epochs = max(1, int(cadence_cfg.get("every_n_epochs", 1)))
    return (epoch_idx + 1) % every_n_epochs == 0


def _resolve_run_output_dir() -> Path:
    if HydraConfig.initialized():
        return Path(HydraConfig.get().runtime.output_dir)

    try:
        return Path(hydra.utils.get_original_cwd())
    except ValueError:
        return Path.cwd()


def get_stage_output_dir(cfg: DictConfig, stage: str) -> Path:
    save_subdir = str(_plot_cfg(cfg).get("save_subdir", "plots"))
    stage_dir = _resolve_run_output_dir() / save_subdir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def _setup_plot_style(cfg: DictConfig):
    plot_cfg = _plot_cfg(cfg)
    style = str(plot_cfg.get("style", "whitegrid"))
    palette = str(plot_cfg.get("palette", "deep"))
    context = str(plot_cfg.get("context", "talk"))

    if sns is not None:
        sns.set_theme(style=style, palette=palette, context=context)
        return

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")


def _save_figure(fig, base_path: Path, cfg: DictConfig) -> list[Path]:
    plot_cfg = _plot_cfg(cfg)
    formats = plot_cfg.get("save_formats", ["png"])
    if isinstance(formats, str):
        formats = [formats]

    dpi = int(plot_cfg.get("dpi", 180))
    saved_paths: list[Path] = []
    for fmt in formats:
        safe_fmt = str(fmt).lstrip(".")
        save_path = base_path.with_suffix(f".{safe_fmt}")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        saved_paths.append(save_path)

    plt.close(fig)
    return saved_paths


def _to_serializable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _save_metrics_json(cfg: DictConfig, stage_dir: Path, filename: str, payload: dict[str, Any]):
    if not bool(_plot_cfg(cfg).get("save_metrics", True)):
        return None

    output_path = stage_dir / filename
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(_to_serializable(payload), fp, indent=2)
    return output_path


def save_train_artifacts(
    cfg: DictConfig,
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    best_epoch: int | None,
    best_checkpoint_path: str | None,
    tag: str,
):
    if not stage_enabled(cfg, "train"):
        return

    _setup_plot_style(cfg)
    stage_dir = get_stage_output_dir(cfg, "train")

    payload = {
        "tag": tag,
        "epochs": len(train_losses),
        "best_epoch": best_epoch,
        "best_checkpoint_path": best_checkpoint_path,
        "train_loss": list(train_losses),
        "val_loss": list(val_losses),
    }
    metrics_path = _save_metrics_json(cfg, stage_dir, f"metrics_{tag}.json", payload)
    if metrics_path is not None:
        logger.info("Saved train metrics: {}", metrics_path)

    train_cfg = _plot_cfg(cfg).get("train", {})
    if not bool(train_cfg.get("plot_loss", True)):
        return

    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses, label="Val Loss", linewidth=2)

    if best_epoch is not None and 1 <= best_epoch <= len(train_losses):
        ax.axvline(best_epoch, linestyle="--", linewidth=1.5, color="black", label="Best Epoch")

    ax.set_title("Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    saved_paths = _save_figure(fig, stage_dir / f"loss_curve_{tag}", cfg)
    logger.info("Saved train plots: {}", ", ".join(str(path) for path in saved_paths))


def _plot_channel_metric(
    cfg: DictConfig,
    stage_dir: Path,
    channel_names: Sequence[str],
    train_history: Sequence[Sequence[float]],
    val_history: Sequence[Sequence[float]],
    metric_name: str,
    tag: str,
):
    if len(train_history) == 0 or len(val_history) == 0:
        return

    train_np = np.asarray(train_history, dtype=float)
    val_np = np.asarray(val_history, dtype=float)
    if train_np.ndim != 2 or val_np.ndim != 2:
        return

    epochs = np.arange(1, train_np.shape[0] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    for idx, channel_name in enumerate(channel_names):
        axes[0].plot(epochs, train_np[:, idx], linewidth=1.8, label=channel_name)
        axes[1].plot(epochs, val_np[:, idx], linewidth=1.8, label=channel_name)

    axes[0].set_title(f"Train Channel {metric_name}")
    axes[1].set_title(f"Val Channel {metric_name}")
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[0].set_ylabel(metric_name)
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    saved_paths = _save_figure(fig, stage_dir / f"channel_{metric_name.lower()}_{tag}", cfg)
    logger.info("Saved pretrain {} plots: {}", metric_name, ", ".join(map(str, saved_paths)))


def save_pretrain_artifacts(
    cfg: DictConfig,
    channel_names: Sequence[str],
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    train_channel_mse: Sequence[Sequence[float]],
    val_channel_mse: Sequence[Sequence[float]],
    train_channel_rmse: Sequence[Sequence[float]],
    val_channel_rmse: Sequence[Sequence[float]],
    best_epoch: int | None,
    best_checkpoint_path: str | None,
    tag: str,
):
    if not stage_enabled(cfg, "pretrain"):
        return

    _setup_plot_style(cfg)
    stage_dir = get_stage_output_dir(cfg, "pretrain")
    payload = {
        "tag": tag,
        "epochs": len(train_losses),
        "channel_names": list(channel_names),
        "best_epoch": best_epoch,
        "best_checkpoint_path": best_checkpoint_path,
        "train_loss": list(train_losses),
        "val_loss": list(val_losses),
        "train_channel_mse": list(train_channel_mse),
        "val_channel_mse": list(val_channel_mse),
        "train_channel_rmse": list(train_channel_rmse),
        "val_channel_rmse": list(val_channel_rmse),
    }
    metrics_path = _save_metrics_json(cfg, stage_dir, f"metrics_{tag}.json", payload)
    if metrics_path is not None:
        logger.info("Saved pretrain metrics: {}", metrics_path)

    pretrain_cfg = _plot_cfg(cfg).get("pretrain", {})

    if bool(pretrain_cfg.get("plot_loss", True)):
        epochs = np.arange(1, len(train_losses) + 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, train_losses, label="Train Loss", linewidth=2)
        ax.plot(epochs, val_losses, label="Val Loss", linewidth=2)
        if best_epoch is not None and 1 <= best_epoch <= len(train_losses):
            ax.axvline(best_epoch, linestyle="--", linewidth=1.5, color="black", label="Best Epoch")
        ax.set_title("Pretraining Reconstruction Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()

        saved_paths = _save_figure(fig, stage_dir / f"loss_curve_{tag}", cfg)
        logger.info("Saved pretrain loss plots: {}", ", ".join(map(str, saved_paths)))

    if bool(pretrain_cfg.get("plot_channel_metrics", True)):
        _plot_channel_metric(
            cfg,
            stage_dir,
            channel_names,
            train_channel_mse,
            val_channel_mse,
            "MSE",
            tag,
        )
        _plot_channel_metric(
            cfg,
            stage_dir,
            channel_names,
            train_channel_rmse,
            val_channel_rmse,
            "RMSE",
            tag,
        )


def save_eval_artifacts(
    cfg: DictConfig,
    overall_metrics: dict[str, float],
    per_step_metrics: dict[str, Sequence[float]],
    residuals: Sequence[float],
    prediction_examples: Sequence[Sequence[float]],
    target_examples: Sequence[Sequence[float]],
    example_subject_ids: Sequence[str],
    subject_metrics: dict[str, dict[str, float]],
    checkpoint_path: str,
    tag: str = "final",
):
    if not stage_enabled(cfg, "eval"):
        return

    _setup_plot_style(cfg)
    stage_dir = get_stage_output_dir(cfg, "eval")

    payload = {
        "tag": tag,
        "checkpoint_path": checkpoint_path,
        "overall": overall_metrics,
        "per_step": per_step_metrics,
        "residual_summary": {
            "count": len(residuals),
            "mean": float(np.mean(residuals)) if len(residuals) > 0 else float("nan"),
            "std": float(np.std(residuals)) if len(residuals) > 0 else float("nan"),
        },
        "subject_metrics": subject_metrics,
        "example_subject_ids": list(example_subject_ids),
    }
    metrics_path = _save_metrics_json(cfg, stage_dir, f"metrics_{tag}.json", payload)
    if metrics_path is not None:
        logger.info("Saved eval metrics: {}", metrics_path)

    eval_cfg = _plot_cfg(cfg).get("eval", {})

    if bool(eval_cfg.get("plot_overall_summary", True)):
        names = ["MSE", "MAE", "RMSE"]
        values = [
            float(overall_metrics.get("mse", float("nan"))),
            float(overall_metrics.get("mae", float("nan"))),
            float(overall_metrics.get("rmse", float("nan"))),
        ]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(names, values)
        ax.set_title("Evaluation Overall Metrics")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3, axis="y")
        saved_paths = _save_figure(fig, stage_dir / f"overall_metrics_{tag}", cfg)
        logger.info("Saved eval overall plots: {}", ", ".join(map(str, saved_paths)))

    if bool(eval_cfg.get("plot_per_step_metrics", True)) and len(per_step_metrics.get("rmse", [])) > 0:
        steps = np.arange(1, len(per_step_metrics["rmse"]) + 1)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(steps, per_step_metrics.get("mse", []), linewidth=1.8, label="MSE")
        ax.plot(steps, per_step_metrics.get("mae", []), linewidth=1.8, label="MAE")
        ax.plot(steps, per_step_metrics.get("rmse", []), linewidth=1.8, label="RMSE")
        ax.set_title("Per-step Forecast Metrics")
        ax.set_xlabel("Forecast Step")
        ax.set_ylabel("Error")
        ax.grid(alpha=0.3)
        ax.legend()
        saved_paths = _save_figure(fig, stage_dir / f"per_step_metrics_{tag}", cfg)
        logger.info("Saved eval per-step plots: {}", ", ".join(map(str, saved_paths)))

    if bool(eval_cfg.get("plot_residual_histogram", True)) and len(residuals) > 0:
        bins = max(5, int(eval_cfg.get("residual_bins", 60)))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(residuals, bins=bins, alpha=0.8)
        ax.set_title("Residual Distribution")
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)
        saved_paths = _save_figure(fig, stage_dir / f"residual_histogram_{tag}", cfg)
        logger.info("Saved eval residual plots: {}", ", ".join(map(str, saved_paths)))

    if bool(eval_cfg.get("plot_prediction_examples", True)):
        pred_np = np.asarray(prediction_examples, dtype=float)
        tgt_np = np.asarray(target_examples, dtype=float)
        if pred_np.ndim == 2 and tgt_np.ndim == 2 and pred_np.shape == tgt_np.shape:
            num_examples = pred_np.shape[0]
            if num_examples > 0:
                rows = min(3, num_examples)
                cols = math.ceil(num_examples / rows)
                fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
                horizon = np.arange(1, pred_np.shape[1] + 1)

                for idx in range(rows * cols):
                    axis = axes[idx // cols][idx % cols]
                    if idx >= num_examples:
                        axis.axis("off")
                        continue

                    axis.plot(horizon, tgt_np[idx], label="Target", linewidth=2)
                    axis.plot(horizon, pred_np[idx], label="Prediction", linewidth=2)
                    subject_suffix = ""
                    if idx < len(example_subject_ids):
                        subject_suffix = f" | {example_subject_ids[idx]}"
                    axis.set_title(f"Example {idx + 1}{subject_suffix}")
                    axis.set_xlabel("Forecast Step")
                    axis.set_ylabel("Knee Angle")
                    axis.grid(alpha=0.3)

                handles, labels = axes[0][0].get_legend_handles_labels()
                if handles:
                    fig.legend(handles, labels, loc="upper center", ncol=2)
                fig.tight_layout(rect=(0, 0, 1, 0.96))
                saved_paths = _save_figure(fig, stage_dir / f"prediction_examples_{tag}", cfg)
                logger.info("Saved eval example plots: {}", ", ".join(map(str, saved_paths)))

    if bool(eval_cfg.get("plot_subject_bars", True)) and len(subject_metrics) > 0:
        max_subjects = max(1, int(eval_cfg.get("max_subjects_in_plot", 20)))
        ranked_subjects = sorted(
            subject_metrics.items(), key=lambda item: item[1].get("rmse", 0.0), reverse=True
        )[:max_subjects]

        labels = [subject for subject, _ in ranked_subjects]
        mae_values = [metrics.get("mae", float("nan")) for _, metrics in ranked_subjects]
        rmse_values = [metrics.get("rmse", float("nan")) for _, metrics in ranked_subjects]

        x = np.arange(len(labels))
        width = 0.4

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 5))
        ax.bar(x - width / 2, mae_values, width=width, label="MAE")
        ax.bar(x + width / 2, rmse_values, width=width, label="RMSE")
        ax.set_title("Per-subject Error Metrics")
        ax.set_xlabel("Subject")
        ax.set_ylabel("Error")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(alpha=0.3, axis="y")
        ax.legend()

        saved_paths = _save_figure(fig, stage_dir / f"subject_metrics_{tag}", cfg)
        logger.info("Saved eval subject plots: {}", ", ".join(map(str, saved_paths)))
