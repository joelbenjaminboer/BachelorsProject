import contextlib
import os
from dataclasses import dataclass
from typing import Any, Optional

import hydra
from loguru import logger
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from src.data.dataloader import build_dataloaders


def _gpu_cfg(cfg: DictConfig):
    return cfg.get("gpu", {})


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _auto_bool(value: Any, auto_default: bool) -> bool:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return auto_default
    return _to_bool(value)


def resolve_device(cfg: DictConfig) -> torch.device:
    requested = str(_gpu_cfg(cfg).get("device", "auto")).strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("gpu.device is set to 'cuda' but CUDA is not available")
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("gpu.device is set to 'mps' but MPS is not available")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported gpu.device value '{requested}'")


def configure_runtime(cfg: DictConfig, device: torch.device):
    gpu_cfg = _gpu_cfg(cfg)
    deterministic = _to_bool(gpu_cfg.get("deterministic", False))
    torch.use_deterministic_algorithms(deterministic, warn_only=True)

    if device.type != "cuda":
        return

    cuda_cfg = gpu_cfg.get("cuda", {})
    tf32 = _to_bool(cuda_cfg.get("tf32", True))
    cudnn_deterministic = deterministic or _to_bool(cuda_cfg.get("cudnn_deterministic", False))
    cudnn_benchmark = _to_bool(cuda_cfg.get("cudnn_benchmark", True)) and not cudnn_deterministic

    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark


def resolve_autocast_kwargs(cfg: DictConfig, device: torch.device):
    if device.type not in {"cuda", "mps"}:
        return None

    autocast_cfg = _gpu_cfg(cfg).get("autocast", {})
    enabled = _auto_bool(autocast_cfg.get("enabled", "auto"), auto_default=(device.type == "cuda"))
    if not enabled:
        return None

    dtype_name = str(autocast_cfg.get("dtype", "float16")).strip().lower()
    if dtype_name in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
    else:
        dtype = torch.float16

    return {
        "device_type": device.type,
        "dtype": dtype,
    }


def autocast_context(autocast_kwargs):
    if autocast_kwargs is None:
        return contextlib.nullcontext()
    return torch.autocast(**autocast_kwargs)


def build_grad_scaler(autocast_kwargs, device: torch.device):
    use_scaler = autocast_kwargs is not None and device.type == "cuda"
    if not use_scaler:
        return None
    return torch.cuda.amp.GradScaler(enabled=True)


def resolve_gradient_settings(cfg: DictConfig):
    gradient_cfg = _gpu_cfg(cfg).get("gradient", {})
    accumulation_steps = max(1, int(gradient_cfg.get("accumulation_steps", 1)))

    raw_clip_grad_norm = gradient_cfg.get("clip_grad_norm", None)
    clip_grad_norm = None if raw_clip_grad_norm is None else float(raw_clip_grad_norm)

    return accumulation_steps, clip_grad_norm


def resolve_dataloader_kwargs(cfg: DictConfig, device: torch.device):
    training_cfg = cfg.get("training", {})

    raw_num_workers = training_cfg.get("num_workers", None)
    if raw_num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)
    else:
        num_workers = max(0, int(raw_num_workers))

    raw_persistent_workers = training_cfg.get("persistent_workers", None)
    if raw_persistent_workers is None:
        persistent_workers = num_workers > 0
    else:
        persistent_workers = _to_bool(raw_persistent_workers)

    if num_workers == 0:
        persistent_workers = False

    raw_pin_memory = training_cfg.get("pin_memory", "auto")
    pin_memory = _auto_bool(raw_pin_memory, auto_default=(device.type == "cuda"))

    prefetch_factor = training_cfg.get("prefetch_factor", None)
    kwargs = {
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = max(2, int(prefetch_factor))

    return kwargs


def use_non_blocking_transfer(cfg: DictConfig, device: torch.device, pin_memory: bool):
    setting = _to_bool(cfg.get("training", {}).get("non_blocking_transfer", True))
    if not setting or device.type == "cpu":
        return False

    if device.type == "cuda":
        return bool(pin_memory)

    return True


def maybe_compile_model(model, cfg: DictConfig):
    compile_cfg = _gpu_cfg(cfg).get("compile", {})
    if not _to_bool(compile_cfg.get("enabled", False)):
        return model

    if not hasattr(torch, "compile"):
        logger.warning("torch.compile is unavailable in this PyTorch build; skipping compile")
        return model

    kwargs = {
        "mode": str(compile_cfg.get("mode", "reduce-overhead")),
        "fullgraph": _to_bool(compile_cfg.get("fullgraph", False)),
        "dynamic": _to_bool(compile_cfg.get("dynamic", False)),
    }

    backend = compile_cfg.get("backend", None)
    if backend is not None and str(backend).strip().lower() not in {"", "null", "none"}:
        kwargs["backend"] = str(backend)

    try:
        compiled_model = torch.compile(model, **kwargs)
        logger.info("Enabled torch.compile with mode={}.", kwargs.get("mode"))
        return compiled_model
    except Exception as exc:  # pragma: no cover - runtime capability guard
        logger.warning("torch.compile failed; falling back to eager mode. reason={}", exc)
        return model


def maybe_wrap_parallel(model, cfg: DictConfig, device: torch.device):
    parallel_cfg = _gpu_cfg(cfg).get("parallel", {})
    strategy = str(parallel_cfg.get("strategy", "auto")).strip().lower()

    if strategy in {"none", "off", "false"}:
        return model

    if device.type != "cuda":
        return model

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        return model

    if strategy in {"auto", "dataparallel"}:
        logger.info("Using DataParallel across {} CUDA devices", gpu_count)
        return torch.nn.DataParallel(model)

    logger.warning("Unknown gpu.parallel.strategy='{}'; using single-device model", strategy)
    return model


def unwrap_model(model):
    while True:
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        elif hasattr(model, "_orig_mod"):  # torch.compile wrapper
            model = model._orig_mod
        else:
            break
    return model


@dataclass
class RunContext:
    device: torch.device
    autocast_kwargs: Optional[dict]
    non_blocking_transfer: bool
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    split_info: dict
    version: str
    accumulation_steps: int
    clip_grad_norm: Optional[float]
    fold_dir: str


def build_run_context(cfg: DictConfig, version: str) -> RunContext:
    device = resolve_device(cfg)
    configure_runtime(cfg, device)
    autocast_kwargs = resolve_autocast_kwargs(cfg, device)
    accumulation_steps, clip_grad_norm = resolve_gradient_settings(cfg)
    loader_kwargs = resolve_dataloader_kwargs(cfg, device)
    non_blocking = use_non_blocking_transfer(
        cfg, device, pin_memory=loader_kwargs.get("pin_memory", False)
    )

    holdout_subjects = cfg.training.get("holdout_subjects", [])
    if not holdout_subjects:
        raise ValueError("training.holdout_subjects must contain at least one subject")
    holdout = holdout_subjects[0]

    fold_dir = os.path.join(
        hydra.utils.to_absolute_path(cfg.dataset.processed_dir), f"fold_{holdout}"
    )

    train_loader, val_loader, test_loader, split_info = build_dataloaders(
        data_dir=fold_dir,
        batch_size=cfg.training.batch_size,
        seed=cfg.training.get("split_seed", 42),
        **loader_kwargs,
    )

    logger.info(f"Using device: {device}")
    logger.info(f"Run version: {version}")
    logger.info(
        f"Holdout subject: {holdout} | "
        f"train: {split_info['train_samples']} | "
        f"val: {split_info['val_samples']} | "
        f"test: {split_info['test_samples']}"
    )

    return RunContext(
        device=device,
        autocast_kwargs=autocast_kwargs,
        non_blocking_transfer=non_blocking,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        split_info=split_info,
        version=version,
        accumulation_steps=accumulation_steps,
        clip_grad_norm=clip_grad_norm,
        fold_dir=fold_dir,
    )


def backward_and_step(
    *,
    loss: torch.Tensor,
    model,
    optimizer,
    scaler,
    accumulation_steps: int,
    clip_grad_norm: Optional[float],
    batch_idx: int,
    total_batches: int,
) -> bool:
    loss_for_backward = loss / accumulation_steps

    if scaler is not None:
        scaler.scale(loss_for_backward).backward()
    else:
        loss_for_backward.backward()

    should_step = ((batch_idx + 1) % accumulation_steps == 0) or (
        batch_idx + 1 == total_batches
    )
    if not should_step:
        return False

    if clip_grad_norm is not None:
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    return True


def load_state_into_model(model, state_dict, *, source: str = "checkpoint") -> None:
    cleaned = dict(state_dict)
    cleaned.pop("positional_layer.pe", None)
    incompatible = unwrap_model(model).load_state_dict(cleaned, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        logger.warning(
            "Loaded {} with missing keys: {} and unexpected keys: {}",
            source,
            incompatible.missing_keys,
            incompatible.unexpected_keys,
        )
    else:
        logger.info(f"Loaded {source} weights into model")
