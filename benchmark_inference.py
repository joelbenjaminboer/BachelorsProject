"""Measure inference latency/throughput of the configured model.

Usage:
    python benchmark_inference.py
    python benchmark_inference.py model=tcn gpu.device=mps
    python benchmark_inference.py model=timesnet benchmark.batch_sizes=[1,32,128]
    python benchmark_inference.py benchmark.checkpoint_path=checkpoints/allfolds_100hz/best_model_epoch_3.pth
"""

import json
from pathlib import Path
import time

import hydra
from loguru import logger
from omegaconf import DictConfig
import torch

from src.logging_utils import setup_logging
from src.models.factory import build_model
from src.runtime import (
    autocast_context,
    configure_runtime,
    load_state_into_model,
    maybe_compile_model,
    maybe_wrap_parallel,
    resolve_autocast_kwargs,
    resolve_device,
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _benchmark_batch_size(
    model, device, autocast_kwargs, seq_length, input_features, batch_size, num_warmup, num_iters
):
    x = torch.randn(batch_size, seq_length, input_features, device=device)

    with torch.inference_mode():
        for _ in range(num_warmup):
            with autocast_context(autocast_kwargs):
                model(x, task="predict")
        _synchronize(device)

        latencies_ms = []
        for _ in range(num_iters):
            start = time.perf_counter()
            with autocast_context(autocast_kwargs):
                model(x, task="predict")
            _synchronize(device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    latencies_ms.sort()
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    p50_ms = latencies_ms[len(latencies_ms) // 2]
    p95_ms = latencies_ms[int(len(latencies_ms) * 0.95)]
    throughput = batch_size / (mean_ms / 1000.0)

    return {
        "batch_size": batch_size,
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "min_ms": latencies_ms[0],
        "max_ms": latencies_ms[-1],
        "throughput_samples_per_sec": throughput,
    }


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging(cfg)

    device = resolve_device(cfg)
    configure_runtime(cfg, device)
    autocast_kwargs = resolve_autocast_kwargs(cfg, device)

    model_type = str(cfg.model.get("model_type", "encoder")).lower()
    arch_key = model_type if model_type in ("encoder", "timesnet", "tcn") else "encoder"
    seq_length = int(cfg.training.context_length)
    forecast_horizon = int(cfg.training.forecast_horizon)
    input_features = int(cfg.model[arch_key].input_features)

    model = build_model(cfg, seq_length=seq_length, forecast_horizon=forecast_horizon).to(device)
    model = maybe_compile_model(model, cfg)
    model = maybe_wrap_parallel(model, cfg, device)
    model.eval()

    bench_cfg = cfg.get("benchmark", {})
    checkpoint_path = bench_cfg.get("checkpoint_path", None)
    if not checkpoint_path and cfg.run.get("load_checkpoint", False):
        checkpoint_path = cfg.run.get("checkpoint_path", None)

    if checkpoint_path:
        checkpoint_path = hydra.utils.to_absolute_path(checkpoint_path)
        if Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location=device)
            load_state_into_model(model, state, source=checkpoint_path)
        else:
            logger.warning(
                "Checkpoint not found at {}; benchmarking randomly initialized weights",
                checkpoint_path,
            )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model: {} | params: {:,} | device: {} | seq_len: {} | horizon: {} | autocast: {}",
        model_type,
        num_params,
        device,
        seq_length,
        forecast_horizon,
        autocast_kwargs is not None,
    )

    batch_sizes = list(bench_cfg.get("batch_sizes", [1, 8, 32, 64]))
    num_warmup = int(bench_cfg.get("num_warmup", 10))
    num_iters = int(bench_cfg.get("num_iters", 50))

    results = []
    for batch_size in batch_sizes:
        result = _benchmark_batch_size(
            model,
            device,
            autocast_kwargs,
            seq_length,
            input_features,
            int(batch_size),
            num_warmup,
            num_iters,
        )
        results.append(result)
        logger.info(
            "batch_size={:>4d} | mean={:7.3f} ms | p50={:7.3f} ms | p95={:7.3f} ms | throughput={:9.1f} samples/s",
            result["batch_size"],
            result["mean_ms"],
            result["p50_ms"],
            result["p95_ms"],
            result["throughput_samples_per_sec"],
        )

    if bool(bench_cfg.get("output_json", True)):
        out_path = Path.cwd() / "inference_benchmark.json"
        out_path.write_text(
            json.dumps(
                {
                    "model_type": model_type,
                    "device": str(device),
                    "num_params": num_params,
                    "seq_length": seq_length,
                    "forecast_horizon": forecast_horizon,
                    "autocast": autocast_kwargs is not None,
                    "checkpoint_path": checkpoint_path,
                    "results": results,
                },
                indent=2,
            )
        )
        logger.info("Saved results to {}", out_path)


if __name__ == "__main__":
    main()
