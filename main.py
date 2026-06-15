from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig
from src.data.download import run_download
from src.data.preprocessing import run_preprocessing
from src.logging_utils import setup_logging
from src.models.factory import build_and_prepare_model
from src.runtime import build_run_context, load_state_into_model
from src.training.eval import run_eval
from src.training.hparam_search import run_hparam_search
from src.training.pretrain import run_pretrain
from src.training.train import run_train
import torch

MODEL_STAGES = ("pretrain", "train", "eval", "hparam_search")

# Absolute path to this script, captured at import (before Hydra may change cwd)
# so child fold subprocesses can re-invoke it from any working directory.
_THIS_SCRIPT = os.path.abspath(__file__)


def _write_fold_result(cfg: DictConfig, eval_results) -> None:
    """When launched as a child fold (FOLD_RESULT_FILE set in env), persist the
    eval RMSE so the parent all-folds orchestrator can aggregate LOSO mean/std."""
    result_file = os.environ.get("FOLD_RESULT_FILE")
    if not result_file or not eval_results:
        return
    holdouts = cfg.training.get("holdout_subjects", [])
    holdout = holdouts[0] if holdouts else "unknown"
    try:
        rmse = float(eval_results["overall"]["rmse"])
    except (KeyError, TypeError, ValueError):
        return
    Path(result_file).write_text(json.dumps({"holdout": holdout, "rmse": rmse}))


def _run_all_folds(cfg: DictConfig) -> None:
    """Run every LOSO fold as an independent `python main.py ...single fold...`
    subprocess.

    Each fold is the exact path used in a normal single-fold run, so it gets its
    own Hydra init, a clean CUDA context, and full release of its ~15 GB of data
    on process exit. Running folds in-process OOM-kills after the first holdout;
    running them in a fork-based pool deadlocks on the first batch because
    CUDA + DataLoader worker forking is unsupported. Subprocesses avoid both.
    """
    processed_dir = hydra.utils.to_absolute_path(cfg.dataset.processed_dir)
    fold_dirs = sorted(glob.glob(os.path.join(processed_dir, "fold_*")))
    if not fold_dirs:
        logger.error("No fold_* directories found in {}. Run preprocessing first.", processed_dir)
        return

    holdouts = [os.path.basename(d).replace("fold_", "") for d in fold_dirs]
    max_workers = int(cfg.get("run", {}).get("all_folds_workers", 1))

    if max_workers > 1 and str(cfg.gpu.get("device", "auto")).lower() != "cpu":
        logger.warning(
            "all_folds_workers={} runs folds concurrently and causes GPU/RAM contention. "
            "Use one worker per physical GPU.",
            max_workers,
        )

    original_cwd = hydra.utils.get_original_cwd()
    results_dir = Path(HydraConfig.get().runtime.output_dir) / "fold_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Forward this invocation's overrides to each fold, minus the all-folds
    # controls and any holdout selection (each fold sets its own holdout).
    passthrough = [
        o
        for o in HydraConfig.get().overrides.task
        if not o.startswith(("run.all_folds", "training.holdout_subjects"))
    ]

    def _run_fold(holdout: str):
        result_file = results_dir / f"{holdout}.json"
        cmd = [
            sys.executable,
            _THIS_SCRIPT,
            *passthrough,
            "run.all_folds=false",
            f"training.holdout_subjects=[{holdout}]",
        ]
        env = {**os.environ, "FOLD_RESULT_FILE": str(result_file)}
        logger.info("Launching fold {} ...", holdout)
        proc = subprocess.run(cmd, cwd=original_cwd, env=env)
        return holdout, proc.returncode, result_file

    all_rmse: dict[str, float] = {}

    def _collect(holdout: str, returncode: int, result_file: Path) -> None:
        if returncode != 0:
            logger.error("Fold {} exited with code {}", holdout, returncode)
            return
        if result_file.exists():
            try:
                all_rmse[holdout] = float(json.loads(result_file.read_text())["rmse"])
            except Exception:
                logger.exception("Could not read result for fold {}", holdout)

    if max_workers <= 1:
        for holdout in holdouts:
            _collect(*_run_fold(holdout))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for fut in as_completed([pool.submit(_run_fold, h) for h in holdouts]):
                _collect(*fut.result())

    if all_rmse:
        rmse_vals = list(all_rmse.values())
        per_fold = "  ".join(f"{h}={v:.4f}°" for h, v in sorted(all_rmse.items()))
        logger.info("LOSO per-fold RMSE: {}", per_fold)
        logger.info(
            "LOSO RMSE over {} folds: mean={:.4f}°  std={:.4f}°",
            len(rmse_vals),
            statistics.mean(rmse_vals),
            statistics.stdev(rmse_vals) if len(rmse_vals) > 1 else 0.0,
        )


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    log_path = setup_logging(cfg)
    if log_path is not None:
        logger.info("Logging to {}", log_path)

    run_cfg = cfg.get("run", {})
    version = cfg.get("version", "0.1.0")

    if run_cfg.get("download", False):
        run_download(cfg)

    if run_cfg.get("preprocess", False):
        run_preprocessing(cfg, version=version)

    if not any(run_cfg.get(stage, False) for stage in MODEL_STAGES):
        return

    # --- LOSO all-folds mode ---
    if run_cfg.get("all_folds", False):
        _run_all_folds(cfg)
        return

    # --- Single-fold mode (default; also the per-fold child of all-folds) ---
    ctx = build_run_context(cfg, version=version)

    if run_cfg.get("hparam_search", False):
        run_hparam_search(cfg, ctx)
        return

    model = build_and_prepare_model(cfg, ctx)

    pretrain_supported = bool(cfg.model.get("supports_pretrain", True))
    will_pretrain = run_cfg.get("pretrain", False) and pretrain_supported

    if run_cfg.get("pretrain", False) and not pretrain_supported:
        logger.info(
            "Skipping pretraining: model_type='{}' does not support MAE pretraining",
            cfg.model.get("model_type", "encoder"),
        )

    pretrained_ckpt = None
    if will_pretrain:
        pretrained_ckpt = run_pretrain(cfg, model=model, ctx=ctx)

    best_checkpoint_path = None
    if run_cfg.get("train", False):
        # Fine-tune from the *best* pretrained checkpoint (lowest val loss), not the
        # last-epoch in-memory weights left behind by the pretraining loop.
        if pretrained_ckpt is not None and Path(pretrained_ckpt).exists():
            state = torch.load(pretrained_ckpt, map_location=ctx.device)
            load_state_into_model(model, state, source=str(pretrained_ckpt))
        elif not will_pretrain and run_cfg.get("load_checkpoint", False):
            checkpoint_path = Path(run_cfg.get("checkpoint_path", ""))
            if checkpoint_path.exists():
                state = torch.load(checkpoint_path, map_location=ctx.device)
                load_state_into_model(model, state, source=str(checkpoint_path))
            else:
                logger.warning(
                    f"Checkpoint not found: {checkpoint_path}. Starting training from scratch."
                )

        best_checkpoint_path = run_train(cfg, model=model, ctx=ctx)

    if run_cfg.get("eval", False):
        eval_results = run_eval(cfg, model=model, ctx=ctx, checkpoint_path=best_checkpoint_path)
        _write_fold_result(cfg, eval_results)


if __name__ == "__main__":
    main()
