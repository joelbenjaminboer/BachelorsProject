from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import os
from pathlib import Path
import statistics

import hydra
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
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


def _run_single_fold(cfg, holdout: str, version: str, log_dir: str | None = None):
    """Run the full pipeline for one LOSO holdout subject. Top-level so it's picklable."""
    # In a worker subprocess Hydra's config isn't active, so loguru would only
    # reach stderr. Give each fold its own file sink under the run directory.
    if log_dir is not None and not HydraConfig.initialized():
        setup_logging(cfg, output_dir=Path(log_dir) / f"fold_{holdout}")

    run_cfg = cfg.get("run", {})
    fold_cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create({"training": {"holdout_subjects": [holdout]}}),
    )

    ctx = build_run_context(fold_cfg, version=version)
    model = build_and_prepare_model(fold_cfg, ctx)

    pretrain_supported = bool(fold_cfg.model.get("supports_pretrain", True))
    will_pretrain = run_cfg.get("pretrain", False) and pretrain_supported

    if will_pretrain:
        run_pretrain(fold_cfg, model=model, ctx=ctx)

    best_checkpoint_path = None
    if run_cfg.get("train", False):
        if not will_pretrain and run_cfg.get("load_checkpoint", False):
            checkpoint_path = Path(run_cfg.get("checkpoint_path", ""))
            if checkpoint_path.exists():
                state = torch.load(checkpoint_path, map_location=ctx.device)
                load_state_into_model(model, state, source=str(checkpoint_path))
        best_checkpoint_path = run_train(fold_cfg, model=model, ctx=ctx)

    results = None
    if run_cfg.get("eval", False):
        results = run_eval(fold_cfg, model=model, ctx=ctx, checkpoint_path=best_checkpoint_path)

    return holdout, results


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
        processed_dir = hydra.utils.to_absolute_path(cfg.dataset.processed_dir)
        fold_dirs = sorted(glob.glob(os.path.join(processed_dir, "fold_*")))
        if not fold_dirs:
            logger.error(
                "No fold_* directories found in {}. Run preprocessing first.", processed_dir
            )
            return

        holdouts = [os.path.basename(d).replace("fold_", "") for d in fold_dirs]
        max_workers = int(run_cfg.get("all_folds_workers", 1))

        if max_workers > 1 and str(cfg.gpu.get("device", "auto")).lower() not in {"cpu"}:
            logger.warning(
                "all_folds_workers={} with GPU training causes VRAM contention. "
                "Use gpu.device=cpu or one worker per physical GPU.",
                max_workers,
            )

        all_rmse: dict[str, float] = {}
        fold_log_dir = HydraConfig.get().runtime.output_dir
        # Run every fold in its own worker that is retired after a single task
        # (max_tasks_per_child=1). When the worker exits, the fold's ~15 GB of
        # data and its CUDA context are returned to the OS before the next fold
        # forks. Running folds in-process accumulates memory across folds and
        # gets OOM-killed ("Killed") after the first holdout. With max_workers=1
        # this is sequential-but-isolated; >1 runs folds concurrently (VRAM/RAM
        # contention — only safe with one worker per physical GPU and spare RAM).
        with ProcessPoolExecutor(max_workers=max_workers, max_tasks_per_child=1) as pool:
            futures = {
                pool.submit(_run_single_fold, cfg, h, version, fold_log_dir): h for h in holdouts
            }
            for fut in as_completed(futures):
                holdout, result = fut.result()
                if result:
                    all_rmse[holdout] = result["overall"]["rmse"]

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
        return

    # --- Single-fold mode (default) ---
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

    if will_pretrain:
        run_pretrain(cfg, model=model, ctx=ctx)

    best_checkpoint_path = None
    if run_cfg.get("train", False):
        if not will_pretrain and run_cfg.get("load_checkpoint", False):
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
        run_eval(cfg, model=model, ctx=ctx, checkpoint_path=best_checkpoint_path)


if __name__ == "__main__":
    main()
