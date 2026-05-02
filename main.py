from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig
import torch

from src.data.download import run_download
from src.data.preprocessing import run_preprocessing
from src.models.factory import build_and_prepare_model
from src.runtime import build_run_context, load_state_into_model
from src.training.eval import run_eval
from src.training.pretrain import run_pretrain
from src.training.train import run_train


MODEL_STAGES = ("pretrain", "train", "eval")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_cfg = cfg.get("run", {})
    version = cfg.get("version", "0.1.0")

    if run_cfg.get("download", False):
        run_download(cfg)

    if run_cfg.get("preprocess", False):
        run_preprocessing(cfg, version=version)

    if not any(run_cfg.get(stage, False) for stage in MODEL_STAGES):
        return

    ctx = build_run_context(cfg, version=version)
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
