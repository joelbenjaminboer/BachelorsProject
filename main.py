from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig
from src.dataset_download import run_download
from src.modeling.eval import run_eval
from src.modeling.pretrain import run_pretrain
from src.modeling.train import run_train
from src.preprocessing import run_preprocessing
import torch


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_cfg = cfg.get("run", {})
    pretrained_state_dict = None
    best_checkpoint_path = None
    version = cfg.get("version", "0.1.0")

    if run_cfg.get("download", False):
        run_download(cfg)
        
    if run_cfg.get("preprocess", False):
        run_preprocessing(cfg, version=version)
        
    if run_cfg.get("pretrain", False):
        if not cfg.model.get("supports_pretrain", True):
            logger.info(
                "Skipping pretraining: model_type='{}' does not support MAE pretraining",
                cfg.model.get("model_type", "encoder"),
            )
        else:
            checkpoint_path = run_pretrain(cfg, version=version)
            if checkpoint_path:
                try:
                    pretrained_state_dict = torch.load(checkpoint_path, map_location="cpu")
                    logger.info(f"Loaded pretrained weights from {checkpoint_path}")
                except FileNotFoundError:
                    logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
    
    if run_cfg.get("train", False):
        if run_cfg.get("load_checkpoint", False) and pretrained_state_dict is None:
            checkpoint_path = Path(run_cfg.get("checkpoint_path", ""))
            if checkpoint_path.exists():
                pretrained_state_dict = torch.load(checkpoint_path, map_location="cpu")
                logger.info(f"Loaded pretrained weights from {checkpoint_path}")
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}. Starting training from scratch.")

        best_checkpoint_path = run_train(cfg, pretrained_state_dict=pretrained_state_dict, version=version)

    if run_cfg.get("eval", False):
        run_eval(cfg, version=version, checkpoint_path=best_checkpoint_path)


if __name__ == "__main__":
    main()
