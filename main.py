from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig

from src.dataset_download import run_download
from src.preprocessing import run_preprocessing
from src.modeling.pretrain import run_pretrain
from src.modeling.train import run_train
from src.modeling.eval import run_eval


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_cfg = cfg.get("run", {})
    pretrained_state_dict = None
    version = cfg.get("version", "0.1.0")

    if run_cfg.get("download", False):
        run_download(cfg)
        
    if run_cfg.get("preprocess", False):
        run_preprocessing(cfg, version=version)
        
    if run_cfg.get("pretrain", False):
        checkpoint_path = run_pretrain(cfg, version=version)
        if checkpoint_path:
            try:
                pretrained_state_dict = torch.load(checkpoint_path, map_location="cpu")
                logger.info(f"Loaded pretrained weights from {checkpoint_path}")
            except FileNotFoundError:
                logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
    
    if run_cfg.get("train", False):
        run_train(cfg, pretrained_state_dict=pretrained_state_dict, version=version)

    if run_cfg.get("eval", False):
        run_eval(cfg, version=version)


if __name__ == "__main__":
    main()
