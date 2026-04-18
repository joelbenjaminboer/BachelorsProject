import hydra
import torch
from loguru import logger
from omegaconf import DictConfig

from src.dataset_download import run_download
from src.preprocessing import run_preprocessing
from src.modeling.pretrain import run_pretrain
from src.modeling.train import run_train


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_cfg = cfg.get("run", {})
    pretrained_state_dict = None

    if run_cfg.get("download", False):
        run_download(cfg)
        
    if run_cfg.get("preprocess", False):
        run_preprocessing(cfg)
        
    if run_cfg.get("pretrain", False):
        checkpoint_path = run_pretrain(cfg)
        if checkpoint_path:
            try:
                pretrained_state_dict = torch.load(checkpoint_path, map_location="cpu")
                logger.info(f"Loaded pretrained weights from {checkpoint_path}")
            except FileNotFoundError:
                logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
    if run_cfg.get("train", False):
        
        run_train(cfg, pretrained_state_dict=pretrained_state_dict)


if __name__ == "__main__":
    main()
