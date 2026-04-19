import os
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from src.dataloader import IMUKneeDataset
from src.modeling.factory import build_encoder, build_loss, build_optimizer
from loguru import logger
from tqdm import tqdm

# Setup device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Trainer:
    def __init__(self, cfg: DictConfig, pretrained_state_dict=None):
        self.cfg = cfg
        self.device = device

        logger.info(f"Using device: {self.device}")

        self.seq_length = cfg.training.context_length
        self.forecast_horizon = cfg.training.forecast_horizon
        self.batch_size = cfg.training.batch_size
        self.processed_dir = hydra.utils.to_absolute_path(cfg.dataset.processed_dir)

        self.dataset = IMUKneeDataset(self.processed_dir, self.seq_length, self.forecast_horizon)
        logger.info(f"Total samples: {len(self.dataset)}")

        train_size = int(cfg.training.split_ratio * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )

        num_workers = cfg.training.get("num_workers", min(os.cpu_count() or 1, 8))
        persistent_workers = cfg.training.get("persistent_workers", num_workers > 0)
        if num_workers == 0:
            persistent_workers = False
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

        logger.info("Initializing IMU_Intent_Encoder from cfg.model")
        self.model = build_encoder(
            cfg=cfg,
            seq_length=self.seq_length,
            forecast_horizon=self.forecast_horizon,
        )
        self.model.to(self.device)

        # In your train.py, before calling load_state_dict:
        if pretrained_state_dict is not None:
            # Remove the old positional encoding from the checkpoint
            if "positional_layer.pe" in pretrained_state_dict:
                del pretrained_state_dict["positional_layer.pe"]
            
            # Now load the remaining weights (Transformer, projections, etc.)
            self.model.load_state_dict(pretrained_state_dict, strict=False)
            print("Loaded pretrained weights into the model (excluding positional encoding)")

        self.optimizer = build_optimizer(cfg, self.model.parameters())
        self.criterion = build_loss(cfg)

        self.epochs = cfg.training.epochs
        self.best_val_loss = float("inf")

    def run(self):
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Train]"):
                past_imu = batch["past_imu"].to(self.device)
                future_knee = batch["future_knee"].to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(past_imu, task="predict")
                loss = self.criterion(predictions, future_knee)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            if len(self.train_loader) > 0:
                train_loss /= len(self.train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Val]"):
                    past_imu = batch["past_imu"].to(self.device)
                    future_knee = batch["future_knee"].to(self.device)

                    predictions = self.model(past_imu, task="predict")
                    loss = self.criterion(predictions, future_knee)
                    val_loss += loss.item()

            if len(self.val_loader) > 0:
                val_loss /= len(self.val_loader)
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = os.path.join(
                    hydra.utils.get_original_cwd(),
                    "checkpoints",
                    f"best_model_epoch_{epoch + 1}.pth",
                )
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(f"Saved new best model checkpoint to {checkpoint_path}")


def run_train(cfg: DictConfig, pretrained_state_dict=None):
    trainer = Trainer(cfg, pretrained_state_dict=pretrained_state_dict)
    trainer.run()


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_train(cfg)


if __name__ == "__main__":
    main()
