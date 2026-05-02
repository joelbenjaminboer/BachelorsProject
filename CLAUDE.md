# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Bachelor's thesis comparing transformer-based encoder (with MAE pretraining) against TCN and TimesNet for continuous knee angle prediction from 6-axis IMU data (ENABL3S dataset).

Do not make any changes until you are 95% confidence in what you need to build. If you are not sure, ask for clarification.

- Architectural decisions:
  - Use a transformer-based model for time series prediction.
  - Preprocess IMU data to create input sequences and corresponding targets.
  - Train the model on the preprocessed data and evaluate its performance.
  - Using Hydra for configuration management to easily switch between different settings and parameters.

Use small HAIKU subagents for any exploration or research and return only summarized information

## Commands

**Run full pipeline:**
```bash
python main.py
```

**Select pipeline stages** (all flags default to values in `conf/run/default.yaml`):
```bash
python main.py run.preprocess=false run.pretrain=false run.train=true run.eval=true
```

**Switch model:**
```bash
python main.py model=tcn        # Temporal Convolutional Network
python main.py model=timesnet   # TimesNet (FFT-based)
# default is the transformer encoder (conf/model/default.yaml)
```

**GPU presets:**
```bash
# Apple Silicon
python main.py gpu.device=mps gpu.autocast.enabled=false training.num_workers=4

# CUDA max throughput
python main.py gpu.device=cuda gpu.autocast.enabled=true gpu.compile.enabled=true gpu.cuda.tf32=true training.num_workers=8
```

**Load a checkpoint:**
```bash
python main.py run.load_checkpoint=true run.checkpoint_path="outputs/.../best_pretrained_epoch_4.pth"
```

**Lint:**
```bash
ruff check src/
ruff format src/
```

## Architecture

### Pipeline (orchestrated by `main.py` via Hydra)

```
download → preprocess → pretrain → train → eval
```

Each stage is toggled via `conf/run/default.yaml`. All config is managed by Hydra; outputs land in `outputs/YYYY-MM-DD/HH-MM-SS/`.

### Data flow

1. **`src/data/preprocessing.py`** — Reads raw ENABL3S CSVs from ZIP archives, downsamples 500 Hz → 100 Hz, filters activity types, creates sliding windows (`context_length=125`, `forecast_horizon=50`), saves per-subject HDF5 files under `data/processed/ENABL3S/fold_<subject>/`.

2. **`src/data/dataloader.py`** — Loads HDF5, computes z-score normalization from the train split, returns `(train_loader, val_loader, test_loader)`. Uses **Leave-One-Subject-Out** (LOSO): one holdout subject is 100% test, the rest are 90/10 train/val.

3. **`src/models/encoder.py`** — `IMU_Intent_Encoder`: linear input projection → CLS token prepend → sinusoidal positional encoding → N × TransformerEncoder layers → dual heads (reconstruction for MAE pretraining, regression for supervised prediction).

4. **`src/training/pretrain.py`** — Masked AutoEncoder: masks random contiguous blocks of the input, trains reconstruction head with per-channel MSE.

5. **`src/training/train.py`** — Two-phase fine-tuning: Phase 1 freezes the encoder and trains only the regression head for `freeze_encoder_epochs`; Phase 2 unfreezes and fine-tunes the full model. Saves best checkpoint by validation loss.

6. **`src/training/eval.py`** — Loads best checkpoint, runs inference on held-out test subjects, reports per-step RMSE/MAE.

### Key supporting modules

- **`src/models/factory.py`** — Builds models, AdamW optimizer, ReduceLROnPlateau scheduler, and loss functions from config.
- **`src/runtime.py`** — `RunContext` dataclass, device selection, mixed precision (autocast + GradScaler), `torch.compile`, multi-GPU (`DataParallel`), shared `backward_and_step` training helper.
- **`src/training/plotting.py`** — Generates loss curves, reconstruction metrics, prediction examples, and per-subject bar charts; controlled by `conf/plotting/default.yaml`.

### Config layout (`conf/`)

| File | Controls |
|------|----------|
| `config.yaml` | Top-level defaults, version string |
| `dataset/enabl3s.yaml` | Paths, source URL |
| `model/default.yaml` | Encoder dims, optimizer, loss |
| `model/tcn.yaml` | TCN-specific overrides |
| `model/timesnet.yaml` | TimesNet-specific overrides |
| `training/default.yaml` | Epochs, batch size, freeze schedule, window sizes |
| `gpu/default.yaml` | Device, autocast, compile, CUDA flags |
| `run/default.yaml` | Stage toggles, checkpoint loading |
| `plotting/default.yaml` | Enable/disable, cadence, per-stage plot types |

## Applied learning

when a recurring failure or workaround is found, add a one-liner to the "Applied learning" section in `README.md` (under 15 words, future-value only).
