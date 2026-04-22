# BachelorsProject

Comparing transformer architecture against current deep neural networks for continous knee angle prediction using a single IMU

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         bachelorsproject and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── bachelorsproject   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes bachelorsproject a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Configurable Plotting

The training pipeline now supports configurable plotting and metric artifact generation for
pretraining, supervised training, and evaluation.

- Configuration file: `conf/plotting/default.yaml`
- Output location: `outputs/YYYY-MM-DD/HH-MM-SS/plots/<stage>/`
- Artifacts include stage metrics JSON and plots (format controlled by config).

### Useful Hydra Overrides

Run full pipeline with plotting enabled:

```bash
python main.py plotting.enabled=true run.pretrain=true run.train=true run.eval=true
```

Generate intermediate plots every 2 epochs (default):

```bash
python main.py plotting.cadence.every_n_epochs=2
```

Disable intermediate artifacts and keep only final reports:

```bash
python main.py plotting.cadence.save_intermediate=false
```

Disable specific eval plot categories:

```bash
python main.py plotting.eval.plot_prediction_examples=false plotting.eval.plot_subject_bars=false
```

Save additional formats:

```bash
python main.py plotting.save_formats='["png","pdf"]'
```

## GPU Optimization

GPU optimization settings are centralized in `conf/gpu/default.yaml` and data pipeline
throughput settings are in `conf/training/default.yaml`.

- Device selection: `gpu.device` (`auto`, `cuda`, `mps`, `cpu`)
- Mixed precision: `gpu.autocast.*`
- Compiler: `gpu.compile.*`
- Multi-GPU (single-process): `gpu.parallel.strategy` (`auto`, `dataparallel`, `none`)
- Data pipeline overlap: `training.num_workers`, `training.pin_memory`,
  `training.prefetch_factor`, `training.non_blocking_transfer`

### Presets

Balanced CUDA training:

```bash
python main.py gpu.device=cuda gpu.autocast.enabled=auto training.num_workers=8
```

Max throughput CUDA training:

```bash
python main.py gpu.device=cuda gpu.autocast.enabled=true gpu.compile.enabled=true gpu.cuda.tf32=true gpu.parallel.strategy=auto training.num_workers=8
```

Apple Silicon MPS training:

```bash
python main.py gpu.device=mps gpu.autocast.enabled=false training.num_workers=4
```

Strict reproducibility mode:

```bash
python main.py gpu.deterministic=true gpu.cuda.cudnn_benchmark=false
```

