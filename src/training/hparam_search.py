"""Optuna hyperparameter search for the encoder model."""

import gc
import json

import optuna
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.models.factory import build_and_prepare_model
from src.runtime import RunContext
from src.training.train import Trainer


class _TrialTrainer(Trainer):
    """Trainer variant for Optuna trials: no checkpoint/artifact saving, reports pruning signals."""

    def __init__(self, cfg: DictConfig, model, ctx: RunContext, trial: optuna.Trial):
        super().__init__(cfg, model, ctx)
        self._trial = trial

    def _save_if_best(self, val_loss: float, epoch: int):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1

    def run(self) -> float:
        freeze_epochs = self._resolve_freeze_epochs()
        if freeze_epochs > 0:
            self._begin_freeze_phase(freeze_epochs)

        for epoch in range(self.epochs):
            if freeze_epochs > 0 and epoch == freeze_epochs:
                self._begin_finetune_phase()

            self._train_epoch(epoch)
            val_loss = self._validate_epoch(epoch)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            self._save_if_best(val_loss, epoch)

            logger.info(
                "Trial {} | Epoch {}/{} | val_loss={:.4f} | lr={:.2e}",
                self._trial.number,
                epoch + 1,
                self.epochs,
                val_loss,
                self.optimizer.param_groups[0]["lr"],
            )

            self._trial.report(val_loss, epoch)
            if self._trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return self.best_val_loss


def _nested_from_dotpaths(flat: dict) -> dict:
    """Convert {'a.b.c': v} → {'a': {'b': {'c': v}}} for OmegaConf.create."""
    result: dict = {}
    for dotpath, value in flat.items():
        parts = dotpath.split(".")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def _suggest_overrides(trial: optuna.Trial, search_space: DictConfig, trial_epochs: int) -> dict:
    flat: dict = {}
    for name, spec in search_space.items():
        param_type = str(spec.type).lower()
        target = str(spec.target)
        if param_type == "categorical":
            value = trial.suggest_categorical(name, list(spec.choices))
        elif param_type == "int":
            value = trial.suggest_int(name, int(spec.low), int(spec.high))
        elif param_type == "float":
            value = trial.suggest_float(
                name, float(spec.low), float(spec.high), log=bool(spec.get("log", False))
            )
        else:
            raise ValueError(f"Unknown search_space type '{param_type}' for param '{name}'")
        flat[target] = value

    flat["training.epochs"] = trial_epochs
    flat["gpu.compile.enabled"] = False
    return _nested_from_dotpaths(flat)


def _run_trial(trial: optuna.Trial, cfg: DictConfig, ctx: RunContext, trial_epochs: int) -> float:
    search_space = cfg.hparam_search.search_space
    overrides = _suggest_overrides(trial, search_space, trial_epochs)
    trial_cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))

    logger.info("Trial {} starting | params: {}", trial.number, trial.params)

    model = build_and_prepare_model(trial_cfg, ctx)
    try:
        return _TrialTrainer(trial_cfg, model, ctx, trial).run()
    finally:
        del model
        gc.collect()
        if ctx.device.type == "cuda":
            torch.cuda.empty_cache()


def _save_results(study: optuna.Study) -> None:
    import csv

    rows = [
        {"trial": t.number, "val_loss": t.value, **t.params}
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if rows:
        csv_path = "hparam_search_trials.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved trial CSV → {}", csv_path)

    best = study.best_trial
    best_path = "hparam_search_best.json"
    with open(best_path, "w") as f:
        json.dump({"val_loss": best.value, "params": best.params}, f, indent=2)
    logger.info("Saved best params → {}", best_path)


def run_hparam_search(cfg: DictConfig, ctx: RunContext) -> optuna.Study:
    search_cfg = cfg.get("hparam_search", {})
    n_trials = int(search_cfg.get("n_trials", 50))
    trial_epochs = int(search_cfg.get("epochs", 10))
    direction = str(search_cfg.get("direction", "minimize"))
    storage = search_cfg.get("storage", None)
    n_jobs = int(search_cfg.get("n_jobs", 1))
    pruner_cfg = search_cfg.get("pruner", {})
    pruner_startup = int(pruner_cfg.get("n_startup_trials", 5))
    pruner_warmup = int(pruner_cfg.get("n_warmup_steps", 5))

    if n_jobs > 1 and storage is not None and "sqlite" in str(storage).lower():
        logger.warning(
            "n_jobs={} with SQLite storage causes write contention; "
            "switch to storage=null (in-memory) or a proper RDB (PostgreSQL/MySQL) "
            "for reliable parallel trials",
            n_jobs,
        )
    if n_jobs > 1 and ctx.device.type == "mps":
        logger.warning(
            "n_jobs={} on MPS is not recommended — MPS has limited thread-safety; "
            "parallel trials may produce incorrect results or crash",
            n_jobs,
        )

    study_name = f"{cfg.get('version', 'search')}_hparam_search"
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=pruner_startup,
        n_warmup_steps=pruner_warmup,
    )
    study = optuna.create_study(
        direction=direction,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
        study_name=study_name,
    )

    logger.info(
        "Optuna search starting | study={} n_trials={} trial_epochs={} n_jobs={}",
        study_name,
        n_trials,
        trial_epochs,
        n_jobs,
    )

    study.optimize(
        lambda trial: _run_trial(trial, cfg, ctx, trial_epochs),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    best = study.best_trial
    logger.info("Search complete | best val_loss={:.6f}", best.value)
    logger.info("Best params: {}", best.params)

    _save_results(study)
    return study
