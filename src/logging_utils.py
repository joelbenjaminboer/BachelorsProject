"""Bridge loguru into Hydra's run directory.

Loguru writes only to stderr by default, so Hydra's per-run ``*.log`` files stay
empty (Hydra only captures the stdlib ``logging`` module). ``setup_logging``
re-points loguru so that:

* a colored sink prints to the console (single source — Hydra's own console
  handler is disabled via ``conf/hydra/job_logging/loguru.yaml``);
* every loguru record is *propagated* into the stdlib logging tree, where
  Hydra's ``FileHandler`` writes it to ``outputs/.../<job>.log``. Third-party
  libraries that log through ``logging`` land in the same file for free.

In a subprocess without an active Hydra config (e.g. parallel LOSO folds), pass
``output_dir`` explicitly to get a direct loguru file sink instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import traceback

from hydra.core.hydra_config import HydraConfig
from loguru import logger

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"


class _PropagateHandler(logging.Handler):
    """Forward loguru records into the stdlib logging tree (and Hydra's handlers)."""

    def emit(self, record: logging.LogRecord) -> None:
        logging.getLogger(record.name).handle(record)


def _install_excepthook() -> None:
    """Route uncaught exceptions into loguru so they also land in the log file.

    Loguru sinks only receive ``logger.*`` calls, so without this an uncaught
    crash (e.g. eval dying inside the baseline step) prints only to the terminal
    and the run's ``*.log`` just ends mid-run with no error. The full traceback
    is formatted into the message so it is recorded regardless of sink type.
    """

    def _hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.critical("Uncaught exception — terminating:\n{}", tb.rstrip())

    sys.excepthook = _hook


def setup_logging(cfg=None, *, output_dir: str | Path | None = None) -> Path | None:
    """Route loguru to a colored console and into Hydra's run-directory log file.

    Returns the path of the file that will receive logs, or ``None`` if it cannot
    be determined. Safe to call multiple times (per process / per fold).
    """
    level = "INFO"
    if cfg is not None:
        level = str(cfg.get("log_level", "INFO")).upper()

    logger.remove()
    logger.add(sys.stderr, level=level, format=_CONSOLE_FORMAT, enqueue=True)
    _install_excepthook()

    hydra_active = output_dir is None and HydraConfig.initialized()

    if hydra_active:
        # Hydra owns the file handler; forward loguru records into stdlib logging
        # so they end up in outputs/.../<job>.log alongside any library logs.
        # The message is pre-formatted here so Hydra's plain "%(message)s"
        # formatter still yields timestamped, leveled file lines.
        logger.add(_PropagateHandler(), level=level, format=_FILE_FORMAT)
        hc = HydraConfig.get()
        return Path(hc.runtime.output_dir) / f"{hc.job.name}.log"

    # No active Hydra logging (e.g. a worker subprocess): write the file ourselves.
    if output_dir is None:
        return None
    log_path = Path(output_dir) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_path, level=level, format=_FILE_FORMAT, enqueue=True)
    return log_path
