"""Logging setup used throughout the pipeline."""
from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Optional

import yaml

_CONFIGURED = False


def setup_logging(
    yaml_path: Optional[Path] = None,
    default_level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> None:
    """Initialise logging once per process.

    Reads config/logging.yaml if available, otherwise falls back to a simple
    stream handler. ``log_file`` overrides the file handler destination.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    cfg_loaded = False
    if yaml_path and Path(yaml_path).exists():
        try:
            with open(yaml_path, "r", encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle)
            if log_file is not None:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                if "handlers" in cfg and "file" in cfg["handlers"]:
                    cfg["handlers"]["file"]["filename"] = str(log_file)
            logging.config.dictConfig(cfg)
            cfg_loaded = True
        except Exception:  # noqa: BLE001
            cfg_loaded = False

    if not cfg_loaded:
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers = [logging.StreamHandler()]
        if log_file is not None:
            handlers.append(logging.FileHandler(str(log_file), encoding="utf-8"))
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
        )

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
