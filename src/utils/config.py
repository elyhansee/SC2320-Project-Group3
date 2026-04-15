"""YAML config loader with attribute-style access and path resolution."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(Exception):
    """Raised when configuration cannot be loaded or is invalid."""


@dataclass
class Config:
    """Wrapper around a config dict with dotted access and root path resolution."""

    data: Dict[str, Any]
    root: Path

    # Dotted access
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    # Path helpers
    def path(self, dotted: str) -> Path:
        value = self.get(dotted)
        if value is None:
            raise ConfigError(f"Missing path entry: {dotted}")
        return (self.root / value).resolve()

    def raw_path(self, key: str) -> Path:
        """Resolve a filename under raw_files against paths.data_raw."""
        filename = self.get(f"raw_files.{key}")
        if filename is None:
            raise ConfigError(f"Missing raw_files entry: {key}")
        return (self.root / self.get("paths.data_raw") / filename).resolve()


def load_config(path: str | Path) -> Config:
    """Load a YAML config file and anchor all paths at the project root."""
    cfg_path = Path(path).resolve()
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    # Project root is assumed to be the parent of the config directory.
    root = cfg_path.parent.parent
    cfg = Config(data=data, root=root)
    _ensure_output_dirs(cfg)
    return cfg


def _ensure_output_dirs(cfg: Config) -> None:
    for key in (
        "paths.data_interim",
        "paths.data_processed",
        "paths.outputs_root",
        "paths.outputs_figures",
        "paths.outputs_tables",
        "paths.outputs_models",
        "paths.outputs_logs",
    ):
        cfg.path(key).mkdir(parents=True, exist_ok=True)
