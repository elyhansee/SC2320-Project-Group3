"""Unit tests for the config loader."""
from pathlib import Path

from src.utils.config import load_config


def test_load_config_basic(tmp_path: Path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "settings.yaml").write_text(
        """
project:
  name: demo
  random_seed: 7
paths:
  data_raw: data/raw
  data_interim: data/interim
  data_processed: data/processed
  outputs_root: outputs
  outputs_figures: outputs/figures
  outputs_tables: outputs/tables
  outputs_models: outputs/models
  outputs_logs: outputs/logs
"""
    )
    cfg = load_config(cfg_dir / "settings.yaml")
    assert cfg.get("project.name") == "demo"
    assert cfg.get("project.random_seed") == 7
    assert cfg.path("paths.outputs_tables").exists()
