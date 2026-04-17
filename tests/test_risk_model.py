"""Unit tests for the risk modelling phase: target builder and classifiers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.modelling.risk_model import metrics_to_table, train_risk_models
from src.modelling.target import build_vulnerability_target
from src.utils.config import load_config


def _write_settings(tmp_path: Path) -> Path:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "settings.yaml").write_text(
        """
project:
  name: demo
  random_seed: 13
paths:
  data_raw: data/raw
  data_interim: data/interim
  data_processed: data/processed
  outputs_root: outputs
  outputs_figures: outputs/figures
  outputs_tables: outputs/tables
  outputs_models: outputs/models
  outputs_logs: outputs/logs
risk_model:
  components:
    - feature: elderly_pct
      direction: positive
      weight: 1.0
      reason: more elderly is more vulnerable
    - feature: nearest_food_km
      direction: positive
      weight: 1.0
      reason: longer distance is worse
    - feature: food_access_score
      direction: negative
      weight: 1.0
      reason: more nearby food is better
  high_risk_quantile: 0.5
  test_size: 0.3
  rf_n_estimators: 50
  rf_min_samples_leaf: 2
  cv_folds: 3
"""
    )
    return cfg_dir / "settings.yaml"


def _toy_features(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    elderly = rng.uniform(0.05, 0.30, n)
    distance = rng.uniform(0.1, 3.0, n)
    food_score = -elderly * 4 - distance + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {
            "subzone_key": [f"S{i}" for i in range(n)],
            "elderly_pct": elderly,
            "nearest_food_km": distance,
            "food_access_score": food_score,
        }
    )


def test_build_vulnerability_target(tmp_path: Path) -> None:
    cfg = load_config(_write_settings(tmp_path))
    features = _toy_features()
    target = build_vulnerability_target(cfg, features)
    assert len(target.scores) == len(features)
    assert target.binary_label.sum() > 0
    assert target.binary_label.sum() < len(features)
    assert "elderly_pct" in target.components["feature"].values


def test_train_risk_models_outputs_metrics(tmp_path: Path) -> None:
    cfg = load_config(_write_settings(tmp_path))
    features = _toy_features()
    target = build_vulnerability_target(cfg, features)
    feature_columns = ["elderly_pct", "nearest_food_km", "food_access_score"]
    report = train_risk_models(
        cfg, features, feature_columns, target.binary_label.values, seed=42
    )
    assert set(report.results.keys()) == {
        "random_forest", "logistic_regression", "decision_tree", "dummy_majority",
    }
    table = metrics_to_table(report)
    assert "accuracy" in table.columns
    rf_acc = table.set_index("model").loc["random_forest", "accuracy"]
    dummy_acc = table.set_index("model").loc["dummy_majority", "accuracy"]
    # The forest should learn the explicit linear target.
    assert rf_acc >= dummy_acc
