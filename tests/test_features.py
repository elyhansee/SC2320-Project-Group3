"""Unit tests for the binary matrix and demographic feature builders."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from src.features.binary import build_binary_matrix
from src.features.demographic import build_demographic_features
from src.utils.config import load_config


def _write_settings(tmp_path: Path) -> Path:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "settings.yaml").write_text(
        """
project:
  name: demo
  random_seed: 1
paths:
  data_raw: data/raw
  data_interim: data/interim
  data_processed: data/processed
  outputs_root: outputs
  outputs_figures: outputs/figures
  outputs_tables: outputs/tables
  outputs_models: outputs/models
  outputs_logs: outputs/logs
crs:
  geographic: EPSG:4326
  projected: EPSG:3414
features:
  population:
    elderly_age_columns: [Total_65_69, Total_70_74]
    total_column: Total_Total
    name_column: Number
binary_thresholds:
  high_elderly_share_q: 0.5
  high_elderly_density_q: 0.5
  poor_food_access_q: 0.5
  poor_transit_q: 0.5
  rich_food_q: 0.5
  rich_transit_q: 0.5
  rich_accessibility_q: 0.5
  poor_accessibility_q: 0.5
  hawker_present_min: 1
  supermarket_present_min: 1
  market_present_min: 1
  diverse_food_min_types: 2
"""
    )
    return cfg_dir / "settings.yaml"


def _toy_subzones() -> gpd.GeoDataFrame:
    polys = [
        Polygon([(103.8, 1.30), (103.81, 1.30), (103.81, 1.31), (103.8, 1.31)]),
        Polygon([(103.81, 1.30), (103.82, 1.30), (103.82, 1.31), (103.81, 1.31)]),
        Polygon([(103.82, 1.30), (103.83, 1.30), (103.83, 1.31), (103.82, 1.31)]),
    ]
    return gpd.GeoDataFrame(
        {"subzone_key": ["A", "B", "C"], "geometry": polys},
        crs="EPSG:4326",
    )


def test_demographic_features(tmp_path: Path) -> None:
    cfg = load_config(_write_settings(tmp_path))
    pop = pd.DataFrame(
        {
            "subzone_key": ["A", "B", "C"],
            "Total_65_69": [10, 0, 30],
            "Total_70_74": [10, 0, 20],
            "Total_Total": [200, 100, 100],
        }
    )
    out = build_demographic_features(cfg, pop, _toy_subzones())
    assert set(out.columns) >= {
        "subzone_key", "elderly_count", "elderly_pct",
        "elderly_density", "elderly_density_log", "area_km2",
    }
    pct = out.set_index("subzone_key")["elderly_pct"]
    assert pct.loc["A"] == pytest.approx(0.10)
    assert pct.loc["B"] == pytest.approx(0.0)
    assert pct.loc["C"] == pytest.approx(0.50)


def test_binary_matrix_uses_real_thresholds(tmp_path: Path) -> None:
    cfg = load_config(_write_settings(tmp_path))
    features = pd.DataFrame(
        {
            "subzone_key": ["A", "B", "C"],
            "elderly_pct": [0.05, 0.20, 0.30],
            "elderly_density": [10.0, 50.0, 100.0],
            "nearest_food_km": [0.2, 1.0, 3.0],
            "food_access_score": [-1.0, 0.0, 1.0],
            "transit_access_score": [-2.0, 0.0, 2.0],
            "accessibility_support_score": [-1.0, 0.0, 1.0],
            "hawker_count": [0, 1, 5],
            "supermarket_count": [0, 2, 3],
            "market_count": [0, 0, 1],
            "food_amenity_diversity": [0, 1, 3],
        }
    )
    matrix = build_binary_matrix(cfg, features)
    assert matrix.shape[0] == 3
    assert "high_elderly_share" in matrix.columns
    assert matrix.loc["C", "high_elderly_share"] == np.True_
    assert matrix.loc["A", "high_elderly_share"] == np.False_
    assert matrix.loc["C", "diverse_food_environment"] == np.True_
