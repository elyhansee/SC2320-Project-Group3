"""Thin IO helpers for tabular, JSON and GeoJSON outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import pandas as pd


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """Persist a DataFrame as CSV, creating parent directories on demand."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def save_json(obj: Dict[str, Any], path: Path) -> Path:
    """Persist a JSON-serialisable object with stable indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=str)
    return path


def read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_geodataframe(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    """Persist a GeoDataFrame as GeoJSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")
    return path
