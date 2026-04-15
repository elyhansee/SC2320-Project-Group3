"""Elderly demographic features from the Census 2020 subzone table."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.geo_utils import compute_area_km2
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_demographic_features(
    cfg: Config,
    population: pd.DataFrame,
    subzones: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Compute elderly indicators per subzone.

    Returns columns:
        subzone_key, total_population, elderly_count, elderly_pct,
        elderly_density (per km2), elderly_density_log, area_km2
    """
    elderly_cols = list(cfg.get("features.population.elderly_age_columns"))
    total_col = cfg.get("features.population.total_column")
    missing = [c for c in elderly_cols + [total_col] if c not in population.columns]
    if missing:
        raise KeyError(f"Population table missing columns: {missing}")

    df = population[["subzone_key"] + elderly_cols + [total_col]].copy()
    df["elderly_count"] = df[elderly_cols].sum(axis=1)
    df["total_population"] = df[total_col]
    df["elderly_pct"] = np.where(
        df["total_population"] > 0,
        df["elderly_count"] / df["total_population"],
        0.0,
    )

    # Subzone area for density.
    projected = cfg.get("crs.projected")
    area_km2 = compute_area_km2(subzones, projected)
    area_df = pd.DataFrame(
        {"subzone_key": subzones["subzone_key"].values, "area_km2": area_km2}
    ).drop_duplicates("subzone_key")
    df = df.merge(area_df, on="subzone_key", how="left")
    df["elderly_density"] = np.where(
        df["area_km2"] > 0, df["elderly_count"] / df["area_km2"], 0.0
    )
    df["elderly_density_log"] = np.log1p(df["elderly_density"])

    out = df[
        [
            "subzone_key",
            "total_population",
            "elderly_count",
            "elderly_pct",
            "elderly_density",
            "elderly_density_log",
            "area_km2",
        ]
    ].copy()
    logger.info(
        "Demographic features built: %d subzones, mean elderly_pct=%.3f",
        len(out),
        out["elderly_pct"].mean(),
    )
    return out
