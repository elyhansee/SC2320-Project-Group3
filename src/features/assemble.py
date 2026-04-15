"""Assemble the master subzone feature table.

Each feature family lives in its own module. This module is the single
join point that produces the canonical ``subzone_features`` DataFrame
used by the mining, clustering and risk modelling stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import geopandas as gpd
import pandas as pd

from src.features.accessibility import build_accessibility_features
from src.features.demographic import build_demographic_features
from src.features.food_access import build_food_access_features
from src.features.landuse import build_landuse_features
from src.features.transit import build_transit_features
from src.utils.config import Config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureBundle:
    """Holds the assembled feature tables and the column groupings."""

    full: pd.DataFrame                      # all subzones with every column
    modelling: pd.DataFrame                 # residential subset used for modelling
    feature_columns: List[str]              # canonical numeric columns
    metadata_columns: List[str] = field(default_factory=list)


def build_feature_bundle(cfg: Config, data: Dict[str, object]) -> FeatureBundle:
    """Build every feature family and return a :class:`FeatureBundle`."""
    subzones: gpd.GeoDataFrame = data["subzones"]  # type: ignore[assignment]

    demo = build_demographic_features(cfg, data["population"], subzones)
    food = build_food_access_features(
        cfg,
        subzones,
        data["hawkers"],
        data["supermarkets"],
        data["markets"],
    )
    transit = build_transit_features(
        cfg,
        subzones,
        data["bus_stops"],
        data["mrt_exits"],
    )
    accessibility = build_accessibility_features(
        cfg,
        subzones,
        data["friendly_buildings"],
        data["senior_centres_ts"],
    )
    landuse = build_landuse_features(cfg, subzones, data["land_use"])

    base_meta = (
        subzones[["subzone_key", "SUBZONE_N", "PLN_AREA_N", "REGION_N"]]
        .drop_duplicates("subzone_key")
        .reset_index(drop=True)
    )
    merged = (
        base_meta
        .merge(demo, on="subzone_key", how="left")
        .merge(food, on="subzone_key", how="left")
        .merge(transit, on="subzone_key", how="left")
        .merge(accessibility, on="subzone_key", how="left")
        .merge(landuse, on="subzone_key", how="left")
    )

    feature_columns = list(cfg.get("clustering.feature_columns"))
    missing = [c for c in feature_columns if c not in merged.columns]
    if missing:
        raise KeyError(
            f"Configured feature columns missing from feature table: {missing}"
        )

    # Drop subzones with no inhabitants and no critical inputs.
    critical = ["elderly_pct", "nearest_food_km", "transit_access_score"]
    before = len(merged)
    merged = merged.dropna(subset=critical).reset_index(drop=True)
    if before != len(merged):
        logger.warning("Dropped %d subzones with NA on critical features", before - len(merged))

    # Fill remaining NA columns conservatively. Counts default to zero.
    merged = merged.fillna(0)

    # Modelling subset = subzones flagged as residential and with >0 elderly.
    modelling_mask = (merged["is_residential"]) & (merged["elderly_count"] > 0)
    modelling = merged[modelling_mask].reset_index(drop=True)
    logger.info(
        "Final feature matrix: %d subzones (modelling subset: %d)",
        len(merged),
        len(modelling),
    )

    return FeatureBundle(
        full=merged,
        modelling=modelling,
        feature_columns=feature_columns,
        metadata_columns=["subzone_key", "SUBZONE_N", "PLN_AREA_N", "REGION_N"],
    )
