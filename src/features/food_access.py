"""Food access features built from hawker, supermarket and market layers.

A subzone's food access is summarised by:

    1. Walking distance from the centroid to the nearest amenity of each type
    2. The number of amenities of each type within a configurable buffer
    3. A composite ``food_access_score`` standardised across subzones
    4. A ``food_amenity_diversity`` count of amenity types present nearby
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.geo_utils import compute_centroids, count_within_radius, nearest_distance_km
from src.utils.logging_utils import get_logger
from src.utils.math_utils import zscore_series

logger = get_logger(__name__)


@dataclass
class FoodAmenitySpec:
    """Description of one food amenity layer."""

    name: str        # short name used as a column prefix
    layer: gpd.GeoDataFrame


def build_food_access_features(
    cfg: Config,
    subzones: gpd.GeoDataFrame,
    hawkers: gpd.GeoDataFrame,
    supermarkets: gpd.GeoDataFrame,
    markets: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Build per-subzone food access features.

    Returns columns:
        subzone_key,
        nearest_hawker_km, nearest_supermarket_km, nearest_market_km,
        hawker_count, supermarket_count, market_count,
        hawker_count_log, supermarket_count_log, market_count_log,
        nearest_food_km, food_amenity_count, food_amenity_count_log,
        food_amenity_diversity, food_access_score
    """
    projected = cfg.get("crs.projected")
    buffer_km = float(cfg.get("features.food_access.buffer_km", 1.0))
    radius_m = buffer_km * 1000.0

    centroids = compute_centroids(subzones[["subzone_key", "geometry"]], projected)

    specs: List[FoodAmenitySpec] = [
        FoodAmenitySpec("hawker", hawkers),
        FoodAmenitySpec("supermarket", supermarkets),
        FoodAmenitySpec("market", markets),
    ]

    out = pd.DataFrame({"subzone_key": centroids["subzone_key"].values})

    nearest_cols: List[str] = []
    count_cols: List[str] = []
    for spec in specs:
        if spec.layer is None or len(spec.layer) == 0:
            logger.warning("Food layer '%s' is empty; skipping", spec.name)
            out[f"nearest_{spec.name}_km"] = np.nan
            out[f"{spec.name}_count"] = 0
            out[f"{spec.name}_count_log"] = 0.0
            continue
        nearest = nearest_distance_km(centroids, spec.layer, projected)
        counts = count_within_radius(centroids, spec.layer, radius_m, projected)
        out[f"nearest_{spec.name}_km"] = nearest
        out[f"{spec.name}_count"] = counts.astype(int)
        out[f"{spec.name}_count_log"] = np.log1p(counts)
        nearest_cols.append(f"nearest_{spec.name}_km")
        count_cols.append(f"{spec.name}_count")

    # Composite features.
    if nearest_cols:
        out["nearest_food_km"] = out[nearest_cols].min(axis=1)
    else:
        out["nearest_food_km"] = np.nan
    out["food_amenity_count"] = out[count_cols].sum(axis=1) if count_cols else 0
    out["food_amenity_count_log"] = np.log1p(out["food_amenity_count"])
    out["food_amenity_diversity"] = (out[count_cols] > 0).sum(axis=1)

    # food_access_score: mean of standardised counts (positive = good access).
    if count_cols:
        z = out[count_cols].apply(zscore_series)
        out["food_access_score"] = z.mean(axis=1)
    else:
        out["food_access_score"] = 0.0

    logger.info(
        "Food access features: mean nearest_food_km=%.2f, mean diversity=%.2f",
        float(out["nearest_food_km"].mean()),
        float(out["food_amenity_diversity"].mean()),
    )
    return out
