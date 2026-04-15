"""Barrier-free accessibility and ageing-in-place support features.

The friendly building density proxies the built-environment barrier-free
support a frail elderly resident can rely on. The senior centre context
is a single national-level number from the annual file (no per-subzone
geocoding) and is exposed as a uniform feature column for transparency.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.geo_utils import compute_centroids, count_within_radius
from src.utils.logging_utils import get_logger
from src.utils.math_utils import zscore_series

logger = get_logger(__name__)


def build_accessibility_features(
    cfg: Config,
    subzones: gpd.GeoDataFrame,
    friendly_buildings: gpd.GeoDataFrame,
    senior_centres_ts: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-subzone accessibility support features.

    Returns columns:
        subzone_key, friendly_building_count, friendly_building_count_log,
        friendly_building_density, accessibility_support_score,
        national_active_ageing_centres
    """
    projected = cfg.get("crs.projected")
    buffer_km = float(cfg.get("features.accessibility.friendly_buffer_km", 0.5))
    radius_m = buffer_km * 1000.0

    centroids = compute_centroids(subzones[["subzone_key", "geometry"]], projected)
    counts = count_within_radius(centroids, friendly_buildings, radius_m, projected)

    df = pd.DataFrame(
        {
            "subzone_key": centroids["subzone_key"].values,
            "friendly_building_count": counts.astype(int),
        }
    )
    df["friendly_building_count_log"] = np.log1p(df["friendly_building_count"])

    # Convert count -> per km2 density of barrier-free buildings within buffer.
    buffer_area_km2 = np.pi * (buffer_km ** 2)
    df["friendly_building_density"] = df["friendly_building_count"] / buffer_area_km2

    df["accessibility_support_score"] = zscore_series(df["friendly_building_count_log"])

    # National context (constant per subzone). Used in the report discussion.
    national = _latest_active_ageing_count(senior_centres_ts)
    df["national_active_ageing_centres"] = national

    logger.info(
        "Accessibility features: mean friendly buildings within %.1fkm = %.1f, "
        "national AAC count (latest year) = %s",
        buffer_km,
        float(df["friendly_building_count"].mean()),
        national,
    )
    return df


def _latest_active_ageing_count(ts: pd.DataFrame) -> int:
    """Return the most recent non-NA active ageing centre count from the file."""
    if ts is None or ts.empty:
        return 0
    if "Number Of Active Ageing Centres" not in ts.index:
        return 0
    row = ts.loc["Number Of Active Ageing Centres"]
    for year in sorted(row.index, reverse=True):
        v = row[year]
        try:
            v_num = int(float(v))
            return v_num
        except (TypeError, ValueError):
            continue
    return 0
