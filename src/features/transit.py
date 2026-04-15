"""Transit support features at the subzone level.

Counts of bus stops and MRT exits within a configurable walking buffer
around each subzone centroid, plus a composite ``transit_access_score``
that mean-standardises the two counts so they are directly comparable.
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


def build_transit_features(
    cfg: Config,
    subzones: gpd.GeoDataFrame,
    bus_stops: gpd.GeoDataFrame,
    mrt_exits: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Build per-subzone transit features.

    Returns columns:
        subzone_key, bus_stop_count_500m, mrt_exit_count_800m,
        bus_stop_count_log, mrt_exit_count_log, transit_access_score
    """
    projected = cfg.get("crs.projected")
    bus_buffer_km = float(cfg.get("features.transit.bus_buffer_km", 0.5))
    mrt_buffer_km = float(cfg.get("features.transit.mrt_buffer_km", 0.8))
    centroids = compute_centroids(subzones[["subzone_key", "geometry"]], projected)

    bus_counts = count_within_radius(centroids, bus_stops, bus_buffer_km * 1000.0, projected)
    mrt_counts = count_within_radius(centroids, mrt_exits, mrt_buffer_km * 1000.0, projected)

    bus_col = f"bus_stop_count_{int(bus_buffer_km * 1000)}m"
    mrt_col = f"mrt_exit_count_{int(mrt_buffer_km * 1000)}m"

    df = pd.DataFrame(
        {
            "subzone_key": centroids["subzone_key"].values,
            bus_col: bus_counts.astype(int),
            mrt_col: mrt_counts.astype(int),
        }
    )
    df["bus_stop_count_log"] = np.log1p(df[bus_col])
    df["mrt_exit_count_log"] = np.log1p(df[mrt_col])

    # Composite z-score; positive means stronger transit support.
    z = pd.DataFrame(
        {
            "bus_z": zscore_series(df["bus_stop_count_log"]),
            "mrt_z": zscore_series(df["mrt_exit_count_log"]),
        }
    )
    df["transit_access_score"] = z.mean(axis=1)
    # Stable canonical names used by downstream binary feature builders.
    df["bus_stop_count_500m"] = df[bus_col]
    df["mrt_exit_count_800m"] = df[mrt_col]
    logger.info(
        "Transit features: mean bus=%.1f, mean mrt=%.1f within buffers",
        float(df[bus_col].mean()),
        float(df[mrt_col].mean()),
    )
    return df


