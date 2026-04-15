"""Master Plan 2025 land use features at the subzone level.

The Master Plan polygons are intersected with the subzone polygons in
metric coordinates so the residential area per subzone can be measured.
This is used to flag subzones that are clearly non-residential (e.g.
industrial estates, water bodies, the central catchment) and exclude
them from the modelling step where elderly food vulnerability is not
meaningful.
"""
from __future__ import annotations

from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.geo_utils import ensure_crs, polygon_overlap_area_km2
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_landuse_features(
    cfg: Config,
    subzones: gpd.GeoDataFrame,
    land_use: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Build per-subzone land-use features.

    Returns columns:
        subzone_key, residential_area_km2, residential_share, is_residential
    """
    projected = cfg.get("crs.projected")
    keywords: List[str] = list(cfg.get("features.land_use.residential_keywords"))
    keyword_set = {k.upper() for k in keywords}

    lu = ensure_crs(land_use, projected).copy()
    mask = lu["LU_DESC"].astype(str).str.upper().isin(keyword_set)
    residential = lu[mask].copy()
    logger.info(
        "Filtered land use to residential: %d / %d polygons", len(residential), len(lu)
    )

    sz = ensure_crs(subzones[["subzone_key", "geometry"]], projected).copy()
    sz["subzone_area_km2"] = sz.geometry.area / 1_000_000.0

    overlap = polygon_overlap_area_km2(
        target=sz, overlay=residential, target_id="subzone_key", projected_crs=projected
    )
    out = sz[["subzone_key", "subzone_area_km2"]].merge(
        overlap.rename(columns={"overlap_km2": "residential_area_km2"}),
        on="subzone_key",
        how="left",
    )
    out["residential_area_km2"] = out["residential_area_km2"].fillna(0.0)
    out["residential_share"] = np.where(
        out["subzone_area_km2"] > 0,
        out["residential_area_km2"] / out["subzone_area_km2"],
        0.0,
    )

    min_area_km2 = float(cfg.get("features.land_use.min_residential_area_m2", 5000)) / 1_000_000.0
    min_share = float(cfg.get("features.land_use.min_residential_share", 0.05))
    out["is_residential"] = (
        (out["residential_area_km2"] >= min_area_km2) & (out["residential_share"] >= min_share)
    )
    logger.info(
        "Residential subzones: %d / %d (min area=%.4f km2, min share=%.2f)",
        int(out["is_residential"].sum()),
        len(out),
        min_area_km2,
        min_share,
    )
    return out[
        ["subzone_key", "residential_area_km2", "residential_share", "is_residential"]
    ]
