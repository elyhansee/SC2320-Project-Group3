"""Geospatial helpers built on GeoPandas and Shapely.

All metric distances are computed in the projected CRS declared in the config
(EPSG:3414, Singapore TM). Inputs are accepted in any CRS and reprojected
on the fly.
"""
from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


# ---------------------------------------------------------------------------
# CRS helpers
# ---------------------------------------------------------------------------

def ensure_crs(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    """Project ``gdf`` to ``crs`` if it is not already in it."""
    if gdf.crs is None:
        return gdf.set_crs(crs)
    if str(gdf.crs).upper() == str(crs).upper():
        return gdf
    return gdf.to_crs(crs)


def to_gdf_points(
    df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Turn a DataFrame with lon/lat into a GeoDataFrame of points."""
    geometry = [Point(xy) for xy in zip(df[lon_col].values, df[lat_col].values)]
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=crs)


# ---------------------------------------------------------------------------
# Centroids and area
# ---------------------------------------------------------------------------

def compute_centroids(
    polygons: gpd.GeoDataFrame, projected_crs: str
) -> gpd.GeoDataFrame:
    """Compute metric-safe centroids by projecting first."""
    projected = ensure_crs(polygons, projected_crs).copy()
    projected["centroid"] = projected.geometry.centroid
    return projected.set_geometry("centroid")


def compute_area_km2(polygons: gpd.GeoDataFrame, projected_crs: str) -> np.ndarray:
    """Return the area of each polygon in km2 using the projected CRS."""
    projected = ensure_crs(polygons, projected_crs)
    return (projected.geometry.area / 1_000_000.0).to_numpy()


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _xy(gdf: gpd.GeoDataFrame) -> np.ndarray:
    return np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])


def _pairwise_distance_matrix(
    origins: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    projected_crs: str,
) -> np.ndarray:
    """Return the dense origin x target distance matrix in metres."""
    o_p = ensure_crs(origins, projected_crs)
    t_p = ensure_crs(targets, projected_crs)
    if len(t_p) == 0:
        return np.empty((len(o_p), 0))
    o_xy = _xy(o_p)
    t_xy = _xy(t_p)
    diffs = o_xy[:, None, :] - t_xy[None, :, :]
    return np.sqrt((diffs ** 2).sum(axis=2))


def nearest_distance_km(
    origins: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    projected_crs: str,
) -> np.ndarray:
    """Return an array of nearest-neighbour distances (km) for each origin."""
    dists = _pairwise_distance_matrix(origins, targets, projected_crs)
    if dists.shape[1] == 0:
        return np.full(dists.shape[0], np.nan)
    return dists.min(axis=1) / 1000.0


def count_within_radius(
    origins: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    radius_m: float,
    projected_crs: str,
) -> np.ndarray:
    """Return an array of counts of targets within ``radius_m`` for each origin."""
    dists = _pairwise_distance_matrix(origins, targets, projected_crs)
    if dists.shape[1] == 0:
        return np.zeros(dists.shape[0], dtype=int)
    return (dists <= radius_m).sum(axis=1).astype(int)


# ---------------------------------------------------------------------------
# Polygon to subzone joins
# ---------------------------------------------------------------------------

def points_in_polygons(
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    polygon_id_col: str,
    projected_crs: Optional[str] = None,
) -> pd.Series:
    """Return the polygon id that each point falls inside, or NaN."""
    pts = points.copy()
    polys = polygons.copy()
    if projected_crs is not None:
        pts = ensure_crs(pts, projected_crs)
        polys = ensure_crs(polys, projected_crs)
    joined = gpd.sjoin(
        pts,
        polys[[polygon_id_col, "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined.loc[~joined.index.duplicated(keep="first")]
    return joined[polygon_id_col]


def polygon_overlap_area_km2(
    target: gpd.GeoDataFrame,
    overlay: gpd.GeoDataFrame,
    target_id: str,
    projected_crs: str,
) -> pd.DataFrame:
    """Return total overlap area (km2) of ``overlay`` features per target polygon.

    Used for the residential land area feature against the Master Plan layer.
    """
    tgt = ensure_crs(target, projected_crs).copy()
    ovl = ensure_crs(overlay, projected_crs).copy()
    inter = gpd.overlay(
        tgt[[target_id, "geometry"]],
        ovl[["geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    if inter.empty:
        return pd.DataFrame({target_id: tgt[target_id], "overlap_km2": 0.0})
    inter["overlap_km2"] = inter.geometry.area / 1_000_000.0
    grouped = inter.groupby(target_id, as_index=False)["overlap_km2"].sum()
    full = tgt[[target_id]].merge(grouped, on=target_id, how="left").fillna({"overlap_km2": 0.0})
    return full


# ---------------------------------------------------------------------------
# Name normalisation (used everywhere a subzone string needs to join)
# ---------------------------------------------------------------------------

def normalise_name(series: pd.Series) -> pd.Series:
    """Upper-case, collapse whitespace and trim a name column for joining."""
    return (
        series.astype(str)
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
