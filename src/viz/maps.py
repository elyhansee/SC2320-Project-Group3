"""Choropleth and overlay maps for the report and the video."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _savefig(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _base_axes(figsize=(9, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_overview_map(subzones: gpd.GeoDataFrame, out_path: Path) -> Path:
    """Singapore subzone overview map (no data overlay)."""
    fig, ax = _base_axes()
    subzones.plot(ax=ax, color="whitesmoke", edgecolor="grey", linewidth=0.3)
    if "REGION_N" in subzones.columns:
        for region, sub in subzones.groupby("REGION_N"):
            centroid = sub.geometry.union_all().centroid
            ax.text(
                centroid.x, centroid.y, str(region).title(),
                fontsize=8, ha="center", color="dimgrey",
            )
    ax.set_title("Singapore subzones")
    ax.axis("off")
    return _savefig(fig, out_path)


def plot_choropleth(
    subzones: gpd.GeoDataFrame,
    feature_df: pd.DataFrame,
    column: str,
    out_path: Path,
    title: str,
    cmap: str = "YlOrRd",
    points: Optional[gpd.GeoDataFrame] = None,
    point_label: Optional[str] = None,
) -> Path:
    """Generic per-subzone choropleth with optional point overlay."""
    gdf = subzones.merge(feature_df[["subzone_key", column]], on="subzone_key", how="left")
    fig, ax = _base_axes()
    gdf.plot(
        column=column,
        cmap=cmap,
        legend=True,
        legend_kwds={"shrink": 0.6, "label": column},
        linewidth=0.2,
        edgecolor="grey",
        missing_kwds={"color": "lightgrey", "label": "No data"},
        ax=ax,
    )
    if points is not None and len(points) > 0:
        points.to_crs(subzones.crs).plot(
            ax=ax,
            markersize=8,
            color="black",
            alpha=0.55,
            label=point_label or "amenity",
        )
        ax.legend(loc="lower left", fontsize=8)
    ax.set_title(title)
    ax.axis("off")
    return _savefig(fig, out_path)


def plot_categorical_map(
    subzones: gpd.GeoDataFrame,
    feature_df: pd.DataFrame,
    column: str,
    out_path: Path,
    title: str,
    cmap: str = "tab10",
) -> Path:
    gdf = subzones.merge(feature_df[["subzone_key", column]], on="subzone_key", how="left")
    fig, ax = _base_axes()
    gdf.plot(
        column=column,
        cmap=cmap,
        categorical=True,
        legend=True,
        legend_kwds={"loc": "upper right", "title": column, "fontsize": 9},
        linewidth=0.2,
        edgecolor="grey",
        missing_kwds={"color": "lightgrey", "label": "No data"},
        ax=ax,
    )
    ax.set_title(title)
    ax.axis("off")
    return _savefig(fig, out_path)


def plot_highlight_map(
    subzones: gpd.GeoDataFrame,
    feature_df: pd.DataFrame,
    highlight_keys: Iterable[str],
    out_path: Path,
    title: str,
    colour: str = "crimson",
) -> Path:
    """Greyscale base map with highlighted subzones (e.g. top-N hotspots)."""
    gdf = subzones.merge(feature_df[["subzone_key"]], on="subzone_key", how="left")
    gdf["__highlight"] = gdf["subzone_key"].isin(set(highlight_keys))
    fig, ax = _base_axes()
    gdf.plot(ax=ax, color="whitesmoke", edgecolor="grey", linewidth=0.2)
    gdf[gdf["__highlight"]].plot(
        ax=ax, color=colour, edgecolor="black", linewidth=0.4
    )
    ax.set_title(title)
    ax.axis("off")
    return _savefig(fig, out_path)


def plot_residential_filter_map(
    subzones: gpd.GeoDataFrame,
    feature_df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Show which subzones survive the residential land-use filter."""
    gdf = subzones.merge(
        feature_df[["subzone_key", "is_residential", "residential_share"]],
        on="subzone_key",
        how="left",
    )
    gdf["status"] = np.where(gdf["is_residential"].fillna(False), "residential", "non-residential")
    fig, ax = _base_axes()
    gdf.plot(
        column="status",
        cmap="Set2",
        categorical=True,
        legend=True,
        linewidth=0.2,
        edgecolor="grey",
        missing_kwds={"color": "lightgrey", "label": "No data"},
        ax=ax,
        legend_kwds={"loc": "upper right", "title": "Land use", "fontsize": 9},
    )
    ax.set_title("Residential filter (Master Plan 2025 land use)")
    ax.axis("off")
    return _savefig(fig, out_path)
