"""High-level preprocessing that materialises every clean intermediate table."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import geopandas as gpd
import pandas as pd

from src.data.loaders import (
    load_bus_stops,
    load_friendly_buildings,
    load_hawkers,
    load_land_use,
    load_markets,
    load_mrt_exits,
    load_od_context,
    load_population,
    load_senior_centres,
    load_subzones,
    load_supermarkets,
)
from src.utils.config import Config
from src.utils.io_utils import save_geodataframe
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_master_tables(cfg: Config) -> Dict[str, object]:
    """Load every raw input and cache the cleaned objects.

    Returns a dictionary that downstream feature builders can index by name.
    """
    interim = cfg.path("paths.data_interim")
    interim.mkdir(parents=True, exist_ok=True)

    subzones = load_subzones(cfg)
    population = load_population(cfg)
    hawkers = load_hawkers(cfg)
    supermarkets = load_supermarkets(cfg)
    markets = load_markets(cfg)
    bus_stops = load_bus_stops(cfg)
    mrt_exits = load_mrt_exits(cfg)
    friendly = load_friendly_buildings(cfg)
    land_use = load_land_use(cfg)
    senior_ts = load_senior_centres(cfg)
    od_context = load_od_context(cfg)

    # Cache the clean point and polygon layers as GeoJSON for inspection.
    _cache(subzones, interim / "subzones.geojson")
    _cache(hawkers, interim / "hawkers.geojson")
    _cache(supermarkets, interim / "supermarkets.geojson")
    _cache(markets, interim / "markets.geojson")
    _cache(bus_stops, interim / "bus_stops.geojson")
    _cache(mrt_exits, interim / "mrt_exits.geojson")
    _cache(friendly, interim / "friendly_buildings.geojson")

    population.to_csv(interim / "population_clean.csv", index=False)
    senior_ts.to_csv(interim / "senior_centres_timeseries.csv")

    return {
        "subzones": subzones,
        "population": population,
        "hawkers": hawkers,
        "supermarkets": supermarkets,
        "markets": markets,
        "bus_stops": bus_stops,
        "mrt_exits": mrt_exits,
        "friendly_buildings": friendly,
        "land_use": land_use,
        "senior_centres_ts": senior_ts,
        "od_context": od_context,
    }


def _cache(gdf: gpd.GeoDataFrame, path: Path) -> None:
    if gdf is None or len(gdf) == 0:
        return
    save_geodataframe(gdf, path)
