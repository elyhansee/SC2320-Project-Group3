"""Loaders for every raw input file used by the project.

Each loader returns a clean pandas / geopandas object with a consistent
schema that downstream feature builders rely on. Raw schemas vary across
data.gov.sg releases so the loaders are deliberately tolerant: missing
optional columns produce a logged warning rather than a hard error.

Datasets handled here:

    1. subzone_boundary.geojson           (Master Plan 2019 subzones)
    2. ResidentPopulation...csv           (Census 2020)
    3. HawkerCentresGEOJSON.geojson       (NEA hawker centres)
    4. SupermarketsGEOJSON.geojson        (Supermarket licences)
    5. NEAMarketandFoodCentre.geojson     (Markets and food centres)
    6. LTABusStop.geojson                 (LTA bus stops)
    7. LTAMRTStationExitGEOJSON.geojson   (LTA MRT exits)
    8. FriendlyBuildings.geojson          (Barrier free access buildings)
    9. MasterPlan2025LandUseLayer.geojson (Master Plan 2025 land use)
   10. SeniorActivityCentresAndActiveAgeingCentresAnnual.csv
   11. od_bus.csv                         (LTA monthly OD, optional context)
   12. od_train.csv                       (LTA monthly OD, optional context)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional

import geopandas as gpd
import pandas as pd

from src.data.schema import SchemaError, require_columns
from src.utils.config import Config
from src.utils.geo_utils import normalise_name
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_geojson(path: Path, default_crs: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(default_crs)
    return gdf


def _filter_status(
    gdf: gpd.GeoDataFrame,
    status_col: str,
    accept: List[str],
    source: str,
) -> gpd.GeoDataFrame:
    if status_col not in gdf.columns:
        logger.warning(
            "%s: status column '%s' not found, keeping all rows.", source, status_col
        )
        return gdf
    accept_set = {s.casefold() for s in accept}
    before = len(gdf)
    mask = gdf[status_col].astype(str).str.casefold().isin(accept_set)
    out = gdf[mask].copy()
    logger.info(
        "%s: filtered by %s in %s -> %d / %d kept", source, status_col, accept, len(out), before
    )
    return out


# ---------------------------------------------------------------------------
# Subzone boundary
# ---------------------------------------------------------------------------

def load_subzones(cfg: Config) -> gpd.GeoDataFrame:
    """Load Singapore subzone polygons and add a normalised join key."""
    path = cfg.raw_path("subzone_boundary")
    logger.info("Loading subzones from %s", path)
    gdf = _read_geojson(path, cfg.get("crs.geographic"))
    require_columns(gdf, ["SUBZONE_N", "PLN_AREA_N"], source="subzone_boundary")
    gdf["subzone_key"] = normalise_name(gdf["SUBZONE_N"])
    gdf["planning_area_key"] = normalise_name(gdf["PLN_AREA_N"])
    if "REGION_N" not in gdf.columns:
        gdf["REGION_N"] = "UNKNOWN"
    # Dissolve multi-polygons sharing the same subzone key.
    gdf = gdf.dissolve(by=["subzone_key", "planning_area_key"], as_index=False)
    logger.info("Loaded %d subzones", len(gdf))
    return gdf


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

def load_population(cfg: Config) -> pd.DataFrame:
    """Load Census 2020 subzone-level age table.

    The raw file mixes top-level totals, planning-area totals (named
    ``<Area> - Total``) and the actual subzone rows. Aggregate rows are
    dropped and a normalised key is added for joining.
    """
    path = cfg.raw_path("population_csv")
    logger.info("Loading population from %s", path)
    df = pd.read_csv(path)
    name_col = cfg.get("features.population.name_column", "Number")
    if name_col not in df.columns:
        raise SchemaError(f"Population CSV missing name column: {name_col}")
    names = df[name_col].astype(str).str.strip()
    df = df[names != ""].copy()
    df = df[df[name_col].astype(str).str.casefold() != "total"].copy()
    df = df[
        ~df[name_col].astype(str).str.contains(r"\s*-\s*total\s*$", case=False, regex=True, na=False)
    ].copy()
    for c in df.columns:
        if c == name_col:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["subzone_key"] = normalise_name(df[name_col])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Food amenities
# ---------------------------------------------------------------------------

def load_hawkers(cfg: Config) -> gpd.GeoDataFrame:
    """Load hawker centre points filtered to Existing centres."""
    path = cfg.raw_path("hawker_centres")
    logger.info("Loading hawker centres from %s", path)
    gdf = _read_geojson(path, cfg.get("crs.geographic"))
    status_col = cfg.get("features.food_access.hawker_status_column", "STATUS")
    accept = list(cfg.get("features.food_access.hawker_status_accept", ["Existing"]))
    gdf = _filter_status(gdf, status_col, accept, "hawkers")
    return gdf.reset_index(drop=True)


def load_supermarkets(cfg: Config) -> gpd.GeoDataFrame:
    """Load supermarket licence points."""
    path = cfg.raw_path("supermarkets")
    logger.info("Loading supermarkets from %s", path)
    gdf = _read_geojson(path, cfg.get("crs.geographic"))
    return gdf.reset_index(drop=True)


def load_markets(cfg: Config) -> gpd.GeoDataFrame:
    """Load NEA wet markets and food centres.

    The file mixes hawker centres (TYPE == 'HC') with markets (TYPE == 'MK')
    and combinations. Only rows whose TYPE involves a market are returned.
    """
    path = cfg.raw_path("markets")
    logger.info("Loading markets from %s", path)
    gdf = _read_geojson(path, cfg.get("crs.geographic"))
    if "TYPE" in gdf.columns:
        before = len(gdf)
        is_market = gdf["TYPE"].astype(str).str.upper().str.contains("MK", na=False)
        gdf = gdf[is_market].copy()
        logger.info("Markets: %d / %d rows are market-type", len(gdf), before)
    return gdf.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Transit
# ---------------------------------------------------------------------------

def load_bus_stops(cfg: Config) -> gpd.GeoDataFrame:
    """Load LTA bus stop points."""
    path = cfg.raw_path("bus_stops")
    logger.info("Loading bus stops from %s", path)
    gdf = _read_geojson(path, cfg.get("crs.geographic"))
    return gdf.reset_index(drop=True)


def load_mrt_exits(cfg: Config) -> gpd.GeoDataFrame:
    """Load MRT station exit points."""
    path = cfg.raw_path("mrt_exits")
    logger.info("Loading MRT exits from %s", path)
    gdf = _read_geojson(path, cfg.get("crs.geographic"))
    return gdf.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Accessibility support
# ---------------------------------------------------------------------------

def load_friendly_buildings(cfg: Config) -> gpd.GeoDataFrame:
    """Load barrier-free access buildings."""
    path = cfg.raw_path("friendly_buildings")
    logger.info("Loading friendly buildings from %s", path)
    gdf = _read_geojson(path, cfg.get("crs.geographic"))
    return gdf.reset_index(drop=True)


def load_senior_centres(cfg: Config) -> pd.DataFrame:
    """Load the annual senior activity / active ageing centre counts.

    This file does not have geographic coordinates: it is a national-level
    annual time series. It is loaded for context (used in the report and
    discussion) and the most recent year is exposed as a single feature
    appended uniformly to every subzone.
    """
    path = cfg.raw_path("senior_centres")
    logger.info("Loading senior activity centres from %s", path)
    df = pd.read_csv(path)
    if "DataSeries" not in df.columns:
        raise SchemaError("senior_centres CSV missing 'DataSeries' column")
    df = df.set_index("DataSeries")
    return df


# ---------------------------------------------------------------------------
# Land use
# ---------------------------------------------------------------------------

def load_land_use(cfg: Config) -> gpd.GeoDataFrame:
    """Load the Master Plan 2025 land use layer.

    The raw file is large (~190 MB). Only the land-use type and geometry are
    retained. Downstream code intersects this against the subzone polygons
    to compute the residential land area per subzone.
    """
    path = cfg.raw_path("land_use")
    logger.info("Loading land use from %s", path)
    gdf = _read_geojson(path, cfg.get("crs.geographic"))
    if "LU_DESC" not in gdf.columns:
        raise SchemaError("land use file missing LU_DESC")
    return gdf[["LU_DESC", "geometry"]].copy()


# ---------------------------------------------------------------------------
# Origin-destination (loaded only when explicitly enabled)
# ---------------------------------------------------------------------------

_OD_REQUIRED = ["ORIGIN_PT_CODE", "DESTINATION_PT_CODE", "TOTAL_TRIPS"]


def iter_od_chunks(path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    """Yield chunks of an LTA OD CSV with the required columns validated."""
    logger.info("Streaming OD file %s (chunksize=%d)", path, chunksize)
    for chunk in pd.read_csv(path, chunksize=chunksize):
        require_columns(chunk, _OD_REQUIRED, source=str(path.name))
        yield chunk


def aggregate_od(path: Path, chunksize: int) -> pd.DataFrame:
    """Aggregate per-hour OD rows into a single (origin, total_trips) table.

    Only the origin total is needed for context: it answers "how busy is
    each origin stop in a typical month".
    """
    totals: dict = {}
    for chunk in iter_od_chunks(path, chunksize):
        if chunk.empty:
            continue
        grouped = chunk.groupby("ORIGIN_PT_CODE", sort=False)["TOTAL_TRIPS"].sum()
        for k, v in grouped.items():
            totals[str(k)] = totals.get(str(k), 0) + int(v)
    return pd.DataFrame(
        [(k, v) for k, v in totals.items()], columns=["origin", "trips"]
    )


def load_od_context(cfg: Config) -> Optional[pd.DataFrame]:
    """Aggregate od_bus.csv + od_train.csv if enabled and present."""
    if not cfg.get("features.od.enable", False):
        return None
    chunksize = int(cfg.get("features.od.chunksize", 500_000))
    parts = []
    for key in ("od_bus_csv", "od_train_csv"):
        try:
            path = cfg.raw_path(key)
        except Exception:  # noqa: BLE001
            continue
        if not path.exists():
            continue
        parts.append(aggregate_od(path, chunksize=chunksize))
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)
