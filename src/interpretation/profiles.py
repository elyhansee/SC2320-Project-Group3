"""Cluster profile and labelling helpers used after the clustering phase."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def cluster_profile_table(
    features: pd.DataFrame,
    feature_columns: List[str],
    labels: np.ndarray,
) -> pd.DataFrame:
    """Mean of each feature column per cluster, plus the cluster size."""
    df = features[feature_columns].copy()
    df["cluster"] = labels.astype(int)
    profile = df.groupby("cluster")[feature_columns].mean()
    profile["cluster_size"] = df.groupby("cluster").size()
    return profile.reset_index()


def label_clusters(profile: pd.DataFrame) -> Dict[int, str]:
    """Assign a short interpretive label to each cluster.

    The label is derived from the relative position of three signal
    columns: ``elderly_pct``, ``food_access_score``,
    ``transit_access_score``. Each is binarised against the median across
    clusters and the resulting tag is concatenated.
    """
    cols_required = ["elderly_pct", "food_access_score", "transit_access_score"]
    if not all(c in profile.columns for c in cols_required):
        return {int(c): f"Cluster {int(c)}" for c in profile["cluster"].unique()}

    medians = profile[cols_required].median()
    out: Dict[int, str] = {}
    for _, row in profile.iterrows():
        elderly = "high-elderly" if row["elderly_pct"] >= medians["elderly_pct"] else "low-elderly"
        food = (
            "rich-food"
            if row["food_access_score"] >= medians["food_access_score"]
            else "poor-food"
        )
        transit = (
            "rich-transit"
            if row["transit_access_score"] >= medians["transit_access_score"]
            else "poor-transit"
        )
        out[int(row["cluster"])] = f"{elderly} | {food} | {transit}"
    return out


def cluster_size_balance(labels: np.ndarray) -> Dict[str, float]:
    """Return min/max/mean cluster sizes for the report."""
    counts = pd.Series(labels).value_counts()
    return {
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "n_clusters": int(len(counts)),
    }
