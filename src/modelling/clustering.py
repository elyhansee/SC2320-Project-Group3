"""Clustering algorithms, scaling and DBSCAN auto-tuning.

This module is intentionally thin: every algorithm is wrapped in a single
function so that the sweep code in :mod:`src.modelling.evaluation` can
treat each method uniformly. DBSCAN's noise label ``-1`` is preserved
through the pipeline so downstream interpretation can flag outliers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ClusterFit:
    """Container for a single clustering result."""

    name: str
    k: int
    labels: np.ndarray
    model: object
    scaler: StandardScaler
    feature_columns: List[str]
    X_scaled: np.ndarray
    params: dict


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def scale_features(
    df: pd.DataFrame, feature_columns: List[str]
) -> Tuple[np.ndarray, StandardScaler]:
    """Standardise the requested feature columns."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_columns].astype(float).values)
    return X, scaler


# ---------------------------------------------------------------------------
# Single-fit entry point
# ---------------------------------------------------------------------------

def fit_clustering(
    df: pd.DataFrame,
    feature_columns: List[str],
    algorithm: str,
    k: Optional[int] = None,
    eps: Optional[float] = None,
    min_samples: Optional[int] = None,
    seed: int = 42,
    kmeans_n_init: int = 25,
) -> ClusterFit:
    """Fit one clustering configuration and return a :class:`ClusterFit`."""
    X, scaler = scale_features(df, feature_columns)
    algorithm = algorithm.lower()
    params: dict = {}

    if algorithm == "kmeans":
        if k is None:
            raise ValueError("kmeans requires k")
        model = KMeans(n_clusters=k, n_init=kmeans_n_init, random_state=seed)
        labels = model.fit_predict(X)
        params = {"k": int(k)}
    elif algorithm == "agglomerative":
        if k is None:
            raise ValueError("agglomerative requires k")
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X)
        params = {"k": int(k)}
    elif algorithm == "dbscan":
        if min_samples is None:
            min_samples = max(2 * X.shape[1], 4)
        if eps is None:
            eps = suggest_dbscan_eps(X, k_neighbors=min_samples)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        params = {"eps": float(eps), "min_samples": int(min_samples)}
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")

    n_clusters = int(len(set(labels) - {-1}))
    logger.info(
        "Fitted %s on %d rows x %d features -> %d clusters (params=%s)",
        algorithm,
        X.shape[0],
        X.shape[1],
        n_clusters,
        params,
    )
    return ClusterFit(
        name=algorithm,
        k=n_clusters,
        labels=labels,
        model=model,
        scaler=scaler,
        feature_columns=list(feature_columns),
        X_scaled=X,
        params=params,
    )


# ---------------------------------------------------------------------------
# DBSCAN eps auto-tuning via the k-distance elbow heuristic
# ---------------------------------------------------------------------------

def suggest_dbscan_eps(X: np.ndarray, k_neighbors: int = 4) -> float:
    """Suggest an eps via the k-distance elbow heuristic.

    Sort the k-th nearest neighbour distance of every point in ascending
    order and return the value at the maximum-curvature knee.
    """
    k_neighbors = min(k_neighbors, max(1, X.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X)
    dists, _ = nn.kneighbors(X)
    k_dists = np.sort(dists[:, -1])
    if k_dists.size < 4:
        return float(k_dists[-1])
    second = np.diff(np.diff(k_dists))
    elbow = int(np.argmax(second)) + 1
    eps = float(k_dists[elbow])
    lo, hi = float(np.quantile(k_dists, 0.25)), float(np.quantile(k_dists, 0.95))
    eps = float(np.clip(eps, lo, hi))
    logger.info("Suggested DBSCAN eps=%.3f (k=%d, knee idx=%d)", eps, k_neighbors, elbow)
    return eps


def k_distance_curve(X: np.ndarray, k_neighbors: int = 4) -> np.ndarray:
    """Return the sorted k-distance curve used by the DBSCAN elbow plot."""
    k_neighbors = min(k_neighbors, max(1, X.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X)
    dists, _ = nn.kneighbors(X)
    return np.sort(dists[:, -1])
