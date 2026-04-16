"""Clustering evaluation: internal metrics, sweeps and stability."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.modelling.clustering import fit_clustering, scale_features
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal metrics
# ---------------------------------------------------------------------------

def internal_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute silhouette, Davies-Bouldin and Calinski-Harabasz.

    DBSCAN noise points (label ``-1``) are excluded from the metrics but
    counted as ``noise_frac``. Returns NaN metrics when fewer than two
    non-noise clusters remain.
    """
    mask = labels != -1
    noise_frac = float(np.mean(~mask))
    valid_labels = labels[mask]
    unique = np.unique(valid_labels)
    out: Dict[str, float] = {"noise_frac": noise_frac}
    if len(unique) < 2 or mask.sum() < len(unique) + 1:
        out.update({"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan})
        return out
    Xv = X[mask]
    out.update(
        {
            "silhouette": float(silhouette_score(Xv, valid_labels)),
            "davies_bouldin": float(davies_bouldin_score(Xv, valid_labels)),
            "calinski_harabasz": float(calinski_harabasz_score(Xv, valid_labels)),
        }
    )
    return out


# ---------------------------------------------------------------------------
# Sweep across algorithms and hyper-parameters
# ---------------------------------------------------------------------------

def sweep_clustering(
    df: pd.DataFrame,
    feature_columns: List[str],
    algorithms: List[str],
    k_range: List[int],
    seed: int = 42,
    kmeans_n_init: int = 25,
    dbscan_min_samples_list: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Sweep every algorithm across its natural hyper-parameter grid."""
    records = []
    for algo in algorithms:
        if algo == "dbscan":
            ms_list = dbscan_min_samples_list or [max(2 * len(feature_columns), 4)]
            for ms in ms_list:
                fit = fit_clustering(
                    df,
                    feature_columns,
                    algorithm="dbscan",
                    min_samples=ms,
                    seed=seed,
                )
                metrics = internal_metrics(fit.X_scaled, fit.labels)
                records.append(
                    {
                        "algorithm": "dbscan",
                        "k": fit.k,
                        "eps": fit.params.get("eps"),
                        "min_samples": fit.params.get("min_samples"),
                        **metrics,
                        "cluster_sizes": str(
                            [int((fit.labels == c).sum()) for c in np.unique(fit.labels)]
                        ),
                    }
                )
        else:
            for k in k_range:
                fit = fit_clustering(
                    df,
                    feature_columns,
                    algorithm=algo,
                    k=k,
                    seed=seed,
                    kmeans_n_init=kmeans_n_init,
                )
                metrics = internal_metrics(fit.X_scaled, fit.labels)
                records.append(
                    {
                        "algorithm": algo,
                        "k": k,
                        "eps": None,
                        "min_samples": None,
                        **metrics,
                        "cluster_sizes": str(
                            [int((fit.labels == c).sum()) for c in np.unique(fit.labels)]
                        ),
                    }
                )
    df_metrics = pd.DataFrame.from_records(records)
    logger.info("Swept %d clustering configurations", len(df_metrics))
    return df_metrics


def choose_best(metrics: pd.DataFrame, min_clusters: int = 2) -> pd.Series:
    """Choose the best (algorithm, k) by mean rank over the three metrics."""
    dfm = metrics.dropna(subset=["silhouette", "davies_bouldin", "calinski_harabasz"]).copy()
    dfm = dfm[dfm["k"] >= min_clusters]
    if dfm.empty:
        return metrics.iloc[0]
    dfm["silhouette_rank"] = dfm["silhouette"].rank(ascending=False)
    dfm["calinski_harabasz_rank"] = dfm["calinski_harabasz"].rank(ascending=False)
    dfm["davies_bouldin_rank"] = dfm["davies_bouldin"].rank(ascending=True)
    dfm["mean_rank"] = dfm[
        ["silhouette_rank", "calinski_harabasz_rank", "davies_bouldin_rank"]
    ].mean(axis=1)
    best = dfm.sort_values(["mean_rank", "silhouette"], ascending=[True, False]).iloc[0]
    return best


# ---------------------------------------------------------------------------
# Bootstrap stability
# ---------------------------------------------------------------------------

def stability_score(
    df: pd.DataFrame,
    feature_columns: List[str],
    k: int,
    n_bootstrap: int = 30,
    seed: int = 42,
    subsample_frac: float = 0.8,
) -> Dict[str, float]:
    """Measure clustering stability via ARI between bootstrap resamples."""
    if k < 2:
        return {"mean_ari": float("nan"), "std_ari": float("nan"), "n_pairs": 0}
    rng = np.random.default_rng(seed)
    X_full, _ = scale_features(df, feature_columns)
    n = X_full.shape[0]
    sample_size = max(int(n * subsample_frac), k + 1)

    prev_idx = prev_labels = None
    aris = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=sample_size, replace=False)
        model = KMeans(n_clusters=k, n_init=20, random_state=int(rng.integers(0, 1_000_000)))
        labels = model.fit_predict(X_full[idx])
        if prev_idx is not None:
            common = np.intersect1d(prev_idx, idx)
            if len(common) >= k + 1:
                a = _lookup_labels(prev_idx, prev_labels, common)
                b = _lookup_labels(idx, labels, common)
                aris.append(adjusted_rand_score(a, b))
        prev_idx, prev_labels = idx, labels

    if not aris:
        return {"mean_ari": float("nan"), "std_ari": float("nan"), "n_pairs": 0}
    return {
        "mean_ari": float(np.mean(aris)),
        "std_ari": float(np.std(aris)),
        "n_pairs": len(aris),
    }


def _lookup_labels(idx: np.ndarray, labels: np.ndarray, common: np.ndarray) -> np.ndarray:
    pos = {i: p for p, i in enumerate(idx)}
    return np.array([labels[pos[i]] for i in common])
