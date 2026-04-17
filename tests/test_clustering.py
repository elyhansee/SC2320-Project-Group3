"""Unit tests for clustering and evaluation utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from src.modelling.clustering import fit_clustering, suggest_dbscan_eps
from src.modelling.evaluation import (
    choose_best,
    internal_metrics,
    stability_score,
    sweep_clustering,
)


def _blob_df(n: int = 120, k: int = 4, seed: int = 0) -> pd.DataFrame:
    X, _ = make_blobs(n_samples=n, centers=k, random_state=seed)
    return pd.DataFrame(
        {
            "subzone_key": [f"S{i}" for i in range(n)],
            "f1": X[:, 0],
            "f2": X[:, 1],
        }
    )


def test_fit_kmeans() -> None:
    df = _blob_df()
    fit = fit_clustering(df, ["f1", "f2"], algorithm="kmeans", k=4, seed=42)
    assert fit.k == 4
    assert fit.labels.shape[0] == len(df)
    metrics = internal_metrics(fit.X_scaled, fit.labels)
    assert metrics["silhouette"] > 0.3


def test_dbscan_runs_and_eps_suggestor() -> None:
    df = _blob_df()
    fit = fit_clustering(df, ["f1", "f2"], algorithm="dbscan", min_samples=4)
    assert fit.k >= 1
    eps = suggest_dbscan_eps(fit.X_scaled, k_neighbors=4)
    assert eps > 0


def test_sweep_and_choose_best_picks_correct_k() -> None:
    df = _blob_df(k=4)
    sweep = sweep_clustering(
        df,
        ["f1", "f2"],
        algorithms=["kmeans", "agglomerative"],
        k_range=[2, 3, 4, 5, 6],
    )
    best = choose_best(sweep[sweep["algorithm"] != "dbscan"])
    assert int(best["k"]) == 4


def test_stability_returns_high_ari_for_well_separated_blobs() -> None:
    df = _blob_df()
    out = stability_score(df, ["f1", "f2"], k=4, n_bootstrap=10, seed=42)
    assert out["mean_ari"] > 0.7
