"""Non-map figures: distributions, heatmaps, sweeps, importances."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, linkage  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")


def _savefig(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Data understanding
# ---------------------------------------------------------------------------

def plot_feature_distributions(
    df: pd.DataFrame, columns: List[str], out_path: Path, title: str = "Feature distributions"
) -> Path:
    n = len(columns)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.0))
    axes = np.atleast_1d(axes).flatten()
    for ax, c in zip(axes, columns):
        sns.histplot(df[c], bins=30, ax=ax, kde=True, color="steelblue")
        ax.set_title(c, fontsize=10)
        ax.set_xlabel("")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13, y=1.02)
    return _savefig(fig, out_path)


def plot_correlation_heatmap(
    df: pd.DataFrame, columns: List[str], out_path: Path, title: str = "Feature correlations"
) -> Path:
    corr = df[columns].corr()
    fig, ax = plt.subplots(
        figsize=(max(6, 0.6 * len(columns) + 4), max(5, 0.5 * len(columns) + 3))
    )
    sns.heatmap(
        corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", square=True,
        linewidths=0.5, cbar_kws={"shrink": 0.7}, ax=ax,
    )
    ax.set_title(title)
    return _savefig(fig, out_path)


def plot_amenity_presence(matrix: pd.DataFrame, out_path: Path) -> Path:
    """Bar chart of how often each binary item appears across subzones."""
    rates = matrix.mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(rates) + 2))
    sns.barplot(y=rates.index, x=rates.values, ax=ax, color="steelblue")
    ax.set_xlabel("Share of subzones (presence rate)")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    ax.set_title("Binary amenity presence rates")
    for i, v in enumerate(rates.values):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    return _savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Clustering diagnostics
# ---------------------------------------------------------------------------

def plot_metric_sweep(metrics: pd.DataFrame, out_path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    metric_list = [
        ("silhouette", "Silhouette (higher better)"),
        ("davies_bouldin", "Davies-Bouldin (lower better)"),
        ("calinski_harabasz", "Calinski-Harabasz (higher better)"),
    ]
    partitional = metrics[metrics["algorithm"] != "dbscan"]
    for ax, (col, title) in zip(axes, metric_list):
        for algo, sub in partitional.groupby("algorithm"):
            ax.plot(sub["k"], sub[col], marker="o", label=algo)
        ax.set_title(title)
        ax.set_xlabel("k (number of clusters)")
    axes[0].legend(fontsize=8, loc="best")
    return _savefig(fig, out_path)


def plot_elbow(metrics: pd.DataFrame, out_path: Path) -> Path:
    """Plain k-vs-inertia is replaced with k-vs-silhouette per algorithm."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sub = metrics[metrics["algorithm"] == "kmeans"].sort_values("k")
    ax.plot(sub["k"], sub["silhouette"], marker="o", color="steelblue")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("K-Means silhouette versus k")
    return _savefig(fig, out_path)


def plot_dbscan_kdistance(
    k_dists: np.ndarray, k_neighbors: int, out_path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(np.arange(len(k_dists)), k_dists, color="darkorange")
    ax.set_xlabel("Sorted index")
    ax.set_ylabel(f"k-distance (k={k_neighbors})")
    ax.set_title("DBSCAN k-distance elbow plot")
    return _savefig(fig, out_path)


def plot_dendrogram(
    X: np.ndarray,
    out_path: Path,
    title: str = "Hierarchical clustering dendrogram",
    labels: list | None = None,
) -> Path:
    n = X.shape[0]
    Z = linkage(X, method="ward")
    width = max(28, n * 0.25)
    fig, ax = plt.subplots(figsize=(width, 9))
    dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=max(6, min(9, 1800 / n)),
        ax=ax,
        color_threshold=None,
    )
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Distance", fontsize=11)
    ax.set_xlabel("Subzones", fontsize=11)
    fig.subplots_adjust(bottom=0.35)
    return _savefig(fig, out_path)


def plot_cluster_sizes(labels, out_path: Path, title: str = "Cluster sizes") -> Path:
    series = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.barplot(x=series.index.astype(str), y=series.values, ax=ax, color="steelblue")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of subzones")
    ax.set_title(title)
    for i, v in enumerate(series.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    return _savefig(fig, out_path)


def plot_pca_scatter(
    X: np.ndarray,
    labels,
    out_path: Path,
    title: str = "PCA projection of clusters",
) -> Path:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    unique = np.unique(labels)
    palette = sns.color_palette("tab10", n_colors=max(int(np.max(labels)) + 1, len(unique)))
    for c in unique:
        mask = labels == c
        colour = "lightgrey" if c == -1 else palette[int(c) % len(palette)]
        label = "noise" if c == -1 else f"Cluster {int(c)}"
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            color=colour, label=label, alpha=0.8, edgecolor="white",
            linewidth=0.3, s=42,
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    return _savefig(fig, out_path)


def plot_cluster_profile_heatmap(
    profile: pd.DataFrame, out_path: Path, title: str = "Cluster profiles"
) -> Path:
    plot_df = profile.drop(columns=["cluster_size"], errors="ignore").set_index("cluster")
    z = (plot_df - plot_df.mean()) / plot_df.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(
        figsize=(max(7, 0.8 * len(plot_df.columns) + 3), 0.6 * len(plot_df) + 2)
    )
    sns.heatmap(z, cmap="RdBu_r", center=0, annot=plot_df.round(2), fmt="", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    return _savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Apriori
# ---------------------------------------------------------------------------

def plot_apriori_scatter(rules: pd.DataFrame, out_path: Path) -> Path:
    if rules.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No rules", ha="center", va="center")
        ax.axis("off")
        return _savefig(fig, out_path)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sc = ax.scatter(
        rules["support"], rules["confidence"],
        c=rules["lift"], cmap="viridis", s=60, edgecolor="white",
    )
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Apriori association rules")
    fig.colorbar(sc, ax=ax, label="Lift")
    return _savefig(fig, out_path)


def plot_apriori_sweep(sweep_df: pd.DataFrame, out_path: Path) -> Path:
    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax1.plot(sweep_df["min_support"], sweep_df["n_rules"], marker="o", color="steelblue", label="rules")
    ax1.set_xlabel("min_support")
    ax1.set_ylabel("Number of rules", color="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(sweep_df["min_support"], sweep_df["n_itemsets"], marker="s", color="darkorange", label="itemsets")
    ax2.set_ylabel("Number of frequent itemsets", color="darkorange")
    ax1.set_title("Apriori sensitivity to min_support")
    return _savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Risk model
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance: pd.DataFrame, out_path: Path, title: str = "Random Forest feature importance"
) -> Path:
    if importance.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No importance data", ha="center", va="center")
        ax.axis("off")
        return _savefig(fig, out_path)
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(importance) + 2))
    sns.barplot(
        data=importance,
        x="importance",
        y="feature",
        ax=ax,
        color="steelblue",
    )
    ax.set_title(title)
    return _savefig(fig, out_path)


def plot_model_comparison(metrics: pd.DataFrame, out_path: Path) -> Path:
    cols = [c for c in ("accuracy", "precision", "recall", "f1", "roc_auc") if c in metrics.columns]
    long = metrics.melt(id_vars=["model"], value_vars=cols, var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=long, x="metric", y="value", hue="model", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Risk model comparison (held-out test set)")
    ax.set_ylabel("Score")
    ax.legend(loc="best", fontsize=8)
    return _savefig(fig, out_path)


def plot_elderly_vs_food(
    features: pd.DataFrame,
    risk_score: pd.Series,
    out_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        features["elderly_pct"],
        features["nearest_food_km"],
        c=risk_score,
        cmap="YlOrRd",
        edgecolor="white",
        s=46,
    )
    ax.set_xlabel("Elderly population share")
    ax.set_ylabel("Distance to nearest food amenity (km)")
    ax.set_title("Elderly concentration versus food access")
    fig.colorbar(sc, ax=ax, label="Vulnerability score")
    return _savefig(fig, out_path)


def plot_score_distribution(
    score: pd.Series, cut: float, out_path: Path, title: str = "Vulnerability score distribution"
) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sns.histplot(score, bins=30, kde=True, color="steelblue", ax=ax)
    ax.axvline(cut, color="crimson", linestyle="--", label=f"high-risk cut={cut:.2f}")
    ax.set_xlabel("Vulnerability score")
    ax.legend()
    ax.set_title(title)
    return _savefig(fig, out_path)
