"""End-to-end orchestrator for the Spatial-Behavioural Analytics pipeline.

The pipeline is split into three phases that match the project story:

    Phase 1  Association rule mining (Apriori) on a binary subzone matrix
    Phase 2  Unsupervised clustering across K-Means, Agglomerative, DBSCAN
    Phase 3  Random Forest risk model on a transparent proxy target

Running ``python run.py`` reproduces every table and figure used in the
report and the video. Each phase has its own helper so the orchestrator
itself stays short and readable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from src.data.preprocess import build_master_tables
from src.features.assemble import FeatureBundle, build_feature_bundle
from src.features.binary import build_binary_matrix, describe_rules
from src.interpretation.profiles import (
    cluster_profile_table,
    cluster_size_balance,
    label_clusters,
)
from src.mining.apriori import AprioriResult, run_apriori, rules_to_table, support_sweep
from src.modelling.clustering import ClusterFit, fit_clustering, k_distance_curve, scale_features
from src.modelling.evaluation import (
    choose_best,
    internal_metrics,
    stability_score,
    sweep_clustering,
)
from src.modelling.risk_model import (
    RiskModelReport,
    feature_importance_table,
    metrics_to_table,
    train_risk_models,
)
from src.modelling.target import VulnerabilityTarget, build_vulnerability_target
from src.utils.config import Config, load_config
from src.utils.io_utils import save_dataframe, save_geodataframe, save_json
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.seed import set_global_seed
from src.viz.maps import (
    plot_categorical_map,
    plot_choropleth,
    plot_highlight_map,
    plot_overview_map,
    plot_residential_filter_map,
)
from src.viz.plots import (
    plot_amenity_presence,
    plot_apriori_scatter,
    plot_apriori_sweep,
    plot_cluster_profile_heatmap,
    plot_cluster_sizes,
    plot_correlation_heatmap,
    plot_dbscan_kdistance,
    plot_dendrogram,
    plot_elbow,
    plot_elderly_vs_food,
    plot_feature_distributions,
    plot_feature_importance,
    plot_metric_sweep,
    plot_model_comparison,
    plot_pca_scatter,
    plot_score_distribution,
)


def run_pipeline(
    config_path: Path = Path("config/settings.yaml"),
    stage: str = "all",
) -> Dict[str, object]:
    """Run the requested stage(s) of the pipeline."""
    cfg = load_config(config_path)
    set_global_seed(int(cfg.get("project.random_seed", 42)))
    setup_logging(
        yaml_path=cfg.root / "config" / "logging.yaml",
        default_level=getattr(logging, cfg.get("logging.level", "INFO")),
        log_file=cfg.path("logging.logfile"),
    )
    logger = get_logger("pipeline")
    logger.info("=" * 80)
    logger.info("Spatial-Behavioural Analytics pipeline starting (stage=%s)", stage)
    logger.info("=" * 80)

    # 1. Validate input files
    _validate_inputs(cfg, logger)

    # 2. Load + preprocess raw layers
    data = build_master_tables(cfg)

    # 3. Build the canonical feature table
    bundle = build_feature_bundle(cfg, data)
    _persist_feature_table(cfg, bundle, data["subzones"])

    # 4. Data understanding figures
    _data_understanding_figures(cfg, bundle, data)

    if stage in ("features",):
        return {"bundle": bundle}

    # 5. Phase 1 — Apriori
    apriori_result = _run_apriori_phase(cfg, bundle)

    if stage in ("apriori",):
        return {"bundle": bundle, "apriori": apriori_result}

    # 6. Phase 2 — Clustering
    cluster_outputs = _run_clustering_phase(cfg, bundle)

    if stage in ("clustering",):
        return {"bundle": bundle, "clustering": cluster_outputs}

    # 7. Phase 3 — Risk modelling
    target, risk_report = _run_risk_phase(cfg, bundle, cluster_outputs["fit"])

    # 8. Final figures + summary
    _final_figures(cfg, bundle, data, cluster_outputs, target, risk_report, apriori_result)
    summary = _build_summary(
        cfg, bundle, apriori_result, cluster_outputs, target, risk_report
    )
    save_json(summary, cfg.path("paths.outputs_tables") / "run_summary.json")
    logger.info("Pipeline complete. Outputs in %s", cfg.path("paths.outputs_root"))
    return {
        "bundle": bundle,
        "apriori": apriori_result,
        "clustering": cluster_outputs,
        "target": target,
        "risk": risk_report,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_RAW_KEYS = [
    "subzone_boundary",
    "population_csv",
    "hawker_centres",
    "supermarkets",
    "markets",
    "bus_stops",
    "mrt_exits",
    "friendly_buildings",
    "land_use",
    "senior_centres",
]


def _validate_inputs(cfg: Config, logger) -> None:
    missing = []
    for key in _REQUIRED_RAW_KEYS:
        path = cfg.raw_path(key)
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Missing required raw data files:\n  " + "\n  ".join(missing)
        )
    logger.info("All %d required raw files are present", len(_REQUIRED_RAW_KEYS))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _persist_feature_table(cfg: Config, bundle: FeatureBundle, subzones) -> None:
    out_dir = cfg.path("paths.data_processed")
    save_dataframe(bundle.full, out_dir / "subzone_features.csv")
    save_dataframe(bundle.modelling, out_dir / "subzone_features_modelling.csv")
    # Also write a GeoJSON for downstream mapping tools.
    geo = subzones[["subzone_key", "geometry"]].drop_duplicates("subzone_key")
    geo = geo.merge(bundle.full, on="subzone_key", how="inner")
    save_geodataframe(geo, out_dir / "subzone_features.geojson")


# ---------------------------------------------------------------------------
# Data understanding
# ---------------------------------------------------------------------------

def _data_understanding_figures(cfg: Config, bundle: FeatureBundle, data: dict) -> None:
    fig_dir = cfg.path("paths.outputs_figures")
    feature_columns = list(cfg.get("clustering.feature_columns"))
    plot_feature_distributions(
        bundle.modelling, columns=feature_columns,
        out_path=fig_dir / "feature_distributions.png",
        title="Engineered feature distributions",
    )
    plot_correlation_heatmap(
        bundle.modelling, columns=feature_columns,
        out_path=fig_dir / "feature_correlations.png",
        title="Correlation between engineered features",
    )
    plot_overview_map(data["subzones"], fig_dir / "map_overview.png")
    plot_choropleth(
        data["subzones"], bundle.full, "elderly_pct",
        fig_dir / "map_elderly_share.png",
        title="Elderly population share by subzone",
        cmap="YlGnBu",
    )
    plot_choropleth(
        data["subzones"], bundle.full, "elderly_density_log",
        fig_dir / "map_elderly_density.png",
        title="Log elderly density (residents per km2)",
        cmap="YlGnBu",
    )
    plot_choropleth(
        data["subzones"], bundle.full, "food_amenity_count",
        fig_dir / "map_food_amenities.png",
        title="Total food amenities within buffer",
        cmap="YlOrBr",
        points=data["hawkers"],
        point_label="hawker centres",
    )
    plot_choropleth(
        data["subzones"], bundle.full, "supermarket_count",
        fig_dir / "map_supermarkets.png",
        title="Supermarkets within buffer",
        cmap="YlOrBr",
        points=data["supermarkets"],
        point_label="supermarkets",
    )
    plot_choropleth(
        data["subzones"], bundle.full, "market_count",
        fig_dir / "map_markets.png",
        title="Wet markets within buffer",
        cmap="YlOrBr",
        points=data["markets"],
        point_label="markets",
    )
    plot_choropleth(
        data["subzones"], bundle.full, "transit_access_score",
        fig_dir / "map_transit_score.png",
        title="Transit access score (bus + MRT)",
        cmap="PuBuGn",
    )
    plot_choropleth(
        data["subzones"], bundle.full, "accessibility_support_score",
        fig_dir / "map_accessibility_score.png",
        title="Barrier-free accessibility support",
        cmap="BuPu",
    )
    plot_residential_filter_map(
        data["subzones"], bundle.full, fig_dir / "map_residential_filter.png"
    )


# ---------------------------------------------------------------------------
# Phase 1 — Apriori
# ---------------------------------------------------------------------------

def _run_apriori_phase(cfg: Config, bundle: FeatureBundle) -> AprioriResult:
    logger = get_logger("pipeline.apriori")
    out_tables = cfg.path("paths.outputs_tables")
    out_figs = cfg.path("paths.outputs_figures")
    out_proc = cfg.path("paths.data_processed")

    matrix = build_binary_matrix(cfg, bundle.modelling)
    matrix.reset_index().to_csv(out_proc / "binary_amenity_matrix.csv", index=False)

    rules_table = describe_rules(cfg, bundle.modelling)
    save_dataframe(rules_table, out_tables / "binary_rules_definition.csv")

    result = run_apriori(
        matrix,
        min_support=float(cfg.get("mining.apriori_min_support")),
        min_confidence=float(cfg.get("mining.apriori_min_confidence")),
        min_lift=float(cfg.get("mining.apriori_min_lift")),
        max_len=int(cfg.get("mining.apriori_max_len")),
    )
    rules_pretty = rules_to_table(result.rules)
    save_dataframe(rules_pretty, out_tables / "apriori_rules.csv")

    sweep_df = support_sweep(
        matrix,
        list(cfg.get("mining.support_sweep")),
        min_confidence=float(cfg.get("mining.apriori_min_confidence")),
        min_lift=float(cfg.get("mining.apriori_min_lift")),
        max_len=int(cfg.get("mining.apriori_max_len")),
    )
    save_dataframe(sweep_df, out_tables / "apriori_support_sweep.csv")

    plot_apriori_scatter(result.rules, out_figs / "apriori_rules_scatter.png")
    plot_apriori_sweep(sweep_df, out_figs / "apriori_support_sweep.png")
    plot_amenity_presence(matrix, out_figs / "binary_amenity_presence.png")

    logger.info("Apriori phase complete: %d rules", len(result.rules))
    return result


# ---------------------------------------------------------------------------
# Phase 2 — Clustering
# ---------------------------------------------------------------------------

def _run_clustering_phase(cfg: Config, bundle: FeatureBundle) -> Dict[str, object]:
    logger = get_logger("pipeline.clustering")
    out_tables = cfg.path("paths.outputs_tables")
    out_figs = cfg.path("paths.outputs_figures")
    out_models = cfg.path("paths.outputs_models")

    features = bundle.modelling
    feature_cols = list(cfg.get("clustering.feature_columns"))
    seed = int(cfg.get("project.random_seed", 42))
    k_range = list(cfg.get("clustering.k_range"))
    algorithms = list(cfg.get("clustering.algorithms"))
    kmeans_n_init = int(cfg.get("clustering.kmeans_n_init", 25))
    dbscan_min_samples_list = list(cfg.get("clustering.dbscan_min_samples_list"))

    metrics = sweep_clustering(
        features, feature_cols, algorithms, k_range,
        seed=seed, kmeans_n_init=kmeans_n_init,
        dbscan_min_samples_list=dbscan_min_samples_list,
    )
    save_dataframe(metrics, out_tables / "clustering_metrics.csv")

    final_algo_cfg = str(cfg.get("clustering.final_algorithm", "kmeans"))
    final_k_cfg = cfg.get("clustering.final_k")
    if final_k_cfg is None:
        partitional = metrics[metrics["algorithm"] != "dbscan"]
        best = choose_best(partitional)
        final_algo = str(best["algorithm"])
        final_k = int(best["k"])
    else:
        final_algo = final_algo_cfg
        final_k = int(final_k_cfg)
    logger.info("Final clustering: algo=%s k=%d", final_algo, final_k)

    fit: ClusterFit = fit_clustering(
        features,
        feature_cols,
        algorithm=final_algo,
        k=final_k,
        seed=seed,
        kmeans_n_init=kmeans_n_init,
    )
    fit_dbscan: ClusterFit = fit_clustering(
        features, feature_cols, algorithm="dbscan", seed=seed
    )

    final_metrics = internal_metrics(fit.X_scaled, fit.labels)
    stability = stability_score(
        features, feature_cols, k=final_k,
        n_bootstrap=30, seed=seed,
    )
    save_json(
        {"final": final_metrics, "stability": stability,
         "algorithm": final_algo, "k": final_k},
        out_tables / "final_clustering_metrics.json",
    )

    profile = cluster_profile_table(features, feature_cols, fit.labels)
    save_dataframe(profile, out_tables / "cluster_profiles.csv")

    cluster_labels_dict = label_clusters(profile)
    save_json(
        {str(k): v for k, v in cluster_labels_dict.items()},
        out_tables / "cluster_labels.json",
    )

    # Diagnostic figures
    plot_metric_sweep(metrics, out_figs / "clustering_metric_sweep.png")
    plot_elbow(metrics, out_figs / "kmeans_silhouette_vs_k.png")
    k_dists = k_distance_curve(fit.X_scaled, k_neighbors=max(2 * len(feature_cols), 4))
    plot_dbscan_kdistance(k_dists, k_neighbors=max(2 * len(feature_cols), 4),
                          out_path=out_figs / "dbscan_kdistance.png")
    subzone_labels = features["subzone_key"].str.replace("_", " ").str.title().tolist()
    plot_dendrogram(fit.X_scaled, out_figs / "hierarchical_dendrogram.png",
                    labels=subzone_labels)
    plot_cluster_sizes(fit.labels, out_figs / "cluster_sizes.png",
                       title=f"Cluster sizes ({final_algo}, k={final_k})")
    plot_pca_scatter(fit.X_scaled, fit.labels, out_figs / "cluster_pca_scatter.png",
                     title=f"PCA projection ({final_algo}, k={final_k})")
    plot_cluster_profile_heatmap(profile, out_figs / "cluster_profile_heatmap.png")

    joblib.dump(fit.model, out_models / "clustering_model.joblib")
    joblib.dump(fit.scaler, out_models / "clustering_scaler.joblib")

    return {
        "fit": fit,
        "fit_dbscan": fit_dbscan,
        "metrics": metrics,
        "profile": profile,
        "labels_dict": cluster_labels_dict,
        "final_metrics": final_metrics,
        "stability": stability,
        "final_algo": final_algo,
        "final_k": final_k,
    }


# ---------------------------------------------------------------------------
# Phase 3 — Risk modelling
# ---------------------------------------------------------------------------

def _run_risk_phase(
    cfg: Config, bundle: FeatureBundle, cluster_fit: ClusterFit
) -> tuple[VulnerabilityTarget, RiskModelReport]:
    logger = get_logger("pipeline.risk")
    out_tables = cfg.path("paths.outputs_tables")
    out_models = cfg.path("paths.outputs_models")
    seed = int(cfg.get("project.random_seed", 42))

    features = bundle.modelling.copy()

    # The proxy target needs the composite "nearest_food_km" column too.
    target = build_vulnerability_target(cfg, features)
    features["vulnerability_score"] = target.scores.values
    features["vulnerability_label"] = target.binary_label.values
    features["cluster"] = cluster_fit.labels.astype(int)

    # Persist the per-subzone score table.
    score_columns = (
        ["subzone_key", "SUBZONE_N", "PLN_AREA_N", "REGION_N",
         "elderly_pct", "elderly_density", "nearest_food_km",
         "food_access_score", "transit_access_score",
         "accessibility_support_score",
         "vulnerability_score", "vulnerability_label", "cluster"]
    )
    score_columns = [c for c in score_columns if c in features.columns]
    save_dataframe(features[score_columns], out_tables / "subzone_vulnerability_scores.csv")

    # Save the explicit description of the proxy target.
    with (out_tables / "vulnerability_target_definition.txt").open("w", encoding="utf-8") as f:
        f.write(target.description)
    save_dataframe(target.components, out_tables / "vulnerability_target_components.csv")

    # Train the Random Forest and the baselines.
    risk_features = list(cfg.get("clustering.feature_columns"))
    report = train_risk_models(
        cfg, features, risk_features, target.binary_label.values, seed=seed
    )
    metrics_table = metrics_to_table(report)
    save_dataframe(metrics_table, out_tables / "risk_model_metrics.csv")

    importance = feature_importance_table(report)
    save_dataframe(importance, out_tables / "risk_model_feature_importance.csv")

    rf_model = report.results["random_forest"].model
    joblib.dump(rf_model, out_models / "random_forest.joblib")
    logger.info("Risk modelling complete; %d models trained", len(report.results))
    return target, report


# ---------------------------------------------------------------------------
# Final figures
# ---------------------------------------------------------------------------

def _final_figures(
    cfg: Config,
    bundle: FeatureBundle,
    data: dict,
    cluster_outputs: dict,
    target: VulnerabilityTarget,
    risk_report: RiskModelReport,
    apriori_result: AprioriResult,
) -> None:
    fig_dir = cfg.path("paths.outputs_figures")
    out_tables = cfg.path("paths.outputs_tables")

    features = bundle.modelling.copy()
    features["cluster"] = cluster_outputs["fit"].labels.astype(int)
    features["vulnerability_score"] = target.scores.values
    features["vulnerability_label"] = target.binary_label.values

    # Cluster map (categorical)
    plot_categorical_map(
        data["subzones"], features, "cluster",
        fig_dir / "map_clusters.png",
        title=f"Subzone clusters ({cluster_outputs['final_algo']}, k={cluster_outputs['final_k']})",
    )
    # Vulnerability score map
    plot_choropleth(
        data["subzones"], features, "vulnerability_score",
        fig_dir / "map_vulnerability_score.png",
        title="Operational nutritional vulnerability score",
        cmap="YlOrRd",
    )
    # Top-N hotspots
    top_n = int(cfg.get("interpretation.hotspot_top_n", 20)) if cfg.get("interpretation.hotspot_top_n") is not None else 20
    hotspots = features.nlargest(top_n, "vulnerability_score")
    save_dataframe(
        hotspots[
            ["subzone_key", "SUBZONE_N", "PLN_AREA_N", "REGION_N",
             "elderly_pct", "nearest_food_km", "food_access_score",
             "transit_access_score", "accessibility_support_score",
             "vulnerability_score", "cluster"]
        ],
        out_tables / "hotspots_top_n.csv",
    )
    plot_highlight_map(
        data["subzones"], features,
        highlight_keys=hotspots["subzone_key"].tolist(),
        out_path=fig_dir / "map_hotspots.png",
        title=f"Top {top_n} vulnerability hotspots",
    )

    # Score distribution + risk model figures
    plot_score_distribution(
        target.scores, target.quantile_cut, fig_dir / "vulnerability_score_distribution.png"
    )
    metrics_table = metrics_to_table(risk_report)
    plot_model_comparison(metrics_table, fig_dir / "risk_model_comparison.png")
    importance = feature_importance_table(risk_report)
    plot_feature_importance(importance, fig_dir / "rf_feature_importance.png")
    plot_elderly_vs_food(features, target.scores, fig_dir / "elderly_vs_food_scatter.png")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _build_summary(
    cfg: Config,
    bundle: FeatureBundle,
    apriori_result: AprioriResult,
    cluster_outputs: dict,
    target: VulnerabilityTarget,
    risk_report: RiskModelReport,
) -> dict:
    rf = risk_report.results["random_forest"]
    return {
        "n_subzones_total": int(len(bundle.full)),
        "n_subzones_modelling": int(len(bundle.modelling)),
        "feature_columns": bundle.feature_columns,
        "apriori": {
            "min_support": apriori_result.min_support,
            "min_confidence": apriori_result.min_confidence,
            "min_lift": apriori_result.min_lift,
            "n_itemsets": int(len(apriori_result.itemsets)),
            "n_rules": int(len(apriori_result.rules)),
        },
        "clustering": {
            "final_algorithm": cluster_outputs["final_algo"],
            "final_k": cluster_outputs["final_k"],
            "metrics": cluster_outputs["final_metrics"],
            "stability": cluster_outputs["stability"],
            "size_balance": cluster_size_balance(cluster_outputs["fit"].labels),
        },
        "risk_model": {
            "high_risk_quantile": float(cfg.get("risk_model.high_risk_quantile")),
            "n_high_risk": int(target.binary_label.sum()),
            "rf_test_metrics": rf.metrics,
            "rf_cv_metrics": rf.cv_metrics,
        },
    }
