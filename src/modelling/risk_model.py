"""Random Forest risk model and baselines for the proxy vulnerability label.

The Random Forest is the headline model. Two simpler baselines provide a
sanity check and a fairness comparison:

    1. Majority-class predictor (lower bound on usefulness)
    2. Logistic regression on the standardised feature matrix
    3. Decision tree at depth 4 (interpretable baseline)

Because the target is constructed by a known formula (see
:mod:`src.modelling.target`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.utils.config import Config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ModelResult:
    """Container for one fitted classifier."""

    name: str
    model: object
    metrics: Dict[str, float]
    cv_metrics: Dict[str, float]
    feature_importances: Dict[str, float] = field(default_factory=dict)


@dataclass
class RiskModelReport:
    """Bundle of every model trained for the risk modelling phase."""

    feature_columns: List[str]
    train_index: np.ndarray
    test_index: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    results: Dict[str, ModelResult]


def train_risk_models(
    cfg: Config,
    features: pd.DataFrame,
    feature_columns: List[str],
    target: np.ndarray,
    seed: int = 42,
) -> RiskModelReport:
    """Train the Random Forest and the baselines on the proxy target."""
    if len(np.unique(target)) < 2:
        raise ValueError("Target has only one class. Adjust the high-risk quantile.")
    X = features[feature_columns].astype(float).values
    y = np.asarray(target).astype(int)

    test_size = float(cfg.get("risk_model.test_size", 0.25))
    cv_folds = int(cfg.get("risk_model.cv_folds", 5))
    n_estimators = int(cfg.get("risk_model.rf_n_estimators", 400))
    max_depth_raw = cfg.get("risk_model.rf_max_depth", None)
    max_depth = None if max_depth_raw in (None, "null", "None") else int(max_depth_raw)
    min_samples_leaf = int(cfg.get("risk_model.rf_min_samples_leaf", 2))

    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_tr, y_tr)
    rf_metrics = _eval_classifier(rf, X_te, y_te)
    rf_cv = _cv_metrics(rf, X, y, seed=seed, cv_folds=cv_folds)
    rf_importance = dict(zip(feature_columns, rf.feature_importances_))

    logreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ]
    )
    logreg.fit(X_tr, y_tr)
    logreg_metrics = _eval_classifier(logreg, X_te, y_te)
    logreg_cv = _cv_metrics(logreg, X, y, seed=seed, cv_folds=cv_folds)

    tree = DecisionTreeClassifier(max_depth=4, random_state=seed, class_weight="balanced")
    tree.fit(X_tr, y_tr)
    tree_metrics = _eval_classifier(tree, X_te, y_te)
    tree_cv = _cv_metrics(tree, X, y, seed=seed, cv_folds=cv_folds)
    tree_importance = dict(zip(feature_columns, tree.feature_importances_))

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_tr, y_tr)
    dummy_metrics = _eval_classifier(dummy, X_te, y_te)
    dummy_cv = _cv_metrics(dummy, X, y, seed=seed, cv_folds=cv_folds)

    results: Dict[str, ModelResult] = {
        "random_forest": ModelResult("random_forest", rf, rf_metrics, rf_cv, rf_importance),
        "logistic_regression": ModelResult(
            "logistic_regression", logreg, logreg_metrics, logreg_cv
        ),
        "decision_tree": ModelResult(
            "decision_tree", tree, tree_metrics, tree_cv, tree_importance
        ),
        "dummy_majority": ModelResult("dummy_majority", dummy, dummy_metrics, dummy_cv),
    }
    return RiskModelReport(
        feature_columns=list(feature_columns),
        train_index=train_idx,
        test_index=test_idx,
        y_train=y_tr,
        y_test=y_te,
        results=results,
    )


# ---------------------------------------------------------------------------
# Helper evaluators
# ---------------------------------------------------------------------------

def _eval_classifier(model, X_te: np.ndarray, y_te: np.ndarray) -> Dict[str, float]:
    pred = model.predict(X_te)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
    }
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_te)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_te, proba))
        except Exception:  # noqa: BLE001
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def _cv_metrics(
    model, X: np.ndarray, y: np.ndarray, seed: int, cv_folds: int
) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    out: Dict[str, float] = {}
    for metric in ("accuracy", "f1", "roc_auc"):
        try:
            scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
            out[f"cv_{metric}_mean"] = float(np.mean(scores))
            out[f"cv_{metric}_std"] = float(np.std(scores))
        except Exception as exc:  # noqa: BLE001
            logger.warning("CV failed for metric %s: %s", metric, exc)
            out[f"cv_{metric}_mean"] = float("nan")
            out[f"cv_{metric}_std"] = float("nan")
    return out


def metrics_to_table(report: RiskModelReport) -> pd.DataFrame:
    """Flatten the report into a long table for CSV export."""
    rows = []
    for name, result in report.results.items():
        row = {"model": name}
        row.update(result.metrics)
        row.update(result.cv_metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def feature_importance_table(report: RiskModelReport) -> pd.DataFrame:
    """Return the Random Forest feature importance table."""
    rf = report.results.get("random_forest")
    if rf is None:
        return pd.DataFrame()
    rows = [
        {"feature": k, "importance": float(v)} for k, v in rf.feature_importances.items()
    ]
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
