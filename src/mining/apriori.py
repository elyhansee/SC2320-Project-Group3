"""Association rule mining for the Apriori phase of the project.

The Apriori implementation is the canonical mlxtend version. Wrapping it
here lets the rest of the codebase pass a binary DataFrame and receive
both the frequent itemsets and the human-readable rules table without
caring about mlxtend's specific column names.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class AprioriResult:
    """Container for one Apriori run."""

    itemsets: pd.DataFrame
    rules: pd.DataFrame
    min_support: float
    min_confidence: float
    min_lift: float


def run_apriori(
    matrix: pd.DataFrame,
    min_support: float = 0.05,
    min_confidence: float = 0.5,
    min_lift: float = 1.1,
    max_len: int = 4,
) -> AprioriResult:
    """Run Apriori on a boolean subzone x item matrix.

    The returned itemsets table has columns ``support`` and ``itemsets``.
    The rules table has the standard mlxtend columns:
    ``antecedents``, ``consequents``, ``support``, ``confidence``, ``lift``,
    ``leverage`` and ``conviction``.
    """
    if matrix.empty:
        raise ValueError("Apriori matrix is empty")
    bool_matrix = matrix.astype(bool)
    itemsets = apriori(
        bool_matrix,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
        low_memory=False,
    )
    if itemsets.empty:
        logger.warning("Apriori produced no frequent itemsets at support=%.3f", min_support)
        rules = pd.DataFrame(
            columns=[
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift",
                "leverage",
                "conviction",
            ]
        )
        return AprioriResult(itemsets, rules, min_support, min_confidence, min_lift)

    rules = association_rules(
        itemsets, metric="confidence", min_threshold=min_confidence
    )
    rules = rules[rules["lift"] >= min_lift].copy()
    rules = rules.sort_values(by=["lift", "confidence"], ascending=[False, False])
    logger.info(
        "Apriori: %d itemsets, %d rules (sup>=%.3f conf>=%.2f lift>=%.2f)",
        len(itemsets),
        len(rules),
        min_support,
        min_confidence,
        min_lift,
    )
    return AprioriResult(itemsets, rules, min_support, min_confidence, min_lift)


def rules_to_table(rules: pd.DataFrame) -> pd.DataFrame:
    """Convert frozenset columns into pretty strings for CSV export."""
    if rules.empty:
        return rules
    out = rules.copy()
    out["antecedents"] = out["antecedents"].apply(_set_to_str)
    out["consequents"] = out["consequents"].apply(_set_to_str)
    return out[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift",
            "leverage",
            "conviction",
        ]
    ]


def support_sweep(
    matrix: pd.DataFrame,
    supports: List[float],
    min_confidence: float,
    min_lift: float,
    max_len: int,
) -> pd.DataFrame:
    """Run Apriori across several min_support values for sensitivity analysis."""
    rows = []
    for s in supports:
        result = run_apriori(
            matrix,
            min_support=s,
            min_confidence=min_confidence,
            min_lift=min_lift,
            max_len=max_len,
        )
        rows.append(
            {
                "min_support": s,
                "n_itemsets": int(len(result.itemsets)),
                "n_rules": int(len(result.rules)),
                "max_lift": float(result.rules["lift"].max()) if len(result.rules) else float("nan"),
                "median_confidence": (
                    float(result.rules["confidence"].median()) if len(result.rules) else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _set_to_str(item) -> str:
    return ", ".join(sorted(map(str, item)))
