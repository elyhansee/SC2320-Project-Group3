"""Binary amenity / vulnerability matrix used as the Apriori input.

Every binary feature is derived from a single real feature column with a
clear, configurable threshold. The thresholds are stored in
``settings.yaml`` so the report can describe each rule literally as
"feature X exceeds quantile Y".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BinaryRule:
    """Description of how a binary item is derived from a continuous feature."""

    name: str
    column: str
    op: str        # one of: ge_quantile, le_quantile, ge_value
    threshold: float
    description: str


def build_binary_matrix(cfg: Config, features: pd.DataFrame) -> pd.DataFrame:
    """Materialise the binary subzone x item matrix used by Apriori.

    Returns a DataFrame indexed by ``subzone_key`` with one boolean column
    per item. The same DataFrame is also written to disk as the input to
    ``mlxtend.frequent_patterns.apriori``.
    """
    rules = _build_rules(cfg, features)
    cols: Dict[str, np.ndarray] = {}
    for rule in rules:
        if rule.column not in features.columns:
            logger.warning("Binary rule references missing column %s", rule.column)
            continue
        values = features[rule.column].astype(float).values
        if rule.op == "ge_quantile":
            cut = float(np.nanquantile(values, rule.threshold))
            cols[rule.name] = values >= cut
        elif rule.op == "le_quantile":
            cut = float(np.nanquantile(values, rule.threshold))
            cols[rule.name] = values <= cut
        elif rule.op == "ge_value":
            cols[rule.name] = values >= float(rule.threshold)
        else:
            raise ValueError(f"Unknown binary op: {rule.op}")
    matrix = pd.DataFrame(cols, index=features["subzone_key"]).astype(bool)
    matrix.index.name = "subzone_key"
    logger.info(
        "Binary matrix built: %d subzones x %d items, mean density=%.3f",
        matrix.shape[0],
        matrix.shape[1],
        float(matrix.values.mean()),
    )
    return matrix


def describe_rules(cfg: Config, features: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame describing each binary rule for the report."""
    rules = _build_rules(cfg, features)
    rows = []
    for rule in rules:
        if rule.column not in features.columns:
            continue
        values = features[rule.column].astype(float).values
        if rule.op == "ge_quantile":
            cut = float(np.nanquantile(values, rule.threshold))
            cut_text = f"value >= q{int(rule.threshold * 100)} = {cut:.3f}"
        elif rule.op == "le_quantile":
            cut = float(np.nanquantile(values, rule.threshold))
            cut_text = f"value <= q{int(rule.threshold * 100)} = {cut:.3f}"
        else:
            cut = float(rule.threshold)
            cut_text = f"value >= {cut:.3f}"
        rows.append(
            {
                "item": rule.name,
                "source_column": rule.column,
                "threshold": cut_text,
                "description": rule.description,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

def _build_rules(cfg: Config, features: pd.DataFrame) -> List[BinaryRule]:
    """Construct the binary rule list from the settings file."""
    t = cfg.get("binary_thresholds")
    rules: List[BinaryRule] = [
        BinaryRule(
            name="high_elderly_share",
            column="elderly_pct",
            op="ge_quantile",
            threshold=float(t["high_elderly_share_q"]),
            description="Subzone in the top tier of elderly population share.",
        ),
        BinaryRule(
            name="high_elderly_density",
            column="elderly_density",
            op="ge_quantile",
            threshold=float(t["high_elderly_density_q"]),
            description="Top tier of elderly residents per km2.",
        ),
        BinaryRule(
            name="poor_food_access",
            column="nearest_food_km",
            op="ge_quantile",
            threshold=float(t["poor_food_access_q"]),
            description="Top tier of distance to nearest food amenity.",
        ),
        BinaryRule(
            name="rich_food_access",
            column="food_access_score",
            op="ge_quantile",
            threshold=float(t["rich_food_q"]),
            description="Top tier of composite food access score.",
        ),
        BinaryRule(
            name="poor_transit_access",
            column="transit_access_score",
            op="le_quantile",
            threshold=1.0 - float(t["poor_transit_q"]),
            description="Bottom tier of transit access score.",
        ),
        BinaryRule(
            name="rich_transit_access",
            column="transit_access_score",
            op="ge_quantile",
            threshold=float(t["rich_transit_q"]),
            description="Top tier of transit access score.",
        ),
        BinaryRule(
            name="rich_accessibility_support",
            column="accessibility_support_score",
            op="ge_quantile",
            threshold=float(t["rich_accessibility_q"]),
            description="Top tier of barrier-free accessibility support.",
        ),
        BinaryRule(
            name="poor_accessibility_support",
            column="accessibility_support_score",
            op="le_quantile",
            threshold=1.0 - float(t["poor_accessibility_q"]),
            description="Bottom tier of barrier-free accessibility support.",
        ),
        BinaryRule(
            name="hawker_present",
            column="hawker_count",
            op="ge_value",
            threshold=float(t["hawker_present_min"]),
            description="At least one hawker centre within the configured buffer.",
        ),
        BinaryRule(
            name="supermarket_present",
            column="supermarket_count",
            op="ge_value",
            threshold=float(t["supermarket_present_min"]),
            description="At least one supermarket within the configured buffer.",
        ),
        BinaryRule(
            name="market_present",
            column="market_count",
            op="ge_value",
            threshold=float(t["market_present_min"]),
            description="At least one wet market within the configured buffer.",
        ),
        BinaryRule(
            name="diverse_food_environment",
            column="food_amenity_diversity",
            op="ge_value",
            threshold=float(t["diverse_food_min_types"]),
            description="At least N distinct food amenity types nearby.",
        ),
    ]
    return rules
