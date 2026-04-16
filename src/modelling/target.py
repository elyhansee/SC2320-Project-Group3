"""Transparent proxy target for nutritional vulnerability.

There is no published validated label for elderly nutritional vulnerability
at the subzone level. Rather than fabricate one, we construct an
**operational** vulnerability score whose every component is named in
``settings.yaml`` together with its direction (positive vs negative) and
weight. The score is the weighted z-score sum of the component features.
The binary risk label flags subzones whose score is in the top ``q``
quantile of the score distribution.

Both the score and the label are exposed to the rest of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.logging_utils import get_logger
from src.utils.math_utils import zscore_array

logger = get_logger(__name__)


@dataclass
class VulnerabilityTarget:
    """Container for the proxy target tables."""

    scores: pd.Series              # raw weighted z-score
    binary_label: pd.Series        # 1 if score >= configured quantile
    components: pd.DataFrame       # per-feature contribution table
    quantile_cut: float            # the actual numerical cut-off used
    description: str


def build_vulnerability_target(cfg: Config, features: pd.DataFrame) -> VulnerabilityTarget:
    """Build the operational nutritional vulnerability score and binary label.

    Each component is z-scored across the input rows. ``positive`` direction
    means a higher value increases vulnerability; ``negative`` means a higher
    value decreases vulnerability and the contribution is flipped before
    summing. The binary label uses the configured high-risk quantile.
    """
    components: List[Dict] = list(cfg.get("risk_model.components"))
    quantile = float(cfg.get("risk_model.high_risk_quantile", 0.75))
    rows = []
    contributions = pd.DataFrame(index=features.index)
    explanation_lines: List[str] = []
    for spec in components:
        col = spec["feature"]
        if col not in features.columns:
            raise KeyError(f"Risk component {col} not in feature table")
        weight = float(spec.get("weight", 1.0))
        direction = spec.get("direction", "positive")
        sign = 1.0 if direction == "positive" else -1.0
        values = features[col].astype(float).values
        z = zscore_array(values)
        contribution = sign * weight * z
        contributions[col] = contribution
        rows.append(
            {
                "feature": col,
                "direction": direction,
                "weight": weight,
                "reason": spec.get("reason", ""),
                "mean_contribution": float(np.nanmean(contribution)),
            }
        )
        explanation_lines.append(
            f"{col} ({direction}, weight={weight}): {spec.get('reason', '')}"
        )

    score = contributions.sum(axis=1)
    cut = float(np.nanquantile(score, quantile))
    label = (score >= cut).astype(int)

    components_df = pd.DataFrame(rows)
    description = (
        "Operational nutritional vulnerability score = weighted sum of "
        "z-scored components. Each component's direction and weight is set "
        "in settings.yaml. Binary label = score >= q{:.0f}.\n\n".format(quantile * 100)
        + "\n".join(explanation_lines)
    )
    logger.info(
        "Built vulnerability target: %d positive labels (cut=%.3f, q=%.2f)",
        int(label.sum()),
        cut,
        quantile,
    )
    return VulnerabilityTarget(
        scores=score,
        binary_label=label,
        components=components_df,
        quantile_cut=cut,
        description=description,
    )
