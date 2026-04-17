"""Unit tests for the Apriori wrapper."""
from __future__ import annotations

import pandas as pd

from src.mining.apriori import run_apriori, rules_to_table, support_sweep


def _toy_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "high_elderly_share":  [True,  True,  True,  False, True,  True,  False, True,  True,  False],
            "poor_food_access":    [True,  True,  True,  False, True,  True,  False, False, True,  False],
            "poor_transit_access": [True,  False, True,  False, True,  False, False, False, True,  False],
            "rich_food_access":    [False, False, False, True,  False, False, True,  True,  False, True],
        }
    )


def test_run_apriori_finds_expected_rule() -> None:
    matrix = _toy_matrix()
    result = run_apriori(matrix, min_support=0.3, min_confidence=0.6, min_lift=1.0)
    assert result.itemsets.shape[0] > 0
    rules = rules_to_table(result.rules)
    assert not rules.empty
    # The "poor_food_access => high_elderly_share" rule should be high lift.
    has_rule = rules.apply(
        lambda r: "poor_food_access" in r["antecedents"] and "high_elderly_share" in r["consequents"],
        axis=1,
    ).any()
    assert has_rule


def test_support_sweep_returns_one_row_per_value() -> None:
    matrix = _toy_matrix()
    sweep = support_sweep(matrix, [0.2, 0.4, 0.6], min_confidence=0.5, min_lift=1.0, max_len=3)
    assert list(sweep["min_support"]) == [0.2, 0.4, 0.6]
    assert (sweep["n_rules"] >= 0).all()
