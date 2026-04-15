"""Schema validation helpers.

These are deliberately light: the goal is to fail fast with a clear message
rather than to enforce rigid type contracts.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SchemaError(Exception):
    """Raised when a dataframe is missing required columns."""


def require_columns(df: pd.DataFrame, required: Iterable[str], source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{source} is missing required columns: {missing}. "
            f"Available: {list(df.columns)[:20]}"
        )


def warn_unexpected_columns(
    df: pd.DataFrame, expected: Iterable[str], source: str
) -> None:
    extras = [c for c in df.columns if c not in expected]
    if extras:
        logger.debug("%s has unexpected columns: %s", source, extras[:10])
