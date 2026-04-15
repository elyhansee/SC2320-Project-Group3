"""Shared numerical helpers used across feature and modelling modules."""
from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_series(s: pd.Series) -> pd.Series:
    """Population z-score of a pandas Series, safe against zero variance."""
    sd = s.std(ddof=0)
    if sd == 0:
        return s * 0.0
    return (s - s.mean()) / sd


def zscore_array(values: np.ndarray) -> np.ndarray:
    """Population z-score of a numpy array, NaN-aware and zero-variance safe."""
    sd = float(np.nanstd(values))
    if sd == 0:
        return np.zeros_like(values, dtype=float)
    return (values - np.nanmean(values)) / sd
