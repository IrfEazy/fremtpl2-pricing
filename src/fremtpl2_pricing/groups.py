"""Group key = tuple of all rating factors (excludes IDpol, Exposure, ClaimNb)."""

from __future__ import annotations

import pandas as pd


def group_ids(freq: pd.DataFrame, factors: list[str]) -> pd.Series:
    """Deterministic integer id per distinct rating-factor tuple (sorted key order)."""
    ids = freq.groupby(factors, observed=True, sort=True, dropna=False).ngroup()
    return ids.astype("int64")
