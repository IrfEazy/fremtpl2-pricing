"""Policy-level modelling targets, built by one function from matched severity rows.

For every policy ``i`` (matched severity rows only, every amount finite and > 0):

- ``N``: number of matched severity rows (frequency target, option b);
- ``S``: sum of the matched amounts;
- ``S_cap``: sum of ``min(amount, cap)``, the cap applied per claim before aggregation;
- ``excess = S - S_cap``;
- ``ClaimNb_raw``: recorded ClaimNb (audit only, not a predictor) and ``claim_discrepancy``
  (``ClaimNb_raw != N``);
- ``freq_count``: the configured frequency target (``sev_rows`` -> N, ``claim_nb`` -> ClaimNb_raw).

Invariant, enforced: ``N == 0`` if and only if ``S == 0``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FREQUENCY_COUNTS = ("sev_rows", "claim_nb")
EXCESS_LOADING_MODES = ("none", "flat_per_exposure_year")
TARGET_COLUMNS = ("ClaimNb_raw", "N", "S", "S_cap", "excess", "claim_discrepancy", "freq_count")


def cap_amounts(amount: pd.Series, cap: float | None) -> pd.Series:
    """Per-claim limited amount ``min(amount, cap)``; no cap returns the amounts unchanged."""
    return amount if cap is None else amount.clip(upper=float(cap))


def valid_amount(amount: pd.Series) -> pd.Series:
    """True where a severity amount is finite and strictly positive."""
    a = amount.astype("float64")
    return pd.Series(np.isfinite(a) & (a > 0), index=amount.index)


def build_target_table(
    freq: pd.DataFrame, sev: pd.DataFrame, cap: float | None, frequency_count: str
) -> pd.DataFrame:
    """Return ``freq`` with the target columns added. ``sev`` must hold matched, valid rows."""
    if frequency_count not in FREQUENCY_COUNTS:
        raise ValueError(f"clean.targets.frequency_count must be one of {FREQUENCY_COUNTS}")
    if not valid_amount(sev["ClaimAmount"]).all():
        raise ValueError("severity amounts must be finite and > 0 (quarantine them first)")
    if not sev["IDpol"].isin(freq["IDpol"]).all():
        raise ValueError("severity rows must all match a policy in freq")
    limited = cap_amounts(sev["ClaimAmount"].astype("float64"), cap)
    g = pd.DataFrame({"IDpol": sev["IDpol"], "a": sev["ClaimAmount"], "c": limited}).groupby(
        "IDpol"
    )
    out = freq.copy()
    out["ClaimNb_raw"] = (
        out["ClaimNb"].astype("int64") if "ClaimNb_raw" not in out else out["ClaimNb_raw"]
    )
    out["N"] = out["IDpol"].map(g.size()).fillna(0).astype("int64")
    out["S"] = out["IDpol"].map(g["a"].sum()).fillna(0.0).astype("float64")
    out["S_cap"] = out["IDpol"].map(g["c"].sum()).fillna(0.0).astype("float64")
    out["excess"] = out["S"] - out["S_cap"]
    out["claim_discrepancy"] = out["ClaimNb_raw"] != out["N"]
    out["freq_count"] = out["N" if frequency_count == "sev_rows" else "ClaimNb_raw"].astype("int64")
    if ((out["N"] == 0) != (out["S"] == 0)).any():
        raise ValueError("invariant violated: N == 0 must hold exactly when S == 0")
    return out


def excess_loading(train: pd.DataFrame, rules: dict[str, Any]) -> float:
    """Flat annual excess loading per exposure-year from TRAIN rows only.

    ``sum(train excess) / sum(train exposure)``; orphans are not in the table, so they are excluded.
    The caller passes the training rows; this function never sees the test set.
    """
    mode = rules["excess_loading"]["mode"]
    if mode not in EXCESS_LOADING_MODES:
        raise ValueError(f"clean.excess_loading.mode must be one of {EXCESS_LOADING_MODES}")
    if mode == "none":
        return 0.0
    exposure = float(train["Exposure"].sum())
    if exposure <= 0:
        raise ValueError("training exposure must be positive to compute the excess loading")
    return float(train["excess"].sum()) / exposure
