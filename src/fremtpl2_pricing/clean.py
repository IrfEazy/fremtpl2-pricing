"""Config-driven cleaning, applied once to the full dataset before any split.

Every rule is row-wise (or per-policy) so the result for a policy does not depend on which set it
falls into: ``clean(full)`` restricted to a subset equals cleaning that subset alone.

Target definitions (``clean.targets``), added as per-policy columns:

- severity sample: policies with at least one severity row (``in_severity_sample``);
  ``sev_weight`` = number of severity rows (or capped ClaimNb, by config), never ClaimNb by default.
- pure-premium observed cost ``cost`` = sum of the (capped) severity amounts of the policy; claims
  without severity rows contribute zero cost.
- ``freq_count``: frequency target, ClaimNb as recorded (after the ClaimNb cap) or the number of
  severity rows, by config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

STATUSES = ("no_claims", "ok", "claims_without_sev", "sev_without_claims", "count_mismatch")


@dataclass
class CleanResult:
    freq: pd.DataFrame  # adds sev_rows and sev_status
    sev: pd.DataFrame  # ClaimAmount possibly capped; ClaimAmount_uncapped kept optionally
    report: dict[str, Any]


def sev_status(claim_nb: pd.Series, sev_rows: pd.Series) -> pd.Series:
    """Classify each policy by agreement between ClaimNb and the number of severity rows."""
    status = np.select(
        [
            (claim_nb == 0) & (sev_rows == 0),
            (claim_nb > 0) & (sev_rows == 0),
            (claim_nb == 0) & (sev_rows > 0),
            claim_nb == sev_rows,
        ],
        ["no_claims", "claims_without_sev", "sev_without_claims", "ok"],
        default="count_mismatch",
    )
    return pd.Series(status, index=claim_nb.index)


FREQUENCY_COUNTS = ("claim_nb", "sev_rows")
SEVERITY_WEIGHTS = ("sev_rows", "claim_nb")
PURE_PREMIUM_MODES = ("zero_cost", "exclude")


def _impact(freq_rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "policies": int(len(freq_rows)),
        "exposure": float(freq_rows["Exposure"].sum()),
        "claims": int(freq_rows["ClaimNb"].sum()),
    }


def add_targets(freq: pd.DataFrame, sev: pd.DataFrame, targets: dict[str, Any]) -> pd.DataFrame:
    """Add freq_count, sev_weight, cost, in_severity_sample, in_pure_premium_sample (per policy)."""
    if targets["frequency_count"] not in FREQUENCY_COUNTS:
        raise ValueError(f"clean.targets.frequency_count must be one of {FREQUENCY_COUNTS}")
    if targets["severity_weight"] not in SEVERITY_WEIGHTS:
        raise ValueError(f"clean.targets.severity_weight must be one of {SEVERITY_WEIGHTS}")
    mode = targets["pure_premium_claims_without_sev"]
    if mode not in PURE_PREMIUM_MODES:
        raise ValueError(
            f"clean.targets.pure_premium_claims_without_sev must be one of {PURE_PREMIUM_MODES}"
        )
    freq = freq.copy()
    cost = sev.groupby("IDpol")["ClaimAmount"].sum()
    freq["cost"] = freq["IDpol"].map(cost).fillna(0.0).astype("float64")
    freq["in_severity_sample"] = freq["sev_rows"] > 0
    freq["freq_count"] = freq[
        "ClaimNb" if targets["frequency_count"] == "claim_nb" else "sev_rows"
    ].astype("int64")
    freq["sev_weight"] = np.where(
        freq["in_severity_sample"],
        freq["sev_rows" if targets["severity_weight"] == "sev_rows" else "ClaimNb"],
        0,
    ).astype("int64")
    freq["in_pure_premium_sample"] = (
        ~(freq["sev_status"] == "claims_without_sev") if mode == "exclude" else True
    )
    return freq


def target_stats(freq: pd.DataFrame, eps: float) -> dict[str, Any]:
    """Measured consequences of the target definitions on a cleaned frame (needs add_targets)."""
    sev_rows = int(freq["sev_rows"].sum())
    cost = float(freq["cost"].sum())
    unmatched = (freq["ClaimNb"] - freq["sev_rows"]).clip(lower=0)
    claims = int(freq["ClaimNb"].sum())
    mean_cost = cost / max(sev_rows, 1)
    implied = float(unmatched.sum()) * mean_cost
    return {
        "severity_sample_policies": int(freq["in_severity_sample"].sum()),
        "severity_sample_sev_rows": sev_rows,
        "severity_weight_total": int(freq["sev_weight"].sum()),
        "frequency_count_total": int(freq["freq_count"].sum()),
        "claims_total": claims,
        "claims_without_sev_rows": int(unmatched.sum()),
        "claims_without_sev_rows_share": float(unmatched.sum() / max(claims, 1)),
        "pure_premium_sample_policies": int(freq["in_pure_premium_sample"].sum()),
        "cost_total": cost,
        "cost_per_sev_row": mean_cost,
        "cost_per_claim_if_all_claims_valued": cost / max(claims, 1),
        "implied_cost_at_mean_severity": implied,
        "implied_cost_share_of_total": float(implied / max(cost + implied, eps)),
    }


def clean(freq: pd.DataFrame, sev: pd.DataFrame, rules: dict[str, Any]) -> CleanResult:
    eps = rules["epsilon"]
    freq = freq.copy()
    sev = sev.copy()
    report: dict[str, Any] = {}

    n_dup = int(freq["IDpol"].duplicated().sum())
    report["duplicate_idpol"] = {"action": rules["duplicate_idpol"], "rows": n_dup}
    if n_dup and rules["duplicate_idpol"] == "error":
        raise ValueError(f"{n_dup} duplicated IDpol in freq and clean.duplicate_idpol=error")

    # Severity rows without a policy in freq.
    orphan = ~sev["IDpol"].isin(freq["IDpol"])
    report["sev_without_policy"] = {
        "action": rules["sev_without_policy"],
        "rows": int(orphan.sum()),
        "amount": float(sev.loc[orphan, "ClaimAmount"].sum()),
        "amount_share": float(
            sev.loc[orphan, "ClaimAmount"].sum() / max(sev["ClaimAmount"].sum(), eps)
        ),
    }
    if rules["sev_without_policy"] == "drop":
        sev = sev[~orphan]

    # ClaimNb versus severity rows (classified on the raw ClaimNb, before any cap).
    counts = sev.groupby("IDpol").size()
    freq["sev_rows"] = freq["IDpol"].map(counts).fillna(0).astype("int64")
    freq["sev_status"] = sev_status(freq["ClaimNb"], freq["sev_rows"])
    cons: dict[str, Any] = {}
    drop = pd.Series(False, index=freq.index)
    for cat, action in rules["sev_consistency"].items():
        m = freq["sev_status"] == cat
        cons[cat] = {
            "action": action,
            **_impact(freq[m]),
            "sev_rows": int(freq.loc[m, "sev_rows"].sum()),
        }
        if action == "drop":
            drop |= m
    report["sev_consistency"] = cons

    # Policy-level exclusions.
    emin = rules.get("exposure_min")
    low = (freq["Exposure"] < emin) if emin is not None else pd.Series(False, index=freq.index)
    report["exposure_min"] = {"threshold": emin, **_impact(freq[low])}
    nb = rules["claim_nb"]
    high = pd.Series(False, index=freq.index)
    if nb["max"] is not None:
        over = freq["ClaimNb"] > nb["max"]
        report["claim_nb"] = {
            "max": nb["max"],
            "action": nb["action"],
            **_impact(freq[over]),
            "excess_claims": int((freq.loc[over, "ClaimNb"] - nb["max"]).sum()),
            # Severity rows of these policies are never capped or dropped by the ClaimNb cap
            # (dropped only if action = exclude).
            "sev_rows": int(freq.loc[over, "sev_rows"].sum()),
            "sev_amount": float(
                sev.loc[sev["IDpol"].isin(freq.loc[over, "IDpol"]), "ClaimAmount"].sum()
            ),
            "sev_rows_kept": nb["action"] == "cap",
        }
        if nb["action"] == "exclude":
            high = over
        else:
            freq.loc[over, "ClaimNb"] = nb["max"]
    else:
        report["claim_nb"] = {"max": None}
    excluded = drop | low | high
    report["policies_excluded"] = _impact(freq[excluded])
    freq = freq[~excluded]
    sev = sev[sev["IDpol"].isin(freq["IDpol"])]

    cap = rules.get("exposure_cap")
    if cap is not None:
        over = freq["Exposure"] > cap
        report["exposure_cap"] = {
            "cap": cap,
            **_impact(freq[over]),
            "exposure_removed": float((freq.loc[over, "Exposure"] - cap).sum()),
        }
        freq.loc[over, "Exposure"] = cap
    else:
        report["exposure_cap"] = {"cap": None}

    ll = rules["large_loss"]
    if ll["cap"] is not None:
        if ll.get("keep_uncapped_column"):
            sev["ClaimAmount_uncapped"] = sev["ClaimAmount"]
        over = sev["ClaimAmount"] > ll["cap"]
        removed = float((sev.loc[over, "ClaimAmount"] - ll["cap"]).sum())
        report["large_loss"] = {
            "cap": ll["cap"],
            "claims_capped": int(over.sum()),
            "amount_removed": removed,
            "amount_removed_share": removed / max(float(sev["ClaimAmount"].sum()), eps),
        }
        sev.loc[over, "ClaimAmount"] = float(ll["cap"])
    else:
        report["large_loss"] = {"cap": None}
    freq = add_targets(freq, sev, rules["targets"]).reset_index(drop=True)
    report["targets"] = {**rules["targets"], **target_stats(freq, eps)}
    return CleanResult(freq, sev.reset_index(drop=True), report)


def totals(freq: pd.DataFrame, sev: pd.DataFrame) -> dict[str, Any]:
    return {
        "policies": int(len(freq)),
        "exposure": float(freq["Exposure"].sum()),
        "claims": int(freq["ClaimNb"].sum()),
        "policies_with_claim": int((freq["ClaimNb"] > 0).sum()),
        "sev_rows": int(len(sev)),
        "claim_amount": float(sev["ClaimAmount"].sum()),
    }
