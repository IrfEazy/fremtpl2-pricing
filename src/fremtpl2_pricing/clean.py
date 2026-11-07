"""Config-driven cleaning, applied once to the full dataset before any split.

Every rule is row-wise (or per-policy) so the result for a policy does not depend on which set it
falls into: ``clean(full)`` restricted to a subset equals cleaning that subset alone.

Order: IDpol normalisation, severity quarantine (amount not finite or <= 0), exposure quarantine
(Exposure not finite or <= 0), orphan severity rows (IDpol not in freq; kept in ``orphans`` for the
ledger), matched-row counts, policy exclusions (all optional, off in the primary rules), exposure
cap, then the single target table (``fremtpl2_pricing.targets``). Sensitivities are named rule
overrides declared in ``clean.sensitivities``; ``clean_sensitivity`` applies one by name.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .targets import build_target_table, cap_amounts, valid_amount

STATUSES = ("no_claims", "ok", "claims_without_sev", "sev_without_claims", "count_mismatch")
SENSITIVITIES_KEY = "sensitivities"


@dataclass(frozen=True)
class CleanResult:
    freq: pd.DataFrame  # policies with target columns, sev_rows and sev_status
    sev: pd.DataFrame  # matched valid rows: ClaimAmount (raw), ClaimAmount_capped
    orphans: pd.DataFrame  # valid severity rows whose IDpol is not in freq: IDpol, ClaimAmount
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


def sensitivity_rules(rules: dict[str, Any], name: str | None) -> dict[str, Any]:
    """Rules with the named sensitivity's overrides deep-merged in; None/'primary' = unchanged."""
    out = copy.deepcopy({k: v for k, v in rules.items() if k != SENSITIVITIES_KEY})
    if name in (None, "primary"):
        return out
    known = rules.get(SENSITIVITIES_KEY, {})
    if name not in known:
        raise ValueError(f"unknown sensitivity '{name}'; declared: {sorted(known)}")

    def merge(base: dict[str, Any], over: dict[str, Any]) -> None:
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                merge(base[k], v)
            else:
                base[k] = copy.deepcopy(v)

    merge(out, known[name])
    return out


def clean_sensitivity(freq: pd.DataFrame, sev: pd.DataFrame, rules: dict[str, Any], name: str):
    """Clean under the named pre-registered sensitivity (cleaning/target side only)."""
    return clean(freq, sev, sensitivity_rules(rules, name))


def _impact(freq_rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "policies": int(len(freq_rows)),
        "exposure": float(freq_rows["Exposure"].sum()),
        "claims": int(freq_rows["ClaimNb"].sum()),
    }


def _as_int_idpol(s: pd.Series, label: str) -> pd.Series:
    """IDpol as int64; floats are accepted only when every value is integral."""
    if pd.api.types.is_integer_dtype(s):
        return s.astype("int64")
    v = pd.to_numeric(s, errors="coerce")
    if v.isna().any() or not np.isfinite(v).all() or (v != np.floor(v)).any():
        raise ValueError(f"{label}.IDpol contains non-integral or missing values")
    return v.astype("int64")


def target_stats(freq: pd.DataFrame, eps: float) -> dict[str, Any]:
    """Measured consequences of the target definitions on a cleaned frame."""
    n, s, sc = int(freq["N"].sum()), float(freq["S"].sum()), float(freq["S_cap"].sum())
    return {
        "policies": int(len(freq)),
        "policies_with_claim_rows": int((freq["N"] > 0).sum()),
        "matched_rows": n,
        "claim_nb_recorded_total": int(freq["ClaimNb_raw"].sum()),
        "frequency_count_total": int(freq["freq_count"].sum()),
        "discrepancy_policies": int(freq["claim_discrepancy"].sum()),
        "cost_total": s,
        "limited_cost_total": sc,
        "excess_total": s - sc,
        "excess_share_of_cost": (s - sc) / max(s, eps),
        "excess_share_of_limited_cost": (s - sc) / max(sc, eps),
        "invariant_n_zero_iff_s_zero": bool(((freq["N"] == 0) == (freq["S"] == 0)).all()),
    }


def clean(freq: pd.DataFrame, sev: pd.DataFrame, rules: dict[str, Any]) -> CleanResult:
    eps = rules["epsilon"]
    freq = freq.copy()
    sev = sev.copy()
    report: dict[str, Any] = {}

    freq_dtype, sev_dtype = str(freq["IDpol"].dtype), str(sev["IDpol"].dtype)
    freq["IDpol"] = _as_int_idpol(freq["IDpol"], "freq")
    sev["IDpol"] = _as_int_idpol(sev["IDpol"], "sev")
    n_dup = int(freq["IDpol"].duplicated().sum())
    report["duplicate_idpol"] = {"action": rules["duplicate_idpol"], "rows": n_dup}
    if n_dup and rules["duplicate_idpol"] == "error":
        raise ValueError(f"{n_dup} duplicated IDpol in freq and clean.duplicate_idpol=error")
    report["idpol"] = {
        "freq_dtype": freq_dtype,
        "sev_dtype": sev_dtype,
        "normalised_dtype": "int64",
        "freq_unique": bool(freq["IDpol"].is_unique),
        "sev_distinct_idpol": int(sev["IDpol"].nunique()),
        "join": "many_to_one (sev -> freq)" if freq["IDpol"].is_unique else "NOT many_to_one",
    }
    raw_sev_rows, raw_sev_total = int(len(sev)), float(sev["ClaimAmount"].sum())

    # Severity amounts must be finite and > 0.
    bad = ~valid_amount(sev["ClaimAmount"])
    report["sev_quarantine"] = {
        "rows": int(bad.sum()),
        "policies": int(sev.loc[bad, "IDpol"].nunique()),
        "amount_finite_sum": float(
            sev.loc[bad, "ClaimAmount"].where(lambda a: np.isfinite(a)).sum()
        ),
    }
    sev = sev[~bad]
    valid_total = float(sev["ClaimAmount"].sum())

    # Exposure must be finite and > 0 (no floor above that).
    e = freq["Exposure"].astype("float64")
    bad_e = ~(np.isfinite(e) & (e > 0))
    quarantined_ids = freq.loc[bad_e, "IDpol"]
    in_q = sev["IDpol"].isin(quarantined_ids)
    report["exposure_quarantine"] = {
        "policies": int(bad_e.sum()),
        "claims": int(freq.loc[bad_e, "ClaimNb"].sum()),
        "sev_rows": int(in_q.sum()),
        "sev_amount": float(sev.loc[in_q, "ClaimAmount"].sum()),
    }

    # Severity rows without a policy in freq (judged on the raw policy table).
    orphan = ~sev["IDpol"].isin(freq["IDpol"])
    orphan_sev = sev[orphan][["IDpol", "ClaimAmount"]].reset_index(drop=True)
    report["sev_without_policy"] = {
        "action": rules["sev_without_policy"],
        "rows": int(orphan.sum()),
        "distinct_idpol": int(orphan_sev["IDpol"].nunique()),
        "amount": float(orphan_sev["ClaimAmount"].sum()),
        "amount_share": float(orphan_sev["ClaimAmount"].sum() / max(sev["ClaimAmount"].sum(), eps)),
    }
    if rules["sev_without_policy"] != "drop":
        raise ValueError("clean.sev_without_policy must be 'drop' (orphans cannot be modelled)")
    sev = sev[~orphan]
    freq = freq[~bad_e]
    sev = sev[~sev["IDpol"].isin(quarantined_ids)]
    matched_rows0, matched_cost0 = int(len(sev)), float(sev["ClaimAmount"].sum())

    # ClaimNb versus matched severity rows (classified on the recorded ClaimNb).
    counts = sev.groupby("IDpol").size()
    freq["ClaimNb_raw"] = freq["ClaimNb"].astype("int64")
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

    # Optional policy-level exclusions (all off in the primary rules).
    emin = rules.get("exposure_min")
    low = (freq["Exposure"] < emin) if emin is not None else pd.Series(False, index=freq.index)
    report["exposure_min"] = {"threshold": emin, **_impact(freq[low])}
    hc = rules["high_count"]["exclude_n_above"]
    high_n = (freq["sev_rows"] > hc) if hc is not None else pd.Series(False, index=freq.index)
    hsev = sev["IDpol"].isin(freq.loc[high_n, "IDpol"])
    report["high_count"] = {
        "exclude_n_above": hc,
        **_impact(freq[high_n]),
        "sev_rows": int(hsev.sum()),
        "sev_amount": float(sev.loc[hsev, "ClaimAmount"].sum()),
    }
    nb = rules["claim_nb"]
    high = pd.Series(False, index=freq.index)
    if nb["max"] is not None:
        over = freq["ClaimNb"] > nb["max"]
        report["claim_nb"] = {
            "max": nb["max"],
            "action": nb["action"],
            **_impact(freq[over]),
            "excess_claims": int((freq.loc[over, "ClaimNb"] - nb["max"]).sum()),
        }
        if nb["action"] == "exclude":
            high = over
        else:
            freq.loc[over, "ClaimNb"] = nb["max"]
    else:
        report["claim_nb"] = {"max": None}
    excluded = drop | low | high | high_n
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

    ll = rules["large_loss"]["cap"]
    sev["ClaimAmount_capped"] = cap_amounts(sev["ClaimAmount"].astype("float64"), ll)
    over_c = sev["ClaimAmount"] > ll if ll is not None else pd.Series(False, index=sev.index)
    removed = float((sev.loc[over_c, "ClaimAmount"] - sev.loc[over_c, "ClaimAmount_capped"]).sum())
    report["large_loss"] = {
        "cap": ll,
        "claims_capped": int(over_c.sum()),
        "amount_removed": removed,
        "amount_removed_share": removed / max(float(sev["ClaimAmount"].sum()), eps),
        "amount_removed_share_of_limited": removed
        / max(float(sev["ClaimAmount_capped"].sum()), eps),
        "applied_per_claim_before_aggregation": True,
    }
    tg = rules["targets"]
    freq = build_target_table(freq, sev, ll, tg["frequency_count"]).reset_index(drop=True)
    report["targets"] = {**tg, **target_stats(freq, eps)}
    report["reconciliation"] = {
        "sev_rows_raw": raw_sev_rows,
        "sev_total_raw": raw_sev_total,
        "quarantined_rows": report["sev_quarantine"]["rows"],
        "orphan_rows": int(len(orphan_sev)),
        "orphan_cost": float(orphan_sev["ClaimAmount"].sum()),
        "valid_total": valid_total,
        "exposure_quarantined_rows": report["exposure_quarantine"]["sev_rows"],
        "exposure_quarantined_cost": report["exposure_quarantine"]["sev_amount"],
        "matched_rows": matched_rows0,
        "matched_cost": matched_cost0,
        "matched_rows_modelled": int(len(sev)),
        "matched_cost_modelled": float(sev["ClaimAmount"].sum()),
    }
    return CleanResult(freq, sev.reset_index(drop=True), orphan_sev, report)


def totals(freq: pd.DataFrame, sev: pd.DataFrame) -> dict[str, Any]:
    return {
        "policies": int(len(freq)),
        "exposure": float(freq["Exposure"].sum()),
        "claims": int(freq["ClaimNb"].sum()),
        "policies_with_claim": int((freq["ClaimNb"] > 0).sum()),
        "sev_rows": int(len(sev)),
        "claim_amount": float(sev["ClaimAmount"].sum()),
    }
