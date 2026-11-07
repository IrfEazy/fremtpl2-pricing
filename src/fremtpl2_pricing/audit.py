"""Data audit: measured numbers (results/audit.json) and the report rendered from them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .clean import STATUSES, clean, sev_status, totals
from .data import load_raw
from .groups import group_ids


def _pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f}%"


def _n(n: int, one: str, many: str) -> str:
    """'1 policy', '2 policies'."""
    return f"{n:,} {one if n == 1 else many}"


def _qpct(q: str) -> str:
    return f"{float(q) * 100:g}%"


def _cap_position(cap: float, quantiles: dict[str, float]) -> str:
    """Where a cap lies relative to the tabulated quantiles (computed, never asserted)."""
    items = sorted(((float(q), q, v) for q, v in quantiles.items()))
    below = [(q, v) for _, q, v in items if v <= cap]
    above = [(q, v) for _, q, v in items if v > cap]
    if below and above:
        (ql, vl), (qh, vh) = below[-1], above[0]
        return (
            f"The cap lies between the {_qpct(ql)} quantile ({vl:,.0f}) and the "
            f"{_qpct(qh)} quantile ({vh:,.0f}) of the raw amounts"
        )
    if above:
        q, v = above[0]
        return f"The cap lies below the lowest tabulated quantile ({_qpct(q)}: {v:,.0f})"
    q, v = below[-1]
    return f"The cap lies at or above the highest tabulated quantile ({_qpct(q)}: {v:,.0f})"


def _exposure(freq: pd.DataFrame, a: dict[str, Any]) -> dict[str, Any]:
    eps = a["epsilon"]
    e = freq["Exposure"]
    n = len(freq)
    over = e > 1
    out: dict[str, Any] = {
        "min": float(e.min()),
        "max": float(e.max()),
        "mean": float(e.mean()),
        "total": float(e.sum()),
        "gt1": {
            "rows": int(over.sum()),
            "share": float(over.mean()),
            "claims": int(freq.loc[over, "ClaimNb"].sum()),
            "claims_per_policy": float(freq.loc[over, "ClaimNb"].sum() / max(over.sum(), 1)),
            "exposure_above_1": float((e[over] - 1).sum()),
        },
        "quantiles": {str(q): float(e.quantile(q)) for q in a["exposure_quantiles"]},
        "small": [],
    }
    for t in a["exposure_small_thresholds"]:
        m = e <= t
        d = freq[m]
        out["small"].append(
            {
                "threshold": t,
                "rows": int(m.sum()),
                "share": float(m.sum() / n),
                "claims": int(d["ClaimNb"].sum()),
                "exposure": float(d["Exposure"].sum()),
                "exposure_share": float(d["Exposure"].sum() / e.sum()),
                "claims_per_exposure_year": float(
                    d["ClaimNb"].sum() / max(d["Exposure"].sum(), eps)
                ),
                "share_with_claim": float((d["ClaimNb"] > 0).mean()) if len(d) else 0.0,
            }
        )
    out["portfolio_claims_per_policy"] = float(freq["ClaimNb"].sum() / n)
    out["portfolio_claims_per_exposure_year"] = float(freq["ClaimNb"].sum() / e.sum())
    out["portfolio_share_with_claim"] = float((freq["ClaimNb"] > 0).mean())
    return out


def _claim_nb(freq: pd.DataFrame, a: dict[str, Any]) -> dict[str, Any]:
    vc = freq["ClaimNb"].value_counts().sort_index()
    thr = a["claim_nb_extreme_above"]
    ext = freq[freq["ClaimNb"] > thr]
    return {
        "distribution": {
            str(k): {
                "policies": int(v),
                "exposure": float(freq.loc[freq["ClaimNb"] == k, "Exposure"].sum()),
            }
            for k, v in vc.items()
        },
        "max": int(freq["ClaimNb"].max()),
        "extreme_threshold": thr,
        "extreme_policies": int(len(ext)),
        "extreme_claims": int(ext["ClaimNb"].sum()),
        "extreme_exposure": float(ext["Exposure"].sum()),
        "extreme_values": sorted(int(v) for v in ext["ClaimNb"]),
        "extreme_exposures": [float(v) for v in ext.sort_values("ClaimNb")["Exposure"]],
        "extreme_claims_per_exposure_year": float(
            ext["ClaimNb"].sum() / max(ext["Exposure"].sum(), a["epsilon"])
        ),
    }


def _cap_row(amt: pd.Series, cap: float | None, eps: float) -> dict[str, Any]:
    total = float(amt.sum())
    m = amt > cap if cap is not None else pd.Series(False, index=amt.index)
    excess = float((amt[m] - cap).sum()) if cap is not None else 0.0
    return {
        "cap": cap,
        "claims_above": int(m.sum()),
        "share_of_claims": float(m.mean()) if len(amt) else 0.0,
        "excess": excess,
        "limited_cost": total - excess,
        "excess_share_of_cost": excess / max(total, eps),
        "excess_share_of_limited_cost": excess / max(total - excess, eps),
    }


def _severity(sev: pd.DataFrame, matched: pd.DataFrame, a: dict[str, Any]) -> dict[str, Any]:
    amt = sev["ClaimAmount"]
    total = float(amt.sum())
    top = amt.nlargest(a["top_n"])
    out = {
        "rows": int(len(sev)),
        "distinct_idpol": int(sev["IDpol"].nunique()),
        "total": total,
        "min": float(amt.min()),
        "mean": float(amt.mean()),
        "small_amount_threshold": a["small_amount_threshold"],
        "rows_below_small_amount": int((amt < a["small_amount_threshold"]).sum()),
        "quantiles": {str(q): float(amt.quantile(q)) for q in a["amount_quantiles"]},
        "top": [{"amount": float(v), "share": float(v / total)} for v in top],
        "top_share": float(top.sum() / total),
        "cap_basis": "matched claims only",
        "matched_rows": int(len(matched)),
        "matched_total": float(matched["ClaimAmount"].sum()),
        "cap_candidates": [
            _cap_row(matched["ClaimAmount"], c, a["epsilon"]) for c in a["large_loss_candidates"]
        ],
        "cap_table": [_cap_row(matched["ClaimAmount"], c, a["epsilon"]) for c in a["cap_table"]],
    }
    return out


def _linkage(freq: pd.DataFrame, sev: pd.DataFrame, eps: float) -> dict[str, Any]:
    orphan = ~sev["IDpol"].isin(freq["IDpol"])
    counts = sev.groupby("IDpol").size()
    nsev = freq["IDpol"].map(counts).fillna(0).astype("int64")
    status = sev_status(freq["ClaimNb"], nsev)
    cats: dict[str, Any] = {}
    for s in STATUSES:
        m = status == s
        cats[s] = {
            "policies": int(m.sum()),
            "claims": int(freq.loc[m, "ClaimNb"].sum()),
            "sev_rows": int(nsev[m].sum()),
            "exposure": float(freq.loc[m, "Exposure"].sum()),
        }
    mm = status == "count_mismatch"
    cats["count_mismatch"]["sev_more_than_claimnb"] = int(
        (nsev[mm] > freq.loc[mm, "ClaimNb"]).sum()
    )
    cats["count_mismatch"]["sev_fewer_than_claimnb"] = int(
        (nsev[mm] < freq.loc[mm, "ClaimNb"]).sum()
    )
    cwo = status == "claims_without_sev"
    return {
        "duplicate_idpol_freq": int(freq["IDpol"].duplicated().sum()),
        "sev_without_freq": {
            "rows": int(orphan.sum()),
            "distinct_idpol": int(sev.loc[orphan, "IDpol"].nunique()),
            "amount": float(sev.loc[orphan, "ClaimAmount"].sum()),
            "amount_share": float(
                sev.loc[orphan, "ClaimAmount"].sum() / max(sev["ClaimAmount"].sum(), eps)
            ),
        },
        "status": cats,
        "claims_without_sev_share_of_claims": float(
            cats["claims_without_sev"]["claims"] / freq["ClaimNb"].sum()
        ),
        "claims_without_sev_share_of_claim_policies": float(
            cwo.sum() / max(int((freq["ClaimNb"] > 0).sum()), 1)
        ),
    }


def _factors(freq: pd.DataFrame, factors: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in factors:
        col = freq[f]
        if pd.api.types.is_numeric_dtype(col):
            out[f] = {
                "kind": "numeric",
                "min": float(col.min()),
                "max": float(col.max()),
                "mean": float(col.mean()),
                "levels": int(col.nunique()),
            }
        else:
            vc = col.astype(str).value_counts()
            out[f] = {
                "kind": "categorical",
                "levels": int(len(vc)),
                "smallest_level": {"name": str(vc.index[-1]), "policies": int(vc.iloc[-1])},
                "largest_level": {"name": str(vc.index[0]), "policies": int(vc.iloc[0])},
            }
    return out


def _leakage(
    freq: pd.DataFrame, factors: list[str], a: dict[str, Any], test_folds: int
) -> dict[str, Any]:
    gid = group_ids(freq, factors)
    size = gid.map(gid.value_counts())
    n = len(freq)
    claims = freq["ClaimNb"]
    multi = size > 1
    edges = a["group_size_bins"]
    open_end = 10**9
    bins = [(lo, hi - 1) for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    bins.append((edges[-1], open_end))
    dist = []
    gsizes = gid.value_counts()
    gclaims = claims.groupby(gid).sum()
    for lo, hi in bins:
        gm = (gsizes >= lo) & (gsizes <= hi)
        dist.append(
            {
                "size": f"{lo}" if lo == hi else (f">={lo}" if hi == open_end else f"{lo}-{hi}"),
                "groups": int(gm.sum()),
                "policies": int(gsizes[gm].sum()),
            }
        )
    return {
        "group_key": factors,
        "policies": n,
        "distinct_groups": int(gid.nunique()),
        "policies_in_multi_groups": int(multi.sum()),
        "share_policies_in_multi_groups": float(multi.mean()),
        "multi_groups": int((gsizes > 1).sum()),
        "largest_group": int(gsizes.max()),
        "fold_count": test_folds,
        "largest_group_share_of_fold_policies": float(gsizes.max() / (n / test_folds)),
        "most_claims_in_one_group": int(gclaims.max()),
        "most_claims_group_share_of_fold_claims": float(
            gclaims.max() / (claims.sum() / test_folds)
        ),
        "size_distribution": dist,
        "claims_in_multi_groups": int(claims[multi].sum()),
        "share_claims_in_multi_groups": float(claims[multi].sum() / claims.sum()),
        "share_with_claim_multi": float((claims[multi] > 0).mean()),
        "share_with_claim_single": float((claims[~multi] > 0).mean()),
    }


def _group_stats(d: pd.DataFrame, key: pd.Series, labels: list[str]) -> list[dict[str, Any]]:
    """Per-label recorded-versus-matched claim counts and exposure (ratio of totals)."""
    out = []
    for lab in labels:
        m = key == lab
        g = d[m]
        n, nb = int(g["N"].sum()), int(g["ClaimNb_raw"].sum())
        e, er = float(g["Exposure"].sum()), float(g["Exposure_raw"].sum())
        out.append(
            {
                "label": lab,
                "policies": int(len(g)),
                "exposure": e,
                "exposure_raw": er,
                "claim_nb": nb,
                "matched_rows": n,
                "claims_gap": nb - n,
                "discrepancy_policies": int(g["claim_discrepancy"].sum()),
                "discrepancy_share": float(g["claim_discrepancy"].mean()) if len(g) else 0.0,
                "freq_n_per_exposure": n / e if e > 0 else None,
                "freq_n_per_raw_exposure": n / er if er > 0 else None,
                "freq_claim_nb_per_exposure": nb / e if e > 0 else None,
            }
        )
    return out


def _fmt_edge(x: float) -> str:
    return f"{x:.10g}"


def _band_key(x: pd.Series, edges: list[float], what: str) -> tuple[pd.Series, list[str]]:
    labels = [f"{what}<={_fmt_edge(edges[0])}"]
    labels += [
        f"({_fmt_edge(lo)},{_fmt_edge(hi)}]" for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]
    labels.append(f">{_fmt_edge(edges[-1])}")
    idx = np.searchsorted(
        np.asarray(edges, dtype="float64"), x.to_numpy(dtype="float64"), side="left"
    )
    return pd.Series(np.array(labels, dtype=object)[idx], index=x.index), labels


def _target_frame(res: Any, freq: pd.DataFrame) -> pd.DataFrame:
    d = res.freq.copy()
    d["Exposure_raw"] = d["IDpol"].map(freq.set_index("IDpol")["Exposure"].astype("float64"))
    return d


def _discrepancy(d: pd.DataFrame, factors: list[str], a: dict[str, Any]) -> dict[str, Any]:
    id_key, id_labels = _band_key(d["IDpol"], a["idpol_band_edges"], "IDpol")
    e_key, e_labels = _band_key(d["Exposure_raw"], a["exposure_bands"], "e")
    fac: dict[str, Any] = {}
    for f in factors:
        col = d[f]
        if pd.api.types.is_numeric_dtype(col) and col.nunique() > a["discrepancy_numeric_bins"]:
            cut = pd.qcut(col, a["discrepancy_numeric_bins"], duplicates="drop")
            labels = [str(c) for c in cut.cat.categories]
            key = cut.astype(str)
        else:
            key = col.astype(str)
            labels = sorted(key.unique())
        fac[f] = _group_stats(d, key, labels)
    return {
        "idpol_bands": _group_stats(d, id_key, id_labels),
        "exposure_bands": _group_stats(d, e_key, e_labels),
        "factors": fac,
    }


def _reconciliation(res: Any, a: dict[str, Any]) -> dict[str, Any]:
    d, rec = res.freq, res.report["reconciliation"]
    gap = d["ClaimNb_raw"] - d["N"]
    by_status = {}
    for st in STATUSES:
        m = d["sev_status"] == st
        by_status[st] = {
            "policies": int(m.sum()),
            "claim_nb": int(d.loc[m, "ClaimNb_raw"].sum()),
            "matched_rows": int(d.loc[m, "N"].sum()),
            "gap": int(gap[m].sum()),
        }
    ext = a["external_reference"]
    claims = int(d["ClaimNb_raw"].sum())
    return {
        **rec,
        "total_cost_check": rec["valid_total"]
        - rec["matched_cost"]
        - rec["orphan_cost"]
        - rec["exposure_quarantined_cost"],
        "rows_check": rec["sev_rows_raw"]
        - rec["quarantined_rows"]
        - rec["orphan_rows"]
        - rec["exposure_quarantined_rows"]
        - rec["matched_rows"],
        "claim_nb_total": claims,
        "claim_nb_minus_matched": claims - int(d["N"].sum()),
        "shortfall_claims": int(gap.clip(lower=0).sum()),
        "surplus_rows": int((-gap).clip(lower=0).sum()),
        "by_status": by_status,
        "external_severity_rows": ext["severity_rows"],
        "matched_equals_external": rec["matched_rows"] == ext["severity_rows"],
    }


def _high_counts(d: pd.DataFrame, a: dict[str, Any]) -> dict[str, Any]:
    thr = a["high_count_n_above"]
    h = d[d["N"] > thr].sort_values("N")
    return {
        "threshold": thr,
        "policies": int(len(h)),
        "exposure": float(h["Exposure"].sum()),
        "matched_rows": int(h["N"].sum()),
        "claim_nb_recorded": int(h["ClaimNb_raw"].sum()),
        "cost": float(h["S"].sum()),
        "limited_cost": float(h["S_cap"].sum()),
        "cost_share_of_matched": float(h["S"].sum() / max(d["S"].sum(), a["epsilon"])),
        "n_values": [int(v) for v in h["N"]],
        "exposures": [float(v) for v in h["Exposure"]],
        "n_per_exposure": float(h["N"].sum() / max(h["Exposure"].sum(), a["epsilon"])),
        "portfolio_n_per_exposure": float(d["N"].sum() / max(d["Exposure"].sum(), a["epsilon"])),
    }


def _version_note(freq: pd.DataFrame, sev: pd.DataFrame, rec: dict[str, Any], a: dict[str, Any]):
    ext = a["external_reference"]
    return {
        "source": ext["source"],
        "external_policies": ext["policies"],
        "external_severity_rows": ext["severity_rows"],
        "file_policies": int(len(freq)),
        "file_severity_rows": int(len(sev)),
        "matched_rows": rec["matched_rows"],
        "policies_difference": int(len(freq)) - ext["policies"],
        "matched_minus_external_rows": rec["matched_rows"] - ext["severity_rows"],
    }


def compute_audit(freq: pd.DataFrame, sev: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    a = cfg["audit"]
    factors = cfg["rating_factors"]
    res = clean(freq, sev, cfg["clean"])
    d = _target_frame(res, freq)
    return {
        "dataset": {"freq_rows": int(len(freq)), "sev_rows": int(len(sev))},
        "exposure": _exposure(freq, a),
        "claim_nb": _claim_nb(freq, a),
        "severity": _severity(sev, res.sev, a),
        "reconciliation": _reconciliation(res, a),
        "discrepancy": _discrepancy(d, factors, a),
        "high_counts": _high_counts(d, a),
        "version_note": _version_note(freq, sev, res.report["reconciliation"], a),
        "linkage": _linkage(freq, sev, a["epsilon"]),
        "factors": _factors(freq, factors),
        "leakage": _leakage(freq, factors, a, cfg["split"]["test_folds"]),
        "cleaning": {
            "rules": cfg["clean"],
            "impact": res.report,
            "before": totals(freq, sev),
            "after": totals(res.freq, res.sev),
        },
        "split_primary": cfg["split"]["primary"],
    }


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def render_markdown(au: dict[str, Any]) -> str:
    ex, cn, sv, lk, lg, cl = (
        au[k] for k in ("exposure", "claim_nb", "severity", "linkage", "leakage", "cleaning")
    )
    rules, imp = cl["rules"], cl["impact"]
    o: list[str] = []
    w = o.append
    w("# Data audit: freMTPL2\n")
    w(
        "Generated by `fremtpl2 audit` from `results/audit.json`; every number below comes from the data "
        "(the raw files were SHA-256 verified against `data/raw/manifest.json` first). "
        "The judgement calls were decided by the human on 2026-11-07 (section 8); the raw numbers in "
        "sections 1 to 6 describe the files as recorded.\n"
    )
    w(
        f"Raw size: {_n(au['dataset']['freq_rows'], 'policy', 'policies')} (freq), {au['dataset']['sev_rows']:,} severity rows (sev).\n"
    )

    w("## 1. Exposure\n")
    w(
        f"Range {ex['min']:.4f} to {ex['max']:.2f} years, mean {ex['mean']:.3f}, total {ex['total']:,.1f} policy-years. "
        f"Portfolio claim frequency {ex['portfolio_claims_per_exposure_year']:.4f} claims per policy-year; "
        f"{_pct(ex['portfolio_share_with_claim'])} of policies have a claim.\n"
    )
    g = ex["gt1"]
    w(
        f"- Exposure > 1: {_n(g['rows'], 'policy', 'policies')} ({_pct(g['share'])}), {_n(g['claims'], 'claim', 'claims')}, "
        f"{g['exposure_above_1']:,.1f} policy-years above the 1-year mark; claims per policy "
        f"{g['claims_per_policy']:.4f} versus {ex['portfolio_claims_per_policy']:.4f} for the portfolio.\n"
    )
    w("Exposure quantiles:\n")
    w(_table(["quantile", "exposure"], [[q, f"{v:.4f}"] for q, v in ex["quantiles"].items()]))
    w("\nVery small exposures (Exposure <= threshold):\n")
    w(
        _table(
            [
                "threshold",
                "policies",
                "share",
                "claims",
                "claims per policy-year",
                "share with a claim",
            ],
            [
                [
                    s["threshold"],
                    f"{s['rows']:,}",
                    _pct(s["share"]),
                    f"{s['claims']:,}",
                    f"{s['claims_per_exposure_year']:.3f}",
                    _pct(s["share_with_claim"]),
                ]
                for s in ex["small"]
            ],
        )
    )
    w("")

    w("## 2. ClaimNb\n")
    w(
        _table(
            ["ClaimNb", "policies", "share", "exposure"],
            [
                [
                    k,
                    f"{v['policies']:,}",
                    _pct(v["policies"] / au["dataset"]["freq_rows"], 3),
                    f"{v['exposure']:,.1f}",
                ]
                for k, v in cn["distribution"].items()
            ],
        )
    )
    w(
        f"\nMaximum ClaimNb {cn['max']}. Extreme values (> {cn['extreme_threshold']}): "
        f"{_n(cn['extreme_policies'], 'policy', 'policies')} with ClaimNb {cn['extreme_values']}, "
        f"total {_n(cn['extreme_claims'], 'claim', 'claims')} on {cn['extreme_exposure']:.2f} policy-years "
        f"(exposures {[round(x, 2) for x in cn['extreme_exposures']]}).\n"
    )

    w("## 3. Severity (ClaimAmount)\n")
    w(
        f"{_n(sv['rows'], 'row', 'rows')} on {sv['distinct_idpol']:,} distinct policies; total {sv['total']:,.2f}, "
        f"mean {sv['mean']:,.2f}, minimum {sv['min']:,.2f} ({_n(sv['rows_below_small_amount'], 'row', 'rows')} below {sv['small_amount_threshold']:g}).\n"
    )
    w(_table(["quantile", "ClaimAmount"], [[q, f"{v:,.2f}"] for q, v in sv["quantiles"].items()]))
    w(f"\nTen largest claims ({_pct(sv['top_share'])} of total cost together):\n")
    w(
        _table(
            ["rank", "ClaimAmount", "share of total cost"],
            [[i + 1, f"{t['amount']:,.2f}", _pct(t["share"])] for i, t in enumerate(sv["top"])],
        )
    )
    w("\nExcess over candidate per-claim caps (matched claims only):\n")
    w(
        _table(
            ["cap", "claims above", "share of claims", "excess as share of matched cost"],
            [
                [
                    f"{c['cap']:,.0f}",
                    c["claims_above"],
                    _pct(c["share_of_claims"], 3),
                    _pct(c["excess_share_of_cost"]),
                ]
                for c in sv["cap_candidates"]
            ],
        )
    )
    w("")

    w("## 4. Linkage of freq and sev\n")
    s = lk["sev_without_freq"]
    w(f"- Duplicate IDpol in freq: {lk['duplicate_idpol_freq']}.")
    w(
        f"- Severity rows whose IDpol is absent from freq: {_n(s['rows'], 'row', 'rows')} ({_n(s['distinct_idpol'], 'policy', 'policies')}), "
        f"amount {s['amount']:,.2f} ({_pct(s['amount_share'])} of total cost)."
    )
    st = lk["status"]
    w(
        f"- ClaimNb > 0 with no severity rows: {_n(st['claims_without_sev']['policies'], 'policy', 'policies')}, "
        f"{_n(st['claims_without_sev']['claims'], 'claim', 'claims')} ({_pct(lk['claims_without_sev_share_of_claims'])} of all claims; "
        f"{_pct(lk['claims_without_sev_share_of_claim_policies'])} of policies with a claim)."
    )
    w(
        f"- Severity rows with ClaimNb = 0: {_n(st['sev_without_claims']['policies'], 'policy', 'policies')}, "
        f"{_n(st['sev_without_claims']['sev_rows'], 'row', 'rows')}."
    )
    cm = st["count_mismatch"]
    w(
        f"- Both > 0 but ClaimNb differs from the number of severity rows: {_n(cm['policies'], 'policy', 'policies')} "
        f"({cm['sev_more_than_claimnb']} with more severity rows than ClaimNb, "
        f"{cm['sev_fewer_than_claimnb']} with fewer)."
    )
    w(
        f"- Consistent (ClaimNb equals severity rows, > 0): {_n(st['ok']['policies'], 'policy', 'policies')}, "
        f"{_n(st['ok']['claims'], 'claim', 'claims')}.\n"
    )

    w("## 5. Rating factors\n")
    rows = []
    for f, d in au["factors"].items():
        if d["kind"] == "numeric":
            rows.append(
                [f, "numeric", d["levels"], f"{d['min']:g} to {d['max']:g} (mean {d['mean']:.2f})"]
            )
        else:
            rows.append(
                [
                    f,
                    "categorical",
                    d["levels"],
                    f"largest {d['largest_level']['name']} ({d['largest_level']['policies']:,}), "
                    f"smallest {d['smallest_level']['name']} ({d['smallest_level']['policies']:,})",
                ]
            )
    w(_table(["factor", "type", "distinct levels", "range / extremes"], rows))
    w("")

    w("## 6. Leakage: identical rating-factor tuples\n")
    w(f"Group key: tuple of {', '.join(lg['group_key'])} (excludes IDpol, Exposure, ClaimNb).\n")
    w(
        f"- Distinct groups: {lg['distinct_groups']:,} for {_n(lg['policies'], 'policy', 'policies')}."
    )
    w(
        f"- Policies in groups of size > 1: {lg['policies_in_multi_groups']:,} "
        f"({_pct(lg['share_policies_in_multi_groups'])}) in {_n(lg['multi_groups'], 'group', 'groups')}; largest group {lg['largest_group']}."
    )
    w(
        f"- Claims in multi-policy groups: {lg['claims_in_multi_groups']:,} ({_pct(lg['share_claims_in_multi_groups'])} of claims); "
        f"share of policies with a claim: {_pct(lg['share_with_claim_multi'])} in multi-policy groups versus "
        f"{_pct(lg['share_with_claim_single'])} in singleton groups.\n"
    )
    w(
        _table(
            ["group size", "groups", "policies"],
            [
                [d["size"], f"{d['groups']:,}", f"{d['policies']:,}"]
                for d in lg["size_distribution"]
            ],
        )
    )
    w(
        f"\nGroup size versus fold size (a fold holds 1/{lg['fold_count']} of policies and claims on average): "
        f"the largest group has {_n(lg['largest_group'], 'policy', 'policies')}, {_pct(lg['largest_group_share_of_fold_policies'])} of an average fold's "
        f"policies; the group with the most claims has {_n(lg['most_claims_in_one_group'], 'claim', 'claims')}, "
        f"{_pct(lg['most_claims_group_share_of_fold_claims'])} of an average fold's claims. "
        f"The realised maximum group share of each set and CV fold is in `results/split_summary.json`.\n"
    )
    w(
        f"Protocol: the grouped split (key above, StratifiedGroupKFold on claim presence) was pre-registered "
        f"as the primary evaluation (PM-1) before this audit was run and stays frozen (section 8). The random "
        f"stratified split is kept as a labelled sensitivity. Currently selected primary: `{au['split_primary']}`. "
        f"This is a rating-profile holdout, not customer-level leakage prevention: identical tuples can belong to "
        f"different people and one person's different tuples can still straddle the split.\n"
    )

    w("## 7. Cleaning rules and targets\n")
    w(
        "All rules live in `configs/default.yaml` (`clean:`), are applied by one function "
        "(`fremtpl2_pricing.clean.clean`) to the full dataset before splitting, and treat every policy "
        "identically regardless of set. The decisions on the judgement calls below were taken by the "
        "human on 2026-11-07 (section 8).\n"
    )
    d = imp["duplicate_idpol"]
    ip = imp["idpol"]
    w(
        f"**R0 identifiers** (`duplicate_idpol: {rules['duplicate_idpol']}`): {d['rows']} duplicated IDpol "
        f"measured, so nothing is removed; the rule makes the pipeline fail if a future data version contains "
        f"duplicates. IDpol is normalised to {ip['normalised_dtype']} (freq dtype {ip['freq_dtype']}, sev dtype "
        f"{ip['sev_dtype']}; a non-integral value is an error). Join cardinality: {ip['join']}, freq IDpol "
        f"unique: {ip['freq_unique']}, {ip['sev_distinct_idpol']:,} distinct IDpol in sev.\n"
    )
    ec = imp["exposure_cap"]
    eq = imp["exposure_quarantine"]
    if ec["cap"] is not None:
        w(
            f"**R1 exposure**: cap at {ec['cap']:g}: {_n(ec['policies'], 'policy', 'policies')} "
            f"({_pct(ec['policies'] / au['dataset']['freq_rows'])}) have Exposure > {ec['cap']:g} "
            f"(maximum {ex['max']:.2f}), carrying {_n(ec['claims'], 'claim', 'claims')}; capping removes "
            f"{ec['exposure_removed']:,.1f} policy-years ({_pct(ec['exposure_removed'] / ex['total'], 3)} of exposure). "
            f"Assumption (not testable from the data): Exposure above {ec['cap']:g} year is a recording artefact; "
            f"the cap keeps the policy and its claims. Exposure that is not finite or <= 0 is quarantined: "
            f"{_n(eq['policies'], 'policy', 'policies')} ({_n(eq['sev_rows'], 'severity row', 'severity rows')}). "
            f"Exposure stays out of the rating-factor list; it is only an offset / weight.\n"
        )
    else:
        w(
            f"**R1 exposure**: no cap (raw exposure); quarantined (not finite or <= 0): "
            f"{_n(eq['policies'], 'policy', 'policies')}.\n"
        )
    em = imp["exposure_min"]
    small = ex["small"]
    base = ex["portfolio_claims_per_exposure_year"]
    if em["threshold"] is None:
        w(
            f"**R2 minimum exposure: none (all positive exposures kept, no floor)**. Measured on recorded ClaimNb: "
            f"{_n(small[0]['rows'], 'policy', 'policies')} ({_pct(small[0]['share'])}) have Exposure <= {small[0]['threshold']}, "
            f"with {_n(small[0]['claims'], 'claim', 'claims')} and an annualised frequency of "
            f"{small[0]['claims_per_exposure_year']:.2f} against {base:.3f} for the portfolio. "
            f"Policies up to {small[-1]['threshold']} years total {_n(small[-1]['rows'], 'row', 'rows')}, "
            f"{_n(small[-1]['claims'], 'claim', 'claims')} ({_pct(small[-1]['claims'] / au['cleaning']['before']['claims'])} "
            f"of claims) and {_pct(small[-1]['exposure_share'], 3)} of exposure. A floor would invent exposure and "
            f"any cut-off would be chosen after looking at outcomes; short-record exclusions are pre-registered "
            f"sensitivities (section 8). Frequencies on matched rows are in section 12.\n"
        )
    else:
        w(
            f"**R2 minimum exposure {em['threshold']}**: {_n(em['policies'], 'policy', 'policies')}, {_n(em['claims'], 'claim', 'claims')} and "
            f"{em['exposure']:,.1f} policy-years dropped.\n"
        )
    nb = imp["claim_nb"]
    if nb["max"] is None:
        w(
            f"**R3 ClaimNb: no clipping**. Recorded ClaimNb above {cn['extreme_threshold']}: "
            f"{_n(cn['extreme_policies'], 'policy', 'policies')} (values {cn['extreme_values']}, exposures "
            f"{[round(x, 2) for x in cn['extreme_exposures']]}). Counts are re-audited with N_i in section 11.\n"
        )
    else:
        w(
            f"**R3 ClaimNb {nb['action']} at {nb['max']}**: {_n(nb['policies'], 'policy', 'policies')} with ClaimNb > {nb['max']}, "
            f"{_n(nb['excess_claims'], 'claim', 'claims')} above the cap.\n"
        )
    hc = imp["high_count"]
    if hc["exclude_n_above"] is not None:
        w(
            f"**R3b high counts excluded**: whole policies with N_i > {hc['exclude_n_above']}: "
            f"{_n(hc['policies'], 'policy', 'policies')}, {_n(hc['sev_rows'], 'severity row', 'severity rows')}.\n"
        )
    sw = imp["sev_without_policy"]
    sq = imp["sev_quarantine"]
    w(
        f"**R4 severity rows**: amounts must be finite and > 0, otherwise quarantined: {_n(sq['rows'], 'row', 'rows')} "
        f"quarantined. Rows whose IDpol is absent from freq ({sw['action']}): {_n(sw['rows'], 'row', 'rows')} "
        f"({_n(sw['distinct_idpol'], 'policy', 'policies')}), amount {sw['amount']:,.2f} ({_pct(sw['amount_share'])} of total cost). "
        f"They have no rating factors, exposure or split side, so they are excluded from modelling and written to "
        f"`results/orphan_ledger.csv` (IDpol, ClaimAmount); they stay in the reconciliation (section 9).\n"
    )
    sc = imp["sev_consistency"]
    tg = imp["targets"]
    w(
        f"**R5 ClaimNb versus severity rows**: claims_without_sev = {sc['claims_without_sev']['action']} "
        f"({_n(sc['claims_without_sev']['policies'], 'policy', 'policies')}, {_n(sc['claims_without_sev']['claims'], 'claim', 'claims')}, "
        f"{_pct(lk['claims_without_sev_share_of_claims'])} of claims), sev_without_claims = "
        f"{sc['sev_without_claims']['action']} ({_n(sc['sev_without_claims']['policies'], 'policy', 'policies')}), count_mismatch = "
        f"{sc['count_mismatch']['action']} ({_n(sc['count_mismatch']['policies'], 'policy', 'policies')}). Policies are kept "
        f'and flagged in `sev_status`. "No severity row" does not prove an unpaid, closed-without-payment or zero-cost '
        f"claim; the target is called the recorded matched claim cost, not an estimate of ultimate cost.\n"
    )
    w("### Target definitions (config `clean.targets`, one policy-level table)\n")
    w(
        f"- N_i = number of matched severity rows ({tg['matched_rows']:,} in total on "
        f"{_n(tg['policies_with_claim_rows'], 'policy', 'policies')}); frequency target `{tg['frequency_count']}` "
        f"(`sev_rows` = N_i, the primary; `claim_nb` = recorded ClaimNb, a sensitivity), total {tg['frequency_count_total']:,}."
    )
    w(
        f"- S_i = sum of matched amounts (total {tg['cost_total']:,.2f}); S_i_cap = sum of min(amount, per-claim cap) "
        f"applied per claim before aggregation (total {tg['limited_cost_total']:,.2f}); excess_i = S_i - S_i_cap "
        f"(total {tg['excess_total']:,.2f}, {_pct(tg['excess_share_of_cost'])} of cost, "
        f"{_pct(tg['excess_share_of_limited_cost'])} of limited cost)."
    )
    w(
        f"- ClaimNb_raw (recorded, total {tg['claim_nb_recorded_total']:,}) and the discrepancy flag "
        f"(ClaimNb_raw != N_i, {_n(tg['discrepancy_policies'], 'policy', 'policies')}) are kept for the audit and are not predictors."
    )
    w(
        f"- Invariant N_i = 0 if and only if S_i = 0 holds: {tg['invariant_n_zero_iff_s_zero']}. "
        f"Severity weight is N_i; Tweedie uses S_i_cap / exposure with weight exposure.\n"
    )
    ll = imp["large_loss"]
    if ll["cap"] is not None:
        w(
            f"**R6 per-claim cap at {ll['cap']:,.0f}**: {_n(ll['claims_capped'], 'claim', 'claims')} of the matched claims exceed it "
            f"({_pct(ll['claims_capped'] / sv['matched_rows'], 3)}) and the excess is {ll['amount_removed']:,.2f} "
            f"({_pct(ll['amount_removed_share'])} of matched cost, {_pct(ll['amount_removed_share_of_limited'])} of limited cost). "
            f"The largest single claim is {_pct(sv['top'][0]['share'])} of total cost and the ten largest together "
            f"{_pct(sv['top_share'])} (all severity rows). {_cap_position(ll['cap'], sv['quantiles'])}. The cap is a chosen "
            f"estimand, not a discovered error boundary. Raw amounts are kept in `ClaimAmount`, limited amounts in "
            f"`ClaimAmount_capped`; cap and excess impacts are computed on matched claims only (section 13).\n"
        )
    else:
        w("**R6 per-claim cap**: none (uncapped).\n")
    b, af = cl["before"], cl["after"]
    w("### Totals before and after cleaning\n")
    w(
        _table(
            ["metric", "before", "after"],
            [
                ["policies", f"{b['policies']:,}", f"{af['policies']:,}"],
                ["exposure (policy-years)", f"{b['exposure']:,.2f}", f"{af['exposure']:,.2f}"],
                ["claims (sum ClaimNb)", f"{b['claims']:,}", f"{af['claims']:,}"],
                [
                    "policies with a claim",
                    f"{b['policies_with_claim']:,}",
                    f"{af['policies_with_claim']:,}",
                ],
                ["severity rows", f"{b['sev_rows']:,}", f"{af['sev_rows']:,}"],
                ["total ClaimAmount", f"{b['claim_amount']:,.2f}", f"{af['claim_amount']:,.2f}"],
            ],
        )
    )
    w("")
    rc, dc, hcn, vn = au["reconciliation"], au["discrepancy"], au["high_counts"], au["version_note"]
    w("## 8. Post-audit, pre-model amendments (human decision 2026-11-07)\n")
    w(
        "Decided after this audit and before any model is run. They are benchmark design choices, not proof "
        "that the underlying records are correct. The split assignment registered earlier stays frozen.\n"
    )
    cap_txt = "uncapped" if ll["cap"] is None else f"{ll['cap']:,.0f}"
    sens = rules.get("sensitivities", {})
    w(
        "1. **Evaluation split.** The grouped split (group = tuple of the rating factors, stratified on claim "
        "presence ClaimNb > 0) stays the primary and is not re-assigned; the random split is a sensitivity. It "
        "is a rating-profile holdout, not customer-level leakage prevention: identical covariates do not "
        "establish the same policyholder, and one customer with changed attributes can straddle the split. "
        "Wüthrich and Merz flag likely renewal-split records; this is cited only as a convention. "
        "Assignments are never re-searched; `fremtpl2 split` recomputes the assignment in memory and fails if "
        "it differs from the stored `data/processed/split.parquet`, which it never overwrites."
    )
    w(
        f"2. **Large losses.** Per-claim cap {cap_txt} applied before aggregation; raw amounts and excess are kept "
        f"separately. Primary comparison on the limited cost without reloading. Optional gross illustration: a "
        f"training-only flat annual loading per exposure-year (`clean.excess_loading.mode: "
        f"{rules['excess_loading']['mode']}`), sum of train excess / sum of train exposure, orphans excluded, "
        f"never estimated on the test set (value in `results/targets_summary.json`). The scikit-learn Tweedie tutorial "
        f"caps at 200,000 per policy after aggregation, not per claim, so it is not equivalent to a per-claim cap; "
        f"cited only as a convention."
    )
    w(
        f"3. **Claims without severity.** Option (b): frequency target N_i = matched severity rows "
        f"(`frequency_count: {tg['frequency_count']}`), so frequency x severity and Tweedie share one observed-cost "
        f"target. ClaimNb_raw and a discrepancy flag are kept for the audit only. Recorded-ClaimNb frequency "
        f"(option a) is the sensitivity `frequency_recorded_claim_nb`. Amounts must be finite and > 0, else "
        f"quarantined ({sq['rows']} found)."
    )
    nbtxt = "disabled" if imp["claim_nb"]["max"] is None else f"set to {imp['claim_nb']['max']}"
    w(
        f"4. **ClaimNb clipping** is {nbtxt} in the primary rules. High counts are re-audited with N_i "
        f"(section 11). Whole-policy exclusion of N_i > {hcn['threshold']} (frequency, severity and Tweedie alike) is "
        f"the sensitivity `exclude_high_count`."
    )
    w(
        "5. **Exposure.** All finite exposures > 0 are kept (no floor), e <= 0 is quarantined, the cap at "
        f"{rules['exposure_cap']} year is unchanged, and exposure is never a rating factor (offset / weight only). "
        "Short-record exclusion and raw (uncapped) exposure are sensitivities."
    )
    w(
        "6. **Orphan severity rows** are excluded from modelling, written to `results/orphan_ledger.csv` and kept in "
        "the reconciliation (section 9); cap and excess impacts use matched claims only."
    )
    w(
        "\nPre-registered sensitivities (declared in `clean.sensitivities`, selectable by name via "
        "`clean.clean_sensitivity`; no model has been run on them): "
        + ", ".join(f"`{k}`" for k in sens)
        + ".\n"
    )

    w("## 9. Reconciliation of matched claims\n")
    w(
        _table(
            ["item", "rows", "cost"],
            [
                [
                    "all severity rows (raw file)",
                    f"{rc['sev_rows_raw']:,}",
                    f"{rc['sev_total_raw']:,.2f}",
                ],
                ["quarantined (amount not finite or <= 0)", f"{rc['quarantined_rows']:,}", "-"],
                [
                    "valid severity rows",
                    f"{rc['sev_rows_raw'] - rc['quarantined_rows']:,}",
                    f"{rc['valid_total']:,.2f}",
                ],
                [
                    "orphans (IDpol not in freq)",
                    f"{rc['orphan_rows']:,}",
                    f"{rc['orphan_cost']:,.2f}",
                ],
                [
                    "policies with exposure <= 0",
                    f"{rc['exposure_quarantined_rows']:,}",
                    f"{rc['exposure_quarantined_cost']:,.2f}",
                ],
                ["matched", f"{rc['matched_rows']:,}", f"{rc['matched_cost']:,.2f}"],
            ],
        )
    )
    w(
        f"\nReconciliation: valid cost = matched + orphan + exposure-quarantined, difference {rc['total_cost_check']:.6f} "
        f"(rows difference {rc['rows_check']}). Matched cost share {_pct(rc['matched_cost'] / max(rc['valid_total'], 1e-12))}, "
        f"orphan cost share {_pct(rc['orphan_cost'] / max(rc['valid_total'], 1e-12))}.\n"
    )
    w(
        f"Matched severity rows: {rc['matched_rows']:,} (external figure {rc['external_severity_rows']:,}: "
        f"{'equal' if rc['matched_equals_external'] else 'different'}). Recorded ClaimNb sum "
        f"{rc['claim_nb_total']:,} minus matched rows = {rc['claim_nb_minus_matched']:,}, explained by "
        f"{rc['shortfall_claims']:,} recorded claims without a matching severity row minus "
        f"{rc['surplus_rows']:,} severity rows beyond the recorded ClaimNb:\n"
    )
    w(
        _table(
            ["sev_status", "policies", "ClaimNb", "matched rows", "ClaimNb - rows"],
            [
                [
                    k,
                    f"{v['policies']:,}",
                    f"{v['claim_nb']:,}",
                    f"{v['matched_rows']:,}",
                    f"{v['gap']:,}",
                ]
                for k, v in rc["by_status"].items()
            ],
        )
    )
    w("")

    def disc_table(rows: list[dict[str, Any]], first: str) -> str:
        return _table(
            [
                first,
                "policies",
                "exposure",
                "ClaimNb",
                "matched rows",
                "ClaimNb - rows",
                "discrepant policies",
                "share",
            ],
            [
                [
                    r["label"],
                    f"{r['policies']:,}",
                    f"{r['exposure']:,.1f}",
                    f"{r['claim_nb']:,}",
                    f"{r['matched_rows']:,}",
                    f"{r['claims_gap']:,}",
                    f"{r['discrepancy_policies']:,}",
                    _pct(r["discrepancy_share"]),
                ]
                for r in rows
            ],
        )

    w("## 10. Discrepancy between recorded ClaimNb and matched rows\n")
    w("By IDpol range (bands from `audit.idpol_band_edges`):\n")
    w(disc_table(dc["idpol_bands"], "IDpol"))
    w("\nBy raw exposure band:\n")
    w(disc_table(dc["exposure_bands"], "raw exposure"))
    for f, rows in dc["factors"].items():
        w(f"\nBy {f}:\n")
        w(disc_table(rows, f))
    w("")

    w("## 11. High counts re-audited with N_i\n")
    w(
        f"Policies with N_i > {hcn['threshold']}: {_n(hcn['policies'], 'policy', 'policies')} (N_i {hcn['n_values']}, "
        f"exposures {[round(x, 3) for x in hcn['exposures']]}), {hcn['exposure']:.3f} policy-years, "
        f"{hcn['matched_rows']:,} matched rows ({hcn['claim_nb_recorded']:,} recorded claims), cost {hcn['cost']:,.2f} "
        f"(limited {hcn['limited_cost']:,.2f}, {_pct(hcn['cost_share_of_matched'])} of matched cost). Matched rows per "
        f"policy-year {hcn['n_per_exposure']:.2f} against {hcn['portfolio_n_per_exposure']:.3f} for the portfolio.\n"
    )

    w("## 12. Exposure bands with matched-row frequency\n")
    w(
        "Frequency = sum N_i / sum e (ratio of totals, not the mean of N_i / e); capped e is the modelling "
        "exposure, raw e the file value.\n"
    )
    w(
        _table(
            [
                "band (raw e)",
                "policies",
                "sum N_i",
                "sum e (capped)",
                "sum e (raw)",
                "N / e capped",
                "N / e raw",
                "recorded ClaimNb / e capped",
            ],
            [
                [
                    r["label"],
                    f"{r['policies']:,}",
                    f"{r['matched_rows']:,}",
                    f"{r['exposure']:,.1f}",
                    f"{r['exposure_raw']:,.1f}",
                    "-" if r["freq_n_per_exposure"] is None else f"{r['freq_n_per_exposure']:.4f}",
                    "-"
                    if r["freq_n_per_raw_exposure"] is None
                    else f"{r['freq_n_per_raw_exposure']:.4f}",
                    "-"
                    if r["freq_claim_nb_per_exposure"] is None
                    else f"{r['freq_claim_nb_per_exposure']:.4f}",
                ]
                for r in dc["exposure_bands"]
            ],
        )
    )
    w("")

    w("## 13. Per-claim cap table (matched claims only)\n")
    w(
        _table(
            [
                "cap",
                "claims above",
                "share of claims",
                "excess",
                "excess share of matched cost",
                "limited cost",
                "excess share of limited cost",
            ],
            [
                [
                    "uncapped" if c["cap"] is None else f"{c['cap']:,.0f}",
                    f"{c['claims_above']:,}",
                    _pct(c["share_of_claims"], 3),
                    f"{c['excess']:,.2f}",
                    _pct(c["excess_share_of_cost"]),
                    f"{c['limited_cost']:,.2f}",
                    _pct(c["excess_share_of_limited_cost"]),
                ]
                for c in sv["cap_table"]
            ],
        )
    )
    w("")

    w("## 14. Dataset version note\n")
    w(
        f"The OpenML file used here has {vn['file_policies']:,} policies and {vn['file_severity_rows']:,} severity rows. "
        f"The {vn['source']} reports {vn['external_policies']:,} policies and {vn['external_severity_rows']:,} severity rows; "
        f"this is an external claim, cited and not verified here. Policy count difference to the file: "
        f"{vn['policies_difference']:,}. Our matched severity rows: {vn['matched_rows']:,} "
        f"(difference to the external severity figure {vn['matched_minus_external_rows']:,}). The raw files are pinned "
        f"by dataset id, version and SHA-256 in `data/raw/manifest.json`.\n"
    )
    return "\n".join(o)


def run_audit(cfg: dict[str, Any]) -> dict[str, Any]:
    freq, sev = load_raw(cfg)
    au = compute_audit(freq, sev, cfg)
    results = Path(cfg["paths"]["results"])
    reports = Path(cfg["paths"]["reports"])
    results.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (results / "audit.json").write_text(
        json.dumps(au, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    (reports / "data_audit.md").write_text(render_markdown(au), encoding="utf-8")
    orphans = clean(freq, sev, cfg["clean"]).orphans.sort_values(["IDpol", "ClaimAmount"])
    orphans.to_csv(results / "orphan_ledger.csv", index=False)
    return au
