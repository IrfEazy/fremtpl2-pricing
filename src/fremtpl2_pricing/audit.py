"""Data audit: measured numbers (results/audit.json) and the report rendered from them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def _severity(sev: pd.DataFrame, a: dict[str, Any]) -> dict[str, Any]:
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
        "cap_candidates": [],
    }
    for c in a["large_loss_candidates"]:
        m = amt > c
        out["cap_candidates"].append(
            {
                "cap": c,
                "claims_above": int(m.sum()),
                "share_of_claims": float(m.mean()),
                "excess_share_of_cost": float((amt[m] - c).sum() / total),
            }
        )
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


def compute_audit(freq: pd.DataFrame, sev: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    a = cfg["audit"]
    factors = cfg["rating_factors"]
    res = clean(freq, sev, cfg["clean"])
    return {
        "dataset": {"freq_rows": int(len(freq)), "sev_rows": int(len(sev))},
        "exposure": _exposure(freq, a),
        "claim_nb": _claim_nb(freq, a),
        "severity": _severity(sev, a),
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
        "Items marked **for human review** are judgement calls; the milestone stops here for that review "
        "before any modelling.\n"
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
    w("\nExcess over candidate per-claim caps (evidence for the large-loss rule):\n")
    w(
        _table(
            ["cap", "claims above", "share of claims", "excess as share of total cost"],
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
        f"as the primary evaluation (PM-1) before this audit was run, and no prevalence threshold is used to "
        f"change it now. The random stratified split is kept as a labelled sensitivity. Currently selected "
        f"primary: `{au['split_primary']}`. Limitation: identical tuples can belong to different people and one "
        f"person's different tuples can still straddle the split, so this is a risk-profile-held-out "
        f"evaluation, not a policyholder-held-out one.\n"
    )

    w("## 7. Cleaning rules and justification\n")
    w(
        "All rules live in `configs/default.yaml` (`clean:`), are applied by one function "
        "(`fremtpl2_pricing.clean.clean`) to the full dataset before splitting, and treat every policy "
        "identically regardless of set.\n"
    )
    d = imp["duplicate_idpol"]
    w(
        f"**R0 duplicate IDpol** (`{rules['duplicate_idpol']}`): {d['rows']} duplicates measured, so nothing "
        f"is removed; the rule makes the pipeline fail if a future data version contains duplicates.\n"
    )
    ec = imp["exposure_cap"]
    if ec["cap"] is not None:
        w(
            f"**R1 exposure cap at {ec['cap']:g}**: {_n(ec['policies'], 'policy', 'policies')} "
            f"({_pct(ec['policies'] / au['dataset']['freq_rows'])}) have Exposure > {ec['cap']:g} "
            f"(maximum {ex['max']:.2f}), carrying {_n(ec['claims'], 'claim', 'claims')}; capping removes "
            f"{ec['exposure_removed']:,.1f} policy-years ({_pct(ec['exposure_removed'] / ex['total'], 3)} of exposure). "
            f"Claims per policy in these rows: {ex['gt1']['claims_per_policy']:.4f}, portfolio: "
            f"{ex['portfolio_claims_per_policy']:.4f}. Assumption (not testable from the data): Exposure above "
            f"{ec['cap']:g} year is a recording artefact and not additional risk time; the cap keeps the policy and "
            f"its claims. Dropping the rows instead would discard {_n(ec['claims'], 'claim', 'claims')}.\n"
        )
    em = imp["exposure_min"]
    small = ex["small"]
    base = ex["portfolio_claims_per_exposure_year"]
    if em["threshold"] is None:
        swc = [x["share_with_claim"] for x in small]
        w(
            f"**R2 minimum exposure: none (all small exposures kept)**. Measured: {_n(small[0]['rows'], 'policy', 'policies')} "
            f"({_pct(small[0]['share'])}) have Exposure <= {small[0]['threshold']}, with {_n(small[0]['claims'], 'claim', 'claims')} and an "
            f"annualised frequency of {small[0]['claims_per_exposure_year']:.2f} against {base:.3f} for the portfolio "
            f"(ratio {small[0]['claims_per_exposure_year'] / base:.1f}). Policies up to {small[-1]['threshold']} years total "
            f"{_n(small[-1]['rows'], 'row', 'rows')}, {_n(small[-1]['claims'], 'claim', 'claims')} ({_pct(small[-1]['claims'] / au['cleaning']['before']['claims'])} "
            f"of claims) and {_pct(small[-1]['exposure_share'], 3)} of exposure. The share of policies with a claim across the "
            f"thresholds ranges from {_pct(min(swc))} to {_pct(max(swc))}, against {_pct(ex['portfolio_share_with_claim'])} "
            f"for the portfolio. Nothing is excluded: any cut-off would be chosen after looking at outcomes. The "
            f"exposure offset and later sensitivity checks must handle these rows. "
            f"**For human review**: decide whether to set `clean.exposure_min` (candidate values are the thresholds "
            f"in `audit.exposure_small_thresholds`) and compare.\n"
        )
    else:
        w(
            f"**R2 minimum exposure {em['threshold']}**: {_n(em['policies'], 'policy', 'policies')}, {_n(em['claims'], 'claim', 'claims')} and "
            f"{em['exposure']:,.1f} policy-years dropped. **For human review.**\n"
        )
    nb = imp["claim_nb"]
    if nb["max"] is not None:
        w(
            f"**R3 ClaimNb {nb['action']} at {nb['max']}**: {_n(nb['policies'], 'policy', 'policies')} with ClaimNb > {nb['max']} "
            f"(values {cn['extreme_values']}, exposures {[round(x, 2) for x in cn['extreme_exposures']]}), "
            f"i.e. {_n(nb['excess_claims'], 'claim', 'claims')} above the cap "
            f"({_pct(nb['excess_claims'] / au['cleaning']['before']['claims'], 3)} of claims, "
            f"{_pct(nb['exposure'] / ex['total'], 3)} of exposure). Their claims per policy-year are "
            f"{cn['extreme_claims_per_exposure_year']:.2f} against {ex['portfolio_claims_per_exposure_year']:.3f} for the portfolio. "
            f"Capping keeps the policy and its exposure; policies at or below {nb['max']} are untouched. "
            f"The threshold {nb['max']} is a judgement call. **For human review.**\n"
        )
        w(
            f"Severity rows of these policies: {_n(nb['sev_rows'], 'row', 'rows')} with amount "
            f"{nb['sev_amount']:,.2f}. The ClaimNb cap changes only the frequency count; severity rows are never "
            f"capped or counted differently because of it (they are "
            f"{'kept' if nb['sev_rows_kept'] else 'dropped with the excluded policies'}), so on these policies "
            f"ClaimNb after the cap and the number of severity rows can differ (see targets below).\n"
        )
    sw = imp["sev_without_policy"]
    w(
        f"**R4 severity rows without a freq policy ({sw['action']})**: {_n(sw['rows'], 'row', 'rows')}, amount {sw['amount']:,.2f} "
        f"({_pct(sw['amount_share'])} of total cost). They have no rating factors, exposure or split side, so "
        f"they cannot be modelled or assigned to train/test; they are dropped and counted here. "
        f"**For human review**: whether to keep them out of all cost totals.\n"
    )
    sc = imp["sev_consistency"]
    tg = imp["targets"]
    w(
        f"**R5 ClaimNb versus severity rows**: claims_without_sev = {sc['claims_without_sev']['action']} "
        f"({_n(sc['claims_without_sev']['policies'], 'policy', 'policies')}, {_n(sc['claims_without_sev']['claims'], 'claim', 'claims')}, "
        f"{_pct(lk['claims_without_sev_share_of_claims'])} of claims), sev_without_claims = "
        f"{sc['sev_without_claims']['action']} ({_n(sc['sev_without_claims']['policies'], 'policy', 'policies')}), count_mismatch = "
        f"{sc['count_mismatch']['action']} ({_n(sc['count_mismatch']['policies'], 'policy', 'policies')}). Policies are kept "
        f"(flagged in `sev_status`) because dropping them would delete {_pct(lk['claims_without_sev_share_of_claims'])} "
        f"of observed claims from the frequency model. Assumption: claims without severity rows were closed "
        f"without payment or not yet valued; the data cannot confirm this. **For human review.**\n"
    )
    w("### Target definitions (config `clean.targets`; for human review)\n")
    w(
        f"- Severity sample: policies with at least one severity row ({_n(tg['severity_sample_policies'], 'policy', 'policies')}, "
        f"{_n(tg['severity_sample_sev_rows'], 'row', 'rows')}). Severity weight (`{tg['severity_weight']}`): "
        f"{'the number of severity rows' if tg['severity_weight'] == 'sev_rows' else 'ClaimNb after the cap'}; "
        f"total weight {tg['severity_weight_total']:,}."
    )
    w(
        f"- Frequency count (`{tg['frequency_count']}`): "
        f"{'ClaimNb as recorded, after the ClaimNb cap' if tg['frequency_count'] == 'claim_nb' else 'the number of severity rows'}; "
        f"total {tg['frequency_count_total']:,} against {tg['claims_total']:,} ClaimNb after cleaning."
    )
    w(
        f"- Pure-premium observed cost: the sum of the capped severity amounts of the policy, total "
        f"{tg['cost_total']:,.2f}; claims without severity rows contribute zero cost "
        f"(`{tg['pure_premium_claims_without_sev']}`; sample {_n(tg['pure_premium_sample_policies'], 'policy', 'policies')}).\n"
    )
    zero_cost = tg["pure_premium_claims_without_sev"] == "zero_cost"
    tweedie = (
        "treats them as zero cost"
        if zero_cost
        else "drops those policies from the pure-premium sample"
    )
    cfg_mark = {
        m: "configured" if tg["frequency_count"] == m else "alternative"
        for m in ("claim_nb", "sev_rows")
    }
    consistent = tg["frequency_count"] == "sev_rows"
    w(
        f"Consequence: {_n(tg['claims_without_sev_rows'], 'claim', 'claims')} "
        f"({_pct(tg['claims_without_sev_rows_share'])} of ClaimNb after cleaning) have no severity row. "
        f"A frequency x severity pure premium counts them at the average severity "
        f"(cost per severity row {tg['cost_per_sev_row']:,.2f}), whereas a Tweedie / pure-premium target "
        f"{tweedie} (`pure_premium_claims_without_sev: {tg['pure_premium_claims_without_sev']}`). "
        f"The two routes are therefore not on the same target unless the frequency "
        f"count is restricted to claims with severity rows. If those claims were valued at the average cost per "
        f"severity row, the cost missing from the Tweedie target would be {tg['implied_cost_at_mean_severity']:,.2f}, "
        f"{_pct(tg['implied_cost_share_of_total'])} of the total (this is an illustration under that assumption, "
        f"not a measured loss). Options, **not chosen here: for human review**: (a) frequency on ClaimNb as recorded "
        f"(`frequency_count: claim_nb`, {cfg_mark['claim_nb']}; keeps all claims, inconsistent with Tweedie); "
        f"(b) frequency on the count of severity rows (`frequency_count: sev_rows`, {cfg_mark['sev_rows']}; "
        f"consistent with Tweedie, drops the claims without severity rows from the frequency target). "
        f"The configured value (`{tg['frequency_count']}`"
        f"{', which matches the sev_rows route' if consistent else ''}) is kept only so downstream "
        f"work has a defined target.\n"
    )
    ll = imp["large_loss"]
    if ll["cap"] is not None:
        w(
            f"**R6 per-claim cap at {ll['cap']:,.0f}**: {_n(ll['claims_capped'], 'claim', 'claims')} exceed it "
            f"({_pct(ll['claims_capped'] / sv['rows'], 3)} of claims) and the capped excess is "
            f"{ll['amount_removed']:,.2f} ({_pct(ll['amount_removed_share'])} of total cost). The largest single claim is "
            f"{_pct(sv['top'][0]['share'])} of total cost and the ten largest together {_pct(sv['top_share'])}. "
            f"{_cap_position(ll['cap'], sv['quantiles'])}; "
            f"the table in section 3 shows how much cost moves for other caps (the table counts all raw severity rows, including the R4 rows without a policy, which are removed before capping). The choice is a judgement call between stability and bias: capped targets understate "
            f"pure premium by the removed share, which must be reported separately. "
            f"Uncapped amounts are kept in `ClaimAmount_uncapped`. **For human review.**\n"
        )
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
    return au
