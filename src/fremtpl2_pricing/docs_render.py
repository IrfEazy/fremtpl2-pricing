"""Generated documentation (``fremtpl2 docs``): the model report and the README / model card blocks.

Renders ``reports/model_report.md`` (fully generated) and the blocks between HTML-comment markers
inside ``README.md`` and ``MODEL_CARD.md`` only from tracked pipeline files: ``results/metrics.json``
and the completed attempt it points to (verified by :func:`report.load_completed`, the same hash check
as ``fremtpl2 report``), ``cv_glm.json``, ``cv_gbm.json``, ``cv_sensitivities.json``,
``interpretation.json``, ``audit.json``, ``split_summary.json``, ``targets_summary.json``, the GLM
relativity tables, ``gbm_models.json`` and, when present, ``reproduction.json``. Every rendered number
is read from those files and only formatted (digits per quantity kind come from ``configs/docs.yaml``);
text outside the markers is preserved byte for byte. Output is deterministic; ``--check`` renders to
memory and reports drift without writing anything. No model is loaded and no data row is read.
"""

# ruff: noqa: E501  (prose templates; the numbers inside are formatted placeholders)
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from .freeze import project_paths
from .report import DECISION_RULE, PRIMARY_CONTRAST, _table, load_completed


class DocsError(ValueError):
    """Raised for a missing input, a missing or duplicated marker, or an unknown block."""


# --------------------------------------------------------------------------- inputs


def _json(path: Path) -> Any:
    if not path.is_file():
        raise DocsError(f"input file is missing: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    """Every input file, the attempt verified by the report loader; reproduction is optional."""
    dc, paths = cfg["docs"], project_paths(cfg, root)
    res, ins = root / cfg["paths"]["results"], cfg["docs"]["inputs"]
    repro = res / ins["reproduction"]
    return {
        "attempt": load_completed(cfg, root),
        "glm": _json(root / paths["cv_glm"]),
        "gbm": _json(root / paths["cv_gbm"]),
        "gbm_models": _json(root / paths["gbm_models"]),
        "audit": _json(root / paths["audit"]),
        "sens": _json(res / ins["sensitivities"]),
        "interp": _json(res / ins["interpretation"]),
        "split": _json(res / ins["split_summary"]),
        "targets": _json(res / ins["targets_summary"]),
        "relativities": {f: _csv(res / f) for f in ins["relativities"]},
        "reproduction": json.loads(repro.read_text(encoding="utf-8")) if repro.is_file() else None,
        "dc": dc,
    }


def _csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DocsError(f"input file is missing: {path.name}")
    return pd.read_csv(path)


class _Fmt:
    """Number formatting; digits per quantity kind from the docs config."""

    def __init__(self, dc: dict[str, Any]) -> None:
        self.dec = dc["decimals"]

    def __call__(self, kind: str, x: Any) -> str:
        if x is None:
            return "n/a"
        if kind == "sci":
            return f"{x:.{self.dec[kind]}e}"
        if kind == "money":
            return f"{x:,.{self.dec[kind]}f}".replace(",", " ")
        if kind == "count":
            return f"{int(x):,}".replace(",", " ")
        if kind == "share":
            return f"{100 * x:.{self.dec[kind] - 2}f}%"
        return f"{x:.{self.dec[kind]}f}"


# --------------------------------------------------------------------------- test tables


def _cands(i: dict[str, Any]) -> list[str]:
    m, want = i["attempt"]["metrics"], i["dc"]["candidates"]
    order = list(m["candidates_order"])
    return order if want == "all" else [c for c in order if c in want]


def _gini_key(m: dict[str, Any], cid: str) -> str | None:
    """Bootstrap interval key of the normalised Gini; only the selected pair has one."""
    return {
        m["selected_gbm_id"]: "gini_normalized:gbm",
        m["selected_glm_id"]: "gini_normalized:glm",
    }.get(cid)


def _ci(f: _Fmt, kind: str, iv: dict[str, float] | None) -> str:
    return "n/a" if iv is None else f"[{f(kind, iv['lower'])}, {f(kind, iv['upper'])}]"


def _cells(i: dict[str, Any], f: _Fmt, cid: str) -> dict[str, str]:
    m = i["attempt"]["metrics"]
    iv, c = m["bootstrap"]["group"]["intervals"], m["candidates"][cid]
    gk = _gini_key(m, cid)
    return {
        "candidate": f"`{cid}`",
        "loss": f("loss", c["primary_loss"]),
        "loss_ci": _ci(f, "loss", iv.get(f"loss:{cid}")),
        "gini": f("gini", c["gini_normalized"]),
        "gini_ci": _ci(f, "gini", iv.get(gk)) if gk else "n/a",
        "ratio": f("ratio", c["calibration_totals"]["ratio"]),
    }


def _power(i: dict[str, Any]) -> str:
    """Reference Tweedie power of the primary loss, read from the recorded candidate metrics."""
    cs = i["attempt"]["metrics"]["candidates"]
    powers = {str(cs[c]["tweedie_power"]) for c in _cands(i)}
    if len(powers) != 1:
        raise DocsError(f"candidates disagree on the primary-loss Tweedie power: {sorted(powers)}")
    return powers.pop()


def _test_table(i: dict[str, Any], f: _Fmt, columns: list[str]) -> list[str]:
    heads = dict(i["dc"]["columns"])
    heads["loss"] = heads["loss"].replace("{power}", _power(i))
    rows = [[_cells(i, f, cid)[k] for k in columns] for cid in _cands(i)]
    return _table([heads[k] for k in columns], rows)


def _contrast_lines(i: dict[str, Any], f: _Fmt) -> list[str]:
    m = i["attempt"]["metrics"]
    g, rows = m["bootstrap"]["group"], m["rows"]
    d, gd = g["delta"], g["gini_difference"]
    return [
        f"Population: the primary grouped test fold, {f('count', rows['policies'])} policies, "
        f"{f('exposure', rows['exposure'])} exposure-years, {f('count', rows['claims'])} claims on "
        f"{f('count', rows['claim_policies'])} policies, {f('count', m['groups']['groups'])} "
        "rating-profile groups.",
        "",
        f"Primary contrast Delta = L(`{m['selected_gbm_id']}`) - L(`{m['selected_glm_id']}`) = "
        f"{f('delta', d['point'])} (95% group-bootstrap CI [{f('delta', d['lower'])}, "
        f"{f('delta', d['upper'])}]): **{d['decision']}**.",
        "",
        f"Gini difference (normalised, secondary) = {f('delta', gd['point'])} (95% CI "
        f"[{f('delta', gd['lower'])}, {f('delta', gd['upper'])}]).",
        "",
    ]


# --------------------------------------------------------------------------- blocks


def block_results(i: dict[str, Any], f: _Fmt) -> list[str]:
    return [*_contrast_lines(i, f), *_test_table(i, f, i["dc"]["results_columns"])][:-1]


def block_metrics(i: dict[str, Any], f: _Fmt) -> list[str]:
    return [*_contrast_lines(i, f), *_test_table(i, f, i["dc"]["model_card_columns"])][:-1]


def block_reproduction(i: dict[str, Any], f: _Fmt) -> list[str]:
    r = i["reproduction"]
    if r is None:
        return ["Reproduction: not yet recorded (`results/reproduction.json` is absent)."]
    n = i["dc"]["short_commit"]
    mins = lambda s: f("minutes", s / 60)  # noqa: E731
    out = [
        f"Verdict: **{r['verdict']}** on commit `{r['tested_commit'][:n]}`; total "
        f"{mins(r['total_seconds'])} minutes. Machine: {r['platform']} ({r['machine']}, "
        f"{r['cpu_count']} CPUs), Python {r['python']}.",
        "",
    ]
    out += _table(["Step", "Minutes"], [[s["name"], mins(s["seconds"])] for s in r["steps"]])
    return out[:-1]


def _top(shap: dict[str, Any], f: _Fmt, n: int) -> str:
    ma, sh = shap["mean_abs"], shap["mean_abs_share"]
    return ", ".join(
        f"{k} ({f('contribution', ma[k])}, {f('share', sh[k])})" for k in shap["ranking"][:n]
    )


def block_drivers(i: dict[str, Any], f: _Fmt) -> list[str]:
    n = i["dc"]["top_drivers"]
    rows = [[c, _top(s, f, n)] for c, s in i["interp"]["shap"].items()]
    head = ["GBM component", "Top drivers (mean absolute contribution, share)"]
    smp = i["interp"]["sample"]
    cap = [
        f"Mean absolute TreeSHAP contribution to the raw score (log scale) on a fixed train sample of "
        f"{f('count', smp['size'])} of {f('count', smp['train_policies'])} train policies; no test row.",
        "",
    ]
    return [*cap, *_table(head, rows)[:-1]]


BLOCKS: dict[str, Callable[[dict[str, Any], _Fmt], list[str]]] = {
    "results": block_results,
    "metrics": block_metrics,
    "reproduction": block_reproduction,
    "drivers": block_drivers,
}


# --------------------------------------------------------------------------- model report


def _kv(rows: list[tuple[str, str]]) -> list[str]:
    return _table(["Item", "Value"], [[a, b] for a, b in rows])


def sec_data(i: dict[str, Any], f: _Fmt) -> list[str]:
    a, imp = i["audit"], i["audit"]["cleaning"]["impact"]
    rec, rules = a["reconciliation"], a["cleaning"]["rules"]
    sc, ec = imp["sev_consistency"], imp["exposure_cap"]
    rows = [
        (
            "Policies / severity rows read",
            f"{f('count', a['dataset']['freq_rows'])} / {f('count', a['dataset']['sev_rows'])}",
        ),
        (
            "Duplicate IDpol (rule: " + rules["duplicate_idpol"] + ")",
            f("count", imp["duplicate_idpol"]["rows"]),
        ),
        ("Severity rows quarantined (non-finite or <= 0)", f("count", rec["quarantined_rows"])),
        ("Exposure rows quarantined (<= 0)", f("count", rec["exposure_quarantined_rows"])),
        (
            f"Orphan severity rows (rule: {rules['sev_without_policy']}), amount",
            f"{f('count', rec['orphan_rows'])}, {f('money', rec['orphan_cost'])}",
        ),
        (
            f"Exposure capped at {rules['exposure_cap']}: policies, exposure removed",
            f"{f('count', ec['policies'])}, {f('exposure', ec['exposure_removed'])}",
        ),
        (
            "Policies with claims but no severity row (flagged), claims",
            f"{f('count', sc['claims_without_sev']['policies'])}, "
            f"{f('count', sc['claims_without_sev']['claims'])}",
        ),
        ("Count mismatches (flagged): policies", f("count", sc["count_mismatch"]["policies"])),
        (
            "Matched severity rows, amount",
            f"{f('count', rec['matched_rows'])}, {f('money', rec['matched_cost'])}",
        ),
        (
            f"Policies with more than {a['high_counts']['threshold']} claims (kept in the primary), "
            "severity rows",
            f"{f('count', a['high_counts']['policies'])}, {f('count', a['high_counts']['matched_rows'])}",
        ),
    ]
    return [
        "## 1. Data and cleaning decisions",
        "",
        "Counts from `results/audit.json`; cleaning runs on the full data before the split.",
        "",
        *_kv([(k, v) for k, v in rows]),
    ]


def sec_split(i: dict[str, Any], f: _Fmt) -> list[str]:
    s, lk = i["split"], i["audit"]["leakage"]
    sets = s["primary_sets"]
    rows = [
        [
            name,
            f("count", v["policies"]),
            f("exposure", v["exposure"]),
            f("count", v["claims_n"]),
            f("count", v["policies_with_claim_rows"]),
        ]
        for name, v in sets.items()
    ]
    return [
        "## 2. Split",
        "",
        f"The primary split is `{s['primary']}` (group = tuple of all rating factors; seed {s['seed']}, "
        f"{s['settings']['test_folds']} test folds, test fold {s['settings']['test_fold']}, "
        f"{s['settings']['cv_folds']} CV folds on train); the random split is a labelled sensitivity "
        f"(test share {f('share', s['random_test_share_of_policies'])}). Test share of policies, "
        f"grouped: {f('share', s['test_share_of_policies'])}. Policy overlap train/test: "
        f"{s['overlap_primary_train_test']['idpol']}; group overlap: "
        f"{s['overlap_primary_train_test']['group_id']}.",
        "",
        *_table(
            ["Set", "Policies", "Exposure-years", "Claims (N)", "Policies with claim rows"], rows
        ),
        f"Groups: {f('count', lk['distinct_groups'])} distinct rating profiles; "
        f"{f('share', lk['share_policies_in_multi_groups'])} of the policies sit in groups of more than one "
        f"policy; largest group {lk['largest_group']} policies.",
        "",
    ]


def sec_targets(i: dict[str, Any], f: _Fmt) -> list[str]:
    t, ll, el = i["targets"]["targets"], i["targets"]["large_loss"], i["targets"]["excess_loading"]
    rows = [
        (
            "Frequency target N",
            f"`{t['frequency_count']}`: {f('count', t['frequency_count_total'])} "
            f"matched rows (recorded ClaimNb total {f('count', t['claim_nb_recorded_total'])}; "
            f"{f('count', t['discrepancy_policies'])} policies differ)",
        ),
        ("Total cost S (matched)", f("money", t["cost_total"])),
        (
            f"Capped cost S_cap (cap {f('count', ll['cap'])} per claim, before aggregation)",
            f("money", t["limited_cost_total"]),
        ),
        (
            "Excess S - S_cap",
            f"{f('money', t['excess_total'])} ({f('share', t['excess_share_of_cost'])} of S; "
            f"{f('count', ll['claims_capped'])} claims capped)",
        ),
        (
            f"Excess loading ({el['mode']}, primary train set only)",
            f"{f('loading', el['value_per_exposure_year'])} per exposure-year",
        ),
    ]
    return [
        "## 3. Targets, cap and excess loading",
        "",
        "From `results/targets_summary.json`.",
        "",
        *_kv(rows),
    ]


def _pp_rows(pooled: dict[str, Any], f: _Fmt) -> list[list[str]]:
    return [
        [
            k,
            f("loss", v["tweedie_ref_deviance_per_exposure"]),
            f("gini", v["gini_normalized"]),
            f("ratio", v["totals"]["ratio"]),
        ]
        for k, v in pooled["pure_premium"].items()
    ]


def sec_glm(i: dict[str, Any], f: _Fmt) -> list[str]:
    g = i["glm"]
    pooled = g["cv"]["pooled"]
    fr = [
        [
            k,
            f("loss", v["deviance_per_exposure"]),
            f("ratio", v["deviance_explained"]),
            f("gini", v["gini_normalized"]),
        ]
        for k, v in pooled["frequency"].items()
    ]
    sv = [
        [k, f("loss", v["deviance_per_claim"]), f("ratio", v["deviance_explained"])]
        for k, v in pooled["severity"].items()
    ]
    cv = g["convergence"]
    rel = [
        [
            f"`{name}`",
            f("count", len(t)),
            f("ratio", t["relativity"].min()),
            f("ratio", t["relativity"].max()),
        ]
        for name, t in i["relativities"].items()
    ]
    return [
        f"## 4. GLM cross-validation (train only, grouped {g['config']['cv_folds']}-fold)",
        "",
        f"{f('count', g['rows']['train_policies'])} train policies in {g['config']['cv_folds']} grouped folds; "
        f"{cv['n_fits']} fits, {cv['n_not_converged']} not converged. Pooled out-of-fold results.",
        "",
        "Frequency (Poisson deviance per exposure-year):",
        "",
        *_table(["Model", "Deviance", "Deviance explained", "Normalised Gini"], fr),
        "Severity (gamma deviance per claim):",
        "",
        *_table(["Model", "Deviance", "Deviance explained"], sv),
        "Pure premium (Tweedie deviance at the reference power per exposure-year):",
        "",
        *_table(
            ["Model", "Tweedie deviance", "Normalised Gini", "Observed / predicted"],
            _pp_rows(pooled, f),
        ),
        "Final-refit relativity tables (`exp(coef)` of every term, intercept included):",
        "",
        *_table(["Table", "Terms", "Smallest relativity", "Largest relativity"], rel),
    ]


def _step_row(s: dict[str, Any], f: _Fmt, a: str, b: str) -> list[str]:
    return [
        f"{a} -> {b}",
        f("delta", s["mean_d"]),
        f("delta", s["sd_d"]),
        f("delta", s["threshold"]),
        s["decision"],
    ]


def sec_selection(i: dict[str, Any], f: _Fmt) -> list[str]:
    s, fin = i["glm"]["selection"], i["glm"]["final"]["power_cv"]
    lad, fv = s["ladder"], s["freq_x_sev_vs_tweedie"]
    steps = [_step_row(st, f, st["incumbent"], st["next"]) for st in lad["steps"]]
    fs = [_step_row(st, f, *st["candidates"]) for st in fv["steps"]]
    head = ["Step", "Mean d", "SD d", "Threshold sd/sqrt(K)", "Decision"]
    losses = [
        [k, f("loss", v), f("ratio", s["severity_deviance_per_claim_mean"][k])]
        for k, v in s["pure_premium_loss_mean"].items()
    ]
    return [
        "## 5. Selections (protocol section 4) and Tweedie power",
        "",
        f"Rule `{lad['rule_version']}`; d = loss(incumbent) - loss(next) per fold, K = {lad['n_folds']}. "
        "A step replaces the incumbent iff mean(d) > sd/sqrt(K).",
        "",
        "Severity ladder (frequency fixed at the full GLM):",
        "",
        *_table(["Rung", "Pure-premium loss (mean)", "Severity deviance per claim (mean)"], losses),
        *_table(head, steps),
        f"Selected severity rung: **{s['selected_severity']}**.",
        "",
        "Frequency x severity against Tweedie (GLM):",
        "",
        *_table(head, fs),
        f"Selected GLM pure-premium candidate: **{fv['selected']}**.",
        "",
        "GBM pure-premium candidate: "
        + ", ".join(
            f"**{st['decision']}** (mean d {f('delta', st['mean_d'])}, threshold {f('delta', st['threshold'])})"
            for st in i["gbm"]["selection"]["gbm_pure_premium"]["steps"]
        )
        + ".",
        "",
        f"Tweedie power: {fin['power']} chosen on the held-out Tweedie log-likelihood, at the grid edge: "
        f"{fin['at_grid_edge']}; best minus next neighbour {f('loss', fin['best_minus_next_nats'])} nats; "
        f"selected by {fin['stability']['folds_selecting_final']} of {fin['stability']['n_folds']} outer "
        f"folds (powers per fold: {', '.join(str(p) for p in fin['stability']['selected_by_outer_fold'])}).",
        "",
    ]


def sec_gbm(i: dict[str, Any], f: _Fmt) -> list[str]:
    g = i["gbm"]
    rows = [
        [
            c,
            ", ".join(f"{k}={v}" for k, v in v_["chosen_hp"].items()),
            v_["rounds"],
            len(v_["grid"]),
            f("loss", v_["mean_cv_loss_optimistic"])
            if c == "tweedie"
            else f("gini", v_["mean_cv_loss_optimistic"]),
            v_["cap_reached_any_fold"],
        ]
        for c, v_ in g["components"].items()
    ]
    mon = [
        [c, m["monotone_constraints"], m["rounds"]] for c, m in g["final_models"]["models"].items()
    ]
    return [
        "## 6. GBM tuning and selected hyperparameters",
        "",
        f"LightGBM frequency, severity and Tweedie components tuned on the same {g['config']['cv_folds']} "
        f"registered folds (seed {g['config']['gbm']['seed']}). {g['config']['scores_label']}.",
        "",
        *_table(
            [
                "Component",
                "Chosen hyperparameters",
                "Rounds",
                "Grid points",
                "Mean CV loss (optimistic)",
                "Round cap reached",
            ],
            rows,
        ),
        "Final refit on train (monotone constraints in the feature order of the model file):",
        "",
        *_table(["Component", "monotone_constraints", "Rounds"], mon),
        f"GBM pure-premium candidate selected in CV: **{g['selection']['selected']}**.",
        "",
    ]


def _eff(d: dict[str, Any], f: _Fmt) -> str:
    return f"{f('delta', d['mean'])} ({d['folds_unconstrained_better']} of {d['folds']} folds better unconstrained)"


def sec_constraints(i: dict[str, Any], f: _Fmt) -> list[str]:
    cr, mono = i["interp"]["constraint_register"], i["interp"]["monotonicity"]
    ce = cr["cv_effect"]
    sep, abl = ce["separately_tuned_procedures"], ce["controlled_ablation"]
    cons = [k for k, v in cr["constrained_components"].items() if v]
    ver = cr["verification"]["min_step_raw_score_by_component"]
    rows = [
        [
            "separately tuned",
            "frequency x severity (primary loss)",
            _eff(sep["diff_freq_x_sev_unconstrained_minus_constrained"], f),
        ],
        [
            "separately tuned",
            "Tweedie (primary loss)",
            _eff(sep["diff_tweedie_unconstrained_minus_constrained"], f),
        ],
        [
            "controlled ablation",
            "frequency (component loss)",
            _eff(abl["frequency"]["component_loss_unconstrained_minus_constrained"], f),
        ],
        [
            "controlled ablation",
            "frequency x severity (primary loss)",
            _eff(abl["frequency"]["primary_loss_unconstrained_minus_constrained"], f),
        ],
        [
            "controlled ablation",
            "Tweedie (primary loss)",
            _eff(abl["tweedie"]["primary_loss_unconstrained_minus_constrained"], f),
        ],
    ]
    mrows = [
        [
            c,
            v["constraint"],
            f("count", v["pure_premium"]["rows_with_decrease"]),
            f("share", v["pure_premium"]["share_rows_with_decrease"]),
        ]
        for c, v in mono["candidates"].items()
    ]
    return [
        "## 7. Monotone constraints and their measured CV effect",
        "",
        f"Constraint: {cr['feature']} non-decreasing in the components {', '.join(cons)}. {cr['rationale']}",
        "",
        "Smallest step of the raw-score sweep (a negative value would be a violation): "
        + "; ".join(f"{c} {f('sci', v['min_step_raw_score'])}" for c, v in ver.items())
        + ".",
        "",
        f"CV effect, unconstrained minus constrained (negative = the unconstrained variant is better). {ce['kind']}.",
        "",
        *_table(["Comparison", "Quantity", "Mean difference"], rows),
        ce["separately_tuned_note"].capitalize() + ".",
        "",
        f"{mono['statement']}",
        "",
        f"BonusMalus sweep of the pure premium over {mono['n_points']} train values:",
        "",
        *_table(["Candidate", "Constraint", "Rows with a decrease", "Share"], mrows),
    ]


def sec_sensitivities(i: dict[str, Any], f: _Fmt) -> list[str]:
    rows = []
    for name, s in i["sens"]["sensitivities"].items():
        pp = s["pooled"]["pure_premium"]
        rows.append(
            [
                f"`{name}`",
                f("count", s["rows"]["policies"]),
                f("count", s["rows"]["claims"]),
                f("loss", pp.get("freq_x_sev", {}).get("tweedie_ref_deviance_per_exposure")),
                f("loss", pp["tweedie"]["tweedie_ref_deviance_per_exposure"]),
                f("gini", pp.get("freq_x_sev", {}).get("gini_normalized")),
                "yes" if s["freq_x_sev_valid"] else "no",
            ]
        )
    return [
        "## 8. Cleaning sensitivities (CV only)",
        "",
        "Declared in `clean.sensitivities`; they change cleaning and targets only, never the split. "
        "Each row is a full grouped CV on train with the Tweedie power fixed at "
        f"{i['sens']['config']['fixed_tweedie_power']}; losses are on each variant's own targets and are not "
        "comparable across rows with a different cap or exposure rule.",
        "",
        *_table(
            [
                "Variant",
                "Train policies",
                "Claims",
                "Freq x sev Tweedie dev.",
                "Tweedie GLM dev.",
                "Freq x sev Gini",
                "Freq x sev valid",
            ],
            rows,
        ),
    ]


def sec_interpretation(i: dict[str, Any], f: _Fmt) -> list[str]:
    ip = i["interp"]
    n = i["dc"]["top_drivers"]
    smp, ia = ip["sample"], ip["interaction"]
    rows = [[c, _top(s, f, n), f("sci", s["max_abs_row_error"])] for c, s in ip["shap"].items()]
    return [
        "## 9. Interpretation (train only)",
        "",
        f"Fixed train sample of {f('count', smp['size'])} of {f('count', smp['train_policies'])} policies "
        f"(seed {smp['seed']}). Top drivers by mean absolute TreeSHAP contribution on the raw score; the "
        "per-row identity (contributions plus base value equal the raw score) holds to the recorded error.",
        "",
        *_table(
            [
                "GBM component",
                f"Top {n} drivers (mean absolute contribution, share)",
                "Max identity error",
            ],
            rows,
        ),
        f"GLM comparison: group means of GBM contributions against the log relativities of the selected GLM "
        f"(`{ip['selected_glm_id']}`) are in `reports/interpretation.md` (aligned log scale, not coefficients).",
        "",
        f"DrivAge x VehPower note ({ia['component']} component): {ia['label']}. Range of the young-driver "
        f"effect across VehPower bands: {f('contribution', ia['young_effect_range_across_vehpower_bands'])}.",
        "",
    ]


def sec_test(i: dict[str, Any], f: _Fmt) -> list[str]:
    m, c = i["attempt"]["metrics"], i["attempt"]["complete"]
    lp = m["loaded_premium"]
    cal = [
        [
            f"`{cid}`",
            f("money", m["candidates"][cid]["calibration_totals"]["observed"]),
            f("money", m["candidates"][cid]["calibration_totals"]["predicted"]),
            f("ratio", m["candidates"][cid]["calibration_totals"]["ratio"]),
        ]
        for cid in _cands(i)
    ]
    loaded = [
        [
            f"`{cid}`",
            f("money", lp["by_candidate"][cid]["predicted_loaded_total"]),
            f("ratio", lp["by_candidate"][cid]["ratio_observed_uncapped_to_loaded"]),
        ]
        for cid in _cands(i)
    ]
    return [
        "## 10. The single test evaluation",
        "",
        f"Attempt {m['attempt']} (`{m['attempt_id']}`), verified against the journal COMPLETE event "
        f"(`{c['outputs_sha256'][:16]}...`). Models are frozen; the journal lists {len(i['attempt']['attempts'])} attempt(s).",
        "",
        PRIMARY_CONTRAST,
        "",
        DECISION_RULE,
        "",
        *_contrast_lines(i, f),
        f"All {len(_cands(i))} candidates (descriptive; only the primary contrast carries a decision):",
        "",
        *_test_table(i, f, list(i["dc"]["columns"])),
        "Calibration (capped pure premium, total observed against predicted):",
        "",
        *_table(["Candidate", "Observed", "Predicted", "Observed / predicted"], cal),
        f"Loaded premium (reported separately; capped prediction plus the training-only flat excess loading "
        f"{f('loading', lp['loading_per_exposure_year'])} per exposure-year; observed uncapped total "
        f"{f('money', lp['observed_uncapped_total'])}):",
        "",
        *_table(["Candidate", "Predicted loaded total", "Observed uncapped / loaded"], loaded),
    ]


def sec_limits(i: dict[str, Any], f: _Fmt) -> list[str]:
    m, a, t = i["attempt"]["metrics"], i["audit"], i["targets"]
    ip = i["interp"]
    d, pol = m["bootstrap"]["group"]["delta"], m["bootstrap"]["policy"]["delta"]
    axes = [
        ax for c in ip["constraint_register"]["grid_edges"].values() for ax in c["axes"].values()
    ]
    unknown = {ax["position"] for ax in axes} - {
        "lowest grid value",
        "highest grid value",
        "interior",
    }
    if unknown:
        raise DocsError(f"unknown grid-edge position(s) in interpretation.json: {sorted(unknown)}")
    edge = sum(ax["position"] != "interior" for ax in axes)
    gt = m["candidates"][m["selected_gbm_id"]]["calibration_totals"]["ratio"]
    gl = m["candidates"][m["selected_glm_id"]]["calibration_totals"]["ratio"]
    lens = "/".join(str(n) for n in sorted({len(ax["grid"]) for ax in axes}))
    sev_expl = i["glm"]["cv"]["pooled"]["severity"]["full"]["deviance_explained"]
    mono = ip["monotonicity"]["candidates"][m["selected_gbm_id"]]["pure_premium"]
    ia = ip["interaction"]
    gbm_ids = [c for c in _cands(i) if c.startswith("gbm_")]
    gbm_cal = ", ".join(
        f"`{c}` {f('ratio', m['candidates'][c]['calibration_totals']['ratio'])}" for c in gbm_ids
    )
    glm_ids = [c for c in _cands(i) if c.startswith("glm_") and c != m["selected_glm_id"]]
    glm_best = min(glm_ids, key=lambda c: m["candidates"][c]["primary_loss"])
    glm_best_loss = m["candidates"][glm_best]["primary_loss"]
    sel_loss = m["candidates"][m["selected_glm_id"]]["primary_loss"]
    lr = [
        m["loaded_premium"]["by_candidate"][c]["ratio_observed_uncapped_to_loaded"]
        for c in _cands(i)
    ]
    out = [
        "## 11. Limits",
        "",
        f"- Intervals are conditional on the frozen fitted models; retraining, tuning and selection "
        f"variability are not included. The group-bootstrap CI of Delta is [{f('delta', d['lower'])}, "
        f"{f('delta', d['upper'])}]; the policy-level sensitivity is narrower ([{f('delta', pol['lower'])}, "
        f"{f('delta', pol['upper'])}]) because it ignores dependence inside rating-profile groups.",
        "- GBM CV scores are optimistic tuning scores (early stopping and grid scoring share the folds).",
        f"- The GBM hyperparameter grid was committed before tuning ({lens} values per axis): {edge} of "
        f"{len(axes)} chosen values lie on the lowest or highest grid value, so the optimum may lie outside the grid.",
        f"- The selected GBM `{m['selected_gbm_id']}` is not guaranteed monotone in {ip['monotonicity']['feature']}: "
        f"{f('share', mono['share_rows_with_decrease'])} of the sweep rows show a decrease (largest relative "
        f"decrease {f('share', mono['largest_relative_decrease'])}); reported, never retuned.",
        f"- Test calibration (observed / predicted): selected GBM `{m['selected_gbm_id']}` {f('ratio', gt)}, "
        f"selected GLM `{m['selected_glm_id']}` {f('ratio', gl)}; a ratio above 1 means total under-prediction "
        f"(by {f('share', gt - 1)} for the selected GBM). Across the GBM candidates: {gbm_cal}. "
        "These are descriptive totals on one test fold, not a decision criterion.",
        f"- Descriptive only, no decision: on the test fold `{glm_best}` has a primary loss "
        f"{f('loss', glm_best_loss)} against {f('loss', sel_loss)} for the selected GLM "
        f"`{m['selected_glm_id']}` (difference {f('loss', sel_loss - glm_best_loss)}; lower is better). "
        "The GLM was selected on train CV before the test evaluation and is not re-chosen on test.",
        f"- GLM interaction add-back is deferred: {ia['label']} (interpretation.json).",
        f"- Loaded premium (flat excess loading): observed uncapped / loaded ranges from "
        f"{f('ratio', min(lr))} to {f('ratio', max(lr))} across the {len(lr)} candidates, so no candidate's "
        "loaded total matches the observed uncapped total.",
        f"- Claims are capped at {f('count', t['large_loss']['cap'])}: {f('share', t['large_loss']['amount_removed_share'])} "
        f"of the matched cost sits above the cap and enters only through a flat loading, "
        f"{f('loading', t['excess_loading']['value_per_exposure_year'])} per exposure-year, "
        f"so the loaded premium cannot reflect risk differences in the excess.",
        f"- {f('share', a['linkage']['claims_without_sev_share_of_claims'])} of the recorded claims have no "
        f"severity row; the frequency target counts severity rows ({f('count', t['targets']['frequency_count_total'])}) "
        f"rather than recorded ClaimNb ({f('count', t['targets']['claim_nb_recorded_total'])}).",
        f"- {f('count', a['reconciliation']['orphan_rows'])} orphan severity rows "
        f"({f('money', a['reconciliation']['orphan_cost'])}) are excluded.",
        f"- The severity model explains {f('share', sev_expl)} of the CV deviance against the constant "
        "(a negative value means no gain over the constant).",
        f"- Evaluation is on one pre-registered grouped test fold ({f('count', m['rows']['claims'])} claims); "
        f"the journal lists {len(i['attempt']['attempts'])} attempt(s).",
        "",
    ]
    return out


def sec_links(i: dict[str, Any], f: _Fmt) -> list[str]:
    dc = i["dc"]
    return [
        "## 12. Detailed reports and figures",
        "",
        *[f"- [{t}]({p})" for p, t in dc["links"]],
        *[f"- [{t}]({p})" for p, t in dc["figures"]],
        "",
    ]


SECTIONS = (sec_data, sec_split, sec_targets, sec_glm, sec_selection, sec_gbm, sec_constraints,
            sec_sensitivities, sec_interpretation, sec_test, sec_limits, sec_links)  # fmt: skip


def render_model_report(i: dict[str, Any]) -> str:
    f = _Fmt(i["dc"])
    m = i["attempt"]["metrics"]
    head = [
        "# Model report",
        "",
        "Generated by `fremtpl2 docs` from the tracked pipeline files (`results/*.json`, the GLM "
        f"relativity tables and the completed test attempt {m['attempt']}). No number in this report is "
        "typed by hand; edit the code or `configs/docs.yaml`, not this file.",
        "",
        f"Selected candidates: GLM `{m['selected_glm_id']}`, GBM `{m['selected_gbm_id']}`.",
        "",
    ]
    body = [line for sec in SECTIONS for line in sec(i, f)]
    return "\n".join([*head, *body]).rstrip() + "\n"


# --------------------------------------------------------------------------- markers


def replace_block(text: str, name: str, body: str, markers: dict[str, str], where: str) -> str:
    """Replace what lies between the begin and end marker of ``name``; everything else is kept."""
    begin, end = markers["begin"].format(name=name), markers["end"].format(name=name)
    if text.count(begin) != 1 or text.count(end) != 1:
        raise DocsError(f"{where}: needs exactly one `{begin}` and one `{end}` marker")
    a, b = text.index(begin) + len(begin), text.index(end)
    if b < a:
        raise DocsError(f"{where}: the end marker of block `{name}` precedes its begin marker")
    return text[:a] + "\n" + body.rstrip("\n") + "\n" + text[b:]


def render_all(cfg: dict[str, Any], root: Path) -> dict[str, str]:
    """Project-relative path -> new text of every target, all rendered in memory."""
    inputs = load_inputs(cfg, root)
    dc, f = cfg["docs"], _Fmt(cfg["docs"])
    out = {dc["outputs"]["model_report"]: render_model_report(inputs)}
    for rel, names in dc["blocks"].items():
        path = root / rel
        if not path.is_file():
            raise DocsError(f"{rel} is missing")
        text = path.read_bytes().decode("utf-8")
        for name in names:
            if name not in BLOCKS:
                raise DocsError(f"unknown block `{name}` for {rel}; known: {sorted(BLOCKS)}")
            text = replace_block(text, name, "\n".join(BLOCKS[name](inputs, f)), dc["markers"], rel)
        out[rel] = text
    return out


def run_docs(cfg: dict[str, Any], root: Path | None = None, check: bool = False) -> dict[str, Any]:
    """Render every target; with ``check`` write nothing and list the targets that would change."""
    root = Path.cwd() if root is None else Path(root)
    new = render_all(cfg, root)
    drift = [
        rel for rel, text in new.items()
        if not (root / rel).is_file() or (root / rel).read_bytes() != text.encode("utf-8")
    ]  # fmt: skip
    if not check:
        for rel in drift:
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_bytes(new[rel].encode("utf-8"))
            tmp.replace(target)
    return {"targets": sorted(new), "drift": drift, "written": [] if check else drift}
