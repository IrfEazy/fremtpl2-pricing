"""Render ``reports/glm_cv.md`` from the parsed ``results/cv_glm.json`` (no hand-typed numbers)."""

from __future__ import annotations

from typing import Any


def _f(v: Any, digits: int = 5) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.1f}" if abs(v) >= 1000 else f"{v:.{digits}g}"
    return str(v)


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_f(c) for c in r) + " |" for r in rows]
    return out + [""]


def _pct(v: float) -> str:
    return f"{100 * v:.2f}%"


def _severity_finding(r: dict[str, Any]) -> str:
    """Sentence on the full severity GLM out of fold, chosen by the sign of the numbers."""
    node = r["cv"]["fold_summary"]["severity"]["full"]["deviance_explained"]
    pooled = r["cv"]["pooled"]["severity"]["full"]["deviance_explained"]
    mean, sd = node["mean"], node["sd"]
    n_neg = sum(f["severity"]["full"]["deviance_explained"] < 0 for f in r["cv"]["folds"])
    head = (
        f"Severity full GLM, out-of-fold deviance explained versus the null: fold mean "
        f"{_pct(mean)} (sd {_pct(sd)}), pooled {_pct(pooled)}; negative in {n_neg} of "
        f"{len(r['cv']['folds'])} folds."
    )
    if pooled < 0:
        tail = (
            " The full severity model predicts claims out of fold worse than the constant "
            "mean: the rating factors carry no usable severity signal at this sample size."
        )
    elif mean - sd < 0:
        tail = (
            " The gain over the constant mean is small relative to its fold-to-fold sd: "
            "weak evidence of severity signal."
        )
    else:
        tail = " The full severity model improves on the constant mean out of fold."
    return head + tail


def _power_text(label: str, d: dict[str, Any]) -> str:
    edge = "at the grid edge" if d["at_grid_edge"] else "interior to the grid"
    return (
        f"{label}: selected {_f(d['power'])}, {edge}; held-out log-likelihood spread over the "
        f"grid {_f(d['loglik_spread_nats'])} nats, best minus runner-up "
        f"{_f(d['best_minus_next_nats'])} nats."
    )


def _neighbour_text(d: dict[str, Any]) -> list[str]:
    """Sentences on the paired fold differences and the capped-policy contribution."""
    out = []
    for nb in d["neighbours"]:
        tot, cap = nb["total_diff_nats"], nb["capped_policies_diff_nats"]
        side = "favours" if tot > 0 else "opposes"
        cap_side = "favour" if cap > 0 else "oppose"
        agree = (
            f"{nb['folds_agreeing_in_sign']} of {nb['n_folds']} folds agree in sign with the total"
        )
        out.append(
            f"Selected {_f(d['power'])} versus neighbour {_f(nb['neighbour'])}: total "
            f"difference {_f(tot)} nats ({side} the selected power; {agree}). Restricted to the "
            f"policies with a capped claim the difference is {_f(cap)} nats (they {cap_side} "
            f"the selected power), for all other policies {_f(nb['other_policies_diff_nats'])} "
            "nats."
        )
    return out


def t_sev(cfg: dict[str, Any]) -> int:
    return cfg["models"]["severity_features"]["min_claims"]


def _ms(node: dict) -> str:
    return f"{_f(node['mean'])} ± {_f(node['sd'])}"


def _scalar_rows(summary: dict, pooled: dict, group: str, metric_keys: list[str]):
    rows = []
    for model, vals in summary[group].items():
        for key in metric_keys:
            rows.append([model, key, _ms(vals[key]), pooled[group][model][key]])
    return rows


def render_report(r: dict[str, Any]) -> str:
    cfg, cv, rows = r["config"], r["cv"], r["rows"]
    t = cfg["models"]["tweedie"]
    lines = [
        "# GLM cross-validation (milestone 2)",
        "",
        "Rendered from `cv_glm.json` by `fremtpl2 train`; every number below comes from that file.",
        "",
        "## Protocol",
        "",
        f"- Test set used: **{cfg['test_set_used']}**. All fits, preprocessing and metrics use the "
        "primary training set only (registered `cv_fold`); the test set is untouched in this "
        "milestone.",
        f"- Split: `{cfg['split_primary']}`, {cfg['cv_folds']} outer folds; features, rare-level "
        "grouping and every model are fitted on the other folds of each outer fold.",
        f"- Tweedie power: grid {t['power_grid']}, chosen per outer fold by an inner CV over the "
        f"remaining folds with the `{t['selection_loss']}` loss: the held-out Tweedie "
        "log-likelihood of S_cap / Exposure with prior weight = exposure, dispersion estimated by "
        "maximum likelihood on the fitting part only (Dunn & Smyth 2005 series density). The "
        "exposure-weighted Poisson deviance per power is kept as a labelled diagnostic. Common "
        f"Tweedie deviance at the fixed reference power {t['reference_power']}, which also "
        "enters the pure premium comparison as the fixed-power sensitivity `tweedie_ref`.",
        "- Power selection by log-likelihood targets distributional fit. The exposure-weighted "
        "Poisson deviance of S_cap / Exposure per power is kept as a labelled diagnostic: it is a "
        "valid conditional-mean scoring rule, so a different optimum under it is not an error. "
        f"Fixed-power Tweedie sensitivities {t['fixed_power_sensitivities']} (models "
        "`tweedie_fixed_<p>`) enter the pure premium comparison; the common deviance stays at the "
        "reference power.",
        "- Severity model: the allowed levels per feature are those of the policy schema of the "
        "same fitting part; the severity map is derived from the claims-only fitting rows "
        f"(minimum {_f(t_sev(cfg))} claims per level): nominal levels below it merge into OTHER "
        "(then the reference), ordinal features "
        f"({', '.join(cfg['models']['severity_features']['ordinal'])}) into the adjacent level "
        "with more claims (ties toward the reference). The mapping is stored in "
        "`feature_map.json`.",
        "- Inference: coefficient standard errors in the relativity tables are policy-level "
        "sandwich standard errors, conditional on the chosen specification (they do not account "
        "for the power selection or the level pooling). Relativities are adjusted multiplicative "
        "effects versus the stated reference level, not causal effects.",
        f"- Targets: frequency count `{cfg['clean_targets']['frequency_count']}`, per-claim cap "
        f"{_f(cfg['large_loss_cap'])}; the excess loading (`{cfg['excess_loading_mode']}`) is in "
        "no CV target or prediction.",
        f"- Bootstrap confidence intervals are not part of this milestone (applied: "
        f"{cfg['bootstrap_applied']}); they come in milestone 3.",
        f"- Config hash `{cfg['config_sha256']}`, seed {cfg['seed']}.",
        "",
        "## Data",
        "",
    ]
    lines += _table(
        ["train policies", "exposure-years", "claims (N)", "policies with claims"],
        [
            [
                rows["train_policies"],
                rows["train_exposure"],
                rows["train_claims"],
                rows["train_claim_policies"],
            ]
        ],
    )
    lines += _table(
        ["fold", "policies", "exposure", "claims"],
        [[x["cv_fold"], x["policies"], x["exposure"], x["claims"]] for x in rows["per_fold"]],
    )
    lines += ["## Selected Tweedie power", ""]
    grid = [str(float(p)) for p in t["power_grid"]]
    lines += _table(
        ["fold", "selected", "at grid edge", "loglik spread (nats)"],
        [
            [s_["fold"], s_["power"], s_["at_grid_edge"], s_["loglik_spread_nats"]]
            for s_ in cv["selected_powers"]
        ],
    )
    lines += [
        "Inner-CV held-out Tweedie log-likelihood per power (total nats; higher is better).",
        "",
    ]
    lines += _table(
        ["fold"] + [f"p={p}" for p in t["power_grid"]],
        [[s_["fold"]] + [s_["inner_loglik"][g] for g in grid] for s_ in cv["selected_powers"]],
    )
    lines += ["Mean maximum-likelihood dispersion over the inner splits.", ""]
    lines += _table(
        ["fold"] + [f"p={p}" for p in t["power_grid"]],
        [[s_["fold"]] + [s_["inner_dispersion"][g] for g in grid] for s_ in cv["selected_powers"]],
    )
    lines += [
        "Diagnostic only (not the selection criterion): inner-CV exposure-weighted Poisson "
        "deviance of S_cap / Exposure per exposure-year. It is a mean-prediction loss and does "
        "not identify the variance power.",
        "",
    ]
    lines += _table(
        ["fold"] + [f"p={p}" for p in t["power_grid"]],
        [
            [s_["fold"]] + [s_["diagnostic_inner_poisson_deviance_per_exposure"][g] for g in grid]
            for s_ in cv["selected_powers"]
        ],
    )
    fp = r["final"]["power_cv"]
    lines += [
        _power_text("Final power (plain CV over the grid on the registered folds)", fp),
        "",
    ]
    lines += _table(
        ["power", "CV loglik (nats)", "CV loglik per policy", "diagnostic: Poisson deviance"],
        [
            [
                p,
                fp["cv_loglik"][str(float(p))],
                fp["cv_loglik_per_policy"][str(float(p))],
                fp["diagnostic_cv_poisson_deviance_per_exposure"][str(float(p))],
            ]
            for p in t["power_grid"]
        ],
    )
    st = fp["stability"]
    n_edge = sum(s_["at_grid_edge"] for s_ in cv["selected_powers"])
    lines += [
        f"Selection stability: the outer folds select {st['selected_by_outer_fold']}; "
        f"{st['folds_selecting_final']} of {st['n_folds']} select the final power "
        f"({n_edge} selections at a grid edge).",
        "",
        "Final-CV held-out log-likelihood per outer fold and power (nats).",
        "",
    ]
    lines += _table(
        ["fold"] + [f"p={p}" for p in t["power_grid"]],
        [[k] + [v[g] for g in grid] for k, v in fp["fold_loglik"].items()],
    )
    lines += ["Paired per-fold differences, selected power minus neighbour (nats).", ""]
    lines += _table(
        ["neighbour", "total"] + [f"fold {k}" for k in fp["fold_loglik"]] + ["folds agreeing"],
        [
            [nb["neighbour"], nb["total_diff_nats"]]
            + nb["per_fold_diff_nats"]
            + [f"{nb['folds_agreeing_in_sign']} of {nb['n_folds']}"]
            for nb in fp["neighbours"]
        ],
    )
    lines += [f"- {x}" for x in _neighbour_text(fp)] + [""]
    lines += ["### Cap diagnostics (from the claim rows)", ""]
    cd = r["cap_diagnostics"]
    lines += _table(
        [
            "cap",
            "claims",
            "capped claims",
            "policies with a capped claim",
            "their exposure",
            "amount removed",
            "removed / uncapped total",
            "capped claims in S_cap",
            "share of S_cap",
        ],
        [
            [
                cd["cap"],
                cd["claims"],
                cd["capped_claims"],
                cd["policies_with_capped_claim"],
                cd["capped_policy_exposure"],
                cd["amount_removed"],
                _pct(cd["amount_removed_share_of_uncapped"]),
                cd["capped_claims_amount_in_s_cap"],
                _pct(cd["share_of_s_cap"]),
            ]
        ],
    )
    rec = cd["reconciliation"]
    lines += [
        f"Reconciliation: claim-row S_cap {_f(rec['claim_rows_s_cap'])} against policy-level "
        f"S_cap {_f(rec['policy_s_cap'])}; policies with a positive excess "
        f"{_f(rec['policies_with_excess'])} against {_f(cd['policies_with_capped_claim'])} with "
        "a capped claim.",
        "",
        "The per-claim cap makes S_cap a censored amount: a capped claim sits exactly at the cap, "
        "an atom that the continuous Tweedie density does not model, so the likelihood is a "
        "working likelihood. "
        f"{_f(cd['capped_claims'])} capped claims on {_f(cd['policies_with_capped_claim'])} "
        f"policies ({_pct(cd['capped_policy_exposure_share'])} of the exposure) carry "
        f"{_pct(cd['share_of_s_cap'])} of S_cap. Their effect on the power choice is the "
        "capped-policy difference quoted above.",
        "",
    ]
    ds = r["dispersion"]
    lines += [
        "## Dispersion search",
        "",
        f"{ds['n_dispersion_fits']} maximum-likelihood dispersion fits; {ds['n_expanded']} needed "
        f"the bracket expanded (total expansions {ds['total_expansions']}, maximum per fit "
        f"{ds['max_expansions']}); allowed: {t['dispersion_max_expansions']} (a fit still on the "
        "bound raises).",
        "",
    ]
    if ds["expanded"]:
        lines += _table(
            ["stage", "fold", "inner fold", "power", "expansions"],
            [
                [x["stage"], x["fold"], x["inner_fold"], x["power"], x["dispersion_expansions"]]
                for x in ds["expanded"]
            ],
        )
    cv_ = r["convergence"]
    lines += ["## Solver convergence", ""]
    lines += [
        f"glum solver settings: max_iter {cv_['solver']['max_iter']}, gradient_tol "
        f"{_f(cv_['solver']['gradient_tol'])}"
        + "".join(
            f" (family {fam}: gradient_tol {_f(o['gradient_tol'])})"
            for fam, o in (cv_["solver"].get("by_family") or {}).items()
            if "gradient_tol" in o
        )
        + f". Fits: {cv_['n_fits']}; **not converged: "
        f"{cv_['n_not_converged']}**; maximum iterations used {cv_['max_n_iter']}.",
        "",
    ]
    if cv_["not_converged"]:
        lines += _table(
            ["stage", "fold", "inner fold", "model", "power", "n_iter"],
            [
                [x["stage"], x["fold"], x["inner_fold"], x["model"], x["power"], x["n_iter"]]
                for x in cv_["not_converged"]
            ],
        )
    sc = r["solver_check"]
    lines += [
        "### Solver accuracy check (real data)",
        "",
        f"The final full-train frequency, severity and Tweedie models and, per outer fold, the CV "
        f"Tweedie (selected power), frequency and severity fits are refitted with gradient_tol "
        f"{_f(sc['tight_solver']['gradient_tol'])} (max_iter {sc['tight_solver']['max_iter']}). "
        f"Maximum relative prediction difference {_f(sc['maxima']['rel_prediction_diff'])} "
        f"(limit {_f(sc['thresholds']['rel_prediction_diff'])}), relative deviance difference "
        f"{_f(sc['maxima']['rel_deviance_diff'])} (limit "
        f"{_f(sc['thresholds']['rel_deviance_diff'])}), coefficient difference "
        f"{_f(sc['maxima']['coef_diff'])} (limit {_f(sc['thresholds']['coef_diff'])}); "
        f"violations: {len(sc['failures'])}; passed: **{sc['passed']}**.",
        "",
    ]
    lines += _table(
        ["stage", "fold", "model", "power", "rel. prediction", "rel. deviance", "coef", "iter"],
        [
            [
                x["stage"],
                x["fold"],
                x["model"],
                x["power"],
                x["rel_prediction_diff"],
                x["rel_deviance_diff"],
                x["coef_diff"],
                f"{x['n_iter']} / {x['n_iter_tight']}",
            ]  # fmt: skip
            for x in sc["records"]
        ],
    )
    pp, ll = sc["pure_premium"], sc["loglik_checks"]
    lines += [
        "",
        "Combined out-of-fold pure premium (frequency x severity) per policy, pipeline solver "
        f"against tight solver: maximum relative difference {_f(pp['max_rel_diff'])} (limit "
        f"{_f(sc['thresholds']['rel_pure_premium_diff'])}) at prediction quantile "
        f"{_f(pp['max_rel_diff_at_prediction_quantile'])}; maximum absolute difference "
        f"{_f(pp['max_abs_diff'])} at prediction quantile "
        f"{_f(pp['max_abs_diff_at_prediction_quantile'])}.",
        "",
    ]
    lines += _table(
        [
            "fold",
            "max rel.",
            "at quantile",
            "prediction there",
            "max abs.",
            "at quantile",
            "prediction there",
            "total rel.",
        ],
        [
            [
                x["fold"],
                x["rel_diff"],
                x["rel_diff_at_prediction_quantile"],
                x["prediction_at_max_rel"],
                x["abs_diff"],
                x["abs_diff_at_prediction_quantile"],
                x["prediction_at_max_abs"],
                x["total_rel_diff"],
            ]  # fmt: skip
            for x in pp["records"]
        ],
    )
    lines += [
        "Held-out log-likelihood difference (selected power minus grid neighbour) with the "
        "pipeline solver and with the tight solver; it must keep its sign and change by less than "
        f"{_pct(sc['thresholds']['loglik_difference_rel_change'])} (maximum relative change "
        f"{_f(ll['max_rel_change'])}; signs kept: {_f(ll['all_signs_kept'])}).",
        "",
    ]
    lines += _table(
        ["fold", "neighbour", "diff (config)", "diff (tight)", "abs. change", "rel. change", "ok"],
        [
            [
                x["fold"],
                x["neighbour"],
                x["diff_config"],
                x["diff_tight"],
                x["abs_change"],
                x["rel_change"],
                x["passed"],
            ]
            for x in ll["records"]
        ],
    )
    ck = r["checks"]
    lines += [
        "Promotion gates of the model artifacts: "
        + ", ".join(f"{k} {_f(v)}" for k, v in ck.items() if k != "passed")
        + f"; all passed: **{ck['passed']}**.",
        "",
    ]
    gs = r["groups"]
    lines += [
        "## Rating-profile groups (train only; the bootstrap unit)",
        "",
        "Groups are the tuples of all rating factors; sizes are policies per group. The largest "
        f"groups are the top {_pct(gs['overall']['top_share'])} by policy count.",
        "",
    ]
    qs = list(gs["overall"]["size_quantiles"])
    lines += _table(
        ["scope", "groups", "policies"]
        + [f"size q{q}" for q in qs]
        + ["max size", "top groups", "top policy share", "top exposure share"],
        [
            [x_name, x["groups"], x["policies"]]
            + [x["size_quantiles"][q] for q in qs]
            + [
                x["max_group_size"],
                x["top_groups"],
                _pct(x["top_policy_share"]),
                _pct(x["top_exposure_share"]),
            ]
            for x_name, x in [("overall", gs["overall"])]
            + [(f"fold {x['fold']}", x) for x in gs["per_fold"]]
        ],
    )
    lines += [
        "## Findings",
        "",
        "- " + _severity_finding(r),
        "- " + _power_text("Tweedie power", fp),
    ] + [f"- {x}" for x in _neighbour_text(fp)]
    lines += [""]  # fmt: skip
    summ, pool = cv["fold_summary"], cv["pooled"]
    lines += ["## Scalar metrics (fold mean ± sd, and pooled out-of-fold)", "", "### Frequency", ""]
    lines += _table(
        ["model", "metric", "fold mean ± sd", "pooled"],
        _scalar_rows(
            summ,
            pool,
            "frequency",
            ["deviance_per_exposure", "deviance_explained", "gini", "gini_normalized"],
        ),
    )
    lines += ["### Severity (claims only, weight N)", ""]
    lines += _table(
        ["model", "metric", "fold mean ± sd", "pooled"],
        _scalar_rows(summ, pool, "severity", ["deviance_per_claim", "deviance_explained"]),
    )
    lines += ["### Pure premium", ""]
    lines += _table(
        ["model", "metric", "fold mean ± sd", "pooled"],
        _scalar_rows(
            summ,
            pool,
            "pure_premium",
            [
                "poisson_deviance_per_exposure",
                "poisson_deviance_explained",
                "tweedie_ref_deviance_per_exposure",
                "tweedie_ref_deviance_explained",
                "gini",
                "gini_normalized",
            ],
        ),
    )
    lines += _table(
        ["model", "observed total (S_cap)", "predicted total", "observed / predicted"],
        [
            [m, v["totals"]["observed"], v["totals"]["predicted"], v["totals"]["ratio"]]
            for m, v in pool["pure_premium"].items()
        ],
    )
    lines += ["## Lift tables (pooled out-of-fold)", ""]
    cols = ["bin", "exposure", "observed", "predicted", "ratio"]
    for m, tab in cv["lift"]["pure_premium"].items():
        lines += [f"### Pure premium: {m}", ""]
        lines += _table(cols, [[x[c] for c in cols] for x in tab])
    lines += ["## Double lift: Tweedie vs frequency x severity", ""]
    dc = ["bin", "exposure", "observed", "pred_a", "pred_b", "ratio_a", "ratio_b"]
    lines += ["`pred_a` is the Tweedie, `pred_b` the frequency x severity prediction.", ""]
    lines += _table(dc, [[x[c] for c in dc] for x in cv["double_lift"]])
    lines += ["## Calibration", ""]
    for m, tab in cv["lift"]["frequency"].items():
        lines += [f"### Frequency by decile: {m}", ""]
        lines += _table(cols, [[x[c] for c in cols] for x in tab])
    cc = ["level", "weight", "observed", "predicted", "ratio"]
    for factor, entry in cv["calibration"].items():
        for target in ("frequency", "pure_premium"):
            model = "full" if target == "frequency" else "tweedie"
            lines += [f"### {target} by {factor}: {model}", ""]
            lines += _table(cc, [[x[c] for c in cc] for x in entry[target][model]])
    lp = r["loaded_premium"]
    lines += [
        "## Loaded premium totals (separate; reporting only)",
        "",
        lp["label"],
        "",
    ]
    lines += _table(
        [
            "mode",
            "observed uncapped cost",
            "predicted (capped Tweedie)",
            "loading total",
            "predicted loaded",
            "observed / loaded",
        ],
        [
            [
                lp["mode"],
                lp["observed_uncapped_cost_total"],
                lp["predicted_tweedie_capped_total"],
                lp["loading_total"],
                lp["predicted_loaded_total"],
                lp["ratio_observed_to_loaded"],
            ]
        ],
    )
    lines += [
        f"Loading per exposure-year on the full training set: {_f(r['excess_loading_train'])}.",
        "",
        "## Final models",
        "",
        f"Refitted on the full training set (Tweedie power {_f(r['final']['power_cv']['power'])}); "
        "files: " + ", ".join(f"`{x}`" for x in r["final"].get("files", [])) + ".",
        "",
        "## Timing (seconds)",
        "",
    ]
    lines += _table(["stage", "seconds"], [[k, v] for k, v in r["timing_seconds"].items()])
    return "\n".join(lines)


def _ms(node: dict[str, Any], pct: bool = False) -> str:
    g = _pct if pct else _f
    return f"{g(node['mean'])} ± {g(node['sd'])}"


def render_sensitivities_report(r: dict[str, Any]) -> str:
    """Render ``reports/glm_sensitivities.md`` from the parsed ``results/cv_sensitivities.json``."""
    cfg, sens = r["config"], r["sensitivities"]
    lines = [
        "# GLM cleaning/target sensitivities (grouped CV, train set only)",
        "",
        "Rendered from `results/cv_sensitivities.json` by `fremtpl2 sensitivities`; no hand-typed "
        "numbers. The test set is untouched: sensitivities are CV-only by protocol "
        "(`reports/evaluation_protocol.md` section 7); the test evaluation uses the primary "
        "definitions only.",
        "",
        f"Outer registered folds only ({cfg['cv_folds']}, {cfg['split_primary']} split, frozen); "
        f"Tweedie power fixed at {_f(cfg['fixed_tweedie_power'])} (final power of "
        f"`{cfg['power_source']}`), no nested selection, no final fit. Policies a sensitivity "
        "drops simply drop out; the split assignment of the others never changes.",
        "",
        "**Comparisons are made within each target definition only. Losses of different "
        "definitions (different targets, caps, exposures or policy sets) are not comparable and "
        "are not ranked.**",
        "",
        "## Row counts (train)",
        "",
    ]
    lines += _table(
        [
            "sensitivity",
            "overrides",
            "policies",
            "exposure",
            "claims (N)",
            "frequency events",
            "S_cap",
        ],
        [
            [
                n,
                json_inline(s["overrides"]),
                s["rows"]["policies"],
                s["rows"]["exposure"],
                s["rows"]["claims"],
                s["rows"]["frequency_events"],
                s["rows"]["s_cap"],
            ]
            for n, s in sens.items()
        ],
    )
    lines += ["## Convergence", ""]
    lines += _table(
        ["sensitivity", "fits", "not converged", "max iterations"],
        [
            [
                n,
                s["convergence"]["n_fits"],
                s["convergence"]["n_not_converged"],
                s["convergence"]["max_n_iter"],
            ]
            for n, s in sens.items()
        ],
    )
    lines += ["## Frequency (full GLM vs null, within definition)", ""]
    lines += _table(
        [
            "sensitivity",
            "deviance / exposure-year (null → full, pooled)",
            "explained pooled",
            "explained fold mean ± sd",
            "Gini normalised pooled",
        ],
        [
            [
                n,
                f"{_f(s['pooled']['frequency']['null']['deviance_per_exposure'])} → "
                f"{_f(s['pooled']['frequency']['full']['deviance_per_exposure'])}",
                _pct(s["pooled"]["frequency"]["full"]["deviance_explained"]),
                _ms(s["fold_summary"]["frequency"]["full"]["deviance_explained"], True),
                s["pooled"]["frequency"]["full"]["gini_normalized"],
            ]
            for n, s in sens.items()
        ],
    )
    lines += ["## Severity (full GLM vs null, claims only, within definition)", ""]
    lines += _table(
        [
            "sensitivity",
            "Gamma deviance / claim (null → full, pooled)",
            "explained pooled",
            "explained fold mean ± sd",
        ],
        [
            [
                n,
                f"{_f(s['pooled']['severity']['null']['deviance_per_claim'])} → "
                f"{_f(s['pooled']['severity']['full']['deviance_per_claim'])}",
                _pct(s["pooled"]["severity"]["full"]["deviance_explained"]),
                _ms(s["fold_summary"]["severity"]["full"]["deviance_explained"], True),
            ]
            for n, s in sens.items()
        ],
    )
    p = _f(cfg["reference_power"])
    lines += [
        f"## Pure premium (common Tweedie deviance at reference power {p}, within definition)",
        "",
        f"The `tweedie` model is fitted at the fixed power {_f(cfg['fixed_tweedie_power'])}; "
        "the deviance is always evaluated at the reference power.",
        "",
    ]
    for n, s in sens.items():
        pool, fs = s["pooled"]["pure_premium"], s["fold_summary"]["pure_premium"]
        lines += [f"### {n}", ""]
        if not s["freq_x_sev_valid"]:
            lines += [f"Frequency x severity not computed: {s['freq_x_sev_note']}.", ""]
        rows = []
        for m, v in pool.items():
            rows.append(
                [
                    m,
                    v["tweedie_ref_deviance_per_exposure"],
                    _pct(v["tweedie_ref_deviance_explained"]),
                    _ms(fs[m]["tweedie_ref_deviance_explained"], True),
                    _pct(v["poisson_deviance_explained"]),
                    v["gini_normalized"],
                    v["totals"]["observed"],
                    v["totals"]["predicted"],
                    v["totals"]["ratio"],
                ]
            )
        lines += _table(
            [
                "model",
                "Tweedie dev. / exposure-year",
                "explained pooled",
                "explained fold mean ± sd",
                "Poisson explained pooled",
                "Gini normalised",
                "observed S_cap",
                "predicted",
                "obs / pred",
            ],
            rows,
        )
    lines += [
        "## Differences inside each definition",
        "",
        "Tweedie minus frequency x severity (negative deviance difference favours the Tweedie), "
        "pooled and fold mean ± sd of the per-fold differences.",
        "",
    ]
    rows = []
    for n, s in sens.items():
        c, cf = s["comparisons"]["pooled"], s["comparisons"]["fold_summary"]
        if s["freq_x_sev_valid"]:
            t, tf = c["tweedie_minus_freq_x_sev"], cf["tweedie_minus_freq_x_sev"]
            t_cells = [
                t["tweedie_ref_deviance_per_exposure"],
                _ms(tf["tweedie_ref_deviance_per_exposure"]),
                t["gini_normalized"],
            ]
        else:
            t_cells = ["not computed"] * 3
        f_, ff = c["frequency_full_minus_null"], cf["frequency_full_minus_null"]
        v_, vf = c["severity_full_minus_null"], cf["severity_full_minus_null"]
        rows.append(
            [
                n,
                *t_cells,
                f_["deviance_per_exposure"],
                _ms(ff["deviance_per_exposure"]),
                v_["deviance_per_claim"],
                _ms(vf["deviance_per_claim"]),
            ]
        )
    lines += _table(
        [
            "sensitivity",
            "Tweedie - freq x sev: deviance pooled",
            "fold mean ± sd",
            "Gini norm. pooled",
            "frequency full - null: deviance pooled",
            "fold mean ± sd",
            "severity full - null: deviance pooled",
            "fold mean ± sd",
        ],
        rows,
    )
    invalid = [n for n, s in sens.items() if not s["freq_x_sev_valid"]]
    for n in invalid:
        lines += ["", f"`{n}`: {sens[n]['freq_x_sev_note']}."]
    lines += [""]
    b = r["bootstrap_unit_diagnostic"]
    lines += [
        "## Bootstrap unit diagnostic (train only, primary definition)",
        "",
        b["label"] + ".",
        "",
        f"Paired bootstrap of the primary out-of-fold predictions; loss: {b['loss']}; "
        f"{b['replicates']} replicates, seed {b['seed']}, {_pct(b['level'])} percentile intervals. "
        "The rating-profile groups are resampled because identical profiles may belong to the "
        "same policyholder (renewals, duplicates), not merely because they share covariates.",
        "",
    ]
    rows = []
    for k in b["ratio_group_over_policy"]:
        for u in ("group", "policy"):
            m = b["units"][u]["metrics"][k]
            rows.append(
                [
                    k,
                    u,
                    b["units"][u]["n_units"],
                    m["point"],
                    m["se"],
                    m["lower"],
                    m["upper"],
                    m["width"],
                ]
            )
    lines += _table(["quantity", "unit", "units", "point", "SE", "lower", "upper", "width"], rows)
    lines += _table(
        ["quantity", "SE ratio group / policy", "interval width ratio group / policy"],
        [[k, v["se"], v["width"]] for k, v in b["ratio_group_over_policy"].items()],
    )
    lines += ["## Timing (seconds)", ""]
    lines += _table(["stage", "seconds"], [[k, v] for k, v in r["timing_seconds"].items()])
    return "\n".join(lines)


def json_inline(d: Any) -> str:
    """Compact one-line rendering of a (nested) override mapping."""
    if not d:
        return "none (primary rules)"
    if isinstance(d, dict):
        return ", ".join(
            f"{k}: {json_inline(v)}" if isinstance(v, dict) else f"{k}: {v}" for k, v in d.items()
        )
    return str(d)
