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


def _power_text(label: str, d: dict[str, Any], n: int | None = None) -> str:
    edge = "at the grid edge" if d["at_grid_edge"] else "interior to the grid"
    ident = (
        "the power is only weakly identified (spread below the configured threshold)"
        if d["weakly_identified"]
        else "the power is identified by the likelihood (spread above the threshold)"
    )
    return (
        f"{label}: selected {_f(d['power'])}, {edge}; held-out log-likelihood spread over the "
        f"grid {_f(d['loglik_spread_nats'])} nats, best minus runner-up "
        f"{_f(d['best_minus_next_nats'])} nats; {ident}."
    )


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
        ["fold", "selected", "at grid edge", "loglik spread (nats)", "weakly identified"],
        [
            [
                s_["fold"],
                s_["power"],
                s_["at_grid_edge"],
                s_["loglik_spread_nats"],
                s_["weakly_identified"],
            ]  # fmt: skip
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
    n_weak = sum(s_["weakly_identified"] for s_ in cv["selected_powers"])
    n_edge = sum(s_["at_grid_edge"] for s_ in cv["selected_powers"])
    lines += [
        f"Per outer fold: {n_edge} of {len(cv['selected_powers'])} selections at a grid edge, "
        f"{n_weak} weakly identified (threshold {_f(t['weak_identification_nats'])} nats).",
        "",
        "### Caveat: point mass at the cap",
        "",
    ]
    cp = r["cap_point_mass"]
    lines += [
        "The per-claim cap makes S_cap a censored amount: policies whose S_cap sits exactly at a "
        "positive multiple of the cap carry a point mass that the continuous Tweedie density "
        "does not model. The likelihood is therefore a working likelihood, adequate for "
        "comparing powers but not an exact model of the capped response.",
        "",
    ]
    lines += _table(
        [
            "cap",
            "policies with claim cost",
            "S_cap at a cap multiple",
            "share",
            "with a capped claim",
        ],
        [
            [
                cp["cap"],
                cp["policies_with_claim_cost"],
                cp["policies_at_cap_multiple"],
                _pct(cp["share_of_claim_policies"]),
                cp["policies_with_a_capped_claim"],
            ]
        ],
    )
    cv_ = r["convergence"]
    lines += ["## Solver convergence", ""]
    lines += [
        f"glum solver settings: max_iter {cv_['solver']['max_iter']}, gradient_tol "
        f"{_f(cv_['solver']['gradient_tol'])}. Fits: {cv_['n_fits']}; **not converged: "
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
    lines += ["## Findings", "", "- " + _severity_finding(r),
        "- " + _power_text("Tweedie power", fp), ""]  # fmt: skip
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
