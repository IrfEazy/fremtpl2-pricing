"""Cleaning/target sensitivities in grouped CV on the TRAIN set only (protocol section 7).

Every declared ``clean.sensitivities`` entry (plus ``primary`` as the baseline) is cleaned on the
full data, joined to the frozen split (train rows only; policies a sensitivity drops simply drop
out, the split assignment of the others never changes) and run through the OUTER cross-validation
only, with the Tweedie power fixed at the final power of ``results/cv_glm.json`` (no nested
selection, no final fit, no coefficient tables). The fits are those of ``train._outer``. The test
set is never loaded into a fit or a metric. A non-converged fit fails the run. A diagnostic
compares the group and the policy bootstrap unit on the primary out-of-fold predictions.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import metrics
from .clean import SENSITIVITIES_KEY, clean_sensitivity, sensitivity_rules
from .cv_report import render_sensitivities_report
from .data import load_raw
from .train import (
    _clean_json,
    _outer_task,
    _records,
    assert_train_only,
    convergence_summary,
    fixed_name,
    outer_splits,
    scalar_metrics,
    summarise_folds,
)

PRIMARY = "primary"
PP_MODELS = ("null", "bonus_malus_only", "freq_x_sev", "tweedie")
BOOT_KEYS = ("freq_x_sev", "tweedie", "tweedie_minus_freq_x_sev")


def fixed_power(cfg: dict[str, Any]) -> float:
    """Final Tweedie power of the committed train run (read from its CV file; error if absent)."""
    path = Path(cfg["paths"]["results"]) / cfg["models"]["outputs"]["cv_json"]
    if not path.is_file():
        raise ValueError(f"{path} not found: run `fremtpl2 train` first (fixed Tweedie power)")
    try:
        power = json.loads(path.read_text(encoding="utf-8"))["final"]["power_cv"]["power"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} has no final.power_cv.power") from exc
    if not isinstance(power, int | float) or isinstance(power, bool):
        raise ValueError(f"{path}: final.power_cv.power is not a number")
    return float(power)


def sensitivity_names(cfg: dict[str, Any]) -> list[str]:
    """Names to run: ``models.sensitivities.names`` (``all`` or a list of declared names)."""
    declared = list(cfg["clean"].get(SENSITIVITIES_KEY, {}))
    sel = cfg["models"]["sensitivities"]["names"]
    names = declared if sel == "all" else list(sel)
    unknown = sorted(set(names) - set(declared))
    if unknown:
        raise ValueError(f"unknown sensitivities {unknown}; declared: {declared}")
    return names


def sensitivity_train_frame(freq: pd.DataFrame, split: pd.DataFrame, cv_folds: int) -> pd.DataFrame:
    """Train rows of cleaned policies joined to the frozen split.

    Unlike ``train.train_frame`` the cleaned IDpol set may be a strict subset of the split (a
    sensitivity can drop policies); an IDpol that the split does not know raises."""
    if not set(freq["IDpol"]) <= set(split["IDpol"]):
        raise ValueError("cleaned policies are not a subset of the stored split")
    d = freq.merge(split, on="IDpol", validate="one_to_one")
    train = d[d["primary_set"] == "train"].reset_index(drop=True)
    if not train["cv_fold"].between(0, cv_folds - 1).all():
        raise ValueError(f"train rows must have cv_fold in 0..{cv_folds - 1}")
    assert_train_only(train)
    return train


def fixed_power_mcfg(mcfg: dict[str, Any], power: float) -> dict[str, Any]:
    """Models config fitting the Tweedie at ``power`` only (plus the reference power, which
    stays ``models.tweedie.reference_power``: the common deviance of every model)."""
    out = copy.deepcopy(mcfg)
    ref = float(out["tweedie"]["reference_power"])
    out["tweedie"]["fixed_power_sensitivities"] = [] if float(power) == ref else [float(power)]
    return out


def fixed_power_column(mcfg: dict[str, Any], power: float) -> str:
    """Out-of-fold prediction column of the Tweedie fitted at the fixed power."""
    if float(power) == float(mcfg["tweedie"]["reference_power"]):
        return "pp_tweedie_ref"
    return f"pp_{fixed_name(float(power))}"


def freq_x_sev_valid(rules: dict[str, Any]) -> bool:
    """Frequency x severity estimates the capped pure premium only if the frequency count is the
    matched severity row count (the severity denominator)."""
    return rules["targets"]["frequency_count"] == "sev_rows"


def _diff(a: dict[str, Any], b: dict[str, Any], keys: tuple[str, ...]) -> dict[str, float]:
    return {k: a[k] - b[k] for k in keys}


def comparisons(m: dict[str, Any]) -> dict[str, Any]:
    """Within-definition differences of one metric tree (a fold or the pooled folds). The
    frequency x severity comparisons exist only if that candidate was computed."""
    pp = m["pure_premium"]
    pp_keys = (
        "tweedie_ref_deviance_per_exposure",
        "tweedie_ref_deviance_explained",
        "gini_normalized",
    )
    fxs = "freq_x_sev" in pp
    return {
        **(
            {"tweedie_minus_freq_x_sev": _diff(pp["tweedie"], pp["freq_x_sev"], pp_keys)}
            if fxs
            else {}
        ),
        "tweedie_minus_null": _diff(pp["tweedie"], pp["null"], pp_keys),
        **({"freq_x_sev_minus_null": _diff(pp["freq_x_sev"], pp["null"], pp_keys)} if fxs else {}),
        "frequency_full_minus_null": _diff(
            m["frequency"]["full"],
            m["frequency"]["null"],
            ("deviance_per_exposure", "deviance_explained", "gini_normalized"),
        ),
        "severity_full_minus_null": _diff(
            m["severity"]["full"],
            m["severity"]["null"],
            ("deviance_per_claim", "deviance_explained"),
        ),
    }


def cv_one(train: pd.DataFrame, cfg: dict[str, Any], power: float, rules: dict[str, Any]):
    """Outer grouped CV at a fixed power. Returns (out-of-fold frame, result dict, fit records)."""
    assert_train_only(train)
    fcfg = cfg["features"]
    mcfg = fixed_power_mcfg(cfg["models"], power)
    ref = float(mcfg["tweedie"]["reference_power"])
    col = fixed_power_column(mcfg, power)
    models = (
        PP_MODELS if freq_x_sev_valid(rules) else tuple(m for m in PP_MODELS if m != "freq_x_sev")
    )
    grouped = cfg["split"]["primary"] == "grouped"
    k_folds = cfg["split"]["cv_folds"]
    parts = list(outer_splits(train, k_folds, grouped))
    done = Parallel(n_jobs=mcfg["parallel"]["n_jobs"])(
        delayed(_outer_task)(fit, valid, fcfg, mcfg, rules, []) for _, fit, valid in parts
    )
    frames, records = [], []
    cols = ["IDpol", "group_id", "cv_fold", "Exposure", "freq_count", "N", "S", "S_cap"]
    for (k, _, _), r in zip(parts, done, strict=True):
        pred = r["pred"].copy()
        pred["pp_tweedie"] = pred[col]
        frames.append(pd.concat([train.loc[pred.index, cols], pred], axis=1))
        records += [{"stage": "outer", "fold": k, **rec} for rec in r["fits"]]
    oof = pd.concat(frames)
    per_fold = [scalar_metrics(f, ref, models) for f in frames]
    pooled = scalar_metrics(oof, ref, models)
    result = {
        "folds": [{"fold": k, **m} for k, m in enumerate(per_fold)],
        "fold_summary": summarise_folds(per_fold),
        "pooled": pooled,
        "comparisons": {
            "pooled": comparisons(pooled),
            "fold_summary": summarise_folds([comparisons(m) for m in per_fold]),
        },
    }
    return oof, result, records


def rows_summary(train: pd.DataFrame) -> dict[str, Any]:
    return {
        "policies": int(len(train)),
        "exposure": float(train["Exposure"].sum()),
        "claims": int(train["N"].sum()),
        "frequency_events": float(train["freq_count"].sum()),
        "s_cap": float(train["S_cap"].sum()),
        "claim_policies": int((train["N"] > 0).sum()),
    }


def bootstrap_diagnostic(oof: pd.DataFrame, power: float, bcfg: dict[str, Any]) -> dict[str, Any]:
    """Group vs policy paired bootstrap of the primary pure-premium loss (Tweedie deviance at the
    fixed power per exposure-year) of the primary out-of-fold GLM predictions. Diagnostic only."""
    e = oof["Exposure"].to_numpy()
    rate_y = (oof["S_cap"] / oof["Exposure"]).to_numpy()
    unit = {
        m: metrics.tweedie_unit_deviance(rate_y, oof[f"pp_{m}"].to_numpy() / e, power)
        for m in ("freq_x_sev", "tweedie")
    }

    def fn(w: np.ndarray) -> dict[str, float]:
        ww = w * e
        den = ww.sum()
        a, b = (ww * unit["freq_x_sev"]).sum() / den, (ww * unit["tweedie"]).sum() / den
        return {"freq_x_sev": a, "tweedie": b, "tweedie_minus_freq_x_sev": b - a}

    n = len(oof)
    ids = {"group": oof["group_id"].to_numpy(), "policy": np.arange(n)}
    out: dict[str, Any] = {
        "label": "diagnostic only (train, primary definition); the primary unit is unchanged",
        "loss": f"Tweedie deviance per exposure-year at power {power}",
        "replicates": bcfg["replicates"],
        "seed": bcfg["seed"],
        "level": bcfg["level"],
        "units": {},
    }
    for u, unit_ids in ids.items():
        boot = metrics.paired_bootstrap(
            fn, n, unit_ids, bcfg["replicates"], bcfg["seed"], bcfg["level"]
        )
        out["units"][u] = {
            "n_units": int(len(np.unique(unit_ids))),
            "metrics": {
                k: {
                    "point": boot["point"][k],
                    "se": float(boot["replicates"][k].std(ddof=1)),
                    "lower": float(boot["ci"].loc[k, "lower"]),
                    "upper": float(boot["ci"].loc[k, "upper"]),
                    "width": float(boot["ci"].loc[k, "upper"] - boot["ci"].loc[k, "lower"]),
                }
                for k in BOOT_KEYS
            },
        }
    g, p = out["units"]["group"]["metrics"], out["units"]["policy"]["metrics"]
    out["ratio_group_over_policy"] = {
        k: {"se": g[k]["se"] / p[k]["se"], "width": g[k]["width"] / p[k]["width"]}
        for k in BOOT_KEYS
    }
    return out


def run_sensitivities(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run every selected sensitivity (and ``primary``) in outer CV; write json and report."""
    t0 = time.perf_counter()
    scfg = cfg["models"]["sensitivities"]
    power = fixed_power(cfg)
    names = [PRIMARY, *sensitivity_names(cfg)]
    results, timing = Path(cfg["paths"]["results"]), {}
    reports = Path(cfg["paths"]["reports"])
    freq, sev = load_raw(cfg)
    split = pd.read_parquet(Path(cfg["paths"]["processed"]) / "split.parquet")
    cv_folds = cfg["split"]["cv_folds"]
    solver = cfg["models"]["glm"]["solver"]
    out: dict[str, Any] = {}
    boot = None
    for name in names:
        t1 = time.perf_counter()
        rules = sensitivity_rules(cfg["clean"], name)
        res = clean_sensitivity(freq, sev, cfg["clean"], name)
        train = sensitivity_train_frame(res.freq, split, cv_folds)
        oof, r, records = cv_one(train, cfg, power, rules)
        conv = convergence_summary(records, solver)
        if conv["n_not_converged"]:
            raise RuntimeError(
                f"sensitivity '{name}': {conv['n_not_converged']} fit(s) did not converge: "
                f"{conv['not_converged'][:3]}"
            )
        if name == PRIMARY:
            boot = bootstrap_diagnostic(oof, power, cfg["models"]["bootstrap"])
        timing[name] = time.perf_counter() - t1
        out[name] = {
            "overrides": cfg["clean"][SENSITIVITIES_KEY][name] if name != PRIMARY else {},
            "rows": rows_summary(train),
            "freq_x_sev_valid": freq_x_sev_valid(rules),
            "freq_x_sev_note": None
            if freq_x_sev_valid(rules)
            else "frequency count (recorded ClaimNb) differs from the severity denominator "
            "(matched severity rows): frequency x severity does not estimate the capped pure "
            "premium, so it and its comparison with the Tweedie are not computed; frequency "
            "metrics are compared only with their own null",
            **r,
            "convergence": conv,
            "fold_rows": _records(
                train.groupby("cv_fold").agg(
                    policies=("IDpol", "size"), exposure=("Exposure", "sum")
                )
            ),
            "seconds": timing[name],
        }
    timing["total"] = time.perf_counter() - t0
    result = _clean_json(
        {
            "config": {
                "fixed_tweedie_power": power,
                "reference_power": float(cfg["models"]["tweedie"]["reference_power"]),
                "power_source": str(
                    Path(cfg["paths"]["results"]) / cfg["models"]["outputs"]["cv_json"]
                ),
                "split_primary": cfg["split"]["primary"],
                "cv_folds": cv_folds,
                "sensitivities": names,
                "solver": solver,
                "test_set_used": False,
                "cv_only": True,
            },
            "sensitivities": out,
            "bootstrap_unit_diagnostic": boot,
            "timing_seconds": timing,
        }
    )
    outs = scfg["outputs"]
    results.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    for path, text in (
        (results / outs["json"], json.dumps(result, indent=2) + "\n"),
        (reports / outs["report"], render_sensitivities_report(result)),
    ):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    return result
