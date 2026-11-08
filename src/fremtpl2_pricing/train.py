"""Milestone 2: grouped CV of the GLMs on the TRAIN set only, final refits and result files.

The test set is never loaded into a fit, a preprocessing step or a metric: ``load_train`` keeps
``primary_set == 'train'`` rows, and every CV/fit function takes an already train-only frame and
raises if a row has ``cv_fold < 0``. Per outer fold (the registered ``cv_fold``) the preprocessing
(``features.fit``) and the models are fitted on the other folds only; the Tweedie power is chosen
by an inner CV over those remaining folds (each inner split refits the features on its own
training part). The excess loading is only reported, in a separate block; it enters no CV target
or prediction. Every setting comes from the ``models`` / ``features`` / ``clean`` config sections.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from . import features, glm, metrics
from .clean import clean
from .cv_report import render_report
from .data import load_raw
from .targets import excess_loading

FREQ_MODELS = ("null", "bonus_malus_only", "full")
PP_BASE_MODELS = ("null", "bonus_malus_only", "freq_x_sev", "tweedie", "tweedie_ref")


def fixed_name(p: float) -> str:
    """Model name of a fixed-power Tweedie sensitivity."""
    return f"tweedie_fixed_{float(p)}"


def pp_models(mcfg: dict[str, Any]) -> tuple[str, ...]:
    """Pure premium models compared out of fold: the base set plus the configured fixed powers."""
    return PP_BASE_MODELS + tuple(
        fixed_name(p) for p in mcfg["tweedie"]["fixed_power_sensitivities"]
    )


# --------------------------------------------------------------------------- data


def assert_train_only(df: pd.DataFrame) -> None:
    """Raise if any row is not a registered training row (test rows have cv_fold -1)."""
    if "cv_fold" in df and (df["cv_fold"] < 0).any():
        raise ValueError("train-only guard: rows with cv_fold < 0 (test set) are not allowed")
    if "primary_set" in df and (df["primary_set"] != "train").any():
        raise ValueError("train-only guard: rows with primary_set != 'train' are not allowed")


def load_train_claims(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw -> clean (primary rules, full data) -> merge the frozen split -> train rows only.

    Also returns the cleaned matched claim rows of the training policies (IDpol, ClaimAmount,
    ClaimAmount_capped), used only for the cap diagnostics."""
    freq, sev = load_raw(cfg)
    res = clean(freq, sev, cfg["clean"])
    split = pd.read_parquet(Path(cfg["paths"]["processed"]) / "split.parquet")
    train = train_frame(res.freq, split, cfg["split"]["cv_folds"])
    return train, res.sev[res.sev["IDpol"].isin(train["IDpol"])].reset_index(drop=True)


def load_train(cfg: dict[str, Any]) -> pd.DataFrame:
    """Train-only policy frame (see :func:`load_train_claims`)."""
    return load_train_claims(cfg)[0]


def train_frame(freq: pd.DataFrame, split: pd.DataFrame, cv_folds: int) -> pd.DataFrame:
    """Merge cleaned policies with the split (IDpol sets must agree) and keep train rows."""
    if set(freq["IDpol"]) != set(split["IDpol"]):
        raise ValueError("IDpol sets of the cleaned data and the stored split differ")
    d = freq.merge(split, on="IDpol", validate="one_to_one")
    train = d[d["primary_set"] == "train"].reset_index(drop=True)
    del d
    if not train["cv_fold"].between(0, cv_folds - 1).all():
        raise ValueError(f"train rows must have cv_fold in 0..{cv_folds - 1}")
    assert_train_only(train)
    return train


# --------------------------------------------------------------------------- splits


def outer_splits(train: pd.DataFrame, cv_folds: int, grouped: bool):
    """Yield (k, fit part, validation part) for the registered folds of a train-only frame."""
    assert_train_only(train)
    for k in range(cv_folds):
        yield k, *_split(train, train["cv_fold"] == k, grouped)


def inner_splits(fit: pd.DataFrame, grouped: bool):
    """Yield (j, inner fit, inner validation) over the folds present in an outer fit part."""
    assert_train_only(fit)
    for j in sorted(fit["cv_fold"].unique()):
        yield int(j), *_split(fit, fit["cv_fold"] == j, grouped)


def _split(df: pd.DataFrame, valid_mask: pd.Series, grouped: bool):
    fit, valid = df[~valid_mask], df[valid_mask]
    if grouped and "group_id" in df and set(fit["group_id"]) & set(valid["group_id"]):
        raise ValueError("group_id overlap between fitting and validation parts")
    return fit, valid


# --------------------------------------------------------------------------- fitting helpers


def _pp(df: pd.DataFrame) -> np.ndarray:
    return (df["S_cap"] / df["Exposure"]).to_numpy()


def _poisson_pp_loss(df: pd.DataFrame, rate: np.ndarray) -> float:
    """Total (not yet divided) exposure-weighted Poisson deviance of S_cap / Exposure.

    Labelled diagnostic only: it is a mean-prediction loss and does not identify the power."""
    return metrics.tweedie_deviance(_pp(df), rate, 1.0, weight=df["Exposure"].to_numpy())


def _fit_record(
    model: glm.GLMModel, name: str, power: float | None = None, expansions: int | None = None
) -> dict[str, Any]:
    """Solver outcome of one fit; ``expansions`` = dispersion bracket expansions (Tweedie CV)."""
    return {
        "model": name,
        "power": power,
        "converged": model.converged,
        "n_iter": model.n_iter,
        "dispersion_expansions": expansions,
    }


def _score_power(model, Xf, fit, Xv, valid, power, mcfg, pointwise=False) -> dict[str, Any]:
    """Held-out scores of one fitted Tweedie power.

    The dispersion is the maximum-likelihood value on the FITTING part (its own means and prior
    weights = exposure); the held-out log-likelihood is the sum of log densities of
    S_cap / Exposure with prior weight = exposure at those means and that dispersion. The
    dispersion bracket is expanded (and the number of expansions returned) when the optimum sits
    at a bound. ``pointwise`` also returns the per-policy log densities."""
    e_fit = fit["Exposure"].to_numpy()
    t = mcfg["tweedie"]
    phi, expansions = metrics.tweedie_mle_dispersion_fit(
        _pp(fit),
        model.predict_rate(Xf),
        power,
        weight=e_fit,
        log_range=t["dispersion_log_range"],
        edge_rtol=t["dispersion_edge_rtol"],
        max_expansions=t["dispersion_max_expansions"],
    )
    rate = model.predict_rate(Xv)
    lp = metrics.tweedie_logpdf(_pp(valid), rate, phi, power, valid["Exposure"].to_numpy())
    out = {
        "loglik": float(lp.sum()),
        "phi": phi,
        "expansions": expansions,
        "poisson_loss": _poisson_pp_loss(valid, rate),
    }
    if pointwise:
        out["logpdf"] = lp
    return out


def _inner_task(fit, valid, fcfg, mcfg, grid) -> dict[str, Any]:
    with threadpool_limits(1):  # single-threaded BLAS/OpenMP: bit-identical reruns
        return _inner(fit, valid, fcfg, mcfg, grid)


def _inner(fit, valid, fcfg, mcfg, grid) -> dict[str, Any]:
    """Features fitted on ``fit`` only; Tweedie full GLM per power; scores on ``valid``."""
    assert_train_only(fit)
    assert_train_only(valid)
    fmap = features.fit(fit, fcfg)
    cols = features.feature_names(fcfg, "full")
    Xf, Xv = features.transform(fit, fmap)[cols], features.transform(valid, fmap)[cols]
    solver = mcfg["glm"]["solver"]
    scores, fits = {}, []
    for p in grid:
        model = glm.fit_tweedie(Xf, _pp(fit), fit["Exposure"].to_numpy(), p, solver)
        scores[p] = _score_power(model, Xf, fit, Xv, valid, p, mcfg)
        fits.append(_fit_record(model, "tweedie", p, scores[p]["expansions"]))
    return {
        "scores": scores,
        "exposure": float(valid["Exposure"].sum()),
        "policies": int(len(valid)),
        "fits": fits,
    }


def _severity_map(claims, fcfg, mcfg, policy_map) -> features.FeatureMap:
    sf = mcfg["severity_features"]
    return features.fit_severity(claims, fcfg, policy_map, sf["min_claims"], sf["ordinal"])


def _severity_inputs(df: pd.DataFrame):
    claims = df[df["N"] > 0]
    return claims, (claims["S_cap"] / claims["N"]).to_numpy(), claims["N"].to_numpy()


def _outer_task(fit, valid, fcfg, mcfg, rules, grid) -> dict[str, Any]:
    with threadpool_limits(1):  # single-threaded BLAS/OpenMP: bit-identical reruns
        return _outer(fit, valid, fcfg, mcfg, rules, grid)


def _outer(fit, valid, fcfg, mcfg, rules, grid) -> dict[str, Any]:
    """Fit every model on ``fit`` only and predict ``valid`` (power grid for the Tweedie)."""
    assert_train_only(fit)
    assert_train_only(valid)
    cols = features.feature_names(fcfg, "full")
    bm_cols = features.feature_names(fcfg, "bonus_malus_only")
    fmap = features.fit(fit, fcfg)
    Xf, Xv = features.transform(fit, fmap), features.transform(valid, fmap)
    e_fit, e_val = fit["Exposure"].to_numpy(), valid["Exposure"].to_numpy()
    count = fit["freq_count"].to_numpy()
    solver = mcfg["glm"]["solver"]
    freq_full = glm.fit_frequency(Xf[cols], count, e_fit, solver)
    freq_bm = glm.fit_frequency(Xf[bm_cols], count, e_fit, solver)
    fits = [_fit_record(freq_full, "frequency_full"), _fit_record(freq_bm, "frequency_bm")]
    null_rate = float(count.sum() / e_fit.sum())
    claims, avg, n_cl = _severity_inputs(fit)
    fmap_s = _severity_map(claims, fcfg, mcfg, fmap)
    sev_full = glm.fit_severity(features.transform(claims, fmap_s)[cols], avg, n_cl, solver)
    fits.append(_fit_record(sev_full, "severity_full"))
    sev_null = float((claims["S_cap"].sum()) / n_cl.sum())
    sev_val = sev_full.predict_rate(features.transform(valid, fmap_s)[cols])
    f_full = freq_full.predict_rate(Xv[cols])
    f_bm = freq_bm.predict_rate(Xv[bm_cols])
    pred = pd.DataFrame(
        {
            "freq_null": null_rate * e_val,
            "freq_bonus_malus_only": f_bm * e_val,
            "freq_full": f_full * e_val,
            "sev_null": np.full(len(valid), sev_null),
            "sev_full": sev_val,
            "pp_null": null_rate * sev_null * e_val,
            "pp_bonus_malus_only": f_bm * sev_null * e_val,
            "pp_freq_x_sev": f_full * sev_val * e_val,
        },
        index=valid.index,
    )
    ref = mcfg["tweedie"]["reference_power"]
    fixed = [float(p) for p in mcfg["tweedie"]["fixed_power_sensitivities"]]
    scores, pointwise = {}, {}
    for p in sorted(set(grid) | {ref} | set(fixed)):
        model = glm.fit_tweedie(Xf[cols], _pp(fit), e_fit, p, solver)
        scores[p] = _score_power(model, Xf[cols], fit, Xv[cols], valid, p, mcfg, pointwise=True)
        fits.append(_fit_record(model, "tweedie", p, scores[p]["expansions"]))
        pointwise[p] = scores[p].pop("logpdf")
        pred[f"pp_tweedie_{p}"] = model.predict_rate(Xv[cols]) * e_val
    pred["pp_tweedie_ref"] = pred[f"pp_tweedie_{ref}"]
    for p in fixed:
        pred[f"pp_{fixed_name(p)}"] = pred[f"pp_tweedie_{p}"]
    return {
        "pred": pred,
        "scores": scores,
        "pointwise": pointwise,
        "exposure": float(e_val.sum()),
        "policies": int(len(valid)),
        "excess_loading": excess_loading(fit, rules),
        "fits": fits,
    }


# --------------------------------------------------------------------------- metrics


def _clean_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return float(obj) if np.isfinite(obj) else None
    return obj


def scalar_metrics(
    d: pd.DataFrame, ref_power: float, models: tuple[str, ...] = PP_BASE_MODELS
) -> dict[str, Any]:
    """Scalar metrics of one out-of-fold frame (a fold or all pooled folds)."""
    e = d["Exposure"].to_numpy()
    out: dict[str, Any] = {"frequency": {}, "severity": {}, "pure_premium": {}}
    y = d["freq_count"].to_numpy()
    dev_null = metrics.poisson_deviance_per_exposure(y, d["freq_null"].to_numpy(), e)
    for m in FREQ_MODELS:
        mu = d[f"freq_{m}"].to_numpy()
        dev = metrics.poisson_deviance_per_exposure(y, mu, e)
        out["frequency"][m] = {
            "deviance_per_exposure": dev,
            "deviance_explained": metrics.deviance_explained(dev, dev_null),
            **metrics.gini(y, mu / e, e),
        }
    cl = d[d["N"] > 0]
    ya, w = (cl["S_cap"] / cl["N"]).to_numpy(), cl["N"].to_numpy()
    dev_null = metrics.gamma_deviance(ya, cl["sev_null"].to_numpy(), w) / w.sum()
    for m in ("null", "full"):
        dev = metrics.gamma_deviance(ya, cl[f"sev_{m}"].to_numpy(), w) / w.sum()
        out["severity"][m] = {
            "deviance_per_claim": dev,
            "deviance_explained": metrics.deviance_explained(dev, dev_null),
        }
    s, rate_y = d["S_cap"].to_numpy(), _pp(d)

    def devs(m):
        mu = d[f"pp_{m}"].to_numpy() / e
        return (
            metrics.tweedie_deviance(rate_y, mu, 1.0, weight=e) / e.sum(),
            metrics.tweedie_deviance(rate_y, mu, ref_power, weight=e) / e.sum(),
        )

    null_pois, null_ref = devs("null")
    for m in models:
        pois, ref = devs(m)
        mu_tot = d[f"pp_{m}"].to_numpy()
        out["pure_premium"][m] = {
            "poisson_deviance_per_exposure": pois,
            "poisson_deviance_explained": metrics.deviance_explained(pois, null_pois),
            "tweedie_ref_deviance_per_exposure": ref,
            "tweedie_ref_deviance_explained": metrics.deviance_explained(ref, null_ref),
            **metrics.gini(s, mu_tot / e, e),
            "totals": metrics.calibration_totals(s, mu_tot),
        }
    return out


def summarise_folds(fold_metrics: list[Any]) -> Any:
    """Mean and sample sd (ddof=1) across folds for every scalar leaf of the metric trees."""
    first = fold_metrics[0]
    if isinstance(first, dict):
        return {k: summarise_folds([f[k] for f in fold_metrics]) for k in first}
    vals = np.array([np.nan if v is None else v for v in fold_metrics], dtype=float)
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
    return {"mean": float(np.mean(vals)), "sd": sd}


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.reset_index().to_dict("records")


def factor_levels(df: pd.DataFrame, name: str, spec: dict, fcfg: dict):
    """Calibration level of every row and the natural level order."""
    kind = spec["kind"]
    if kind == "raw":
        lv = df[name].astype(str)
        return lv, sorted(lv.unique())
    edges = spec["lower_edges"] if kind == "edges" else fcfg["definitions"][name]["lower_edges"]
    if kind not in ("edges", "feature_band"):
        raise ValueError(f"models.evaluation.calibration_factors.{name}: unknown kind {kind!r}")
    return features._band(df[name], edges), features._band_labels(edges)


def tables(oof: pd.DataFrame, train: pd.DataFrame, mcfg, fcfg) -> dict[str, Any]:
    """Pooled lift, double-lift and calibration tables on out-of-fold predictions."""
    ev = mcfg["evaluation"]
    nb = ev["n_bins"]
    e, s, y = oof["Exposure"], oof["S_cap"], oof["freq_count"]
    lift = {
        "frequency": {
            m: _records(metrics.lift_table(y, oof[f"freq_{m}"], e, nb)) for m in FREQ_MODELS
        },
        "pure_premium": {
            m: _records(metrics.lift_table(s, oof[f"pp_{m}"], e, nb)) for m in pp_models(mcfg)
        },
    }
    double = _records(metrics.double_lift_table(s, oof["pp_tweedie"], oof["pp_freq_x_sev"], e, nb))
    calib: dict[str, Any] = {}
    for name, spec in ev["calibration_factors"].items():
        lv, order = factor_levels(train.loc[oof.index], name, spec, fcfg)
        entry = {}
        for target, obs, models in (
            ("frequency", y, {m: oof[f"freq_{m}"] for m in FREQ_MODELS}),
            ("pure_premium", s, {m: oof[f"pp_{m}"] for m in pp_models(mcfg)}),
        ):
            entry[target] = {}
            for m, pred in models.items():
                t = metrics.calibration_by_level(obs, pred, lv.to_numpy())
                t = t.reindex([o for o in order if o in t.index])
                entry[target][m] = _records(t.rename_axis("level"))
        calib[name] = entry
    return {"lift": lift, "double_lift": double, "calibration": calib}


# --------------------------------------------------------------------------- run


def _grid(mcfg) -> list[float]:
    return [float(p) for p in mcfg["tweedie"]["power_grid"]]


def check_level_coverage(train: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Fail fast, before any model is fitted, if the policy schema of a fitting part does not
    cover the validation part it is applied to.

    For each outer fold the policy map is fitted on the fitting part, the severity map is derived
    from it and the claims-only fitting rows, and both are applied to the validation part; for
    each inner split the policy map is applied to the inner validation part (the inner loop fits
    no severity model). A level that is absent from the policy schema of the fitting part (and is
    not grouped into the other level) raises in ``features.transform``; the severity map covers
    every level of the policy schema by construction (merge rules), so only the policy schema can
    fail. All offending splits are listed in one error instead of a late crash.
    """
    mcfg, fcfg = cfg["models"], cfg["features"]
    grouped = cfg["split"]["primary"] == "grouped"
    problems: list[str] = []

    def check(label, fit, valid, severity):
        try:
            fmap = features.fit(fit, fcfg)
            if severity:
                fmap = _severity_map(fit[fit["N"] > 0], fcfg, mcfg, fmap)
            features.transform(valid, fmap)
        except ValueError as exc:
            problems.append(f"{label}: {exc}")

    for k, fit, valid in outer_splits(train, cfg["split"]["cv_folds"], grouped):
        check(f"outer fold {k} policy map", fit, valid, False)
        check(f"outer fold {k} severity map", fit, valid, True)
        for j, ifit, ivalid in inner_splits(fit, grouped):
            check(f"outer fold {k} inner fold {j} policy map", ifit, ivalid, False)
    if problems:
        raise ValueError(
            "feature levels of a validation part are not covered by the policy schema of its "
            "fitting part (raise the rare-level grouping or lower the thresholds):\n  "
            + "\n  ".join(problems)
        )


def _power_diagnostics(grid, loglik: dict[float, float]) -> dict[str, Any]:
    """Selected power plus boundary / flatness diagnostic of a held-out log-likelihood curve."""
    ll = np.array([loglik[p] for p in grid])
    best = int(np.argmax(ll))
    return {
        "power": grid[best],
        "at_grid_edge": best in (0, len(grid) - 1),
        "loglik_spread_nats": float(ll.max() - ll.min()),
        "best_minus_next_nats": float(ll[best] - np.sort(ll)[-2]) if len(ll) > 1 else None,
    }


def grid_neighbours(grid: list[float], power: float) -> list[float]:
    """Grid neighbours (lower, upper) of ``power``; missing at a grid edge."""
    i = grid.index(power)
    return [grid[j] for j in (i - 1, i + 1) if 0 <= j < len(grid)]


def final_power_diagnostics(
    grid, outer: dict, k_folds: int, selected: list, capped_mask: dict[int, np.ndarray]
) -> dict[str, Any]:
    """Paired diagnostics of the final-CV power choice (outer folds only; no test row).

    For the selected power and each neighbour on the grid: the per-fold held-out log-likelihood
    difference (selected minus neighbour), the number of folds with the same sign as the total,
    and the total difference split into the policies with a capped claim and the others.
    Also the per-fold log-likelihood of every power and the stability of the per-fold choice."""
    cv_ll = {p: sum(outer[k]["scores"][p]["loglik"] for k in range(k_folds)) for p in grid}
    diag = _power_diagnostics(grid, cv_ll)
    neighbours = []
    for q in grid_neighbours(grid, diag["power"]):
        per_fold = [
            outer[k]["scores"][diag["power"]]["loglik"] - outer[k]["scores"][q]["loglik"]
            for k in range(k_folds)
        ]
        total = float(sum(per_fold))
        cap_diff = other_diff = 0.0
        for k in range(k_folds):
            d = outer[k]["pointwise"][diag["power"]] - outer[k]["pointwise"][q]
            cap_diff += float(d[capped_mask[k]].sum())
            other_diff += float(d[~capped_mask[k]].sum())
        neighbours.append(
            {
                "neighbour": q,
                "total_diff_nats": total,
                "per_fold_diff_nats": per_fold,
                "folds_agreeing_in_sign": int(sum(np.sign(x) == np.sign(total) for x in per_fold)),
                "n_folds": k_folds,
                "capped_policies_diff_nats": cap_diff,
                "other_policies_diff_nats": other_diff,
            }
        )
    chosen = [s_["power"] for s_ in selected]
    return {
        **diag,
        "fold_loglik": {
            k: {p: outer[k]["scores"][p]["loglik"] for p in grid} for k in range(k_folds)
        },
        "neighbours": neighbours,
        "stability": {
            "selected_by_outer_fold": chosen,
            "distinct_powers": sorted(set(chosen)),
            "folds_selecting_final": int(sum(c == diag["power"] for c in chosen)),
            "n_folds": k_folds,
        },
    }


def cross_validate(train: pd.DataFrame, cfg: dict[str, Any], capped_ids=None) -> dict[str, Any]:
    """Grouped CV on a train-only frame. Returns the CV result dict and the final-power scores.

    ``capped_ids``: IDpol of the policies with a capped claim (default: ``excess > 0``)."""
    assert_train_only(train)
    mcfg, fcfg = cfg["models"], cfg["features"]
    grid, ref = _grid(mcfg), mcfg["tweedie"]["reference_power"]
    if mcfg["tweedie"]["selection_loss"] != "tweedie_loglik":
        raise ValueError("models.tweedie.selection_loss: only 'tweedie_loglik' is supported")
    grouped = cfg["split"]["primary"] == "grouped"
    k_folds = cfg["split"]["cv_folds"]
    timing: dict[str, float] = {}
    t0 = time.perf_counter()
    check_level_coverage(train, cfg)
    timing["coverage_check"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    jobs = []
    for k, fit, valid in outer_splits(train, k_folds, grouped):
        jobs.append(("outer", k, None, fit, valid))
        for j, ifit, ivalid in inner_splits(fit, grouped):
            jobs.append(("inner", k, j, ifit, ivalid))
    n_jobs = mcfg["parallel"]["n_jobs"]
    done = Parallel(n_jobs=n_jobs)(
        delayed(_outer_task if kind == "outer" else _inner_task)(
            *(
                (fit, valid, fcfg, mcfg, cfg["clean"], grid)
                if kind == "outer"
                else (fit, valid, fcfg, mcfg, grid)
            )
        )
        for kind, _, _, fit, valid in jobs
    )
    timing["cv_fits"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    fit_records = [
        {"stage": kind, "fold": k, "inner_fold": j, **rec}
        for (kind, k, j, _, _), r in zip(jobs, done, strict=True)
        for rec in r["fits"]
    ]
    outer = {k: r for (kind, k, _, _, _), r in zip(jobs, done, strict=True) if kind == "outer"}
    selected, frames, fold_metrics = [], [], []
    for k in range(k_folds):
        inner = [r for (kind, kk, _, _, _), r in zip(jobs, done, strict=True)
                 if kind == "inner" and kk == k]  # fmt: skip
        n_pol = sum(r["policies"] for r in inner)
        ll = {p: sum(r["scores"][p]["loglik"] for r in inner) for p in grid}
        expo = sum(r["exposure"] for r in inner)
        diag = _power_diagnostics(grid, ll)
        selected.append(
            {
                "fold": k,
                **diag,
                "inner_loglik": ll,
                "inner_loglik_per_policy": {p: v / n_pol for p, v in ll.items()},
                "inner_dispersion": {
                    p: float(np.mean([r["scores"][p]["phi"] for r in inner])) for p in grid
                },
                "diagnostic_inner_poisson_deviance_per_exposure": {
                    p: sum(r["scores"][p]["poisson_loss"] for r in inner) / expo for p in grid
                },
            }
        )
        pred = outer[k]["pred"].copy()
        pred["pp_tweedie"] = pred[f"pp_tweedie_{diag['power']}"]
        v = train.loc[pred.index, ["IDpol", "cv_fold", "Exposure", "freq_count", "N", "S", "S_cap"]]
        frame = pd.concat([v, pred], axis=1)
        frame["excess_loading"] = outer[k]["excess_loading"]
        frames.append(frame)
        fold_metrics.append({"fold": k, **scalar_metrics(frame, ref, pp_models(mcfg))})
    oof = pd.concat(frames)
    per_fold = [{k_: v for k_, v in m.items() if k_ != "fold"} for m in fold_metrics]
    # final power: plain CV over the grid on the registered folds (dispersion per outer fit part)
    n_all = sum(outer[k]["policies"] for k in range(k_folds))
    tot = sum(outer[k]["exposure"] for k in range(k_folds))
    cv_ll = {p: sum(outer[k]["scores"][p]["loglik"] for k in range(k_folds)) for p in grid}
    if capped_ids is None:
        capped = train["excess"] > 0
    else:
        capped = train["IDpol"].isin(capped_ids)
    capped_mask = {k: capped.loc[outer[k]["pred"].index].to_numpy() for k in range(k_folds)}
    final_diag = final_power_diagnostics(grid, outer, k_folds, selected, capped_mask)
    result = {
        "selected_powers": selected,
        "folds": [{"fold": k, **per_fold[k]} for k in range(k_folds)],
        "fold_summary": summarise_folds(per_fold),
        "pooled": scalar_metrics(oof, ref, pp_models(mcfg)),
        **tables(oof, train, mcfg, fcfg),
    }
    loaded = loaded_premium(oof, cfg["clean"])
    timing["cv_metrics"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    sel = final_diag["power"]
    nbrs = grid_neighbours(grid, sel)
    check_cv = Parallel(n_jobs=n_jobs)(
        delayed(_solver_check_task)(
            fit,
            valid,
            fcfg,
            mcfg,
            sel,
            k,
            nbrs,
            {p: outer[k]["scores"][p]["loglik"] for p in [sel, *nbrs]},
        )  # fmt: skip
        for k, fit, valid in outer_splits(train, k_folds, grouped)
    )
    timing["solver_check_cv"] = time.perf_counter() - t2
    return {
        "cv": result,
        "final_power_cv": {
            **final_diag,
            "cv_loglik": cv_ll,
            "cv_loglik_per_policy": {p: v / n_all for p, v in cv_ll.items()},
            "diagnostic_cv_poisson_deviance_per_exposure": {
                p: sum(outer[k]["scores"][p]["poisson_loss"] for k in range(k_folds)) / tot
                for p in grid
            },
        },
        "loaded_premium": loaded,
        "fit_records": fit_records,
        "solver_check_cv": {
            "records": [r_ for c in check_cv for r_ in c["records"]],
            "pure_premium": [c["pure_premium"] for c in check_cv],
            "loglik_checks": loglik_difference_checks(
                [c["loglik"] for c in check_cv],
                sel,
                nbrs,
                mcfg["glm"]["solver_check"]["max_loglik_diff_rel_change"],
            ),
        },
        "timing": timing,
    }


def loaded_premium(oof: pd.DataFrame, rules: dict[str, Any]) -> dict[str, Any]:
    """Separately labelled totals including the excess loading (per fold, from its fitting part).

    Informational only: the loading enters no CV target, prediction or metric."""
    load = oof["excess_loading"] * oof["Exposure"]
    obs, pred = float(oof["S"].sum()), float(oof["pp_tweedie"].sum())
    return {
        "label": "loaded premium totals (reporting only; not part of any CV metric)",
        "mode": rules["excess_loading"]["mode"],
        "loading_per_exposure_year_by_fold": [
            float(oof.loc[oof["cv_fold"] == k, "excess_loading"].iloc[0])
            for k in sorted(oof["cv_fold"].unique())
        ],
        "observed_uncapped_cost_total": obs,
        "predicted_tweedie_capped_total": pred,
        "loading_total": float(load.sum()),
        "predicted_loaded_total": pred + float(load.sum()),
        "ratio_observed_to_loaded": obs / (pred + float(load.sum())),
    }


def _compare_fits(a: glm.GLMModel, b: glm.GLMModel, X_list, deviance) -> dict[str, Any]:
    """Accuracy of fit ``a`` (pipeline solver) against ``b`` (tight solver) on the same data.

    Relative prediction difference (max over the frames in ``X_list``), relative deviance
    difference of ``deviance(rate)`` on the fitting rows (first frame) and the largest coefficient
    difference (intercept included)."""
    rel_pred = 0.0
    for X in X_list:
        ra, rb = a.predict_rate(X), b.predict_rate(X)
        rel_pred = max(rel_pred, float(np.max(np.abs(ra - rb) / rb)))
    da, db = deviance(a.predict_rate(X_list[0])), deviance(b.predict_rate(X_list[0]))
    coef = np.abs(np.r_[a.intercept, a.coef] - np.r_[b.intercept, b.coef])
    return {
        "rel_prediction_diff": rel_pred,
        "rel_deviance_diff": float(abs(da - db) / abs(db)) if db else 0.0,
        "coef_diff": float(coef.max()),
        "deviance": float(db),
        "n_iter": a.n_iter,
        "n_iter_tight": b.n_iter,
        "tight_converged": bool(b.converged),
    }


def _solver_check_task(fit, valid, fcfg, mcfg, power, k, neighbours, cfg_ll) -> dict[str, Any]:
    """Tight-solver refits of one outer fold: the Tweedie at the selected power, the frequency and
    severity fits (and their combined out-of-fold pure premium) and the Tweedie held-out
    log-likelihood at the selected power and its grid neighbours (``cfg_ll``: pipeline-solver
    log-likelihoods of those powers)."""
    with threadpool_limits(1):
        assert_train_only(fit)
        assert_train_only(valid)
        cols = features.feature_names(fcfg, "full")
        fmap = features.fit(fit, fcfg)
        Xf, Xv = features.transform(fit, fmap)[cols], features.transform(valid, fmap)[cols]
        e, pp, solver = fit["Exposure"].to_numpy(), _pp(fit), mcfg["glm"]["solver"]
        tight_s = _tight_solver(mcfg)
        e_val = valid["Exposure"].to_numpy()
        base = {"stage": "cv", "fold": k, "power": None}
        records = []
        cfg_fit = glm.fit_tweedie(Xf, pp, e, power, solver)
        tight = glm.fit_tweedie(Xf, pp, e, power, tight_s)
        cmp = _compare_fits(
            cfg_fit, tight, [Xf, Xv], lambda r: metrics.tweedie_deviance(pp, r, power, weight=e)
        )
        records.append({**base, "model": "tweedie", "power": power, **cmp})
        count = fit["freq_count"].to_numpy()
        f_cfg = glm.fit_frequency(Xf, count, e, solver)
        f_tight = glm.fit_frequency(Xf, count, e, tight_s)
        records.append(
            {
                **base,
                "model": "frequency",
                **_compare_fits(
                    f_cfg, f_tight, [Xf, Xv], lambda r: metrics.poisson_deviance(count, r * e)
                ),
            }
        )
        claims, avg, n_cl = _severity_inputs(fit)
        fmap_s = _severity_map(claims, fcfg, mcfg, fmap)
        Xs, Xsv = features.transform(claims, fmap_s)[cols], features.transform(valid, fmap_s)[cols]
        s_cfg = glm.fit_severity(Xs, avg, n_cl, solver)
        s_tight = glm.fit_severity(Xs, avg, n_cl, tight_s)
        records.append(
            {
                **base,
                "model": "severity",
                **_compare_fits(
                    s_cfg, s_tight, [Xs, Xsv], lambda r: metrics.gamma_deviance(avg, r, n_cl)
                ),
            }
        )
        pp_cfg = f_cfg.predict_rate(Xv) * e_val * s_cfg.predict_rate(Xsv)
        pp_tight = f_tight.predict_rate(Xv) * e_val * s_tight.predict_rate(Xsv)
        absd = np.abs(pp_cfg - pp_tight)
        reld = absd / pp_tight
        i_rel, i_abs = int(np.argmax(reld)), int(np.argmax(absd))
        pure_premium = {
            "stage": "cv",
            "fold": k,
            "model": "freq_x_sev",
            "rel_diff": float(reld[i_rel]),
            "abs_diff": float(absd[i_abs]),
            "rel_diff_at_prediction_quantile": float(np.mean(pp_tight <= pp_tight[i_rel])),
            "abs_diff_at_prediction_quantile": float(np.mean(pp_tight <= pp_tight[i_abs])),
            "prediction_at_max_rel": float(pp_tight[i_rel]),
            "prediction_at_max_abs": float(pp_tight[i_abs]),
            "mean_abs_diff": float(absd.mean()),
            "total_rel_diff": float(abs(pp_cfg.sum() - pp_tight.sum()) / pp_tight.sum()),
        }
        loglik = {}
        for q in [power, *neighbours]:
            model = tight if q == power else glm.fit_tweedie(Xf, pp, e, q, tight_s)
            loglik[q] = _score_power(model, Xf, fit, Xv, valid, q, mcfg)["loglik"]
        return {
            "records": records,
            "pure_premium": pure_premium,
            "loglik": {"fold": k, "tight": loglik, "config": cfg_ll},
        }


def loglik_difference_checks(
    checks: list[dict[str, Any]], power: float, neighbours: list[float], max_rel_change: float
) -> list[dict[str, Any]]:
    """Per fold and neighbour: the selected-minus-neighbour held-out log-likelihood difference
    with the pipeline solver and with the tight solver; sign kept and relative change below
    ``max_rel_change``; plus the same on the total over folds."""

    def row(fold, q, d_cfg, d_tight):
        change = abs(d_tight - d_cfg) / abs(d_cfg) if d_cfg else float("inf")
        return {
            "fold": fold,
            "neighbour": q,
            "diff_config": d_cfg,
            "diff_tight": d_tight,
            "abs_change": abs(d_tight - d_cfg),
            "rel_change": change,
            "sign_kept": bool(np.sign(d_cfg) == np.sign(d_tight)),
            "passed": bool(np.sign(d_cfg) == np.sign(d_tight) and change < max_rel_change),
        }

    out = []
    for q in neighbours:
        diffs = [
            (c["fold"], c["config"][power] - c["config"][q], c["tight"][power] - c["tight"][q])
            for c in checks
        ]
        out += [row(f, q, a, b) for f, a, b in diffs]
        out.append(row("total", q, sum(a for _, a, _ in diffs), sum(b for _, _, b in diffs)))
    return out


def _tight_solver(mcfg: dict[str, Any]) -> dict[str, Any]:
    sc = mcfg["glm"]["solver_check"]
    return {"max_iter": sc["max_iter"], "gradient_tol": sc["gradient_tol"]}


def solver_check_summary(
    records: list[dict[str, Any]],
    mcfg: dict[str, Any],
    pure_premium: list[dict[str, Any]] | None = None,
    loglik_checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Maxima over the solver-check records and the verdict against the configured thresholds.

    ``pure_premium``: per outer fold, the combined freq x sev out-of-fold prediction differences
    (relative limit ``max_rel_pure_premium_diff``; absolute differences are recorded only);
    ``loglik_checks``: selected-minus-neighbour log-likelihood difference checks."""
    sc = mcfg["glm"]["solver_check"]
    pure_premium, loglik_checks = pure_premium or [], loglik_checks or []
    limits = {k: sc[f"max_{k}"] for k in ("rel_prediction_diff", "rel_deviance_diff", "coef_diff")}
    maxima = {k: max((r[k] for r in records), default=0.0) for k in limits}
    failures = [
        {k: r[k] for k in ("stage", "fold", "model", "power")} | {"metric": m, "value": r[m]}
        for r in records
        for m in limits
        if r[m] > limits[m]
    ] + [
        {k: r[k] for k in ("stage", "fold", "model", "power")} | {"metric": "tight_converged"}
        for r in records
        if not r["tight_converged"]
    ]
    pp_limit = sc["max_rel_pure_premium_diff"]
    failures += [
        {"stage": r["stage"], "fold": r["fold"], "model": r["model"], "power": None}
        | {"metric": "pure_premium_rel_diff", "value": r["rel_diff"]}
        for r in pure_premium
        if r["rel_diff"] > pp_limit
    ]
    failures += [
        {"stage": "cv", "fold": r["fold"], "model": "tweedie_loglik_difference", "power": None}
        | {"metric": "loglik_difference", "neighbour": r["neighbour"], "value": r["rel_change"]}
        for r in loglik_checks
        if not r["passed"]
    ]
    worst_rel = max(pure_premium, key=lambda r: r["rel_diff"], default=None)
    worst_abs = max(pure_premium, key=lambda r: r["abs_diff"], default=None)
    return {
        "tight_solver": _tight_solver(mcfg),
        "thresholds": {
            **limits,
            "rel_pure_premium_diff": pp_limit,
            "loglik_difference_rel_change": sc["max_loglik_diff_rel_change"],
        },
        "n_checks": len(records),
        "records": records,
        "maxima": maxima,
        "pure_premium": {
            "records": pure_premium,
            "max_rel_diff": worst_rel["rel_diff"] if worst_rel else 0.0,
            "max_rel_diff_at_prediction_quantile": (
                worst_rel["rel_diff_at_prediction_quantile"] if worst_rel else None
            ),
            "max_abs_diff": worst_abs["abs_diff"] if worst_abs else 0.0,
            "max_abs_diff_at_prediction_quantile": (
                worst_abs["abs_diff_at_prediction_quantile"] if worst_abs else None
            ),
        },
        "loglik_checks": {
            "records": loglik_checks,
            "max_rel_change": max((r["rel_change"] for r in loglik_checks), default=0.0),
            "all_signs_kept": all(r["sign_kept"] for r in loglik_checks),
        },
        "failures": failures,
        "passed": not failures,
    }


def group_summary(train: pd.DataFrame, gcfg: dict[str, Any], cv_folds: int) -> dict[str, Any]:
    """Rating-profile group concentration on the train set (the bootstrap unit): number of
    groups, size quantiles (policies per group) and the exposure share of the largest groups
    (by policy count; ties by group id), overall and per CV fold."""
    assert_train_only(train)

    def one(df: pd.DataFrame) -> dict[str, Any]:
        g = df.groupby("group_id").agg(size=("IDpol", "size"), exposure=("Exposure", "sum"))
        g = g.sort_values(["size"], ascending=False, kind="stable")
        n_top = max(1, int(np.ceil(gcfg["top_share"] * len(g))))
        return {
            "groups": int(len(g)),
            "policies": int(g["size"].sum()),
            "size_quantiles": {
                str(q): float(g["size"].quantile(q)) for q in gcfg["size_quantiles"]
            },
            "max_group_size": int(g["size"].max()),
            "top_share": gcfg["top_share"],
            "top_groups": n_top,
            "top_policy_share": float(g["size"].iloc[:n_top].sum() / g["size"].sum()),
            "top_exposure_share": float(g["exposure"].iloc[:n_top].sum() / g["exposure"].sum()),
        }

    return {
        "overall": one(train),
        "per_fold": [{"fold": k, **one(train[train["cv_fold"] == k])} for k in range(cv_folds)],
    }


def cap_diagnostics(
    train: pd.DataFrame, claims: pd.DataFrame, cap: float | None, rtol: float = 1e-9
) -> dict[str, Any]:
    """Cap effect from the cleaned claim rows (ClaimAmount vs ClaimAmount_capped) of the training
    policies: capped claims, policies with one, their exposure, the amount removed and the share
    of S_cap carried by the capped claims. The claim-row S_cap total must equal the policy-level
    total within the relative tolerance ``rtol`` (models.cap_reconciliation_rtol), else raise."""
    assert_train_only(train)
    if set(claims["IDpol"]) - set(train["IDpol"]):
        raise ValueError("cap diagnostics: claim rows of policies outside the training set")
    is_capped = (claims["ClaimAmount"] > claims["ClaimAmount_capped"]).to_numpy()
    capped = claims[is_capped]
    ids = set(capped["IDpol"])
    s_cap = float(train["S_cap"].sum())
    removed = float((capped["ClaimAmount"] - capped["ClaimAmount_capped"]).sum())
    at_cap = float(capped["ClaimAmount_capped"].sum())
    in_capped = train["IDpol"].isin(ids)
    claim_s_cap = float(claims["ClaimAmount_capped"].sum())
    if abs(claim_s_cap - s_cap) > rtol * max(abs(s_cap), 1.0):
        raise ValueError(
            f"cap reconciliation failed: claim-row S_cap {claim_s_cap!r} differs from the "
            f"policy-level S_cap {s_cap!r} by more than rtol {rtol}"
        )
    return {
        "cap": cap,
        "claims": int(len(claims)),
        "capped_claims": int(is_capped.sum()),
        "policies_with_capped_claim": len(ids),
        "capped_policy_exposure": float(train.loc[in_capped, "Exposure"].sum()),
        "capped_policy_exposure_share": float(
            train.loc[in_capped, "Exposure"].sum() / train["Exposure"].sum()
        ),
        "amount_removed": removed,
        "amount_removed_share_of_uncapped": float(removed / claims["ClaimAmount"].sum()),
        "capped_claims_amount_in_s_cap": at_cap,
        "share_of_s_cap": at_cap / s_cap if s_cap else None,
        "s_cap_total": s_cap,
        "reconciliation": {
            "claim_rows_s_cap": claim_s_cap,
            "policy_s_cap": s_cap,
            "rtol": rtol,
            "relative_difference": abs(claim_s_cap - s_cap) / max(abs(s_cap), 1.0),
            "passed": True,
            "policies_with_excess": int((train["excess"] > 0).sum()),
        },
    }


def dispersion_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Dispersion bracket expansions over the CV Tweedie fits (per fit: in the records)."""
    rows = [r for r in records if r.get("dispersion_expansions") is not None]
    exp = [r for r in rows if r["dispersion_expansions"] > 0]
    return {
        "n_dispersion_fits": len(rows),
        "n_expanded": len(exp),
        "total_expansions": int(sum(r["dispersion_expansions"] for r in rows)),
        "max_expansions": max((r["dispersion_expansions"] for r in rows), default=0),
        "expanded": [
            {k: r[k] for k in ("stage", "fold", "inner_fold", "power", "dispersion_expansions")}
            for r in exp
        ],
    }


def final_fit(train: pd.DataFrame, cfg: dict[str, Any], power: float, out_dir: Path):
    with threadpool_limits(1):
        return _final_fit(train, cfg, power, out_dir)


def _final_fit(train: pd.DataFrame, cfg: dict[str, Any], power: float, out_dir: Path):
    """Refit on the full train set; write coefficient tables, feature maps and model dicts."""
    assert_train_only(train)
    mcfg, fcfg = cfg["models"], cfg["features"]
    outs, g = mcfg["outputs"], mcfg["glm"]
    cols, bm_cols = (
        features.feature_names(fcfg, "full"),
        features.feature_names(fcfg, "bonus_malus_only"),
    )
    fmap = features.fit(train, fcfg)
    X = features.transform(train, fmap)
    e, count = train["Exposure"].to_numpy(), train["freq_count"].to_numpy()
    solver = g["solver"]
    freq = glm.fit_frequency(X[cols], count, e, solver)
    freq_tab = glm.coefficient_table(
        freq,
        X[cols],
        count,
        np.ones(len(e)),
        offset=np.log(e),
        robust=g["robust_se"],
        level=g["ci_level"],
    )
    freq_bm = glm.fit_frequency(X[bm_cols], count, e, solver)
    claims, avg, n_cl = _severity_inputs(train)
    fmap_s = _severity_map(claims, fcfg, mcfg, fmap)
    Xs = features.transform(claims, fmap_s)[cols]
    sev = glm.fit_severity(Xs, avg, n_cl, solver)
    sev_tab = glm.coefficient_table(
        sev, Xs, avg, n_cl, offset=None, robust=g["robust_se"], level=g["ci_level"]
    )
    pp = _pp(train)
    tw = glm.fit_tweedie(X[cols], pp, e, power, solver)
    tw_tab = glm.coefficient_table(
        tw, X[cols], pp, e, offset=None, robust=g["robust_se"], level=g["ci_level"]
    )
    tight = _tight_solver(mcfg)
    mu_dev = {
        "frequency": lambda r: metrics.poisson_deviance(count, r * e),
        "severity": lambda r: metrics.gamma_deviance(avg, r, n_cl),
        "tweedie": lambda r: metrics.tweedie_deviance(pp, r, power, weight=e),
    }
    checks = [
        ("frequency", None, freq, glm.fit_frequency(X[cols], count, e, tight), X[cols]),
        ("severity", None, sev, glm.fit_severity(Xs, avg, n_cl, tight), Xs),
        ("tweedie", power, tw, glm.fit_tweedie(X[cols], pp, e, power, tight), X[cols]),
    ]
    solver_check = [
        {
            "stage": "final",
            "fold": None,
            "model": name,
            "power": pw,
            **_compare_fits(m_cfg, m_tight, [Xc], mu_dev[name]),
        }
        for name, pw, m_cfg, m_tight, Xc in checks
    ]
    freq_tab.to_csv(out_dir / outs["relativities_frequency"], index=False)
    sev_tab.to_csv(out_dir / outs["relativities_severity"], index=False)
    tw_tab.to_csv(out_dir / outs["relativities_tweedie"], index=False)
    (out_dir / outs["feature_map"]).write_text(
        json.dumps({"policy": fmap.to_dict(), "severity": fmap_s.to_dict()}, indent=2) + "\n",
        encoding="utf-8",
    )
    models = {
        "frequency": freq.to_dict(),
        "frequency_bonus_malus_only": freq_bm.to_dict(),
        "severity": sev.to_dict(),
        "tweedie": tw.to_dict(),
    }
    (out_dir / outs["models"]).write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    return {
        "tweedie_power": power,
        "severity_claim_rows": int(len(claims)),
        "solver_check": solver_check,
        "fits": [
            _fit_record(freq, "frequency_full"),
            _fit_record(freq_bm, "frequency_bm"),
            _fit_record(sev, "severity_full"),
            _fit_record(tw, "tweedie", power),
        ],
        "files": [
            outs[k]
            for k in (
                "relativities_frequency",
                "relativities_severity",
                "relativities_tweedie",
                "feature_map",
                "models",
            )
        ],
    }


def config_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return {
        "config_sha256": hashlib.sha256(blob).hexdigest(),
        "seed": cfg["seed"],
        "split_primary": cfg["split"]["primary"],
        "cv_folds": cfg["split"]["cv_folds"],
        "clean_targets": cfg["clean"]["targets"],
        "large_loss_cap": cfg["clean"]["large_loss"]["cap"],
        "excess_loading_mode": cfg["clean"]["excess_loading"]["mode"],
        "models": cfg["models"],
        "test_set_used": False,
        "bootstrap_applied": False,
    }


def convergence_summary(records: list[dict[str, Any]], solver: dict[str, Any]) -> dict[str, Any]:
    """Count of fits and of fits that warned of non-convergence (with which fit it was)."""
    bad = [r for r in records if r["converged"] is False]
    return {
        "solver": solver,
        "n_fits": len(records),
        "n_not_converged": len(bad),
        "max_n_iter": max((r["n_iter"] for r in records), default=0),
        "not_converged": bad,
    }


def pipeline_checks(solver_check, convergence, cap_diag) -> dict[str, Any]:
    """Gates of the promotion of the model artifacts: solver accuracy, cap reconciliation (already
    enforced by ``cap_diagnostics``) and convergence of every fit."""
    gates = {
        "solver_check": bool(solver_check["passed"]),
        "cap_reconciliation": bool(cap_diag["reconciliation"]["passed"]),
        "convergence": convergence["n_not_converged"] == 0,
    }
    return {**gates, "passed": all(gates.values())}


def train_models(
    train: pd.DataFrame, cfg: dict[str, Any], claims: pd.DataFrame, out_dir: Path | None = None
):
    """Whole milestone on a train-only frame and its cleaned claim rows (cap diagnostics only);
    writes files when ``out_dir`` is given."""
    t0 = time.perf_counter()
    cap_diag = cap_diagnostics(
        train, claims, cfg["clean"]["large_loss"]["cap"], cfg["models"]["cap_reconciliation_rtol"]
    )  # raises on a reconciliation failure before any model is fitted
    capped_ids = set(claims.loc[claims["ClaimAmount"] > claims["ClaimAmount_capped"], "IDpol"])
    res = cross_validate(train, cfg, capped_ids)
    solver_cv_all = res.pop("solver_check_cv")
    timing = dict(res.pop("timing"))
    power_cv = res.pop("final_power_cv")
    fit_records = res.pop("fit_records")
    final_power = power_cv["power"]
    t1 = time.perf_counter()
    final = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        final = final_fit(train, cfg, final_power, out_dir)
    fit_records += [{"stage": "final", "fold": None, "inner_fold": None, **r}
                    for r in (final or {}).get("fits", [])]  # fmt: skip
    timing["final_fit"] = time.perf_counter() - t1
    solver_check = solver_check_summary(
        solver_cv_all["records"] + (final or {}).get("solver_check", []),
        cfg["models"],
        solver_cv_all["pure_premium"],
        solver_cv_all["loglik_checks"],
    )
    timing["total"] = time.perf_counter() - t0
    convergence = convergence_summary(fit_records, cfg["models"]["glm"]["solver"])
    fold_rows = train.groupby("cv_fold").agg(
        policies=("IDpol", "size"), exposure=("Exposure", "sum"), claims=("N", "sum")
    )
    result = {
        "config": config_summary(cfg),
        "rows": {
            "train_policies": int(len(train)),
            "train_exposure": float(train["Exposure"].sum()),
            "train_claims": int(train["N"].sum()),
            "train_claim_policies": int((train["N"] > 0).sum()),
            "per_fold": _records(fold_rows),
        },
        **res,
        "final": {
            "power_cv": power_cv,
            **{k: v for k, v in (final or {}).items() if k not in ("fits", "solver_check")},
        },
        "convergence": convergence,
        "cap_diagnostics": cap_diag,
        "dispersion": dispersion_summary(fit_records),
        "solver_check": solver_check,
        "checks": pipeline_checks(solver_check, convergence, cap_diag),
        "groups": group_summary(train, cfg["models"]["group_summary"], cfg["split"]["cv_folds"]),
        "excess_loading_train": excess_loading(train, cfg["clean"]),
        "timing_seconds": timing,
    }
    return result


def _promote(staging: Path, dest: Path, names: list[str]) -> None:
    """Copy staged files to ``dest`` (each through a temporary name and an atomic replace)."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        tmp = dest / (name + ".tmp")
        shutil.copyfile(staging / name, tmp)
        os.replace(tmp, dest / name)


def _write_failure(processed: Path, name: str, payload: dict[str, Any]) -> Path:
    processed.mkdir(parents=True, exist_ok=True)
    path = processed / name
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def run_train(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load, cross-validate, refit. The model artifacts (final models, feature maps, relativity
    tables, cv_glm.json and the report) are written to a staging directory under paths.processed
    and promoted to paths.results / paths.reports only if every gate passes (solver check, cap
    reconciliation, convergence). On a failure the previously committed artifacts stay untouched
    and a failure diagnostic (with the full result, when there is one) is written under
    paths.processed."""
    t0 = time.perf_counter()
    outs = cfg["models"]["outputs"]
    processed = Path(cfg["paths"]["processed"])
    results, reports = Path(cfg["paths"]["results"]), Path(cfg["paths"]["reports"])
    staging = processed / outs["staging_dir"]
    failure_name = outs["failure_diagnostic"]
    shutil.rmtree(staging, ignore_errors=True)
    (processed / failure_name).unlink(missing_ok=True)
    try:
        train, claims = load_train_claims(cfg)
        load_s = time.perf_counter() - t0
        result = train_models(train, cfg, claims, staging)
    except Exception as exc:  # e.g. cap reconciliation: nothing was promoted
        path = _write_failure(
            processed, failure_name, {"error": f"{type(exc).__name__}: {exc}", "result": None}
        )
        raise RuntimeError(f"train failed ({exc}); diagnostic: {path}") from exc
    result["timing_seconds"]["load_and_clean"] = load_s
    result["timing_seconds"]["total"] += load_s
    clean_result = _clean_json(result)
    (staging / outs["cv_json"]).write_text(
        json.dumps(clean_result, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    report_text = render_report(clean_result)
    (staging / outs["report"]).write_text(report_text, encoding="utf-8")
    checks = clean_result["checks"]
    if not checks["passed"]:
        failed = sorted(k for k, v in checks.items() if k != "passed" and not v)
        path = _write_failure(
            processed,
            failure_name,
            {"error": f"gates failed: {failed}", "checks": checks, "result": clean_result},
        )
        raise RuntimeError(
            f"train gates failed {failed} (solver check violations: "
            f"{clean_result['solver_check']['failures'][:3]}); no artifact was promoted; "
            f"diagnostic: {path}"
        )
    model_files = [
        outs[k] for k in ("relativities_frequency", "relativities_severity", "relativities_tweedie")
    ] + [outs["feature_map"], outs["models"], outs["cv_json"]]
    _promote(staging, results, model_files)
    _promote(staging, reports, [outs["report"]])
    shutil.rmtree(staging, ignore_errors=True)
    return clean_result
