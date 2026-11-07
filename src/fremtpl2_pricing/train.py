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
PP_MODELS = ("null", "bonus_malus_only", "freq_x_sev", "tweedie", "tweedie_ref")


# --------------------------------------------------------------------------- data


def assert_train_only(df: pd.DataFrame) -> None:
    """Raise if any row is not a registered training row (test rows have cv_fold -1)."""
    if "cv_fold" in df and (df["cv_fold"] < 0).any():
        raise ValueError("train-only guard: rows with cv_fold < 0 (test set) are not allowed")
    if "primary_set" in df and (df["primary_set"] != "train").any():
        raise ValueError("train-only guard: rows with primary_set != 'train' are not allowed")


def load_train(cfg: dict[str, Any]) -> pd.DataFrame:
    """Raw -> clean (primary rules, full data) -> merge the frozen split -> train rows only."""
    freq, sev = load_raw(cfg)
    res = clean(freq, sev, cfg["clean"])
    split = pd.read_parquet(Path(cfg["paths"]["processed"]) / "split.parquet")
    return train_frame(res.freq, split, cfg["split"]["cv_folds"])


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


def _fit_record(model: glm.GLMModel, name: str, power: float | None = None) -> dict[str, Any]:
    return {"model": name, "power": power, "converged": model.converged, "n_iter": model.n_iter}


def _score_power(model, Xf, fit, Xv, valid, power, mcfg) -> dict[str, float]:
    """Held-out scores of one fitted Tweedie power.

    The dispersion is the maximum-likelihood value on the FITTING part (its own means and prior
    weights = exposure); the held-out log-likelihood is the sum of log densities of
    S_cap / Exposure with prior weight = exposure at those means and that dispersion."""
    e_fit = fit["Exposure"].to_numpy()
    phi = metrics.tweedie_mle_dispersion(
        _pp(fit),
        model.predict_rate(Xf),
        power,
        weight=e_fit,
        log_range=mcfg["tweedie"]["dispersion_log_range"],
    )
    rate = model.predict_rate(Xv)
    ll = metrics.tweedie_loglik(_pp(valid), rate, phi, power, valid["Exposure"].to_numpy())
    return {"loglik": ll, "phi": phi, "poisson_loss": _poisson_pp_loss(valid, rate)}


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
        fits.append(_fit_record(model, "tweedie", p))
        scores[p] = _score_power(model, Xf, fit, Xv, valid, p, mcfg)
    return {
        "scores": scores,
        "exposure": float(valid["Exposure"].sum()),
        "policies": int(len(valid)),
        "fits": fits,
    }


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
    fmap_s = features.fit(
        claims,
        fcfg,
        weight_column="N",
        min_weight=mcfg["severity_features"]["min_claims"],
    )
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
    scores = {}
    for p in sorted(set(grid) | {ref}):
        model = glm.fit_tweedie(Xf[cols], _pp(fit), e_fit, p, solver)
        fits.append(_fit_record(model, "tweedie", p))
        scores[p] = _score_power(model, Xf[cols], fit, Xv[cols], valid, p, mcfg)
        pred[f"pp_tweedie_{p}"] = model.predict_rate(Xv[cols]) * e_val
    pred["pp_tweedie_ref"] = pred[f"pp_tweedie_{ref}"]
    return {
        "pred": pred,
        "scores": scores,
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


def scalar_metrics(d: pd.DataFrame, ref_power: float) -> dict[str, Any]:
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
    for m in PP_MODELS:
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
            m: _records(metrics.lift_table(s, oof[f"pp_{m}"], e, nb)) for m in PP_MODELS
        },
    }
    double = _records(metrics.double_lift_table(s, oof["pp_tweedie"], oof["pp_freq_x_sev"], e, nb))
    calib: dict[str, Any] = {}
    for name, spec in ev["calibration_factors"].items():
        lv, order = factor_levels(train.loc[oof.index], name, spec, fcfg)
        entry = {}
        for target, obs, models in (
            ("frequency", y, {m: oof[f"freq_{m}"] for m in FREQ_MODELS}),
            ("pure_premium", s, {m: oof[f"pp_{m}"] for m in PP_MODELS}),
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
    """Fail fast, before any model is fitted: every feature map used by a CV job must cover the
    levels of the part it is applied to.

    For each outer fold the policy map and the claims-only severity map (weight N) are fitted on
    the fitting part and applied to the validation part; for each inner split the policy map is
    applied to the inner validation part (the inner loop fits no severity model). Levels that
    are not grouped (``OTHER``) and are absent from a fitting part raise in ``features.transform``;
    all offending splits are listed in one error instead of a late crash.
    """
    mcfg, fcfg = cfg["models"], cfg["features"]
    grouped = cfg["split"]["primary"] == "grouped"
    problems: list[str] = []

    def check(label, fit, valid, severity):
        try:
            if severity:
                claims = fit[fit["N"] > 0]
                fmap = features.fit(
                    claims,
                    fcfg,
                    weight_column="N",
                    min_weight=mcfg["severity_features"]["min_claims"],
                )
            else:
                fmap = features.fit(fit, fcfg)
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
            "feature levels of a validation part are not covered by its fitting part "
            "(raise the rare-level grouping or lower the thresholds):\n  " + "\n  ".join(problems)
        )


def _power_diagnostics(grid, loglik: dict[float, float], cfg_t) -> dict[str, Any]:
    """Selected power plus boundary / flatness diagnostic of a held-out log-likelihood curve."""
    ll = np.array([loglik[p] for p in grid])
    best = int(np.argmax(ll))
    spread = float(ll.max() - ll.min())
    return {
        "power": grid[best],
        "at_grid_edge": best in (0, len(grid) - 1),
        "loglik_spread_nats": spread,
        "best_minus_next_nats": float(ll[best] - np.sort(ll)[-2]) if len(ll) > 1 else None,
        "weakly_identified": spread < cfg_t["weak_identification_nats"],
    }


def cross_validate(train: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Grouped CV on a train-only frame. Returns the CV result dict and the final-power scores."""
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
        diag = _power_diagnostics(grid, ll, mcfg["tweedie"])
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
        fold_metrics.append({"fold": k, **scalar_metrics(frame, ref)})
    oof = pd.concat(frames)
    per_fold = [{k_: v for k_, v in m.items() if k_ != "fold"} for m in fold_metrics]
    # final power: plain CV over the grid on the registered folds (dispersion per outer fit part)
    n_all = sum(outer[k]["policies"] for k in range(k_folds))
    tot = sum(outer[k]["exposure"] for k in range(k_folds))
    cv_ll = {p: sum(outer[k]["scores"][p]["loglik"] for k in range(k_folds)) for p in grid}
    final_diag = _power_diagnostics(grid, cv_ll, mcfg["tweedie"])
    result = {
        "selected_powers": selected,
        "folds": [{"fold": k, **per_fold[k]} for k in range(k_folds)],
        "fold_summary": summarise_folds(per_fold),
        "pooled": scalar_metrics(oof, ref),
        **tables(oof, train, mcfg, fcfg),
    }
    loaded = loaded_premium(oof, cfg["clean"])
    timing["cv_metrics"] = time.perf_counter() - t1
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
    fmap_s = features.fit(
        claims, fcfg, weight_column="N", min_weight=mcfg["severity_features"]["min_claims"]
    )
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


def cap_point_mass(train: pd.DataFrame, cap: float) -> dict[str, Any]:
    """Policies whose S_cap sits exactly at a positive multiple of the per-claim cap.

    The Tweedie likelihood treats S_cap / Exposure as continuous; these policies carry a point
    mass the density does not model (all their claims at least reach the cap)."""
    s = train["S_cap"].to_numpy()
    at_cap = (s > 0) & (np.mod(s, cap) == 0)
    n_pos = int((s > 0).sum())
    return {
        "cap": float(cap),
        "policies_with_claim_cost": n_pos,
        "policies_at_cap_multiple": int(at_cap.sum()),
        "share_of_claim_policies": float(at_cap.sum() / n_pos) if n_pos else None,
        "policies_with_a_capped_claim": int((train["excess"] > 0).sum()),
    }


def train_models(train: pd.DataFrame, cfg: dict[str, Any], out_dir: Path | None = None):
    """Whole milestone on a train-only frame; writes files when ``out_dir`` is given."""
    t0 = time.perf_counter()
    res = cross_validate(train, cfg)
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
    timing["total"] = time.perf_counter() - t0
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
        "final": {"power_cv": power_cv, **{k: v for k, v in (final or {}).items() if k != "fits"}},
        "convergence": convergence_summary(fit_records, cfg["models"]["glm"]["solver"]),
        "cap_point_mass": cap_point_mass(train, cfg["clean"]["large_loss"]["cap"]),
        "excess_loading_train": excess_loading(train, cfg["clean"]),
        "timing_seconds": timing,
    }
    return result


def run_train(cfg: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    train = load_train(cfg)
    load_s = time.perf_counter() - t0
    results, reports = Path(cfg["paths"]["results"]), Path(cfg["paths"]["reports"])
    result = train_models(train, cfg, results)
    result["timing_seconds"]["load_and_clean"] = load_s
    result["timing_seconds"]["total"] += load_s
    outs = cfg["models"]["outputs"]
    clean_result = _clean_json(result)
    (results / outs["cv_json"]).write_text(
        json.dumps(clean_result, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    reports.mkdir(parents=True, exist_ok=True)
    (reports / outs["report"]).write_text(render_report(clean_result), encoding="utf-8")
    return clean_result
