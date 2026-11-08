"""Milestone 3a: LightGBM frequency, severity and Tweedie candidates (grouped CV, TRAIN only).

Protocol: ``reports/evaluation_protocol.md`` sections 4-5. The test set is never loaded: the train
frame comes from ``train.load_train`` and every fold/fit function takes an already train-only frame
and raises on ``cv_fold < 0``. Every setting comes from ``models.gbm`` (declared before tuning).

Prediction contract. ``log(E)`` is the ``init_score`` of the Poisson GBM in the training and the
validation set; the booster is only asked for raw scores (``raw_score=True``, which never include
an init score) and the expected counts are ``exp(raw + log E)`` computed here; annual rates are
``exp(raw)``. Exposure is never a feature and never a sample weight of the Poisson GBM. Severity:
response ``S_cap / N``, weight ``N``, claims only. Tweedie: response ``S_cap / E``, weight ``E``,
power read from ``results/cv_glm.json`` (``final.power_cv.power``).

GBM CV scores are optimistic tuning scores: early stopping and grid scoring use the same registered
fold. The unconstrained twins and the constraint ablation are a CV-only diagnostic that never feeds
a selection.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import metrics, selection
from .gbm_report import render_report
from .train import (
    _clean_json,
    _write_failure,
    assert_train_only,
    load_train,
    outer_splits,
    summarise_folds,
)

COMPONENTS = ("frequency", "severity", "tweedie")
UNSEEN_POLICIES = ("missing", "error")
PP_FXS, PP_TW = "gbm_freq_x_sev", "gbm_tweedie"


# --------------------------------------------------------------------------- config


def validate_gbm(gcfg: dict[str, Any], features: dict[str, Any] | None = None) -> None:
    """Fail on an incomplete ``models.gbm`` block (nothing is defaulted in code)."""
    for key in (
        "seed",
        "features",
        "unseen_level",
        "deterministic",
        "base_params",
        "early_stopping",
        "components",
        "diagnostic",
        "checks",
        "outputs",
    ):
        if key not in gcfg:
            raise ValueError(f"models.gbm.{key} is required")
    if not isinstance(gcfg["seed"], int):
        raise ValueError("models.gbm.seed must be an integer")
    if gcfg["unseen_level"] not in UNSEEN_POLICIES:
        raise ValueError(f"models.gbm.unseen_level must be one of {UNSEEN_POLICIES}")
    feats = gcfg["features"]
    names = list(feats["numeric"]) + list(feats["categorical"])
    if len(set(names)) != len(names) or not names:
        raise ValueError("models.gbm.features: names must be unique and non-empty")
    if "Exposure" in names:
        raise ValueError("models.gbm.features: Exposure is never a feature")
    es = gcfg["early_stopping"]
    if es["round_rounding"] != "nearest":
        raise ValueError("models.gbm.early_stopping.round_rounding: only 'nearest' is supported")
    if es["at_cap"] != "record_use_cap":
        raise ValueError("models.gbm.early_stopping.at_cap: only 'record_use_cap' is supported")
    if es["round_cap"] < 1 or es["patience"] < 1:
        raise ValueError("models.gbm.early_stopping: round_cap and patience must be >= 1")
    if set(gcfg["components"]) != set(COMPONENTS):
        raise ValueError(f"models.gbm.components must declare exactly {COMPONENTS}")
    for comp, spec in gcfg["components"].items():
        for key in ("objective", "metric", "monotone", "params", "grid"):
            if key not in spec:
                raise ValueError(f"models.gbm.components.{comp}.{key} is required")
        if not spec["grid"] or any(not v for v in spec["grid"].values()):
            raise ValueError(f"models.gbm.components.{comp}.grid must list values per key")
        unknown = set(spec["monotone"]) - set(names)
        if unknown:
            raise ValueError(f"models.gbm.components.{comp}.monotone: unknown {sorted(unknown)}")
        if any(int(v) not in (-1, 1) for v in spec["monotone"].values()):
            raise ValueError(f"models.gbm.components.{comp}.monotone: values must be -1 or +1")
    if gcfg["components"]["severity"]["monotone"]:
        raise ValueError("models.gbm: the severity GBM is unconstrained (protocol section 5)")
    diag = gcfg["diagnostic"]
    for comp in diag["components"]:
        if comp not in ("frequency", "tweedie"):
            raise ValueError("models.gbm.diagnostic.components: frequency and tweedie only")
        if not gcfg["components"][comp]["monotone"]:
            raise ValueError(f"diagnostic component {comp!r} has no monotone constraint to remove")


def feature_names(gcfg: dict[str, Any]) -> list[str]:
    """Final feature order: numeric features first, then categorical, as declared."""
    return list(gcfg["features"]["numeric"]) + list(gcfg["features"]["categorical"])


def tweedie_power(cfg: dict[str, Any]) -> float:
    """Selected Tweedie power, read from the GLM CV result file (never retyped)."""
    path = Path(cfg["paths"]["results"]) / cfg["models"]["outputs"]["cv_json"]
    power = float(json.loads(path.read_text(encoding="utf-8"))["final"]["power_cv"]["power"])
    if not 1.0 < power < 2.0:
        raise ValueError(f"{path}: Tweedie power {power} outside (1, 2)")
    return power


def grid_points(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Cartesian product of the declared grid, keys in sorted order, values in declared order."""
    keys = sorted(spec["grid"])
    return [
        dict(zip(keys, vals, strict=True))
        for vals in itertools.product(*(spec["grid"][k] for k in keys))
    ]


def monotone_vector(names: list[str], monotone: dict[str, int]) -> list[int]:
    """LightGBM constraint vector built BY NAME for the final feature order."""
    unknown = set(monotone) - set(names)
    if unknown:
        raise ValueError(f"monotone constraints on unknown features {sorted(unknown)}")
    return [int(monotone.get(n, 0)) for n in names]


def build_params(
    gcfg: dict[str, Any],
    component: str,
    hp: dict[str, Any],
    power: float,
    constrained: bool,
    with_metric: bool,
) -> dict[str, Any]:
    spec = gcfg["components"][component]
    names = feature_names(gcfg)
    params: dict[str, Any] = {
        **gcfg["base_params"],
        **gcfg["deterministic"],
        "seed": gcfg["seed"],
        "objective": spec["objective"],
        **spec["params"],
        **hp,
    }
    if with_metric:
        params["metric"] = spec["metric"]
    if component == "tweedie":
        params["tweedie_variance_power"] = power
    if constrained and spec["monotone"]:
        params["monotone_constraints"] = monotone_vector(names, spec["monotone"])
    return params


# --------------------------------------------------------------------------- encoding


def fit_encoding(df: pd.DataFrame, gcfg: dict[str, Any]) -> dict[str, Any]:
    """Category levels (sorted) of every categorical feature, from the fitting part only."""
    levels = {c: sorted(df[c].astype(str).unique()) for c in gcfg["features"]["categorical"]}
    return {
        "numeric": list(gcfg["features"]["numeric"]),
        "categorical": list(gcfg["features"]["categorical"]),
        "levels": levels,
        "unseen_level": gcfg["unseen_level"],
    }


def encode(df: pd.DataFrame, enc: dict[str, Any]) -> np.ndarray:
    """Float matrix in the final feature order. Categoricals are codes of the fitted levels; a
    level unseen in the fitting part becomes NaN (LightGBM missing) or raises, as configured."""
    cols = []
    for c in enc["numeric"]:
        v = df[c].to_numpy(dtype=float)
        if not np.isfinite(v).all():
            raise ValueError(f"{c}: non-finite values")
        cols.append(v)
    for c in enc["categorical"]:
        codes = pd.Index(enc["levels"][c]).get_indexer(df[c].astype(str))
        if (codes < 0).any():
            if enc["unseen_level"] == "error":
                bad = sorted(set(df[c].astype(str)[codes < 0]))
                raise ValueError(f"{c}: levels unseen in the fitting data: {bad}")
        cols.append(np.where(codes < 0, np.nan, codes.astype(float)))
    return np.column_stack(cols)


# --------------------------------------------------------------------------- contract


def component_data(component: str, X, e, count, s, n):
    """(X, label, weight, init_score) of a component. Exposure is only the init score (Poisson),
    the weight (Tweedie) or absent."""
    if component == "frequency":
        return X, count.astype(float), None, np.log(e)
    if component == "severity":
        m = n > 0
        return X[m], s[m] / n[m], n[m].astype(float), None
    if component == "tweedie":
        return X, s / e, e.astype(float), None
    raise ValueError(f"unknown component {component!r}")


def raw_score(booster: lgb.Booster, X, num_iteration: int | None = None) -> np.ndarray:
    """Raw score WITHOUT any init score (``raw_score=True``)."""
    return booster.predict(X, raw_score=True, num_iteration=num_iteration)


def expected_counts(raw: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    """Expected counts exp(raw + log E); explicit, never via predict()."""
    return np.exp(raw + np.log(exposure))


def annual_rate(raw: np.ndarray) -> np.ndarray:
    return np.exp(raw)


def check_feature_order(booster: lgb.Booster, names: list[str]) -> None:
    if list(booster.feature_name()) != list(names):
        raise ValueError(f"feature order mismatch: {booster.feature_name()} != {names}")


def component_loss(component: str, fd: dict[str, Any], raw: np.ndarray, power: float) -> float:
    """Own objective deviance on the validation part (raw: scores of all validation rows)."""
    ev, nv, sv = fd["ev"], fd["nv"], fd["sv"]
    if component == "frequency":
        return metrics.poisson_deviance_per_exposure(fd["cv"], expected_counts(raw, ev), ev)
    if component == "severity":
        m = nv > 0
        return metrics.gamma_deviance(sv[m] / nv[m], annual_rate(raw[m]), nv[m]) / nv[m].sum()
    return metrics.tweedie_deviance(sv / ev, annual_rate(raw), power, ev) / ev.sum()


# --------------------------------------------------------------------------- fitting


def _dataset(X, label, weight, init, names, cat, params, reference=None):
    return lgb.Dataset(
        X,
        label=label,
        weight=weight,
        init_score=init,
        feature_name=names,
        categorical_feature=cat,
        reference=reference,
        params={k: params[k] for k in ("max_bin", "min_data_in_bin", "verbosity") if k in params},
        free_raw_data=False,
    )


def hit_round_cap(current: int, best: int, round_cap: int, patience: int) -> bool:
    """True only if all ``round_cap`` rounds ran without the patience rule having fired.

    Early stopping fires after exactly ``patience`` rounds past the best one, so a run that
    ends at ``best + patience`` stopped by patience (even when that equals the cap).
    """
    return bool(current >= round_cap and best + patience > current)


def _train_early(component, params, fd, names, cat, es, power):
    """Fit on the fitting part with early stopping on the validation part of the same fold."""
    Xf, y, w, i = component_data(component, fd["Xf"], fd["ef"], fd["cf"], fd["sf"], fd["nf"])
    Xv, yv, wv, iv = component_data(component, fd["Xv"], fd["ev"], fd["cv"], fd["sv"], fd["nv"])
    train_set = _dataset(Xf, y, w, i, names, cat, params)
    valid_set = _dataset(Xv, yv, wv, iv, names, cat, params, reference=train_set)
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=es["round_cap"],
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(
                es["patience"],
                first_metric_only=True,
                verbose=False,
                min_delta=es["min_improvement"],
            )
        ],
    )
    best = int(booster.best_iteration)
    cap_reached = hit_round_cap(booster.current_iteration(), best, es["round_cap"], es["patience"])
    return booster, best, cap_reached


def _tune_task(component, params, fd, names, cat, es, power):
    assert_train_only_fold(fd)
    t0 = time.perf_counter()
    booster, best, cap_reached = _train_early(component, params, fd, names, cat, es, power)
    check_feature_order(booster, names)
    raw = raw_score(booster, fd["Xv"], best)
    return {
        "best_iteration": best,
        "cap_reached": cap_reached,
        "loss": component_loss(component, fd, raw, power),
        "raw": raw,
        "seconds": time.perf_counter() - t0,
    }


def _fixed_task(component, params, fd, names, cat, rounds, power):
    """Fit with a fixed number of rounds on the fitting part; predict the validation part."""
    assert_train_only_fold(fd)
    t0 = time.perf_counter()
    Xf, y, w, i = component_data(component, fd["Xf"], fd["ef"], fd["cf"], fd["sf"], fd["nf"])
    booster = lgb.train(params, _dataset(Xf, y, w, i, names, cat, params), num_boost_round=rounds)
    raw = raw_score(booster, fd["Xv"], rounds)
    return {
        "loss": component_loss(component, fd, raw, power),
        "raw": raw,
        "seconds": time.perf_counter() - t0,
    }


def _final_task(component, params, X, e, count, s, n, names, cat, rounds):
    t0 = time.perf_counter()
    Xc, y, w, i = component_data(component, X, e, count, s, n)
    booster = lgb.train(params, _dataset(Xc, y, w, i, names, cat, params), num_boost_round=rounds)
    return {
        "model": booster.model_to_string(num_iteration=rounds),
        "seconds": time.perf_counter() - t0,
    }


def assert_train_only_fold(fd: dict[str, Any]) -> None:
    if fd["k"] < 0:
        raise ValueError("train-only guard: fold index < 0 (test set) is not allowed")


def make_fold(
    fit: pd.DataFrame, valid: pd.DataFrame, gcfg: dict[str, Any], k: int
) -> dict[str, Any]:
    """Numpy arrays of one registered fold; encoding fitted on the fitting part only."""
    assert_train_only(fit)
    assert_train_only(valid)
    enc = fit_encoding(fit, gcfg)
    arr = lambda d, c: d[c].to_numpy()  # noqa: E731
    return {
        "k": k,
        "enc": enc,
        "Xf": encode(fit, enc),
        "Xv": encode(valid, enc),
        "ef": arr(fit, "Exposure"),
        "ev": arr(valid, "Exposure"),
        "cf": arr(fit, "freq_count"),
        "cv": arr(valid, "freq_count"),
        "sf": arr(fit, "S_cap"),
        "sv": arr(valid, "S_cap"),
        "nf": arr(fit, "N"),
        "nv": arr(valid, "N"),
    }


def best_rounds(best_iterations: list[int], rounding: str) -> int:
    """Mean of the per-fold best iterations rounded to the nearest integer (half up)."""
    if rounding != "nearest":
        raise ValueError(f"unsupported rounding {rounding!r}")
    return max(1, int(np.floor(float(np.mean(best_iterations)) + 0.5)))


# --------------------------------------------------------------------------- tuning


def _variants(gcfg: dict[str, Any]) -> list[tuple[str, str, bool]]:
    """(variant, component, constrained): the main procedures and the unconstrained twins."""
    out = [("main", c, True) for c in COMPONENTS]
    out += (
        [("twin", c, False) for c in gcfg["diagnostic"]["components"]]
        if gcfg["diagnostic"]["enabled"]
        else []
    )
    return out


def tune(folds: list[dict[str, Any]], gcfg: dict[str, Any], power: float, n_jobs: int):
    """Grid x fold tuning of every main procedure and twin with identical grids/folds/seed/budget.

    Returns {(variant, component): {"points": [...], "chosen": index, "rounds": int}}."""
    names, cat = feature_names(gcfg), list(gcfg["features"]["categorical"])
    es = gcfg["early_stopping"]
    jobs = []
    for variant, comp, constrained in _variants(gcfg):
        for gi, hp in enumerate(grid_points(gcfg["components"][comp])):
            params = build_params(gcfg, comp, hp, power, constrained, with_metric=True)
            for fd in folds:
                jobs.append((variant, comp, gi, hp, params, fd))
    done = Parallel(n_jobs=n_jobs)(
        delayed(_tune_task)(comp, params, fd, names, cat, es, power)
        for _, comp, _, _, params, fd in jobs
    )
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for (variant, comp, gi, hp, _, fd), r in zip(jobs, done, strict=True):
        entry = out.setdefault((variant, comp), {"points": {}})
        pt = entry["points"].setdefault(gi, {"hp": hp, "folds": {}})
        pt["folds"][fd["k"]] = r
    for entry in out.values():
        pts = []
        for gi in sorted(entry["points"]):
            pt = entry["points"][gi]
            ks = sorted(pt["folds"])
            losses = [pt["folds"][k]["loss"] for k in ks]
            bests = [pt["folds"][k]["best_iteration"] for k in ks]
            pts.append(
                {
                    "hp": pt["hp"],
                    "fold_loss": losses,
                    "mean_loss": float(np.mean(losses)),
                    "fold_best_iteration": bests,
                    "fold_cap_reached": [pt["folds"][k]["cap_reached"] for k in ks],
                    "fold_seconds": [pt["folds"][k]["seconds"] for k in ks],
                    "rounds": best_rounds(bests, es["round_rounding"]),
                    "raw": [pt["folds"][k]["raw"] for k in ks],
                }
            )
        means = [p["mean_loss"] for p in pts]
        entry["points"] = pts
        entry["chosen"] = int(np.argmin(means))  # ties: lowest grid index
        entry["rounds"] = pts[entry["chosen"]]["rounds"]
    return out


# --------------------------------------------------------------------------- OOF metrics


def _fold_metrics(fd, rates: dict[str, np.ndarray], power: float, ref: float) -> dict[str, Any]:
    """Metrics of one validation fold. ``rates``: annual rates of frequency / severity / tweedie
    (only those that are given); pure premium of freq_x_sev = E * frequency rate * severity."""
    ev, cv, sv, nv = fd["ev"], fd["cv"], fd["sv"], fd["nv"]
    out: dict[str, Any] = {}
    if "frequency" in rates:
        mu = rates["frequency"] * ev
        out["frequency"] = {
            "poisson_deviance_per_exposure": metrics.poisson_deviance_per_exposure(cv, mu, ev),
            **metrics.gini(cv, rates["frequency"], ev),
            "totals": metrics.calibration_totals(cv, mu),
        }
    if "severity" in rates:
        m = nv > 0
        sev = rates["severity"][m]
        out["severity"] = {
            "gamma_deviance_per_claim": metrics.gamma_deviance(sv[m] / nv[m], sev, nv[m])
            / nv[m].sum(),
            "totals": metrics.calibration_totals(sv[m], sev * nv[m]),
        }
    pps = {}
    if "frequency" in rates and "severity" in rates:
        pps[PP_FXS] = rates["frequency"] * rates["severity"]
    if "tweedie" in rates:
        pps[PP_TW] = rates["tweedie"]
    out["pure_premium"] = {}
    for name, rate in pps.items():
        y = sv / ev
        out["pure_premium"][name] = {
            "poisson_deviance_per_exposure": metrics.tweedie_deviance(y, rate, 1.0, ev) / ev.sum(),
            "tweedie_power_deviance_per_exposure": metrics.tweedie_deviance(y, rate, power, ev)
            / ev.sum(),
            "primary_loss": metrics.tweedie_deviance(y, rate, ref, ev) / ev.sum(),
            **metrics.gini(sv, rate, ev),
            "totals": metrics.calibration_totals(sv, rate * ev),
        }
    return out


def _rates(chosen_raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {c: annual_rate(r) for c, r in chosen_raw.items()}


def _chosen_raw(tuned, variant: str, comp: str, k: int) -> np.ndarray:
    entry = tuned[(variant, comp)]
    return entry["points"][entry["chosen"]]["raw"][k]


def _primary(per_fold: list[dict[str, Any]], name: str) -> list[float]:
    return [f["pure_premium"][name]["primary_loss"] for f in per_fold]


def _stats(d: list[float]) -> dict[str, Any]:
    a = np.asarray(d, dtype=float)
    sd = float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")
    return {"d_k": [float(x) for x in a], "mean": float(a.mean()), "sd": sd}


# --------------------------------------------------------------------------- diagnostic


def diagnostic(folds, gcfg, tuned, power, ref, n_jobs, main_folds) -> dict[str, Any]:
    """CV-only monotonicity diagnostic; never feeds a selection (labelled)."""
    names, cat = feature_names(gcfg), list(gcfg["features"]["categorical"])
    k_folds = len(folds)
    block: dict[str, Any] = {
        "label": "CV-only diagnostic of the frozen monotone constraint; never a candidate and "
        "never an input of any selection",
        "separately_tuned": {},
        "ablation": {},
    }
    comps = gcfg["diagnostic"]["components"]
    for comp in comps:
        main, twin = tuned[("main", comp)], tuned[("twin", comp)]
        mp, tp = main["points"][main["chosen"]], twin["points"][twin["chosen"]]
        block["separately_tuned"][comp] = {
            "constrained": {"hp": mp["hp"], "rounds": main["rounds"], "fold_loss": mp["fold_loss"]},
            "unconstrained": {
                "hp": tp["hp"],
                "rounds": twin["rounds"],
                "fold_loss": tp["fold_loss"],
            },
            "component_loss_diff_unconstrained_minus_constrained": _stats(
                [u - c for u, c in zip(tp["fold_loss"], mp["fold_loss"], strict=True)]
            ),
        }
    sev_raw = [_chosen_raw(tuned, "main", "severity", k) for k in range(k_folds)]
    twin_folds = []
    for k, fd in enumerate(folds):
        raw: dict[str, np.ndarray] = {"severity": sev_raw[k]}
        for comp in ("frequency", "tweedie"):
            variant = "twin" if comp in comps else "main"
            raw[comp] = _chosen_raw(tuned, variant, comp, k)
        twin_folds.append(_fold_metrics(fd, _rates(raw), power, ref))
    block["separately_tuned"]["primary_loss"] = {
        "constrained_freq_x_sev": _primary(main_folds, PP_FXS),
        "constrained_tweedie": _primary(main_folds, PP_TW),
        "unconstrained_freq_x_sev": _primary(twin_folds, PP_FXS),
        "unconstrained_tweedie": _primary(twin_folds, PP_TW),
        "diff_freq_x_sev_unconstrained_minus_constrained": _stats(
            [
                u - c
                for u, c in zip(
                    _primary(twin_folds, PP_FXS), _primary(main_folds, PP_FXS), strict=True
                )
            ]
        ),
        "diff_tweedie_unconstrained_minus_constrained": _stats(
            [
                u - c
                for u, c in zip(
                    _primary(twin_folds, PP_TW), _primary(main_folds, PP_TW), strict=True
                )
            ]
        ),
    }
    if not gcfg["diagnostic"]["ablation"]:
        return block
    # controlled ablation: chosen hyperparameters and rounds of the constrained procedure,
    # constraint toggled, same folds and seed
    jobs = []
    for comp in comps:
        main = tuned[("main", comp)]
        hp, rounds = main["points"][main["chosen"]]["hp"], main["rounds"]
        for constrained in (True, False):
            params = build_params(gcfg, comp, hp, power, constrained, with_metric=False)
            for fd in folds:
                jobs.append((comp, constrained, params, rounds, fd))
    done = Parallel(n_jobs=n_jobs)(
        delayed(_fixed_task)(comp, params, fd, names, cat, rounds, power)
        for comp, _, params, rounds, fd in jobs
    )
    res: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for (comp, constrained, _, _, _), r in zip(jobs, done, strict=True):
        res.setdefault((comp, constrained), []).append(r)
    for comp in comps:
        row: dict[str, Any] = {"rounds": tuned[("main", comp)]["rounds"]}
        pp = {}
        for constrained in (True, False):
            tag = "constrained" if constrained else "unconstrained"
            runs = res[(comp, constrained)]
            row[f"{tag}_fold_loss"] = [r["loss"] for r in runs]
            pp[tag] = []
            for k, fd in enumerate(folds):
                raw = {"severity": sev_raw[k], comp: runs[k]["raw"]}
                pp[tag].append(_fold_metrics(fd, _rates(raw), power, ref))
        name = PP_FXS if comp == "frequency" else PP_TW
        row["primary_loss_constrained"] = _primary(pp["constrained"], name)
        row["primary_loss_unconstrained"] = _primary(pp["unconstrained"], name)
        row["component_loss_diff_unconstrained_minus_constrained"] = _stats(
            [
                u - c
                for u, c in zip(
                    row["unconstrained_fold_loss"], row["constrained_fold_loss"], strict=True
                )
            ]
        )
        row["primary_loss_diff_unconstrained_minus_constrained"] = _stats(
            [
                u - c
                for u, c in zip(
                    row["primary_loss_unconstrained"], row["primary_loss_constrained"], strict=True
                )
            ]
        )
        block["ablation"][comp] = row
    return block


# --------------------------------------------------------------------------- checks


def monotone_sweep(
    booster: lgb.Booster, X: np.ndarray, names: list[str], feature: str, values, sign: int = 1
) -> float:
    """Smallest signed step of the raw score along ``values`` (ascending) with all other features
    fixed, over every row of ``X``. >= 0 for a model constrained to be non-decreasing (sign=1)."""
    j = names.index(feature)
    raws = []
    for v in values:
        Xv = X.copy()
        Xv[:, j] = v
        raws.append(raw_score(booster, Xv))
    steps = np.diff(np.vstack(raws), axis=0) * sign
    return float(steps.min())


def final_checks(boosters, files, folds_complete, train, gcfg) -> dict[str, Any]:
    """Gates of the final models: reload equivalence, monotone sweep, finite positive rates."""
    chk = gcfg["checks"]
    names = feature_names(gcfg)
    enc = fit_encoding(train, gcfg)
    rng = np.random.default_rng(gcfg["seed"])
    n = min(chk["reload_rows"], len(train))
    sample = train.iloc[np.sort(rng.choice(len(train), size=n, replace=False))]
    X = encode(sample, enc)
    out: dict[str, Any] = {"reload_equivalence": {}, "monotone_sweep": {}, "finite_positive": {}}
    ok_reload = ok_mono = ok_pos = True
    for comp, b in boosters.items():
        check_feature_order(b, names)
        reloaded = lgb.Booster(model_file=str(files[comp]))
        diff = float(np.max(np.abs(raw_score(b, X) - raw_score(reloaded, X))))
        out["reload_equivalence"][comp] = diff
        ok_reload &= diff <= chk["reload_tolerance"]
        rate = annual_rate(raw_score(b, X))
        pos = bool(np.isfinite(rate).all() and (rate > 0).all())
        out["finite_positive"][comp] = pos
        ok_pos &= pos
        for feat, sign in gcfg["components"][comp]["monotone"].items():
            sub = X[: chk["sweep_rows"]]
            vals = np.unique(
                np.quantile(
                    sample[feat].to_numpy(dtype=float), np.linspace(0, 1, chk["sweep_points"])
                )
            )
            mn = monotone_sweep(b, sub, names, feat, vals, int(sign))
            out["monotone_sweep"][f"{comp}:{feat}"] = mn
            ok_mono &= mn >= -chk["monotone_tolerance"]
    out["folds_complete"] = bool(folds_complete)
    out["passed"] = bool(ok_reload and ok_mono and ok_pos and folds_complete)
    return out


# --------------------------------------------------------------------------- run


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_summary(cfg: dict[str, Any], power: float) -> dict[str, Any]:
    g = cfg["models"]["gbm"]
    return {
        "gbm": json.loads(json.dumps(g)),
        "tweedie_power": power,
        "reference_power": cfg["models"]["tweedie"]["reference_power"],
        "cv_folds": cfg["split"]["cv_folds"],
        "n_jobs": cfg["models"]["parallel"]["n_jobs"],
        "scores_label": "GBM CV scores are optimistic tuning scores: early stopping and grid "
        "scoring use the same registered fold (protocol section 5)",
    }


def gbm_models(train: pd.DataFrame, cfg: dict[str, Any], tuned, power: float, out_dir: Path):
    """Refit the three components on full train with the chosen settings; write the model files."""
    assert_train_only(train)
    gcfg = cfg["models"]["gbm"]
    names, cat = feature_names(gcfg), list(gcfg["features"]["categorical"])
    enc = fit_encoding(train, gcfg)
    X = encode(train, enc)
    arrays = (
        X,
        train["Exposure"].to_numpy(),
        train["freq_count"].to_numpy(),
        train["S_cap"].to_numpy(),
        train["N"].to_numpy(),
    )
    jobs = []
    for comp in COMPONENTS:
        entry = tuned[("main", comp)]
        hp = entry["points"][entry["chosen"]]["hp"]
        params = build_params(gcfg, comp, hp, power, True, with_metric=False)
        jobs.append((comp, params, entry["rounds"]))
    done = Parallel(n_jobs=cfg["models"]["parallel"]["n_jobs"])(
        delayed(_final_task)(comp, params, *arrays, names, cat, rounds)
        for comp, params, rounds in jobs
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    boosters, files, meta = {}, {}, {}
    for (comp, params, rounds), r in zip(jobs, done, strict=True):
        path = out_dir / f"{comp}.txt"
        path.write_text(r["model"], encoding="utf-8")
        boosters[comp] = lgb.Booster(model_str=r["model"])
        files[comp] = path
        meta[comp] = {
            "file": f"{comp}.txt",
            "sha256": _sha256(path),
            "params": params,
            "rounds": rounds,
            "feature_names": names,
            "monotone_constraints": params.get("monotone_constraints"),
            "fit_seconds": r["seconds"],
        }
    return boosters, files, meta, enc


def run_gbm_cv(train: pd.DataFrame, cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Tune, score out of fold, select, diagnose and refit (all on the train-only frame)."""
    assert_train_only(train)
    gcfg = cfg["models"]["gbm"]
    validate_gbm(gcfg)
    power, ref = tweedie_power(cfg), cfg["models"]["tweedie"]["reference_power"]
    grouped = cfg["split"]["primary"] == "grouped"
    k_folds, n_jobs = cfg["split"]["cv_folds"], cfg["models"]["parallel"]["n_jobs"]
    timing: dict[str, float] = {}
    t0 = time.perf_counter()
    folds = [
        make_fold(fit, valid, gcfg, k) for k, fit, valid in outer_splits(train, k_folds, grouped)
    ]
    timing["prepare_folds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    tuned = tune(folds, gcfg, power, n_jobs)
    timing["tuning"] = time.perf_counter() - t0
    main_folds = []
    for k, fd in enumerate(folds):
        raw = {c: _chosen_raw(tuned, "main", c, k) for c in COMPONENTS}
        main_folds.append(_fold_metrics(fd, _rates(raw), power, ref))
    trace = selection.freq_x_sev_vs_tweedie(
        _primary(main_folds, PP_FXS), _primary(main_folds, PP_TW), names=(PP_FXS, PP_TW)
    )
    t0 = time.perf_counter()
    diag = (
        diagnostic(folds, gcfg, tuned, power, ref, n_jobs, main_folds)
        if gcfg["diagnostic"]["enabled"]
        else None
    )
    timing["diagnostic_ablation"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    boosters, files, meta, enc = gbm_models(train, cfg, tuned, power, out_dir / "models")
    timing["final_refits"] = time.perf_counter() - t0
    checks = final_checks(boosters, files, len(folds) == k_folds, train, gcfg)
    components = {}
    for (variant, comp), entry in tuned.items():
        if variant != "main":
            continue
        components[comp] = _component_block(entry)
    pooled = {
        "frequency_totals": _pooled_totals(folds, tuned, "frequency"),
    }
    result = {
        "config": config_summary(cfg, power),
        "components": components,
        "oof": {
            "per_fold": main_folds,
            "summary": summarise_folds(main_folds),
            "pooled": pooled,
        },
        "selection": {"gbm_pure_premium": trace, "selected": trace["selected"]},
        "diagnostic": diag,
        "final_models": {"encoding": enc, "models": meta},
        "checks": checks,
        "timing_seconds": timing,
    }
    if diag is not None:
        result["twin_components"] = {
            comp: _component_block(entry) for (v, comp), entry in tuned.items() if v == "twin"
        }
    timing["total"] = float(sum(timing.values()))
    return result


def _pooled_totals(folds, tuned, comp: str) -> dict[str, Any]:
    ev = np.concatenate([fd["ev"] for fd in folds])
    cnt = np.concatenate([fd["cv"] for fd in folds])
    raw = np.concatenate([_chosen_raw(tuned, "main", comp, k) for k in range(len(folds))])
    return metrics.calibration_totals(cnt, expected_counts(raw, ev))


def _component_block(entry: dict[str, Any]) -> dict[str, Any]:
    pts = [{k: v for k, v in p.items() if k != "raw"} for p in entry["points"]]
    chosen = entry["points"][entry["chosen"]]
    return {
        "grid": pts,
        "chosen_index": entry["chosen"],
        "chosen_hp": chosen["hp"],
        "mean_cv_loss_optimistic": chosen["mean_loss"],
        "rounds": entry["rounds"],
        "cap_reached_any_fold": bool(any(chosen["fold_cap_reached"])),
        "cap_reached_any_point": bool(any(any(p["fold_cap_reached"]) for p in pts)),
    }


def load_frozen_booster(path: Path, meta: dict[str, Any]) -> lgb.Booster:
    """Prediction adapter of a stored final model (no fitting): the file must match the recorded
    sha256 and feature order. Use with :func:`raw_score` and :func:`annual_rate` (or
    :func:`expected_counts`); the booster never adds an init score."""
    if not path.is_file() or _sha256(path) != meta["sha256"]:
        raise ValueError(f"model file {path} does not match its recorded sha256")
    booster = lgb.Booster(model_file=str(path))
    check_feature_order(booster, list(meta["feature_names"]))
    return booster


def predict_annual_rate(
    booster: lgb.Booster, df: pd.DataFrame, enc: dict[str, Any], num_threads: int | None = None
) -> np.ndarray:
    """Annual rate exp(raw score) of a frozen booster for the rows of ``df`` (stored encoding);
    ``num_threads`` is passed to LightGBM's predict when given."""
    X = encode(df, enc)
    if num_threads is None:
        return annual_rate(raw_score(booster, X))
    return annual_rate(booster.predict(X, raw_score=True, num_threads=int(num_threads)))


def verify_staged_hashes(model_dir: Path, models: dict[str, Any]) -> None:
    """Every model file named in the results metadata must exist with the recorded sha256."""
    for comp, m in models.items():
        path = model_dir / m["file"]
        if not path.is_file() or _sha256(path) != m["sha256"]:
            raise ValueError(f"staged model file {m['file']} ({comp}) does not match its sha256")


def promote_all(plan: list[tuple[Path, Path, list[str]]], backup: Path) -> None:
    """Promote (staging, dest, names) groups together: all copies are made under temporary
    names first (a failure there changes nothing), then replaced one by one; if a replace
    fails the already replaced files are restored from ``backup`` and the error is re-raised."""
    tmps: list[tuple[Path, Path]] = []
    try:
        for src_dir, dest, names in plan:
            dest.mkdir(parents=True, exist_ok=True)
            for name in names:
                tmp = dest / (name + ".tmp")
                shutil.copyfile(src_dir / name, tmp)
                tmps.append((tmp, dest / name))
    except Exception:
        for tmp, _ in tmps:
            tmp.unlink(missing_ok=True)
        raise
    backup.mkdir(parents=True, exist_ok=True)
    done: list[tuple[Path, Path | None]] = []
    try:
        for i, (tmp, final) in enumerate(tmps):
            saved = None
            if final.exists():
                saved = backup / f"{i}_{final.name}"
                shutil.copyfile(final, saved)
            os.replace(tmp, final)
            done.append((final, saved))
    except Exception:
        for final, saved in reversed(done):
            if saved is None:
                final.unlink(missing_ok=True)
            else:
                os.replace(shutil.copyfile(saved, final.with_name(final.name + ".tmp")), final)
        for tmp, _ in tmps:
            tmp.unlink(missing_ok=True)
        raise


def run_gbm(cfg: dict[str, Any]) -> dict[str, Any]:
    """``fremtpl2 gbm``: results are staged and promoted only if every check passes."""
    t0 = time.perf_counter()
    gcfg = cfg["models"]["gbm"]
    outs = gcfg["outputs"]
    processed = Path(cfg["paths"]["processed"])
    results, reports = Path(cfg["paths"]["results"]), Path(cfg["paths"]["reports"])
    staging = processed / outs["staging_dir"]
    shutil.rmtree(staging, ignore_errors=True)
    (processed / outs["failure_diagnostic"]).unlink(missing_ok=True)
    try:
        train = load_train(cfg)
        load_s = time.perf_counter() - t0
        result = run_gbm_cv(train, cfg, staging)
    except Exception as exc:
        path = _write_failure(
            processed, outs["failure_diagnostic"], {"error": f"{type(exc).__name__}: {exc}"}
        )
        raise RuntimeError(f"gbm failed ({exc}); diagnostic: {path}") from exc
    result["timing_seconds"]["load_and_clean"] = load_s
    result["timing_seconds"]["total"] += load_s
    clean_result = _clean_json(result)
    meta = {
        "tweedie_power": clean_result["config"]["tweedie_power"],
        "selected_pure_premium_candidate": clean_result["selection"]["selected"],
        "encoding": clean_result["final_models"]["encoding"],
        "models": clean_result["final_models"]["models"],
    }
    (staging / outs["cv_json"]).write_text(
        json.dumps(clean_result, indent=2) + "\n", encoding="utf-8"
    )
    (staging / outs["models_json"]).write_text(
        json.dumps(_clean_json(meta), indent=2) + "\n", encoding="utf-8"
    )
    (staging / outs["report"]).write_text(render_report(clean_result), encoding="utf-8")
    if not clean_result["checks"]["passed"]:
        path = _write_failure(
            processed,
            outs["failure_diagnostic"],
            {"error": "checks failed", "checks": clean_result["checks"]},
        )
        raise RuntimeError(f"gbm checks failed; nothing promoted; diagnostic: {path}")
    model_dir = processed / outs["model_dir"]
    try:
        verify_staged_hashes(staging / "models", clean_result["final_models"]["models"])
        promote_all(
            [
                (
                    staging / "models",
                    model_dir,
                    sorted(p.name for p in (staging / "models").iterdir()),
                ),
                (staging, results, [outs["cv_json"], outs["models_json"]]),
                (staging, reports, [outs["report"]]),
            ],
            staging / "_backup",
        )
    except Exception as exc:
        path = _write_failure(
            processed, outs["failure_diagnostic"], {"error": f"promotion failed: {exc}"}
        )
        raise RuntimeError(f"gbm promotion failed ({exc}); diagnostic: {path}") from exc
    shutil.rmtree(staging, ignore_errors=True)
    return clean_result
