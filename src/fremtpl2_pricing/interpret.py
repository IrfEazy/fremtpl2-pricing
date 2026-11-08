"""Train-only interpretation of the frozen candidates (``fremtpl2 interpret``, before the freeze).

Reads the committed training artifacts through the same loaders as the freeze and evaluate stages
(``freeze.read_inputs`` / ``recompute_selections`` / ``build_recipes`` / ``load_artifacts``; the
hash-checked LightGBM files) and never fits anything. It uses TRAIN rows only: a fixed sample of the
train set (config ``interpret.sample``: size, seed; the ids are recorded), with a guard that every
row has ``cv_fold >= 0``. Outputs: ``results/interpretation.json``,
``reports/interpretation.md`` and figures under ``reports/figures/``; every number of the report
comes from the JSON.

Contents:

- TreeSHAP contributions of each GBM component (LightGBM ``pred_contrib``), checked per row to sum
  (with the base value) to the raw score; the exposure offset of the frequency model is not part of
  the contributions (the contributions explain the annual log rate);
- mean contributions by factor level / bin next to the GLM log relativities (the group mean of the
  contributions, re-centred on the GLM reference level, against the log relativity; an individual
  contribution is never a coefficient);
- a descriptive, exploratory DrivAge x VehPower note on the GBM frequency contributions;
- the BonusMalus monotonicity diagnostic of the pure premium of the six candidates (PM-M3-1);
- the constraint register built from the diagnostic block of ``results/cv_gbm.json``.

Nothing here feeds a selection or changes a frozen choice.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import features, figures, freeze, gbm
from .config import DEFAULT_REPRODUCE_CONFIG, load_reproduce_config
from .manifest import sha256_file
from .train import assert_train_only, load_train

COMPONENTS = gbm.COMPONENTS
MAP_KEY = {"frequency": "policy", "severity": "severity", "tweedie": "policy"}
OFFSET_NOTE = (
    "The frequency GBM was trained with log(Exposure) as init_score. The exposure offset is "
    "outside the contributions: the contributions (and the base value) explain the raw score, the "
    "log of the ANNUAL claim rate; the expected count is exp(raw score + log Exposure). The "
    "severity GBM has no offset; the Tweedie GBM uses Exposure only as a weight."
)
CONTRIBUTION_NOTE = (
    "A contribution is the TreeSHAP attribution of one row's raw (log-scale) score to one feature, "
    "relative to the base value of the model. It describes that model on that row; it is not a "
    "coefficient and individual contributions are never equated with GLM coefficients. Group means "
    "of contributions are compared with GLM log relativities only on the aligned log scale after "
    "re-centring on the GLM reference level, and the GBM mean contribution of a level also carries "
    "the part of the effect that the model shares with correlated factors and interactions."
)
RATIONALE = (
    "A worse claims record (higher BonusMalus) should not lower the expected claim frequency or "
    "pure premium: the constraint removes non-monotone artefacts of a tree ensemble on this "
    "factor. It is frozen in protocol section 5 (frequency Poisson and Tweedie GBMs "
    "non-decreasing in BonusMalus; severity Gamma GBM and all other factors unconstrained)."
)


class InterpretError(ValueError):
    """Raised when the interpretation inputs or invariants are not met."""


# --------------------------------------------------------------------------- helpers


def jsonable(obj: Any, where: str = "$") -> Any:
    """JSON-ready copy; a non-finite number fails closed."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v, f"{where}.{k}") for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [jsonable(v, f"{where}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, bool | np.bool_):
        return bool(obj)
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        if not np.isfinite(obj):
            raise InterpretError(f"non-finite value at {where}")
        return float(obj)
    if obj is None or isinstance(obj, str):
        return obj
    raise InterpretError(f"unexpected value type {type(obj).__name__} at {where}")


_NUMLIST = re.compile(r"\[\s*(-?[0-9][-0-9.eE+,\s]*?)\s*\]")


def dump_json(obj: Any) -> str:
    """Indented JSON in which lists of plain numbers stay on one line (the recorded ids)."""
    text = json.dumps(jsonable(obj), indent=2, allow_nan=False)
    return (
        _NUMLIST.sub(
            lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",") if x.strip()) + "]",
            text,
        )
        + "\n"
    )


def draw_sample(train: pd.DataFrame, scfg: dict[str, Any]) -> pd.DataFrame:
    """Fixed train sample: rows sorted by IDpol, ``size`` drawn without replacement by ``seed``."""
    assert_train_only(train)
    if "cv_fold" not in train or not (train["cv_fold"] >= 0).all():
        raise InterpretError("interpret uses train rows only (cv_fold >= 0)")
    t = train.sort_values("IDpol", kind="stable").reset_index(drop=True)
    n = min(int(scfg["size"]), len(t))
    pos = np.sort(np.random.default_rng(int(scfg["seed"])).choice(len(t), size=n, replace=False))
    return t.iloc[pos].reset_index(drop=True)


def _wgroup(labels, values, w) -> pd.DataFrame:
    """Per label: rows, weight sum and weighted mean of ``values`` (order of first appearance)."""
    d = pd.DataFrame({"l": np.asarray(labels), "v": np.asarray(values) * np.asarray(w)})
    d["w"] = np.asarray(w, dtype=float)
    g = d.groupby("l", sort=False).agg(n=("w", "size"), exposure=("w", "sum"), sv=("v", "sum"))
    g["mean"] = g["sv"] / g["exposure"]
    return g.drop(columns="sv")


def _num_bins(x: np.ndarray, nbins: int) -> tuple[np.ndarray, list[str]]:
    """Bin index per row and bin labels: one bin per distinct value if there are at most
    ``nbins`` of them, otherwise quantile bins."""
    uniq = np.unique(x)
    if len(uniq) <= nbins:
        return np.searchsorted(uniq, x), [f"{u:g}" for u in uniq]
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, nbins + 1)))
    idx = np.searchsorted(edges[1:-1], x, side="right")
    labels = []
    for b in range(int(idx.max()) + 1):
        sel = x[idx == b]
        labels.append(f"{sel.min():g}-{sel.max():g}" if len(sel) else f"bin{b}")
    return idx, labels


def _binned_rows(x, c, w, nbins: int) -> list[dict[str, Any]]:
    idx, labels = _num_bins(np.asarray(x, dtype=float), nbins)
    rows = []
    for b in sorted(set(idx.tolist())):
        m = idx == b
        rows.append(
            {
                "label": labels[b],
                "x": float(np.average(np.asarray(x, dtype=float)[m], weights=w[m])),
                "n": int(m.sum()),
                "exposure": float(w[m].sum()),
                "mean": float(np.average(c[m], weights=w[m])),
            }
        )
    return rows


# --------------------------------------------------------------------------- contributions


def contributions(booster, X: np.ndarray, num_threads: int, tolerance: float) -> dict[str, Any]:
    """TreeSHAP contributions (n x (features + 1); the last column is the base value) and the
    per-row identity check: contributions sum to the raw score."""
    contrib = np.asarray(booster.predict(X, pred_contrib=True, num_threads=int(num_threads)))
    raw = np.asarray(booster.predict(X, raw_score=True, num_threads=int(num_threads)))
    err = np.abs(contrib.sum(axis=1) - raw)
    worst = float(err.max())
    if not np.isfinite(contrib).all() or not worst <= tolerance:
        raise InterpretError(
            f"contributions do not sum to the raw score: max abs error {worst:.3g} > {tolerance}"
        )
    base = contrib[:, -1]
    return {
        "phi": contrib[:, :-1],
        "raw": raw,
        "base_value": float(base[0]),
        "base_value_constant": bool(np.ptp(base) <= tolerance),
        "max_abs_row_error": worst,
        "tolerance": float(tolerance),
        "rows_checked": int(len(raw)),
    }


def shap_block(
    comp: str, contrib: dict[str, Any], sample: pd.DataFrame, enc: dict[str, Any], icfg: dict
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summary (mean |contribution|) and dependence data of one component; also the plot panels."""
    names = list(enc["numeric"]) + list(enc["categorical"])
    phi, w = contrib["phi"], sample["Exposure"].to_numpy(dtype=float)
    mean_abs = {n: float(np.abs(phi[:, j]).mean()) for j, n in enumerate(names)}
    total = sum(mean_abs.values())
    dcfg = icfg["dependence"]
    pick = np.sort(
        np.random.default_rng(int(dcfg["point_seed"])).choice(
            len(sample), size=min(int(dcfg["max_points"]), len(sample)), replace=False
        )
    )
    dep: dict[str, Any] = {}
    panels: list[dict[str, Any]] = []
    for f in dcfg["factors"]:
        if f not in names:
            raise InterpretError(f"dependence factor {f!r} is not a GBM feature")
        c = phi[:, names.index(f)]
        if f in enc["categorical"]:
            lv = sample[f].astype(str).to_numpy()
            g = _wgroup(lv, c, w)
            order = [x for x in enc["levels"][f] if x in g.index]
            rows = [
                {"label": x, "n": int(g.loc[x, "n"]), "exposure": float(g.loc[x, "exposure"]),
                 "mean": float(g.loc[x, "mean"])}
                for x in order
            ]  # fmt: skip
            dep[f] = {"kind": "categorical", "bins": rows}
            panels.append({"name": f, "kind": "categorical", "bins": rows, "xlabel": f})
        else:
            x = sample[f].to_numpy(dtype=float)
            rows = _binned_rows(x, c, w, int(dcfg["numeric_bins"]))
            dep[f] = {"kind": "numeric", "bins": rows}
            panels.append(
                {"name": f, "kind": "numeric", "bins": rows, "xlabel": f, "x": x[pick],
                 "y": c[pick], "log_x": f == "Density"}
            )  # fmt: skip
    block = {
        "base_value": contrib["base_value"],
        "base_value_constant": contrib["base_value_constant"],
        "max_abs_row_error": contrib["max_abs_row_error"],
        "tolerance": contrib["tolerance"],
        "rows_checked": contrib["rows_checked"],
        "mean_abs": mean_abs,
        "mean_abs_share": {n: v / total for n, v in mean_abs.items()},
        "ranking": sorted(mean_abs, key=lambda n: (-mean_abs[n], n)),
        "dependence": dep,
    }
    return block, panels


# --------------------------------------------------------------------------- GLM comparison


def _centred_summary(gbm_c: np.ndarray, glm_c: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    diff = gbm_c - glm_c
    corr = None
    if len(gbm_c) >= 3 and np.ptp(gbm_c) > 0 and np.ptp(glm_c) > 0:
        corr = float(np.corrcoef(gbm_c, glm_c)[0, 1])
    return {
        "levels": int(len(gbm_c)),
        "max_abs_diff": float(np.abs(diff).max()),
        "exposure_weighted_mean_abs_diff": float(np.average(np.abs(diff), weights=w)),
        "pearson": corr,
    }


# weight used to average the contributions of each component (the severity GBM is claims-only)
CMP_WEIGHT = {"frequency": "Exposure", "severity": "N", "tweedie": "Exposure"}
CMP_POPULATION = {
    "frequency": "all sample rows, weighted by exposure",
    "severity": "claim rows of the sample only (N > 0), weighted by claim count N: the severity "
    "GBM and GLM are fitted on claims only, so rows without a claim are outside their training "
    "population and are not used",
    "tweedie": "all sample rows, weighted by exposure",
}
OTHER_GLM = {"severity": ["severity"], "tweedie": ["tweedie"]}
GLM_LABEL = {
    "frequency": "full GLM frequency",
    "severity": "full GLM severity (ladder reference)",
    "tweedie": "GLM Tweedie",
}


def glm_sources(
    comp: str, recipes: dict[str, Any], sel: dict[str, Any], models: dict[str, Any]
) -> list[dict[str, Any]]:
    """GLMs compared with the GBM component: first the component of the SELECTED GLM candidate
    (role ``selected``), then any other stored GLM of the same target (role ``other``)."""
    rec = recipes[sel["selected_glm_id"]]
    spec = rec.get("pure_premium" if comp == "tweedie" else comp)
    out: list[dict[str, Any]] = []
    keys: set[str] = set()
    sid = sel["selected_glm_id"]
    if spec is not None and spec["kind"] == "glm":
        key = spec["model_id"].split("#", 1)[1]
        out.append({"key": key, "role": "selected", "model": models[key],
                    "label": f"selected GLM candidate {sid}: {key}"})  # fmt: skip
        keys.add(key)
    elif spec is not None:
        out.append({"key": "constant", "role": "selected", "model": None,
                    "label": f"selected {sid}: constant {comp}"})  # fmt: skip
        keys.add("constant")
    for key in OTHER_GLM.get(comp, []):
        if key not in keys:
            out.append({"key": key, "role": "other", "model": models[key],
                        "label": f"other GLM (not selected): {GLM_LABEL[comp]}"})  # fmt: skip
    return out


def glm_effects(model: dict[str, Any] | None, fmap: features.FeatureMap) -> dict[str, Any]:
    """Log-scale effects per factor from the stored coefficients (intercept excluded): for a
    categorical factor ``{level: coef}`` with the model's reference level at 0, for a continuous
    term the coefficient. Factors the model does not contain are absent (``None``)."""
    out: dict[str, Any] = {}
    if model is None:
        return out
    i = 0
    for spec in model["features"]:
        name = spec["name"]
        if spec["kind"] == "num":
            out[name] = float(model["coef"][i])
            i += 1
            continue
        cats = [str(c) for c in spec["categories"]]
        ref = fmap.levels[name]["reference"]
        if cats[0] != ref or set(cats) != set(fmap.levels[name]["levels"]):
            raise InterpretError(f"{name}: model levels differ from its feature map")
        eff = {cats[0]: 0.0}
        for lv in cats[1:]:
            eff[lv] = float(model["coef"][i])
            i += 1
        out[name] = eff
    if i != len(model["coef"]):
        raise InterpretError("GLM coefficient vector does not match its features")
    return out


def glm_comparison(
    comp: str,
    phi: np.ndarray,
    names: list[str],
    sample: pd.DataFrame,
    fmap: features.FeatureMap,
    sources: list[dict[str, Any]],
    nbins: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Per factor: weighted mean GBM contribution per GLM level (re-centred on the GLM reference
    level) against the log relativity of each GLM source; for the factors that enter the GLM as
    continuous terms the same on weighted quantile bins, both centred on the weighted mean of the
    bins. A factor a GLM does not contain has effect 0 and is marked absent. The severity
    component is evaluated on claim rows only, weighted by N."""
    wcol = CMP_WEIGHT[comp]
    keep = (sample[wcol] > 0).to_numpy()
    if not keep.any():
        raise InterpretError(f"{comp}: no rows with {wcol} > 0 in the sample")
    sample, phi = sample[keep].reset_index(drop=True), phi[keep]
    w = sample[wcol].to_numpy(dtype=float)
    mapped = features.transform(sample, fmap)
    defs = fmap.config["definitions"]
    eff = {s["key"]: glm_effects(s["model"], fmap) for s in sources}
    out: dict[str, Any] = {}
    panels: list[dict[str, Any]] = []
    for f in features.feature_names(fmap.config, "full"):
        c = phi[:, names.index(f)]
        present = {s["key"]: f in eff[s["key"]] for s in sources}
        if defs[f]["kind"] in ("numeric_cap", "log"):
            g = mapped[f].to_numpy(dtype=float)
            idx, _ = _num_bins(g, nbins)  # the transform is monotone: bins of g are bins of x
            raw = sample[f].to_numpy(dtype=float)
            rows = []
            for b in sorted(set(idx.tolist())):
                m = idx == b
                gm = float(np.average(g[m], weights=w[m]))
                rows.append(
                    {"label": f"{raw[m].min():g}-{raw[m].max():g}", "n": int(m.sum()),
                     "weight": float(w[m].sum()), "gbm_mean": float(np.average(c[m], weights=w[m])),
                     "glm_effect": {k: (eff[k][f] * gm if present[k] else 0.0) for k in eff}}
                )  # fmt: skip
            ew = np.array([r["weight"] for r in rows])
            mu = float(np.average([r["gbm_mean"] for r in rows], weights=ew))
            for r in rows:
                r["gbm_mean_centred"] = r["gbm_mean"] - mu
            glm_c: dict[str, np.ndarray] = {}
            for k in eff:
                mk = float(np.average([r["glm_effect"][k] for r in rows], weights=ew))
                for r in rows:
                    r.setdefault("glm_effect_centred", {})[k] = r["glm_effect"][k] - mk
                glm_c[k] = np.array([r["glm_effect_centred"][k] for r in rows])
            gbm_c = np.array([r["gbm_mean_centred"] for r in rows])
            for r in rows:
                r["difference"] = {
                    k: r["gbm_mean_centred"] - r["glm_effect_centred"][k] for k in eff
                }
            out[f] = {
                "kind": "continuous term",
                "transform": defs[f]["kind"],
                "coef": {k: (eff[k][f] if present[k] else None) for k in eff},
                "centring": "weighted mean of the bins (no reference level exists)",
                "present": present,
                "rows": rows,
                "summary": {k: _centred_summary(gbm_c, glm_c[k], ew) for k in eff},
            }
            vals = glm_c
            plabels = [r["label"] for r in rows]
            pgbm = gbm_c
            ylabel = "centred log scale"
        else:
            info = fmap.levels[f]
            ref = info["reference"]
            lv = np.asarray(mapped[f].astype(str))
            g = _wgroup(lv, c, w)
            order = [x for x in info["levels"] if x in g.index]
            if ref not in g.index:
                raise InterpretError(f"{comp}/{f}: reference level {ref} absent from the sample")
            m_ref = float(g.loc[ref, "mean"])
            rows = []
            for x in order:
                gb = float(g.loc[x, "mean"]) - m_ref
                rl = {k: (eff[k][f][x] if present[k] else 0.0) for k in eff}
                rows.append(
                    {"label": x, "n": int(g.loc[x, "n"]), "weight": float(g.loc[x, "exposure"]),
                     "gbm_mean": float(g.loc[x, "mean"]), "gbm_mean_centred": gb,
                     "glm_log_relativity": rl, "difference": {k: gb - rl[k] for k in eff}}
                )  # fmt: skip
            ew = np.array([r["weight"] for r in rows])
            gbm_c = np.array([r["gbm_mean_centred"] for r in rows])
            vals = {k: np.array([r["glm_log_relativity"][k] for r in rows]) for k in eff}
            out[f] = {
                "kind": "levels",
                "reference": ref,
                "centring": f"GBM mean contribution minus that of the GLM reference level {ref}",
                "present": present,
                "rows": rows,
                "summary": {k: _centred_summary(gbm_c, vals[k], ew) for k in eff},
            }
            plabels = [r["label"] for r in rows]
            pgbm = gbm_c
            ylabel = "log scale, reference = 0"
        panels.append(
            {"name": f, "labels": plabels, "gbm": pgbm, "ylabel": ylabel,
             "glms": [{"label": s["label"] + ("" if present[s["key"]] else " (factor absent: 0)"),
                       "role": s["role"], "values": vals[s["key"]]} for s in sources]}
        )  # fmt: skip
    return out, panels


# --------------------------------------------------------------------------- interaction


def band_labels(values, edges: list) -> np.ndarray:
    return features._band(pd.Series(np.asarray(values)), edges).to_numpy()


def interaction_note(
    phi: np.ndarray, names: list[str], sample: pd.DataFrame, icfg: dict[str, Any], fcfg: dict
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """DrivAge contribution of the GBM frequency model by VehPower band (descriptive)."""
    age_edges = fcfg["definitions"]["DrivAge"]["lower_edges"]
    pow_edges = icfg["vehpower_bands"]["lower_edges"]
    age_labels = features._band_labels(age_edges)
    pow_labels = features._band_labels(pow_edges)
    w = sample["Exposure"].to_numpy(dtype=float)
    c = phi[:, names.index("DrivAge")]
    age_b = band_labels(sample["DrivAge"].to_numpy(), age_edges)
    pow_b = band_labels(sample["VehPower"].to_numpy(), pow_edges)
    ref, young = icfg["reference_drivage_band"], list(icfg["young_drivage_bands"])
    thin = float(icfg["min_exposure_years"])
    cells, per_power, lines = [], {}, []
    for pb in pow_labels:
        mp = pow_b == pb
        if not mp.any():
            continue
        g = _wgroup(age_b[mp], c[mp], w[mp])
        if ref not in g.index:
            raise InterpretError(f"interaction: no rows in the reference DrivAge band {ref}")
        ref_mean = float(g.loc[ref, "mean"])
        ys = []
        for ab in age_labels:
            if ab not in g.index:
                ys.append(float("nan"))
                continue
            cell = {
                "vehpower_band": pb, "drivage_band": ab, "n": int(g.loc[ab, "n"]),
                "exposure": float(g.loc[ab, "exposure"]),
                "mean_contribution": float(g.loc[ab, "mean"]),
                "centred_on_reference_band": float(g.loc[ab, "mean"]) - ref_mean,
                "thin": bool(g.loc[ab, "exposure"] < thin),
            }  # fmt: skip
            cells.append(cell)
            ys.append(cell["centred_on_reference_band"])
        yg = g.loc[[b for b in young if b in g.index]]
        yw = float(yg["exposure"].sum())
        young_mean = float((yg["mean"] * yg["exposure"]).sum() / yw) if yw > 0 else None
        per_power[pb] = {
            "n": int(mp.sum()),
            "exposure": float(w[mp].sum()),
            "young_bands_mean_minus_reference_band": None
            if young_mean is None
            else young_mean - ref_mean,
        }
        lines.append({"label": pb, "x": age_labels, "y": ys})
    vals = [
        v["young_bands_mean_minus_reference_band"]
        for v in per_power.values()
        if v["young_bands_mean_minus_reference_band"] is not None
    ]
    block = {
        "label": "exploratory and descriptive, train sample only; no GLM interaction is refitted "
        "(deferred by the 2026-11-08 amendment); the DrivAge contribution varies with VehPower "
        "only through the interactions the trees learned, shared with the VehPower contribution",
        "component": icfg["component"],
        "reference_drivage_band": ref,
        "young_drivage_bands": young,
        "vehpower_bands": pow_labels,
        "drivage_bands": age_labels,
        "min_exposure_years": thin,
        "per_vehpower_band": per_power,
        "cells": cells,
        "young_effect_range_across_vehpower_bands": (max(vals) - min(vals)) if vals else None,
    }
    cloud = {"x": sample["DrivAge"].to_numpy(dtype=float), "y": c,
             "c": sample["VehPower"].to_numpy(dtype=float)}  # fmt: skip
    return block, cloud, lines


# --------------------------------------------------------------------------- monotonicity


def sweep_points(train: pd.DataFrame, mcfg: dict[str, Any]) -> tuple[np.ndarray, str]:
    """BonusMalus sweep points from the TRAIN set: its distinct values, or, when there are more
    than ``max_points``, ``max_points`` equally spaced values from the train minimum to maximum."""
    assert_train_only(train)
    x = train[mcfg["feature"]].to_numpy(dtype=float)
    uniq = np.unique(x)
    mx = int(mcfg["max_points"])
    if len(uniq) <= mx:
        return uniq, "distinct train values"
    return np.linspace(uniq[0], uniq[-1], mx), f"dense grid of {mx} points, train min to max"


def monotone_diagnostic(
    recipes: dict[str, Any], art: dict[str, Any], sample: pd.DataFrame, mcfg: dict[str, Any],
    pts: np.ndarray, pts_mode: str = "given",
) -> dict[str, Any]:  # fmt: skip
    """Sweep ``mcfg.feature`` over the points ``pts`` (see ``sweep_points``) with every other
    feature fixed and report, per candidate, the share of rows whose predicted pure premium
    decreases anywhere along the sweep and the largest relative decrease (also for the
    frequency / severity components)."""
    feat = mcfg["feature"]
    n = min(int(mcfg["rows"]), len(sample))
    pos = np.sort(np.random.default_rng(int(mcfg["seed"])).choice(len(sample), n, replace=False))
    rows = sample.iloc[pos].reset_index(drop=True)
    tol = float(mcfg["tolerance"])
    series: dict[str, dict[str, list[np.ndarray]]] = {}
    for v in pts:
        d = rows.copy()
        d[feat] = np.asarray(v).astype(rows[feat].dtype)
        for cid, comps in freeze.predict(recipes, art, d).items():
            for k, arr in comps.items():
                series.setdefault(cid, {}).setdefault(k, []).append(np.asarray(arr))
    out: dict[str, Any] = {}
    for cid, comps in series.items():
        rec = recipes[cid]
        res: dict[str, Any] = {}
        for k, lst in comps.items():
            m = np.vstack(lst)
            rel = m[1:] / m[:-1] - 1.0
            dec = rel < -tol
            res[k] = {
                "rows_with_decrease": int(dec.any(axis=0).sum()),
                "share_rows_with_decrease": float(dec.any(axis=0).mean()),
                "largest_relative_decrease": float(max(0.0, -rel.min())),
            }
        out[cid] = {"family": rec["family"], "constraint": _constraint_label(rec, feat), **res}
    return {
        "feature": feat,
        "rows": int(n),
        "points": [float(p) for p in pts],
        "n_points": int(len(pts)),
        "points_mode": pts_mode,
        "tolerance": tol,
        "label": "pure premium of each candidate along a BonusMalus sweep, all other features "
        "fixed at the row's own values; a decrease is a relative step below -tolerance between "
        "consecutive points; the sweep points are the distinct train BonusMalus values or, if "
        "there are more than max_points, an equally spaced grid from the train minimum to the "
        "train maximum (so a decrease between two consecutive points of the grid is not seen); "
        "the rows swept are a fixed train sub-sample; a diagnostic on train rows that feeds no "
        "selection",
        "candidates": out,
        "statement": "Only the frequency and the Tweedie GBM are constrained to be non-decreasing "
        "in BonusMalus. The GBM frequency x severity pure premium is a product with the "
        "unconstrained severity GBM and is not guaranteed to be monotone; a decrease is reported "
        "as a limit, never retuned.",
    }


def _constraint_label(rec: dict[str, Any], feat: str) -> str:
    parts = []
    for k in ("frequency", "severity", "pure_premium"):
        spec = rec.get(k)
        if spec is None:
            continue
        if spec["kind"] == "gbm":
            mc = spec["monotone_constraints"]
            on = bool(mc) and mc[spec["feature_names"].index(feat)] != 0
            parts.append(f"{k}: GBM {'constrained' if on else 'unconstrained'}")
        elif spec["kind"] == "glm":
            parts.append(f"{k}: GLM (log-linear in {feat})")
        else:
            parts.append(f"{k}: constant")
    return "; ".join(parts)


# --------------------------------------------------------------------------- constraint register


def _stats(d: list[float]) -> dict[str, Any]:
    return {"mean": float(np.mean(d)), "folds_unconstrained_better": int(np.sum(np.array(d) < 0)),
            "folds": len(d), "d_k": [float(x) for x in d]}  # fmt: skip


def grid_edges(cv: dict[str, Any]) -> dict[str, Any]:
    """Where each chosen value lies in its grid axis (lowest / highest value or interior)."""
    out: dict[str, Any] = {}
    gcomp = cv["config"]["gbm"]["components"]
    for comp in COMPONENTS:
        chosen, axes = cv["components"][comp]["chosen_hp"], {}
        for hp, values in gcomp[comp]["grid"].items():
            lo, hi, v = min(values), max(values), chosen[hp]
            where = (
                "lowest grid value" if v == lo else "highest grid value" if v == hi else "interior"
            )
            axes[hp] = {"chosen": v, "grid": list(values), "position": where}
        out[comp] = {
            "axes": axes,
            "all_on_boundary": all(a["position"] != "interior" for a in axes.values()),
            "cap_reached_any_fold": bool(cv["components"][comp]["cap_reached_any_fold"]),
        }
    return out


def grid_edge_disclosure(cv: dict[str, Any]) -> str:
    """Disclosure text derived from the grid actually used (axis sizes, chosen positions)."""
    ge = grid_edges(cv)
    sizes = sorted({len(a["grid"]) for e in ge.values() for a in e["axes"].values()})
    n_axes = sum(len(e["axes"]) for e in ge.values())
    n_edge = sum(a["position"] != "interior" for e in ge.values() for a in e["axes"].values())
    txt = (
        "The hyperparameter grid was committed before tuning and was not extended after seeing "
        f"the CV results. The grid axes have {', '.join(map(str, sizes))} values; "
        f"{n_edge} of the {n_axes} chosen values lie on the lowest or highest value of their "
        "axis"
    )
    if n_edge:
        txt += ", so the optimum may lie outside the grid along those axes"
    if sizes == [2]:
        txt += " (with two values per axis every chosen value is on the boundary)"
    return (
        txt + "; the chosen points are not claimed to be the most complex or the simplest model "
        "of the grid."
    )


def constraint_register(
    inputs: dict[str, Any], recipes: dict[str, Any], art: dict[str, Any], sample: pd.DataFrame,
    mcfg: dict[str, Any], pts: np.ndarray,
) -> dict[str, Any]:  # fmt: skip
    """Rationale, implementation, verification and CV effect of the BonusMalus constraint."""
    cv, gm = inputs["cv_gbm"], inputs["gbm_models"]
    feat = mcfg["feature"]
    constrained = {}
    for comp, spec in cv["config"]["gbm"]["components"].items():
        constrained[comp] = {k: int(v) for k, v in spec["monotone"].items()}
    impl = {}
    for comp, meta in gm["models"].items():
        impl[comp] = {
            "monotone_constraints": meta["monotone_constraints"],
            "feature_names": meta["feature_names"],
            "method": cv["config"]["gbm"]["base_params"].get("monotone_constraints_method"),
            "rounds": meta["rounds"],
        }
    enc = art["encoding"]
    X = gbm.encode(sample.iloc[: int(cv["config"]["gbm"]["checks"]["sweep_rows"])], enc)
    sweep = {}
    for comp, booster in art["boosters"].items():
        names = list(gm["models"][comp]["feature_names"])
        sweep[comp] = {
            "constrained": bool(constrained[comp]),
            "min_step_raw_score": gbm.monotone_sweep(booster, X, names, feat, pts, 1),
        }
    st, ab = cv["diagnostic"]["separately_tuned"], cv["diagnostic"]["ablation"]
    separately, ablation = {}, {}
    for comp in [c for c, v in st.items() if "constrained" in v]:
        s = st[comp]
        separately[comp] = {
            "constrained": s["constrained"],
            "unconstrained": s["unconstrained"],
            "component_loss_unconstrained_minus_constrained": _stats(
                s["component_loss_diff_unconstrained_minus_constrained"]["d_k"]
            ),
        }
    for key in ("diff_freq_x_sev_unconstrained_minus_constrained",
                "diff_tweedie_unconstrained_minus_constrained"):  # fmt: skip
        separately[key] = _stats(st["primary_loss"][key]["d_k"])
    for comp, a in ab.items():
        ablation[comp] = {
            "rounds": a["rounds"],
            "component_loss_unconstrained_minus_constrained": _stats(
                a["component_loss_diff_unconstrained_minus_constrained"]["d_k"]
            ),
            "primary_loss_unconstrained_minus_constrained": _stats(
                a["primary_loss_diff_unconstrained_minus_constrained"]["d_k"]
            ),
        }
    return {
        "feature": feat,
        "rationale": RATIONALE,
        "constrained_components": constrained,
        "implementation": {
            "how": "LightGBM monotone_constraints on the BonusMalus column "
            "(1 = non-decreasing), fixed in config models.gbm.components.*.monotone",
            "by_component": impl,
        },
        "verification": {
            "sweep_points": [float(p) for p in pts],
            "rows": int(len(X)),
            "min_step_raw_score_by_component": sweep,
            "training_gate": cv["checks"]["monotone_sweep"],
            "tolerance": cv["config"]["gbm"]["checks"]["monotone_tolerance"],
            "note": "the raw-score sweep of the constrained components must not decrease "
            "(minimum step >= -tolerance); the severity GBM is unconstrained and listed for "
            "comparison; the pure premium sweep is in the monotonicity block",
        },
        "cv_effect": {
            "label": cv["diagnostic"]["label"],
            "scores_label": cv["config"]["scores_label"],
            "sign": "differences are unconstrained minus constrained: a negative value means the "
            "unconstrained variant has the lower (better) loss",
            "separately_tuned_procedures": separately,
            "separately_tuned_note": "comparison of two separately tuned procedures (identical "
            "grid, folds, seed and budget), so the hyperparameters and rounds differ between "
            "the variants",
            "controlled_ablation": ablation,
            "controlled_ablation_note": "same chosen hyperparameters and rounds, the constraint "
            "toggled",
            "kind": "optimistic tuning scores of CV folds also used for early stopping and grid "
            "scoring; not unbiased estimates and not test results",
        },
        "grid_edges": grid_edges(cv),
        "grid_edge_disclosure": grid_edge_disclosure(cv),
    }


# --------------------------------------------------------------------------- orchestration


def compute(
    train: pd.DataFrame,
    cfg: dict[str, Any],
    inputs: dict[str, Any],
    sel: dict[str, Any],
    art: dict[str, Any],
    recipes: dict[str, Any],
    root: Path,
    fig_dir: Path | None,
) -> dict[str, Any]:
    """The whole interpretation (pure given its arguments; figures are drawn into ``fig_dir``
    unless it is None)."""
    assert_train_only(train)
    icfg, fcfg = cfg["interpret"], cfg["figures"]
    prefix = icfg["outputs"]["figure_prefix"]
    timing: dict[str, float] = {}
    t0 = time.perf_counter()
    sample = draw_sample(train, icfg["sample"])
    enc = art["encoding"]
    names = list(enc["numeric"]) + list(enc["categorical"])
    X = gbm.encode(sample, enc)
    fig_files: list[str] = []
    shap: dict[str, Any] = {}
    contribs: dict[str, dict[str, Any]] = {}
    panels_by: dict[str, list[dict[str, Any]]] = {}
    for comp in COMPONENTS:
        booster = art["boosters"][comp]
        gbm.check_feature_order(booster, names)
        contribs[comp] = contributions(
            booster, X, art["num_threads"], float(icfg["contributions"]["tolerance"])
        )
        shap[comp], panels_by[comp] = shap_block(comp, contribs[comp], sample, enc, icfg)
    timing["contributions"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    comparison: dict[str, Any] = {}
    cmp_panels: dict[str, list[dict[str, Any]]] = {}
    for comp in COMPONENTS:
        fmap = art["feature_maps"][MAP_KEY[comp]]
        srcs = glm_sources(comp, recipes, sel, inputs["glm_models"])
        comparison[comp], cmp_panels[comp] = glm_comparison(
            comp, contribs[comp]["phi"], names, sample, fmap, srcs,
            int(icfg["glm_comparison"]["numeric_bins"]),
        )  # fmt: skip
        comparison[comp] = {
            "population": CMP_POPULATION[comp],
            "weight": CMP_WEIGHT[comp],
            "glms": [{k: v for k, v in x.items() if k != "model"} for x in srcs],
            "factors": comparison[comp],
        }
    inter, cloud, lines = interaction_note(
        contribs[icfg["interaction"]["component"]]["phi"], names, sample, icfg["interaction"],
        cfg["features"],
    )  # fmt: skip
    timing["comparison"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    pts, pts_mode = sweep_points(train, icfg["monotonicity"])
    mono = monotone_diagnostic(recipes, art, sample, icfg["monotonicity"], pts, pts_mode)
    register = constraint_register(inputs, recipes, art, sample, icfg["monotonicity"], pts)
    timing["monotonicity"] = time.perf_counter() - t0
    if fig_dir is not None:
        t0 = time.perf_counter()
        fig_files.append(
            figures.bar_importance(
                fcfg,
                fig_dir / f"{prefix}shap_summary.png",
                {c: shap[c]["mean_abs"] for c in COMPONENTS},
                "Mean |contribution| per feature (train sample, log scale)",
                "mean |contribution|",
            )  # fmt: skip
        )
        for comp in COMPONENTS:
            fig_files.append(
                figures.dependence_grid(
                    fcfg,
                    fig_dir / f"{prefix}dependence_{comp}.png",
                    panels_by[comp],
                    f"GBM {comp}: contribution by factor (train sample)",
                )  # fmt: skip
            )
            fig_files.append(
                figures.compare_grid(
                    fcfg,
                    fig_dir / f"{prefix}glm_comparison_{comp}.png",
                    cmp_panels[comp],
                    f"GBM {comp}: mean contribution vs log GLM relativity",
                )  # fmt: skip
            )
        fig_files.append(
            figures.interaction_figure(
                fcfg,
                fig_dir / f"{prefix}interaction_drivage_vehpower.png",
                {**cloud, "xlabel": "DrivAge", "clabel": "VehPower"},
                lines,
                "GBM frequency: DrivAge contribution by VehPower (exploratory)",
            )  # fmt: skip
        )
        timing["figures"] = time.perf_counter() - t0
    ids = [int(i) for i in sample["IDpol"]]
    res = {
        "label": "Train-only interpretation of the frozen candidates (descriptive; feeds no "
        "selection and changes no frozen choice). Produced by `fremtpl2 interpret`.",
        "test_set_used": False,
        "selected_glm_id": sel["selected_glm_id"],
        "selected_gbm_id": sel["selected_gbm_id"],
        "config": {"interpret": icfg, "figures": fcfg},
        "sources": _sources(root, cfg, inputs),
        "sample": {
            "size": int(len(sample)),
            "seed": int(icfg["sample"]["seed"]),
            "train_policies": int(len(train)),
            "cv_fold_range": [int(sample["cv_fold"].min()), int(sample["cv_fold"].max())],
            "exposure": float(sample["Exposure"].sum()),
            "claims": int(sample["N"].sum()),
            "ids_sha256": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(),
            "ids": ids,
        },
        "notes": {"exposure_offset": OFFSET_NOTE, "contributions": CONTRIBUTION_NOTE},
        "shap": shap,
        "glm_comparison": {
            "label": "mean GBM contribution per GLM level, re-centred on the GLM reference level, "
            "against the log GLM relativity exp(beta) of the stored coefficients of the GLM of "
            "the SELECTED GLM candidate for the same component (results/glm_models.json, on its "
            "own feature map and reference levels; a factor the GLM does not contain is absent, "
            "effect 0), plus any other stored GLM of the same target as a labelled extra column "
            "(aligned log scale; not coefficients)",
            "components": comparison,
        },
        "interaction": inter,
        "monotonicity": mono,
        "constraint_register": register,
        "figures": fig_files,
    }
    return {"result": jsonable(res), "timing_seconds": timing}


def _policy_path(root: Path) -> Path:
    """The verification policy of the project, else the one shipped with the source tree."""
    own = root / DEFAULT_REPRODUCE_CONFIG
    return own if own.is_file() else Path(__file__).resolve().parents[2] / DEFAULT_REPRODUCE_CONFIG


def _drop_keys(obj: Any, drop: set[str]) -> Any:
    if isinstance(obj, dict):
        return {k: _drop_keys(v, drop) for k, v in obj.items() if k not in drop}
    if isinstance(obj, list):
        return [_drop_keys(v, drop) for v in obj]
    return obj


def _source_entry(file: Path, rel: str, volatile: list[str]) -> dict[str, Any]:
    """Provenance of an input: a JSON file that carries telemetry is hashed as canonical JSON with
    the volatile telemetry keys removed (so a rerun gives the same hash); other files by bytes."""
    if file.suffix == ".json":
        doc = json.loads(file.read_text(encoding="utf-8"))
        if _drop_keys(doc, set(volatile)) != doc:
            canon = json.dumps(
                _drop_keys(doc, set(volatile)), sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            )  # fmt: skip
            return {
                "path": rel,
                "content_sha256_excluding": volatile,
                "sha256": hashlib.sha256(canon.encode("utf-8")).hexdigest(),
            }
    return {"path": rel, "sha256": sha256_file(file)}


def _sources(root: Path, cfg: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    paths = freeze.project_paths(cfg, root)
    files = {k: paths[k] for k in ("cv_glm", "glm_models", "feature_map", "cv_gbm", "gbm_models")}
    volatile = sorted(load_reproduce_config(_policy_path(root))["check"]["volatile_keys"])
    out = {k: _source_entry(root / p, p, volatile) for k, p in files.items()}
    for comp, meta in inputs["gbm_models"]["models"].items():
        out[f"gbm_model_{comp}"] = {
            "path": f"{paths['model_dir']}/{meta['file']}",
            "sha256": meta["sha256"],
        }
    return out


def run_interpret(
    cfg: dict[str, Any], root: Path | None = None, train: pd.DataFrame | None = None
) -> dict[str, Any]:
    """``fremtpl2 interpret``: train rows only; nothing is fitted; outputs are staged and moved
    into place only when everything succeeded."""
    root = Path.cwd() if root is None else Path(root)
    paths = freeze.project_paths(cfg, root)
    inputs = freeze.read_inputs(root, paths)
    sel = freeze.recompute_selections(inputs, cfg)
    train = load_train(cfg) if train is None else train
    assert_train_only(train)
    null_rate = float(train["freq_count"].sum() / train["Exposure"].sum())
    recipes = freeze.build_recipes(inputs, sel, cfg, paths, null_rate)
    art = freeze.load_artifacts(root, paths, inputs, cfg["freeze"]["runtime"]["num_threads"])
    outs = cfg["interpret"]["outputs"]
    stage = root / Path(cfg["paths"]["processed"]) / outs["staging_dir"]
    if stage.exists():
        shutil.rmtree(stage)
    fig_stage = stage / "figures"
    out = compute(train, cfg, inputs, sel, art, recipes, root, fig_stage)
    res = out["result"]
    (stage / outs["json"]).write_text(dump_json(res), encoding="utf-8")
    (stage / outs["report"]).write_text(render_report(res), encoding="utf-8")
    results = root / cfg["paths"]["results"]
    reports = root / cfg["paths"]["reports"]
    fig_dir = reports / cfg["figures"]["dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    for f in fig_stage.iterdir():
        f.replace(fig_dir / f.name)
    (stage / outs["json"]).replace(results / outs["json"])
    (stage / outs["report"]).replace(reports / outs["report"])
    shutil.rmtree(stage)
    res["timing_seconds"] = out["timing_seconds"]
    return res


# --------------------------------------------------------------------------- report


def _f(x: Any, nd: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def _row(cells: list[Any]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    return [_row(header), _row(["---"] * len(header)), *(_row(r) for r in rows), ""]


def render_report(res: dict[str, Any]) -> str:
    """``reports/interpretation.md``: every number is read from the payload."""
    fig_dir = res["config"]["figures"]["dir"]
    sm = res["sample"]
    out = [
        "# Train-only interpretation of the frozen candidates",
        "",
        "Generated by `fremtpl2 interpret` from `results/interpretation.json`. "
        "Descriptive, train rows only, no model is refitted, no test row is read. Nothing here "
        "feeds a selection or changes a frozen choice.",
        "",
        f"Fixed train sample: {sm['size']} of {sm['train_policies']} train policies "
        f"(seed {sm['seed']}, cv_fold {sm['cv_fold_range'][0]}..{sm['cv_fold_range'][1]}, "
        f"exposure {_f(sm['exposure'], 1)} years, {sm['claims']} claims); the ids are recorded in "
        f"the JSON (sha256 `{sm['ids_sha256'][:16]}...`). Selected candidates: GLM "
        f"`{res['selected_glm_id']}`, GBM `{res['selected_gbm_id']}`.",
        "",
        "## 1. Contributions (TreeSHAP) of the GBM components",
        "",
        res["notes"]["contributions"],
        "",
        res["notes"]["exposure_offset"],
        "",
    ]
    rows = []
    for comp, b in res["shap"].items():
        rows.append(
            [
                comp,
                _f(b["base_value"], 5),
                f"{b['max_abs_row_error']:.2e}",
                f"{b['tolerance']:.0e}",
                b["rows_checked"],
                ", ".join(f"{n} ({_f(b['mean_abs'][n], 4)})" for n in b["ranking"][:4]),
            ]
        )
    out += ["Per-row identity: contributions plus base value equal the raw score.", ""]
    out += _table(
        [
            "component",
            "base value",
            "max abs row error",
            "tolerance",
            "rows",
            "top features (mean abs)",
        ],
        rows,
    )
    out += [f"![mean absolute contribution]({fig_dir}/{_fig(res, 'shap_summary')})", ""]
    for comp in res["shap"]:
        out += [
            f"### Dependence: {comp}",
            "",
            f"![{comp} dependence]({fig_dir}/{_fig(res, f'dependence_{comp}')})",
            "",
        ]
    out += [
        "## 2. GBM contributions against GLM log relativities",
        "",
        res["glm_comparison"]["label"] + ".",
        "",
    ]
    for comp, cc in res["glm_comparison"]["components"].items():
        facs, glms = cc["factors"], cc["glms"]
        out += [f"### {comp}", "", f"Population and weights: {cc['population']}.", ""]
        out += ["GLM columns: " + "; ".join(f"{g['key']}: {g['label']}" for g in glms) + ".", ""]
        if comp == "severity":
            out += [
                "Limitation: the severity GBM is compared on claim rows only and the GLM column "
                "is the severity GLM of the selected candidate; factors it does not contain "
                "show an effect of 0 (absent), whatever the GBM contribution is.",
                "",
            ]
        srows = []
        for f, v in facs.items():
            row = [f, v["kind"], next(iter(v["summary"].values()))["levels"]]
            for g in glms:
                sm = v["summary"][g["key"]]
                tag = "" if v["present"][g["key"]] else "absent: "
                row.append(f"{tag}{_f(sm['max_abs_diff'])} / {_f(sm['pearson'], 3)}")
            srows.append(row)
        out += _table(
            ["factor", "kind", "levels / bins"]
            + [f"{g['key']} ({g['role']}): max abs diff / Pearson" for g in glms],
            srows,
        )
        out += [f"![{comp} comparison]({fig_dir}/{_fig(res, f'glm_comparison_{comp}')})", ""]
        for f, v in facs.items():
            out += [f"{f} ({v['centring']})", ""]
            absent = [g["key"] for g in glms if not v["present"][g["key"]]]
            if absent:
                out += ["Absent from: " + ", ".join(absent) + " (effect 0 on the log scale).", ""]
            lev = v["kind"] == "levels"
            header = ["level" if lev else "bin", "n", "weight", "GBM mean contribution (centred)"]
            for g in glms:
                header += [
                    f"{g['key']} " + ("log relativity" if lev else "effect (centred)"),
                    f"diff {g['key']}",
                ]
            body = []
            for r in v["rows"]:
                row = [r["label"], r["n"], _f(r["weight"], 1), _f(r["gbm_mean_centred"])]
                src = r["glm_log_relativity"] if lev else r["glm_effect_centred"]
                for g in glms:
                    row += [_f(src[g["key"]]), _f(r["difference"][g["key"]])]
                body.append(row)
            out += _table(header, body)
    out += _interaction_md(res, fig_dir)
    out += _monotone_md(res)
    out += _register_md(res)
    return "\n".join(out).rstrip() + "\n"


def _fig(res: dict[str, Any], stem: str) -> str:
    prefix = res["config"]["interpret"]["outputs"]["figure_prefix"]
    name = f"{prefix}{stem}.png"
    if name not in res["figures"]:
        raise InterpretError(f"figure {name} was not produced")
    return name


def _interaction_md(res: dict[str, Any], fig_dir: str) -> list[str]:
    it = res["interaction"]
    out = [
        "## 3. Interaction note: young drivers x vehicle power (exploratory)",
        "",
        f"Component: GBM {it['component']}. " + it["label"] + ".",
        "",
        f"Mean DrivAge contribution per DrivAge band within each VehPower band, centred on the "
        f"reference band {it['reference_drivage_band']} of the same VehPower band "
        "(exposure-weighted over the sample; cells below "
        f"{_f(it['min_exposure_years'], 0)} exposure-years are marked thin and not to be read).",
        "",
    ]
    cells = {(c["vehpower_band"], c["drivage_band"]): c for c in it["cells"]}
    rows = []
    for ab in it["drivage_bands"]:
        row = [ab]
        for pb in it["per_vehpower_band"]:
            c = cells.get((pb, ab))
            row.append(
                "n/a"
                if c is None
                else _f(c["centred_on_reference_band"], 3) + (" (thin)" if c["thin"] else "")
            )
        rows.append(row)
    out += _table(["DrivAge band \\ VehPower band", *it["per_vehpower_band"]], rows)
    young = ", ".join(it["young_drivage_bands"])
    out += _table(
        ["VehPower band", "n", "exposure", f"young bands ({young}) minus reference band"],
        [
            [pb, v["n"], _f(v["exposure"], 1), _f(v["young_bands_mean_minus_reference_band"], 3)]
            for pb, v in it["per_vehpower_band"].items()
        ],
    )
    out += [
        "Range of that young-driver effect across VehPower bands: "
        f"{_f(it['young_effect_range_across_vehpower_bands'], 3)} (log scale). "
        "The pattern is a description of what the trees learned on the train sample, not a "
        "tested interaction; no GLM interaction was refitted.",
        "",
        f"![interaction]({fig_dir}/{_fig(res, 'interaction_drivage_vehpower')})",
        "",
    ]
    return out


def _monotone_md(res: dict[str, Any]) -> list[str]:
    m = res["monotonicity"]
    out = [
        "## 4. BonusMalus monotonicity of the pure premium (PM-M3-1)",
        "",
        m["label"] + f" ({m['rows']} rows, {len(m['points'])} {m['feature']} points "
        f"{_f(m['points'][0], 0)}..{_f(m['points'][-1], 0)}, {m['points_mode']}; "
        f"tolerance {m['tolerance']:g}).",
        "",
        m["statement"],
        "",
    ]
    rows = []
    for cid, c in m["candidates"].items():
        pp = c["pure_premium"]
        extra = ", ".join(
            f"{k}: {_pct(c[k]['share_rows_with_decrease'])} / "
            f"{_pct(c[k]['largest_relative_decrease'])}"
            for k in ("frequency", "severity")
            if k in c
        )
        rows.append(
            [
                cid,
                c["constraint"],
                pp["rows_with_decrease"],
                _pct(pp["share_rows_with_decrease"]),
                _pct(pp["largest_relative_decrease"]),
                extra or "-",
            ]
        )
    out += _table(
        [
            "candidate",
            "constraint",
            "rows with a decrease",
            "share of rows with any decrease",
            "largest relative decrease",
            "components (share / largest)",
        ],
        rows,
    )
    return out


def _register_md(res: dict[str, Any]) -> list[str]:
    r = res["constraint_register"]
    ce = r["cv_effect"]
    out = [
        "## 5. Constraint register",
        "",
        f"**Rationale.** {r['rationale']}",
        "",
        f"**Implementation.** {r['implementation']['how']}.",
        "",
    ]
    out += _table(
        ["component", "constraint vector", "feature order", "method", "rounds"],
        [
            [c, v["monotone_constraints"], ", ".join(v["feature_names"]), v["method"], v["rounds"]]
            for c, v in r["implementation"]["by_component"].items()
        ],
    )
    ver = r["verification"]
    out += [
        f"**Verification.** {ver['note']}. Sweep of {ver['rows']} train rows over "
        f"{len(ver['sweep_points'])} {r['feature']} points; the training gate recorded in "
        f"`results/cv_gbm.json` is {ver['training_gate']}.",
        "",
    ]
    out += _table(
        ["component", "constrained", "min step of the raw score"],
        [
            [c, v["constrained"], f"{v['min_step_raw_score']:.3e}"]
            for c, v in ver["min_step_raw_score_by_component"].items()
        ],
    )
    out += ["**CV effect.** " + ce["label"] + ". " + ce["sign"] + ". " + ce["kind"] + ".", ""]
    out += ["Comparison of separately tuned procedures: " + ce["separately_tuned_note"] + ".", ""]
    rows = []
    for k, v in ce["separately_tuned_procedures"].items():
        if "d_k" in v:
            rows.append([k, _f(v["mean"], 5), f"{v['folds_unconstrained_better']}/{v['folds']}"])
        else:
            s = v["component_loss_unconstrained_minus_constrained"]
            rows.append(
                [
                    f"{k} component loss",
                    _f(s["mean"], 5),
                    f"{s['folds_unconstrained_better']}/{s['folds']}",
                ]
            )
    out += _table(
        [
            "quantity (unconstrained minus constrained)",
            "mean over folds",
            "folds with the unconstrained variant better",
        ],
        rows,
    )
    out += ["Controlled ablation: " + ce["controlled_ablation_note"] + ".", ""]
    rows = []
    for k, v in ce["controlled_ablation"].items():
        for q in ("component_loss", "primary_loss"):
            s = v[f"{q}_unconstrained_minus_constrained"]
            rows.append(
                [
                    k,
                    q.replace("_", " "),
                    v["rounds"],
                    _f(s["mean"], 5),
                    f"{s['folds_unconstrained_better']}/{s['folds']}",
                ]
            )
    out += _table(
        [
            "component",
            "loss",
            "rounds",
            "mean over folds (unconstrained minus constrained)",
            "folds with the unconstrained variant better",
        ],
        rows,
    )
    out += ["**Grid-edge disclosure.** " + r["grid_edge_disclosure"], ""]
    rows = []
    for comp, e in r["grid_edges"].items():
        for hp, a in e["axes"].items():
            rows.append([comp, hp, a["chosen"], a["grid"], a["position"]])
    out += _table(["component", "hyperparameter", "chosen", "grid", "position in the grid"], rows)
    return out
