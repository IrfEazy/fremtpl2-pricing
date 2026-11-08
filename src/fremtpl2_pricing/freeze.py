"""Freeze stage (train only): the explicit manifest that the single test evaluation asserts.

``fremtpl2 freeze`` never reads a test row. It reads the committed training artifacts
(``cv_glm.json``, ``glm_models.json``, ``feature_map.json``, ``cv_gbm.json``, ``gbm_models.json``
and the LightGBM model files), recomputes both pure-premium selections with ``selection.py`` from
the recorded per-fold losses (a mismatch with the recorded trace fails), checks that the current
config still equals the training record (except the evaluation-only keys), checks that the stored
feature maps cover the full-data factor domains of ``results/audit.json`` and writes
``results/freeze_manifest.json`` with:

- the six prespecified candidates (protocol section 3) and a complete, fit-free prediction recipe
  for each (component model ids, feature map ids, power, scale conventions);
- ``selected_glm_id`` and ``selected_gbm_id`` with the selection rule version and trace hashes;
- the sha256 of every input and model file, the config, the protocol, ``uv.lock``, the frozen
  split and the raw data files, the training commit (clean work tree required) and the runtime
  identity (python, library versions, lock-file hash, threads), which ``evaluate`` asserts;
- the evaluation specification (population, primary loss, primary contrast, bootstrap, decision,
  technical-rerun and recovery policy) and a train-row prediction sample (random rows plus edge
  cases: extreme values and every level present in train) that ``evaluate`` reproduces, with a
  determinism check (repeat and permuted row order), before any test access.

The factor-level coverage check uses the factor domains of the milestone 1 full-data audit
(``results/audit.json``: covariate levels and ranges only, no outcomes; protocol amendment).

The prediction code here (``load_artifacts`` / ``predict``) is shared with ``evaluate``; it only
loads stored parameters (``GLMModel.from_dict``, ``FeatureMap.from_dict``, the hashed LightGBM text
files) and never fits anything.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import features, gbm, glm, selection
from .manifest import sha256_file
from .train import assert_train_only, load_train

MANIFEST_VERSION = "freeze-manifest-v1"
PP_NULL, PP_BM = "pp_null", "bonus_malus_only"
GLM_TW = "glm_tweedie"
GBM_FXS, GBM_TW = gbm.PP_FXS, gbm.PP_TW
COMPONENT_KEYS = ("pure_premium", "frequency", "severity")
SCALE_CONVENTIONS = {
    "rate_unit": "per exposure-year (Exposure capped at 1 by clean)",
    "frequency": "annual claim rate; expected claim count = rate * Exposure",
    "severity": "expected capped cost per claim (S_cap / N scale)",
    "pure_premium": "annual capped cost rate (S_cap / Exposure scale); expected capped cost = "
    "rate * Exposure; freq x sev candidates: frequency rate * severity",
    "glm": "rate = exp(intercept + sum_j design_ij coef_j) of the stored model on "
    "features.transform with the stored feature map (per-row reduction: row-order invariant)",
    "gbm": "rate = exp(raw score), raw_score=True never includes an init score; expected counts "
    "= exp(raw + log Exposure) computed explicitly",
    "capped": "capped prediction = expected capped-claim cost, never a truncation",
}


class FreezeError(ValueError):
    """Raised when the freeze or the evaluation preconditions are not met."""


# --------------------------------------------------------------------------- git and hashing


def git(root: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise FreezeError(f"git {' '.join(args)} failed: {detail.strip()}") from exc
    return out.stdout


def assert_clean_tree(root: Path, ignore: frozenset[str] = frozenset()) -> None:
    """The work tree must be clean, including untracked non-ignored files (paths in ``ignore``
    excepted)."""
    status = git(root, "status", "--porcelain", "--untracked-files=normal")
    lines = [ln for ln in status.splitlines() if ln.strip() and ln[3:].strip('"') not in ignore]
    if lines:
        raise FreezeError(
            "the work tree is not clean (commit or remove these first):\n" + "\n".join(lines)
        )


def head_commit(root: Path) -> str:
    return git(root, "rev-parse", "HEAD").strip()


def canonical_sha256(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(blob).hexdigest()


def file_entry(root: Path, rel: str, role: str) -> dict[str, Any]:
    path = root / rel
    if not path.is_file():
        raise FreezeError(f"missing file: {rel}")
    return {"path": rel, "role": role, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_files(root: Path, entries: list[dict[str, Any]], append_only: set[str]) -> list[str]:
    """Problems of the hashed files; a path in ``append_only`` may have grown if its first
    ``bytes`` bytes still have the recorded sha256 (technical-rerun amendments)."""
    problems = []
    for e in entries:
        path = root / e["path"]
        if not path.is_file():
            problems.append(f"missing file: {e['path']}")
            continue
        size = path.stat().st_size
        if e["path"] in append_only and size >= e["bytes"]:
            with path.open("rb") as fh:
                digest = hashlib.sha256(fh.read(e["bytes"])).hexdigest()
            if digest != e["sha256"]:
                problems.append(f"{e['path']}: the frozen content was changed (not an append)")
            continue
        if size != e["bytes"] or sha256_file(path) != e["sha256"]:
            problems.append(f"sha256 mismatch: {e['path']}")
    return problems


def environment(packages: list[str]) -> dict[str, Any]:
    """Descriptive environment record (the asserted part is :func:`runtime_identity`)."""
    return {
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {p: importlib.metadata.version(p) for p in packages},
    }


def runtime_identity(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    """What must be equal between the freeze and the evaluation (and for a technical rerun):
    python version and implementation, machine architecture, the versions of
    ``freeze.environment_packages``, the sha256 of the lock file and the prediction threads."""
    fz = cfg["freeze"]
    lock = root / fz["runtime"]["lock_file"]
    if not lock.is_file():
        raise FreezeError(f"missing lock file: {fz['runtime']['lock_file']}")
    ident = {
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "machine": platform.machine(),
        "packages": {p: importlib.metadata.version(p) for p in fz["environment_packages"]},
        "lock_file": fz["runtime"]["lock_file"],
        "lock_sha256": sha256_file(lock),
        "num_threads": int(fz["runtime"]["num_threads"]),
    }
    return {**ident, "sha256": canonical_sha256(ident)}


def runtime_differences(current: dict[str, Any], frozen: dict[str, Any]) -> list[str]:
    keys = sorted((set(current) | set(frozen)) - {"sha256"})
    diff = [
        f"{k}: {frozen.get(k)!r} -> {current.get(k)!r}"
        for k in keys
        if current.get(k) != frozen.get(k)
    ]
    if not diff and current.get("sha256") != frozen.get("sha256"):
        diff.append("runtime identity digest differs")
    return diff


# --------------------------------------------------------------------------- paths


def project_paths(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    """Project-relative paths of every file the freeze and evaluate stages read or write."""
    p, fz = cfg["paths"], cfg["freeze"]
    outs, gouts = cfg["models"]["outputs"], cfg["models"]["gbm"]["outputs"]
    results, processed = Path(p["results"]), Path(p["processed"])
    rel = lambda x: Path(x).as_posix()  # noqa: E731
    return {
        "root": root,
        "results": rel(results),
        "cv_glm": rel(results / outs["cv_json"]),
        "glm_models": rel(results / outs["models"]),
        "feature_map": rel(results / outs["feature_map"]),
        "cv_gbm": rel(results / gouts["cv_json"]),
        "gbm_models": rel(results / gouts["models_json"]),
        "model_dir": rel(processed / gouts["model_dir"]),
        "audit": rel(results / fz["audit"]),
        "protocol": rel(Path(p["reports"]) / fz["protocol"]),
        "split": rel(processed / "split.parquet"),
        "raw": [rel(Path(p["raw"]) / f) for f in fz["hashed_raw_files"]],
        "project_files": [rel(Path(f)) for f in fz["hashed_project_files"]],
        "manifest": rel(results / fz["manifest"]),
        "state": rel(results / fz["test_state"]),
        "mirror": rel(processed / fz["test_state_mirror"]),
        "lock": rel(processed / fz["lock"]),
        "attempts": rel(results / fz["attempts_dir"]),
        "pointer": rel(results / fz["metrics_pointer"]),
        "staging": rel(processed / fz["staging_dir"]),
        "rehearsal": rel(processed / fz["rehearsal"]["dir"]),
    }


def _read_json(root: Path, rel: str) -> Any:
    path = root / rel
    if not path.is_file():
        raise FreezeError(f"missing file: {rel}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_inputs(root: Path, paths: dict[str, Any]) -> dict[str, Any]:
    keys = ("cv_glm", "glm_models", "feature_map", "cv_gbm", "gbm_models", "audit")
    return {k: _read_json(root, paths[k]) for k in keys}


# --------------------------------------------------------------------------- selection


def _same(recomputed: Any, recorded: Any, what: str) -> None:
    if json.loads(json.dumps(recomputed)) != recorded:
        raise FreezeError(f"{what}: the recomputed selection differs from the recorded trace")


def recompute_selections(inputs: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Recompute both pure-premium selections (protocol section 4) from the recorded per-fold
    losses with ``selection.py``; fail unless every trace and selected id agrees."""
    sel = inputs["cv_glm"]["selection"]
    if sel["rule_version"] != selection.RULE_VERSION:
        raise FreezeError(f"GLM selection rule {sel['rule_version']} != {selection.RULE_VERSION}")
    ids = [c["id"] for c in sel["candidates"]]
    if ids != [e["id"] for e in cfg["models"]["glm"]["severity_ladder"]]:
        raise FreezeError("the recorded severity ladder differs from models.glm.severity_ladder")
    pp = sel["pure_premium_loss_per_fold"]
    lad = selection.ladder({i: pp[i] for i in ids})
    _same(lad, sel["ladder"], "GLM severity ladder")
    chosen = lad["selected"]
    if chosen != sel["selected_severity"]:
        raise FreezeError("GLM selected severity differs from the recorded one")
    step6 = selection.freq_x_sev_vs_tweedie(pp[chosen], sel["tweedie_loss_per_fold"])
    rec6 = dict(sel["freq_x_sev_vs_tweedie"])
    if (
        rec6.pop("freq_x_sev_severity") != chosen
        or rec6.pop("freq_x_sev_loss_per_fold") != pp[chosen]
    ):
        raise FreezeError("GLM freq x sev vs Tweedie: recorded severity inputs differ")
    _same(step6, rec6, "GLM freq x sev vs Tweedie")
    fam = step6["selected"]
    glm_id = f"glm_freq_x_sev[{chosen}]" if fam == "freq_x_sev" else GLM_TW
    rec_pp = sel["selected_glm_pure_premium"]
    exp_pp = (
        {"id": f"freq_x_sev[{chosen}]", "family": "freq_x_sev", "severity": chosen}
        if fam == "freq_x_sev"
        else {
            "id": "tweedie",
            "family": "tweedie",
            "tweedie_power": sel["tweedie_power_final"],
        }
    )
    _same(exp_pp, rec_pp, "GLM selected pure premium")

    gsel = inputs["cv_gbm"]["selection"]
    per_fold = inputs["cv_gbm"]["oof"]["per_fold"]
    fxs = [f["pure_premium"][GBM_FXS]["primary_loss"] for f in per_fold]
    tw = [f["pure_premium"][GBM_TW]["primary_loss"] for f in per_fold]
    gtrace = selection.freq_x_sev_vs_tweedie(fxs, tw, names=(GBM_FXS, GBM_TW))
    _same(gtrace, gsel["gbm_pure_premium"], "GBM freq x sev vs Tweedie")
    gbm_id = gtrace["selected"]
    if gsel["selected"] != gbm_id or inputs["gbm_models"]["selected_pure_premium_candidate"] != (
        gbm_id
    ):
        raise FreezeError("GBM selected candidate differs between the trace and the files")
    k = inputs["cv_glm"]["config"]["cv_folds"]
    if lad["n_folds"] != k or gtrace["n_folds"] != k or step6["n_folds"] != k:
        raise FreezeError("a selection trace does not cover every registered fold")
    traces = {"glm_ladder": lad, "glm_freq_x_sev_vs_tweedie": step6, "gbm": gtrace}
    return {
        "rule_version": selection.RULE_VERSION,
        "selected_glm_id": glm_id,
        "selected_glm_severity": chosen,
        "selected_gbm_id": gbm_id,
        "trace_sha256": {k_: canonical_sha256(v) for k_, v in traces.items()},
        "traces": traces,
    }


# --------------------------------------------------------------------------- training record


def check_training_record(inputs: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Training gates passed, the test set was never used, and the current config equals the
    training record (models.* except the evaluation-only keys; models.gbm against cv_gbm)."""
    cv, cg = inputs["cv_glm"], inputs["cv_gbm"]
    problems = []
    if not cv["checks"]["passed"] or cv["convergence"]["n_not_converged"] != 0:
        problems.append("GLM training gates did not pass")
    if not cg["checks"]["passed"]:
        problems.append("GBM training checks did not pass")
    if cv["config"]["test_set_used"] is not False:
        problems.append("cv_glm.json records test-set use")
    cur = json.loads(json.dumps(cfg))
    skip = set(cfg["freeze"]["evaluation_only_model_keys"])
    for key, rec in cv["config"]["models"].items():
        if key not in skip and cur["models"].get(key) != rec:
            problems.append(f"models.{key} differs from the GLM training record")
    scalars = {
        "seed": cur["seed"],
        "split_primary": cur["split"]["primary"],
        "cv_folds": cur["split"]["cv_folds"],
        "clean_targets": cur["clean"]["targets"],
        "large_loss_cap": cur["clean"]["large_loss"]["cap"],
        "excess_loading_mode": cur["clean"]["excess_loading"]["mode"],
    }
    for key, val in scalars.items():
        if cv["config"][key] != val:
            problems.append(f"{key} differs from the GLM training record")
    if cg["config"]["gbm"] != cur["models"]["gbm"]:
        problems.append("models.gbm differs from the GBM training record")
    power = cv["final"]["power_cv"]["power"]
    if cg["config"]["tweedie_power"] != power or inputs["gbm_models"]["tweedie_power"] != power:
        problems.append("GBM Tweedie power differs from the GLM final power")
    ref = cur["models"]["tweedie"]["reference_power"]
    if cg["config"]["reference_power"] != ref or cv["selection"]["primary_loss"]["power"] != ref:
        problems.append("reference power differs from the training record")
    if cur["split"]["primary"] != "grouped":
        problems.append("the pre-registered primary split is grouped")
    if problems:
        raise FreezeError("training record check failed: " + "; ".join(problems))
    return {
        "glm_checks_passed": True,
        "gbm_checks_passed": True,
        "glm_config_sha256_at_training": cv["config"]["config_sha256"],
        "evaluation_only_model_keys": sorted(skip),
        "tweedie_power": power,
        "reference_power": ref,
    }


# --------------------------------------------------------------------------- level coverage


def _expected_labels(name: str, spec: dict, fac: dict) -> set[str] | None:
    """Labels the full data can produce for a banded / capped feature (None: by count only)."""
    if spec["kind"] == "band":
        s = pd.Series([fac["min"], fac["max"]], name=name)
        lo, hi = features._band(s, spec["lower_edges"])
        labels = features._band_labels(spec["lower_edges"])
        return set(labels[labels.index(lo) : labels.index(hi) + 1])
    if fac["kind"] == "numeric":
        lo, hi = fac["min"], fac["max"]
        if not (float(lo).is_integer() and float(hi).is_integer()) or fac["levels"] != (
            hi - lo + 1
        ):
            raise FreezeError(f"{name}: cannot verify the level coverage from the audit summary")
        vals = np.arange(int(lo), int(hi) + 1)
        if spec.get("cap") is not None:
            vals = np.minimum(vals, spec["cap"])
        return {str(int(v)) for v in vals}
    return None


def check_level_coverage(inputs: dict[str, Any], fcfg: dict[str, Any]) -> dict[str, Any]:
    """Every level the full data can take (``results/audit.json``: counts and ranges only, no
    outcomes) is handled by the stored maps, so the fit-free transform cannot fail on an unseen
    level at evaluation time."""
    factors = inputs["audit"]["factors"]
    other = fcfg["rare_levels"]["other_label"]
    out: dict[str, Any] = {"glm_policy_map": {}, "gbm_encoding": {}}
    problems = []
    pol = inputs["feature_map"]["policy"]["levels"]
    sev = inputs["feature_map"]["severity"]["levels"]
    for name, spec in fcfg["definitions"].items():
        if spec["kind"] not in ("band", "categorical"):
            continue
        info, fac = pol[name], factors[name]
        known = set(info["levels"]) | set(info["rare"])
        if set(sev[name]["merge"]) != set(info["levels"]):
            problems.append(f"{name}: the severity map does not cover the policy levels")
        if spec.get("group_rare") and other in info["levels"]:
            out["glm_policy_map"][name] = "covered (unseen levels map to the other label)"
            continue
        expected = _expected_labels(name, spec, fac)
        if expected is None:
            ok = len(known) == fac["levels"]
        else:
            ok = expected <= known
        out["glm_policy_map"][name] = "covered" if ok else "NOT covered"
        if not ok:
            problems.append(f"{name}: full-data levels are missing from the policy feature map")
    enc = inputs["gbm_models"]["encoding"]
    for name in enc["categorical"]:
        ok = len(enc["levels"][name]) == factors[name]["levels"]
        if ok:
            out["gbm_encoding"][name] = "covered"
        elif enc["unseen_level"] == "missing":
            out["gbm_encoding"][name] = "unseen levels use the LightGBM missing branch"
        else:
            out["gbm_encoding"][name] = "NOT covered"
            problems.append(f"{name}: full-data levels are missing from the GBM encoding")
    if problems:
        raise FreezeError("level coverage check failed: " + "; ".join(problems))
    return out


# --------------------------------------------------------------------------- recipes


def _glm_component(models: dict, key: str, fmap: str, fcfg: dict, feature_set: str) -> dict:
    if key not in models:
        raise FreezeError(f"glm_models.json has no model {key!r}")
    names = [f["name"] for f in models[key]["features"]]
    if names != features.feature_names(fcfg, feature_set):
        raise FreezeError(f"{key}: stored features differ from feature set {feature_set!r}")
    return {
        "kind": "glm",
        "model_id": f"glm_models.json#{key}",
        "feature_map_id": f"feature_map.json#{fmap}",
        "features": names,
        "family": models[key]["family"],
        "power": models[key]["power"],
    }


def _gbm_component(gm: dict, comp: str, paths: dict) -> dict:
    meta = gm["models"][comp]
    return {
        "kind": "gbm",
        "model_id": f"gbm_models.json#models.{comp}",
        "encoding_id": "gbm_models.json#encoding",
        "file": f"{paths['model_dir']}/{meta['file']}",
        "sha256": meta["sha256"],
        "rounds": meta["rounds"],
        "objective": meta["params"]["objective"],
        "feature_names": meta["feature_names"],
        "monotone_constraints": meta["monotone_constraints"],
    }


def build_recipes(
    inputs: dict[str, Any], sel: dict[str, Any], cfg: dict[str, Any], paths: dict, null_rate: float
) -> dict[str, Any]:
    """Fit-free prediction recipe of each of the six candidates (protocol section 3)."""
    models, gm, fcfg = inputs["glm_models"], inputs["gbm_models"], cfg["features"]
    sev_key = f"severity_{sel['selected_glm_severity']}"
    const = models["severity_constant"]
    if const.get("kind") != "constant":
        raise FreezeError("glm_models.json: severity_constant is not a constant model")
    sev_const = {
        "kind": "constant",
        "model_id": "glm_models.json#severity_constant",
        "value": float(const["value"]),
    }
    ladder = {e["id"]: e for e in cfg["models"]["glm"]["severity_ladder"]}
    entry = ladder[sel["selected_glm_severity"]]
    if entry["kind"] == "constant":
        sev_sel = sev_const
    else:
        sev_sel = _glm_component(models, sev_key, "severity", fcfg, entry["feature_set"])
    power = inputs["cv_glm"]["final"]["power_cv"]["power"]
    if models["tweedie"]["power"] != power:
        raise FreezeError("glm_models.json: Tweedie power differs from the final power")
    glm_fxs = f"glm_freq_x_sev[{sel['selected_glm_severity']}]"
    return {
        PP_NULL: {
            "family": "freq_x_sev",
            "description": "training mean: constant frequency x constant severity",
            "frequency": {
                "kind": "constant",
                "model_id": "train mean freq_count / Exposure (full train set)",
                "value": float(null_rate),
            },
            "severity": sev_const,
        },
        PP_BM: {
            "family": "freq_x_sev",
            "description": "BonusMalus-only GLM frequency x constant severity",
            "frequency": _glm_component(
                models, "frequency_bonus_malus_only", "policy", fcfg, "bonus_malus_only"
            ),
            "severity": sev_const,
        },
        glm_fxs: {
            "family": "freq_x_sev",
            "description": "full GLM frequency x selected GLM severity candidate",
            "frequency": _glm_component(models, "frequency", "policy", fcfg, "full"),
            "severity": sev_sel,
        },
        GLM_TW: {
            "family": "tweedie",
            "description": "GLM Tweedie on S_cap / Exposure, weight Exposure",
            "pure_premium": _glm_component(models, "tweedie", "policy", fcfg, "full"),
            "tweedie_power": power,
        },
        GBM_FXS: {
            "family": "freq_x_sev",
            "description": "LightGBM Poisson frequency x LightGBM Gamma severity",
            "frequency": _gbm_component(gm, "frequency", paths),
            "severity": _gbm_component(gm, "severity", paths),
        },
        GBM_TW: {
            "family": "tweedie",
            "description": "LightGBM Tweedie on S_cap / Exposure, weight Exposure",
            "pure_premium": _gbm_component(gm, "tweedie", paths),
            "tweedie_power": gm["tweedie_power"],
        },
    }


# --------------------------------------------------------------------------- prediction


def load_artifacts(
    root: Path, paths: dict[str, Any], inputs: dict[str, Any], num_threads: int = 1
) -> dict[str, Any]:
    """Stored parameters only: GLM dicts, feature maps, hashed LightGBM files (no fitting)."""
    fm = inputs["feature_map"]
    gm = inputs["gbm_models"]
    boosters = {
        comp: gbm.load_frozen_booster(root / paths["model_dir"] / meta["file"], meta)
        for comp, meta in gm["models"].items()
    }
    return {
        "glm_models": inputs["glm_models"],
        "feature_maps": {k: features.FeatureMap.from_dict(v) for k, v in fm.items()},
        "boosters": boosters,
        "encoding": gm["encoding"],
        "num_threads": int(num_threads),
    }


def _component_rate(spec: dict, df: pd.DataFrame, art: dict, cache: dict) -> np.ndarray:
    if spec["kind"] == "constant":
        return np.full(len(df), float(spec["value"]))
    if spec["kind"] == "glm":
        fmap = spec["feature_map_id"].split("#", 1)[1]
        if fmap not in cache:
            cache[fmap] = features.transform(df, art["feature_maps"][fmap])
        key = spec["model_id"].split("#", 1)[1]
        model = glm.GLMModel.from_dict(art["glm_models"][key])
        return model.predict_rate_rowwise(cache[fmap][spec["features"]])
    if spec["kind"] == "gbm":
        comp = spec["model_id"].rsplit(".", 1)[1]
        return gbm.predict_annual_rate(
            art["boosters"][comp], df, art["encoding"], num_threads=art["num_threads"]
        )
    raise FreezeError(f"unknown component kind {spec['kind']!r}")


def predict(recipes: dict[str, Any], art: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Annual rates per candidate: pure_premium, and frequency / severity for freq x sev
    (BLAS/OpenMP threads limited to ``art['num_threads']``)."""
    with threadpool_limits(art["num_threads"]):
        return _predict(recipes, art, df)


def _predict(recipes: dict[str, Any], art: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    cache: dict[str, pd.DataFrame] = {}
    out: dict[str, Any] = {}
    for cid, rec in recipes.items():
        if rec["family"] == "freq_x_sev":
            f = _component_rate(rec["frequency"], df, art, cache)
            s = _component_rate(rec["severity"], df, art, cache)
            out[cid] = {"pure_premium": f * s, "frequency": f, "severity": s}
        else:
            out[cid] = {"pure_premium": _component_rate(rec["pure_premium"], df, art, cache)}
        for k, v in out[cid].items():
            if not (np.isfinite(v).all() and (v > 0).all()):
                raise FreezeError(f"{cid}: non-finite or non-positive {k} predictions")
    return out


# --------------------------------------------------------------------------- self-check


def self_check_rows(train: pd.DataFrame, sc: dict[str, Any]) -> pd.DataFrame:
    """Fixed sample of train rows (sorted by IDpol): ``rows`` drawn by seed, plus edge cases, the
    rows with the min and max of every ``extreme_columns`` column and one row (lowest IDpol) of
    every level of every ``level_columns`` column present in train."""
    assert_train_only(train)
    t = train.sort_values("IDpol", kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(sc["seed"])
    n = min(int(sc["rows"]), len(t))
    pos = set(rng.choice(len(t), size=n, replace=False).tolist())
    for col in sc.get("extreme_columns", []):
        v = t[col].to_numpy(dtype=float)
        pos.update((int(np.argmin(v)), int(np.argmax(v))))
    for col in sc.get("level_columns", []):
        pos.update(int(i) for i in t.groupby(t[col].astype(str), sort=True).head(1).index)
    return t.iloc[sorted(pos)].reset_index(drop=True)


def edge_case_summary(sample: pd.DataFrame, train: pd.DataFrame, sc: dict[str, Any]) -> dict:
    """Which edge cases the sample covers (every one must be covered)."""
    out: dict[str, Any] = {"extremes": {}, "levels": {}}
    for col in sc.get("extreme_columns", []):
        lo, hi = float(train[col].min()), float(train[col].max())
        ok = bool(sample[col].min() == lo and sample[col].max() == hi)
        out["extremes"][col] = {"min": lo, "max": hi, "covered": ok}
    for col in sc.get("level_columns", []):
        want = set(train[col].astype(str))
        ok = want <= set(sample[col].astype(str))
        out["levels"][col] = {"levels": len(want), "covered": ok}
    bad = [c for part in out.values() for c, v in part.items() if not v["covered"]]
    if bad:
        raise FreezeError(f"self-check sample misses edge cases of {bad}")
    return out


def determinism_check(recipes, art, rows: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Predicting twice, and on a permuted row order, must give bit-identical predictions."""
    first = predict(recipes, art, rows)
    again = predict(recipes, art, rows)
    perm = np.random.default_rng(seed).permutation(len(rows))
    shuffled = predict(recipes, art, rows.iloc[perm].reset_index(drop=True))
    inv = np.argsort(perm)
    for cid, comps in first.items():
        for k, v in comps.items():
            if not np.array_equal(v, again[cid][k]):
                raise FreezeError(f"determinism: {cid} {k} differs between two predictions")
            if not np.array_equal(v, shuffled[cid][k][inv]):
                raise FreezeError(f"determinism: {cid} {k} depends on the row order")
    return {"repeat": "identical", "permuted_row_order": "identical", "rows": int(len(rows))}


def prediction_record(df: pd.DataFrame, preds: dict[str, Any]) -> dict[str, Any]:
    return {
        "IDpol": [int(x) for x in df["IDpol"]],
        "predictions": {
            cid: {k: [float(x) for x in v] for k, v in p.items()} for cid, p in preds.items()
        },
    }


def verify_self_check(stored: dict[str, Any], train: pd.DataFrame, recipes, art, rtol: float):
    """Reproduce the stored train predictions from the loaded artifacts (before test access)."""
    assert_train_only(train)
    by_id = train.set_index("IDpol", drop=False)
    missing = set(stored["IDpol"]) - set(by_id.index)
    if missing:
        raise FreezeError(f"self-check: {len(missing)} stored train rows are not in the train set")
    rows = by_id.loc[stored["IDpol"]].reset_index(drop=True)
    preds = predict(recipes, art, rows)
    det = determinism_check(recipes, art, rows, int(stored["permutation_seed"]))
    worst = 0.0
    for cid, comps in stored["predictions"].items():
        for k, vals in comps.items():
            ref = np.asarray(vals, dtype=float)
            diff = float(np.max(np.abs(preds[cid][k] - ref) / np.abs(ref)))
            worst = max(worst, diff)
            if not diff <= rtol:
                raise FreezeError(
                    f"self-check: {cid} {k} predictions differ from the stored ones "
                    f"(max rel diff {diff:.3g} > {rtol})"
                )
    return {
        "rows": len(rows),
        "max_rel_diff": worst,
        "rtol": rtol,
        "determinism": det,
        "passed": True,
    }


# --------------------------------------------------------------------------- manifest


def evaluation_spec(cfg: dict[str, Any], sel: dict[str, Any], inputs: dict[str, Any]) -> dict:
    mc, b = cfg["models"], cfg["models"]["bootstrap"]
    ref = mc["tweedie"]["reference_power"]
    level = float(b["level"])
    if b["interval"] != "two_sided":
        raise FreezeError("models.bootstrap.interval: only two_sided is implemented")
    return {
        "population": {
            "rows": "primary_set == 'test' (cv_fold == -1) of the frozen grouped split",
            "split": cfg["split"]["primary"],
            "test_fold": cfg["split"]["test_fold"],
            "targets": "primary definitions only (clean without a sensitivity)",
        },
        "primary_loss": {
            "name": "exposure-weighted Tweedie deviance of Y = S_cap / Exposure",
            "formula": "L_m = sum_i E_i d_p(Y_i, mu_mi) / sum_i E_i",
            "power": ref,
            "power_key": "models.tweedie.reference_power",
        },
        "primary_contrast": {
            "delta": "L(selected_gbm_id) - L(selected_glm_id)",
            "gbm": sel["selected_gbm_id"],
            "glm": sel["selected_glm_id"],
            "secondary": "normalised Gini difference (selected GBM - selected GLM)",
            "other_pairs": "descriptive only",
        },
        "decision": {
            "improvement": "upper endpoint of the Delta interval < 0",
            "deterioration": "lower endpoint of the Delta interval > 0",
            "otherwise": "no demonstrated difference",
        },
        "bootstrap": {
            "unit": b["unit"],
            "unit_column": "group_id",
            "sensitivity_unit": b["sensitivity_unit"],
            "sensitivity_unit_column": "IDpol",
            "replicates": int(b["replicates"]),
            "seed": int(b["seed"]),
            "level": level,
            "interval": "two-sided percentile interval",
            "percentiles": [(1 - level) / 2, (1 + level) / 2],
            "quantile_function": "numpy.nanquantile",
            "quantile_method": b["quantile_method"],
            "multiplicities": "per replicate, as many units as there are drawn uniformly with "
            "replacement (numpy default_rng(seed).integers); the same draws for every model",
            "statistic": "Delta^(b) = sum_g k_g (A_g,GBM - A_g,GLM) / sum_g k_g E_g, "
            "A_gm = sum_i e_i d_p(y_i, mu_mi); denominators recomputed per replicate; Gini "
            "rankings and oracle normalisation recomputed with the multiplicity weights",
            "conditional_on": "the frozen fitted models; no fit, tuning or selection inside "
            "replicates",
        },
        "metrics": {
            "per_candidate": [
                "primary loss",
                "Tweedie deviance per exposure at the selected power",
                "Poisson deviance per exposure of the pure premium (Poisson-form score)",
                "raw and normalised Gini of the pure premium",
                "Poisson deviance per exposure, Gini and totals of frequency components",
                "Gamma deviance per claim (weight N) and totals of severity components",
                "calibration totals, by equal-exposure decile (lift) and by factor",
                "Lorenz points",
            ],
            "double_lift": "selected GBM vs selected GLM",
            "loaded_premium": "reported separately; mu + training-only flat excess loading",
            "evaluation_config": mc["evaluation"],
            "group_summary_config": mc["group_summary"],
        },
        "excess_loading_per_exposure_year": float(inputs["cv_glm"]["excess_loading_train"]),
        "nonfinite_policy": "fail closed: a non-finite or non-positive prediction, a non-finite "
        "loss or metric, a zero or non-finite denominator, or a non-finite bootstrap replicate "
        "aborts the attempt; nothing is dropped",
        "technical_rerun_policy": " ".join(cfg["freeze"]["technical_rerun"]["policy"].split()),
        "recovery_policy": " ".join(cfg["freeze"]["recovery"]["policy"].split()),
        "rehearsal": {
            "population": "train rows with cv_fold == freeze.rehearsal.fold (never cv_fold < 0)",
            "fold": int(cfg["freeze"]["rehearsal"]["fold"]),
            "note": "identical evaluate code path; outputs under paths.processed only",
        },
    }


def build_manifest(
    cfg: dict[str, Any], config_path: str | Path, root: Path | None = None, train=None
) -> dict[str, Any]:
    """The manifest (nothing is written). ``train`` defaults to ``train.load_train(cfg)``."""
    root = Path.cwd() if root is None else Path(root)
    paths = project_paths(cfg, root)
    inputs = read_inputs(root, paths)
    sel = recompute_selections(inputs, cfg)
    record = check_training_record(inputs, cfg)
    coverage = check_level_coverage(inputs, cfg["features"])
    train = load_train(cfg) if train is None else train
    assert_train_only(train)
    rows = inputs["cv_glm"]["rows"]
    e_sum = float(train["Exposure"].sum())
    if len(train) != rows["train_policies"] or int(train["N"].sum()) != rows["train_claims"]:
        raise FreezeError("the train rows differ from the training record")
    if abs(e_sum - rows["train_exposure"]) > 1e-9 * rows["train_exposure"]:
        raise FreezeError("the train exposure differs from the training record")
    null_rate = float(train["freq_count"].sum() / e_sum)
    recipes = build_recipes(inputs, sel, cfg, paths, null_rate)
    if (
        len(recipes) != 6
        or sel["selected_glm_id"] not in recipes
        or (sel["selected_gbm_id"] not in recipes)
    ):
        raise FreezeError("the six candidates do not contain the selected ids")
    art = load_artifacts(root, paths, inputs, cfg["freeze"]["runtime"]["num_threads"])
    sc = cfg["freeze"]["self_check"]
    sample = self_check_rows(train, sc)
    edges = edge_case_summary(sample, train, sc)
    stored = prediction_record(sample, predict(recipes, art, sample))
    det = determinism_check(recipes, art, sample, int(sc["permutation_seed"]))
    config_rel = Path(config_path).as_posix()
    files = [
        file_entry(root, paths["cv_glm"], "glm cv results and selection trace"),
        file_entry(root, paths["glm_models"], "glm final models"),
        file_entry(root, paths["feature_map"], "glm feature maps"),
        file_entry(root, paths["cv_gbm"], "gbm cv results and selection trace"),
        file_entry(root, paths["gbm_models"], "gbm final model metadata"),
        *[
            file_entry(root, f"{paths['model_dir']}/{m['file']}", f"gbm model {comp}")
            for comp, m in inputs["gbm_models"]["models"].items()
        ],
        file_entry(root, paths["audit"], "full-data factor domains (coverage check)"),
        file_entry(root, config_rel, "config"),
        file_entry(root, paths["protocol"], "evaluation protocol"),
        file_entry(root, paths["split"], "frozen split"),
        *[file_entry(root, f, "project file") for f in paths["project_files"]],
        *[file_entry(root, f, "raw data") for f in paths["raw"]],
    ]
    recorded = {
        f"{paths['model_dir']}/{m['file']}": m["sha256"]
        for m in inputs["gbm_models"]["models"].values()
    }
    for f in files:
        if f["path"] in recorded and f["sha256"] != recorded[f["path"]]:
            raise FreezeError(f"{f['path']}: sha256 differs from gbm_models.json")
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "test_set_used": False,
        "training_commit": head_commit(root),
        "config_path": config_rel,
        "candidates": list(recipes),
        "selected_glm_id": sel["selected_glm_id"],
        "selected_gbm_id": sel["selected_gbm_id"],
        "selection": sel,
        "recipes": recipes,
        "scale_conventions": SCALE_CONVENTIONS,
        "training_record": record,
        "level_coverage": coverage,
        "evaluation": evaluation_spec(cfg, sel, inputs),
        "files": files,
        "environment": environment(cfg["freeze"]["environment_packages"]),
        "runtime_identity": runtime_identity(cfg, root),
        "self_check": {
            **sc,
            "edge_cases": edges,
            "determinism_at_freeze": det,
            **stored,
        },
    }


def has_test_state(root: Path, paths: dict[str, Any]) -> bool:
    return (root / paths["state"]).exists() or (root / paths["mirror"]).exists()


def run_freeze(
    cfg: dict[str, Any], config_path: str | Path, force: bool = False, root: Path | None = None
) -> dict[str, Any]:
    """``fremtpl2 freeze``: clean tree, no test access yet; writes the manifest."""
    root = Path.cwd() if root is None else Path(root)
    paths = project_paths(cfg, root)
    if has_test_state(root, paths):
        raise FreezeError("a test-state file exists: the test set was accessed; freeze refused")
    target = root / paths["manifest"]
    if target.exists() and not force:
        raise FreezeError(f"{paths['manifest']} exists; use --force (only before test access)")
    # --force replaces an existing manifest (before any test access); it may be uncommitted
    assert_clean_tree(root, frozenset({paths["manifest"]}) if force else frozenset())
    manifest = build_manifest(cfg, config_path, root)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(target)
    return manifest
