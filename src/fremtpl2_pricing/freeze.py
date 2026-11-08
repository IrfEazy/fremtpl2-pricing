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
import fnmatch
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
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


def git_blob(root: Path, rev: str, rel: str) -> bytes:
    """Bytes of ``rel`` at revision ``rev`` (the committed original)."""
    try:
        out = subprocess.run(
            ["git", "show", f"{rev}:{rel}"], cwd=root, capture_output=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreezeError(f"git show {rev}:{rel} failed") from exc
    return out.stdout


def git_tracked(root: Path, rev: str = "HEAD") -> set[str]:
    """Paths tracked at revision ``rev``."""
    return set(git(root, "ls-tree", "-r", "--name-only", "-z", rev).split("\0")) - {""}


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


# --------------------------------------------------------------------------- strict check


@dataclass
class Comparison:
    """Result of :func:`compare_values`. ``numeric`` mismatches are float differences beyond the
    tolerance; every other difference (keys, types, ints, strings, list lengths, ``exact_keys``
    subtrees) is ``structural`` and is never tolerated."""

    mismatches: list[dict[str, Any]] = field(default_factory=list)
    n_numeric: int = 0
    n_structural: int = 0
    n_within_tolerance: int = 0
    max_abs: float = 0.0
    max_rel: float = 0.0
    limit: int = 10

    @property
    def ok(self) -> bool:
        return self.n_numeric + self.n_structural == 0

    @property
    def numeric_only(self) -> bool:
        return self.n_structural == 0

    def add(self, path: str, kind: str, base: Any, new: Any) -> None:
        if kind == "numeric":
            self.n_numeric += 1
        else:
            self.n_structural += 1
        if len(self.mismatches) < self.limit:
            self.mismatches.append(
                {"path": path, "kind": kind, "base": _short(base), "new": _short(new)}
            )

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "numeric_only": self.numeric_only,
            "n_numeric": self.n_numeric,
            "n_structural": self.n_structural,
            "n_within_tolerance": self.n_within_tolerance,
            "max_abs": self.max_abs,
            "max_rel": self.max_rel,
            "mismatches": self.mismatches,
        }


def _short(v: Any) -> str:
    text = repr(v)
    return text if len(text) <= 80 else text[:77] + "..."


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def strict_json_loads(text: str) -> Any:
    """``json.loads`` that rejects duplicate object keys (``ValueError``). Non-finite numbers are
    parsed, not accepted silently: :func:`compare_values` treats a non-finite value as a
    difference unless the committed value is identical."""
    return json.loads(text, object_pairs_hook=_no_duplicates)


def compare_values(
    base: Any,
    new: Any,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    drop: frozenset[str] | set[str] = frozenset(),
    exact_keys: frozenset[str] | set[str] = frozenset(),
    numeric_paths: Sequence[str] | None = None,
    limit: int = 10,
) -> Comparison:
    """Compare two parsed JSON values; ``base`` is the committed reference. Dict keys in ``drop``
    are ignored at any depth (matched by name). The shape (keys, types, list lengths) is compared
    first; only if it is identical are values compared: floats may differ within
    ``|a - b| <= atol + rtol * |a|`` with ``a`` the committed value (strict mode: zero
    tolerance), non-finite floats only if identical; ints, bools, strings and everything below a
    key in ``exact_keys`` (descendants included) must be equal. If
    ``numeric_paths`` (fnmatch patterns on the ``/``-joined path) is given, a float difference
    is only classed numeric (and only tolerated) at a matching path; anywhere else it is
    structural."""
    out = Comparison(limit=limit)

    def shape(a: Any, b: Any, path: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            ka, kb = set(a) - drop, set(b) - drop
            for k in sorted(ka - kb):
                out.add(f"{path}/{k}", "structural", a[k], "<missing>")
            for k in sorted(kb - ka):
                out.add(f"{path}/{k}", "structural", "<missing>", b[k])
            for k in sorted(ka & kb):
                shape(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                out.add(path, "structural", f"list[{len(a)}]", f"list[{len(b)}]")
                return
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                shape(x, y, f"{path}[{i}]")
        elif type(a) is not type(b):
            out.add(path, "structural", a, b)

    def walk(a: Any, b: Any, path: str, tolerant: bool) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            ka, kb = set(a) - drop, set(b) - drop
            for k in sorted(ka - kb):
                out.add(f"{path}/{k}", "structural", a[k], "<missing>")
            for k in sorted(kb - ka):
                out.add(f"{path}/{k}", "structural", "<missing>", b[k])
            for k in sorted(ka & kb):
                walk(a[k], b[k], f"{path}/{k}", tolerant and k not in exact_keys)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                out.add(path, "structural", f"list[{len(a)}]", f"list[{len(b)}]")
                return
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                walk(x, y, f"{path}[{i}]", tolerant)
        elif type(a) is float and type(b) is float:
            if a == b or (math.isnan(a) and math.isnan(b)):
                return
            if not (math.isfinite(a) and math.isfinite(b)):
                out.add(path, "structural", a, b)
                return
            diff = abs(a - b)
            rel = diff / max(abs(a), abs(b))
            out.max_abs, out.max_rel = max(out.max_abs, diff), max(out.max_rel, rel)
            numeric = tolerant and (
                numeric_paths is None or any(fnmatch.fnmatchcase(path, p) for p in numeric_paths)
            )
            if numeric and diff <= atol + rtol * abs(a):
                out.n_within_tolerance += 1
            else:
                out.add(path, "numeric" if numeric else "structural", a, b)
        elif type(a) is not type(b) or a != b:
            out.add(path, "structural", a, b)

    shape(base, new, "")
    if out.n_structural:  # a different shape is never compared numerically
        return out
    walk(base, new, "", True)
    return out


def _check(name: str, cmp: Comparison | None = None, problems: list[str] | None = None,
           numeric: bool = False) -> dict[str, Any]:  # fmt: skip
    """One entry of the structured result. ``cmp`` and/or free-text ``problems`` (each counted as
    one mismatch; structural unless ``numeric``)."""
    problems = problems or []
    res: dict[str, Any] = cmp.summary() if cmp else {
        "ok": True, "numeric_only": True, "n_numeric": 0, "n_structural": 0,
        "n_within_tolerance": 0, "max_abs": 0.0, "max_rel": 0.0, "mismatches": [],
    }  # fmt: skip
    for text in problems:
        res["ok"] = False
        res["numeric_only"] = res["numeric_only"] and numeric
        res["n_numeric" if numeric else "n_structural"] += 1
        res["mismatches"].append({"path": text, "kind": "numeric" if numeric else "structural"})
    return {"name": name, "status": "pass" if res["ok"] else "fail", **res}


def _manifest_groups(manifest: dict[str, Any], run_specific: set[str]) -> dict[str, Any]:
    groups = {
        "selection": ("candidates", "selected_glm_id", "selected_gbm_id", "selection"),
        "recipes": ("recipes",),
        "self_check": ("self_check",),
    }
    used = {k for keys in groups.values() for k in keys} | {"files"} | run_specific
    out = {name: {k: manifest.get(k) for k in keys} for name, keys in groups.items()}
    out["manifest_fields"] = {k: v for k, v in manifest.items() if k not in used}
    out["file_list"] = [(f["path"], f["role"]) for f in manifest.get("files", [])]
    return out


def _evidence_inventory(root: Path, paths: dict[str, Any], events: list[dict[str, Any]],
                        complete: dict[str, Any]) -> list[str]:  # fmt: skip
    """Exact inventory of the published evidence against the COMPLETE event: one COMPLETE and
    unique attempt identities, no symlinked or escaping artifact, a pointer that resolves inside
    the attempts directory, and exactly the staged outputs in the attempt directory (neither
    missing nor unexpected entries)."""
    problems = []
    real_root = root.resolve()
    done = [e for e in events if e["event"] == "COMPLETE"]
    if len(done) != 1:
        problems.append(f"{len(done)} COMPLETE events (exactly one is allowed)")
    for key in ("attempt_id", "attempt"):
        values = [e[key] for e in events if e["event"] == "TEST_OPENED" and key in e]
        if len(values) != len(set(values)):
            problems.append(f"duplicate attempt identity ({key}) in the journal")
    attempts = (root / paths["attempts"]).resolve()
    adir = root / complete["publish_dir"]
    if (
        adir.is_symlink()
        or not adir.resolve().is_relative_to(attempts)
        or adir.resolve() == attempts
    ):
        problems.append(f"{complete['publish_dir']}: resolves outside {paths['attempts']}")
        return problems
    for rel in (paths["state"], paths["pointer"]):
        f = root / rel
        if f.is_symlink() or not f.resolve().is_relative_to(real_root):
            problems.append(f"{rel}: symlinked or escaping artifact path")
    found = set()
    for f in sorted(adir.rglob("*")):
        rel = f.relative_to(adir).as_posix()
        if f.is_symlink():
            problems.append(f"{complete['publish_dir']}/{rel}: symlinked artifact")
        elif f.is_file():
            found.add(rel)
    staged = set(complete["staged"])
    for name in sorted(staged - found):
        problems.append(f"{complete['publish_dir']}/{name}: missing")
    for name in sorted(found - staged):
        problems.append(f"{complete['publish_dir']}/{name}: unexpected entry (not in COMPLETE)")
    return problems


def _evidence_problems(cfg: dict[str, Any], root: Path, paths: dict[str, Any]) -> list[str]:
    """Verify the published evaluation with the report loader (journal COMPLETE, pointer
    resolution, every attempt output against the COMPLETE hashes) and its exact inventory.
    Nothing to verify, and no problem, before any test access."""
    if not has_test_state(root, paths):
        return []
    from .evaluate import real_sink  # local imports: both import this module
    from .report import _journal_events, load_completed

    try:
        data = load_completed(cfg, root)
        events, _ = _journal_events(real_sink(root, paths))
        return _evidence_inventory(root, paths, events, data["complete"])
    except (FreezeError, ValueError, OSError, KeyError, TypeError) as exc:
        return [f"published evidence: {exc}"]


def evidence_identity(cfg: dict[str, Any], root: Path) -> dict[str, Any] | None:
    """Identities of the preserved published evidence (journal, pointer, attempt, manifest);
    ``None`` before any test access. Preserved, not recomputed."""
    paths = project_paths(cfg, root)
    if not has_test_state(root, paths):
        return None
    out: dict[str, Any] = {}
    try:
        for key, rel in (("journal", "state"), ("pointer", "pointer"), ("manifest", "manifest")):
            out[f"{key}_sha256"] = sha256_file(root / paths[rel])
        from .report import load_completed

        complete = load_completed(cfg, root)["complete"]
        out.update(attempt_id=complete["attempt_id"], attempt=complete["attempt"])
    except (FreezeError, ValueError, OSError, KeyError, TypeError) as exc:
        out["error"] = str(exc)
    return out


def run_freeze_check(
    cfg: dict[str, Any], config_path: str | Path, root: Path | None = None
) -> dict[str, Any]:
    """``fremtpl2 freeze --check``: strict, read-only verification of the committed manifest.

    Writes nothing and does not need a clean tree. Never reads a test row. Returns a structured
    result with one entry per check; ``passed`` is true only if every check passes; ``numeric_only``
    tells whether every failure is a numeric one (float differences, bytes of regenerated model
    files), which ``reproduce`` may accept under its declared tolerance."""
    root = Path.cwd() if root is None else Path(root)
    paths = project_paths(cfg, root)
    chk = cfg["freeze"]["check"]
    limit = int(chk["max_mismatches"])
    run_specific = set(chk["run_specific_keys"])
    volatile = frozenset(chk["volatile_keys"])
    exact_keys = frozenset(chk["exact_keys"])
    numeric_paths = list(chk["numeric_paths"])
    telemetry = {f"{paths['results']}/{f}" for f in chk["telemetry_files"]}
    checks: list[dict[str, Any]] = []
    mpath = root / paths["manifest"]
    if not mpath.is_file():
        raise FreezeError(f"missing manifest: {paths['manifest']}")
    manifest = strict_json_loads(mpath.read_text(encoding="utf-8"))
    tracked = git_tracked(root)
    entries = manifest["files"]

    # (1) original-byte integrity: the committed bytes are the frozen bytes
    problems = []
    if paths["manifest"] not in tracked:
        problems.append(f"{paths['manifest']}: the manifest is not tracked at HEAD")
    elif git_blob(root, "HEAD", paths["manifest"]) != mpath.read_bytes():
        problems.append(f"{paths['manifest']}: differs from the committed manifest")
    for e in entries:
        if e["path"] in tracked:
            blob = git_blob(root, "HEAD", e["path"])
            if hashlib.sha256(blob).hexdigest() != e["sha256"] or len(blob) != e["bytes"]:
                problems.append(f"{e['path']}: HEAD blob differs from the manifest hash")
    checks.append(_check("original_bytes", problems=problems))

    # (2) the manifest rebuilt in memory equals the committed one (run-specific keys excluded)
    try:
        rebuilt = build_manifest(cfg, config_path, root)
    except (FreezeError, OSError, KeyError, ValueError) as exc:
        checks.append(_check("rebuild", problems=[f"manifest rebuild failed: {exc}"]))
        rebuilt = None
    if rebuilt is not None:
        base = _manifest_groups(manifest, run_specific)
        new = _manifest_groups(rebuilt, run_specific)
        for name in base:
            # float differences are numeric only where config declares it (losses, constants,
            # stored predictions); ids, hyperparameters and every non-float are structural
            cmp = compare_values(
                base[name],
                new[name],
                exact_keys=exact_keys,
                numeric_paths=numeric_paths if name != "manifest_fields" else [],
                limit=limit,
            )
            checks.append(_check(name, cmp))

    # (3) the hashed files
    wrong, regenerated = [], []
    model_prefix = paths["model_dir"] + "/"
    for e in entries:
        path = root / e["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            wrong.append(f"symlinked or escaping artifact path: {e['path']}")
        elif not path.is_file():
            wrong.append(f"missing file: {e['path']}")
        elif e["path"] in telemetry and e["path"] in tracked:
            try:
                cmp = compare_values(
                    strict_json_loads(git_blob(root, "HEAD", e["path"]).decode("utf-8")),
                    strict_json_loads(path.read_text(encoding="utf-8")),
                    drop=volatile,
                    exact_keys=exact_keys,
                    limit=limit,
                )
            except ValueError:
                wrong.append(f"{e['path']}: not valid JSON")
                continue
            checks.append(_check(f"telemetry:{e['path']}", cmp))
        elif sha256_file(path) != e["sha256"] or path.stat().st_size != e["bytes"]:
            (regenerated if e["path"].startswith(model_prefix) else wrong).append(
                f"sha256 mismatch: {e['path']}"
            )
    checks.append(_check("files", problems=wrong))
    checks.append(
        _check("model_files", problems=regenerated)
    )  # byte-different models block portability

    # (4) the published test evidence (journal COMPLETE, pointer, attempt hashes), read-only
    checks.append(_check("published_evidence", problems=_evidence_problems(cfg, root, paths)))

    failed = [c for c in checks if c["status"] == "fail"]
    return {
        "mode": "freeze-check",
        "passed": not failed,
        "numeric_only": all(c["numeric_only"] for c in failed),
        "tolerance": {"rtol": 0.0, "atol": 0.0},
        "manifest": paths["manifest"],
        "head": head_commit(root),
        "checks": checks,
    }
