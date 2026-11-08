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

import copy
import datetime as dt
import fnmatch
import hashlib
import importlib.metadata
import json
import math
import platform
import re
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
from .train import assert_train_only, config_summary, load_train

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
    excepted; an entry ending in ``/`` ignores everything below that directory)."""
    status = git(root, "status", "--porcelain", "--untracked-files=normal")
    dirs = tuple(i for i in ignore if i.endswith("/"))

    def ignored(path: str) -> bool:
        return path in ignore or path.startswith(dirs)

    lines = [ln for ln in status.splitlines() if ln.strip() and not ignored(ln[3:].strip('"'))]
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


# --------------------------------------------------------------------------- declared deviations
#
# Post-hoc amendment of the verification policy (configs/reproduce.yaml ``amendments``): a deviation
# is a pinned, artifact- and field-path-specific exception that is verified from the actual files on
# every run. Nothing is ignored by key name and nothing is expanded at runtime: an unmatched
# difference stays a failure. ``freeze --check`` keeps its strict result; it only reports which
# failures a satisfied deviation covers.

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def is_iso_date(value: Any) -> bool:
    """A ``YYYY-MM-DD`` string that is a real calendar date (no other ISO 8601 form)."""
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        return False
    try:
        dt.date.fromisoformat(value)
    except ValueError:
        return False
    return True


def main_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """The main config as ``fremtpl2 train --config`` loaded it: without the verification-policy
    keys that ``with_reproduce`` merges in (their presence would change the config hash)."""
    out = copy.deepcopy(cfg)
    for key in ("reproduce", "amendments", "reproduce_policy_sha256"):
        out.pop(key, None)
    if isinstance(out.get("freeze"), dict):
        out["freeze"].pop("check", None)
    return out


def _ptr_get(obj: Any, ptr: str) -> Any:
    for part in [p for p in ptr.split("/") if p]:
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def _ptr_set(obj: Any, ptr: str, value: Any) -> None:
    parts = [p for p in ptr.split("/") if p]
    parent = _ptr_get(obj, "/".join(parts[:-1]))
    parent[int(parts[-1]) if isinstance(parent, list) else parts[-1]] = value


def _key_diff(committed: Any, new: Any, prefix: str = "") -> tuple[list[str], list[str]]:
    """Dotted key paths present only in ``new`` (added) and only in ``committed`` (removed),
    recursing into mappings present on both sides; an added mapping is reported at its root."""
    added: list[str] = []
    removed: list[str] = []
    if isinstance(committed, dict) and isinstance(new, dict):
        for key in sorted(set(new) - set(committed)):
            added.append(f"{prefix}{key}")
        for key in sorted(set(committed) - set(new)):
            removed.append(f"{prefix}{key}")
        for key in sorted(set(committed) & set(new)):
            a, r = _key_diff(committed[key], new[key], f"{prefix}{key}.")
            added += a
            removed += r
    return added, removed


def _restrict(current: Any, committed: Any) -> Any:
    """``current`` restricted to the keys of ``committed`` (recursively over mappings; lists and
    scalars unchanged, so their values and structure must then be equal)."""
    if isinstance(current, dict) and isinstance(committed, dict):
        return {k: _restrict(current[k], committed[k]) for k in committed if k in current}
    return current


def scan_readers(spec: dict[str, Any], root: Path) -> dict[str, list[str]]:
    """Static evidence: for each key, the modules of ``spec['source_dir']`` whose text matches the
    key's declared pattern (the config-read idiom), as a sorted list of module names."""
    src = root / spec["source_dir"]
    texts = {f.stem: f.read_text(encoding="utf-8") for f in sorted(src.glob("*.py"))}
    return {
        key: sorted(m for m, t in texts.items() if re.search(item["pattern"], t))
        for key, item in spec["keys"].items()
    }


def _new_result(dev: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": dev["id"], "kind": dev["kind"], "satisfied": True, "problems": [], "applied": [],
        "covers_checks": {}, "covers_files": [],
    }  # fmt: skip


def _verify_download_metadata(dev: dict[str, Any], root: Path, manifest: dict[str, Any]):
    """D1: the download metadata differs from the frozen bytes only in the pinned ``retrieved``
    fields, whose new values are ISO dates; re-serialised with the committed values it hashes to
    the frozen entry, so every other field is exact. The raw data files stay byte-identical."""
    res = _new_result(dev)
    art = dev["artifact"]
    entries = {e["path"]: e for e in manifest["files"]}
    file = root / art
    if art not in entries or not file.is_file():
        return res  # a missing or unlisted file is a failure no deviation covers
    entry = entries[art]
    if file.stat().st_size == entry["bytes"] and sha256_file(file) == entry["sha256"]:
        return res  # no difference to excuse
    problems = res["problems"]
    try:
        doc = strict_json_loads(file.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        problems.append(f"{art}: not valid JSON ({exc})")
        return res
    rebuilt = copy.deepcopy(doc)
    for field_ in dev["fields"]:
        try:
            value = _ptr_get(doc, field_["path"])
        except (KeyError, IndexError, ValueError, TypeError):
            problems.append(f"{art}{field_['path']}: field not found")
            continue
        if not is_iso_date(value):
            problems.append(f"{art}{field_['path']}: {value!r} is not an ISO date (YYYY-MM-DD)")
            continue
        _ptr_set(rebuilt, field_["path"], field_["committed"])
        res["applied"].append(
            {"place": art, "path": field_["path"], "base": field_["committed"], "new": value}
        )
    if not problems:
        # same serialisation as manifest.write_manifest
        blob = (json.dumps(rebuilt, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if len(blob) != entry["bytes"] or hashlib.sha256(blob).hexdigest() != entry["sha256"]:
            problems.append(f"{art}: differs from the frozen file outside the pinned fields")
    for rel in dev["raw_files"]:
        if rel not in entries:
            problems.append(f"{rel}: not in the frozen manifest")
        else:
            problems += verify_files(root, [entries[rel]], set())
    if problems:
        res["satisfied"] = False
    else:
        res["covers_checks"] = {"files": [f"sha256 mismatch: {art}"]}
    return res


def _verify_glm_config_echo(dev, cfg: dict[str, Any], root: Path, manifest: dict[str, Any]):
    """D2: the GLM training stage echoes the training config (hash and the ``models`` subtree) in
    three places (``train.config_summary`` -> cv_glm.json /config, ``freeze.check_training_record``
    -> manifest /training_record, ``cv_report`` -> the hash line of glm_cv.md). The committed echo
    must equal the current config restricted to the committed keys; the added keys are exactly the
    pinned ones with the pinned value hashes; the old/new hashes are pinned; every other field of
    cv_glm.json is exact (volatile telemetry keys excepted, as in the strict check)."""
    res = _new_result(dev)
    problems = res["problems"]
    cv_rel, md_rel = dev["echoes"]["cv_json"], dev["echoes"]["report"]
    old, new = dev["old_config_sha256"], dev["new_config_sha256"]
    chk = cfg["freeze"]["check"]
    cv_file, md_file = root / cv_rel, root / md_rel
    if not cv_file.is_file():
        return res
    cv_new_bytes = cv_file.read_bytes()
    try:
        committed_bytes = git_blob(root, "HEAD", cv_rel)
    except FreezeError:
        return res
    md_base = md_new = None
    try:
        md_base = git_blob(root, "HEAD", md_rel).decode("utf-8")
        md_new = md_file.read_text(encoding="utf-8") if md_file.is_file() else None
    except (FreezeError, OSError, ValueError):
        pass
    cv_changed = cv_new_bytes != committed_bytes
    md_changed = md_new is not None and md_base is not None and md_new != md_base
    if not cv_changed and not md_changed:
        return res
    try:
        base_doc = strict_json_loads(committed_bytes.decode("utf-8"))
        new_doc = strict_json_loads(cv_new_bytes.decode("utf-8"))
        bcfg, ncfg = base_doc["config"], new_doc["config"]
    except (ValueError, KeyError, TypeError) as exc:
        problems.append(f"{cv_rel}: unreadable echo ({exc})")
        res["satisfied"] = False
        return res
    expected = json.loads(json.dumps(config_summary(main_config(cfg)), default=str))
    if expected["config_sha256"] != new:
        problems.append("the hash of the current config differs from the pinned new hash")
    if ncfg != expected:
        problems.append(f"{cv_rel}: the new echo is not the echo of the current config")
    if bcfg.get("config_sha256") != old:
        problems.append(f"{cv_rel}: the committed hash is not the pinned old hash")
    if ncfg.get("config_sha256") != new:
        problems.append(f"{cv_rel}: the new hash is not the pinned new hash")
    added, removed = _key_diff(bcfg, ncfg)
    if removed or sorted(added) != sorted(dev["added_keys"]):
        problems.append(
            f"{cv_rel}: added keys {added} / removed keys {removed} differ from the pin"
        )
    for dotted, pinned in dev["added_value_sha256"].items():
        try:
            value = _ptr_get(ncfg, dotted.replace(".", "/"))
        except (KeyError, IndexError, ValueError, TypeError):
            problems.append(f"{cv_rel}: pinned added key {dotted} not found")
            continue
        if canonical_sha256(value) != pinned:
            problems.append(f"{cv_rel}: value of the added key {dotted} differs from its pin")
    restricted = copy.deepcopy(_restrict(expected, bcfg))
    restricted["config_sha256"] = old
    if not compare_values(bcfg, restricted).ok:
        problems.append(f"{cv_rel}: the committed echo is not the current config restricted to it")
    rest = lambda d: {k: v for k, v in d.items() if k != "config"}  # noqa: E731
    if not compare_values(
        rest(base_doc), rest(new_doc), drop=frozenset(chk["volatile_keys"]),
        exact_keys=frozenset(chk["exact_keys"]),
    ).ok:  # fmt: skip
        problems.append(f"{cv_rel}: differs from the committed file outside /config")
    tr = manifest.get("training_record", {})
    if tr.get("glm_config_sha256_at_training") != old:
        problems.append(
            "manifest /training_record/glm_config_sha256_at_training is not the old pin"
        )
    if md_changed and (md_base.count(old) != 1 or md_new.count(new) != 1 or old in md_new):
        problems.append(
            f"{md_rel}: the hash line is not the pinned old hash replaced by the new one"
        )
    # Static evidence is a recorded, labelled grep listing (a regex over module text): it can miss a
    # read through another idiom, so it is documentation only and never a pass/fail input.
    evidence = dev["static_evidence"]
    scanned = scan_readers(evidence, root)
    declared = {k: sorted(v["modules"]) for k, v in evidence["keys"].items()}
    glm_modules = set(evidence["glm_training_modules"])
    res["static_evidence"] = {
        "label": "informational grep listing (regex); not proof, not a pass/fail input",
        "scanned": scanned,
        "listing_at_amendment": declared,
        "listing_unchanged": scanned == declared,
        "glm_training_modules": evidence["glm_training_modules"],
        "glm_training_modules_matching": {
            k: sorted(set(m) & glm_modules) for k, m in scanned.items()
        },
    }
    if cv_changed:
        res["applied"].append({"place": f"{cv_rel} /config/config_sha256", "base": old, "new": new})
        res["applied"] += [
            {"place": f"{cv_rel} /config/{k.replace('.', '/')}", "base": "<absent>",
             "new": f"sha256 {dev['added_value_sha256'][k]}"}
            for k in dev["added_keys"]
        ]  # fmt: skip
        res["applied"].append(
            {"place": "manifest /training_record/glm_config_sha256_at_training",
             "base": old, "new": new}
        )  # fmt: skip
    if md_changed:
        res["applied"].append({"place": f"{md_rel} hash line", "base": old, "new": new})
    if problems:
        res["satisfied"] = False
        return res
    paths = {"/config/" + k.replace(".", "/") for k in dev["added_keys"]} | {
        "/config/config_sha256"
    }
    if cv_changed:
        res["covers_checks"] = {
            "manifest_fields": ["/training_record/glm_config_sha256_at_training"],
            f"telemetry:{cv_rel}": sorted(paths),
        }
        res["covers_files"].append(cv_rel)
    if md_changed:
        # the whole file (this substitution plus the pinned timing cells) is judged by D3 only
        res["md_substitutions"] = {md_rel: [old, new]}
    return res


_DECIMAL = r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"  # a finite non-negative decimal literal
_TABLE_CELL = re.compile(rf"( *)({_DECIMAL})( *)")
_BULLET_SECONDS = re.compile(rf"- (\w+): ({_DECIMAL}) s")


def _is_number(text: str) -> bool:
    return re.fullmatch(_DECIMAL, text) is not None and math.isfinite(float(text))


def _producer_format(renderer: str) -> Any:
    """The producer's own formatter of a timing value: the ``cv_report`` table cell formatter or the
    ``gbm_report`` bullet formatter (one decimal, as ``- <stage>: <n> s``)."""
    from . import cv_report, gbm_report

    if renderer == "cv_report":
        return lambda v: cv_report._f(v)
    if renderer == "gbm_report":
        return lambda v: f"{gbm_report._f(v, 1)}"
    raise KeyError(renderer)


def _telemetry_cell(a: str, b: str, style: str, column: int, timings: dict[str, Any],
                    fmt: Any) -> str | None:  # fmt: skip
    """Why the two differing lines of a pinned section are not a pure telemetry-cell difference,
    or None. ``table``: only the pinned column differs, both sides numbers with equal padding;
    ``bullet_seconds``: ``- key: <number> s`` on both sides with the same key. In both cases the new
    text must equal the producer's rendering of the stage's value in the regenerated JSON."""
    if style == "table":
        pa, pb = a.split("|"), b.split("|")
        i = column + 1  # the line starts with "|"
        if not (a.startswith("|") and len(pa) == len(pb) and i < len(pa) - 1):
            return "not a table row of the same shape"
        if [x for j, x in enumerate(pa) if j != i] != [x for j, x in enumerate(pb) if j != i]:
            return "a cell outside the pinned column differs"
        ma, mb = _TABLE_CELL.fullmatch(pa[i]), _TABLE_CELL.fullmatch(pb[i])
        if not (ma and mb and ma[1] == mb[1] and ma[3] == mb[3]):
            return "the pinned cell is not a plain number on both sides"
        if not (_is_number(ma[2]) and _is_number(mb[2])):
            return "the pinned cell is not finite"
        key, value, text = pa[1].strip(), None, mb[2]
    else:
        ma, mb = _BULLET_SECONDS.fullmatch(a), _BULLET_SECONDS.fullmatch(b)
        if not (ma and mb and ma[1] == mb[1]):
            return "not a '- key: <number> s' line with the same key"
        if not (_is_number(ma[2]) and _is_number(mb[2])):
            return "the value is not finite"
        key, text = ma[1], mb[2] + " s"
    if key not in timings:
        return f"stage {key!r} is not in the regenerated JSON timing_seconds"
    value = timings[key]
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return f"the JSON timing of {key!r} is not a finite number"
    expected = fmt(value) + (" s" if style != "table" else "")
    if text != expected:
        return f"{text!r} is not the producer's rendering {expected!r} of the JSON value"
    return None


def md_telemetry_differences(base: str, new: str, spec: dict[str, Any],
                             timings: dict[str, Any] | None = None):  # fmt: skip
    """Compare a markdown file with its committed text: every byte must be identical except the
    pinned telemetry cells inside the pinned section, each equal to the producer's rendering of the
    regenerated JSON value (``timings``). The pinned heading, marker and table header must exist
    exactly once in both texts. Returns (problems, changed cells)."""
    lb, ln = base.split("\n"), new.split("\n")
    if len(lb) != len(ln):
        return ["the number of lines differs"], []
    head = spec["heading"]
    starts = [i for i, x in enumerate(lb) if x == head]
    if len(starts) != 1 or ln[starts[0]] != head or ln.count(head) != 1:
        return [f"pinned heading {head!r} is not unique at the same line in both files"], []
    start = starts[0]
    end = next((i for i in range(start + 1, len(lb)) if re.match(r"#{1,2} ", lb[i])), len(lb))
    cells, problems = spec["cells"], []
    try:
        fmt = _producer_format(spec["renderer"])
    except KeyError:
        return [f"unknown renderer {spec.get('renderer')!r}"], []
    if not isinstance(timings, dict):
        return ["the regenerated JSON has no timing_seconds mapping"], []
    marker = cells.get("after")
    first = start + 1
    if marker is not None:
        marks = [i for i in range(start + 1, end) if lb[i].startswith(marker)]
        if len(marks) != 1 or sum(x.startswith(marker) for x in ln[start + 1 : end]) != 1:
            return [f"marker {marker!r} is not unique in the pinned section"], []
        first = marks[0] + 1
    header = cells.get("header")
    if header is not None and (
        lb[start + 1 : end].count(header) != 1 or ln[start + 1 : end].count(header) != 1
    ):
        return [f"table header {header!r} is not unique in the pinned section"], []
    changed = []
    for i, (a, b) in enumerate(zip(lb, ln, strict=True)):
        if a == b:
            continue
        if not first <= i < end:
            problems.append(f"line {i + 1} differs outside the pinned cells")
            continue
        why = _telemetry_cell(a, b, cells["style"], int(cells.get("column", 0)), timings, fmt)
        if why:
            problems.append(f"line {i + 1}: {why}")
        else:
            changed.append({"line": i + 1, "base": a.strip(), "new": b.strip()})
    return problems, changed


def _verify_markdown_telemetry(dev, cfg: dict[str, Any], root: Path, manifest: dict[str, Any],
                               prior: list[dict[str, Any]]):  # fmt: skip
    """D3: the markdown renderings of the volatile telemetry (``timing_seconds``). Within the pinned
    section of each pinned file only the pinned cells may differ; each equals the producer's
    rendering of the corresponding value of the regenerated JSON; every other byte is identical to
    the committed text (after the hash-line substitution of a satisfied D2). Any other file or
    section is not covered."""
    res = _new_result(dev)
    problems = res["problems"]
    volatile = set(cfg["freeze"]["check"]["volatile_keys"])
    if not set(dev["volatile_keys"]) <= volatile:
        problems.append("the rendered keys are not declared volatile in the check policy")
    subs: dict[str, list[str]] = {}
    for r in prior:
        if r["satisfied"]:
            subs.update(r.get("md_substitutions", {}))
    for spec in dev["files"]:
        rel, file = spec["path"], root / spec["path"]
        try:
            base = git_blob(root, "HEAD", rel).decode("utf-8")
            new = file.read_text(encoding="utf-8") if file.is_file() else None
        except (FreezeError, OSError, ValueError):
            continue
        if new is None or new == base:
            continue
        if rel in subs:
            base = base.replace(*subs[rel])
        try:
            doc = strict_json_loads((root / spec["json"]).read_text(encoding="utf-8"))
            timings = doc["timing_seconds"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f"{rel}: the regenerated JSON {spec.get('json')} is unreadable ({exc})")
            continue
        bad, changed = md_telemetry_differences(base, new, spec, timings)
        problems += [f"{rel}: {p}" for p in bad]
        if not bad:
            res["covers_files"].append(rel)
            res["applied"] += [
                {"place": f"{rel}:{c['line']}", "base": c["base"], "new": c["new"]} for c in changed
            ]
    if problems:
        res["satisfied"] = False
        res["covers_files"] = []
    return res


_VERIFIERS = {
    "download_metadata": _verify_download_metadata,
    "glm_config_echo": _verify_glm_config_echo,
    "markdown_telemetry": _verify_markdown_telemetry,
}


def verify_amendment(
    amendment: dict[str, Any], cfg: dict[str, Any], root: Path
) -> list[dict[str, Any]]:
    """Evaluate every declared deviation of ``amendment`` against the files under ``root``. A
    deviation is ``satisfied`` unless a difference it would excuse breaks one of its pins; it
    ``covers`` failed checks and compared files only if satisfied and the difference exists."""
    paths = project_paths(cfg, root)
    out = []
    try:
        manifest = strict_json_loads((root / paths["manifest"]).read_text(encoding="utf-8"))
        manifest["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return [
            {**_new_result(d), "satisfied": False, "problems": ["frozen manifest unreadable"]}
            for d in amendment["declared_deviations"]
        ]
    for dev in amendment["declared_deviations"]:
        verifier = _VERIFIERS.get(dev["kind"])
        if verifier is None:
            out.append({**_new_result(dev), "satisfied": False, "problems": ["unknown kind"]})
        elif dev["kind"] == "download_metadata":
            out.append(verifier(dev, root, manifest))
        elif dev["kind"] == "markdown_telemetry":
            out.append(verifier(dev, cfg, root, manifest, out))  # after D2: its hash substitution
        else:
            out.append(verifier(dev, cfg, root, manifest))
    return out


def declared_coverage(
    checks: list[dict[str, Any]], results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Which failed checks of a strict result are covered by satisfied deviations: all mismatches
    of the check are structural, complete (not truncated) and at paths a deviation declares."""
    allowed: dict[str, dict[str, set[str]]] = {}
    for r in results:
        if r["satisfied"]:
            for name, paths in r["covers_checks"].items():
                allowed.setdefault(name, {}).setdefault(r["id"], set()).update(paths)
    covered, uncovered, used = [], [], set()
    for c in checks:
        if c["status"] != "fail":
            continue
        ids = allowed.get(c["name"], {})
        paths = set().union(*ids.values()) if ids else set()
        mm = c["mismatches"]
        ok = (
            bool(paths)
            and c["n_numeric"] == 0
            and c["n_structural"] == len(mm)
            and all(m["kind"] == "structural" and m["path"] in paths for m in mm)
        )
        (covered if ok else uncovered).append(c["name"])
        if ok:
            used.update(ids)
    return {
        "covered_failures": covered,
        "uncovered_failures": uncovered,
        "all_failures_covered": bool(covered) and not uncovered,
        "deviations_used": sorted(used),
    }


def deviation_report(
    amendments: list[dict[str, Any]], cfg: dict[str, Any], root: Path, checks: list[dict[str, Any]]
) -> dict[str, Any]:
    """The ``declared_deviations`` block of a ``freeze --check`` result (report only: ``passed``
    and the exit code stay strict)."""
    results = [r for a in amendments for r in verify_amendment(a, cfg, root)]
    cov = declared_coverage(checks, results)
    return {
        "amendments": [a["id"] for a in amendments],
        "deviations": [
            {k: r[k] for k in ("id", "satisfied", "problems", "applied")} for r in results
        ],
        **cov,
    }


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
    result = {
        "mode": "freeze-check",
        "passed": not failed,
        "numeric_only": all(c["numeric_only"] for c in failed),
        "tolerance": {"rtol": 0.0, "atol": 0.0},
        "manifest": paths["manifest"],
        "head": head_commit(root),
        "checks": checks,
    }
    if cfg.get("amendments"):  # report only: the strict result above is not changed
        result["declared_deviations"] = deviation_report(cfg["amendments"], cfg, root, checks)
    return result
