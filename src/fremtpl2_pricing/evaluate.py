"""The single test evaluation (protocol section 6): the ONLY code path that reads test rows.

``fremtpl2 evaluate`` is fit-free and one-shot. :func:`run_attempt` is the whole code path; its
``population`` parameter is the ONLY difference between the real evaluation (the primary grouped
test fold) and ``--rehearsal`` (a registered train CV fold, ``freeze.rehearsal.fold``), whose
journal, staging and published outputs live under ``paths.processed`` instead of ``results/``:

1. an exclusive lock (``O_EXCL`` under ``paths.processed``) prevents concurrent runs;
2. preflight: the work tree must be clean (untracked non-ignored files included), the freeze
   manifest committed, only ``freeze.post_freeze_paths`` changed since its training commit (a
   rehearsal records these git checks instead of enforcing them); every hashed file, the runtime
   identity (python, packages, lock file, threads) and both selections (recomputed as an
   assertion, including the manifest's primary contrast) must match;
3. the journal ``results/test_access.json`` is the single authoritative record; its git-ignored
   mirror is only a check (any disagreement, a torn record or a missing journal fails closed). A
   new attempt is refused once ``TEST_OPENED`` exists, except ``--technical-rerun --reason`` after
   an attempt with ``TEST_OPENED`` and no ``COMPLETE``, with the same manifest digest and runtime
   identity (a deterministic replay, linked to the failed attempt);
4. the stored train predictions are reproduced from the loaded artifacts (with a determinism
   check);
5. ``FROZEN`` (first run) and ``TEST_OPENED`` (attempt id, manifest digest, runtime digest) are
   appended and fsynced with the journal directory; only then is the population loaded;
6. every candidate is scored from the frozen recipes, both bootstraps run (non-finite values fail
   closed), the outputs are written to a private staging directory, validated and hashed;
   ``COMPLETE`` records the hashes (``FAILED`` records the error type only); only then are the
   outputs published to ``results/test/attempt-<n>/`` and the pointer ``results/metrics.json``.
   ``--publish-only`` republishes a COMPLETE attempt's staged outputs (hash-checked) without
   recomputing. A crash leaves ``TEST_OPENED``, which is never reset; completion is never inferred
   from directories or the pointer.
"""

from __future__ import annotations

import datetime as dt
import fnmatch
import json
import os
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import metrics
from .clean import clean
from .data import load_raw
from .freeze import (
    FreezeError,
    assert_clean_tree,
    canonical_sha256,
    git,
    head_commit,
    load_artifacts,
    predict,
    project_paths,
    read_inputs,
    recompute_selections,
    runtime_differences,
    runtime_identity,
    verify_files,
    verify_self_check,
)
from .manifest import sha256_file
from .train import assert_train_only, factor_levels, load_train

EVENTS = ("FROZEN", "TEST_OPENED", "COMPLETE", "FAILED")
REQUIRED_KEYS = {
    "FROZEN": {"event", "timestamp", "manifest_sha256", "commit"},
    "TEST_OPENED": {
        "event",
        "timestamp",
        "attempt",
        "attempt_id",
        "manifest_sha256",
        "runtime_sha256",
        "commit",
        "population",
        "technical_rerun",
        "reason",
        "rerun_of",
        "original_attempt_id",
    },  # fmt: skip
    "COMPLETE": {
        "event",
        "timestamp",
        "attempt",
        "attempt_id",
        "staging_dir",
        "publish_dir",
        "staged",
        "outputs_sha256",
    },  # fmt: skip
    "FAILED": {"event", "timestamp", "attempt", "attempt_id", "error_type"},
}
OUTPUT_FILES = (
    "metrics.json",
    "tables.json",
    "bootstrap_replicates_group.csv",
    "bootstrap_replicates_policy.csv",
)
METRICS_KEYS = {
    "attempt", "attempt_id", "population", "manifest_sha256", "runtime_sha256", "commit",
    "training_commit", "selected_glm_id", "selected_gbm_id", "candidates_order",
    "technical_rerun", "rerun_of", "self_check", "access_log", "conditional_on", "rows", "groups",
    "candidates", "loaded_premium", "bootstrap", "tables_file",
}  # fmt: skip
TABLE_KEYS = {"lorenz", "lift", "calibration_by_factor", "double_lift"}
TEST_POPULATION = {"kind": "test", "rows": "primary_set == 'test' (cv_fold == -1)"}


class EvaluateError(FreezeError):
    """Raised when the evaluation must not (or did not) run."""


# --------------------------------------------------------------------------- durable writes


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds")


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdirs(path: Path) -> None:
    """Create ``path`` and its missing parents, fsyncing each new entry's parent directory."""
    missing = []
    p = path
    while not p.exists():
        missing.append(p)
        p = p.parent
    for d in reversed(missing):
        d.mkdir()
        _fsync_dir(d.parent)


def _durable_write(path: Path, data: bytes) -> None:
    """Write through a temporary file, fsync it, rename it into place and fsync the directory."""
    _mkdirs(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


# --------------------------------------------------------------------------- journal


def real_sink(root: Path, paths: dict[str, Any]) -> dict[str, Path]:
    """Journal and output locations of the real evaluation."""
    return {
        "state": root / paths["state"],
        "mirror": root / paths["mirror"],
        "staging": root / paths["staging"],
        "attempts": root / paths["attempts"],
        "pointer": root / paths["pointer"],
    }


def rehearsal_sink(root: Path, paths: dict[str, Any]) -> dict[str, Path]:
    """A fresh rehearsal directory under ``paths.processed`` with the same layout."""
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    base = root / paths["rehearsal"] / f"run-{stamp}-{uuid.uuid4().hex[:8]}"
    return {
        "base": base,
        "state": base / "journal.json",
        "mirror": base / "journal_mirror.jsonl",
        "staging": base / "staging",
        "attempts": base / "results" / "test",
        "pointer": base / "results" / "metrics.json",
    }


def _parse_journal(path: Path) -> list[dict[str, Any]]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        events = doc["events"]
    except (ValueError, KeyError, TypeError) as exc:
        raise EvaluateError(f"{path.name}: torn or unparseable journal; fail closed") from exc
    if not isinstance(events, list):
        raise EvaluateError(f"{path.name}: the journal has no event list; fail closed")
    return events


def _parse_mirror(path: Path) -> list[dict[str, Any]]:
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").split("\n")):
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError as exc:
            raise EvaluateError(f"{path.name}: torn record on line {i + 1}; fail closed") from exc
    return out


def read_events(sink: dict[str, Path]) -> list[dict[str, Any]]:
    """Events of the authoritative journal, checked against the mirror (fail closed)."""
    state, mirror = sink["state"], sink["mirror"]
    events = _parse_journal(state) if state.exists() else None
    mirrored = _parse_mirror(mirror) if mirror.exists() else None
    if events is None and mirrored:
        raise EvaluateError(
            f"the journal {state.name} is missing while the mirror {mirror.name} records events: "
            "the test-state journal was reset or deleted; refusing (fail closed)"
        )
    if events and mirrored is None:
        raise EvaluateError(
            f"the mirror {mirror.name} is missing while the journal records events; fail closed "
            "(restore the mirror from the journal only after checking the journal's git history)"
        )
    if (events or []) != (mirrored or []):
        raise EvaluateError(
            f"the journal {state.name} and its mirror {mirror.name} disagree (reset, edit or an "
            "interrupted append); fail closed"
        )
    validate_events(events or [])
    return events or []


def validate_events(events: list[dict[str, Any]]) -> None:
    """Complete records; FROZEN first and once; attempts numbered 1..n, each TEST_OPENED followed
    by at most one COMPLETE / FAILED; at most one COMPLETE, never followed by an attempt; every
    rerun is a technical rerun linked to the previous attempt with the same manifest digest and
    runtime identity."""
    if not events:
        return
    for ev in events:
        kind = ev.get("event") if isinstance(ev, dict) else None
        if kind not in EVENTS:
            raise EvaluateError(f"journal: unexpected event {kind!r}")
        missing = REQUIRED_KEYS[kind] - set(ev)
        if missing:
            raise EvaluateError(f"journal: incomplete {kind} record (missing {sorted(missing)})")
    if events[0]["event"] != "FROZEN" or any(e["event"] == "FROZEN" for e in events[1:]):
        raise EvaluateError("journal: the first and only FROZEN event must come first")
    attempts: list[dict[str, Any]] = []
    current = None
    completed = False
    for ev in events[1:]:
        kind = ev["event"]
        if kind == "TEST_OPENED":
            if completed:
                raise EvaluateError("journal: an attempt after COMPLETE")
            if ev["attempt"] != len(attempts) + 1:
                raise EvaluateError("journal: attempts are not numbered consecutively")
            if attempts:
                first, last = attempts[0], attempts[-1]
                if (
                    not ev["technical_rerun"]
                    or ev["rerun_of"] != last["attempt_id"]
                    or ev["original_attempt_id"] != first["attempt_id"]
                    or ev["manifest_sha256"] != first["manifest_sha256"]
                    or ev["runtime_sha256"] != first["runtime_sha256"]
                ):
                    raise EvaluateError("journal: a rerun is not a linked deterministic replay")
            attempts.append(ev)
            current = ev
        else:
            if current is None or (ev["attempt_id"], ev["attempt"]) != (
                current["attempt_id"],
                current["attempt"],
            ):
                raise EvaluateError(f"journal: {kind} without a matching TEST_OPENED")
            completed = completed or kind == "COMPLETE"
            current = None


def append_event(sink: dict[str, Path], event: dict[str, Any]) -> None:
    """Append one event: journal (temporary file, fsync, atomic rename, directory fsync) first,
    then the mirror line (fsync file and directory); the result is re-read and checked.

    A crash between the journal rename and the mirror write leaves the two files disagreeing;
    ``read_events`` then fails closed. Recovery (technical note, operator only): inspect both
    files and restore the mirror from the authoritative journal, never the reverse."""
    events = read_events(sink)
    events = [*events, event]
    validate_events(events)
    payload = {
        "description": "authoritative append-only test access journal (fremtpl2 evaluate); "
        "never edit or reset",
        "events": events,
    }
    _durable_write(sink["state"], (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode())
    mirror = sink["mirror"]
    _mkdirs(mirror.parent)
    with mirror.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    _fsync_dir(mirror.parent)
    if read_events(sink) != events:
        raise EvaluateError("journal: the appended event could not be read back")


def access_log(events: list[dict[str, Any]], current: str | None = None) -> list[dict[str, Any]]:
    """Every attempt of the journal (failed ones with their error type only, no metric)."""
    out: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for ev in events:
        if ev["event"] == "TEST_OPENED":
            rec = {
                "attempt": ev["attempt"],
                "attempt_id": ev["attempt_id"],
                "opened_utc": ev["timestamp"],
                "technical_rerun": ev["technical_rerun"],
                "reason": ev["reason"],
                "rerun_of": ev["rerun_of"],
                "status": "interrupted (TEST_OPENED without a terminal event)",
                "error_type": None,
            }
            by_id[ev["attempt_id"]] = rec
            out.append(rec)
        elif ev["event"] in ("COMPLETE", "FAILED"):
            by_id[ev["attempt_id"]]["status"] = ev["event"]
            by_id[ev["attempt_id"]]["error_type"] = ev.get("error_type")
    if current is not None:
        by_id[current]["status"] = "this attempt"
    return out


class Lock:
    """Exclusive lock file (O_CREAT | O_EXCL); a stale lock after a crash is removed by hand."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            raise EvaluateError(
                f"lock {self.path} exists: another evaluation is running, or a crashed one left "
                "it; remove it by hand only after checking that no evaluation is running"
            ) from exc
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps({"pid": os.getpid(), "created_utc": _now()}) + "\n")
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
        return False


# --------------------------------------------------------------------------- pre-test checks


def git_problems(
    cfg: dict[str, Any], root: Path, paths: dict[str, Any], technical_rerun: bool
) -> list[str]:
    """Work-tree and commit checks: clean tree, committed manifest, training commit an ancestor
    of HEAD, only ``freeze.post_freeze_paths`` changed since (the protocol only for a rerun)."""
    problems = []
    try:
        assert_clean_tree(root)
    except FreezeError as exc:
        problems.append(str(exc))
    mpath = paths["manifest"]
    try:
        if not git(root, "ls-files", mpath).strip():
            problems.append(f"{mpath} is not committed")
        commit = json.loads((root / mpath).read_text(encoding="utf-8"))["training_commit"]
        try:
            git(root, "merge-base", "--is-ancestor", commit, "HEAD")
        except FreezeError:
            problems.append(f"training commit {commit} is not an ancestor of HEAD")
            return problems
        changed = [p for p in git(root, "diff", "--name-only", commit, "HEAD").splitlines() if p]
    except FreezeError as exc:
        problems.append(str(exc))
        return problems
    allowed = cfg["freeze"]["post_freeze_paths"]
    bad = [p for p in changed if not any(fnmatch.fnmatch(p, pat) for pat in allowed)]
    if bad:
        problems.append(f"paths changed since the training commit: {bad}")
    if paths["protocol"] in changed and not technical_rerun:
        problems.append(f"{paths['protocol']} changed after the freeze")
    return problems


def preflight(
    cfg: dict[str, Any],
    config_path: str | Path,
    root: Path,
    technical_rerun: bool,
    enforce_git: bool = True,
) -> dict[str, Any]:
    """Every check that needs no population row. Git checks are enforced for the real
    evaluation and recorded (not enforced) for a rehearsal; everything else is enforced."""
    paths = project_paths(cfg, root)
    if enforce_git:
        try:
            assert_clean_tree(root)
        except FreezeError as exc:
            raise EvaluateError(str(exc)) from exc
    mpath = root / paths["manifest"]
    if not mpath.is_file():
        raise EvaluateError(f"{paths['manifest']} does not exist (run fremtpl2 freeze)")
    problems = git_problems(cfg, root, paths, technical_rerun)
    if enforce_git and problems:
        raise EvaluateError("; ".join(problems))
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest["config_path"] != Path(config_path).as_posix():
        raise EvaluateError(
            f"config {config_path} is not the frozen config {manifest['config_path']}"
        )
    append_only = {paths["protocol"]} if technical_rerun else set()
    bad_files = verify_files(root, manifest["files"], append_only)
    if bad_files:
        raise EvaluateError("frozen files do not match the manifest: " + "; ".join(bad_files))
    runtime = runtime_identity(cfg, root)
    diff = runtime_differences(runtime, manifest["runtime_identity"])
    if diff:
        raise EvaluateError("the runtime identity differs from the freeze: " + "; ".join(diff))
    inputs = read_inputs(root, paths)
    sel = recompute_selections(inputs, cfg)
    for key in ("selected_glm_id", "selected_gbm_id"):
        if sel[key] != manifest[key]:
            raise EvaluateError(f"recomputed {key} {sel[key]} != frozen {manifest[key]}")
    pc = manifest["evaluation"]["primary_contrast"]
    if (pc["gbm"], pc["glm"]) != (sel["selected_gbm_id"], sel["selected_glm_id"]):
        raise EvaluateError(
            f"the manifest primary contrast {pc['gbm']} vs {pc['glm']} differs from the "
            f"recomputed selected ids {sel['selected_gbm_id']} vs {sel['selected_glm_id']}"
        )
    if sel["trace_sha256"] != manifest["selection"]["trace_sha256"]:
        raise EvaluateError("selection trace hashes differ from the manifest")
    if len(manifest["candidates"]) != 6 or set(manifest["candidates"]) != set(manifest["recipes"]):
        raise EvaluateError("the manifest does not define exactly six candidates")
    try:
        commit = head_commit(root)
    except FreezeError:
        if enforce_git:
            raise
        commit = None
    return {
        "paths": paths,
        "manifest": manifest,
        "manifest_sha256": sha256_file(mpath),
        "runtime": runtime,
        "inputs": inputs,
        "commit": commit,
        "git_problems": problems,
    }


def attempt_plan(
    events: list[dict[str, Any]],
    technical_rerun: bool,
    reason: str | None,
    manifest_sha256: str,
    runtime_sha256: str,
):
    """(attempt number, previous TEST_OPENED events) or raise (protocol section 6)."""
    opened = [e for e in events if e["event"] == "TEST_OPENED"]
    if not technical_rerun:
        if opened:
            raise EvaluateError(
                "the test set was already accessed (TEST_OPENED in the journal); a second "
                "evaluation is refused (only --technical-rerun --reason after an attempt that "
                "did not complete)"
            )
        return 1, opened
    if not reason or not reason.strip():
        raise EvaluateError("--technical-rerun needs a non-empty --reason")
    if not opened:
        raise EvaluateError("--technical-rerun: there is no earlier test access to rerun")
    if any(e["event"] == "COMPLETE" for e in events):
        raise EvaluateError(
            "--technical-rerun: an attempt already completed; a completed attempt is never rerun"
        )
    last = opened[-1]
    if last["manifest_sha256"] != manifest_sha256:
        raise EvaluateError(
            "--technical-rerun: the manifest digest differs from the failed attempt"
        )
    if last["runtime_sha256"] != runtime_sha256:
        raise EvaluateError(
            "--technical-rerun: the runtime identity differs from the failed attempt"
        )
    return len(opened) + 1, opened


# --------------------------------------------------------------------------- population


def rehearsal_population(cfg: dict[str, Any]) -> dict[str, Any]:
    fold = cfg["freeze"]["rehearsal"]["fold"]
    k = int(cfg["split"]["cv_folds"])
    if not isinstance(fold, int) or isinstance(fold, bool) or not 0 <= fold < k:
        raise EvaluateError(f"freeze.rehearsal.fold must be a registered train fold 0..{k - 1}")
    return {
        "kind": "rehearsal",
        "cv_fold": fold,
        "rows": f"primary_set == 'train' and cv_fold == {fold} (in-sample for the frozen "
        "models: a code rehearsal, not a result)",
    }


def assert_rehearsal_rows(rows: pd.DataFrame) -> None:
    """A rehearsal population is train rows only; any cv_fold < 0 fails closed."""
    if rows.empty:
        raise EvaluateError("rehearsal: the substituted population is empty")
    if (rows["cv_fold"] < 0).any() or (rows["primary_set"] != "train").any():
        raise EvaluateError(
            "rehearsal: the substituted population contains rows with cv_fold < 0 (test); "
            "fail closed"
        )
    assert_train_only(rows)


def select_population(d: pd.DataFrame, population: dict[str, Any]) -> pd.DataFrame:
    """The rows of the cleaned, split-merged frame that form the evaluated population."""
    if population["kind"] == "test":
        rows = d[d["primary_set"] == "test"].reset_index(drop=True)
        if rows.empty or not (rows["cv_fold"] == -1).all():
            raise EvaluateError("test rows must exist and all have cv_fold == -1")
        return rows
    if population["kind"] == "rehearsal":
        mask = (d["primary_set"] == "train") & (d["cv_fold"] == population["cv_fold"])
        rows = d[mask].reset_index(drop=True)
        assert_rehearsal_rows(rows)
        return rows
    raise EvaluateError(f"unknown population {population['kind']!r}")


def load_population(cfg: dict[str, Any], population: dict[str, Any]) -> pd.DataFrame:
    """Raw -> clean (primary rules, full data) -> frozen split -> the population's rows.

    Policy-level targets come from the ``clean`` columns (as in ``train.load_train``); the
    claim-level severity join is not needed by any metric."""
    if cfg["split"]["primary"] != "grouped":
        raise EvaluateError("the pre-registered primary split is grouped")
    freq, sev = load_raw(cfg)
    res = clean(freq, sev, cfg["clean"])
    split = pd.read_parquet(Path(cfg["paths"]["processed"]) / "split.parquet")
    if set(res.freq["IDpol"]) != set(split["IDpol"]):
        raise EvaluateError("IDpol sets of the cleaned data and the stored split differ")
    d = res.freq.merge(split, on="IDpol", validate="one_to_one")
    return select_population(d, population)


def load_test(cfg: dict[str, Any]) -> pd.DataFrame:
    """The only test loader of the package; called by :func:`run_attempt` after TEST_OPENED."""
    return load_population(cfg, TEST_POPULATION)


def population_rows(cfg: dict[str, Any], population: dict[str, Any]) -> pd.DataFrame:
    if population["kind"] == "test":
        return load_test(cfg)
    return load_population(cfg, population)


# --------------------------------------------------------------------------- metrics


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.reset_index().to_dict("records")


def decision(lower: float, upper: float) -> str:
    """Protocol section 6: improvement iff upper < 0, deterioration iff lower > 0."""
    if upper < 0:
        return "improvement"
    if lower > 0:
        return "deterioration"
    return "no demonstrated difference"


def group_summary(df: pd.DataFrame, gcfg: dict[str, Any]) -> dict[str, Any]:
    """Number of rating-profile groups, size quantiles and exposure concentration."""
    g = df.groupby("group_id").agg(size=("IDpol", "size"), exposure=("Exposure", "sum"))
    g = g.sort_values(["size"], ascending=False, kind="stable")
    n_top = max(1, int(np.ceil(gcfg["top_share"] * len(g))))
    return {
        "groups": int(len(g)),
        "policies": int(g["size"].sum()),
        "exposure": float(g["exposure"].sum()),
        "size_quantiles": {str(q): float(g["size"].quantile(q)) for q in gcfg["size_quantiles"]},
        "max_group_size": int(g["size"].max()),
        "top_share": gcfg["top_share"],
        "top_groups": n_top,
        "top_policy_share": float(g["size"].iloc[:n_top].sum() / g["size"].sum()),
        "top_exposure_share": float(g["exposure"].iloc[:n_top].sum() / g["exposure"].sum()),
    }


def unit_deviance_numerators(df: pd.DataFrame, preds: dict[str, Any], power: float):
    """Per row e_i * d_p(y_i, mu_mi) of every candidate (y = S_cap / E, mu = pure premium rate)."""
    e = df["Exposure"].to_numpy(dtype=float)
    y = df["S_cap"].to_numpy(dtype=float) / e
    return {
        cid: e * metrics.tweedie_unit_deviance(y, p["pure_premium"], power)
        for cid, p in preds.items()
    }


def bootstrap_block(df, preds, spec, unit_column: str, label: str) -> tuple[dict, pd.DataFrame]:
    """Paired bootstrap over ``unit_column`` (group_id primary, IDpol sensitivity)."""
    b, ref = spec["bootstrap"], spec["primary_loss"]["power"]
    gbm_id, glm_id = spec["primary_contrast"]["gbm"], spec["primary_contrast"]["glm"]
    e = df["Exposure"].to_numpy(dtype=float)
    s = df["S_cap"].to_numpy(dtype=float)
    num = unit_deviance_numerators(df, preds, ref)
    nums = {f"loss:{cid}": v for cid, v in num.items()}
    nums["delta"] = num[gbm_id] - num[glm_id]
    g_gbm = metrics.WeightedGini(s, preds[gbm_id]["pure_premium"], e)
    g_glm = metrics.WeightedGini(s, preds[glm_id]["pure_premium"], e)

    def gini_fn(w: np.ndarray) -> dict[str, float]:
        a, c = g_gbm(w)["gini_normalized"], g_glm(w)["gini_normalized"]
        return {"gini_normalized:gbm": a, "gini_normalized:glm": c, "gini_difference": a - c}

    res = metrics.ratio_bootstrap(
        df[unit_column].to_numpy(),
        nums,
        e,
        int(b["replicates"]),
        int(b["seed"]),
        float(b["level"]),
        b["quantile_method"],
        row_metric_fn=gini_fn,
    )
    ci = res["ci"]
    lo, hi = float(ci.loc["delta", "lower"]), float(ci.loc["delta", "upper"])
    out = {
        "label": label,
        "unit_column": unit_column,
        "n_units": res["n_units"],
        "replicates": int(b["replicates"]),
        "seed": int(b["seed"]),
        "level": float(b["level"]),
        "quantile_method": b["quantile_method"],
        "delta": {
            "definition": f"L({gbm_id}) - L({glm_id})",
            "point": res["point"]["delta"],
            "lower": lo,
            "upper": hi,
            "decision": decision(lo, hi),
        },
        "gini_difference": {
            "definition": f"normalised Gini({gbm_id}) - normalised Gini({glm_id})",
            "point": res["point"]["gini_difference"],
            "lower": float(ci.loc["gini_difference", "lower"]),
            "upper": float(ci.loc["gini_difference", "upper"]),
        },
        "intervals": {
            k: {
                "point": res["point"][k],
                "lower": float(ci.loc[k, "lower"]),
                "upper": float(ci.loc[k, "upper"]),
            }
            for k in res["replicates"].columns
        },
    }
    return out, res["replicates"]


def lorenz_points(s, rate, e, n_points: int) -> dict[str, list[float]]:
    lc = metrics.lorenz_curve(s, rate, e)
    grid = np.linspace(0.0, 1.0, int(n_points))
    return {
        "cum_exposure": [float(x) for x in grid],
        "cum_loss": [float(x) for x in np.interp(grid, lc["cum_exposure"], lc["cum_loss"])],
    }


def candidate_metrics(df, preds, spec, cfg) -> tuple[dict[str, Any], dict[str, Any]]:
    """Scalar metrics and tables of every candidate on the given (test) rows."""
    ev, fcfg = cfg["models"]["evaluation"], cfg["features"]
    ref = spec["primary_loss"]["power"]
    nb = int(ev["n_bins"])
    e = df["Exposure"].to_numpy(dtype=float)
    s = df["S_cap"].to_numpy(dtype=float)
    cnt = df["freq_count"].to_numpy(dtype=float)
    n = df["N"].to_numpy(dtype=float)
    cl = n > 0
    if not (e.sum() > 0 and n[cl].sum() > 0):
        raise EvaluateError("zero total exposure or no claims in the population; fail closed")
    y = s / e
    loading = spec["excess_loading_per_exposure_year"]
    levels = {
        name: factor_levels(df, name, fspec, fcfg)
        for name, fspec in ev["calibration_factors"].items()
    }
    scal: dict[str, Any] = {}
    tabs: dict[str, Any] = {"lorenz": {}, "lift": {}, "calibration_by_factor": {}}
    for cid, p in preds.items():
        rate = p["pure_premium"]
        power = spec["candidate_powers"][cid]
        m: dict[str, Any] = {
            "primary_loss": metrics.tweedie_deviance(y, rate, ref, e) / e.sum(),
            "tweedie_deviance_per_exposure": metrics.tweedie_deviance(y, rate, power, e) / e.sum(),
            "tweedie_power": power,
            "poisson_deviance_per_exposure": metrics.tweedie_deviance(y, rate, 1.0, e) / e.sum(),
            **metrics.gini(s, rate, e),
            "calibration_totals": metrics.calibration_totals(s, rate * e),
        }
        tabs["lorenz"][cid] = lorenz_points(s, rate, e, ev["lorenz_points"])
        tabs["lift"][cid] = {"pure_premium": _records(metrics.lift_table(s, rate * e, e, nb))}
        cal: dict[str, Any] = {}
        for name, (lv, order) in levels.items():
            t = metrics.calibration_by_level(s, rate * e, lv.to_numpy())
            cal[name] = {"pure_premium": _records(t.reindex([o for o in order if o in t.index]))}
        if "frequency" in p:
            f = p["frequency"]
            m["frequency"] = {
                "poisson_deviance_per_exposure": metrics.poisson_deviance_per_exposure(
                    cnt, f * e, e
                ),
                **metrics.gini(cnt, f, e),
                "calibration_totals": metrics.calibration_totals(cnt, f * e),
            }
            tabs["lift"][cid]["frequency"] = _records(metrics.lift_table(cnt, f * e, e, nb))
            for name, (lv, order) in levels.items():
                t = metrics.calibration_by_level(cnt, f * e, lv.to_numpy())
                cal[name]["frequency"] = _records(t.reindex([o for o in order if o in t.index]))
        if "severity" in p:
            sv = p["severity"][cl]
            m["severity"] = {
                "gamma_deviance_per_claim": metrics.gamma_deviance(s[cl] / n[cl], sv, n[cl])
                / n[cl].sum(),
                "calibration_totals": metrics.calibration_totals(s[cl], sv * n[cl]),
            }
        tabs["calibration_by_factor"][cid] = cal
        scal[cid] = m
    gbm_id, glm_id = spec["primary_contrast"]["gbm"], spec["primary_contrast"]["glm"]
    tabs["double_lift"] = {
        "a": gbm_id,
        "b": glm_id,
        "table": _records(
            metrics.double_lift_table(
                s, preds[gbm_id]["pure_premium"] * e, preds[glm_id]["pure_premium"] * e, e, nb
            )
        ),
    }
    uncapped = float(df["S"].sum())
    loaded = {
        "label": "loaded premium = capped prediction + training-only flat excess loading per "
        "exposure-year; reported separately, never compared with capped targets, no claim of "
        "predictive adequacy",
        "loading_per_exposure_year": loading,
        "observed_uncapped_total": uncapped,
        "by_candidate": {},
    }
    for cid, p in preds.items():
        tot = float(np.sum(p["pure_premium"] * e)) + loading * float(e.sum())
        loaded["by_candidate"][cid] = {
            "predicted_loaded_total": tot,
            "ratio_observed_uncapped_to_loaded": uncapped / tot,
        }
    return {"candidates": scal, "loaded_premium": loaded}, tabs


def evaluate_frame(df, preds, spec, cfg) -> dict[str, Any]:
    """Everything computed on the test frame: metrics, tables, both bootstraps (pure function)."""
    scal, tabs = candidate_metrics(df, preds, spec, cfg)
    boot, reps = bootstrap_block(df, preds, spec, "group_id", "primary: rating-profile groups")
    boot_p, reps_p = bootstrap_block(
        df, preds, spec, "IDpol", "labelled sensitivity: policies as the resampling unit"
    )
    return {
        "scalars": scal,
        "tables": tabs,
        "bootstrap": {"group": boot, "policy": boot_p},
        "replicates": {"group": reps, "policy": reps_p},
        "groups": group_summary(df, cfg["models"]["group_summary"]),
        "rows": {
            "policies": int(len(df)),
            "exposure": float(df["Exposure"].sum()),
            "claims": int(df["N"].sum()),
            "claim_policies": int((df["N"] > 0).sum()),
        },
    }


# --------------------------------------------------------------------------- staging


def _jsonable(obj: Any, where: str = "$") -> Any:
    """JSON-ready copy; a non-finite number anywhere fails closed (never written as null)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, f"{where}.{k}") for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v, f"{where}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, bool | np.bool_):
        return bool(obj)
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        if not np.isfinite(obj):
            raise EvaluateError(f"non-finite value at {where}; fail closed")
        return float(obj)
    if obj is None or isinstance(obj, str):
        return obj
    raise EvaluateError(f"unexpected value type {type(obj).__name__} at {where}")


def _spec(manifest: dict[str, Any]) -> dict[str, Any]:
    spec = dict(manifest["evaluation"])
    spec["candidate_powers"] = {
        cid: float(r.get("tweedie_power") or manifest["training_record"]["tweedie_power"])
        for cid, r in manifest["recipes"].items()
    }
    return spec


def output_documents(out: dict[str, Any], header: dict[str, Any]) -> dict[str, bytes]:
    """The bytes of every output file (non-finite values fail closed)."""
    metrics_doc = {
        **header,
        "rows": out["rows"],
        "groups": out["groups"],
        "candidates": out["scalars"]["candidates"],
        "loaded_premium": out["scalars"]["loaded_premium"],
        "bootstrap": out["bootstrap"],
        "tables_file": "tables.json",
    }
    docs = {
        "metrics.json": json.dumps(_jsonable(metrics_doc), indent=2, allow_nan=False) + "\n",
        "tables.json": json.dumps(_jsonable(out["tables"]), indent=2, allow_nan=False) + "\n",
    }
    for unit in ("group", "policy"):
        reps = out["replicates"][unit]
        if not np.isfinite(reps.to_numpy(dtype=float)).all():
            raise EvaluateError(f"non-finite {unit} bootstrap replicate; fail closed")
        docs[f"bootstrap_replicates_{unit}.csv"] = reps.to_csv(index=False)
    return {k: v.encode("utf-8") for k, v in docs.items()}


def validate_staged(sdir: Path, header: dict[str, Any], spec: dict[str, Any]) -> None:
    """Read the staged files back: expected files and keys, every candidate, finite values,
    one replicate row per bootstrap replicate."""
    names = sorted(p.name for p in sdir.iterdir() if p.is_file())
    if names != sorted(OUTPUT_FILES):
        raise EvaluateError(f"staged outputs {names} != expected {sorted(OUTPUT_FILES)}")
    m = json.loads((sdir / "metrics.json").read_text(encoding="utf-8"))
    t = json.loads((sdir / "tables.json").read_text(encoding="utf-8"))
    _jsonable(m)
    _jsonable(t)
    if set(m) != METRICS_KEYS:
        raise EvaluateError(f"metrics.json keys differ: {sorted(set(m) ^ METRICS_KEYS)}")
    if set(t) != TABLE_KEYS:
        raise EvaluateError(f"tables.json keys differ: {sorted(set(t) ^ TABLE_KEYS)}")
    if list(m["candidates"]) != list(header["candidates_order"]):
        raise EvaluateError("metrics.json does not hold every candidate in the frozen order")
    b = int(spec["bootstrap"]["replicates"])
    for unit in ("group", "policy"):
        blk = m["bootstrap"][unit]["delta"]
        if not {"point", "lower", "upper", "decision"} <= set(blk):
            raise EvaluateError(f"bootstrap {unit}: incomplete delta block")
        reps = pd.read_csv(sdir / f"bootstrap_replicates_{unit}.csv")
        if len(reps) != b or "delta" not in reps:
            raise EvaluateError(f"bootstrap {unit}: {len(reps)} replicate rows != {b}")
        if not np.isfinite(reps.to_numpy(dtype=float)).all():
            raise EvaluateError(f"bootstrap {unit}: non-finite staged replicates")


def stage_outputs(
    sdir: Path, out: dict[str, Any], header: dict[str, Any], spec: dict[str, Any]
) -> dict[str, str]:
    """Write the outputs to the (new, private) staging directory durably, validate and hash."""
    docs = output_documents(out, header)
    for name, data in docs.items():
        _durable_write(sdir / name, data)
    validate_staged(sdir, header, spec)
    return {name: sha256_file(sdir / name) for name in OUTPUT_FILES}


# --------------------------------------------------------------------------- publication


def pointer_bytes(complete: dict[str, Any], metrics_raw: bytes) -> bytes:
    doc = {
        "source": {
            "attempt_dir": complete["publish_dir"],
            "attempt": complete["attempt"],
            "attempt_id": complete["attempt_id"],
            "metrics_sha256": complete["staged"]["metrics.json"],
            "outputs_sha256": complete["outputs_sha256"],
            "note": "the completed attempt of the single test evaluation (journal COMPLETE)",
        },
        **json.loads(metrics_raw.decode("utf-8")),
    }
    return (json.dumps(doc, indent=2, allow_nan=False) + "\n").encode("utf-8")


def publish(root: Path, sink: dict[str, Path], complete: dict[str, Any]) -> dict[str, Any]:
    """Copy the staged outputs of a COMPLETE event to its publish directory and write the pointer.

    Every staged file must match the hash recorded in COMPLETE; an already published file must
    match too (idempotent republication); any mismatch fails closed. Nothing is recomputed."""
    if complete.get("event") != "COMPLETE":
        raise EvaluateError("only a COMPLETE event can be published")
    staging, dest = root / complete["staging_dir"], root / complete["publish_dir"]
    staged = complete["staged"]
    if canonical_sha256(staged) != complete["outputs_sha256"]:
        raise EvaluateError("COMPLETE: the recorded output hashes are inconsistent; fail closed")
    blobs = {}
    for name, sha in staged.items():
        src = staging / name
        if not src.is_file():
            raise EvaluateError(f"staged output {name} is missing; fail closed")
        data = src.read_bytes()
        if sha256_file(src) != sha:
            raise EvaluateError(f"staged output {name}: sha256 mismatch with COMPLETE; fail closed")
        blobs[name] = data
    for name, data in blobs.items():
        target = dest / name
        if target.exists():
            if sha256_file(target) != staged[name]:
                raise EvaluateError(f"published {target} differs from the staged output; refusing")
            continue
        _durable_write(target, data)
        if sha256_file(target) != staged[name]:
            raise EvaluateError(f"published {target}: sha256 mismatch after the copy")
    ptr = pointer_bytes(complete, blobs["metrics.json"])
    pointer = sink["pointer"]
    if pointer.exists():
        if pointer.read_bytes() != ptr:
            raise EvaluateError(f"{pointer} exists and differs from the COMPLETE attempt; refusing")
    else:
        _durable_write(pointer, ptr)
    return json.loads(ptr.decode("utf-8"))


# --------------------------------------------------------------------------- run


def run_attempt(
    cfg: dict[str, Any],
    config_path: str | Path,
    population: dict[str, Any],
    technical_rerun: bool = False,
    reason: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """The evaluate code path for a ``population`` (the test fold, or a rehearsal train fold).

    Only the population and the journal / output locations differ between the real evaluation
    and the rehearsal; every check, load, prediction, metric, bootstrap, staging, journal and
    publication step is the same code."""
    root = Path.cwd() if root is None else Path(root)
    paths = project_paths(cfg, root)
    rehearsal = population["kind"] != "test"
    if rehearsal and technical_rerun:
        raise EvaluateError("a rehearsal is never a technical rerun")
    with Lock(root / paths["lock"]):
        pre = preflight(cfg, config_path, root, technical_rerun, enforce_git=not rehearsal)
        manifest, msha, rsha = pre["manifest"], pre["manifest_sha256"], pre["runtime"]["sha256"]
        real_events = read_events(real_sink(root, paths))  # checked in both modes, read only
        sink = rehearsal_sink(root, paths) if rehearsal else real_sink(root, paths)
        events = read_events(sink) if rehearsal else real_events
        n, opened = attempt_plan(events, technical_rerun, reason, msha, rsha)
        adir = sink["attempts"] / f"attempt-{n}"
        sdir = sink["staging"] / f"attempt-{n}"
        for d in (adir, sdir):
            if d.exists():
                raise EvaluateError(f"{d} exists; attempts are never overwritten")
        recipes = manifest["recipes"]
        art = load_artifacts(root, paths, pre["inputs"], cfg["freeze"]["runtime"]["num_threads"])
        sc = manifest["self_check"]
        self_check = verify_self_check(sc, load_train(cfg), recipes, art, float(sc["rtol"]))
        attempt_id = uuid.uuid4().hex
        base = {"manifest_sha256": msha, "commit": pre["commit"]}
        if not events:
            append_event(sink, {"event": "FROZEN", "timestamp": _now(), **base})
        opened_ev = {
            "event": "TEST_OPENED",
            "timestamp": _now(),
            "attempt": n,
            "attempt_id": attempt_id,
            **base,
            "runtime_sha256": rsha,
            "population": population,
            "technical_rerun": bool(technical_rerun),
            "reason": reason if technical_rerun else None,
            "rerun_of": opened[-1]["attempt_id"] if opened else None,
            "original_attempt_id": opened[0]["attempt_id"] if opened else None,
        }
        append_event(sink, opened_ev)
        spec = _spec(manifest)
        try:
            _mkdirs(sdir.parent)
            sdir.mkdir()
            _fsync_dir(sdir.parent)
            rows = population_rows(cfg, population)
            preds = predict(recipes, art, rows)
            out = evaluate_frame(rows, preds, spec, cfg)
            header = {
                "attempt": n,
                "attempt_id": attempt_id,
                "population": population,
                "manifest_sha256": msha,
                "runtime_sha256": rsha,
                "commit": pre["commit"],
                "training_commit": manifest["training_commit"],
                "selected_glm_id": manifest["selected_glm_id"],
                "selected_gbm_id": manifest["selected_gbm_id"],
                "candidates_order": manifest["candidates"],
                "technical_rerun": opened_ev["technical_rerun"],
                "rerun_of": opened_ev["rerun_of"],
                "self_check": self_check,
                "access_log": access_log([*events, opened_ev], current=attempt_id),
                "conditional_on": "frozen fitted models; retraining, tuning and selection "
                "variability are not included",
            }
            staged = stage_outputs(sdir, out, header, spec)
        except Exception as exc:
            if sdir.is_dir():
                (sdir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            try:
                append_event(
                    sink,
                    {
                        "event": "FAILED",
                        "timestamp": _now(),
                        "attempt": n,
                        "attempt_id": attempt_id,
                        "error_type": type(exc).__name__,
                    },
                )
            except Exception as append_exc:  # never mask the original failure
                print(
                    f"evaluate: could not append the FAILED event "
                    f"({type(append_exc).__name__}: {append_exc}); the journal keeps TEST_OPENED",
                    file=sys.stderr,
                )
                exc.add_note(f"the FAILED event could not be appended: {append_exc!r}")
            raise
        complete = {
            "event": "COMPLETE",
            "timestamp": _now(),
            "attempt": n,
            "attempt_id": attempt_id,
            "staging_dir": _rel(root, sdir),
            "publish_dir": _rel(root, adir),
            "staged": staged,
            "outputs_sha256": canonical_sha256(staged),
        }
        append_event(sink, complete)
        doc = publish(root, sink, complete)
    if rehearsal:
        doc["rehearsal"] = {
            "dir": _rel(root, sink["base"]),
            "git_problems_for_the_real_run": pre["git_problems"],
            "note": "a rehearsal of the evaluate code path on train rows; not a test result",
        }
    return doc


def run_evaluate(
    cfg: dict[str, Any],
    config_path: str | Path,
    technical_rerun: bool = False,
    reason: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """``fremtpl2 evaluate``: the single test evaluation (see the module docstring)."""
    return run_attempt(cfg, config_path, TEST_POPULATION, technical_rerun, reason, root)


def run_rehearsal(
    cfg: dict[str, Any], config_path: str | Path, root: Path | None = None
) -> dict[str, Any]:
    """``fremtpl2 evaluate --rehearsal``: the identical code path on ``freeze.rehearsal.fold``."""
    return run_attempt(cfg, config_path, rehearsal_population(cfg), root=root)


def run_publish(
    cfg: dict[str, Any], config_path: str | Path, root: Path | None = None
) -> dict[str, Any]:
    """``fremtpl2 evaluate --publish-only``: republish the COMPLETE attempt's staged, hashed
    outputs (recovery after a failed publication); never recomputes anything."""
    root = Path.cwd() if root is None else Path(root)
    paths = project_paths(cfg, root)
    with Lock(root / paths["lock"]):
        sink = real_sink(root, paths)
        events = read_events(sink)
        done = [e for e in events if e["event"] == "COMPLETE"]
        if not done:
            raise EvaluateError(
                "--publish-only: the journal has no COMPLETE event; nothing to publish "
                "(completion is never inferred from directories or the pointer)"
            )
        complete = done[-1]
        opened = next(
            e for e in events
            if e["event"] == "TEST_OPENED" and e["attempt_id"] == complete["attempt_id"]
        )  # fmt: skip
        msha = sha256_file(root / paths["manifest"])
        if msha != opened["manifest_sha256"]:
            raise EvaluateError("--publish-only: the manifest differs from the completed attempt")
        return publish(root, sink, complete)
