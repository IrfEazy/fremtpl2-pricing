"""Reproduction of the pre-test pipeline in an isolated clean clone of HEAD.

``fremtpl2 reproduce`` runs the steps of ``reproduce.steps`` (setup, download, audit, split, train,
gbm, sensitivities, interpret, the strict ``freeze --check`` and the report re-render) in a fresh
clone under ``paths.processed/reproduce.workdir``; the clone starts with no ``data/`` so a stale
artifact cannot pass. Afterwards every tracked file under ``results/`` and ``reports/`` of the clone
is compared with the HEAD blob (exact bytes, or JSON whose only differences are declared volatile
keys; parsed-equal files with other byte differences are "equivalent", differences within the
declared tolerances "tolerance": portable at most). Before the steps the clone deletes every
tracked output except the declared published paths, so a step that does not regenerate an output
fails. Renderer outputs of committed evidence (``byte_exact_paths``) must be byte-identical; only
figures may differ as renders. Any other tracked file that changed, or any untracked file outside
the output roots, or any new untracked file under them that is not a declared step output, fails
the run. The record is written to ``results/reproduction.json``
of the SOURCE tree.

Reproduction never recomputes held-out metrics: ``freeze`` and ``evaluate`` are not steps and
``report`` only re-renders the published evaluation.

Scope of ``EXACT_PASS``: exact reproduction of the declared recomputable artifacts plus
verification of the preserved published evidence; held-out metrics are not recomputed. The record
lists the preserved-evidence identities and the recomputable inventory.

Rules are snapshotted from the SOURCE tree (config, inventory of tracked outputs) before any step
and never re-read from the clone. A ``freeze --check`` failure whose mismatches are all numeric does
not stop the run (the strict check stays failed, the remaining steps and the classification run);
integrity, structural and execution failures stop it.

Verdicts: ``EXACT_PASS``, ``PORTABLE_NUMERIC_PASS`` (the strict check failed only on numeric
differences and the outputs agree within the declared tolerance, or only the encoding of a figure
differs), ``FAIL`` and ``UNSUPPORTED_ENVIRONMENT`` (only an identified missing tool, exit 127 of
``uv``/``git``, or the transport exit code 75).
"""

from __future__ import annotations

import copy
import csv
import datetime as dt
import fnmatch
import io
import json
import os
import platform
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .freeze import (
    FreezeError,
    assert_clean_tree,
    compare_values,
    evidence_identity,
    git,
    git_blob,
    git_tracked,
    head_commit,
    project_paths,
    strict_json_loads,
)

EXACT, PORTABLE, FAIL, UNSUPPORTED = (
    "EXACT_PASS",
    "PORTABLE_NUMERIC_PASS",
    "FAIL",
    "UNSUPPORTED_ENVIRONMENT",
)
OK_EXACT = ("exact", "volatile_only")
OK_PORTABLE = (*OK_EXACT, "equivalent", "tolerance", "pixel_equal")
SCOPE = (
    "EXACT_PASS = exact reproduction of the declared recomputable artifacts plus verification of "
    "preserved published evidence; held-out metrics are not recomputed"
)
_NUMBER = re.compile(r"(-?\d+\.\d+(?:[eE][-+]?\d+)?)")

Runner = Callable[[list[str], Path], tuple[int, str, str]]


# --------------------------------------------------------------------------- file comparison


def _csv_cell(cell: str) -> Any:
    for conv in (int, float):
        try:
            return conv(cell)
        except ValueError:
            pass
    return cell


def _parsed(rel: str, data: bytes) -> Any:
    """Parsed form of a tracked output: JSON as is, CSV as rows of typed cells, markdown as the
    alternating list of text segments and decimal numbers."""
    text = data.decode("utf-8")
    if rel.endswith(".json"):
        return strict_json_loads(text)
    if rel.endswith(".csv"):
        return [[_csv_cell(c) for c in row] for row in csv.reader(io.StringIO(text))]
    parts = _NUMBER.split(text)
    return [float(p) if i % 2 else p for i, p in enumerate(parts)]


def _literal_view(data: bytes, drop: frozenset[str]) -> Any:
    """JSON with number literals, key order and every non-volatile value preserved and the
    declared volatile keys removed, for the byte-level test of a volatile-only difference."""

    def obj(pairs: list[tuple[str, Any]]) -> Any:
        if len({k for k, _ in pairs}) != len(pairs):
            raise ValueError("duplicate JSON key")
        return ("obj", [(k, v) for k, v in pairs if k not in drop])

    return json.loads(
        data.decode("utf-8"),
        object_pairs_hook=obj,
        parse_float=lambda t: ("float", t),
        parse_int=lambda t: ("int", t),
    )


def _volatile_only(base: bytes, new: bytes, drop: frozenset[str]) -> bool:
    """True if the two JSON documents differ only in the values of declared volatile keys:
    the same literals, key order and line count everywhere else."""
    if base.count(b"\n") != new.count(b"\n"):
        return False
    try:
        return _literal_view(base, drop) == _literal_view(new, drop)
    except ValueError:
        return False


def _pixels_equal(a: bytes, b: bytes) -> bool | None:
    """True/False: decoded pixels (``matplotlib.image.imread``) equal or not; None: undecodable."""
    import matplotlib.image as mpimg
    import numpy as np

    try:
        x, y = (mpimg.imread(io.BytesIO(d), format="png") for d in (a, b))
    except Exception:  # noqa: BLE001  (any decoder error means the image is not comparable)
        return None
    return bool(x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y))


def compare_file(rel: str, base: bytes, new: bytes | None, rp: dict[str, Any],
                 drop: frozenset[str]) -> dict[str, Any]:  # fmt: skip
    """Compare one regenerated output with its committed bytes. ``status`` is ``exact``,
    ``volatile_only``, ``equivalent``, ``tolerance``, ``pixel_equal`` (figure bytes differ, decoded
    pixels identical), ``fail`` or ``missing``."""
    if new is None:
        return {"path": rel, "status": "missing"}
    if new == base:
        return {"path": rel, "status": "exact"}
    if any(fnmatch.fnmatch(rel, pat) for pat in rp["byte_exact_paths"]):
        reason = "bytes differ (renderer output of committed evidence)"
        return {"path": rel, "status": "fail", "reason": reason}
    suffix = Path(rel).suffix
    if suffix in rp["render_suffixes"]:
        same = _pixels_equal(base, new)
        if same:
            return {"path": rel, "status": "pixel_equal"}
        reason = "figure not decodable" if same is None else "decoded pixels differ"
        return {"path": rel, "status": "fail", "reason": reason}
    if suffix not in (".json", ".csv", ".md"):
        return {"path": rel, "status": "fail", "reason": "bytes differ"}
    try:
        cmp = compare_values(
            _parsed(rel, base),
            _parsed(rel, new),
            rtol=float(rp["rtol"]),
            atol=float(rp["atol"]),
            drop=drop if suffix == ".json" else frozenset(),
            exact_keys=frozenset(rp["exact_keys"]),
            limit=int(rp["max_mismatches"]),
        )
    except ValueError as exc:
        return {"path": rel, "status": "fail", "reason": f"unparsable: {exc}"}
    if not cmp.ok:
        status = "fail"
    elif cmp.n_within_tolerance:
        status = "tolerance"
    elif suffix == ".json" and _volatile_only(base, new, drop):
        status = "volatile_only"
    else:
        status = (
            "equivalent"  # parsed-equal, bytes differ for another reason (formatting): never exact
        )
    return {"path": rel, "status": status, "comparison": cmp.summary()}


def _category(rel: str, rp: dict[str, Any]) -> str:
    suffix = Path(rel).suffix
    return "render" if suffix in rp["render_suffixes"] else (suffix.lstrip(".") or "other")


def snapshot_rules(cfg: dict[str, Any], root: Path, commit: str) -> dict[str, Any]:
    """The comparison rules, taken from the SOURCE tree and the tested commit before any step:
    the reproduce block (tolerances, exact keys, byte-exact and preserved paths, steps), the
    volatile keys and the expected inventory (tracked outputs to regenerate and compare). The
    clone's config is never read, so a step cannot redefine them."""
    rp = copy.deepcopy(cfg["reproduce"])
    paths = project_paths(cfg, root)
    roots = (paths["results"] + "/", Path(cfg["paths"]["reports"]).as_posix() + "/")
    record = f"{paths['results']}/{rp['record']}"
    is_in = lambda p, pats: any(fnmatch.fnmatch(p, pat) for pat in pats)  # noqa: E731
    outputs = sorted(p for p in git_tracked(root, commit) if p.startswith(roots) and p != record)
    return {
        "rp": rp,
        "roots": roots,
        "record": record,
        "drop": frozenset(cfg["freeze"]["check"]["volatile_keys"]),
        "purge": [p for p in outputs if not is_in(p, rp["preserve_paths"])],
        "compared": [p for p in outputs if not is_in(p, rp["protected_paths"])],
        "preserved": [p for p in outputs if is_in(p, rp["protected_paths"])],
    }


def compare_clone(root: Path, clone: Path, commit: str, rules: dict[str, Any]) -> dict[str, Any]:
    """Every tracked file under ``results/`` and ``reports/`` of the clone against its HEAD blob,
    per category, plus the tracked files that changed outside the declared output paths. ``rules``
    is the snapshot of :func:`snapshot_rules`."""
    rp, roots, record, drop = rules["rp"], rules["roots"], rules["record"], rules["drop"]
    outside = lambda p: not p.startswith(roots)  # noqa: E731
    changed = set(git(clone, "diff", "--name-only", "-z", "HEAD").split("\0")) - {"", record}
    others = set(git(clone, "ls-files", "--others", "--exclude-standard", "-z").split("\0")) - {""}
    untracked = sorted(p for p in others if not outside(p) and p != record)
    untracked_source = sorted(p for p in others if outside(p))
    declared = {o for step in rp["steps"] for o in step.get("outputs", [])}
    untracked_undeclared = [p for p in untracked if p not in declared]
    compared = rules["compared"]
    real_clone = clone.resolve()
    results = []
    for rel in compared:
        file = clone / rel
        if file.is_symlink() or not file.resolve().is_relative_to(real_clone):
            reason = "symlinked or escaping artifact path"
            results.append({"path": rel, "status": "fail", "reason": reason})
        elif rel not in changed:
            results.append({"path": rel, "status": "exact"})
        else:
            new = file.read_bytes() if file.is_file() else None
            results.append(compare_file(rel, git_blob(root, commit, rel), new, rp, drop))
    categories: dict[str, dict[str, Any]] = {}
    for r in results:
        cat = categories.setdefault(
            _category(r["path"], rp),
            {"files": 0, "statuses": {}, "max_abs": 0.0, "max_rel": 0.0, "mismatches": []},
        )
        cat["files"] += 1
        cat["statuses"][r["status"]] = cat["statuses"].get(r["status"], 0) + 1
        c = r.get("comparison")
        if c:
            cat["max_abs"] = max(cat["max_abs"], c["max_abs"])
            cat["max_rel"] = max(cat["max_rel"], c["max_rel"])
        if r["status"] in ("fail", "missing") and len(cat["mismatches"]) < int(
            rp["max_mismatches"]
        ):
            cat["mismatches"].append(
                {
                    "path": r["path"],
                    "status": r["status"],
                    "details": (c or {}).get("mismatches", r.get("reason", "")),
                }  # fmt: skip
            )
    source_changed = sorted(p for p in changed if p not in compared)
    statuses = {r["status"] for r in results}
    blocked = source_changed or untracked_source or untracked_undeclared
    return {
        "categories": categories,
        "n_files": len(results),
        "source_changed": source_changed,
        "untracked_new": untracked,
        "untracked_source": untracked_source,
        "untracked_undeclared": untracked_undeclared,
        "exact_ok": statuses <= set(OK_EXACT) and not blocked,
        "portable_ok": statuses <= set(OK_PORTABLE) and not blocked,
    }


def decide(check: dict[str, Any] | None, comparison: dict[str, Any]) -> str:
    """Verdict of a run whose steps all ran: from the strict ``freeze --check`` result and the
    comparison of the outputs."""
    if check is None:
        return FAIL
    if check["passed"] and comparison["exact_ok"]:
        return EXACT
    if (check["passed"] or check["numeric_only"]) and comparison["portable_ok"]:
        return PORTABLE
    return FAIL


# --------------------------------------------------------------------------- runner


def subprocess_runner(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    try:
        done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:  # the tool is not installed
        return 127, "", str(exc)
    except OSError as exc:  # e.g. not executable: an execution failure, not a missing tool
        return 126, "", str(exc)
    return done.returncode, done.stdout, done.stderr


def _workdir(root: Path, cfg: dict[str, Any], force_clean: bool) -> Path:
    """The reproduction workdir, validated before anything is created or deleted: its real path
    must be strictly inside the configured processed directory (itself inside the project), the
    path must not be a symlink, and it must not be the project root or one of its ancestors."""
    rp = cfg["reproduce"]
    real_root = root.resolve()
    processed = (root / cfg["paths"]["processed"]).resolve()
    candidate = root / cfg["paths"]["processed"] / rp["workdir"]
    work = candidate.resolve()
    if candidate.is_symlink() or (processed / rp["workdir"]).is_symlink():
        raise FreezeError(f"reproduce.workdir {candidate} is a symlink; refused")
    if (
        not processed.is_relative_to(real_root)
        or processed == real_root
        or work == processed
        or not work.is_relative_to(processed)
        or real_root.is_relative_to(work)
    ):
        raise FreezeError("reproduce.workdir must be a directory strictly inside paths.processed")
    if work.exists() and not work.is_dir():
        raise FreezeError(f"{work} exists and is not a directory")
    if work.exists() and any(work.iterdir()):
        if not force_clean:
            raise FreezeError(f"{work} is not empty; use --force-clean to delete it")
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    return work


def purge_outputs(clone: Path, rules: dict[str, Any]) -> list[str]:
    """Delete from the clone the tracked outputs of the source snapshot (``rules["purge"]``: every
    tracked file under ``results/`` and ``reports/`` except the preserved published paths), so
    that a step which does not rewrite its outputs cannot leave the committed bytes in place.
    Returns the deleted paths."""
    deleted = []
    for rel in rules["purge"]:
        file = clone / rel
        if file.is_file() or file.is_symlink():
            file.unlink()
            deleted.append(rel)
    return deleted


def missing_outputs(clone: Path, step: dict[str, Any]) -> list[str]:
    """Declared ``outputs`` of a step (paths relative to the clone, ignored paths included) that
    are not regular, non-symlinked files inside the clone."""
    real = clone.resolve()
    bad = []
    for rel in step.get("outputs", []):
        file = clone / rel
        if Path(rel).is_absolute() or not (
            file.is_file() and not file.is_symlink() and file.resolve().is_relative_to(real)
        ):
            bad.append(rel)
    return bad


def tool_versions() -> dict[str, str]:
    """Versions of the tools the reproduction depends on (uv, git, python)."""
    out = {"python": platform.python_version()}
    for tool in ("uv", "git"):
        try:
            done = subprocess.run(
                [tool, "--version"], capture_output=True, text=True, check=False, timeout=30
            )
            out[tool] = done.stdout.strip() or f"exit {done.returncode}"
        except (OSError, subprocess.SubprocessError):
            out[tool] = "unavailable"
    return out


def _json_result(stdout: str) -> dict[str, Any] | None:
    start = stdout.find("{")
    try:
        return json.loads(stdout[start:]) if start >= 0 else None
    except ValueError:
        return None


def _is_environment(rules: dict[str, Any], cmd: list[str], code: int) -> bool:
    """Only an identified tool absence (exit 127 of a declared tool command) or a declared
    transport exit code is an environment problem; every other failure is a FAIL."""
    rp = rules["rp"]
    return (code == 127 and cmd[0] in rp["tool_commands"]) or code in rp["transport_exit_codes"]


def run_reproduce(
    cfg: dict[str, Any],
    config_path: str | Path,
    force_clean: bool = False,
    root: Path | None = None,
    runner: Runner = subprocess_runner,
) -> dict[str, Any]:
    """``fremtpl2 reproduce``: clean source tree required; returns (and writes) the record."""
    root = Path.cwd() if root is None else Path(root)
    paths = project_paths(cfg, root)
    record_rel = f"{paths['results']}/{cfg['reproduce']['record']}"
    assert_clean_tree(root, frozenset({record_rel}))
    commit = head_commit(root)
    rules = snapshot_rules(cfg, root, commit)  # before any step; never re-read from the clone
    rp = rules["rp"]
    identity = evidence_identity(cfg, root)
    config_rel = Path(config_path)
    if config_rel.is_absolute():
        config_rel = config_rel.relative_to(root)
    work = _workdir(root, cfg, force_clean)
    clone, logs = work / rp["clone_dir"], work / rp["log_dir"]
    logs.mkdir()
    start, t0 = dt.datetime.now(dt.UTC), time.perf_counter()
    git(root, "clone", "--quiet", "--no-hardlinks", str(root), str(clone))
    clone_commit = head_commit(clone)
    purged = purge_outputs(clone, rules) if clone_commit == commit else []
    steps, check, failure = [], None, None
    if clone_commit != commit:
        reason = f"the clone is at {clone_commit}, not at the tested commit"
        failure = {"step": "clone", "exit_code": None, "environment": False, "reason": reason}
    for step in [] if failure else rp["steps"]:
        cmd = [c.replace("{config}", config_rel.as_posix()) for c in step["cmd"]]
        s0 = time.perf_counter()
        code, out, err = runner(cmd, clone)
        entry = {"name": step["name"], "cmd": cmd, "exit_code": code,
                 "seconds": round(time.perf_counter() - s0, 3)}  # fmt: skip
        (logs / f"{step['name']}.out").write_text(out, encoding="utf-8")
        (logs / f"{step['name']}.err").write_text(err, encoding="utf-8")
        steps.append(entry)
        base = {"step": step["name"], "exit_code": code}
        if step.get("json_result"):
            check = _json_result(out)
            valid = isinstance(check, dict) and {"passed", "numeric_only"} <= set(check)
            if _is_environment(rules, cmd, code):
                failure = {**base, "environment": True}
            elif not valid or (code == 0) != check["passed"]:
                failure = {**base, "environment": False, "reason": "no valid strict-check result"}
            elif not check["passed"] and not check["numeric_only"]:
                reason = "integrity or structural mismatch in the strict check"
                failure = {**base, "environment": False, "reason": reason}
            # an all-numeric mismatch records the failed strict check and continues
        elif code != 0:
            failure = {**base, "environment": _is_environment(rules, cmd, code)}
        if failure is None and (missing := missing_outputs(clone, step)):
            failure = {**base, "environment": False, "missing_outputs": missing}
        if failure:
            break
    end_commit = head_commit(clone)
    if failure is None and end_commit != commit:
        reason = f"the clone moved to {end_commit} during the steps"
        failure = {"step": "clone", "exit_code": None, "environment": False, "reason": reason}
    comparison = None
    if failure:
        verdict = UNSUPPORTED if failure["environment"] else FAIL
    else:
        comparison = compare_clone(root, clone, commit, rules)
        verdict = decide(check, comparison)
    end = dt.datetime.now(dt.UTC)
    record = {
        "tested_commit": commit,
        "clone_commit": clone_commit,
        "start_utc": start.isoformat(timespec="seconds"),
        "end_utc": end.isoformat(timespec="seconds"),
        "total_seconds": round(time.perf_counter() - t0, 3),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "tool_versions": tool_versions(),
        "download_included": any(s["name"] == "download" for s in steps),
        "scope": SCOPE,
        "preserved_evidence": {"identity": identity, "paths": rules["preserved"]},
        "recomputable_inventory": {"compared": rules["compared"], "purged": purged},
        "steps": steps,
        "failure": failure,
        "verdict": verdict,
        "freeze_check": check,
        "comparison": comparison,
        "tolerances": {
            "rtol": rp["rtol"],
            "atol": rp["atol"],
            "formula": "|a - b| <= atol + rtol * |a|, a = committed value",
        },
    }
    target = root / record_rel
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(target)
    return record


__all__ = [
    "compare_clone",
    "compare_file",
    "decide",
    "purge_outputs",
    "run_reproduce",
    "snapshot_rules",
]
