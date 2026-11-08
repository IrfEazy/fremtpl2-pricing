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
the run. The record is written to ``results/reproduction/run-<n>.json`` of the SOURCE tree (one
immutable record per run; an existing record is never overwritten and the directory is excluded
from the comparison).

Reproduction never recomputes held-out metrics: ``freeze`` and ``evaluate`` are not steps and
``report`` only re-renders the published evaluation.

Scope of ``EXACT_PASS``: exact reproduction of the declared recomputable artifacts plus
verification of the preserved published evidence; held-out metrics are not recomputed. The record
lists the preserved-evidence identities and the recomputable inventory.

Rules are snapshotted from the SOURCE tree (config, inventory of tracked outputs) before any step
and never re-read from the clone. A ``freeze --check`` failure whose mismatches are all numeric does
not stop the run (the strict check stays failed, the remaining steps and the classification run);
integrity, structural and execution failures stop it.

Verdicts (``strict_verdict``, the original policy): ``EXACT_PASS``, ``PORTABLE_NUMERIC_PASS``
(the strict check failed only on numeric differences and the outputs agree within the declared
tolerance, or only the encoding of a figure differs), ``FAIL`` and ``UNSUPPORTED_ENVIRONMENT``
(only an identified missing tool, exit 127 of ``uv``/``git``, or the transport exit code 75).

Amendment (post-hoc, configs/reproduce.yaml ``amendments``): the record also carries
``amended_verdict``, which is ``PASS_WITH_DECLARED_DEVIATIONS`` only if every difference found is
matched by a declared deviation with all its pins satisfied (:mod:`freeze`) and every step and every
comparison completed; otherwise it equals the strict verdict. It is never an exact pass under the
original policy, and the strict verdict is never changed. A ``freeze_check`` step whose failures
are all covered by declared deviations does not stop the run.
"""

from __future__ import annotations

import copy
import csv
import datetime as dt
import fnmatch
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .freeze import (
    FreezeError,
    assert_clean_tree,
    compare_values,
    declared_coverage,
    evidence_identity,
    git,
    git_blob,
    git_tracked,
    head_commit,
    project_paths,
    strict_json_loads,
    verify_amendment,
)

AMENDED_PASS = "PASS_WITH_DECLARED_DEVIATIONS"
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
    """The comparison rules, taken from the SOURCE tree and the tested commit before any step
    (the run history directory is not an output; the amendments are part of the snapshot):
    the reproduce block (tolerances, exact keys, byte-exact and preserved paths, steps), the
    volatile keys and the expected inventory (tracked outputs to regenerate and compare). The
    clone's config is never read, so a step cannot redefine them."""
    rp = copy.deepcopy(cfg["reproduce"])
    paths = project_paths(cfg, root)
    roots = (paths["results"] + "/", Path(cfg["paths"]["reports"]).as_posix() + "/")
    history = f"{paths['results']}/{rp['history_dir']}/"
    is_in = lambda p, pats: any(fnmatch.fnmatch(p, pat) for pat in pats)  # noqa: E731
    tracked = git_tracked(root, commit)
    outputs = sorted(p for p in tracked if p.startswith(roots) and not p.startswith(history))
    return {
        "rp": rp,
        "roots": roots,
        "history": history,
        "amendments": copy.deepcopy(cfg.get("amendments") or []),
        "policy_sha256": cfg.get("reproduce_policy_sha256"),
        "drop": frozenset(cfg["freeze"]["check"]["volatile_keys"]),
        "purge": [p for p in outputs if not is_in(p, rp["preserve_paths"])],
        "compared": [p for p in outputs if not is_in(p, rp["protected_paths"])],
        "preserved": [p for p in outputs if is_in(p, rp["protected_paths"])],
    }


def compare_clone(root: Path, clone: Path, commit: str, rules: dict[str, Any]) -> dict[str, Any]:
    """Every tracked file under ``results/`` and ``reports/`` of the clone against its HEAD blob,
    per category, plus the tracked files that changed outside the declared output paths. ``rules``
    is the snapshot of :func:`snapshot_rules`."""
    rp, roots, history, drop = rules["rp"], rules["roots"], rules["history"], rules["drop"]
    outside = lambda p: not p.startswith(roots)  # noqa: E731
    changed = {
        p
        for p in git(clone, "diff", "--name-only", "-z", "HEAD").split("\0")
        if p and not p.startswith(history)
    }
    others = set(git(clone, "ls-files", "--others", "--exclude-standard", "-z").split("\0")) - {""}
    untracked = sorted(p for p in others if not outside(p) and not p.startswith(history))
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
        "files": [{"path": r["path"], "status": r["status"]} for r in results],
        "source_changed": source_changed,
        "untracked_new": untracked,
        "untracked_source": untracked_source,
        "untracked_undeclared": untracked_undeclared,
        "exact_ok": statuses <= set(OK_EXACT) and not blocked,
        "portable_ok": statuses <= set(OK_PORTABLE) and not blocked,
    }


def incomplete(
    rules: dict[str, Any],
    steps: list[dict[str, Any]],
    check: Any,
    comparison: dict[str, Any] | None,
) -> list[str]:
    """Why a run is not complete (fail closed): every configured step ran, the strict-check result
    is usable and every expected comparison was executed (no missing artifact, no extra or absent
    comparison). An empty list means complete; only then may a passing verdict be given."""
    out = []
    configured = [st["name"] for st in rules["rp"]["steps"]]
    ran = [st["name"] for st in steps]
    if ran != configured:
        out.append(f"steps run {ran} differ from the configured {configured}")
    if not (_usable_check(check) and isinstance(check.get("passed"), bool)):
        out.append("no complete strict freeze-check result")
    if comparison is None:
        out.append("comparison not executed")
        return out
    files = comparison.get("files") or []
    paths = [f.get("path") for f in files]
    if sorted(paths) != sorted(rules["compared"]):
        out.append("the executed comparisons differ from the expected inventory")
    if any(f.get("status") == "missing" for f in files):
        out.append("an expected artifact is missing")
    return out


def decide(
    check: dict[str, Any] | None, comparison: dict[str, Any], reasons: Sequence[str] = ()
) -> str:
    """Verdict of a run whose steps all ran: from the strict ``freeze --check`` result and the
    comparison of the outputs. ``reasons`` (:func:`incomplete`) forbid any passing verdict."""
    if check is None or reasons:
        return FAIL
    if check["passed"] and comparison["exact_ok"]:
        return EXACT
    if (check["passed"] or check["numeric_only"]) and comparison["portable_ok"]:
        return PORTABLE
    return FAIL


# --------------------------------------------------------------------------- history, amendment


def _history_files(directory: Path, pattern: str) -> dict[int, Path]:
    """Existing run records ``run-<n>.json`` of the history directory, by run number."""
    rx = re.compile("^" + re.escape(pattern).replace(re.escape("{n}"), r"(\d+)") + "$")
    found = {}
    if directory.is_dir():
        for f in directory.iterdir():
            if (m := rx.match(f.name)) and f.is_file():
                found[int(m.group(1))] = f
    return found


def assert_history_contiguous(directory: Path, pattern: str, retry_policy: Any) -> int:
    """Enforce ``operational_retry: none``: the history is exactly run-1..run-k (no gap, so no run
    was deleted or replaced); returns k. A failed or incomplete run is its own record."""
    if retry_policy != "none":
        raise FreezeError(f"reproduce.operational_retry must be 'none', not {retry_policy!r}")
    found = sorted(_history_files(directory, pattern))
    if found != list(range(1, len(found) + 1)):
        raise FreezeError(f"the run history {found} is not contiguous from run-1; refused")
    return len(found)


def write_run_record(directory: Path, pattern: str, record: dict[str, Any]) -> Path:
    """Write the record as the next ``run-<n>.json`` (highest existing + 1). The history is
    immutable: the file is created exclusively and an existing one is never overwritten."""
    directory.mkdir(parents=True, exist_ok=True)
    n = max(_history_files(directory, pattern), default=0) + 1
    target = directory / pattern.format(n=n)
    record["run"] = n
    try:
        with target.open("x", encoding="utf-8") as fh:
            fh.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    except FileExistsError as exc:
        raise FreezeError(f"{target} exists; the run history is never overwritten") from exc
    return target


def _usable_check(check: Any) -> bool:
    return isinstance(check, dict) and isinstance(check.get("checks"), list)


def tolerable_check(check: dict[str, Any], results: list[dict[str, Any]]) -> bool:
    """A failed strict check does not stop the run if every failed check is numeric-only (the
    declared tolerance case) or covered by a satisfied declared deviation."""
    if not _usable_check(check):
        return False
    failed = [c for c in check["checks"] if isinstance(c, dict) and c.get("status") == "fail"]
    if not failed:
        return False  # a non-zero exit with no classified failure is never tolerated
    try:
        covered = set(declared_coverage(check["checks"], results)["covered_failures"])
    except (KeyError, TypeError):
        return False  # a malformed check entry is unclassified, never covered
    return all(c.get("name") in covered or c.get("numeric_only") is True for c in failed)


def assess_amendment(
    amendment: dict[str, Any],
    cfg: dict[str, Any],
    rules: dict[str, Any],
    clone: Path,
    steps: list[dict[str, Any]],
    failure: dict[str, Any] | None,
    check: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
    strict: str,
) -> tuple[dict[str, Any], str]:
    """The amended assessment of a run and its ``amended_verdict``.
    ``PASS_WITH_DECLARED_DEVIATIONS`` requires: no failure, every configured step run, a complete
    strict-check result and a complete comparison (no file missing), every failed check covered
    by a satisfied deviation, every compared file either exact (or volatile-only) or covered by a
    satisfied deviation (within-tolerance numeric, reformatted and pixel-equal files are NOT
    declared, so they leave the amended verdict equal to the strict one), and at least one applied
    deviation. Otherwise the amended verdict equals the strict one; unmatched differences are
    listed."""
    results = verify_amendment(amendment, cfg, clone)
    configured = [st["name"] for st in rules["rp"]["steps"]]
    ran = [st["name"] for st in steps]
    unmatched: list[str] = []
    if failure is not None:
        unmatched.append(f"run stopped at {failure.get('step')}")
    unmatched += incomplete(rules, steps, check, comparison) if failure is None else []
    checks = check["checks"] if _usable_check(check) else []
    try:
        cov = declared_coverage(checks, results)
    except (KeyError, TypeError):
        cov = {"covered_failures": [], "uncovered_failures": ["malformed"], "deviations_used": []}
    unmatched += [f"freeze check {n} (not covered)" for n in cov["uncovered_failures"]]
    covered_files = {f for r in results if r["satisfied"] for f in r["covers_files"]}
    files = comparison["files"] if comparison else []
    inv_files = []
    for f in files:
        # only an exact or volatile-only file is within the policy: an in-tolerance numeric
        # difference, a parsed-equal reformatting or a pixel-equal re-encoding is a difference that
        # no deviation declares, unless a satisfied deviation covers that very file
        ok = f["status"] in OK_EXACT
        cov_file = f["status"] not in (*OK_EXACT, "missing") and f["path"] in covered_files
        inv_files.append({**f, "covered_by_deviation": cov_file})
        if not (ok or cov_file):
            unmatched.append(f"{f['path']}: {f['status']}")
    for key in ("source_changed", "untracked_source", "untracked_undeclared"):
        if comparison and comparison[key]:
            unmatched.append(f"{key}: {comparison[key]}")
    failed_files = {f["path"] for f in files if f["status"] == "fail"}
    used = set(cov["deviations_used"]) | {
        r["id"] for r in results if r["satisfied"] and failed_files & set(r["covers_files"])
    }
    applied = [r for r in results if r["satisfied"] and r["applied"] and r["id"] in used]
    failed_check_names = [c["name"] for c in checks if c["status"] == "fail"]
    if strict in (EXACT, PORTABLE, UNSUPPORTED):
        verdict = strict  # nothing for a deviation to excuse (or no assessment possible)
    elif not unmatched and applied:
        verdict = AMENDED_PASS
    else:
        verdict = strict
    record = {
        "id": amendment["id"],
        "date": amendment["date"],
        "post_hoc": amendment["post_hoc"],
        "revealed_by": amendment["revealed_by"],
        "reason": amendment["reason"],
        "original_policy_sha256": amendment["original_policy_sha256"],
        "amended_policy_sha256": rules["policy_sha256"],
        "applied_deviations": [
            {k: r[k] for k in ("id", "kind", "applied", "static_evidence") if k in r}
            for r in applied
        ],
        "deviations": [{k: r[k] for k in ("id", "satisfied", "problems")} for r in results],
        "unmatched_differences": unmatched,
        "inventory": {
            "steps": {"configured": configured, "ran": ran},
            "freeze_checks": [
                {
                    "name": c["name"],
                    "status": c["status"],
                    "covered_by_deviation": c["name"] in cov["covered_failures"],
                }
                for c in checks
            ],  # fmt: skip
            "freeze_checks_failed": failed_check_names,
            "comparison_files": inv_files,
            "n_expected_files": len(rules["compared"]),
        },
    }
    return record, verdict


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


def _source_sha256(name: str) -> str:
    return hashlib.sha256((Path(__file__).with_name(name)).read_bytes()).hexdigest()


def decision_procedure(
    rules: dict[str, Any], amendment: dict[str, Any] | None, commit: str,
    command: Sequence[str] | None,
) -> dict[str, Any]:  # fmt: skip
    """The frozen decision procedure of a run: the policy hashes (original and amended), the sha256
    of the verifier sources, the tested commit, the exact rerun command and the retry policy."""
    amended = rules.get("policy_sha256")
    return {
        "original_policy_sha256": amendment["original_policy_sha256"] if amendment else amended,
        "amended_policy_sha256": amended,
        "policy_file": "configs/reproduce.yaml",
        "verifier_sources_sha256": {
            f"src/fremtpl2_pricing/{n}": _source_sha256(n) for n in ("reproduce.py", "freeze.py")
        },
        "tested_commit": commit,
        "rerun_command": list(command) if command is not None else None,
        "operational_retry": rules["rp"].get("operational_retry"),
    }


def _crash_record(rules, commit, start, t0, exc, command) -> dict[str, Any]:
    """The record of a run that raised: its own run-<n>, FAIL on both verdicts."""
    failure = {"step": "exception", "exit_code": None, "environment": False,
               "reason": f"{type(exc).__name__}: {exc}"}  # fmt: skip
    first = rules["amendments"][0] if rules["amendments"] else None
    return {
        "tested_commit": commit,
        "start_utc": start.isoformat(timespec="seconds"),
        "end_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "total_seconds": round(time.perf_counter() - t0, 3),
        "scope": SCOPE,
        "steps": [],
        "failure": failure,
        "strict_verdict": FAIL,
        "amended_verdict": FAIL,
        "verdict": FAIL,
        "decision_procedure": decision_procedure(rules, first, commit, command),
        "amendment": None,
        "freeze_check": None,
        "comparison": None,
    }


def run_reproduce(
    cfg: dict[str, Any],
    config_path: str | Path,
    force_clean: bool = False,
    root: Path | None = None,
    runner: Runner = subprocess_runner,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """``fremtpl2 reproduce``: clean source tree required; returns (and writes) the record.
    ``operational_retry: none``: the history must be contiguous and every run, failed or incomplete
    included, is its own ``run-<n>`` record (an unexpected exception after the start is recorded as
    a failed run and re-raised)."""
    root = Path.cwd() if root is None else Path(root)
    paths = project_paths(cfg, root)
    history_rel = f"{paths['results']}/{cfg['reproduce']['history_dir']}/"
    assert_clean_tree(root, frozenset({history_rel}))  # new (untracked) run records are fine
    if git(root, "diff", "--name-only", "HEAD", "--", history_rel).strip():
        raise FreezeError(f"a committed run record under {history_rel} was modified (immutable)")
    assert_history_contiguous(
        root / history_rel,
        cfg["reproduce"]["history_pattern"],
        cfg["reproduce"].get("operational_retry"),
    )
    commit = head_commit(root)
    rules = snapshot_rules(cfg, root, commit)  # before any step; never re-read from the clone
    rp = rules["rp"]
    identity = evidence_identity(cfg, root)
    config_rel = Path(config_path)
    if config_rel.is_absolute():
        config_rel = config_rel.relative_to(root)
    work = _workdir(root, cfg, force_clean)
    start, t0 = dt.datetime.now(dt.UTC), time.perf_counter()
    try:
        record = _execute(
            cfg, config_rel, root, runner, work, rules, commit, identity, start, t0, command
        )
    except Exception as exc:
        record = _crash_record(rules, commit, start, t0, exc, command)
        write_run_record(root / history_rel, rp["history_pattern"], record)
        raise
    write_run_record(root / history_rel, rp["history_pattern"], record)
    return record


def _execute(cfg, config_rel, root, runner, work, rules, commit, identity, start, t0, command):
    rp = rules["rp"]
    clone, logs = work / rp["clone_dir"], work / rp["log_dir"]
    logs.mkdir()
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
                devs = [r for a in rules["amendments"] for r in verify_amendment(a, cfg, clone)]
                if not tolerable_check(check, devs):
                    reason = "integrity or structural mismatch in the strict check"
                    failure = {**base, "environment": False, "reason": reason}
            # an all-numeric mismatch, or one fully covered by declared deviations, records the
            # failed strict check and continues
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
        verdict = decide(check, comparison, incomplete(rules, steps, check, comparison))
    amended, amendment_rec = verdict, None
    first = rules["amendments"][0] if rules["amendments"] else None
    if rules["amendments"]:
        amendment_rec, amended = assess_amendment(
            rules["amendments"][0], cfg, rules, clone, steps, failure, check, comparison, verdict
        )
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
        "strict_verdict": verdict,  # the original policy, unchanged semantics
        "amended_verdict": amended,  # PASS_WITH_DECLARED_DEVIATIONS only via the pinned amendment
        "verdict": verdict,  # alias of strict_verdict (kept for callers of the v1 record)
        "decision_procedure": decision_procedure(rules, first, commit, command),
        "amendment": amendment_rec,
        "freeze_check": check,
        "comparison": comparison,
        "tolerances": {
            "rtol": rp["rtol"],
            "atol": rp["atol"],
            "formula": "|a - b| <= atol + rtol * |a|, a = committed value",
        },
    }
    return record
    return record


__all__ = [
    "compare_clone",
    "compare_file",
    "assess_amendment",
    "assert_history_contiguous",
    "incomplete",
    "decide",
    "purge_outputs",
    "run_reproduce",
    "snapshot_rules",
    "write_run_record",
]
