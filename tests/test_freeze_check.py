"""``fremtpl2 freeze --check``: the strict, read-only verification of the committed manifest.

Runs on the synthetic git project of ``test_freeze_evaluate`` (no network, no real data).
"""

# ruff: noqa: F811  (fixtures imported from test_freeze_evaluate)

import hashlib
import json
from pathlib import Path

import pytest

from fremtpl2_pricing import cli, freeze
from fremtpl2_pricing.config import load_reproduce_config, with_reproduce
from test_freeze_evaluate import (  # noqa: F401  (fixtures)
    CONFIG,
    _commit,
    _forbid_fitting,
    _git,
    cfg,
    frozen,
    project,
    template,
)

REPRODUCE = str(Path(__file__).resolve().parents[1] / "configs" / "reproduce.yaml")


def _snapshot(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and ".git" not in p.relative_to(root).parts
    }


def _check(cfg) -> dict:
    return freeze.run_freeze_check(with_reproduce(cfg, load_reproduce_config(REPRODUCE)), CONFIG)


def _cli(*extra: str) -> int:
    args = ["freeze", "--check", *extra, "--config", CONFIG, "--reproduce-config", REPRODUCE]
    return cli.main(args)


def _failed(res: dict) -> set[str]:
    return {c["name"] for c in res["checks"] if c["status"] == "fail"}


def _edit_json(path: Path, fn) -> None:
    obj = json.loads(path.read_text())
    fn(obj)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _first_float(obj, path=()):
    if isinstance(obj, float):
        return path
    items = (
        obj.items() if isinstance(obj, dict) else enumerate(obj) if isinstance(obj, list) else []
    )
    for k, v in items:
        found = _first_float(v, (*path, k))
        if found is not None:
            return found
    return None


def test_check_passes_and_writes_nothing(project, cfg, frozen):
    (project / "untracked.txt").write_text("a dirty tree is allowed")
    before = _snapshot(project)
    res = _check(cfg)
    assert res["passed"], [c for c in res["checks"] if c["status"] == "fail"]
    assert {c["name"] for c in res["checks"]} >= {
        "original_bytes",
        "selection",
        "recipes",
        "self_check",
    }
    assert _snapshot(project) == before


def test_check_cli_exit_codes(project, cfg, frozen, capsys):
    assert _cli() == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    assert _cli("--force") == 1
    (project / "results/feature_map.json").write_text("{}")
    assert _cli() == 1


def test_only_declared_volatile_keys_are_ignored(project, cfg, frozen):
    _edit_json(project / "results/cv_glm.json", lambda o: o["timing_seconds"].update(total=123.0))
    assert _check(cfg)["passed"]
    _edit_json(project / "results/cv_glm.json", lambda o: o.update(undeclared_key=1))
    res = _check(cfg)
    assert not res["passed"] and "telemetry:results/cv_glm.json" in _failed(res)


def test_tampered_model_file_fails(project, cfg, frozen):
    model = next((project / "data/processed/models").glob("*.txt"))
    model.write_text(model.read_text() + "\n# tampered\n")
    res = _check(cfg)
    assert not res["passed"]


def test_tampered_split_fails(project, cfg, frozen):
    split = project / "data/processed/split.parquet"
    split.write_bytes(split.read_bytes() + b"x")
    res = _check(cfg)
    assert not res["passed"] and "files" in _failed(res)
    assert not res["numeric_only"]


def test_tampered_config_fails(project, cfg, frozen):
    config = project / CONFIG
    config.write_text(config.read_text() + "\n# edit\n")
    res = _check(cfg)
    assert not res["passed"] and {"files", "original_bytes"} & _failed(res)


def test_tampered_manifest_fails_even_when_committed(project, cfg, frozen):
    path = project / "results/freeze_manifest.json"
    _edit_json(path, lambda o: o.update(selected_gbm_id="other"))
    assert "original_bytes" in _failed(_check(cfg))  # not the committed bytes
    _commit(project, "tamper manifest")
    res = _check(cfg)
    assert not res["passed"] and "selection" in _failed(res)


def test_tampered_frozen_original_fails(project, cfg, frozen):
    """The manifest hash of a tracked file is checked against the HEAD blob."""
    path = project / "results/freeze_manifest.json"

    def tamper(o):
        entry = next(f for f in o["files"] if f["path"] == "results/audit.json")
        entry["sha256"] = "0" * 64

    _edit_json(path, tamper)
    _commit(project, "tamper manifest hash")
    res = _check(cfg)
    assert not res["passed"] and "original_bytes" in _failed(res)


def test_approximate_agreement_never_passes_strict(project, cfg, frozen):
    path = project / "results/freeze_manifest.json"
    where = _first_float(json.loads(path.read_text())["self_check"]["predictions"])
    assert where is not None

    def nudge(o):
        node = o["self_check"]["predictions"]
        for k in where[:-1]:
            node = node[k]
        node[where[-1]] *= 1 + 1e-12

    _edit_json(path, nudge)
    _commit(project, "nudge a stored prediction")
    res = _check(cfg)
    assert not res["passed"] and res["numeric_only"]
    assert "self_check" in _failed(res)


def _nudge(path: Path, getter, factor: float = 1 + 1e-12) -> None:
    def fn(o):
        parent, key = getter(o)
        parent[key] *= factor

    _edit_json(path, fn)


def test_tiny_coefficient_change_is_portable_never_exact(project, cfg, frozen):
    """A float difference where config declares numeric is numeric (PORTABLE at most): the strict
    check still fails, and it is classed numeric only."""
    path = project / "results/freeze_manifest.json"
    recipes = json.loads(path.read_text())["recipes"]
    rid = next(
        k for k, v in recipes.items() if isinstance(v.get("frequency", {}).get("value"), float)
    )
    _nudge(path, lambda o: (o["recipes"][rid]["frequency"], "value"))
    _commit(project, "nudge a constant")
    res = _check(cfg)
    assert not res["passed"] and res["numeric_only"] and "recipes" in _failed(res)


def test_changed_selected_id_or_hyperparameter_is_structural(project, cfg, frozen):
    path = project / "results/freeze_manifest.json"
    orig = path.read_text()
    # a selected id
    _edit_json(path, lambda o: o.update(selected_glm_id="other"))
    _commit(project, "tamper id")
    res = _check(cfg)
    assert not res["passed"] and not res["numeric_only"] and "selection" in _failed(res)
    # a float hyperparameter (the Tweedie power), even a tiny change
    path.write_text(orig)
    _commit(project, "restore")
    assert _check(cfg)["passed"]
    recipes = json.loads(orig)["recipes"]
    rid = next(k for k, v in recipes.items() if isinstance(v.get("tweedie_power"), float))
    _nudge(path, lambda o: (o["recipes"][rid], "tweedie_power"))
    _commit(project, "tamper power")
    res = _check(cfg)
    assert not res["passed"] and not res["numeric_only"] and "recipes" in _failed(res)
    # a float in a selection trace is numeric, a trace string is not
    path.write_text(orig)
    _commit(project, "restore again")
    _edit_json(path, lambda o: o["selection"].update(rule_version="x"))
    _commit(project, "tamper rule version")
    assert not _check(cfg)["numeric_only"]


def test_check_works_after_test_access_and_verifies_the_evidence(project, cfg, frozen, monkeypatch):
    from fremtpl2_pricing import evaluate

    _forbid_fitting(monkeypatch)
    evaluate.run_evaluate(cfg, CONFIG)
    before = _snapshot(project)
    res = _check(cfg)
    assert res["passed"], [c for c in res["checks"] if c["status"] == "fail"]
    assert "published_evidence" in {c["name"] for c in res["checks"]}
    assert _snapshot(project) == before
    with pytest.raises(freeze.FreezeError, match="test-state"):
        freeze.run_freeze(cfg, CONFIG, force=True)


def test_tampered_pointer_or_attempt_file_fails(project, cfg, frozen, monkeypatch):
    from fremtpl2_pricing import evaluate

    _forbid_fitting(monkeypatch)
    evaluate.run_evaluate(cfg, CONFIG)
    pointer = project / "results/metrics.json"
    attempt = project / "results/test/attempt-1/tables.json"
    good_pointer, good_attempt = pointer.read_text(), attempt.read_text()
    _edit_json(pointer, lambda o: o["source"].update(attempt_id="forged"))
    res = _check(cfg)
    assert "published_evidence" in _failed(res) and not res["numeric_only"]
    pointer.write_text(good_pointer)
    assert _check(cfg)["passed"]
    attempt.write_text(good_attempt.replace("0.", "1.", 1))
    assert "published_evidence" in _failed(_check(cfg))
    attempt.write_text(good_attempt)
    assert _check(cfg)["passed"]
    pointer.unlink()
    assert "published_evidence" in _failed(_check(cfg))


def test_journal_without_complete_fails(project, cfg, frozen):
    (project / "results/test_access.json").write_text(json.dumps({"events": []}))
    assert "published_evidence" in _failed(_check(cfg))


def test_untracked_manifest_fails(project, cfg, frozen):
    _git(project, "rm", "-q", "--cached", "results/freeze_manifest.json")
    _git(project, "commit", "-q", "-m", "untrack the manifest")
    assert "original_bytes" in _failed(_check(cfg))


# --------------------------------------------------------------------------- compare_values


def test_compare_values_rules():
    base = {"a": 1.0, "ids": {"selected": "x", "power": 1.5}, "n": 3, "t": {"seconds": 1.0}}
    same = {"a": 1.0, "ids": {"selected": "x", "power": 1.5}, "n": 3, "t": {"seconds": 9.0}}
    assert freeze.compare_values(base, same, drop={"seconds"}).ok
    assert not freeze.compare_values(base, same).ok
    near = {**same, "a": 1.0 + 1e-10}
    strict = freeze.compare_values(base, near, drop={"seconds"})
    assert not strict.ok and strict.numeric_only and strict.max_abs > 0
    loose = freeze.compare_values(base, near, rtol=1e-6, atol=0.0, drop={"seconds"})
    assert loose.ok and loose.n_within_tolerance == 1
    # ints, strings, list lengths and exact_keys are never tolerated
    assert not freeze.compare_values(base, {**same, "n": 4}, rtol=1.0, drop={"seconds"}).ok
    exact = {**same, "ids": {"selected": "x", "power": 1.5 + 1e-12}}
    cmp = freeze.compare_values(base, exact, rtol=1e-6, drop={"seconds"}, exact_keys={"ids"})
    assert not cmp.ok and not cmp.numeric_only
    assert not freeze.compare_values([1.0], [1.0, 2.0], rtol=1.0).ok


# --------------------------------------------------------------------------- rework round 2


def test_byte_different_model_file_is_structural_and_blocks_portability(project, cfg, frozen):
    model = next((project / "data/processed/models").glob("*.txt"))
    model.write_text(model.read_text() + "\n# regenerated with other bytes\n")
    res = _check(cfg)
    assert "model_files" in _failed(res) and not res["numeric_only"]


def _evaluated(project, cfg, monkeypatch):
    from fremtpl2_pricing import evaluate

    _forbid_fitting(monkeypatch)
    evaluate.run_evaluate(cfg, CONFIG)
    return project / "results/test/attempt-1"


def _evidence_failure(cfg) -> str:
    res = _check(cfg)
    assert "published_evidence" in _failed(res) and not res["numeric_only"]
    return json.dumps(next(c for c in res["checks"] if c["name"] == "published_evidence"))


def test_unexpected_or_missing_attempt_entries_fail(project, cfg, frozen, monkeypatch):
    adir = _evaluated(project, cfg, monkeypatch)
    assert _check(cfg)["passed"]
    extra = adir / "extra.txt"
    extra.write_text("x")
    assert "unexpected entry" in _evidence_failure(cfg)
    extra.unlink()
    (adir / "nested").mkdir()
    (adir / "nested/more.txt").write_text("x")
    assert "unexpected entry" in _evidence_failure(cfg)
    (adir / "nested/more.txt").unlink()
    (adir / "nested").rmdir()
    victim = adir / "tables.json"
    saved = victim.read_bytes()
    victim.unlink()
    assert "missing" in _evidence_failure(cfg)
    victim.write_bytes(saved)
    assert _check(cfg)["passed"]


def test_pointer_resolving_outside_the_attempts_directory_fails(project, cfg, frozen, monkeypatch):
    adir = _evaluated(project, cfg, monkeypatch)
    journal = project / "results/test_access.json"
    good = journal.read_text()
    outside = project / "results/elsewhere"
    adir.rename(outside)
    adir.symlink_to(outside, target_is_directory=True)  # the attempt directory is a symlink
    assert "resolves outside" in _evidence_failure(cfg)
    adir.unlink()
    outside.rename(adir)
    journal.write_text(good.replace("results/test/attempt-1", "results/test/../elsewhere"))
    assert "published_evidence" in _failed(_check(cfg))
    journal.write_text(good)
    assert _check(cfg)["passed"]


def test_symlinked_attempt_file_or_model_file_fails(project, cfg, frozen, monkeypatch, tmp_path):
    adir = _evaluated(project, cfg, monkeypatch)
    victim = adir / "tables.json"
    saved = victim.read_bytes()
    elsewhere = tmp_path / "copy.json"
    elsewhere.write_bytes(saved)
    victim.unlink()
    victim.symlink_to(elsewhere)  # same bytes, but not the published file
    assert "symlinked" in _evidence_failure(cfg)
    victim.unlink()
    victim.write_bytes(saved)
    model = next((project / "data/processed/models").glob("*.txt"))
    content = model.read_bytes()
    moved = tmp_path / "model.txt"
    moved.write_bytes(content)
    model.unlink()
    model.symlink_to(moved)
    res = _check(cfg)
    assert "files" in _failed(res) and not res["numeric_only"]


def test_duplicate_journal_identity_is_rejected(project, cfg, frozen, monkeypatch):
    _evaluated(project, cfg, monkeypatch)
    events = json.loads((project / "results/test_access.json").read_text())
    assert freeze._evidence_inventory(
        project,
        freeze.project_paths(cfg, project),
        [*events["events"], dict(next(e for e in events["events"] if e["event"] == "COMPLETE"))],
        next(e for e in events["events"] if e["event"] == "COMPLETE"),
    )


def test_evidence_identity_lists_journal_pointer_attempt_and_manifest(
    project, cfg, frozen, monkeypatch
):
    assert freeze.evidence_identity(cfg, project) is None
    _evaluated(project, cfg, monkeypatch)
    ident = freeze.evidence_identity(cfg, project)
    assert {"journal_sha256", "pointer_sha256", "manifest_sha256", "attempt_id", "attempt"} <= set(
        ident
    )


def test_strict_json_rejects_duplicate_keys():
    assert freeze.strict_json_loads('{"a": 1, "b": [1.5]}') == {"a": 1, "b": [1.5]}
    with pytest.raises(ValueError, match="duplicate"):
        freeze.strict_json_loads('{"a": 1, "a": 2}')
    with pytest.raises(ValueError, match="duplicate"):
        freeze.strict_json_loads('{"x": {"a": 1, "a": 1}}')


def test_compare_values_non_finite_shape_and_tolerance_formula():
    cv = freeze.compare_values
    nan, inf = float("nan"), float("inf")
    assert cv({"a": nan, "b": inf}, {"a": nan, "b": inf}).ok  # identical non-finite
    assert not cv({"a": 1.0}, {"a": nan}, rtol=1.0, atol=1.0).ok
    assert not cv({"a": 1.0}, {"a": inf}, rtol=1.0, atol=1.0).ok
    assert not cv({"a": nan}, {"a": 1.0}, rtol=1.0, atol=1.0).ok
    # formula |a - b| <= atol + rtol * |a|, a = the committed (first) value
    assert cv([100.0], [101.0], rtol=0.01).ok
    assert not cv([100.0], [101.0001], rtol=0.01).ok
    assert cv([0.0], [1e-9], atol=1e-9).ok and not cv([0.0], [1e-8], atol=1e-9).ok
    assert not cv([1.0], [1.0 + 1e-9 + 1e-6 + 1e-12], rtol=1e-6, atol=1e-9).ok
    assert cv([0.0], [1e-9], rtol=1.0).ok is False  # the reference is the committed value (0)
    # type, key and list differences are shape differences, never tolerated
    assert not cv({"a": 1}, {"a": 1.0}, rtol=1.0).ok
    assert not cv({"a": True}, {"a": 1}, rtol=1.0).ok
    assert not cv({"a": 1.0}, {"a": 1.0, "b": 2.0}, rtol=1.0).ok
    assert not cv([1.0, 2.0], [2.0, 1.0], rtol=1e-9).ok  # order matters
    shape = cv({"a": 1.0, "b": [1.0]}, {"a": 1.0 + 1e-12, "b": [1.0, 2.0]}, rtol=1e-6)
    assert not shape.ok and shape.n_within_tolerance == 0 and shape.n_numeric == 0
    # an exact key covers all its descendants, including lists of floats
    deep = cv({"params": {"x": [{"y": 1.0}]}}, {"params": {"x": [{"y": 1.0 + 1e-12}]}},
              rtol=1e-6, exact_keys={"params"})  # fmt: skip
    assert not deep.ok and not deep.numeric_only
