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


# --------------------------------------------------------------------------- declared deviations

SRC = Path(__file__).resolve().parents[1] / "src" / "fremtpl2_pricing"
OLD_HASH = "1" * 64
ADDED = [
    "models.gbm",
    "models.bootstrap.interval",
    "models.bootstrap.quantile_method",
    "models.evaluation.lorenz_points",
]
OLD_DATE, NEW_DATE = "2026-11-07", "2026-11-08"


def _raw_manifest(date_0: str, date_1: str | None = None, extra: dict | None = None) -> str:
    """The download metadata as `manifest.write_manifest` writes it."""
    doc = {"datasets": [{"retrieved": date_0}, {"retrieved": date_1 or date_0}], **(extra or {})}
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def _amendment(cfg, **over) -> dict:
    """The real amendment with the pins of D2 recomputed for the synthetic project's config."""
    from fremtpl2_pricing.config import load_reproduce_config
    from fremtpl2_pricing.train import config_summary

    am = load_reproduce_config(REPRODUCE)["amendments"][0]
    d1, d2, *_ = am["declared_deviations"]
    d2["old_config_sha256"] = OLD_HASH
    d2["new_config_sha256"] = config_summary(cfg)["config_sha256"]
    d2["added_value_sha256"] = {
        k: freeze.canonical_sha256(freeze._ptr_get(cfg, k.replace(".", "/"))) for k in ADDED
    }
    d2["static_evidence"]["source_dir"] = str(SRC)  # the real modules (absolute path)
    for key, value in over.items():
        (d1 if key.startswith("d1_") else d2)[key.split("_", 1)[1]] = value  # d1_x / d2_x
    return am


def _pins(cfg) -> dict:
    return _amendment(cfg)["declared_deviations"][1]["added_value_sha256"]


def _regenerated(project, cfg, old_edit=None, new_edit=None, old_raw=None, new_raw=None):
    """The state of the real situation: the frozen (committed) training output and download
    metadata are the OLD ones; the working tree holds the regenerated NEW ones."""
    raw = project / "data/raw/manifest.json"
    raw.write_text(old_raw or _raw_manifest(OLD_DATE))
    freeze.run_freeze(cfg, CONFIG)
    cv_path, mf_path = project / "results/cv_glm.json", project / "results/freeze_manifest.json"
    new_bytes = cv_path.read_bytes()
    old = json.loads(new_bytes)
    for dotted in ADDED:
        node = old["config"]
        *head, last = dotted.split(".")
        for part in head:
            node = node[part]
        del node[last]
    old["config"]["config_sha256"] = OLD_HASH
    if old_edit:
        old_edit(old)
    cv_path.write_text(json.dumps(old, indent=2) + "\n")

    def fix_manifest(m):
        m["training_record"]["glm_config_sha256_at_training"] = OLD_HASH
        entry = next(f for f in m["files"] if f["path"] == "results/cv_glm.json")
        entry["sha256"] = hashlib.sha256(cv_path.read_bytes()).hexdigest()
        entry["bytes"] = cv_path.stat().st_size

    _edit_json(mf_path, fix_manifest)
    _commit(project, "frozen with the old training output")
    new = json.loads(new_bytes)
    if new_edit:
        new_edit(new)
        new_bytes = (json.dumps(new, indent=2) + "\n").encode()
    cv_path.write_bytes(new_bytes)  # the regenerated output (uncommitted)
    raw.write_text(new_raw or _raw_manifest(NEW_DATE))


def _amended_check(cfg, am):
    merged = with_reproduce(cfg, load_reproduce_config(REPRODUCE))
    merged["amendments"] = [am]
    return freeze.run_freeze_check(merged, CONFIG)


def _dev(res, dev_id):
    return next(d for d in res["declared_deviations"]["deviations"] if d["id"] == dev_id)


def test_declared_deviations_cover_exactly_the_pinned_differences(project, cfg):
    _regenerated(project, cfg)
    res = _amended_check(cfg, _amendment(cfg))
    assert not res["passed"] and not res["numeric_only"]  # the strict result is unchanged
    assert _failed(res) == {"manifest_fields", "telemetry:results/cv_glm.json", "files"}
    dd = res["declared_deviations"]
    assert dd["all_failures_covered"] and dd["uncovered_failures"] == []
    assert sorted(dd["covered_failures"]) == sorted(_failed(res))
    assert _dev(res, "D1")["satisfied"] and _dev(res, "D2")["satisfied"]
    d1 = _dev(res, "D1")["applied"]
    assert [(a["path"], a["base"], a["new"]) for a in d1] == [
        ("/datasets/0/retrieved", OLD_DATE, NEW_DATE),
        ("/datasets/1/retrieved", OLD_DATE, NEW_DATE),
    ]
    places = [a["place"] for a in _dev(res, "D2")["applied"]]
    assert "manifest /training_record/glm_config_sha256_at_training" in places
    assert "results/cv_glm.json /config/models/gbm" in places


def test_strict_cli_exit_code_is_unchanged_with_deviations_present(project, cfg, capsys):
    _regenerated(project, cfg)
    assert _cli() == 1  # the real policy file carries the amendment; the strict exit code is 1
    out = json.loads(capsys.readouterr().out)
    assert out["passed"] is False and "declared_deviations" in out


def test_without_a_difference_nothing_is_applied(project, cfg, frozen):
    res = _amended_check(cfg, _amendment(cfg))
    assert res["passed"]
    dd = res["declared_deviations"]
    assert dd["covered_failures"] == [] and not dd["all_failures_covered"]
    assert all(d["applied"] == [] for d in dd["deviations"])


@pytest.mark.parametrize(
    "bad_date", ["20261108", "2026-11-08T10:00:00", "08/11/2026", "2026-13-01", "", "today"]
)
def test_d1_requires_an_iso_date(project, cfg, bad_date):
    _regenerated(project, cfg, new_raw=_raw_manifest(bad_date))
    res = _amended_check(cfg, _amendment(cfg))
    d1 = _dev(res, "D1")
    assert not d1["satisfied"] and "files" in res["declared_deviations"]["uncovered_failures"]
    assert not res["declared_deviations"]["all_failures_covered"]


def test_d1_covers_nothing_beyond_the_pinned_fields(project, cfg):
    changed = _raw_manifest(NEW_DATE, extra={"note": "x"})  # another field differs
    _regenerated(project, cfg, new_raw=changed)
    res = _amended_check(cfg, _amendment(cfg))
    assert not _dev(res, "D1")["satisfied"]
    assert "files" in res["declared_deviations"]["uncovered_failures"]
    # a different committed value than the pinned one
    _regenerated_pin = _amendment(cfg)
    _regenerated_pin["declared_deviations"][0]["fields"][0]["committed"] = "2026-11-06"
    assert not _dev(_amended_check(cfg, _regenerated_pin), "D1")["satisfied"]


def test_d1_keeps_the_raw_data_files_byte_equal(project, cfg):
    _regenerated(project, cfg)
    manifest = json.loads((project / "results/freeze_manifest.json").read_text())
    d1 = _amendment(cfg)["declared_deviations"][0]
    assert freeze._verify_download_metadata(d1, project, manifest)["satisfied"]
    entry = next(f for f in manifest["files"] if f["path"] == "data/raw/freMTPL2freq.parquet")
    entry["sha256"] = "0" * 64  # the raw file no longer has the frozen bytes
    res = freeze._verify_download_metadata(d1, project, manifest)
    assert not res["satisfied"] and any("sha256 mismatch" in p for p in res["problems"])
    assert res["covers_checks"] == {}


def test_the_retrieved_field_is_not_ignored_in_another_artifact(project, cfg):
    _regenerated(project, cfg)
    _edit_json(project / "results/cv_gbm.json", lambda o: o.update(retrieved=NEW_DATE))
    res = _amended_check(cfg, _amendment(cfg))
    dd = res["declared_deviations"]
    assert "telemetry:results/cv_gbm.json" in dd["uncovered_failures"]
    assert not dd["all_failures_covered"]


def test_d2_requires_the_pinned_added_keys_only(project, cfg):
    def extra_key(new):
        new["config"]["models"]["bootstrap"]["extra_option"] = 1

    _regenerated(project, cfg, new_edit=extra_key)
    res = _amended_check(cfg, _amendment(cfg))
    assert not _dev(res, "D2")["satisfied"]
    dd = res["declared_deviations"]
    assert "telemetry:results/cv_glm.json" in dd["uncovered_failures"]
    # a pinned key that is missing from the new echo
    _regenerated_missing = _amendment(cfg, added_keys=[*ADDED, "models.parallel.n_jobs"])
    assert not _dev(_amended_check(cfg, _regenerated_missing), "D2")["satisfied"]


def test_d2_requires_an_exactly_retained_committed_echo(project, cfg):
    def tweak(old):
        old["config"]["cv_folds"] = old["config"]["cv_folds"] + 1  # a changed retained value

    _regenerated(project, cfg, old_edit=tweak)
    res = _amended_check(cfg, _amendment(cfg))
    d2 = _dev(res, "D2")
    assert not d2["satisfied"] and any("restricted" in p for p in d2["problems"])
    assert not res["declared_deviations"]["all_failures_covered"]


def test_d2_requires_the_pinned_hashes_and_value_hashes(project, cfg):
    _regenerated(project, cfg)
    for over in (
        {"d2_old_config_sha256": "2" * 64},
        {"d2_new_config_sha256": "3" * 64},
        {"d2_added_value_sha256": {**_pins(cfg), "models.gbm": "4" * 64}},
    ):
        res = _amended_check(cfg, _amendment(cfg, **over))
        assert not _dev(res, "D2")["satisfied"], over
        assert not res["declared_deviations"]["all_failures_covered"]


def test_d2_changed_value_in_the_regenerated_echo_is_not_covered(project, cfg):
    def tweak(new):
        new["config"]["models"]["bootstrap"]["interval"] = "one_sided"

    _regenerated(project, cfg, new_edit=tweak)
    res = _amended_check(cfg, _amendment(cfg))
    assert not _dev(res, "D2")["satisfied"]


def test_d2_requires_the_rest_of_the_file_to_be_exact(project, cfg):
    def tweak(new):
        new["rows"]["train_policies"] += 1

    _regenerated(project, cfg, new_edit=tweak)
    res = _amended_check(cfg, _amendment(cfg))
    assert not _dev(res, "D2")["satisfied"]
    assert not res["declared_deviations"]["all_failures_covered"]


def test_d2_static_evidence_is_a_labelled_listing_not_a_pass_fail_input(project, cfg):
    _regenerated(project, cfg)
    merged = with_reproduce(cfg, load_reproduce_config(REPRODUCE))

    def d2(am):
        return freeze.verify_amendment(am, merged, project)[1]

    res = d2(_amendment(cfg))
    ev = res["static_evidence"]
    assert res["satisfied"] and "not a pass/fail input" in ev["label"]
    assert ev["listing_unchanged"] is True
    am = _amendment(cfg)
    am["declared_deviations"][1]["static_evidence"]["glm_training_modules"].append("freeze")
    am["declared_deviations"][1]["static_evidence"]["keys"]["models.gbm"]["modules"].append("train")
    res = d2(am)
    assert res["satisfied"]  # a differing listing is recorded, never decisive
    assert res["static_evidence"]["listing_unchanged"] is False
    assert res["static_evidence"]["glm_training_modules_matching"]["models.gbm"]


def test_the_same_hash_field_in_the_manifest_alone_is_not_covered(project, cfg, frozen):
    """A manifest whose training-record hash differs although cv_glm.json is unchanged."""
    path = project / "results/freeze_manifest.json"
    _edit_json(path, lambda m: m["training_record"].update(glm_config_sha256_at_training=OLD_HASH))
    _commit(project, "tamper the recorded hash")
    res = _amended_check(cfg, _amendment(cfg))
    assert "manifest_fields" in _failed(res)
    assert not res["declared_deviations"]["all_failures_covered"]


def test_unreadable_manifest_or_unknown_kind_covers_nothing(project, cfg, frozen):
    am = _amendment(cfg)
    am["declared_deviations"][0]["kind"] = "unknown"
    res = _amended_check(cfg, am)
    assert not _dev(res, "D1")["satisfied"]


# --------------------------------------------------------------------------- D3: markdown telemetry

GLM_MD = (
    "# GLM\n\nConfig hash: `{h}`\n\n## Fits\n\n| a | b |\n|---|---|\n| 1 | 2.5 |\n\n"
    "## Timing (seconds)\n\n| stage | seconds |\n|---|---|\n| fit | {t1} |\n| total | {t2} |\n"
)
GBM_MD = (
    "# GBM\n\n## Tuning\n\n- tuning: 5 rounds\n\n## Checks and timings\n\n- folds_complete: True\n"
    "- passed: True\n\n- tuning: {t1} s\n- total: {t2} s\n"
)


def _d3(cfg) -> dict:
    return _amendment(cfg)["declared_deviations"][2]


def _glm_md(h=OLD_HASH, t=(1.5, 3.5)) -> str:
    """glm_cv.md as the producer renders it (cv_report formatter)."""
    from fremtpl2_pricing import cv_report

    return GLM_MD.format(h=h, t1=cv_report._f(t[0]), t2=cv_report._f(t[1]))


def _gbm_md(t=(1.5, 3.5)) -> str:
    from fremtpl2_pricing import gbm_report

    return GBM_MD.format(t1=gbm_report._f(t[0], 1), t2=gbm_report._f(t[1], 1))


def _write_timing(project, glm_t=(1.5, 3.5), gbm_t=(1.5, 3.5)):
    """The regenerated JSON whose timing_seconds the markdown reports render."""
    for rel, (a, b), keys in (
        ("results/cv_glm.json", glm_t, ("fit", "total")),
        ("results/cv_gbm.json", gbm_t, ("tuning", "total")),
    ):
        path = project / rel
        doc = json.loads(path.read_text()) if path.is_file() else {}
        doc["timing_seconds"] = {keys[0]: a, keys[1]: b}
        path.write_text(json.dumps(doc, indent=2) + "\n")


def _md_project(project, glm_t=(1.5, 3.5), gbm_t=(1.5, 3.5), h=OLD_HASH):
    (project / "reports").mkdir(exist_ok=True)
    (project / "reports/glm_cv.md").write_text(_glm_md(h, glm_t))
    (project / "reports/gbm_cv.md").write_text(_gbm_md(gbm_t))
    _write_timing(project, glm_t, gbm_t)


def _d3_result(cfg, **over) -> dict:
    return _dev(_amended_check(cfg, _amendment(cfg, **over)), "D3")


def test_d3_pins_come_from_the_policy_file(cfg):
    d3 = _d3(cfg)
    assert d3["kind"] == "markdown_telemetry" and d3["volatile_keys"] == ["timing_seconds"]
    assert [f["path"] for f in d3["files"]] == [
        "reports/glm_cv.md", "reports/glm_sensitivities.md", "reports/gbm_cv.md"
    ]  # fmt: skip


def test_d3_a_timing_cell_change_in_the_pinned_section_is_covered(project, cfg, frozen):
    _md_project(project)
    _commit(project, "reports")
    _md_project(project, glm_t=(7.25, 100.0), gbm_t=(9.0, 10.5))
    res = _d3_result(cfg)
    assert res["satisfied"] and res["problems"] == []
    assert len(res["applied"]) == 4
    assert {a["place"].split(":")[0] for a in res["applied"]} == {
        "reports/glm_cv.md", "reports/gbm_cv.md"
    }  # fmt: skip


def test_d3_nothing_changed_applies_nothing(project, cfg, frozen):
    _md_project(project)
    _commit(project, "reports")
    res = _d3_result(cfg)
    assert res["satisfied"] and res["applied"] == []


@pytest.mark.parametrize(
    "edit",
    [
        lambda t: t.replace("| 1 | 2.5 |", "| 1 | 2.6 |"),  # a cell in a non-pinned section
        lambda t: t.replace("# GLM", "# GLM2"),  # another byte
        lambda t: t.replace(
            "| fit | 1.5 |", "| fit2 | 7 |"
        ),  # the label cell, not the pinned column
        lambda t: t.replace("| fit | 1.5 |", "| fit | n/a |"),  # a non-numeric cell
        lambda t: t.replace("| fit | 1.5 |", "| fit | nan |"),
        lambda t: t.replace("| fit | 1.5 |", "| fit | -1.5 |"),  # negative
        lambda t: t.replace("| fit | 1.5 |", "| fit | 1e999 |"),  # not finite
        lambda t: t.replace("| fit | 1.5 |", "| fit |  1.5 |"),  # padding is a byte change
        lambda t: t.replace("## Timing (seconds)", "## Timing (seconds) "),  # the heading
        lambda t: t + "extra\n",  # a line more
    ],
)
def test_d3_any_other_change_is_not_covered(project, cfg, frozen, edit):
    _md_project(project)
    _commit(project, "reports")
    path = project / "reports/glm_cv.md"
    path.write_text(edit(path.read_text()))
    res = _d3_result(cfg)
    assert not res["satisfied"] and res["problems"] and res["applied"] == []


def test_d3_bullets_only_after_the_checks_and_with_the_same_key(project, cfg, frozen):
    _md_project(project)
    _commit(project, "reports")
    path = project / "reports/gbm_cv.md"
    for old, new in (
        ("- folds_complete: True", "- folds_complete: False"),  # a check, before the marker
        ("- passed: True", "- passed: False"),
        ("- tuning: 1.5 s", "- other: 1.5 s"),
        ("- tuning: 1.5 s", "- tuning: fast s"),
        ("- tuning: 5 rounds", "- tuning: 6 rounds"),  # another section
    ):
        path.write_text(_gbm_md().replace(old, new))
        assert not _d3_result(cfg)["satisfied"], (old, new)


def test_d3_other_files_and_missing_pins_are_not_covered(project, cfg, frozen):
    _md_project(project)
    (project / "reports/data_audit.md").write_text("# Audit\n\n| a | b |\n|--|--|\n| x | 1.5 |\n")
    _commit(project, "reports")
    (project / "reports/data_audit.md").write_text("# Audit\n\n| a | b |\n|--|--|\n| x | 9.5 |\n")
    merged = with_reproduce(cfg, load_reproduce_config(REPRODUCE))
    d3 = freeze.verify_amendment(_amendment(cfg), merged, project)[2]
    assert d3["satisfied"] and d3["covers_files"] == []  # the file is not pinned: not covered
    am = _amendment(cfg)
    am["declared_deviations"][2]["volatile_keys"] = ["not_volatile"]
    _md_project(project, glm_t=(2.0, 3.5))
    assert not _dev(_amended_check(cfg, am), "D3")["satisfied"]


def test_d3_covers_the_d2_hash_line_together_with_timing_cells(project, cfg):
    _md_project(project)
    _commit(project, "reports")
    _regenerated(project, cfg)
    new_hash = _amendment(cfg)["declared_deviations"][1]["new_config_sha256"]
    _md_project(project, glm_t=(8.5, 9.5), gbm_t=(1.5, 3.5), h=new_hash)
    res = _amended_check(cfg, _amendment(cfg))
    assert _dev(res, "D2")["satisfied"] and _dev(res, "D3")["satisfied"]
    assert any(a["place"].startswith("reports/glm_cv.md:") for a in _dev(res, "D3")["applied"])
    # a hash line that is not the pinned replacement is not covered by either deviation
    _md_project(project, glm_t=(8.5, 9.5), h="9" * 64)
    res = _amended_check(cfg, _amendment(cfg))
    assert not _dev(res, "D2")["satisfied"]
    assert not _dev(res, "D3")["satisfied"]
    # the hash line alone, without D2 satisfied (wrong pin), is not covered
    _md_project(project, h=new_hash)
    res = _amended_check(cfg, _amendment(cfg, d2_old_config_sha256="2" * 64))
    assert not _dev(res, "D3")["satisfied"]


def test_d3_changed_cell_must_equal_the_producers_rendering_of_the_json_value(project, cfg, frozen):
    _md_project(project)
    _commit(project, "reports")
    _md_project(project, glm_t=(7.25, 100.0), gbm_t=(9.0, 10.5))
    assert _d3_result(cfg)["satisfied"]
    # markdown says 7.25 but the regenerated JSON says another value
    _write_timing(project, glm_t=(7.26, 100.0), gbm_t=(9.0, 10.5))
    res = _d3_result(cfg)
    assert not res["satisfied"] and any("producer's rendering" in p for p in res["problems"])
    # the bullet formatter has one decimal: 9.04 s is not its rendering of 9.04
    _md_project(project, glm_t=(7.25, 100.0), gbm_t=(9.0, 10.5))
    (project / "reports/gbm_cv.md").write_text(_gbm_md((9.0, 10.5)).replace("9.0 s", "9.04 s"))
    _write_timing(project, glm_t=(7.25, 100.0), gbm_t=(9.04, 10.5))
    assert not _d3_result(cfg)["satisfied"]


def test_d3_missing_json_or_stage_is_not_covered(project, cfg, frozen):
    _md_project(project)
    _commit(project, "reports")
    _md_project(project, glm_t=(7.25, 100.0))
    path = project / "results/cv_glm.json"
    doc = json.loads(path.read_text())
    del doc["timing_seconds"]["fit"]
    path.write_text(json.dumps(doc))
    assert not _d3_result(cfg)["satisfied"]
    path.write_text("not json")
    assert not _d3_result(cfg)["satisfied"]


def test_d3_anchors_must_exist_exactly_once(project, cfg, frozen):
    _md_project(project)
    _commit(project, "reports")
    glm = _glm_md((1.5, 3.5))
    dup = glm + glm[glm.index("## Timing") :]
    (project / "reports/glm_cv.md").write_text(dup.replace("1.5", "7.25").replace("3.5", "100"))
    assert not _d3_result(cfg)["satisfied"]  # duplicate heading (and different line count)
    gbm = _gbm_md((1.5, 3.5))
    for text in (gbm.replace("- passed: True\n", "- passed: True\n- passed: True\n"),
                 gbm.replace("- passed: True", "- ok: True")):  # fmt: skip
        (project / "reports/gbm_cv.md").write_text(text)
        assert not _d3_result(cfg)["satisfied"]
    (project / "reports/gbm_cv.md").write_text(gbm)
    # a missing table header
    base = _glm_md((1.5, 3.5)).replace("| stage | seconds |", "| st | secs |")
    (project / "reports/glm_cv.md").write_text(base)
    _commit(project, "no header")
    (project / "reports/glm_cv.md").write_text(base.replace("1.5", "7.25"))
    _write_timing(project, glm_t=(7.25, 3.5))
    assert not _d3_result(cfg)["satisfied"]


# --------------------------------------------------------------------------- D4: derived hashes

IN_REL, ART_REL = "results/cv_glm.json", "results/interpretation.json"
D2_PRIOR = {"id": "D2", "satisfied": True, "covers_files": [IN_REL]}


def _d4_spec() -> dict:
    return load_reproduce_config(REPRODUCE)["amendments"][0]["derived_hashes"][0]


def _src_hash(cfg, data: bytes, rel: str = IN_REL) -> str:
    """The recorded hash, from the producer's own function."""
    from fremtpl2_pricing.interpret import source_entry_from_bytes

    vol = sorted(cfg["freeze"]["check"]["volatile_keys"])
    return source_entry_from_bytes(data, rel, vol)["sha256"]


def _interp(cfg, data: bytes, key="cv_glm", rel=IN_REL, extra=None) -> str:
    entry = {"path": rel, "sha256": _src_hash(cfg, data, rel)}
    return json.dumps({"sources": {key: entry, **(extra or {})}, "other": 1}, indent=2) + "\n"


def _d4_project(project, cfg, old=b'{"config": {"h": 1}, "timing_seconds": {"t": 1}}',
                new=b'{"config": {"h": 2}, "timing_seconds": {"t": 9}}'):  # fmt: skip
    """HEAD: input and artifact in step; working tree: the regenerated input and artifact."""
    merged = with_reproduce(cfg, load_reproduce_config(REPRODUCE))
    (project / "results").mkdir(exist_ok=True)
    (project / IN_REL).write_bytes(old)
    (project / ART_REL).write_text(_interp(merged, old))
    if freeze.git(project, "status", "--porcelain", "--", "results").strip():
        _commit(project, "d4 base")
    (project / IN_REL).write_bytes(new)
    (project / ART_REL).write_text(_interp(merged, new))
    return merged


def _d4(merged, prior=(D2_PRIOR,), root=None):
    return freeze._verify_derived_hash(_d4_spec(), merged, root, list(prior))


def test_d4_spec_is_declared_in_the_policy_without_observed_values():
    spec = _d4_spec()
    assert spec["artifact"] == ART_REL and spec["sources_pointer"] == "/sources"
    assert not any(len(str(v)) == 64 for v in spec.values())  # no pinned hash


def test_d4_derived_hash_is_covered_when_its_input_differs_only_by_a_covered_deviation(
    project, cfg
):
    merged = _d4_project(project, cfg)
    res = _d4(merged, root=project)
    assert res["satisfied"], res["problems"]
    assert res["covers_files"] == [ART_REL]
    [app] = res["applied"]
    assert app["key"] == "cv_glm" and app["covered_by"] == ["D2"]
    assert app["base"] != app["new"]
    assert app["base"] == _src_hash(merged, freeze.git_blob(project, "HEAD", IN_REL))
    assert app["new"] == _src_hash(merged, (project / IN_REL).read_bytes())


def test_d4_nothing_changed_applies_nothing(project, cfg):
    merged = _d4_project(project, cfg)
    freeze.git(project, "checkout", "--", ART_REL)
    res = _d4(merged, root=project)
    assert res["satisfied"] and not res["applied"] and res["covers_files"] == []


def test_d4_not_covered_when_the_input_has_an_uncovered_difference(project, cfg):
    merged = _d4_project(project, cfg)
    for prior in ((), ({**D2_PRIOR, "satisfied": False},), ({**D2_PRIOR, "covers_files": []},)):
        res = _d4(merged, prior, root=project)
        assert not res["satisfied"] and res["covers_files"] == []
        assert any("not covered" in p for p in res["problems"])


def test_d4_not_covered_when_the_recorded_hash_does_not_match_the_recomputation(project, cfg):
    merged = _d4_project(project, cfg)
    art = project / ART_REL
    doc = json.loads(art.read_text())
    doc["sources"]["cv_glm"]["sha256"] = "9" * 64  # not what the producer computes from the file
    art.write_text(json.dumps(doc, indent=2) + "\n")
    res = _d4(merged, root=project)
    assert not res["satisfied"] and any(
        "recomputation from the clone" in p for p in res["problems"]
    )
    # the committed (HEAD) value is not the recomputation from the HEAD blob either
    merged = _d4_project(project, cfg, old=b'{"config": {"h": 3}}', new=b'{"config": {"h": 4}}')
    doc = json.loads(freeze.git_blob(project, "HEAD", ART_REL))
    doc["sources"]["cv_glm"]["sha256"] = "8" * 64
    (project / ART_REL).write_text(_interp(merged, (project / IN_REL).read_bytes()))
    _commit_bytes = (json.dumps(doc, indent=2) + "\n").encode()
    (project / ART_REL).write_bytes(_commit_bytes)
    _commit(project, "bad head hash")
    (project / ART_REL).write_text(_interp(merged, (project / IN_REL).read_bytes()))
    res = _d4(merged, root=project)
    assert not res["satisfied"] and any("recomputation from HEAD" in p for p in res["problems"])


def test_d4_not_covered_for_a_key_or_path_outside_the_sources(project, cfg):
    merged = _d4_project(project, cfg)
    art = project / ART_REL
    doc = json.loads(art.read_text())
    doc["other"] = 2  # a difference outside /sources
    art.write_text(json.dumps(doc, indent=2) + "\n")
    res = _d4(merged, root=project)
    assert not res["satisfied"] and any("outside the covered hashes" in p for p in res["problems"])
    # the hash moves under a key that is only on one side: nothing is substituted
    merged = _d4_project(project, cfg)
    doc = json.loads(art.read_text())
    doc["sources"]["new_key"] = doc["sources"].pop("cv_glm")
    art.write_text(json.dumps(doc, indent=2) + "\n")
    res = _d4(merged, root=project)
    assert not res["applied"] and res["covers_files"] == []  # the difference stays unmatched
    # an untracked input is not a covered input
    merged = _d4_project(project, cfg)
    doc = json.loads(art.read_text())
    doc["sources"]["cv_glm"]["path"] = "results/untracked.json"
    art.write_text(json.dumps(doc, indent=2) + "\n")
    assert not _d4(merged, root=project)["satisfied"]


def test_d4_uses_the_producers_function_not_a_copy():
    import inspect

    src = inspect.getsource(freeze._verify_derived_hash)
    assert "source_entry_from_bytes" in src and "hashlib" not in src and "sort_keys" not in src


def test_d4_runs_after_the_deviations_it_depends_on(project, cfg, frozen):
    merged = _d4_project(project, cfg)
    am = {**load_reproduce_config(REPRODUCE)["amendments"][0], "declared_deviations": []}
    res = freeze.verify_amendment(am, merged, project)
    assert [r["id"] for r in res] == ["D4"] and not res[0]["satisfied"]  # no covering deviation
