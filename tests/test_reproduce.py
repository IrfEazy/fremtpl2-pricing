"""``fremtpl2 reproduce``: clone, steps, comparison with the committed baseline and verdicts.

Offline: a tiny git repository stands in for the project and the steps are fake commands run by an
injected runner (no uv, no network, no real data, no pipeline run).
"""

import copy
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fremtpl2_pricing import freeze, reproduce
from synth import _DEFAULTS

PASS = {"passed": True, "numeric_only": True}
NUMERIC = {"passed": False, "numeric_only": True}
STRUCTURAL = {"passed": False, "numeric_only": False}


def _png(value: int = 10, level: int = 9) -> bytes:
    """A tiny PNG; ``level`` changes the encoding bytes, ``value`` the decoded pixels."""
    buf = io.BytesIO()
    pixels = np.full((4, 4, 3), value, dtype=np.uint8)
    Image.fromarray(pixels).save(buf, format="PNG", compress_level=level)
    return buf.getvalue()


def _git(root: Path, *args: str) -> None:
    cmd = ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "-c",
           "commit.gpgsign=false", *args]  # fmt: skip
    subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=True)


def _write(root: Path, rel: str, text: str | bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text if isinstance(text, bytes) else text.encode())


@pytest.fixture
def cfg():
    cfg = copy.deepcopy(_DEFAULTS)
    cfg["reproduce"]["steps"] = [
        {"name": "setup", "cmd": ["setup"], "environment": True},
        {"name": "regen", "cmd": ["regen", "{config}"]},
        {"name": "freeze_check", "cmd": ["fc"], "json_result": True},
        {"name": "report", "cmd": ["report"]},
    ]
    cfg["reproduce"]["tool_commands"] = ["setup"]
    return cfg


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "src_repo"
    root.mkdir()
    _git(root, "init", "-q")
    _write(root, ".gitignore", "data/processed/\n")
    _write(root, "src/mod.py", "X = 1\n")
    _write(root, "results/a.json", json.dumps({"v": 1.0, "n": 3, "timing_seconds": 1.0}))
    _write(root, "results/rel.csv", "name,coef\nx,0.5\ny,1.25\n")
    _write(root, "results/metrics.json", '{"m": 0.123}')
    _write(root, "results/test/attempt-1/m.json", '{"m": 1}')
    _write(root, "reports/r.md", "# Report\n\nvalue 1.234567 end\n")
    _write(root, "reports/figures/f.png", _png())
    _write(root, "reports/test_evaluation.md", "# Test\n\nvalue 0.5\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "baseline")
    return root


def _runner(check=PASS, edit=None, codes=None, seen=None, regenerate=True):
    """Fake step runner: `regen` restores the committed outputs the clone deleted (as a faithful
    pipeline regenerates them; ``regenerate=False`` skips them) and calls ``edit(clone)``; `fc`
    prints ``check`` as JSON."""
    codes = codes or {}

    def run(cmd, cwd):
        if seen is not None:
            seen.append((cmd[0], (cwd / "data/processed").exists()))
        if cmd[0] in codes:
            return codes[cmd[0]], "", "boom"
        if cmd[0] == "regen":
            if regenerate:
                subprocess.run(
                    ["git", "checkout", "-q", "HEAD", "--", "results", "reports"],
                    cwd=cwd,
                    check=True,
                )
            if edit:
                edit(cwd)
        if cmd[0] == "fc":
            if check is None:
                return 1, "crashed", ""
            return (0 if check["passed"] else 1), json.dumps(check), ""
        return 0, "", ""

    return run


def _run(repo, cfg, **kw):
    return reproduce.run_reproduce(cfg, "configs/default.yaml", root=repo, **kw)


def _volatile_edit(clone):
    _write(clone, "results/a.json", json.dumps({"v": 1.0, "n": 3, "timing_seconds": 99.0}))


def test_exact_pass_and_record(repo, cfg):
    seen: list = []
    rec = _run(repo, cfg, runner=_runner(edit=_volatile_edit, seen=seen))
    assert rec["verdict"] == "EXACT_PASS"
    assert not any(has_data for _, has_data in seen[:1])  # the clone starts without data/
    assert [s["name"] for s in rec["steps"]] == ["setup", "regen", "freeze_check", "report"]
    assert all("seconds" in s and "exit_code" in s for s in rec["steps"])
    written = json.loads((repo / "results/reproduction.json").read_text())
    assert written["tested_commit"] == freeze.head_commit(repo)
    assert written["download_included"] is False
    assert {"platform", "machine", "cpu_count", "python", "start_utc", "end_utc"} <= set(written)
    # only the record was written in the source tree (the workdir is git-ignored)
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    )
    assert status.stdout.split() == ["??", "results/reproduction.json"]
    # a second run: the record does not make the tree dirty, the workdir needs --force-clean
    with pytest.raises(freeze.FreezeError, match="not empty"):
        _run(repo, cfg, runner=_runner())
    assert _run(repo, cfg, runner=_runner(), force_clean=True)["verdict"] == "EXACT_PASS"


def test_portable_pass_only_within_declared_tolerance(repo, cfg):
    def edit(clone):
        _write(
            clone, "results/a.json", json.dumps({"v": 1.0 + 1e-9, "n": 3, "timing_seconds": 2.0})
        )
        _write(clone, "results/rel.csv", "name,coef\nx,0.5000000001\ny,1.25\n")
        _write(clone, "reports/r.md", "# Report\n\nvalue 1.2345671 end\n")
        _write(clone, "reports/figures/f.png", _png(level=0))  # other encoding, same pixels

    rec = _run(repo, cfg, runner=_runner(check=NUMERIC, edit=edit))
    assert rec["verdict"] == "PORTABLE_NUMERIC_PASS"
    assert rec["freeze_check"]["passed"] is False  # strict stays failed
    cats = rec["comparison"]["categories"]
    assert cats["json"]["statuses"]["tolerance"] == 1 and cats["json"]["max_abs"] > 0
    assert cats["render"]["statuses"] == {"pixel_equal": 1}
    # the same differences with a passing strict check are still not an exact pass
    rec = _run(repo, cfg, runner=_runner(check=PASS, edit=edit), force_clean=True)
    assert rec["verdict"] == "PORTABLE_NUMERIC_PASS"


def test_out_of_tolerance_and_non_numeric_differences_fail(repo, cfg):
    edits = {
        "float": lambda c: _write(
            c, "results/a.json", '{"v": 1.001, "n": 3, "timing_seconds": 1.0}'
        ),
        "int": lambda c: _write(c, "results/a.json", '{"v": 1.0, "n": 4, "timing_seconds": 1.0}'),
        "key": lambda c: _write(c, "results/a.json", '{"v": 1.0, "n": 3, "new": 1}'),
        "text": lambda c: _write(c, "reports/r.md", "# Other\n\nvalue 1.234567 end\n"),
        "deleted": lambda c: (c / "results/rel.csv").unlink(),
    }
    for name, edit in edits.items():
        rec = _run(repo, cfg, runner=_runner(check=NUMERIC, edit=edit), force_clean=True)
        assert rec["verdict"] == "FAIL", name
    assert _run(repo, cfg, runner=_runner(check=STRUCTURAL), force_clean=True)["verdict"] == "FAIL"
    assert _run(repo, cfg, runner=_runner(check=None), force_clean=True)["verdict"] == "FAIL"


def test_exact_keys_are_never_tolerated(repo, cfg):
    _write(repo, "results/a.json", '{"config": {"lr": 0.1}}')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "config echo")
    edit = lambda c: _write(c, "results/a.json", '{"config": {"lr": 0.1000000001}}')  # noqa: E731
    assert _run(repo, cfg, runner=_runner(check=NUMERIC, edit=edit))["verdict"] == "FAIL"


@pytest.mark.parametrize(
    "rel", ["src/mod.py", "results/metrics.json", "results/test/attempt-1/m.json"]
)
def test_changed_source_or_published_evidence_fails(repo, cfg, rel):
    rec = _run(repo, cfg, runner=_runner(edit=lambda c: _write(c, rel, "changed\n")))
    assert rec["verdict"] == "FAIL"
    assert rel in rec["comparison"]["source_changed"]


def test_untracked_new_outputs_are_listed(repo, cfg):
    rec = _run(repo, cfg, runner=_runner(edit=lambda c: _write(c, "results/new.json", "{}")))
    assert rec["comparison"]["untracked_new"] == ["results/new.json"]
    assert rec["verdict"] == "FAIL"


def test_failed_step_stops_the_run(repo, cfg):
    seen: list = []
    rec = _run(repo, cfg, runner=_runner(codes={"regen": 2}, seen=seen))
    assert rec["verdict"] == "FAIL" and rec["failure"]["step"] == "regen"
    assert [name for name, _ in seen] == ["setup", "regen"]


def test_only_identified_tool_absence_or_transport_is_unsupported(repo, cfg):
    cases = {  # (step, exit code) -> verdict
        ("setup", 127): "UNSUPPORTED_ENVIRONMENT",  # a declared tool command is not installed
        ("regen", 75): "UNSUPPORTED_ENVIRONMENT",  # the transport exit code
        ("setup", 1): "FAIL",  # a failing setup is not an environment verdict by itself
        ("regen", 127): "FAIL",  # 127 of a command that is not a declared tool
        ("regen", 2): "FAIL",
    }
    for (step, code), verdict in cases.items():
        rec = _run(repo, cfg, runner=_runner(codes={step: code}), force_clean=True)
        assert rec["verdict"] == verdict, (step, code)


def test_refuses_dirty_source_tree(repo, cfg):
    _write(repo, "src/mod.py", "X = 2\n")
    with pytest.raises(freeze.FreezeError, match="not clean"):
        _run(repo, cfg, runner=_runner())
    assert not (repo / "results/reproduction.json").exists()


def test_workdir_must_stay_inside_processed(repo, cfg):
    cfg["reproduce"]["workdir"] = "../.."
    with pytest.raises(freeze.FreezeError, match="inside"):
        _run(repo, cfg, runner=_runner(), force_clean=True)
    assert (repo / "src/mod.py").exists()


def test_force_clean_deletes_only_the_workdir(repo, cfg):
    work = repo / "data/processed/reproduce"
    _write(repo, "data/processed/reproduce/stale.txt", "x")
    _write(repo, "data/processed/keep.txt", "x")
    _run(repo, cfg, runner=_runner(), force_clean=True)
    assert not (work / "stale.txt").exists() and (repo / "data/processed/keep.txt").exists()


def test_default_runner_reports_a_missing_command(tmp_path):
    code, _, err = reproduce.subprocess_runner(["definitely-not-a-command-xyz"], tmp_path)
    assert code == 127 and err


def test_outputs_are_deleted_before_the_steps_except_published_evidence(repo, cfg):
    seen: dict = {}

    def run(cmd, cwd):
        if cmd[0] == "setup":
            seen["deleted"] = [
                p for p in ("results/a.json", "reports/r.md", "reports/figures/f.png")
                if not (cwd / p).exists()
            ]  # fmt: skip
            seen["kept"] = [
                p for p in ("results/metrics.json", "results/test/attempt-1/m.json")
                if (cwd / p).exists()
            ]  # fmt: skip
        return 0, "", ""

    _run(repo, cfg, runner=run)
    assert len(seen["deleted"]) == 3 and len(seen["kept"]) == 2


def test_a_skipped_step_cannot_pass(repo, cfg):
    """A step that exits 0 without rewriting the committed outputs leaves them missing."""
    rec = _run(repo, cfg, runner=_runner(regenerate=False))
    assert rec["verdict"] == "FAIL"
    assert rec["comparison"]["categories"]["json"]["statuses"]["missing"] >= 1
    # the same run with the outputs regenerated passes
    assert _run(repo, cfg, runner=_runner(), force_clean=True)["verdict"] == "EXACT_PASS"


def test_formatting_only_differences_are_equivalent_never_exact(repo, cfg):
    def edit(clone):
        _write(clone, "results/rel.csv", "name,coef\nx,0.50\ny,1.25\n")  # 0.50 vs 0.5
        _write(clone, "results/a.json", '{"v": 1.00, "n": 3, "timing_seconds": 1.0}')

    rec = _run(repo, cfg, runner=_runner(edit=edit))
    statuses = rec["comparison"]["categories"]
    assert statuses["csv"]["statuses"] == {"equivalent": 1}
    assert statuses["json"]["statuses"].get("equivalent") == 1
    assert rec["verdict"] == "PORTABLE_NUMERIC_PASS"


def test_json_key_order_change_is_not_exact(repo, cfg):
    edit = lambda c: _write(c, "results/a.json", '{"n": 3, "v": 1.0, "timing_seconds": 1.0}')  # noqa: E731
    assert _run(repo, cfg, runner=_runner(edit=edit))["verdict"] == "PORTABLE_NUMERIC_PASS"


def test_numeric_strict_failure_with_identical_outputs_is_not_exact(repo, cfg):
    rec = _run(repo, cfg, runner=_runner(check=NUMERIC))
    assert rec["verdict"] == "PORTABLE_NUMERIC_PASS" and rec["freeze_check"]["passed"] is False


def test_untracked_file_outside_the_outputs_fails(repo, cfg):
    rec = _run(repo, cfg, runner=_runner(edit=lambda c: _write(c, "src/new.py", "X = 1\n")))
    assert rec["verdict"] == "FAIL" and rec["comparison"]["untracked_source"] == ["src/new.py"]


def test_rendered_report_must_be_byte_identical_but_figures_may_differ(repo, cfg):
    edit = lambda c: _write(c, "reports/test_evaluation.md", "# Test\n\nvalue 0.5000000001\n")  # noqa: E731
    rec = _run(repo, cfg, runner=_runner(edit=edit))
    assert rec["verdict"] == "FAIL"
    edit = lambda c: _write(c, "reports/figures/f.png", _png(level=0))  # noqa: E731
    rec = _run(repo, cfg, runner=_runner(edit=edit), force_clean=True)
    assert rec["verdict"] == "PORTABLE_NUMERIC_PASS"
    assert rec["comparison"]["categories"]["render"]["statuses"] == {"pixel_equal": 1}


def test_download_checksum_mismatch_fails_but_no_network_is_unsupported(repo, cfg):
    cfg["reproduce"]["steps"] = [{"name": "download", "cmd": ["dl"]}]
    assert _run(repo, cfg, runner=_runner(codes={"dl": 1}))["verdict"] == "FAIL"
    rec = _run(repo, cfg, runner=_runner(codes={"dl": 75}), force_clean=True)
    assert rec["verdict"] == "UNSUPPORTED_ENVIRONMENT"


def test_default_config_maps_download_network_failure_only():
    step = next(s for s in _DEFAULTS["reproduce"]["steps"] if s["name"] == "download")
    assert not step.get("environment") and not step.get("environment_exit_codes")
    assert _DEFAULTS["reproduce"]["transport_exit_codes"] == [75]
    assert _DEFAULTS["reproduce"]["tool_commands"] == ["uv", "git"]


@pytest.mark.parametrize("workdir", [".", "..", "../..", "../../.."])
def test_force_clean_refuses_the_root_an_ancestor_or_processed(repo, cfg, workdir):
    cfg["reproduce"]["workdir"] = workdir
    with pytest.raises(freeze.FreezeError, match="inside"):
        _run(repo, cfg, runner=_runner(), force_clean=True)
    assert (repo / "src/mod.py").exists() and (repo / ".git").exists()


def test_force_clean_refuses_a_symlink_or_a_file(repo, cfg, tmp_path):
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "keep.txt").write_text("x")
    (repo / "data/processed").mkdir(parents=True)
    (repo / "data/processed/reproduce").symlink_to(outside, target_is_directory=True)
    with pytest.raises(freeze.FreezeError, match="symlink"):
        _run(repo, cfg, runner=_runner(), force_clean=True)
    assert (outside / "keep.txt").exists()
    (repo / "data/processed/reproduce").unlink()
    (repo / "data/processed/reproduce").write_text("a file")
    with pytest.raises(freeze.FreezeError, match="not a directory"):
        _run(repo, cfg, runner=_runner(), force_clean=True)
    # a link elsewhere in the path that leaves processed is refused too
    (repo / "data/processed/reproduce").unlink()
    (repo / "data/processed/link").symlink_to(outside, target_is_directory=True)
    cfg["reproduce"]["workdir"] = "link/work"
    with pytest.raises(freeze.FreezeError, match="inside"):
        _run(repo, cfg, runner=_runner(), force_clean=True)
    assert (outside / "keep.txt").exists()


# --------------------------------------------------------------------------- rework round 2


def test_numeric_only_strict_failure_continues_to_the_report_and_reaches_portable(repo, cfg):
    seen: list = []
    rec = _run(repo, cfg, runner=_runner(check=NUMERIC, seen=seen))
    assert [name for name, _ in seen][-2:] == ["fc", "report"]  # the remaining steps ran
    assert rec["failure"] is None and rec["verdict"] == "PORTABLE_NUMERIC_PASS"
    assert rec["freeze_check"]["passed"] is False  # the strict check stays failed
    assert [s["name"] for s in rec["steps"]][-1] == "report"


def test_integrity_structural_or_execution_failure_of_the_check_stops(repo, cfg):
    for check in (STRUCTURAL, None, {"passed": True}):
        seen: list = []
        rec = _run(repo, cfg, runner=_runner(check=check, seen=seen), force_clean=True)
        assert rec["verdict"] == "FAIL" and rec["failure"]["step"] == "freeze_check"
        assert "report" not in [name for name, _ in seen]
    # a passing result with a failing exit code (or the reverse) is an execution failure
    rec = _run(repo, cfg, runner=_runner(check={"passed": False, "numeric_only": True},
                                         codes={"fc": 0}), force_clean=True)  # fmt: skip
    assert rec["verdict"] == "FAIL"


def test_record_defines_the_scope_and_lists_preserved_and_recomputable_inventories(repo, cfg):
    rec = _run(repo, cfg, runner=_runner())
    assert "held-out metrics are not recomputed" in rec["scope"]
    assert "preserved published evidence" in rec["scope"]
    assert set(rec["preserved_evidence"]["paths"]) == {
        "results/metrics.json",
        "results/test/attempt-1/m.json",
    }
    inv = rec["recomputable_inventory"]
    assert "results/a.json" in inv["compared"] and "results/metrics.json" not in inv["compared"]
    assert "results/a.json" in inv["purged"]
    assert rec["preserved_evidence"]["identity"] is None  # the synthetic project has no journal
    assert rec["tolerances"]["formula"] == "|a - b| <= atol + rtol * |a|, a = committed value"


def test_figure_with_different_pixels_or_undecodable_bytes_fails(repo, cfg):
    different = lambda c: _write(c, "reports/figures/f.png", _png(value=200))  # noqa: E731
    assert _run(repo, cfg, runner=_runner(edit=different))["verdict"] == "FAIL"
    garbage = lambda c: _write(c, "reports/figures/f.png", b"not a png")  # noqa: E731
    assert _run(repo, cfg, runner=_runner(edit=garbage), force_clean=True)["verdict"] == "FAIL"
    same = lambda c: _write(c, "reports/figures/f.png", _png(level=0))  # noqa: E731
    assert _png(level=0) != _png()
    rec = _run(repo, cfg, runner=_runner(edit=same), force_clean=True)
    assert rec["verdict"] == "PORTABLE_NUMERIC_PASS"  # pixel_equal is never exact


def test_comparison_rules_come_from_the_source_snapshot_not_the_clone(repo, cfg):
    """A step that loosens the config in the clone (file) or in memory cannot change the rules."""
    loose = lambda c: _write(c, "results/a.json", '{"v": 1.001, "n": 3, "timing_seconds": 1.0}')  # noqa: E731

    def edit(clone):
        loose(clone)
        _write(clone, "configs/default.yaml", "reproduce: {rtol: 1.0, atol: 1.0}\n")
        cfg["reproduce"]["rtol"] = 1.0  # an in-memory mutation after the snapshot
        cfg["reproduce"]["exact_keys"] = []
        cfg["reproduce"]["protected_paths"] = []

    rec = _run(repo, cfg, runner=_runner(check=NUMERIC, edit=edit))
    assert rec["verdict"] == "FAIL"
    assert rec["comparison"]["categories"]["json"]["statuses"].get("fail") == 1
    assert "configs/default.yaml" in rec["comparison"]["untracked_source"]
    assert rec["tolerances"]["rtol"] == 1.0e-6


def test_producer_exiting_zero_without_its_outputs_fails(repo, cfg):
    cfg["reproduce"]["steps"][1]["outputs"] = ["data/processed/split.parquet"]  # ignored path
    rec = _run(repo, cfg, runner=_runner())
    assert rec["verdict"] == "FAIL" and rec["failure"]["step"] == "regen"
    assert rec["failure"]["missing_outputs"] == ["data/processed/split.parquet"]
    assert rec["comparison"] is None
    ok = lambda c: _write(c, "data/processed/split.parquet", b"x")  # noqa: E731
    assert _run(repo, cfg, runner=_runner(edit=ok), force_clean=True)["verdict"] == "EXACT_PASS"

    # a symlink does not count as an output
    def link(c):
        _write(c, "outside.bin", b"x")
        (c / "data/processed").mkdir(parents=True, exist_ok=True)
        (c / "data/processed/split.parquet").symlink_to(c / "outside.bin")

    assert _run(repo, cfg, runner=_runner(edit=link), force_clean=True)["verdict"] == "FAIL"


def test_symlinked_or_escaping_artifact_fails(repo, cfg, tmp_path):
    secret = tmp_path / "elsewhere.json"
    secret.write_text(json.dumps({"v": 1.0, "n": 3, "timing_seconds": 1.0}))

    def edit(clone):
        (clone / "results/a.json").unlink()
        (clone / "results/a.json").symlink_to(secret)

    rec = _run(repo, cfg, runner=_runner(edit=edit))
    assert rec["verdict"] == "FAIL"
    assert "symlinked" in rec["comparison"]["categories"]["json"]["mismatches"][0]["details"]


def test_duplicate_keys_and_non_finite_numbers_in_regenerated_json_fail(repo, cfg):
    dup = lambda c: _write(c, "results/a.json", '{"v": 1.0, "v": 1.0, "n": 3, "timing_seconds": 1}')  # noqa: E731
    assert _run(repo, cfg, runner=_runner(edit=dup))["verdict"] == "FAIL"
    nan = lambda c: _write(c, "results/a.json", '{"v": NaN, "n": 3, "timing_seconds": 1.0}')  # noqa: E731
    assert _run(repo, cfg, runner=_runner(edit=nan), force_clean=True)["verdict"] == "FAIL"


def test_clone_commit_and_tool_versions_are_recorded(repo, cfg, monkeypatch):
    rec = _run(repo, cfg, runner=_runner())
    assert rec["clone_commit"] == rec["tested_commit"] == freeze.head_commit(repo)
    assert {"uv", "git", "python"} <= set(rec["tool_versions"])
    real = reproduce.head_commit
    monkeypatch.setattr(reproduce, "head_commit", lambda p: real(p) if p == repo else "0" * 40)
    seen: list = []
    rec = _run(repo, cfg, runner=_runner(seen=seen), force_clean=True)
    assert rec["verdict"] == "FAIL" and rec["failure"]["step"] == "clone" and not seen


def test_undeclared_untracked_output_fails_but_declared_one_passes(repo, cfg):
    rec = _run(repo, cfg, runner=_runner(edit=lambda c: _write(c, "results/new.json", "{}")))
    assert rec["verdict"] == "FAIL"
    assert rec["comparison"]["untracked_undeclared"] == ["results/new.json"]
