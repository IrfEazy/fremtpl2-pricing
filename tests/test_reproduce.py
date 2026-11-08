"""``fremtpl2 reproduce``: clone, steps, comparison with the committed baseline and verdicts.

Offline: a tiny git repository stands in for the project and the steps are fake commands run by an
injected runner (no uv, no network, no real data, no pipeline run).
"""

import copy
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fremtpl2_pricing import freeze, reproduce
from fremtpl2_pricing.config import load_reproduce_config, with_reproduce
from synth import _DEFAULTS as _MAIN

# the verification policy lives in configs/reproduce.yaml, not in the frozen main config
_DEFAULTS = with_reproduce(
    _MAIN, load_reproduce_config(Path(__file__).resolve().parents[1] / "configs/reproduce.yaml")
)


def _chk(status, numeric_only):
    return {"name": "files", "status": status, "numeric_only": numeric_only, "n_numeric": 0,
            "n_structural": 0, "mismatches": []}  # fmt: skip


PASS = {"passed": True, "numeric_only": True, "checks": [_chk("pass", True)]}
NUMERIC = {"passed": False, "numeric_only": True, "checks": [_chk("fail", True)]}
STRUCTURAL = {"passed": False, "numeric_only": False, "checks": [_chk("fail", False)]}


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
    written = json.loads((repo / "results/reproduction/run-1.json").read_text())
    assert written["run"] == 1 and rec["run"] == 1
    assert written["strict_verdict"] == written["verdict"] == written["amended_verdict"]
    assert written["tested_commit"] == freeze.head_commit(repo)
    assert written["download_included"] is False
    assert {"platform", "machine", "cpu_count", "python", "start_utc", "end_utc"} <= set(written)
    # only the record was written in the source tree (the workdir is git-ignored)
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    )
    assert status.stdout.split() == ["??", "results/reproduction/"]  # untracked directory
    # a second run: the record does not make the tree dirty, the workdir needs --force-clean
    with pytest.raises(freeze.FreezeError, match="not empty"):
        _run(repo, cfg, runner=_runner())
    assert _run(repo, cfg, runner=_runner(), force_clean=True)["verdict"] == "EXACT_PASS"
    assert sorted(p.name for p in (repo / "results/reproduction").iterdir()) == [
        "run-1.json",
        "run-2.json",
    ]


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


def test_reproduce_config_change_in_the_clone_is_a_source_change(repo, cfg):
    _write(repo, "configs/reproduce.yaml", "check: {}\nreproduce: {}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "policy")
    edit = lambda c: _write(c, "configs/reproduce.yaml", "check: {}\nreproduce: {rtol: 1.0}\n")  # noqa: E731
    rec = _run(repo, cfg, runner=_runner(edit=edit))
    assert rec["verdict"] == "FAIL"
    assert rec["comparison"]["source_changed"] == ["configs/reproduce.yaml"]


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


# --------------------------------------------------------------------------- run history


def test_history_is_never_overwritten(repo, cfg, monkeypatch):
    _run(repo, cfg, runner=_runner())
    first = (repo / "results/reproduction/run-1.json").read_bytes()
    monkeypatch.setattr(reproduce, "_history_files", lambda *a: {})  # a stale view of the directory
    with pytest.raises(freeze.FreezeError, match="never overwritten"):
        _run(repo, cfg, runner=_runner(), force_clean=True)
    assert (repo / "results/reproduction/run-1.json").read_bytes() == first


def test_numbers_increment_past_the_highest_existing_record(tmp_path):
    d = tmp_path / "h"
    pat = "run-{n}.json"
    names = [reproduce.write_run_record(d, pat, {"x": i}).name for i in range(2)]
    (d / "run-7.json").write_text("{}")
    names.append(reproduce.write_run_record(d, pat, {}).name)
    assert names == ["run-1.json", "run-2.json", "run-8.json"]
    assert json.loads((d / "run-2.json").read_text()) == {"x": 1, "run": 2}


def test_a_modified_committed_record_refuses_the_run(repo, cfg):
    _write(repo, "results/reproduction/run-1.json", '{"verdict": "FAIL"}')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "run-1")
    _run(repo, cfg, runner=_runner())  # run-2 is written next to the committed run-1
    assert (repo / "results/reproduction/run-2.json").is_file()
    _write(repo, "results/reproduction/run-1.json", '{"verdict": "EXACT_PASS"}')
    with pytest.raises(freeze.FreezeError, match="immutable"):
        _run(repo, cfg, runner=_runner(), force_clean=True)


def test_the_history_directory_is_not_compared_or_purged(repo, cfg):
    _write(repo, "results/reproduction/run-1.json", '{"verdict": "FAIL"}')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "run-1")
    rules = reproduce.snapshot_rules(cfg, repo, freeze.head_commit(repo))
    assert not any(
        p.startswith("results/reproduction/") for p in rules["compared"] + rules["purge"]
    )
    rec = _run(repo, cfg, runner=_runner())
    assert rec["strict_verdict"] == "EXACT_PASS"
    assert "results/reproduction/run-1.json" not in rec["recomputable_inventory"]["compared"]


# --------------------------------------------------------------------------- amended verdict

D1 = {"id": "D1", "kind": "download_metadata", "satisfied": True, "problems": [],
      "applied": [{"place": "data/raw/manifest.json", "path": "/datasets/0/retrieved",
                   "base": "2026-11-07", "new": "2026-11-08"}],
      "covers_checks": {"files": ["sha256 mismatch: data/raw/manifest.json"]},
      "covers_files": []}  # fmt: skip
D2 = {"id": "D2", "kind": "glm_config_echo", "satisfied": True, "problems": [],
      "applied": [{"place": "results/a.json /config/config_sha256", "base": "o", "new": "n"}],
      "covers_checks": {}, "covers_files": ["results/a.json"]}  # fmt: skip


def _failed_check(name="files", path="sha256 mismatch: data/raw/manifest.json", extra=()):
    fail = {"name": name, "status": "fail", "numeric_only": False, "n_numeric": 0,
            "n_structural": 1 + len(extra),
            "mismatches": [{"path": p, "kind": "structural"} for p in (path, *extra)]}  # fmt: skip
    ok = {"name": "selection", "status": "pass", "numeric_only": True, "n_numeric": 0,
          "n_structural": 0, "mismatches": []}  # fmt: skip
    return {"passed": False, "numeric_only": False, "checks": [ok, fail]}


def _amended_cfg(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["amendments"] = [
        {"id": "A1", "date": "2026-11-08", "post_hoc": True, "revealed_by": "run-1",
         "reason": "r", "original_policy_sha256": "0" * 64, "declared_deviations": []}
    ]  # fmt: skip
    cfg["reproduce_policy_sha256"] = "f" * 64
    return cfg


def _edit_a(clone):
    _write(clone, "results/a.json", json.dumps({"v": 9.0, "n": 3, "timing_seconds": 1.0}))


@pytest.fixture
def acfg(cfg):
    return _amended_cfg(cfg)


def _with_deviations(monkeypatch, results):
    monkeypatch.setattr(reproduce, "verify_amendment", lambda *a, **k: copy.deepcopy(results))


def test_covered_failures_continue_and_give_the_amended_pass_but_keep_the_strict_failure(
    repo, acfg, monkeypatch
):
    _with_deviations(monkeypatch, [D1, D2])
    rec = _run(repo, acfg, runner=_runner(check=_failed_check(), edit=_edit_a))
    assert [s["name"] for s in rec["steps"]] == ["setup", "regen", "freeze_check", "report"]
    assert rec["failure"] is None
    assert rec["strict_verdict"] == rec["verdict"] == "FAIL"
    assert rec["freeze_check"]["passed"] is False  # the strict status is recorded as failed
    assert rec["amended_verdict"] == "PASS_WITH_DECLARED_DEVIATIONS"
    am = rec["amendment"]
    assert am["original_policy_sha256"] == "0" * 64 and am["amended_policy_sha256"] == "f" * 64
    assert [d["id"] for d in am["applied_deviations"]] == ["D1", "D2"]
    assert am["applied_deviations"][0]["applied"][0]["base"] == "2026-11-07"
    assert am["applied_deviations"][0]["applied"][0]["new"] == "2026-11-08"
    assert am["unmatched_differences"] == []
    inv = am["inventory"]
    assert inv["steps"]["ran"] == inv["steps"]["configured"]
    assert {c["name"]: c["covered_by_deviation"] for c in inv["freeze_checks"]} == {
        "selection": False, "files": True,
    }  # fmt: skip
    assert {f["path"]: f["covered_by_deviation"] for f in inv["comparison_files"]}[
        "results/a.json"
    ]  # fmt: skip
    assert len(inv["comparison_files"]) == inv["n_expected_files"]
    written = json.loads((acfg and (repo / "results/reproduction/run-1.json")).read_text())
    assert written["amended_verdict"] == "PASS_WITH_DECLARED_DEVIATIONS"


def test_an_uncovered_failure_stops_the_run_and_gives_no_amended_pass(repo, acfg, monkeypatch):
    _with_deviations(monkeypatch, [D1, D2])
    check = _failed_check(extra=("sha256 mismatch: data/processed/split.parquet",))
    rec = _run(repo, acfg, runner=_runner(check=check, edit=_edit_a))
    assert rec["failure"]["step"] == "freeze_check"
    assert [s["name"] for s in rec["steps"]] == ["setup", "regen", "freeze_check"]
    assert rec["strict_verdict"] == rec["amended_verdict"] == "FAIL"
    assert rec["comparison"] is None


def test_an_unsatisfied_deviation_covers_nothing(repo, acfg, monkeypatch):
    bad = {**D1, "satisfied": False, "problems": ["not an ISO date"]}
    _with_deviations(monkeypatch, [bad, D2])
    rec = _run(repo, acfg, runner=_runner(check=_failed_check(), edit=_edit_a))
    assert rec["failure"]["step"] == "freeze_check" and rec["amended_verdict"] == "FAIL"


def test_an_unmatched_output_difference_blocks_the_amended_pass(repo, acfg, monkeypatch):
    _with_deviations(monkeypatch, [D1, {**D2, "covers_files": []}])
    rec = _run(repo, acfg, runner=_runner(check=_failed_check(), edit=_edit_a))
    assert rec["strict_verdict"] == "FAIL" and rec["amended_verdict"] == "FAIL"
    assert "results/a.json: fail" in rec["amendment"]["unmatched_differences"]


def test_a_difference_in_another_file_is_not_covered_by_the_pinned_one(repo, acfg, monkeypatch):
    _with_deviations(monkeypatch, [D1, D2])

    def edit(clone):
        _edit_a(clone)
        _write(clone, "results/rel.csv", "name,coef\nx,0.5\ny,9.0\n")

    rec = _run(repo, acfg, runner=_runner(check=_failed_check(), edit=edit))
    assert rec["amended_verdict"] == "FAIL"
    assert "results/rel.csv: fail" in rec["amendment"]["unmatched_differences"]


def test_a_skipped_step_or_a_missing_comparison_gives_no_amended_pass(repo, acfg, monkeypatch):
    _with_deviations(monkeypatch, [D1, D2])
    rec = _run(repo, acfg, runner=_runner(check=_failed_check(), edit=_edit_a, regenerate=False))
    assert rec["amended_verdict"] == "FAIL"  # the purged outputs were not regenerated: missing
    assert any(u.endswith(": missing") for u in rec["amendment"]["unmatched_differences"])
    # a comparison that lacks files is incomplete
    rules = reproduce.snapshot_rules(acfg, repo, freeze.head_commit(repo))
    comp = {"files": [{"path": "results/a.json", "status": "fail"}], "source_changed": [],
            "untracked_source": [], "untracked_undeclared": []}  # fmt: skip
    steps = [{"name": s["name"]} for s in rules["rp"]["steps"]]
    am, verdict = reproduce.assess_amendment(
        acfg["amendments"][0], acfg, rules, repo, steps, None, _failed_check(), comp, "FAIL"
    )
    assert (
        verdict == "FAIL" and "differ from the expected inventory" in am["unmatched_differences"][0]
    )
    # a skipped step (the report did not run) is incomplete as well
    am, verdict = reproduce.assess_amendment(
        acfg["amendments"][0], acfg, rules, repo, steps[:-1], None, _failed_check(), None, "FAIL"
    )
    assert verdict == "FAIL" and any("steps run" in u for u in am["unmatched_differences"])


def test_amended_verdict_equals_the_strict_one_when_nothing_needs_excusing(repo, acfg, monkeypatch):
    _with_deviations(monkeypatch, [D1, D2])
    rec = _run(repo, acfg, runner=_runner(edit=_volatile_edit))
    assert rec["strict_verdict"] == rec["amended_verdict"] == "EXACT_PASS"
    assert rec["amendment"]["applied_deviations"] == []
    rec = _run(repo, acfg, runner=_runner(check=NUMERIC), force_clean=True)
    assert rec["strict_verdict"] == rec["amended_verdict"] == "PORTABLE_NUMERIC_PASS"


def test_without_amendments_the_amended_verdict_is_the_strict_one(repo, cfg):
    cfg = copy.deepcopy(cfg)
    cfg["amendments"] = []
    rec = _run(repo, cfg, runner=_runner(edit=_volatile_edit))
    assert rec["amended_verdict"] == rec["strict_verdict"] == "EXACT_PASS" and not rec["amendment"]
    assert _run(repo, cfg, runner=_runner(check=STRUCTURAL), force_clean=True)["failure"]


def test_a_within_tolerance_numeric_difference_forbids_the_amended_pass(repo, acfg, monkeypatch):
    _with_deviations(monkeypatch, [D1, D2])

    def edit(clone):
        _edit_a(clone)  # covered by D2
        _write(clone, "results/rel.csv", "name,coef\nx,0.5\ny,1.2500000001\n")  # within rtol

    rec = _run(repo, acfg, runner=_runner(check=_failed_check(), edit=edit))
    inv = {f["path"]: f["status"] for f in rec["amendment"]["inventory"]["comparison_files"]}
    assert inv["results/rel.csv"] == "tolerance"
    assert rec["strict_verdict"] == rec["amended_verdict"] == "FAIL"
    assert "results/rel.csv: tolerance" in rec["amendment"]["unmatched_differences"]


def test_a_pixel_equal_png_reencoding_or_reformatting_forbids_the_amended_pass(
    repo, acfg, monkeypatch
):
    _with_deviations(monkeypatch, [D1, D2])
    for rel, data, status in (
        ("reports/figures/f.png", _png(level=0), "pixel_equal"),
        ("results/rel.csv", "name,coef\nx,0.50\ny,1.25\n", "equivalent"),
    ):

        def edit(clone, rel=rel, data=data):
            _edit_a(clone)
            _write(clone, rel, data)

        rec = _run(repo, acfg, runner=_runner(check=_failed_check(), edit=edit), force_clean=True)
        files = {f["path"]: f["status"] for f in rec["amendment"]["inventory"]["comparison_files"]}
        assert files[rel] == status
        assert rec["amended_verdict"] == "FAIL", rel


def test_an_out_of_scope_byte_change_forbids_the_amended_pass(repo, acfg, monkeypatch):
    _with_deviations(monkeypatch, [D1, D2])

    def edit(clone):
        _edit_a(clone)
        _write(clone, "src/mod.py", "X = 2\n")

    rec = _run(repo, acfg, runner=_runner(check=_failed_check(), edit=edit))
    assert rec["amended_verdict"] == "FAIL"
    assert any(u.startswith("source_changed") for u in rec["amendment"]["unmatched_differences"])


def test_a_passing_verdict_needs_a_complete_check_result_and_comparison(repo, cfg):
    # a check result without its check entries is unusable: fail closed even if "passed"
    rec = _run(repo, cfg, runner=_runner(check={"passed": True, "numeric_only": True}))
    assert rec["verdict"] == "FAIL"
    rules = reproduce.snapshot_rules(cfg, repo, freeze.head_commit(repo))
    steps = [{"name": s["name"]} for s in rules["rp"]["steps"]]
    assert reproduce.incomplete(rules, steps, PASS, {"files": []})  # comparisons not executed
    full = {"files": [{"path": p, "status": "exact"} for p in rules["compared"]]}
    assert reproduce.incomplete(rules, steps, PASS, full) == []
    full["files"][0]["status"] = "missing"
    assert any("missing" in r for r in reproduce.incomplete(rules, steps, PASS, full))
    assert reproduce.decide(PASS, {"exact_ok": True, "portable_ok": True}, ["x"]) == "FAIL"
    assert reproduce.incomplete(rules, steps[:-1], PASS, full)


def test_a_generic_nonzero_check_is_never_tolerated():
    cov = {"id": "D", "satisfied": True, "covers_checks": {}, "covers_files": []}
    assert not reproduce.tolerable_check({"checks": []}, [cov])  # no classified failure
    assert not reproduce.tolerable_check({"checks": [{"status": "fail"}]}, [cov])  # malformed
    assert not reproduce.tolerable_check(_failed_check(), [cov])  # failed but not covered
    assert not reproduce.tolerable_check(None, [cov])
    ok = {**cov, "covers_checks": {"files": ["sha256 mismatch: data/raw/manifest.json"]}}
    assert reproduce.tolerable_check(_failed_check(), [ok])


def test_the_record_pins_the_decision_procedure(repo, acfg, monkeypatch):
    _with_deviations(monkeypatch, [D1, D2])
    cmd = ["fremtpl2", "reproduce", "--config", "configs/default.yaml"]
    rec = _run(repo, acfg, runner=_runner(edit=_volatile_edit), command=cmd)
    dp = rec["decision_procedure"]
    assert dp["original_policy_sha256"] == "0" * 64 and dp["amended_policy_sha256"] == "f" * 64
    assert dp["rerun_command"] == cmd and dp["operational_retry"] == "none"
    assert dp["tested_commit"] == rec["tested_commit"] == freeze.head_commit(repo)
    src = Path(reproduce.__file__).parent
    assert dp["verifier_sources_sha256"] == {
        f"src/fremtpl2_pricing/{n}": hashlib.sha256((src / n).read_bytes()).hexdigest()
        for n in ("reproduce.py", "freeze.py")
    }


def test_operational_retry_none_numbers_every_run_and_refuses_gaps(repo, cfg):
    first = _run(repo, cfg, runner=_runner(check=STRUCTURAL))  # a failed run is its own record
    second = _run(repo, cfg, runner=_runner(), force_clean=True)
    assert (first["run"], second["run"]) == (1, 2)
    assert first["strict_verdict"] == "FAIL" and second["strict_verdict"] == "EXACT_PASS"
    first_text = (repo / "results/reproduction/run-1.json").read_text()
    (repo / "results/reproduction/run-1.json").unlink()  # a gap: run-1 deleted
    with pytest.raises(freeze.FreezeError, match="not contiguous"):
        _run(repo, cfg, runner=_runner(), force_clean=True)
    (repo / "results/reproduction/run-1.json").write_text(first_text)
    bad = copy.deepcopy(cfg)
    bad["reproduce"]["operational_retry"] = "retry"
    with pytest.raises(freeze.FreezeError, match="operational_retry"):
        _run(repo, bad, runner=_runner(), force_clean=True)


def test_a_crash_is_recorded_as_its_own_failed_run_and_reraised(repo, cfg):
    def boom(cmd, cwd):
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError):
        _run(repo, cfg, runner=boom)
    rec = json.loads((repo / "results/reproduction/run-1.json").read_text())
    assert rec["amended_verdict"] == rec["strict_verdict"] == "FAIL"
    assert rec["failure"]["step"] == "exception" and "kaboom" in rec["failure"]["reason"]


# --------------------------------------------------------------------------- the real amendment

REPO_ROOT = Path(__file__).resolve().parents[1]


def _real_amendment():
    amendments = load_reproduce_config(REPO_ROOT / "configs/reproduce.yaml")["amendments"]
    assert len(amendments) == 1
    return amendments[0]


def test_the_real_amendment_is_dated_post_hoc_and_pins_exactly_three_deviations():
    a = _real_amendment()
    assert (a["date"], a["post_hoc"], a["revealed_by"]) == ("2026-11-08", True, "run-1")
    assert a["original_policy_sha256"] == (
        "d35439ce4deb892114161823d9f87bdb8cf5bc04707fc1f6d0df38fd85641bb6"
    )
    d1, d2, d3 = a["declared_deviations"]
    assert (d1["id"], d1["artifact"], d1["new_value_format"]) == (
        "D1", "data/raw/manifest.json", "iso_date",
    )  # fmt: skip
    assert [(f["path"], f["committed"]) for f in d1["fields"]] == [
        ("/datasets/0/retrieved", "2026-11-07"),
        ("/datasets/1/retrieved", "2026-11-07"),
    ]
    assert d2["added_keys"] == [
        "models.gbm", "models.bootstrap.interval", "models.bootstrap.quantile_method",
        "models.evaluation.lorenz_points",
    ]  # fmt: skip
    assert (d2["old_config_sha256"], d2["new_config_sha256"]) == (
        "75d7109f4ca0325c26b1f54a8c62d5d46cde78c689825aadebfb634d9c5d05fe",
        "cd341a439ed871bdc03ed9c7cc1f34a469e1d0c46b648d027eddd2a1d6505793",
    )


def test_the_pinned_hashes_match_the_frozen_default_config():
    from fremtpl2_pricing.config import load_config
    from fremtpl2_pricing.train import config_summary

    d2 = _real_amendment()["declared_deviations"][1]
    cfg = load_config(REPO_ROOT / "configs/default.yaml")
    assert config_summary(cfg)["config_sha256"] == d2["new_config_sha256"]
    for dotted, pinned in d2["added_value_sha256"].items():
        node = cfg
        for part in dotted.split("."):
            node = node[part]
        assert freeze.canonical_sha256(node) == pinned


def test_the_original_policy_hash_matches_the_commit_it_names():
    ref = _real_amendment()["original_policy_ref"]
    try:
        blob = subprocess.run(
            ["git", "show", ref], cwd=REPO_ROOT, capture_output=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("the original policy commit is not available")
    import hashlib

    assert hashlib.sha256(blob).hexdigest() == _real_amendment()["original_policy_sha256"]


def test_static_evidence_of_the_added_keys_matches_the_source():
    ev = _real_amendment()["declared_deviations"][1]["static_evidence"]
    scanned = freeze.scan_readers(ev, REPO_ROOT)
    assert scanned == {k: sorted(v["modules"]) for k, v in ev["keys"].items()}
    assert scanned["models.evaluation.lorenz_points"] == ["evaluate"]
    glm_stage = set(ev["glm_training_modules"])
    assert {"train", "glm", "cv_report"} <= glm_stage
    assert not glm_stage & {m for mods in scanned.values() for m in mods}


def test_the_reproduce_config_carries_the_amendment_and_its_hash():
    from fremtpl2_pricing.config import ConfigError, with_reproduce

    rcfg = load_reproduce_config(REPO_ROOT / "configs/reproduce.yaml")
    merged = with_reproduce(_MAIN, rcfg)
    assert merged["amendments"] == rcfg["amendments"]
    assert merged["reproduce_policy_sha256"] == rcfg["policy_sha256"]
    assert len(rcfg["policy_sha256"]) == 64
    with pytest.raises(ConfigError, match="amendments"):
        with_reproduce({**_MAIN, "amendments": []}, rcfg)
    # the verification-policy keys are not part of the main config seen by the trainer
    assert freeze.main_config(merged) == _MAIN


def test_the_real_policy_declares_no_operational_retry():
    rp = load_reproduce_config(REPO_ROOT / "configs/reproduce.yaml")["reproduce"]
    assert rp["operational_retry"] == "none"


def test_the_real_d3_pins_the_timing_sections_of_the_committed_reports():
    d3 = _real_amendment()["declared_deviations"][2]
    assert d3["kind"] == "markdown_telemetry" and d3["volatile_keys"] == ["timing_seconds"]
    pinned = {f["path"] for f in d3["files"]}
    for spec in d3["files"]:
        text = (REPO_ROOT / spec["path"]).read_text(encoding="utf-8")
        # an unchanged file differs nowhere; a doubled value inside the section is a covered cell
        timings = json.loads((REPO_ROOT / spec["json"]).read_text())["timing_seconds"]
        assert freeze.md_telemetry_differences(text, text, spec, timings) == ([], [])
        lines = text.split("\n")
        head = lines.index(spec["heading"])
        pat = r"\| (\d+\.?\d*) \|$" if spec["cells"]["style"] == "table" else r"(\d+\.?\d*) s$"
        idx = next(i for i in range(head + 1, len(lines)) if re.search(pat, lines[i]))
        mutated = list(lines)
        mutated[idx] = re.sub(pat, lambda m: m[0].replace(m[1], "123.5"), lines[idx])
        key = re.match(r"\| ([^|]+?) \|" if pat.startswith(r"\|") else r"- (\w+):", lines[idx])[1]
        new_timings = {**timings, key: 123.5}  # the regenerated JSON value the cell renders
        args = (text, "\n".join(mutated), spec)
        problems, changed = freeze.md_telemetry_differences(*args, new_timings)
        assert problems == [] and len(changed) == 1, (spec["path"], problems)
        problems, _ = freeze.md_telemetry_differences(*args, timings)  # JSON does not say 123.5
        assert any("producer's rendering" in p for p in problems), spec["path"]
    # every other tracked markdown report renders no timing section
    for md in sorted((REPO_ROOT / "reports").glob("*.md")):
        rel = f"reports/{md.name}"
        if rel not in pinned:
            text = md.read_text(encoding="utf-8")
            assert "## Timing" not in text and "timings" not in text.lower(), rel


def _cli_record(strict, amended):
    return {"strict_verdict": strict, "amended_verdict": amended, "verdict": strict,
            "total_seconds": 1.0}  # fmt: skip


@pytest.mark.parametrize(
    ("strict", "amended", "code"),
    [
        ("EXACT_PASS", "EXACT_PASS", 0),
        ("FAIL", "PASS_WITH_DECLARED_DEVIATIONS", 0),
        ("PORTABLE_NUMERIC_PASS", "PORTABLE_NUMERIC_PASS", 1),
        ("FAIL", "FAIL", 1),
        ("UNSUPPORTED_ENVIRONMENT", "UNSUPPORTED_ENVIRONMENT", 1),
    ],
)
def test_reproduce_cli_prints_both_verdicts_and_exits_zero_only_for_exact_or_amended_pass(
    monkeypatch, capsys, strict, amended, code
):
    from fremtpl2_pricing import cli

    monkeypatch.setattr(cli, "run_reproduce", lambda *a, **k: _cli_record(strict, amended))
    args = ["reproduce", "--config", str(REPO_ROOT / "configs/default.yaml"),
            "--reproduce-config", str(REPO_ROOT / "configs/reproduce.yaml")]  # fmt: skip
    assert cli.main(args) == code
    out = capsys.readouterr().out
    assert f"strict {strict}" in out and f"amended {amended}" in out


def test_a_markdown_timing_deviation_covers_only_the_files_it_names(repo, acfg, monkeypatch):
    d3 = {"id": "D3", "kind": "markdown_telemetry", "satisfied": True, "problems": [],
          "applied": [{"place": "reports/r.md:3", "base": "1.2", "new": "9.0"}],
          "covers_checks": {}, "covers_files": ["reports/r.md"]}  # fmt: skip
    edit = lambda c: _write(c, "reports/r.md", "# Report\n\nvalue 9.0 end\n")  # noqa: E731
    _with_deviations(monkeypatch, [D1, d3])
    rec = _run(repo, acfg, runner=_runner(check=_failed_check(), edit=edit))
    assert rec["strict_verdict"] == "FAIL"
    assert rec["amended_verdict"] == "PASS_WITH_DECLARED_DEVIATIONS"
    _with_deviations(monkeypatch, [D1, {**d3, "covers_files": ["reports/other.md"]}])
    rec = _run(repo, acfg, runner=_runner(check=_failed_check(), edit=edit), force_clean=True)
    assert rec["amended_verdict"] == "FAIL"
    assert "reports/r.md: fail" in rec["amendment"]["unmatched_differences"]
    _with_deviations(
        monkeypatch, [D1, {**d3, "satisfied": False}]
    )  # an unsatisfied D3 covers nothing
    rec = _run(repo, acfg, runner=_runner(check=_failed_check(), edit=edit), force_clean=True)
    assert rec["amended_verdict"] == "FAIL"
