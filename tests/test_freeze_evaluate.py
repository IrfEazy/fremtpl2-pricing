"""Freeze stage and the one-shot test evaluation, rehearsed on a synthetic git project.

Every test runs in a temporary copy of a tiny project (synthetic data, config, frozen split, GLM
and GBM artifacts produced by the real `train` / `gbm` stages, its own git repository). The real
project data, results and test set are never touched.
"""

import copy
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fremtpl2_pricing import cli, evaluate, features, freeze, gbm, glm, train
from fremtpl2_pricing.audit import _factors
from fremtpl2_pricing.clean import clean
from fremtpl2_pricing.config import load_config
from fremtpl2_pricing.split import assign_split
from synth import _DEFAULTS, FACTORS, make_cfg, make_freq, make_sev, tiny_gbm_block

CONFIG = "configs/synthetic.yaml"
N_POLICIES = 4000
CAPTURED: dict = {}  # in-memory fitted objects of the template's final train / gbm refits


def _git(root: Path, *args: str) -> str:
    cmd = ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "-c",
           "commit.gpgsign=false", *args]  # fmt: skip
    return subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=True).stdout


def _commit(root: Path, msg: str) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", msg)


def _project_cfg() -> dict:
    cfg = make_cfg()
    cfg["models"]["gbm"] = tiny_gbm_block()
    cfg["models"]["bootstrap"]["replicates"] = 40
    cfg["paths"] = {
        "raw": "data/raw",
        "processed": "data/processed",
        "results": "results",
        "reports": "reports",
    }
    cfg["download"] = copy.deepcopy(_DEFAULTS["download"])
    cfg["freeze"] = copy.deepcopy(_DEFAULTS["freeze"])
    cfg["freeze"]["self_check"]["rows"] = 100
    return cfg


def _capture_final_fits(mp: pytest.MonkeyPatch) -> None:
    """Record the fitted model objects of the final refits (GLM: the glum estimators with their
    training design; GBM: the in-memory boosters and the fitted encoding)."""
    real_final, real_fit, real_gbm = train.final_fit, glm._fit, gbm.gbm_models
    fits: list = []

    def fit_spy(kind, X, *a, **k):
        model = real_fit(kind, X, *a, **k)
        fits.append((X.copy(), model))
        return model

    def final_spy(tr, *a, **k):
        mp.setattr(glm, "_fit", fit_spy)
        try:
            return real_final(tr, *a, **k)
        finally:
            mp.setattr(glm, "_fit", real_fit)
            CAPTURED["glm_fits"] = list(fits)
            CAPTURED["glm_train_ids"] = tr["IDpol"].to_numpy().copy()

    def gbm_spy(tr, *a, **k):
        out = real_gbm(tr, *a, **k)
        CAPTURED["boosters"], CAPTURED["encoding"] = out[0], out[3]
        CAPTURED["gbm_train_ids"] = tr["IDpol"].to_numpy().copy()
        return out

    mp.setattr(train, "final_fit", final_spy)
    mp.setattr(gbm, "gbm_models", gbm_spy)


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    """A committed synthetic project after `train` and `gbm` (before the freeze)."""
    root = tmp_path_factory.mktemp("project")
    mp = pytest.MonkeyPatch()
    mp.chdir(root)
    try:
        cfg = _project_cfg()
        (root / "configs").mkdir()
        (root / CONFIG).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        for d in ("data/raw", "data/processed", "results", "reports"):
            (root / d).mkdir(parents=True)
        freq = make_freq(N_POLICIES)
        sev = make_sev(freq)
        freq.to_parquet(root / "data/raw/freMTPL2freq.parquet", index=False)
        sev.to_parquet(root / "data/raw/freMTPL2sev.parquet", index=False)
        (root / "data/raw/manifest.json").write_text(json.dumps({"datasets": []}))
        res = clean(freq, sev, cfg["clean"])
        assign_split(res.freq, cfg).to_parquet(root / "data/processed/split.parquet", index=False)
        (root / "results/audit.json").write_text(json.dumps({"factors": _factors(freq, FACTORS)}))
        (root / "reports/evaluation_protocol.md").write_text("# Protocol\n\nSynthetic.\n")
        (root / "uv.lock").write_text("synthetic lock\n")
        (root / ".gitignore").write_text("data/raw/\ndata/processed/\n__pycache__/\n")
        cfg = load_config(CONFIG)
        _capture_final_fits(mp)
        train.run_train(cfg)
        gbm.run_gbm(cfg)
        _git(root, "init", "-q")
        _commit(root, "synthetic project")
    finally:
        mp.undo()
    return root


@pytest.fixture
def project(template, tmp_path, monkeypatch):
    root = tmp_path / "p"
    shutil.copytree(template, root, symlinks=True)
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def cfg(project):
    return load_config(CONFIG)


@pytest.fixture
def frozen(project, cfg):
    """Freeze and commit the manifest."""
    manifest = freeze.run_freeze(cfg, CONFIG)
    _commit(project, "freeze")
    return manifest


@pytest.fixture
def no_test_read(monkeypatch):
    """Record (and forbid) any call of the test loader."""
    calls = []

    def boom(cfg):
        calls.append(1)
        raise AssertionError("the test rows were read")

    monkeypatch.setattr(evaluate, "load_test", boom)
    return calls


def _events(root: Path) -> list[dict]:
    return json.loads((root / "results/test_access.json").read_text())["events"]


# --------------------------------------------------------------------------- freeze


def test_freeze_manifest_contents(project, cfg, frozen):
    m = json.loads((project / "results/freeze_manifest.json").read_text())
    sev = m["selection"]["selected_glm_severity"]
    assert m["candidates"] == [
        "pp_null", "bonus_malus_only", f"glm_freq_x_sev[{sev}]", "glm_tweedie",
        "gbm_freq_x_sev", "gbm_tweedie",
    ]  # fmt: skip
    assert m["selected_glm_id"] in m["candidates"] and m["selected_gbm_id"] in m["candidates"]
    assert m["training_commit"] == _git(project, "rev-parse", "HEAD~1").strip()
    paths = {f["path"] for f in m["files"]}
    for p in (
        "results/cv_glm.json", "results/glm_models.json", "results/feature_map.json",
        "results/cv_gbm.json", "results/gbm_models.json", CONFIG,
        "reports/evaluation_protocol.md", "uv.lock", "data/processed/split.parquet",
        "data/processed/models/frequency.txt", "data/processed/models/severity.txt",
        "data/processed/models/tweedie.txt",
    ):  # fmt: skip
        assert p in paths, p
    r = m["recipes"]
    assert r["gbm_freq_x_sev"]["frequency"]["kind"] == "gbm"
    assert r[f"glm_freq_x_sev[{sev}]"]["frequency"]["feature_map_id"] == "feature_map.json#policy"
    assert r["glm_tweedie"]["tweedie_power"] == m["training_record"]["tweedie_power"]
    b = m["evaluation"]["bootstrap"]
    assert b["unit"] == "group" and b["sensitivity_unit"] == "policy"
    assert b["quantile_method"] == "linear" and b["percentiles"] == pytest.approx([0.025, 0.975])
    assert b["replicates"] == 40 and b["seed"] == cfg["models"]["bootstrap"]["seed"]
    assert m["evaluation"]["primary_loss"]["power"] == cfg["models"]["tweedie"]["reference_power"]
    assert m["evaluation"]["technical_rerun_policy"]
    sc = m["self_check"]
    assert len(sc["IDpol"]) >= 100 and sc["IDpol"] == sorted(sc["IDpol"])
    assert set(sc["predictions"]) == set(m["candidates"])
    assert all(v["covered"] for part in sc["edge_cases"].values() for v in part.values())
    assert sc["determinism_at_freeze"]["permuted_row_order"] == "identical"
    rt = m["runtime_identity"]
    assert set(rt["packages"]) == {
        "numpy", "pandas", "scipy", "scikit-learn", "glum", "lightgbm", "pyarrow",
    }  # fmt: skip
    assert rt["python"] and rt["num_threads"] == 1 and len(rt["lock_sha256"]) == 64
    assert m["evaluation"]["recovery_policy"] and m["evaluation"]["nonfinite_policy"]
    assert set(m["selection"]["trace_sha256"]) == {"glm_ladder", "glm_freq_x_sev_vs_tweedie", "gbm"}


def test_freeze_refuses_dirty_tree(project, cfg):
    (project / "stray.txt").write_text("x")
    with pytest.raises(freeze.FreezeError, match="not clean"):
        freeze.run_freeze(cfg, CONFIG)


def test_freeze_refuses_existing_manifest_unless_force(project, cfg, frozen):
    with pytest.raises(freeze.FreezeError, match="--force"):
        freeze.run_freeze(cfg, CONFIG)
    freeze.run_freeze(cfg, CONFIG, force=True)
    (project / "results/test_access.json").write_text('{"events": []}')
    with pytest.raises(freeze.FreezeError, match="test-state"):
        freeze.run_freeze(cfg, CONFIG, force=True)


def test_freeze_fails_when_a_recorded_trace_differs(project, cfg):
    path = project / "results/cv_gbm.json"
    d = json.loads(path.read_text())
    step = d["selection"]["gbm_pure_premium"]["steps"][0]
    step["mean_d"] += 1.0
    path.write_text(json.dumps(d))
    _commit(project, "tamper")
    with pytest.raises(freeze.FreezeError, match="GBM freq x sev vs Tweedie"):
        freeze.run_freeze(cfg, CONFIG)


def test_freeze_fails_when_training_config_changed(project, cfg):
    cfg = copy.deepcopy(cfg)
    cfg["models"]["gbm"]["seed"] += 1
    with pytest.raises(freeze.FreezeError, match="models.gbm"):
        freeze.build_manifest(cfg, CONFIG, project)


# --------------------------------------------------------------------------- evaluate guards


def test_manifest_hash_tamper_aborts_before_test_read(project, cfg, frozen, no_test_read):
    path = project / "results/freeze_manifest.json"
    m = json.loads(path.read_text())
    m["files"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(m, indent=2))
    _commit(project, "tamper manifest")
    with pytest.raises(evaluate.EvaluateError, match="do not match"):
        evaluate.run_evaluate(cfg, CONFIG)
    assert not no_test_read
    assert not (project / "results/test_access.json").exists()


def test_model_file_tamper_aborts_before_test_read(project, cfg, frozen, no_test_read):
    with (project / "data/processed/models/severity.txt").open("a") as fh:
        fh.write("\n")  # git-ignored: the tree stays clean, the hash catches it
    with pytest.raises(evaluate.EvaluateError, match="severity.txt"):
        evaluate.run_evaluate(cfg, CONFIG)
    assert not no_test_read


def test_dirty_tree_aborts(project, cfg, frozen, no_test_read):
    (project / "notes.txt").write_text("untracked")
    with pytest.raises(freeze.FreezeError, match="not clean"):
        evaluate.run_evaluate(cfg, CONFIG)
    assert not no_test_read


def test_uncommitted_manifest_aborts(project, cfg, no_test_read):
    freeze.run_freeze(cfg, CONFIG)
    with pytest.raises(freeze.FreezeError, match="not clean"):
        evaluate.run_evaluate(cfg, CONFIG)
    assert not no_test_read


def test_code_change_after_freeze_aborts(project, cfg, frozen, no_test_read):
    (project / "reports/extra.md").write_text("late change\n")
    _commit(project, "late change")
    with pytest.raises(evaluate.EvaluateError, match="changed since the training commit"):
        evaluate.run_evaluate(cfg, CONFIG)
    assert not no_test_read


def test_concurrent_run_is_locked(project, cfg, frozen, no_test_read):
    (project / "data/processed/evaluate.lock").write_text("held")
    with pytest.raises(evaluate.EvaluateError, match="lock"):
        evaluate.run_evaluate(cfg, CONFIG)
    assert not no_test_read


def test_self_check_mismatch_aborts_before_test_read(project, cfg, frozen, no_test_read):
    path = project / "results/freeze_manifest.json"
    m = json.loads(path.read_text())
    m["self_check"]["predictions"]["glm_tweedie"]["pure_premium"][0] *= 1.01
    path.write_text(json.dumps(m, indent=2))
    _commit(project, "tamper self-check")
    with pytest.raises(freeze.FreezeError, match="self-check"):
        evaluate.run_evaluate(cfg, CONFIG)
    assert not no_test_read


# --------------------------------------------------------------------------- evaluate runs


def _forbid_fitting(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a fitting entry point was called during evaluate")

    import glum
    import lightgbm

    monkeypatch.setattr(glum.GeneralizedLinearRegressor, "fit", boom)
    monkeypatch.setattr(lightgbm, "train", boom)
    monkeypatch.setattr(lightgbm.Booster, "update", boom)
    for mod, name in (
        (glm, "_fit"), (glm, "fit_frequency"), (glm, "fit_severity"), (glm, "fit_tweedie"),
        (features, "fit"), (features, "fit_severity"), (gbm, "fit_encoding"),
        (gbm, "_train_early"), (gbm, "_final_task"),
    ):  # fmt: skip
        monkeypatch.setattr(mod, name, boom)


def test_end_to_end_rehearsal_is_fit_free(project, cfg, frozen, monkeypatch):
    _forbid_fitting(monkeypatch)
    out = evaluate.run_evaluate(cfg, CONFIG)
    adir = project / "results/test/attempt-1"
    for name in ("metrics.json", "tables.json", "bootstrap_replicates_group.csv",
                 "bootstrap_replicates_policy.csv"):  # fmt: skip
        assert (adir / name).is_file()
    assert [e["event"] for e in _events(project)] == ["FROZEN", "TEST_OPENED", "COMPLETE"]
    assert set(out["candidates"]) == set(frozen["candidates"])
    delta = out["bootstrap"]["group"]["delta"]
    assert delta["lower"] <= delta["point"] <= delta["upper"]
    assert delta["decision"] == evaluate.decision(delta["lower"], delta["upper"])
    assert out["bootstrap"]["policy"]["unit_column"] == "IDpol"
    assert out["groups"]["groups"] == out["bootstrap"]["group"]["n_units"]
    assert out["self_check"]["passed"]
    pointer = json.loads((project / "results/metrics.json").read_text())
    assert pointer["source"]["attempt_dir"] == "results/test/attempt-1"
    reps = pd.read_csv(adir / "bootstrap_replicates_group.csv")
    assert len(reps) == 40 and "delta" in reps
    tables = json.loads((adir / "tables.json").read_text())
    assert tables["double_lift"]["a"] == frozen["selected_gbm_id"]
    assert len(tables["lorenz"]["glm_tweedie"]["cum_exposure"]) == 101
    # the test rows scored are exactly the frozen test fold
    split = pd.read_parquet(project / "data/processed/split.parquet")
    assert out["rows"]["policies"] == int((split["primary_set"] == "test").sum())
    # second run: refused, as is a technical rerun after a completed attempt
    _commit(project, "attempt 1")
    with pytest.raises(evaluate.EvaluateError, match="already accessed"):
        evaluate.run_evaluate(cfg, CONFIG)
    with pytest.raises(evaluate.EvaluateError, match="already completed"):
        evaluate.run_evaluate(cfg, CONFIG, technical_rerun=True, reason="again")


class Crash(BaseException):
    """Simulates a hard crash (not caught as a failure)."""


def test_crash_keeps_state_and_rerun_creates_new_attempt(project, cfg, frozen, monkeypatch):
    real_load = evaluate.load_test

    def crash(c):
        raise Crash()

    monkeypatch.setattr(evaluate, "load_test", crash)
    with pytest.raises(Crash):
        evaluate.run_evaluate(cfg, CONFIG)
    ev = _events(project)
    assert [e["event"] for e in ev] == ["FROZEN", "TEST_OPENED"]  # no reset, no terminal event
    first_id = ev[1]["attempt_id"]
    assert ev[1]["manifest_sha256"] and ev[1]["runtime_sha256"]
    assert not (project / "results/test").exists()  # nothing is published without COMPLETE
    staged1 = project / "data/processed/evaluate_staging/attempt-1"
    (staged1 / "marker.txt").write_text("attempt 1 output\n")
    _commit(project, "attempt 1 crashed")
    monkeypatch.setattr(evaluate, "load_test", real_load)
    with pytest.raises(evaluate.EvaluateError, match="already accessed"):
        evaluate.run_evaluate(cfg, CONFIG)
    with pytest.raises(evaluate.EvaluateError, match="reason"):
        evaluate.run_evaluate(cfg, CONFIG, technical_rerun=True, reason=" ")
    evaluate.run_evaluate(cfg, CONFIG, technical_rerun=True, reason="I/O error in attempt 1")
    ev = _events(project)
    assert [e["event"] for e in ev] == ["FROZEN", "TEST_OPENED", "TEST_OPENED", "COMPLETE"]
    assert ev[2]["attempt"] == 2 and ev[2]["rerun_of"] == first_id
    assert ev[2]["original_attempt_id"] == first_id and ev[2]["technical_rerun"]
    assert ev[2]["manifest_sha256"] == ev[1]["manifest_sha256"]
    assert ev[2]["runtime_sha256"] == ev[1]["runtime_sha256"]
    assert sorted(p.name for p in staged1.iterdir()) == ["marker.txt"]
    assert (staged1 / "marker.txt").read_text() == "attempt 1 output\n"
    assert not (project / "results/test/attempt-1").exists()
    assert (project / "results/test/attempt-2/metrics.json").is_file()
    pointer = json.loads((project / "results/metrics.json").read_text())
    assert pointer["source"]["attempt_dir"] == "results/test/attempt-2"
    log = pointer["access_log"]  # every attempt is reported, the interrupted one included
    assert [a["attempt"] for a in log] == [1, 2]
    assert log[0]["status"].startswith("interrupted") and log[1]["status"] == "this attempt"
    assert log[1]["rerun_of"] == first_id


def test_failure_after_opening_is_logged(project, cfg, frozen, monkeypatch):
    def fail(c):
        raise OSError("disk error")

    monkeypatch.setattr(evaluate, "load_test", fail)
    with pytest.raises(OSError):
        evaluate.run_evaluate(cfg, CONFIG)
    ev = _events(project)
    assert [e["event"] for e in ev] == ["FROZEN", "TEST_OPENED", "FAILED"]
    assert ev[-1]["error_type"] == "OSError" and "error" not in ev[-1]  # type only, no message
    err = project / "data/processed/evaluate_staging/attempt-1/error.txt"
    assert "disk error" in err.read_text()  # private staging, git-ignored
    assert not (project / "results/test").exists()
    assert not (project / "results/metrics.json").exists()
    assert not (project / "data/processed/evaluate.lock").exists()


def test_failed_append_does_not_mask_the_original_error(project, cfg, frozen, monkeypatch, capsys):
    def fail(c):
        raise OSError("disk error")

    real_append = evaluate.append_event

    def append(sink, event):
        if event["event"] == "FAILED":
            raise RuntimeError("journal full")
        return real_append(sink, event)

    monkeypatch.setattr(evaluate, "load_test", fail)
    monkeypatch.setattr(evaluate, "append_event", append)
    with pytest.raises(OSError, match="disk error") as info:
        evaluate.run_evaluate(cfg, CONFIG)
    assert any("journal full" in n for n in info.value.__notes__)
    assert "could not append the FAILED event" in capsys.readouterr().err
    assert [e["event"] for e in _events(project)] == ["FROZEN", "TEST_OPENED"]


def test_reset_of_the_tracked_log_is_refused(project, cfg, frozen, monkeypatch):
    def fail(c):
        raise OSError("disk error")

    monkeypatch.setattr(evaluate, "load_test", fail)
    with pytest.raises(OSError):
        evaluate.run_evaluate(cfg, CONFIG)
    (project / "results/test_access.json").unlink()  # tree clean again, mirror remains
    with pytest.raises(evaluate.EvaluateError, match="reset"):
        evaluate.run_evaluate(cfg, CONFIG)


# --------------------------------------------------------------------------- pure parts


def _frame_and_preds(seed=3):
    r = np.random.default_rng(seed)
    n = 600
    sizes = r.integers(1, 6, size=200)
    gid = np.repeat(np.arange(200), sizes)[:n]
    n = len(gid)
    e = r.uniform(0.2, 1.0, n)
    nclaims = (r.random(n) < 0.2).astype(int)
    s = nclaims * r.gamma(2.0, 400.0, n)
    df = pd.DataFrame(
        {
            "IDpol": np.arange(n) + 10,
            "group_id": gid,
            "Exposure": e,
            "S_cap": s,
            "S": s * 1.1,
            "N": nclaims,
            "freq_count": nclaims,
            "BonusMalus": r.choice([50, 80, 120], n),
            "DrivAge": r.integers(18, 90, n),
            "VehAge": r.integers(0, 20, n),
            "Region": r.choice(["R1", "R2"], n),
        }
    )
    a = r.uniform(50, 300, n)
    preds = {
        "glm": {"pure_premium": a, "frequency": a / 1000, "severity": np.full(n, 1000.0)},
        "gbm": {"pure_premium": a * r.uniform(0.8, 1.2, n)},
    }
    return df, preds


def _spec(gbm_id="gbm", glm_id="glm"):
    b = dict(_DEFAULTS["models"]["bootstrap"], replicates=50)
    return {
        "primary_loss": {"power": 1.5},
        "primary_contrast": {"gbm": gbm_id, "glm": glm_id},
        "bootstrap": b,
        "excess_loading_per_exposure_year": 10.0,
        "candidate_powers": {"glm": 1.5, "gbm": 1.5},
    }


def test_evaluate_frame_swap_and_identical_models():
    df, preds = _frame_and_preds()
    cfg = {**_DEFAULTS, "features": _DEFAULTS["features"]}
    ab = evaluate.evaluate_frame(df, preds, _spec(), cfg)
    ba = evaluate.evaluate_frame(df, preds, _spec("glm", "gbm"), cfg)
    np.testing.assert_array_equal(
        ab["replicates"]["group"]["delta"], -ba["replicates"]["group"]["delta"]
    )
    d1, d2 = ab["bootstrap"]["group"]["delta"], ba["bootstrap"]["group"]["delta"]
    assert d2["lower"] == pytest.approx(-d1["upper"]) and d2["upper"] == pytest.approx(-d1["lower"])
    g1, g2 = (
        ab["bootstrap"]["group"]["gini_difference"],
        ba["bootstrap"]["group"]["gini_difference"],
    )
    assert g2["point"] == pytest.approx(-g1["point"])
    same = {"glm": preds["glm"], "gbm": {"pure_premium": preds["glm"]["pure_premium"].copy()}}
    z = evaluate.evaluate_frame(df, same, _spec(), cfg)
    assert (z["replicates"]["group"]["delta"] == 0).all()
    assert (z["replicates"]["group"]["gini_difference"] == 0).all()
    assert z["bootstrap"]["group"]["delta"]["decision"] == "no demonstrated difference"
    # group sums = direct weighted computation for the point estimate
    e = df["Exposure"].to_numpy()
    y = df["S_cap"].to_numpy() / e
    direct = [
        evaluate.metrics.tweedie_deviance(y, preds[k]["pure_premium"], 1.5, e) / e.sum()
        for k in ("gbm", "glm")
    ]
    assert ab["bootstrap"]["group"]["delta"]["point"] == pytest.approx(direct[0] - direct[1])
    assert ab["scalars"]["candidates"]["glm"]["primary_loss"] == pytest.approx(direct[1])
    assert "frequency" in ab["scalars"]["candidates"]["glm"]
    assert "severity" not in ab["scalars"]["candidates"]["gbm"]


def test_decision_rule():
    assert evaluate.decision(-2.0, -0.1) == "improvement"
    assert evaluate.decision(0.1, 2.0) == "deterioration"
    assert evaluate.decision(-0.1, 0.1) == "no demonstrated difference"
    assert evaluate.decision(-0.1, 0.0) == "no demonstrated difference"


def test_cli_freeze_and_evaluate_are_wired(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "run_freeze", lambda c, p, force: calls.append(("f", p, force)))
    monkeypatch.setattr(cli, "run_evaluate", lambda c, p, r, why: calls.append(("e", r, why)))
    monkeypatch.setattr(cli, "run_rehearsal", lambda c, p: calls.append(("r", p)))
    monkeypatch.setattr(cli, "run_publish", lambda c, p: calls.append(("p", p)))
    assert cli.main(["freeze", "--force"]) == 0
    assert cli.main(["evaluate"]) == 0
    assert cli.main(["evaluate", "--technical-rerun", "--reason", "io"]) == 0
    assert cli.main(["evaluate", "--reason", "io"]) == 1
    assert cli.main(["evaluate", "--rehearsal"]) == 0
    assert cli.main(["evaluate", "--publish-only"]) == 0
    assert cli.main(["evaluate", "--rehearsal", "--publish-only"]) == 1
    assert cli.main(["evaluate", "--rehearsal", "--technical-rerun", "--reason", "x"]) == 1
    assert calls == [
        ("f", "configs/default.yaml", True),
        ("e", False, None),
        ("e", True, "io"),
        ("r", "configs/default.yaml"),
        ("p", "configs/default.yaml"),
    ]


# --------------------------------------------------------------------------- rehearsal


def _rehearsal_dirs(root: Path) -> list[Path]:
    return sorted((root / "data/processed/rehearsal").glob("run-*"))


def test_rehearsal_runs_the_evaluate_path_on_a_train_fold(
    project, cfg, frozen, monkeypatch, no_test_read
):
    _forbid_fitting(monkeypatch)
    seen = []
    real_select = evaluate.select_population

    def spy(d, population):
        rows = real_select(d, population)
        seen.append(rows)
        return rows

    monkeypatch.setattr(evaluate, "select_population", spy)
    (project / "notes.txt").write_text("a dirty tree is allowed in a rehearsal\n")
    out = evaluate.run_rehearsal(cfg, CONFIG)
    assert not no_test_read  # the test loader is never called
    fold = cfg["freeze"]["rehearsal"]["fold"]
    assert len(seen) == 1 and (seen[0]["cv_fold"] == fold).all()
    assert (seen[0]["primary_set"] == "train").all()
    split = pd.read_parquet(project / "data/processed/split.parquet")
    assert out["rows"]["policies"] == int(
        ((split["primary_set"] == "train") & (split["cv_fold"] == fold)).sum()
    )
    assert out["population"]["kind"] == "rehearsal" and out["population"]["cv_fold"] == fold
    assert any("not clean" in p for p in out["rehearsal"]["git_problems_for_the_real_run"])
    (run,) = _rehearsal_dirs(project)
    ev = json.loads((run / "journal.json").read_text())["events"]
    assert [e["event"] for e in ev] == ["FROZEN", "TEST_OPENED", "COMPLETE"]
    assert ev[1]["population"]["kind"] == "rehearsal"
    for name in evaluate.OUTPUT_FILES:
        staged = run / "staging/attempt-1" / name
        published = run / "results/test/attempt-1" / name
        assert freeze.sha256_file(staged) == freeze.sha256_file(published) == ev[2]["staged"][name]
    assert (
        json.loads((run / "results/metrics.json").read_text())["source"]["attempt_id"]
        == (ev[1]["attempt_id"])
    )
    # nothing of the real evaluation was written
    untouched = (
        "results/test_access.json",
        "results/test",
        "results/metrics.json",
        "data/processed/test_access_mirror.jsonl",
        "data/processed/evaluate_staging",
    )
    for rel in untouched:
        assert not (project / rel).exists(), rel
    assert _git(project, "status", "--porcelain").strip() == "?? notes.txt"
    # the real evaluation afterwards runs the same code and produces the same structure
    (project / "notes.txt").unlink()
    monkeypatch.setattr(
        evaluate, "load_test", lambda c: evaluate.load_population(c, evaluate.TEST_POPULATION)
    )
    real = evaluate.run_evaluate(cfg, CONFIG)
    assert set(real) == set(out) - {"rehearsal"}
    assert set(real["candidates"]) == set(out["candidates"])
    for c in real["candidates"]:
        assert set(real["candidates"][c]) == set(out["candidates"][c])
    assert set(real["bootstrap"]["group"]) == set(out["bootstrap"]["group"])


def test_rehearsal_and_evaluate_share_one_code_path(monkeypatch):
    calls = []
    monkeypatch.setattr(evaluate, "run_attempt", lambda *a, **k: calls.append(a[2]["kind"]))
    cfg = copy.deepcopy(_DEFAULTS)
    evaluate.run_rehearsal(cfg, "c.yaml")
    evaluate.run_evaluate(cfg, "c.yaml")
    assert calls == ["rehearsal", "test"]


def test_rehearsal_fails_closed_on_test_rows():
    cfg = copy.deepcopy(_DEFAULTS)
    for bad in (-1, cfg["split"]["cv_folds"], "0", True):
        cfg["freeze"]["rehearsal"]["fold"] = bad
        with pytest.raises(evaluate.EvaluateError, match="registered train fold"):
            evaluate.rehearsal_population(cfg)
    d = pd.DataFrame(
        {"IDpol": [1, 2, 3], "primary_set": ["train", "test", "train"], "cv_fold": [0, -1, 0]}
    )
    with pytest.raises(evaluate.EvaluateError, match="fail closed"):
        evaluate.assert_rehearsal_rows(d)
    # a fold selection can never pick test rows: cv_fold < 0 is never a rehearsal fold
    with pytest.raises(evaluate.EvaluateError, match="fail closed"):
        evaluate.select_population(
            d.assign(primary_set="train"), {"kind": "rehearsal", "cv_fold": -1}
        )
    ok = evaluate.select_population(d, {"kind": "rehearsal", "cv_fold": 0})
    assert ok["IDpol"].tolist() == [1, 3]


# --------------------------------------------------------------------------- staging, publication


def test_publication_failure_is_recovered_by_publish_only(project, cfg, frozen, monkeypatch):
    real_publish = evaluate.publish

    def broken(*a, **k):
        raise OSError("disk full while publishing")

    monkeypatch.setattr(evaluate, "publish", broken)
    with pytest.raises(OSError, match="disk full"):
        evaluate.run_evaluate(cfg, CONFIG)
    ev = _events(project)
    assert [e["event"] for e in ev] == ["FROZEN", "TEST_OPENED", "COMPLETE"]
    assert (
        not (project / "results/test").exists() and not (project / "results/metrics.json").exists()
    )
    monkeypatch.setattr(evaluate, "publish", real_publish)
    _commit(project, "journal after COMPLETE")
    with pytest.raises(evaluate.EvaluateError, match="already completed"):
        evaluate.run_evaluate(cfg, CONFIG, technical_rerun=True, reason="publication failed")
    staged = project / "data/processed/evaluate_staging/attempt-1"
    before = {p.name: p.stat().st_mtime_ns for p in staged.iterdir()}
    doc = evaluate.run_publish(cfg, CONFIG)
    assert {p.name: p.stat().st_mtime_ns for p in staged.iterdir()} == before  # no recompute
    for name, sha in ev[2]["staged"].items():
        assert freeze.sha256_file(project / "results/test/attempt-1" / name) == sha
    assert doc["source"]["attempt_id"] == ev[1]["attempt_id"]
    # idempotent: a second republication finds identical files
    assert evaluate.run_publish(cfg, CONFIG) == doc
    # a tampered published file is refused
    with (project / "results/test/attempt-1/tables.json").open("a") as fh:
        fh.write(" ")
    with pytest.raises(evaluate.EvaluateError, match="differs from the staged"):
        evaluate.run_publish(cfg, CONFIG)


def test_publish_only_fails_closed_on_staged_hash_mismatch(project, cfg, frozen, monkeypatch):
    real_publish = evaluate.publish

    def broken(*a, **k):
        raise OSError("publication failed")

    monkeypatch.setattr(evaluate, "publish", broken)
    with pytest.raises(OSError):
        evaluate.run_evaluate(cfg, CONFIG)
    monkeypatch.setattr(evaluate, "publish", real_publish)
    staged = project / "data/processed/evaluate_staging/attempt-1/metrics.json"
    staged.write_text(staged.read_text().replace('"attempt": 1', '"attempt": 1 '))
    with pytest.raises(evaluate.EvaluateError, match="sha256 mismatch"):
        evaluate.run_publish(cfg, CONFIG)
    assert (
        not (project / "results/test").exists() and not (project / "results/metrics.json").exists()
    )


def test_completion_is_never_inferred_from_directories(project, cfg, frozen):
    adir = project / "results/test/attempt-1"
    adir.mkdir(parents=True)
    (adir / "metrics.json").write_text("{}")
    (project / "results/metrics.json").write_text("{}")
    with pytest.raises(evaluate.EvaluateError, match="no COMPLETE"):
        evaluate.run_publish(cfg, CONFIG)


def test_nonfinite_metric_fails_closed_without_publication(project, cfg, frozen, monkeypatch):
    real_gini = evaluate.metrics.gini

    def nan_gini(*a, **k):
        out = real_gini(*a, **k)
        return {**out, "gini_normalized": float("nan")}

    monkeypatch.setattr(evaluate.metrics, "gini", nan_gini)
    with pytest.raises(evaluate.EvaluateError, match="non-finite"):
        evaluate.run_evaluate(cfg, CONFIG)
    ev = _events(project)
    assert [e["event"] for e in ev] == ["FROZEN", "TEST_OPENED", "FAILED"]
    assert ev[-1]["error_type"] == "EvaluateError"
    assert not (project / "results/test").exists()


def test_output_documents_reject_nonfinite_values():
    with pytest.raises(evaluate.EvaluateError, match="non-finite value at"):
        evaluate._jsonable({"a": [1.0, float("inf")]})
    assert evaluate._jsonable({"a": np.float64(1.5), "b": np.int64(2), "c": None}) == {
        "a": 1.5, "b": 2, "c": None,
    }  # fmt: skip


# --------------------------------------------------------------------------- journal


def _sink(tmp_path: Path) -> dict:
    return {"state": tmp_path / "results/j.json", "mirror": tmp_path / "processed/m.jsonl"}


def _ev(kind: str, **kw) -> dict:
    base = {"event": kind, "timestamp": "t"}
    if kind == "FROZEN":
        return {**base, "manifest_sha256": "m", "commit": "c", **kw}
    if kind == "TEST_OPENED":
        return {
            **base, "attempt": 1, "attempt_id": "a1", "manifest_sha256": "m",
            "runtime_sha256": "r", "commit": "c", "population": {"kind": "test"},
            "technical_rerun": False, "reason": None, "rerun_of": None,
            "original_attempt_id": None, **kw,
        }  # fmt: skip
    if kind == "FAILED":
        return {**base, "attempt": 1, "attempt_id": "a1", "error_type": "OSError", **kw}
    raise ValueError(kind)


def test_journal_is_authoritative_and_the_mirror_fails_closed(tmp_path):
    sink = _sink(tmp_path)
    evaluate.append_event(sink, _ev("FROZEN"))
    evaluate.append_event(sink, _ev("TEST_OPENED"))
    assert [e["event"] for e in evaluate.read_events(sink)] == ["FROZEN", "TEST_OPENED"]
    mirror_text = sink["mirror"].read_text()
    journal_text = sink["state"].read_text()
    # mirror ahead of the journal (or edited)
    sink["mirror"].write_text(mirror_text + json.dumps(_ev("FAILED")) + "\n")
    with pytest.raises(evaluate.EvaluateError, match="disagree"):
        evaluate.read_events(sink)
    # torn mirror record
    sink["mirror"].write_text(mirror_text + '{"event": "FAI')
    with pytest.raises(evaluate.EvaluateError, match="torn"):
        evaluate.read_events(sink)
    sink["mirror"].write_text(mirror_text)
    # torn journal
    sink["state"].write_text(journal_text[: len(journal_text) // 2])
    with pytest.raises(evaluate.EvaluateError, match="torn"):
        evaluate.read_events(sink)
    # missing journal while the mirror says opened
    sink["state"].unlink()
    with pytest.raises(evaluate.EvaluateError, match="reset"):
        evaluate.read_events(sink)
    # missing mirror while the journal has events
    sink["state"].write_text(journal_text)
    sink["mirror"].unlink()
    with pytest.raises(evaluate.EvaluateError, match="mirror"):
        evaluate.read_events(sink)
    # appends refuse on an inconsistent state as well
    with pytest.raises(evaluate.EvaluateError):
        evaluate.append_event(sink, _ev("FAILED"))


def test_journal_validation_rejects_incomplete_or_unlinked_records():
    with pytest.raises(evaluate.EvaluateError, match="incomplete"):
        evaluate.validate_events([_ev("FROZEN"), {"event": "TEST_OPENED", "attempt": 1}])
    rerun = _ev("TEST_OPENED", attempt=2, attempt_id="a2", technical_rerun=True,
                reason="io", rerun_of="a1", original_attempt_id="a1")  # fmt: skip
    evaluate.validate_events([_ev("FROZEN"), _ev("TEST_OPENED"), _ev("FAILED"), rerun])
    for bad in ({"runtime_sha256": "other"}, {"manifest_sha256": "other"}, {"rerun_of": "x"}):
        with pytest.raises(evaluate.EvaluateError, match="replay"):
            evaluate.validate_events([_ev("FROZEN"), _ev("TEST_OPENED"), {**rerun, **bad}])


def test_rerun_requires_the_same_manifest_and_runtime():
    events = [_ev("FROZEN"), _ev("TEST_OPENED"), _ev("FAILED")]
    assert evaluate.attempt_plan(events, True, "io", "m", "r")[0] == 2
    with pytest.raises(evaluate.EvaluateError, match="manifest digest"):
        evaluate.attempt_plan(events, True, "io", "m2", "r")
    with pytest.raises(evaluate.EvaluateError, match="runtime identity"):
        evaluate.attempt_plan(events, True, "io", "m", "r2")


# --------------------------------------------------------------------------- preflight


def test_runtime_identity_mismatch_aborts_before_test_read(project, cfg, frozen, no_test_read,
                                                           monkeypatch):  # fmt: skip
    real = evaluate.runtime_identity

    def drifted(c, r):
        ident = real(c, r)
        ident["packages"] = {**ident["packages"], "numpy": "0.0.0"}
        return ident

    monkeypatch.setattr(evaluate, "runtime_identity", drifted)
    with pytest.raises(evaluate.EvaluateError, match="runtime identity differs"):
        evaluate.run_evaluate(cfg, CONFIG)
    assert not no_test_read
    assert not (project / "results/test_access.json").exists()


def test_primary_contrast_mismatch_aborts_before_test_read(project, cfg, frozen, no_test_read):
    path = project / "results/freeze_manifest.json"
    m = json.loads(path.read_text())
    other = next(c for c in m["candidates"] if c.startswith("gbm") and c != m["selected_gbm_id"])
    m["evaluation"]["primary_contrast"]["gbm"] = other
    path.write_text(json.dumps(m, indent=2))
    _commit(project, "tamper contrast")
    with pytest.raises(evaluate.EvaluateError, match="primary contrast"):
        evaluate.run_evaluate(cfg, CONFIG)
    assert not no_test_read


# --------------------------------------------------------------------------- self-check


def test_recipes_equal_the_fitted_model_objects(project, cfg, frozen):
    """Non-circular: the frozen recipes (loaded from files) equal the predictions of the model
    objects fitted by `train` / `gbm` (glum's own predict on the training design; LightGBM's own
    transformed predict on the training encoding), for every train row."""
    tr = train.load_train(cfg)
    m = frozen
    paths = freeze.project_paths(cfg, project)
    inputs = freeze.read_inputs(project, paths)
    art = freeze.load_artifacts(project, paths, inputs)
    preds = freeze.predict(m["recipes"], art, tr)
    np.testing.assert_array_equal(CAPTURED["glm_train_ids"], tr["IDpol"].to_numpy())
    np.testing.assert_array_equal(CAPTURED["gbm_train_ids"], tr["IDpol"].to_numpy())
    claims = (tr["N"] > 0).to_numpy()

    def fitted(key):
        stored = inputs["glm_models"][key]
        hits = [(X, mod) for X, mod in CAPTURED["glm_fits"] if mod.to_dict() == stored]
        assert hits, f"no fitted object equals the stored {key}"
        X, mod = hits[0]
        D, _ = glm._design(X, mod.features)
        return mod.estimator.predict(D)

    rtol = 1e-12
    sev = m["selection"]["selected_glm_severity"]
    fxs = preds[f"glm_freq_x_sev[{sev}]"]
    np.testing.assert_allclose(fxs["frequency"], fitted("frequency"), rtol=rtol)
    np.testing.assert_allclose(
        preds["bonus_malus_only"]["frequency"], fitted("frequency_bonus_malus_only"), rtol=rtol
    )
    np.testing.assert_allclose(preds["glm_tweedie"]["pure_premium"], fitted("tweedie"), rtol=rtol)
    const = float(tr["S_cap"].sum() / tr["N"].sum())
    if inputs["glm_models"][f"severity_{sev}"].get("kind") == "constant":
        np.testing.assert_allclose(fxs["severity"], const, rtol=rtol)
    else:
        np.testing.assert_allclose(fxs["severity"][claims], fitted(f"severity_{sev}"), rtol=rtol)
    np.testing.assert_allclose(fxs["pure_premium"], fxs["frequency"] * fxs["severity"], rtol=rtol)
    null = float(tr["freq_count"].sum() / tr["Exposure"].sum())
    np.testing.assert_allclose(preds["pp_null"]["frequency"], null, rtol=rtol)
    np.testing.assert_allclose(preds["pp_null"]["severity"], const, rtol=rtol)
    assert CAPTURED["encoding"] == inputs["gbm_models"]["encoding"]
    X = gbm.encode(tr, CAPTURED["encoding"])
    b = CAPTURED["boosters"]
    g = preds["gbm_freq_x_sev"]
    np.testing.assert_allclose(g["frequency"], b["frequency"].predict(X), rtol=rtol)
    np.testing.assert_allclose(g["severity"], b["severity"].predict(X), rtol=rtol)
    np.testing.assert_allclose(
        preds["gbm_tweedie"]["pure_premium"], b["tweedie"].predict(X), rtol=rtol
    )


def test_self_check_sample_covers_edge_cases(project, cfg):
    tr = train.load_train(cfg)
    sc = cfg["freeze"]["self_check"]
    rows = freeze.self_check_rows(tr, sc)
    assert len(rows) >= sc["rows"] and rows["IDpol"].is_monotonic_increasing
    for col in sc["extreme_columns"]:
        assert rows[col].min() == tr[col].min() and rows[col].max() == tr[col].max(), col
    for col in sc["level_columns"]:
        assert set(rows[col].astype(str)) == set(tr[col].astype(str)), col
    assert freeze.edge_case_summary(rows, tr, sc)
    with pytest.raises(freeze.FreezeError, match="edge cases"):
        freeze.edge_case_summary(rows[rows["Exposure"] != tr["Exposure"].max()], tr, sc)


def test_predictions_are_deterministic_and_row_order_invariant(project, cfg, frozen, monkeypatch):
    tr = train.load_train(cfg)
    paths = freeze.project_paths(cfg, project)
    art = freeze.load_artifacts(project, paths, freeze.read_inputs(project, paths))
    rows = tr.iloc[:700].reset_index(drop=True)
    det = freeze.determinism_check(frozen["recipes"], art, rows, 5)
    assert det["permuted_row_order"] == "identical"
    real = freeze.predict
    calls = []

    def flaky(rec, a, df):
        calls.append(1)
        out = real(rec, a, df)
        if len(calls) == 2:
            out["glm_tweedie"]["pure_premium"] = out["glm_tweedie"]["pure_premium"] * (1 + 1e-15)
        return out

    monkeypatch.setattr(freeze, "predict", flaky)
    with pytest.raises(freeze.FreezeError, match="determinism"):
        freeze.determinism_check(frozen["recipes"], art, rows, 5)
