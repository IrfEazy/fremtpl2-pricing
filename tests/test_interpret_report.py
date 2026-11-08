"""`fremtpl2 interpret` (train-only interpretation) and `fremtpl2 report` (test report).

Both run on the synthetic project of ``test_freeze_evaluate`` (artifacts of the real `train` /
`gbm` stages on synthetic data, its own git repository); the real project is never touched.
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fremtpl2_pricing import cli, evaluate, figures, freeze, gbm, interpret, report
from fremtpl2_pricing.train import load_train
from synth import _DEFAULTS
from test_freeze_evaluate import (  # noqa: F401  (fixtures)
    CONFIG,
    _forbid_fitting,
    cfg,
    frozen,
    project,
    template,
)


def _with_sections(cfg: dict) -> dict:
    """Shrunk interpret / report / figures sections for the tiny synthetic data."""
    cfg = copy.deepcopy(cfg)
    cfg["figures"] = {**copy.deepcopy(_DEFAULTS["figures"]), "dpi": 40, "width": 6.0, "height": 2.5}
    ic = copy.deepcopy(_DEFAULTS["interpret"])
    ic["sample"]["size"] = 1500
    ic["dependence"].update(
        max_points=300, numeric_bins=4, factors=["BonusMalus", "DrivAge", "Region"]
    )
    ic["glm_comparison"]["numeric_bins"] = 3
    ic["interaction"].update(
        reference_drivage_band="26-30", young_drivage_bands=["18-20", "21-25"],
        min_exposure_years=5.0,
    )  # fmt: skip
    ic["interaction"]["vehpower_bands"] = {"lower_edges": [4, 6]}
    ic["monotonicity"].update(rows=200, max_points=300)
    cfg["interpret"] = ic
    cfg["report"] = copy.deepcopy(_DEFAULTS["report"])
    cfg["report"]["bootstrap_histogram_bins"] = 10
    return cfg


@pytest.fixture
def icfg(cfg):
    return _with_sections(cfg)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- interpret


def test_interpret_end_to_end_train_only_and_fit_free(project, icfg, monkeypatch):
    _forbid_fitting(monkeypatch)
    res = interpret.run_interpret(icfg, root=project)
    out = json.loads((project / "results/interpretation.json").read_text())
    assert set(res["timing_seconds"]) == {"contributions", "comparison", "monotonicity", "figures"}
    assert out["test_set_used"] is False
    # the recorded ids are train rows of the frozen split, all with cv_fold >= 0
    split = pd.read_parquet(project / "data/processed/split.parquet").set_index("IDpol")
    ids = out["sample"]["ids"]
    assert len(ids) == out["sample"]["size"] == 1500 and ids == sorted(ids)
    assert (split.loc[ids, "primary_set"] == "train").all() and (
        split.loc[ids, "cv_fold"] >= 0
    ).all()
    assert out["sample"]["cv_fold_range"][0] >= 0
    assert (
        out["sample"]["ids_sha256"] == hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
    )
    # contributions sum to the raw score within the recorded tolerance; offset is stated
    for b in out["shap"].values():
        assert b["max_abs_row_error"] <= b["tolerance"] == 1e-9
        assert b["rows_checked"] == 1500 and b["base_value_constant"]
        assert set(b["mean_abs"]) == set(gbm.feature_names(icfg["models"]["gbm"]))
        assert b["ranking"][0] == max(b["mean_abs"], key=b["mean_abs"].get)
        assert set(b["dependence"]) == {"BonusMalus", "DrivAge", "Region"}
    assert "outside the contributions" in out["notes"]["exposure_offset"]
    assert "never equated" in out["notes"]["contributions"]
    # GLM comparison: reference level is zero on both sides; continuous terms are centred
    cmp_ = {c: v["factors"] for c, v in out["glm_comparison"]["components"].items()}
    assert set(cmp_) == {"frequency", "severity", "tweedie"}
    for facs in cmp_.values():
        for v in facs.values():
            if v["kind"] == "levels":
                ref = next(r for r in v["rows"] if r["label"] == v["reference"])
                assert ref["gbm_mean_centred"] == 0.0
                assert all(x == 0.0 for x in ref["glm_log_relativity"].values())
            else:
                w = np.array([r["weight"] for r in v["rows"]])
                c = np.array([r["gbm_mean_centred"] for r in v["rows"]])
                assert abs(np.average(c, weights=w)) < 1e-12
    # the log relativities equal log(exp(coef)) of the GLM table (frequency: selected = full GLM)
    freq = pd.read_csv(project / "results/glm_relativities_frequency.csv", dtype={"level": str})
    area = freq[freq["feature"] == "Area"].set_index("level")["relativity"]
    for r in cmp_["frequency"]["Area"]["rows"]:
        assert r["glm_log_relativity"]["frequency"] == pytest.approx(
            float(np.log(area[r["label"]]))
        )
    # severity: the GLM of the selected candidate is compared, on claim rows weighted by N
    sevc = out["glm_comparison"]["components"]["severity"]
    sel_sev = out["selected_glm_id"].split("[")[1][:-1]
    roles = {g["key"]: g["role"] for g in sevc["glms"]}
    if sel_sev != "constant":
        assert roles[f"severity_{sel_sev}"] == "selected"
    assert "claim rows" in sevc["population"] and sevc["weight"] == "N"
    assert (
        sum(r["n"] for r in next(iter(sevc["factors"].values()))["rows"]) <= out["sample"]["claims"]
    )
    # interaction note and monotonicity of six candidates
    assert out["interaction"]["component"] == "frequency" and out["interaction"]["cells"]
    mono = out["monotonicity"]["candidates"]
    assert list(mono) == [
        "pp_null", "bonus_malus_only", out["selected_glm_id"], "glm_tweedie",
        "gbm_freq_x_sev", "gbm_tweedie",
    ]  # fmt: skip
    for c in mono.values():
        assert 0.0 <= c["pure_premium"]["share_rows_with_decrease"] <= 1.0
    assert "not guaranteed to be monotone" in out["monotonicity"]["statement"]
    # constraint register
    reg = out["constraint_register"]
    assert set(reg["constrained_components"]) == {"frequency", "severity", "tweedie"}
    assert reg["constrained_components"]["severity"] == {}
    sweep = reg["verification"]["min_step_raw_score_by_component"]
    assert sweep["frequency"]["min_step_raw_score"] >= -1e-12
    assert sweep["tweedie"]["min_step_raw_score"] >= -1e-12
    assert "separately tuned" in reg["cv_effect"]["separately_tuned_note"]
    assert reg["cv_effect"]["controlled_ablation"]["frequency"]["rounds"] > 0
    assert "optimistic" in reg["cv_effect"]["kind"]
    assert "most complex or the simplest" in reg["grid_edge_disclosure"]
    for e in reg["grid_edges"].values():
        assert all(a["position"] in {"lowest grid value", "highest grid value", "interior"}
                   for a in e["axes"].values())  # fmt: skip
    # report and figures
    md = (project / "reports/interpretation.md").read_text()
    assert "no test row is read" in md and "never equated" in md
    assert "(thin)" in md or "exposure-years are marked thin" in md
    assert str(out["monotonicity"]["rows"]) in md
    pm = out["monotonicity"]["candidates"]["gbm_freq_x_sev"]["pure_premium"]
    assert f"{100 * pm['share_rows_with_decrease']:.2f}%" in md
    assert len(out["figures"]) == 8
    for f in out["figures"]:
        assert (project / "reports/figures" / f).is_file()
        assert f"figures/{f}" in md
    for f in out["figures"]:
        assert (project / "reports/figures" / f).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not (project / "data/processed/interpret_staging").exists()


def test_interpret_is_deterministic(project, icfg):
    interpret.run_interpret(icfg, root=project)
    files = ["results/interpretation.json", "reports/interpretation.md"]
    files += [
        f"reports/figures/{p.name}" for p in sorted((project / "reports/figures").glob("*.png"))
    ]
    first = {f: _sha(project / f) for f in files}
    interpret.run_interpret(icfg, root=project)
    assert {f: _sha(project / f) for f in files} == first
    assert len(files) > 8


def test_interpret_refuses_test_rows(icfg):
    split = pd.DataFrame(
        {"IDpol": [1, 2, 3], "primary_set": ["train", "test", "train"], "cv_fold": [0, -1, 1]}
    )
    with pytest.raises(ValueError, match="train-only guard"):
        interpret.draw_sample(split, icfg["interpret"]["sample"])
    only_train = split[split["primary_set"] == "train"].assign(cv_fold=[0, -1])
    with pytest.raises(ValueError, match="train-only guard"):
        interpret.draw_sample(only_train, icfg["interpret"]["sample"])


def test_sample_is_fixed_by_seed(project, icfg):
    train = load_train(icfg)
    a = interpret.draw_sample(train.sample(frac=1.0, random_state=1), icfg["interpret"]["sample"])
    b = interpret.draw_sample(train, icfg["interpret"]["sample"])
    assert a["IDpol"].tolist() == b["IDpol"].tolist()
    other = copy.deepcopy(icfg["interpret"]["sample"])
    other["seed"] += 1
    assert interpret.draw_sample(train, other)["IDpol"].tolist() != b["IDpol"].tolist()


def test_contribution_identity_is_enforced(project, icfg):
    inputs = freeze.read_inputs(project, freeze.project_paths(icfg, project))
    paths = freeze.project_paths(icfg, project)
    art = freeze.load_artifacts(project, paths, inputs, 1)
    train = load_train(icfg)
    sample = interpret.draw_sample(train, icfg["interpret"]["sample"])
    X = gbm.encode(sample, art["encoding"])
    ok = interpret.contributions(art["boosters"]["frequency"], X, 1, 1e-9)
    assert ok["phi"].shape == (len(sample), X.shape[1])
    # the identity holds per row: contributions + base value = raw score
    raw = gbm.raw_score(art["boosters"]["frequency"], X)
    assert np.allclose(ok["phi"].sum(axis=1) + ok["base_value"], raw, atol=1e-9)
    with pytest.raises(interpret.InterpretError, match="do not sum to the raw score"):
        interpret.contributions(art["boosters"]["frequency"], X, 1, -1.0)


def test_monotone_diagnostic_detects_decreases(project, icfg, monkeypatch):
    paths = freeze.project_paths(icfg, project)
    inputs = freeze.read_inputs(project, paths)
    sel = freeze.recompute_selections(inputs, icfg)
    train = load_train(icfg)
    recipes = freeze.build_recipes(inputs, sel, icfg, paths, 0.1)
    art = freeze.load_artifacts(project, paths, inputs, 1)
    sample = interpret.draw_sample(train, icfg["interpret"]["sample"])
    mcfg = copy.deepcopy(icfg["interpret"]["monotonicity"])
    pts = np.array([1.0, 2.0])
    calls = []

    def fake_predict(rec, a, df):
        calls.append(df["BonusMalus"].iloc[0])
        # candidate x decreases by 10% at the second point for the first half of the rows
        n = len(df)
        level = 1.0 + 0.0 * np.arange(n)
        if len(calls) == 2:
            level = np.where(np.arange(n) < n // 2, 0.9, 1.0)
        return {"x": {"pure_premium": level, "frequency": level, "severity": 1.0 + 0 * level}}

    monkeypatch.setattr(freeze, "predict", fake_predict)
    monkeypatch.setitem(recipes, "x", {"family": "freq_x_sev", "frequency": {"kind": "constant"},
                                       "severity": {"kind": "constant"}})  # fmt: skip
    res = interpret.monotone_diagnostic({"x": recipes["x"]}, art, sample, mcfg, pts)
    pp = res["candidates"]["x"]["pure_premium"]
    assert pp["share_rows_with_decrease"] == pytest.approx(0.5)
    assert pp["largest_relative_decrease"] == pytest.approx(0.1)
    assert res["candidates"]["x"]["severity"]["share_rows_with_decrease"] == 0.0
    assert len(calls) == 2


def test_sweep_points_distinct_train_values_or_dense_grid(project, icfg):
    train = load_train(icfg)
    mcfg = copy.deepcopy(icfg["interpret"]["monotonicity"])
    pts, mode = interpret.sweep_points(train, mcfg)
    assert np.array_equal(pts, np.unique(train["BonusMalus"].to_numpy(dtype=float)))
    assert "distinct" in mode
    dense = train.iloc[:40].copy()
    dense["BonusMalus"] = np.arange(50.0, 90.0)
    mcfg["max_points"] = 5
    pts, mode = interpret.sweep_points(dense, mcfg)
    assert len(pts) == 5 and pts[0] == 50.0 and pts[-1] == 89.0 and "dense grid" in mode


def test_glm_effects_use_stored_coefficients_and_mark_absent_factors(project, icfg):
    paths = freeze.project_paths(icfg, project)
    inputs = freeze.read_inputs(project, paths)
    art = freeze.load_artifacts(project, paths, inputs, 1)
    fmap = art["feature_maps"]["severity"]
    model = inputs["glm_models"]["severity_simplified"]
    eff = interpret.glm_effects(model, fmap)
    assert set(eff) == {f["name"] for f in model["features"]}
    assert eff["BonusMalus"] == model["coef"][0]
    for f in model["features"]:
        if f["kind"] == "cat":
            assert eff[f["name"]][f["categories"][0]] == 0.0
    assert interpret.glm_effects(None, fmap) == {}


def test_severity_comparison_claim_rows_weighted_by_n_and_absent_factor(project, icfg):
    paths = freeze.project_paths(icfg, project)
    inputs = freeze.read_inputs(project, paths)
    art = freeze.load_artifacts(project, paths, inputs, 1)
    fmap = art["feature_maps"]["severity"]
    sample = interpret.draw_sample(load_train(icfg), icfg["interpret"]["sample"])
    names = list(art["encoding"]["numeric"]) + list(art["encoding"]["categorical"])
    phi = np.random.default_rng(0).normal(size=(len(sample), len(names) + 1))
    src = [{"key": "severity_simplified", "role": "selected", "label": "s",
            "model": inputs["glm_models"]["severity_simplified"]}]  # fmt: skip
    out, panels = interpret.glm_comparison("severity", phi, names, sample, fmap, src, 3)
    claims = sample[sample["N"] > 0]
    absent = [f for f, v in out.items() if not v["present"]["severity_simplified"]]
    assert absent  # the simplified GLM drops factors
    for f in absent:
        for r in out[f]["rows"]:
            val = r.get("glm_log_relativity", r.get("glm_effect_centred"))
            assert val["severity_simplified"] == 0.0
    f0 = next(iter(out))
    assert sum(r["n"] for r in out[f0]["rows"]) == len(claims)
    assert sum(r["weight"] for r in out[f0]["rows"]) == pytest.approx(float(claims["N"].sum()))
    assert any("absent" in g["label"] for p in panels for g in p["glms"])


def test_grid_edge_disclosure_follows_the_grid():
    def cv(grid, chosen):
        return {
            "config": {"gbm": {"components": {c: {"grid": grid} for c in gbm.COMPONENTS}}},
            "components": {c: {"chosen_hp": chosen, "cap_reached_any_fold": False}
                           for c in gbm.COMPONENTS},
        }  # fmt: skip

    two = interpret.grid_edge_disclosure(cv({"a": [1, 2]}, {"a": 2}))
    assert "two values per axis" in two and "3 of the 3" in two
    three = interpret.grid_edge_disclosure(cv({"a": [1, 2, 3]}, {"a": 2}))
    assert "two values" not in three and "0 of the 3" in three and "outside the grid" not in three


def test_grid_edge_positions():
    cv = {
        "config": {"gbm": {"components": {c: {"grid": {"a": [1, 2], "b": [1, 2, 3]}}
                                          for c in gbm.COMPONENTS}}},
        "components": {c: {"chosen_hp": {"a": 2, "b": 2}, "cap_reached_any_fold": False}
                       for c in gbm.COMPONENTS},
    }  # fmt: skip
    e = interpret.grid_edges(cv)["frequency"]
    assert e["axes"]["a"]["position"] == "highest grid value"
    assert e["axes"]["b"]["position"] == "interior" and not e["all_on_boundary"]


def test_dump_json_keeps_number_lists_on_one_line_and_rejects_nan():
    text = interpret.dump_json({"ids": list(range(30)), "name": ["a", "b"], "x": 1.5})
    assert "[0, 1, 2" in text.splitlines()[1] and json.loads(text)["ids"] == list(range(30))
    with pytest.raises(interpret.InterpretError):
        interpret.dump_json({"x": float("nan")})


def test_figures_are_deterministic_png(tmp_path):
    fcfg = {"dpi": 40, "width": 4.0, "height": 2.0}
    panels = [
        {
            "name": "p",
            "xlabel": "x",
            "ylabel": "y",
            "series": [{"label": "s", "x": [0, 1], "y": [1, 2]}],
        }
    ]
    a = figures.line_panels(fcfg, tmp_path / "a.png", panels, "t", ncol=1)
    b = figures.line_panels(fcfg, tmp_path / "b.png", panels, "t", ncol=1)
    assert a == "a.png" and b == "b.png"
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()
    assert b"Software" not in (tmp_path / "a.png").read_bytes()


# --------------------------------------------------------------------------- report


def _written(root: Path) -> set[str]:
    return {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
    }


def test_report_without_journal_writes_nothing(project, icfg):
    before = _written(project)
    with pytest.raises(report.ReportError, match="no test-access journal"):
        report.run_report(icfg, root=project)
    assert _written(project) == before


def test_report_without_completed_attempt_writes_nothing(project, icfg, frozen, monkeypatch):
    class Crash(BaseException):
        pass

    def crash(c):
        raise Crash

    monkeypatch.setattr(evaluate, "load_test", crash)
    with pytest.raises(Crash):
        evaluate.run_evaluate(icfg, CONFIG)
    before = _written(project)
    with pytest.raises(report.ReportError, match="no completed attempt.*interrupted"):
        report.run_report(icfg, root=project)
    assert _written(project) == before


@pytest.fixture
def evaluated(project, icfg, frozen):
    out = evaluate.run_evaluate(icfg, CONFIG)
    return out


def test_report_renders_the_completed_attempt(project, icfg, evaluated, monkeypatch):
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("a data file was read"))
    res = report.run_report(icfg, root=project)
    md = (project / "reports/test_evaluation.md").read_text()
    d = evaluated["bootstrap"]["group"]["delta"]
    assert res["delta"] == d
    # protocol section 6 wording, decision from the file
    assert report.PRIMARY_CONTRAST in md and report.DECISION_RULE in md
    assert f"**Result: {d['decision']}.**" in md
    assert f"{d['upper']:.5f}" in md and f"{d['lower']:.5f}" in md
    assert "conditional on the frozen fitted models" in md
    assert "labelled sensitivity" in md and "secondary" in md and "descriptive" in md
    for cid in evaluated["candidates_order"]:
        assert f"| {cid}" in md
    # every attempt of the journal is listed
    assert evaluated["attempt_id"] in md and "COMPLETE" in md or "this attempt" in md
    expected = {
        "lorenz", "lift", "double_lift", "calibration_decile", "calibration_factor",
        "bootstrap_delta", "bootstrap_delta_policy",
    }  # fmt: skip
    prefix = icfg["report"]["outputs"]["figure_prefix"]
    for stem in expected:
        f = project / "reports/figures" / f"{prefix}{stem}.png"
        assert f.is_file() and f"figures/{prefix}{stem}.png" in md
    assert not (project / "data/processed/report_staging").exists()
    # deterministic: a second run gives byte-identical outputs
    files = ["reports/test_evaluation.md"] + [
        f"reports/figures/{prefix}{s}.png" for s in sorted(expected)
    ]
    first = {f: _sha(project / f) for f in files}
    report.run_report(icfg, root=project)
    assert {f: _sha(project / f) for f in files} == first


def test_report_fails_closed_on_a_modified_attempt_file(project, icfg, evaluated):
    tables = project / "results/test/attempt-1/tables.json"
    tables.write_text(tables.read_text().replace("0.", "1.", 1))
    before = _written(project)
    with pytest.raises(report.ReportError, match="sha256 differs"):
        report.run_report(icfg, root=project)
    assert _written(project) == before


def test_report_fails_closed_when_the_pointer_differs(project, icfg, evaluated):
    ptr = project / "results/metrics.json"
    doc = json.loads(ptr.read_text())
    doc["source"]["attempt_id"] = "other"
    ptr.write_text(json.dumps(doc))
    with pytest.raises(report.ReportError, match="source.attempt_id differs"):
        report.run_report(icfg, root=project)


def test_cli_wires_interpret_and_report(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "run_interpret", lambda c: calls.append("i") or {"timing_seconds": {}})
    monkeypatch.setattr(cli, "run_report", lambda c: calls.append("r"))
    assert cli.main(["interpret", "--config", "configs/default.yaml"]) == 0
    assert cli.main(["report", "--config", "configs/default.yaml"]) == 0
    assert calls == ["i", "r"]

    def boom(c):
        raise report.ReportError("no completed attempt")

    monkeypatch.setattr(cli, "run_report", boom)
    assert cli.main(["report", "--config", "configs/default.yaml"]) == 1


# --------------------------------------------------------------------------- run-state independence


def test_report_does_not_depend_on_the_mirror_but_still_checks_it(project, icfg, evaluated):
    from fremtpl2_pricing.freeze import project_paths

    mirror = project / project_paths(icfg, project)["mirror"]
    assert mirror.is_file()
    report.run_report(icfg, root=project)
    with_mirror = (project / "reports/test_evaluation.md").read_bytes()
    saved = mirror.read_bytes()
    mirror.write_bytes(saved + b'{"event": "FORGED"}\n')  # a disagreeing mirror still fails
    with pytest.raises(Exception):  # noqa: B017
        report.run_report(icfg, root=project)
    mirror.unlink()
    report.run_report(icfg, root=project)
    assert (project / "reports/test_evaluation.md").read_bytes() == with_mirror
    assert report.JOURNAL_CHECK.encode() in with_mirror


def test_interpret_source_hash_ignores_volatile_keys_only(tmp_path):
    vol = interpret.load_reproduce_config(interpret._policy_path(tmp_path))["check"][
        "volatile_keys"
    ]
    key = vol[0]

    def entry(doc):
        f = tmp_path / "x.json"
        f.write_text(json.dumps(doc))
        return interpret._source_entry(f, "x.json", sorted(vol))

    base = {"loss": 1.5, "folds": [{"a": 1, key: 3.0}], key: 9.0}
    a = entry(base)
    b = entry({"loss": 1.5, "folds": [{"a": 1, key: 77.0}], key: 0.1})
    c = entry({"loss": 1.6, "folds": [{"a": 1, key: 3.0}], key: 9.0})
    assert a["sha256"] == b["sha256"] and a["sha256"] != c["sha256"]
    assert a["content_sha256_excluding"] == sorted(vol)
    plain = entry({"loss": 1.5})  # no telemetry: byte hash
    assert "content_sha256_excluding" not in plain
