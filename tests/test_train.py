"""Offline tests for the milestone-2 CV pipeline on synthetic data."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fremtpl2_pricing import cli, features, train
from fremtpl2_pricing.clean import clean
from fremtpl2_pricing.cv_report import render_report
from fremtpl2_pricing.split import assign_split
from synth import make_cfg, make_freq, make_sev


def build(cfg, n=5000):
    freq = make_freq(n)
    res = clean(freq, make_sev(freq), cfg["clean"])
    split = assign_split(res.freq, cfg)
    return res, split


@pytest.fixture(scope="module")
def cfg():
    return make_cfg()


@pytest.fixture(scope="module")
def train_df(cfg):
    res, split = build(cfg)
    return train.train_frame(res.freq, split, cfg["split"]["cv_folds"])


@pytest.fixture(scope="module")
def result(cfg, train_df):
    return json.loads(json.dumps(train._clean_json(train.train_models(train_df, cfg))))


def metrics_only(res):
    return {
        k: v
        for k, v in res.items()
        if k not in ("timing_seconds", "loaded_premium", "config", "excess_loading_train")
    }


def test_train_frame_is_train_only(cfg):
    res, split = build(cfg)
    df = train.train_frame(res.freq, split, 5)
    assert set(df["primary_set"]) == {"train"} and df["cv_fold"].between(0, 4).all()
    assert len(df) < len(res.freq)
    with pytest.raises(ValueError, match="IDpol"):
        train.train_frame(res.freq.iloc[1:], split, 5)


def test_guard_raises_on_test_rows(cfg, train_df):
    bad = train_df.copy()
    bad.loc[bad.index[0], "cv_fold"] = -1
    with pytest.raises(ValueError, match="train-only"):
        train.assert_train_only(bad)
    with pytest.raises(ValueError, match="train-only"):
        train.cross_validate(bad, cfg)
    with pytest.raises(ValueError, match="train-only"):
        train.final_fit(bad, cfg, 1.5, Path("."))
    res, split = build(cfg)
    split.loc[split.index[0], "cv_fold"] = 9
    with pytest.raises(ValueError, match="cv_fold"):
        train.train_frame(res.freq, split, 5)


def test_no_group_overlap_in_outer_and_inner_splits(train_df):
    n_inner = 0
    for k, fit, valid in train.outer_splits(train_df, 5, True):
        assert set(valid["cv_fold"]) == {k} and k not in set(fit["cv_fold"])
        assert not set(fit["group_id"]) & set(valid["group_id"])
        for j, ifit, ivalid in train.inner_splits(fit, True):
            n_inner += 1
            assert j != k and not set(ifit["group_id"]) & set(ivalid["group_id"])
            assert not set(ifit["IDpol"]) & set(ivalid["IDpol"])
    assert n_inner == 20
    # an overlap is detected
    bad = train_df.copy()
    bad.loc[bad["cv_fold"] == 1, "group_id"] = bad.loc[bad["cv_fold"] == 0, "group_id"].iloc[0]
    with pytest.raises(ValueError, match="overlap"):
        list(train.outer_splits(bad, 5, True))


def test_inner_selection_never_sees_outer_fold_rows(cfg, train_df, monkeypatch):
    seen = []
    real = features.fit

    def spy(df, *a, **kw):
        seen.append((set(df["IDpol"]), set(df["cv_fold"])))
        return real(df, *a, **kw)

    monkeypatch.setattr(features, "fit", spy)
    for k, fit, _ in train.outer_splits(train_df, 5, True):
        seen.clear()
        for _, ifit, ivalid in train.inner_splits(fit, True):
            train._inner_task(ifit, ivalid, cfg["features"], cfg["models"], [1.5])
        assert len(seen) == 4
        for ids, folds in seen:
            assert k not in folds and ids <= set(fit["IDpol"])


def test_excess_loading_invariance(cfg, train_df):
    other = copy.deepcopy(cfg)
    other["clean"]["excess_loading"]["mode"] = "none"
    a, b = train.train_models(train_df, cfg), train.train_models(train_df, other)
    assert metrics_only(train._clean_json(a)) == metrics_only(train._clean_json(b))
    assert a["loaded_premium"]["loading_total"] > 0 == b["loaded_premium"]["loading_total"]
    assert a["excess_loading_train"] > 0 and b["excess_loading_train"] == 0


def test_determinism(cfg, train_df, result):
    again = json.loads(json.dumps(train._clean_json(train.train_models(train_df, cfg))))
    assert metrics_only(again) == metrics_only(result)


def test_result_keys_and_report(result):
    assert set(result) >= {
        "config", "rows", "cv", "final", "loaded_premium", "excess_loading_train", "timing_seconds",
    }  # fmt: skip
    cv = result["cv"]
    assert len(cv["selected_powers"]) == 5 and len(cv["folds"]) == 5
    assert all(s["power"] in (1.4, 1.5, 1.6) for s in cv["selected_powers"])
    assert set(cv["pooled"]) == {"frequency", "severity", "pure_premium"}
    assert set(cv["pooled"]["pure_premium"]) == set(train.PP_MODELS)
    assert cv["fold_summary"]["frequency"]["full"]["deviance_per_exposure"]["sd"] is not None
    assert len(cv["double_lift"]) == 10
    assert set(cv["calibration"]) == {"BonusMalus", "DrivAge", "VehAge", "Region"}
    assert result["final"]["power_cv"]["power"] in (1.4, 1.5, 1.6)
    assert result["config"]["test_set_used"] is False
    md = render_report(result)
    assert "test set is untouched" in md and "milestone 3" in md
    assert "Selected Tweedie power" in md and "Double lift" in md
    assert "## Findings" in md and "Severity full GLM, out-of-fold deviance explained" in md
    assert "point mass at the cap" in md and "not converged" in md


def test_power_selection_diagnostics_and_convergence(result, cfg):
    t = cfg["models"]["tweedie"]
    grid = [str(float(p)) for p in t["power_grid"]]
    for s in result["cv"]["selected_powers"]:
        ll = s["inner_loglik"]
        assert set(ll) == set(grid)
        assert s["power"] == float(max(grid, key=lambda g: ll[g]))  # argmax of the loglik
        assert s["at_grid_edge"] == (s["power"] in (1.4, 1.6))
        assert s["loglik_spread_nats"] == pytest.approx(max(ll.values()) - min(ll.values()))
        assert s["weakly_identified"] == (s["loglik_spread_nats"] < t["weak_identification_nats"])
        assert set(s["diagnostic_inner_poisson_deviance_per_exposure"]) == set(grid)
    assert "tweedie_ref" in result["cv"]["pooled"]["pure_premium"]
    conv = result["convergence"]
    n_grid = len(grid)
    assert conv["n_fits"] == 5 * (3 + n_grid) + 5 * 4 * n_grid  # no final fit without out_dir
    assert conv["n_not_converged"] == len(conv["not_converged"])
    assert conv["solver"] == cfg["models"]["glm"]["solver"]
    cp = result["cap_point_mass"]
    assert cp["policies_at_cap_multiple"] <= cp["policies_with_claim_cost"]


def test_coverage_check_fails_fast_before_fitting(cfg, train_df, monkeypatch):
    bad = train_df.copy()
    # a non-grouped level (VehGas) that exists in a single fold only and has no claims elsewhere
    bad["VehGas"] = bad["VehGas"].astype(str)
    only = bad.index[bad["cv_fold"] == 0][:1]
    bad.loc[only, "VehGas"] = "Hydrogen"
    monkeypatch.setattr(train, "Parallel", lambda *a, **k: pytest.fail("fits started"))
    with pytest.raises(ValueError, match="not covered by its fitting part"):
        train.cross_validate(bad, cfg)


def test_end_to_end_files(cfg, tmp_path, monkeypatch):
    cfg = copy.deepcopy(cfg)
    cfg["paths"] = {k: str(tmp_path / k) for k in ("raw", "processed", "results", "reports")}
    res, split = build(cfg)
    Path(cfg["paths"]["processed"]).mkdir()
    split.to_parquet(Path(cfg["paths"]["processed"]) / "split.parquet", index=False)
    monkeypatch.setattr(train, "load_raw", lambda c: (make_freq(5000), make_sev(make_freq(5000))))
    out = train.run_train(cfg)
    outs = cfg["models"]["outputs"]
    for key in outs:
        base = cfg["paths"]["reports"] if key == "report" else cfg["paths"]["results"]
        assert (Path(base) / outs[key]).stat().st_size > 0, key
    on_disk = json.loads((Path(cfg["paths"]["results"]) / outs["cv_json"]).read_text())
    assert on_disk["cv"]["selected_powers"] == out["cv"]["selected_powers"]
    tab = pd.read_csv(Path(cfg["paths"]["results"]) / outs["relativities_tweedie"])
    assert {"term", "coef", "se", "relativity"} <= set(tab.columns)
    assert np.isfinite(tab["coef"]).all()
    fm = json.loads((Path(cfg["paths"]["results"]) / outs["feature_map"]).read_text())
    assert set(fm) == {"policy", "severity"}


def test_cli_train_is_wired(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(cli, "run_train", lambda c: called.append(c))
    assert cli.main(["train"]) == 0 and called
    assert cli.main(["evaluate"]) == 2
