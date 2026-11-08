"""Offline tests of the LightGBM candidates on synthetic data (train rows only)."""

import copy
import json

import lightgbm as lgb
import numpy as np
import pytest

from fremtpl2_pricing import gbm, train
from fremtpl2_pricing.clean import clean
from fremtpl2_pricing.gbm_report import render_report
from fremtpl2_pricing.split import assign_split
from synth import make_cfg, make_freq, make_sev


def tiny_gbm_block():
    comp = lambda obj, metric, mono: {  # noqa: E731
        "objective": obj,
        "metric": metric,
        "monotone": mono,
        "params": {},
        "grid": {"num_leaves": [3, 5], "min_child_samples": [20], "learning_rate": [0.2]},
    }
    return {
        "seed": 11,
        "features": {
            "numeric": ["DrivAge", "VehAge", "VehPower", "BonusMalus", "Density"],
            "categorical": ["Area", "VehGas", "VehBrand", "Region"],
        },
        "unseen_level": "missing",
        "deterministic": {"deterministic": True, "force_row_wise": True, "num_threads": 1},
        "base_params": {
            "verbosity": -1,
            "max_bin": 63,
            "min_data_per_group": 10,
            "cat_smooth": 5.0,
            "monotone_constraints_method": "basic",
        },
        "early_stopping": {
            "round_cap": 25,
            "patience": 5,
            "min_improvement": 0.0,
            "round_rounding": "nearest",
            "at_cap": "record_use_cap",
        },
        "components": {
            "frequency": comp("poisson", "poisson", {"BonusMalus": 1}),
            "severity": comp("gamma", "gamma_deviance", {}),
            "tweedie": comp("tweedie", "tweedie", {"BonusMalus": 1}),
        },
        "diagnostic": {"enabled": True, "components": ["frequency", "tweedie"], "ablation": True},
        "checks": {
            "reload_rows": 500,
            "reload_tolerance": 1e-9,
            "sweep_rows": 50,
            "sweep_points": 8,
            "monotone_tolerance": 1e-12,
        },
        "outputs": {
            "cv_json": "cv_gbm.json",
            "models_json": "gbm_models.json",
            "report": "gbm_cv.md",
            "model_dir": "models",
            "staging_dir": "gbm_staging",
            "failure_diagnostic": "gbm_FAILED.json",
        },
    }


@pytest.fixture(scope="module")
def cfg():
    c = make_cfg()
    c["models"]["gbm"] = tiny_gbm_block()
    return c


@pytest.fixture(scope="module")
def train_df(cfg):
    freq = make_freq(5000)
    res = clean(freq, make_sev(freq), cfg["clean"])
    split = assign_split(res.freq, cfg)
    return train.train_frame(res.freq, split, cfg["split"]["cv_folds"])


@pytest.fixture(scope="module")
def fold(cfg, train_df):
    gcfg = cfg["models"]["gbm"]
    k, fit, valid = next(train.outer_splits(train_df, 5, True))
    return gbm.make_fold(fit, valid, gcfg, k), fit, valid


def fit_component(comp, cfg, fd, constrained=True, rounds=15, hp=None):
    gcfg = cfg["models"]["gbm"]
    hp = hp or {"num_leaves": 5, "min_child_samples": 20, "learning_rate": 0.2}
    params = gbm.build_params(gcfg, comp, hp, 1.5, constrained, with_metric=False)
    X, y, w, i = gbm.component_data(comp, fd["Xf"], fd["ef"], fd["cf"], fd["sf"], fd["nf"])
    names = gbm.feature_names(gcfg)
    ds = gbm._dataset(X, y, w, i, names, gcfg["features"]["categorical"], params)
    return lgb.train(params, ds, num_boost_round=rounds), names


def test_validate_rejects_constrained_severity(cfg):
    bad = copy.deepcopy(cfg["models"]["gbm"])
    bad["components"]["severity"]["monotone"] = {"BonusMalus": 1}
    with pytest.raises(ValueError, match="severity"):
        gbm.validate_gbm(bad)
    gbm.validate_gbm(cfg["models"]["gbm"])


def test_exposure_doubles_counts_not_rates(cfg, fold):
    fd, _, _ = fold
    booster, _ = fit_component("frequency", cfg, fd)
    raw = gbm.raw_score(booster, fd["Xv"])
    e = fd["ev"]
    np.testing.assert_allclose(gbm.expected_counts(raw, 2 * e), 2 * gbm.expected_counts(raw, e))
    np.testing.assert_allclose(gbm.expected_counts(raw, 2 * e) / (2 * e), gbm.annual_rate(raw))
    # the raw score is a rate: counts / exposure equals exp(raw), independent of exposure
    np.testing.assert_allclose(gbm.expected_counts(raw, e) / e, gbm.annual_rate(raw))
    # exposure is no feature and no weight of the Poisson GBM
    args = (fd["Xf"], fd["ef"], fd["cf"], fd["sf"], fd["nf"])
    _, _, w, init = gbm.component_data("frequency", *args)
    assert w is None
    np.testing.assert_allclose(init, np.log(fd["ef"]))
    assert "Exposure" not in gbm.feature_names(cfg["models"]["gbm"])


def test_severity_and_tweedie_data(cfg, fold):
    fd, _, _ = fold
    X, y, w, i = gbm.component_data("severity", fd["Xf"], fd["ef"], fd["cf"], fd["sf"], fd["nf"])
    m = fd["nf"] > 0
    assert len(y) == m.sum() and i is None
    np.testing.assert_allclose(y * w, fd["sf"][m])
    X, y, w, i = gbm.component_data("tweedie", fd["Xf"], fd["ef"], fd["cf"], fd["sf"], fd["nf"])
    np.testing.assert_allclose(w, fd["ef"])
    np.testing.assert_allclose(y * w, fd["sf"])


def test_save_load_equivalence(cfg, fold, tmp_path):
    fd, _, _ = fold
    for comp in gbm.COMPONENTS:
        booster, _ = fit_component(comp, cfg, fd)
        path = tmp_path / f"{comp}.txt"
        booster.save_model(str(path))
        reloaded = lgb.Booster(model_file=str(path))
        np.testing.assert_allclose(
            gbm.raw_score(booster, fd["Xv"]), gbm.raw_score(reloaded, fd["Xv"]), atol=1e-12
        )


def test_feature_order_and_monotone_vector(cfg, fold):
    fd, _, _ = fold
    booster, names = fit_component("frequency", cfg, fd)
    gbm.check_feature_order(booster, names)
    with pytest.raises(ValueError, match="feature order"):
        gbm.check_feature_order(booster, names[::-1])
    vec = gbm.monotone_vector(names, {"BonusMalus": 1})
    assert vec[names.index("BonusMalus")] == 1 and sum(map(abs, vec)) == 1
    with pytest.raises(ValueError, match="unknown"):
        gbm.monotone_vector(names, {"Nope": 1})
    assert booster.params["monotone_constraints"] == vec


def test_unseen_category_policy(cfg, fold):
    fd, fit, valid = fold
    gcfg = cfg["models"]["gbm"]
    enc = gbm.fit_encoding(fit, gcfg)
    odd = valid.head(3).copy()
    odd["Region"] = "R-never-seen"
    X = gbm.encode(odd, enc)
    j = gbm.feature_names(gcfg).index("Region")
    assert np.isnan(X[:, j]).all()
    X_known = gbm.encode(valid.head(3), enc)
    assert not np.isnan(X_known[:, j]).any()
    with pytest.raises(ValueError, match="unseen"):
        gbm.encode(odd, {**enc, "unseen_level": "error"})
    booster, _ = fit_component("frequency", cfg, fd)
    assert np.isfinite(gbm.raw_score(booster, X)).all()


def test_monotone_sweep_and_finite_positive(cfg, fold):
    fd, _, _ = fold
    names = gbm.feature_names(cfg["models"]["gbm"])
    vals = np.linspace(50, 150, 15)
    for comp in ("frequency", "tweedie"):
        booster, _ = fit_component(comp, cfg, fd, constrained=True, rounds=30)
        assert gbm.monotone_sweep(booster, fd["Xv"][:100], names, "BonusMalus", vals) >= -1e-12
        rate = gbm.annual_rate(gbm.raw_score(booster, fd["Xv"]))
        assert np.isfinite(rate).all() and (rate > 0).all()
    sev, _ = fit_component("severity", cfg, fd)
    rate = gbm.annual_rate(gbm.raw_score(sev, fd["Xv"]))
    assert np.isfinite(rate).all() and (rate > 0).all()


def test_determinism(cfg, fold):
    fd, _, _ = fold
    for comp in gbm.COMPONENTS:
        a, _ = fit_component(comp, cfg, fd)
        b, _ = fit_component(comp, cfg, fd)
        np.testing.assert_array_equal(gbm.raw_score(a, fd["Xv"]), gbm.raw_score(b, fd["Xv"]))


def test_rounds_rounding():
    assert gbm.best_rounds([10, 11], "nearest") == 11  # 10.5 rounds half up
    assert gbm.best_rounds([10, 10, 11], "nearest") == 10
    with pytest.raises(ValueError):
        gbm.best_rounds([1], "floor")


def test_train_only_guards(cfg, train_df):
    gcfg = cfg["models"]["gbm"]
    bad = train_df.copy()
    bad.loc[bad.index[0], "cv_fold"] = -1
    k, fit, valid = next(train.outer_splits(train_df, 5, True))
    with pytest.raises(ValueError, match="train-only"):
        gbm.make_fold(bad.head(50), valid, gcfg, 0)
    with pytest.raises(ValueError, match="train-only"):
        gbm.run_gbm_cv(bad, cfg, None)


def test_cv_end_to_end(cfg, train_df, tmp_path):
    cfg = copy.deepcopy(cfg)
    res_dir = tmp_path / "results"
    res_dir.mkdir()
    cfg["paths"] = {"results": str(res_dir)}
    cv_name = cfg["models"]["outputs"]["cv_json"]
    (res_dir / cv_name).write_text(json.dumps({"final": {"power_cv": {"power": 1.5}}}))
    result = gbm.run_gbm_cv(train_df, cfg, tmp_path / "stage")
    assert result["config"]["tweedie_power"] == 1.5
    assert result["checks"]["passed"]
    for comp in gbm.COMPONENTS:
        block = result["components"][comp]
        assert len(block["grid"]) == 2
        assert all(len(p["fold_loss"]) == 5 for p in block["grid"])
    sel = result["selection"]
    assert sel["selected"] in (gbm.PP_FXS, gbm.PP_TW)
    assert sel["gbm_pure_premium"]["n_folds"] == 5
    d = result["diagnostic"]
    assert set(d["ablation"]) == {"frequency", "tweedie"}
    assert len(d["separately_tuned"]["primary_loss"]["constrained_tweedie"]) == 5
    for comp in gbm.COMPONENTS:
        assert (tmp_path / "stage" / "models" / f"{comp}.txt").is_file()
        assert result["final_models"]["models"][comp]["sha256"]
    text = render_report(train._clean_json(result))
    assert "optimistic tuning scores" in text and "Monotonicity diagnostic" in text


def test_hit_round_cap_only_without_patience_stop():
    assert gbm.hit_round_cap(25, 25, 25, 5)  # ran to the cap, best at the cap
    assert gbm.hit_round_cap(25, 22, 25, 5)  # ran to the cap, patience not yet elapsed
    assert not gbm.hit_round_cap(25, 20, 25, 5)  # patience fired exactly at the cap
    assert not gbm.hit_round_cap(18, 13, 25, 5)  # stopped early


def test_fitted_poisson_gbm_exposure_contract(cfg, fold):
    """Real fit path (_train_early) and raw-score prediction: exposure only enters as log offset."""
    fd, _, _ = fold
    gcfg = cfg["models"]["gbm"]
    names, cat = gbm.feature_names(gcfg), list(gcfg["features"]["categorical"])
    hp = {"num_leaves": 5, "min_child_samples": 20, "learning_rate": 0.2}
    params = gbm.build_params(gcfg, "frequency", hp, 1.5, True, with_metric=True)
    booster, best, _ = gbm._train_early(
        "frequency", params, fd, names, cat, gcfg["early_stopping"], 1.5
    )
    assert best >= 1
    assert booster.feature_name() == names and "Exposure" not in names
    _, _, w, init = gbm.component_data(
        "frequency", fd["Xf"], fd["ef"], fd["cf"], fd["sf"], fd["nf"]
    )
    assert w is None
    np.testing.assert_allclose(init, np.log(fd["ef"]))
    raw = gbm.raw_score(booster, fd["Xv"], best)
    e = fd["ev"]
    c1, c2 = gbm.expected_counts(raw, e), gbm.expected_counts(raw, 2 * e)
    np.testing.assert_allclose(c2, 2 * c1, rtol=1e-12)
    np.testing.assert_allclose(c1 / e, c2 / (2 * e), rtol=1e-12)
    # raw prediction does not depend on exposure at all (it is not an input of the model)
    np.testing.assert_allclose(gbm.raw_score(booster, fd["Xv"], best), raw)
    assert (gbm.annual_rate(raw) > 0).all()


def test_promote_all_consistent_and_rolls_back(cfg, train_df, tmp_path):
    cfg = copy.deepcopy(cfg)
    res_dir = tmp_path / "results"
    res_dir.mkdir()
    cfg["paths"] = {"results": str(res_dir)}
    (res_dir / cfg["models"]["outputs"]["cv_json"]).write_text(
        json.dumps({"final": {"power_cv": {"power": 1.5}}})
    )
    result = gbm.run_gbm_cv(train_df, cfg, tmp_path / "stage")
    models = result["final_models"]["models"]
    stage_models = tmp_path / "stage" / "models"
    gbm.verify_staged_hashes(stage_models, models)
    names = sorted(p.name for p in stage_models.iterdir())
    dest = tmp_path / "dest"
    gbm.promote_all([(stage_models, dest, names)], tmp_path / "bk")
    for m in models.values():
        assert gbm._sha256(dest / m["file"]) == m["sha256"]
    # tampered staging file is detected before anything is promoted
    (stage_models / "frequency.txt").write_text("tampered")
    with pytest.raises(ValueError, match="sha256"):
        gbm.verify_staged_hashes(stage_models, models)
    # a failing group rolls back files already replaced
    before = (dest / "frequency.txt").read_text()
    (tmp_path / "s2").mkdir()
    (tmp_path / "s2" / "frequency.txt").write_text("new")
    with pytest.raises(FileNotFoundError):
        gbm.promote_all(
            [(tmp_path / "s2", dest, ["frequency.txt"]), (tmp_path / "s2", dest, ["absent.txt"])],
            tmp_path / "bk2",
        )
    assert (dest / "frequency.txt").read_text() == before
    assert not list(dest.glob("*.tmp"))
