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


def claim_rows(res, train_df):
    return res.sev[res.sev["IDpol"].isin(train_df["IDpol"])].reset_index(drop=True)


@pytest.fixture(scope="module")
def claims(cfg, train_df):
    res, _ = build(cfg)
    return claim_rows(res, train_df)


@pytest.fixture(scope="module")
def result(cfg, train_df, claims, tmp_path_factory):
    out = tmp_path_factory.mktemp("final")
    return json.loads(json.dumps(train._clean_json(train.train_models(train_df, cfg, claims, out))))


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


def test_excess_loading_invariance(cfg, train_df, claims):
    other = copy.deepcopy(cfg)
    other["clean"]["excess_loading"]["mode"] = "none"
    a, b = train.train_models(train_df, cfg, claims), train.train_models(train_df, other, claims)
    assert metrics_only(train._clean_json(a)) == metrics_only(train._clean_json(b))
    assert a["loaded_premium"]["loading_total"] > 0 == b["loaded_premium"]["loading_total"]
    assert a["excess_loading_train"] > 0 and b["excess_loading_train"] == 0


def test_determinism(cfg, train_df, claims, result, tmp_path):
    again = json.loads(
        json.dumps(train._clean_json(train.train_models(train_df, cfg, claims, tmp_path)))
    )
    assert metrics_only(again) == metrics_only(result)


def test_result_keys_and_report(result, cfg):
    assert set(result) >= {
        "config", "rows", "cv", "final", "loaded_premium", "excess_loading_train", "timing_seconds",
    }  # fmt: skip
    cv = result["cv"]
    assert len(cv["selected_powers"]) == 5 and len(cv["folds"]) == 5
    assert all(s["power"] in (1.4, 1.5, 1.6) for s in cv["selected_powers"])
    assert set(cv["pooled"]) == {"frequency", "severity", "pure_premium"}
    assert set(cv["pooled"]["pure_premium"]) == set(train.pp_models(cfg["models"]))
    assert cv["fold_summary"]["frequency"]["full"]["deviance_per_exposure"]["sd"] is not None
    assert len(cv["double_lift"]) == 10
    assert set(cv["calibration"]) == {"BonusMalus", "DrivAge", "VehAge", "Region"}
    assert result["final"]["power_cv"]["power"] in (1.4, 1.5, 1.6)
    assert result["config"]["test_set_used"] is False
    md = render_report(result)
    assert "test set is untouched" in md and "milestone 3" in md
    assert "Selected Tweedie power" in md and "Double lift" in md
    assert "## Findings" in md and "Severity full GLM, out-of-fold deviance explained" in md
    assert "Cap diagnostics" in md and "not converged" in md
    assert "sandwich standard errors" in md and "not causal" in md
    assert "Solver accuracy check" in md and "Dispersion search" in md and "Rating-profile" in md


def test_power_selection_diagnostics_and_convergence(result, cfg):
    t = cfg["models"]["tweedie"]
    grid = [str(float(p)) for p in t["power_grid"]]
    for s in result["cv"]["selected_powers"]:
        ll = s["inner_loglik"]
        assert set(ll) == set(grid)
        assert s["power"] == float(max(grid, key=lambda g: ll[g]))  # argmax of the loglik
        assert s["at_grid_edge"] == (s["power"] in (1.4, 1.6))
        assert s["loglik_spread_nats"] == pytest.approx(max(ll.values()) - min(ll.values()))
        assert "weakly_identified" not in s
        assert set(s["diagnostic_inner_poisson_deviance_per_exposure"]) == set(grid)
    assert "tweedie_ref" in result["cv"]["pooled"]["pure_premium"]
    conv = result["convergence"]
    n_grid = len(grid)
    n_outer = 3 + n_grid + len(set(t["fixed_power_sensitivities"]) - {float(g) for g in grid})
    n_l2 = len(cfg["models"]["glm"]["l2_grid"])
    # outer + inner fits, ladder fits (simplified, L2 per penalty, full) per outer fold, L2 inner
    # fits (frequency + one per penalty) per inner split, and the six final fits (out_dir given)
    assert conv["n_fits"] == (
        5 * n_outer + 5 * 4 * n_grid + 5 * (n_l2 + 2) + 5 * 4 * (1 + n_l2) + 6
    )
    assert conv["n_not_converged"] == len(conv["not_converged"])
    assert conv["solver"] == cfg["models"]["glm"]["solver"]
    assert "weak_identification_nats" not in t


def test_final_power_paired_diagnostics_and_stability(result, cfg):
    fp = result["final"]["power_cv"]
    grid = [float(p) for p in cfg["models"]["tweedie"]["power_grid"]]
    sel = fp["power"]
    assert sel == max(grid, key=lambda p: fp["cv_loglik"][str(p)])
    for row in fp["fold_loglik"].values():
        assert set(row) == {str(p) for p in grid}
    for nb in fp["neighbours"]:
        i = grid.index(sel)
        assert nb["neighbour"] in (grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)])
        per = [
            fp["fold_loglik"][str(k)][str(sel)] - fp["fold_loglik"][str(k)][str(nb["neighbour"])]
            for k in range(5)
        ]
        assert nb["per_fold_diff_nats"] == pytest.approx(per)
        assert nb["total_diff_nats"] == pytest.approx(sum(per))
        assert nb["total_diff_nats"] == pytest.approx(
            fp["cv_loglik"][str(sel)] - fp["cv_loglik"][str(nb["neighbour"])]
        )
        # the policy split reconciles with the total
        assert nb["capped_policies_diff_nats"] + nb["other_policies_diff_nats"] == pytest.approx(
            nb["total_diff_nats"]
        )
        assert 0 <= nb["folds_agreeing_in_sign"] <= 5
    st = fp["stability"]
    assert st["folds_selecting_final"] == st["selected_by_outer_fold"].count(sel)


def test_fixed_power_sensitivities_are_in_the_comparison(result, cfg):
    fixed = cfg["models"]["tweedie"]["fixed_power_sensitivities"]
    pooled = result["cv"]["pooled"]["pure_premium"]
    for p in fixed:
        assert train.fixed_name(p) in pooled
    # the common deviance stays at the reference power for every model
    assert "tweedie_ref_deviance_per_exposure" in pooled[train.fixed_name(fixed[0])]


def test_cap_diagnostics_from_claim_rows(result, train_df, claims, cfg):
    cd = result["cap_diagnostics"]
    cap = cfg["clean"]["large_loss"]["cap"]
    over = claims[claims["ClaimAmount"] > cap]
    assert cd["capped_claims"] == len(over) and cd["claims"] == len(claims)
    assert cd["policies_with_capped_claim"] == over["IDpol"].nunique()
    assert cd["amount_removed"] == pytest.approx((over["ClaimAmount"] - cap).sum())
    assert cd["capped_claims_amount_in_s_cap"] == pytest.approx(cap * len(over))
    assert cd["share_of_s_cap"] == pytest.approx(cap * len(over) / train_df["S_cap"].sum())
    assert cd["reconciliation"]["claim_rows_s_cap"] == pytest.approx(train_df["S_cap"].sum())
    assert cd["policies_with_capped_claim"] == cd["reconciliation"]["policies_with_excess"]
    assert cd["policies_with_capped_claim"] > 0


def test_dispersion_summary_and_solver_check_recorded(result, cfg):
    assert result["dispersion"]["n_dispersion_fits"] > 0
    assert result["dispersion"]["max_expansions"] >= 0
    sc = result["solver_check"]
    assert sc["passed"] and sc["failures"] == []
    assert {r["stage"] for r in sc["records"]} == {"cv", "final"}
    assert (
        len(sc["records"]) == 5 * 3 + 3 + 2
    )  # per fold tweedie, frequency, severity; 3 final + simplified and L2
    assert {(r["stage"], r["model"]) for r in sc["records"]} == {
        ("cv", "tweedie"), ("cv", "frequency"), ("cv", "severity"),
        ("final", "tweedie"), ("final", "frequency"), ("final", "severity"),
        ("final", "severity_simplified"), ("final", "severity_l2"),
    }  # fmt: skip
    pp = sc["pure_premium"]
    assert len(pp["records"]) == 5
    assert pp["max_rel_diff"] <= sc["thresholds"]["rel_pure_premium_diff"]
    assert pp["max_abs_diff"] >= 0 and 0 < pp["max_rel_diff_at_prediction_quantile"] <= 1
    assert 0 < pp["max_abs_diff_at_prediction_quantile"] <= 1
    ll = sc["loglik_checks"]
    sel = result["final"]["power_cv"]
    nbrs = [n["neighbour"] for n in sel["neighbours"]]
    assert {r["neighbour"] for r in ll["records"]} == set(nbrs)
    assert len(ll["records"]) == len(nbrs) * (5 + 1)  # per fold and the total
    assert ll["all_signs_kept"]
    assert ll["max_rel_change"] < sc["thresholds"]["loglik_difference_rel_change"]
    assert result["checks"]["passed"]
    assert (
        sc["tight_solver"]["gradient_tol"] == cfg["models"]["glm"]["solver_check"]["gradient_tol"]
    )
    for key in sc["maxima"]:
        assert 0 <= sc["maxima"][key] <= sc["thresholds"][key]
    assert all(r["tight_converged"] for r in sc["records"])


def _run_cfg(cfg, tmp_path, monkeypatch):
    cfg = copy.deepcopy(cfg)
    cfg["paths"] = {k: str(tmp_path / k) for k in ("raw", "processed", "results", "reports")}
    res, split = build(cfg)
    Path(cfg["paths"]["processed"]).mkdir()
    split.to_parquet(Path(cfg["paths"]["processed"]) / "split.parquet", index=False)
    monkeypatch.setattr(train, "load_raw", lambda c: (make_freq(5000), make_sev(make_freq(5000))))
    return cfg


def _seed_committed(cfg):
    """Pre-existing artifacts of an earlier successful run (sentinel content)."""
    outs = cfg["models"]["outputs"]
    paths = [
        Path(cfg["paths"]["results"]) / outs[k]
        for k in (
            "cv_json", "relativities_frequency", "relativities_severity", "relativities_tweedie",
            "feature_map", "models",
        )
    ] + [Path(cfg["paths"]["reports"]) / outs["report"]]  # fmt: skip
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("COMMITTED\n", encoding="utf-8")
    return paths


def test_solver_check_failure_keeps_committed_artifacts(cfg, tmp_path, monkeypatch):
    bad = _run_cfg(cfg, tmp_path, monkeypatch)
    bad["models"]["glm"]["solver_check"]["max_coef_diff"] = 0.0
    committed = _seed_committed(bad)
    with pytest.raises(RuntimeError, match="gates failed.*solver_check"):
        train.run_train(bad)
    assert all(p.read_text() == "COMMITTED\n" for p in committed)  # untouched
    processed = Path(bad["paths"]["processed"])
    diag = json.loads((processed / bad["models"]["outputs"]["failure_diagnostic"]).read_text())
    assert diag["checks"]["solver_check"] is False and diag["result"]["solver_check"]["failures"]
    assert not list(Path(bad["paths"]["results"]).glob("*.tmp"))


def test_convergence_failure_blocks_promotion(cfg, tmp_path, monkeypatch):
    bad = _run_cfg(cfg, tmp_path, monkeypatch)
    bad["models"]["glm"]["solver"] = {"max_iter": 1, "gradient_tol": 1.0e-5}
    bad["models"]["glm"]["solver_check"]["max_coef_diff"] = 1e9
    bad["models"]["glm"]["solver_check"]["max_rel_prediction_diff"] = 1e9
    bad["models"]["glm"]["solver_check"]["max_rel_deviance_diff"] = 1e9
    bad["models"]["glm"]["solver_check"]["max_rel_pure_premium_diff"] = 1e9
    bad["models"]["glm"]["solver_check"]["max_loglik_diff_rel_change"] = 1e9
    committed = _seed_committed(bad)
    with pytest.raises(RuntimeError, match="convergence"):
        train.run_train(bad)
    assert all(p.read_text() == "COMMITTED\n" for p in committed)
    assert (
        Path(bad["paths"]["processed"]) / bad["models"]["outputs"]["failure_diagnostic"]
    ).exists()


def test_success_promotes_and_clears_staging_and_failure_file(cfg, tmp_path, monkeypatch):
    ok = _run_cfg(cfg, tmp_path, monkeypatch)
    committed = _seed_committed(ok)
    processed = Path(ok["paths"]["processed"])
    stale = processed / ok["models"]["outputs"]["failure_diagnostic"]
    stale.write_text("{}")
    train.run_train(ok)
    assert all(p.read_text() != "COMMITTED\n" for p in committed)
    assert not stale.exists()
    assert not (processed / ok["models"]["outputs"]["staging_dir"]).exists()


def test_cap_reconciliation_is_enforced(cfg, train_df, claims):
    ok = train.cap_diagnostics(train_df, claims, 100.0, 1e-9)
    assert ok["reconciliation"]["passed"]
    bad = claims.copy()
    bad.loc[bad.index[0], "ClaimAmount_capped"] += 1.0
    with pytest.raises(ValueError, match="cap reconciliation failed"):
        train.cap_diagnostics(train_df, bad, 100.0, 1e-9)
    assert train.cap_diagnostics(train_df, bad, 100.0, 1.0)  # a looser configured tolerance


def test_reconciliation_failure_writes_diagnostic_only(cfg, tmp_path, monkeypatch):
    bad = _run_cfg(cfg, tmp_path, monkeypatch)
    committed = _seed_committed(bad)
    monkeypatch.setattr(train, "cap_diagnostics", lambda *a, **k: (_ for _ in ()).throw(
        ValueError("cap reconciliation failed: x")))  # fmt: skip
    with pytest.raises(RuntimeError, match="cap reconciliation failed"):
        train.run_train(bad)
    assert all(p.read_text() == "COMMITTED\n" for p in committed)
    assert (
        Path(bad["paths"]["processed"]) / bad["models"]["outputs"]["failure_diagnostic"]
    ).exists()


def test_loglik_difference_checks_flag_sign_flip_and_large_change():
    checks = [
        {"fold": 0, "config": {1.5: -10.0, 1.4: -20.0}, "tight": {1.5: -10.0, 1.4: -20.0}},
        {"fold": 1, "config": {1.5: -10.0, 1.4: -11.0}, "tight": {1.5: -10.0, 1.4: -9.5}},
    ]
    rows = train.loglik_difference_checks(checks, 1.5, [1.4], 0.01)
    by = {r["fold"]: r for r in rows}
    assert by[0]["passed"] and not by[1]["sign_kept"] and not by[1]["passed"]
    assert by["total"]["neighbour"] == 1.4


def test_group_summary_train_only(result, train_df, cfg):
    gs = result["groups"]
    assert gs["overall"]["groups"] == train_df["group_id"].nunique()
    assert gs["overall"]["policies"] == len(train_df)
    assert [f["groups"] for f in gs["per_fold"]] == [
        train_df.loc[train_df["cv_fold"] == k, "group_id"].nunique() for k in range(5)
    ]
    top = gs["overall"]
    sizes = train_df.groupby("group_id").size().sort_values(ascending=False)
    assert top["top_groups"] == max(1, int(np.ceil(0.01 * len(sizes))))
    assert top["top_policy_share"] == pytest.approx(
        sizes.iloc[: top["top_groups"]].sum() / len(train_df)
    )
    assert 0 < top["top_exposure_share"] <= 1
    with pytest.raises(ValueError, match="train-only"):
        bad = train_df.copy()
        bad.loc[bad.index[0], "cv_fold"] = -1
        train.group_summary(bad, cfg["models"]["group_summary"], 5)


def test_coverage_check_fails_fast_before_fitting(cfg, train_df, monkeypatch):
    bad = train_df.copy()
    # a non-grouped level (VehGas) that exists in a single fold only and has no claims elsewhere
    bad["VehGas"] = bad["VehGas"].astype(str)
    only = bad.index[bad["cv_fold"] == 0][:1]
    bad.loc[only, "VehGas"] = "Hydrogen"
    monkeypatch.setattr(train, "Parallel", lambda *a, **k: pytest.fail("fits started"))
    with pytest.raises(ValueError, match="not covered by the policy schema"):
        train.cross_validate(bad, cfg)


def test_end_to_end_files(cfg, tmp_path, monkeypatch):
    cfg = copy.deepcopy(cfg)
    cfg["paths"] = {k: str(tmp_path / k) for k in ("raw", "processed", "results", "reports")}
    res, split = build(cfg)
    Path(cfg["paths"]["processed"]).mkdir()
    split.to_parquet(Path(cfg["paths"]["processed"]) / "split.parquet", index=False)
    monkeypatch.setattr(train, "load_raw", lambda c: (make_freq(5000), make_sev(make_freq(5000))))
    out = train.run_train(cfg)
    assert out["solver_check"]["passed"]
    outs = cfg["models"]["outputs"]
    for key in set(outs) - {"staging_dir", "failure_diagnostic"}:
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
    assert cli.main(["report"]) == 2  # stub; never call the real evaluate on the project


def test_selection_block(result, cfg):
    s = result["selection"]
    ids = [e["id"] for e in cfg["models"]["glm"]["severity_ladder"]]
    assert ids == ["constant", "simplified", "l2", "full"]
    assert set(s["pure_premium_loss_per_fold"]) == set(ids)
    assert all(len(v) == 5 for v in s["pure_premium_loss_per_fold"].values())
    assert s["selected_severity"] in ids and s["ladder"]["selected"] == s["selected_severity"]
    assert [st["next"] for st in s["ladder"]["steps"]] == ids[1:]
    assert s["ladder"]["rule_version"] == s["rule_version"]
    # the full rung equals the existing freq x sev out-of-fold primary loss of each fold
    cv = result["cv"]["folds"]
    ref = [f["pure_premium"]["freq_x_sev"]["tweedie_ref_deviance_per_exposure"] for f in cv]
    assert np.allclose(s["pure_premium_loss_per_fold"]["full"], ref, rtol=1e-8)
    tw = [f["pure_premium"]["tweedie"]["tweedie_ref_deviance_per_exposure"] for f in cv]
    assert np.allclose(s["tweedie_loss_per_fold"], tw)
    assert s["freq_x_sev_vs_tweedie"]["selected"] in ("freq_x_sev", "tweedie")
    assert s["selected_glm_pure_premium"]["id"]
    l2 = s["l2"]
    grid = cfg["models"]["glm"]["l2_grid"]
    assert l2["grid"] == grid and min(grid) > 0 and l2["final_penalty"] in grid
    assert all(p in grid for p in l2["penalty_by_outer_fold"])
    assert len(l2["inner"]["0"]["per_inner_fold"]) == 4
    assert isinstance(l2["final_penalty_at_grid_edge"], bool) and "alpha" in l2["convention"]
    md = render_report(result)
    assert "Severity ladder and GLM pure premium selection" in md
    assert "not a significance test" in md and "Selected GLM pure premium candidate" in md


def test_selection_fits_in_convergence_and_solver_records(result, cfg):
    assert result["convergence"]["n_not_converged"] == 0
    models = {r["model"] for r in result["solver_check"]["records"] if r["stage"] == "final"}
    assert {"severity_simplified", "severity_l2"} <= models


def test_final_fit_writes_all_severity_candidates(cfg, train_df, tmp_path):
    train.final_fit(train_df, cfg, 1.5, tmp_path, l2=0.01)
    models = json.loads((tmp_path / cfg["models"]["outputs"]["models"]).read_text())
    assert {"severity", "severity_constant", "severity_simplified", "severity_l2"} <= set(models)
    assert models["severity_l2"]["l2"] == 0.01 and models["severity_constant"]["value"] > 0
    assert [f["name"] for f in models["severity_simplified"]["features"]] == [
        "VehAge", "DrivAge", "BonusMalus", "VehGas"
    ] or len(models["severity_simplified"]["features"]) == 4  # fmt: skip
    with pytest.raises(ValueError, match="no penalty"):
        train.final_fit(train_df, cfg, 1.5, tmp_path)
