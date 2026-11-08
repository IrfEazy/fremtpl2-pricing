"""Offline tests for `fremtpl2 sensitivities` on synthetic data."""

import copy
import json
from pathlib import Path

import pytest

from fremtpl2_pricing import cli, metrics, sensitivities, train
from fremtpl2_pricing.clean import clean
from fremtpl2_pricing.split import assign_split
from synth import make_cfg, make_freq, make_sev

POWER = 1.6


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("sens")
    cfg = make_cfg()
    cfg["paths"] = {k: str(root / k) for k in ("raw", "processed", "results", "reports")}
    freq = make_freq(5000)
    sev = make_sev(freq)
    split = assign_split(clean(freq, sev, cfg["clean"]).freq, cfg)
    Path(cfg["paths"]["processed"]).mkdir()
    split.to_parquet(Path(cfg["paths"]["processed"]) / "split.parquet", index=False)
    Path(cfg["paths"]["results"]).mkdir()
    cv = Path(cfg["paths"]["results"]) / cfg["models"]["outputs"]["cv_json"]
    cv.write_text(json.dumps({"final": {"power_cv": {"power": POWER}}}))
    cfg["models"]["bootstrap"]["replicates"] = 20
    return cfg, freq, sev, split


@pytest.fixture(scope="module")
def run(world):
    cfg, freq, sev, _ = world
    mp = pytest.MonkeyPatch()
    mp.setattr(sensitivities, "load_raw", lambda c: (freq, sev))
    try:
        yield sensitivities.run_sensitivities(cfg), sensitivities.run_sensitivities(cfg)
    finally:
        mp.undo()


def test_every_declared_sensitivity_runs_and_converges(world, run):
    cfg = world[0]
    res = run[0]
    names = ["primary", *cfg["clean"]["sensitivities"]]
    assert list(res["sensitivities"]) == names and res["config"]["test_set_used"] is False
    assert res["config"]["fixed_tweedie_power"] == POWER
    for n in names:
        s = res["sensitivities"][n]
        assert s["convergence"]["n_not_converged"] == 0 and s["convergence"]["n_fits"] > 0
        assert len(s["folds"]) == 5
        expect = set(sensitivities.PP_MODELS) - (set() if s["freq_x_sev_valid"] else {"freq_x_sev"})
        assert set(s["pooled"]["pure_premium"]) == expect
        assert s["freq_x_sev_valid"] == (n != "frequency_recorded_claim_nb")
        assert "deviance_explained" in s["pooled"]["frequency"]["full"]
    outs = cfg["models"]["sensitivities"]["outputs"]
    assert (Path(cfg["paths"]["results"]) / outs["json"]).stat().st_size > 0
    text = (Path(cfg["paths"]["reports"]) / outs["report"]).read_text()
    assert "untouched" in text and "section 7" in text and "not comparable" in text


def test_excluding_sensitivity_drops_rows_but_keeps_split(world, run):
    cfg, freq, sev, split = world
    rows = run[0]["sensitivities"]
    assert rows["short_records"]["rows"]["policies"] < rows["primary"]["rows"]["policies"]
    assert rows["cap_500"]["rows"]["policies"] == rows["primary"]["rows"]["policies"]
    res = sensitivities.clean_sensitivity(freq, sev, cfg["clean"], "short_records")
    tr = sensitivities.sensitivity_train_frame(res.freq, split, 5)
    ref = split.set_index("IDpol").loc[tr["IDpol"]]
    assert (tr["cv_fold"].to_numpy() == ref["cv_fold"].to_numpy()).all()
    assert (tr["group_id"].to_numpy() == ref["group_id"].to_numpy()).all()
    assert set(tr["primary_set"]) == {"train"}


def test_fixed_power_read_from_cv_json(world, tmp_path):
    cfg = copy.deepcopy(world[0])
    assert sensitivities.fixed_power(cfg) == POWER
    cfg["paths"]["results"] = str(tmp_path)
    with pytest.raises(ValueError, match="fremtpl2 train"):
        sensitivities.fixed_power(cfg)
    (tmp_path / cfg["models"]["outputs"]["cv_json"]).write_text("{}")
    with pytest.raises(ValueError, match="power_cv"):
        sensitivities.fixed_power(cfg)


def test_unknown_name_and_selection(world):
    cfg = copy.deepcopy(world[0])
    assert sensitivities.sensitivity_names(cfg) == list(cfg["clean"]["sensitivities"])
    cfg["models"]["sensitivities"]["names"] = ["cap_500"]
    assert sensitivities.sensitivity_names(cfg) == ["cap_500"]
    cfg["models"]["sensitivities"]["names"] = ["nope"]
    with pytest.raises(ValueError, match="unknown"):
        sensitivities.sensitivity_names(cfg)


def test_train_only_guard(world):
    cfg, freq, sev, split = world
    res = clean(freq, sev, cfg["clean"])
    tr = sensitivities.sensitivity_train_frame(res.freq, split, 5)
    bad = tr.copy()
    bad.loc[bad.index[0], "cv_fold"] = -1
    with pytest.raises(ValueError, match="train-only"):
        sensitivities.cv_one(bad, cfg, POWER, cfg["clean"])
    with pytest.raises(ValueError, match="subset"):
        sensitivities.sensitivity_train_frame(res.freq, split.iloc[1:], 5)
    assert train.assert_train_only(tr) is None


def test_determinism_and_bootstrap_diagnostic(run):
    a, b = run
    for r in (a, b):
        del r["timing_seconds"]
        for s in r["sensitivities"].values():
            del s["seconds"]
    assert a == b
    d = a["bootstrap_unit_diagnostic"]
    assert set(d["units"]) == {"group", "policy"}
    assert d["units"]["group"]["n_units"] < d["units"]["policy"]["n_units"]
    for k in sensitivities.BOOT_KEYS:
        assert d["ratio_group_over_policy"][k]["se"] > 0
        assert d["units"]["policy"]["metrics"][k]["width"] > 0


def test_cli_is_wired(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "run_sensitivities", lambda c: called.append(c))
    assert cli.main(["sensitivities"]) == 0 and called


def test_recorded_claim_nb_has_no_freq_x_sev(run):
    s = run[0]["sensitivities"]["frequency_recorded_claim_nb"]
    assert s["freq_x_sev_valid"] is False and "differs" in s["freq_x_sev_note"]
    assert "freq_x_sev" not in s["pooled"]["pure_premium"]
    assert set(s["comparisons"]["pooled"]) == {
        "tweedie_minus_null",
        "frequency_full_minus_null",
        "severity_full_minus_null",
    }
    assert s["pooled"]["frequency"]["full"]["deviance_explained"] is not None
    assert run[0]["sensitivities"]["primary"]["freq_x_sev_note"] is None


def test_report_states_why_freq_x_sev_is_missing(world, run):
    cfg = world[0]
    outs = cfg["models"]["sensitivities"]["outputs"]
    text = (Path(cfg["paths"]["reports"]) / outs["report"]).read_text()
    assert "Frequency x severity not computed" in text and "severity denominator" in text


def test_reference_power_kept_and_fixed_power_fitted_separately(world):
    cfg, freq, sev, split = world
    mcfg = sensitivities.fixed_power_mcfg(cfg["models"], POWER)
    ref = cfg["models"]["tweedie"]["reference_power"]
    assert POWER != ref and mcfg["tweedie"]["reference_power"] == ref
    assert mcfg["tweedie"]["fixed_power_sensitivities"] == [POWER]
    assert sensitivities.fixed_power_column(mcfg, POWER) == f"pp_tweedie_fixed_{POWER}"
    same = sensitivities.fixed_power_mcfg(cfg["models"], ref)
    assert same["tweedie"]["fixed_power_sensitivities"] == []
    assert sensitivities.fixed_power_column(same, ref) == "pp_tweedie_ref"
    tr = sensitivities.sensitivity_train_frame(clean(freq, sev, cfg["clean"]).freq, split, 5)
    oof, res, _ = sensitivities.cv_one(tr, cfg, POWER, cfg["clean"])
    assert not (oof["pp_tweedie"] == oof["pp_tweedie_ref"]).all()
    assert (oof["pp_tweedie"] == oof[f"pp_tweedie_{POWER}"]).all()
    got = (
        metrics.tweedie_deviance(
            (oof["S_cap"] / oof["Exposure"]).to_numpy(),
            oof["pp_tweedie"].to_numpy() / oof["Exposure"].to_numpy(),
            ref,
            weight=oof["Exposure"].to_numpy(),
        )
        / oof["Exposure"].sum()
    )
    assert res["pooled"]["pure_premium"]["tweedie"][
        "tweedie_ref_deviance_per_exposure"
    ] == pytest.approx(got)
