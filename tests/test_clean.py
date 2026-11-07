import copy

import numpy as np
import pandas as pd
import pytest

from fremtpl2_pricing.clean import clean, clean_sensitivity, sensitivity_rules, sev_status
from fremtpl2_pricing.split import assign_split
from fremtpl2_pricing.targets import build_target_table, excess_loading
from synth import make_cfg, make_freq, make_sev

freq = make_freq()
sev = make_sev(freq)
rules = make_cfg()["clean"]


def tiny(extra_exposure=None):
    base = {"Exposure": 1.0}
    f = pd.DataFrame(
        [
            {"IDpol": 1, "ClaimNb": 0, **base},
            {"IDpol": 2, "ClaimNb": 2, **base},  # 2 sev rows
            {"IDpol": 3, "ClaimNb": 1, **base},  # no sev rows
            {"IDpol": 4, "ClaimNb": 6, **base},  # 7 sev rows, N > 5 once rows are counted
            {"IDpol": 5, "ClaimNb": 1, "Exposure": 0.5},  # 1 sev row
        ]
    )
    s = pd.DataFrame(
        {
            "IDpol": [2, 2] + [4] * 7 + [5, 99],
            "ClaimAmount": [10.0, 20.0] + [2000.0] * 7 + [300.0, 7.0],  # 99 is an orphan
        }
    )
    return f, s


def test_rules_applied():
    r = clean(freq, sev, rules)
    assert r.freq["Exposure"].max() <= 1.0
    assert r.freq["ClaimNb"].equals(freq["ClaimNb"])  # no ClaimNb clipping in the primary rules
    assert (r.sev["ClaimAmount_capped"] <= 1000.0).all()
    assert (r.sev["ClaimAmount"] >= r.sev["ClaimAmount_capped"]).all()
    assert r.sev["IDpol"].isin(r.freq["IDpol"]).all()
    assert r.report["sev_without_policy"]["rows"] == 2
    assert r.report["claim_nb"] == {"max": None}
    assert r.report["exposure_cap"]["policies"] == int((freq["Exposure"] > 1).sum())
    assert len(r.freq) == len(freq)
    assert list(r.orphans["IDpol"]) == [-1, -2]


def test_target_construction_cap_before_aggregation():
    f, s = tiny()
    r = clean(f, s, rules)
    t = r.freq.set_index("IDpol")
    assert t["N"].to_dict() == {1: 0, 2: 2, 3: 0, 4: 7, 5: 1}
    assert t["S"].to_dict() == {1: 0.0, 2: 30.0, 3: 0.0, 4: 14000.0, 5: 300.0}
    # cap 1000 per claim, then aggregation: 7 x 1000, not min(14000, 1000)
    assert t["S_cap"].to_dict() == {1: 0.0, 2: 30.0, 3: 0.0, 4: 7000.0, 5: 300.0}
    assert t["excess"].to_dict() == {1: 0.0, 2: 0.0, 3: 0.0, 4: 7000.0, 5: 0.0}
    assert t["ClaimNb_raw"].to_dict() == {1: 0, 2: 2, 3: 1, 4: 6, 5: 1}
    assert t["claim_discrepancy"].to_dict() == {1: False, 2: False, 3: True, 4: True, 5: False}
    assert t["freq_count"].equals(t["N"])
    assert ((t["N"] == 0) == (t["S"] == 0)).all()
    pos = t[t["N"] > 0]
    assert np.allclose(pos["N"] * (pos["S_cap"] / pos["N"]), pos["S_cap"])


def test_recorded_claimnb_frequency_is_selectable_option():
    f, s = tiny()
    rr = copy.deepcopy(rules)
    rr["targets"]["frequency_count"] = "claim_nb"
    t = clean(f, s, rr).freq.set_index("IDpol")
    assert t["freq_count"].to_dict() == {1: 0, 2: 2, 3: 1, 4: 6, 5: 1}
    assert t["N"].to_dict() == {1: 0, 2: 2, 3: 0, 4: 7, 5: 1}  # N_i unchanged
    rr["targets"]["frequency_count"] = "bogus"
    with pytest.raises(ValueError):
        clean(f, s, rr)


def test_nonpositive_or_nonfinite_amounts_are_quarantined_and_counted():
    f, s = tiny()
    bad = pd.DataFrame({"IDpol": [2, 2, 5], "ClaimAmount": [0.0, -5.0, np.nan]})
    r = clean(f, pd.concat([s, bad], ignore_index=True), rules)
    assert r.report["sev_quarantine"]["rows"] == 3
    assert (r.sev["ClaimAmount"] > 0).all() and np.isfinite(r.sev["ClaimAmount"]).all()
    t = r.freq.set_index("IDpol")
    assert t.loc[2, "N"] == 2 and t.loc[5, "N"] == 1  # quarantined rows do not count
    with pytest.raises(ValueError):
        build_target_table(f, bad, 1000.0, "sev_rows")  # the table itself refuses bad amounts


def test_invariant_n_zero_iff_s_zero_enforced():
    f, s = tiny()
    t = build_target_table(f, s[s["IDpol"] != 99], 1000.0, "sev_rows")
    assert ((t["N"] == 0) == (t["S"] == 0)).all()
    with pytest.raises(ValueError):
        build_target_table(f, s, 1000.0, "sev_rows")  # orphan row present


def test_reconciliation_matched_plus_orphan_equals_total():
    f, s = tiny()
    r = clean(f, s, rules)
    rec = r.report["reconciliation"]
    assert rec["matched_rows"] == 10 and rec["orphan_rows"] == 1
    assert rec["matched_cost"] + rec["orphan_cost"] == pytest.approx(s["ClaimAmount"].sum())
    assert rec["matched_cost"] == r.sev["ClaimAmount"].sum()
    big = clean(freq, sev, rules).report["reconciliation"]
    assert big["valid_total"] == pytest.approx(big["matched_cost"] + big["orphan_cost"])
    assert big["sev_rows_raw"] == big["matched_rows"] + big["orphan_rows"]


def test_idpol_normalised_to_integers_and_non_integral_rejected():
    f, s = tiny()
    f2, s2 = f.copy(), s.copy()
    f2["IDpol"] = f2["IDpol"].astype("float64")
    s2["IDpol"] = s2["IDpol"].astype("float64")
    r = clean(f2, s2, rules)
    assert r.freq["IDpol"].dtype == "int64" and r.report["idpol"]["freq_dtype"] == "float64"
    assert r.report["idpol"]["join"].startswith("many_to_one")
    f2.loc[0, "IDpol"] = 1.5
    with pytest.raises(ValueError):
        clean(f2, s2, rules)


def test_exposure_nonpositive_quarantined_no_floor():
    f, s = tiny()
    f.loc[f["IDpol"] == 3, "Exposure"] = 0.0
    f.loc[f["IDpol"] == 1, "Exposure"] = -1.0
    f.loc[f["IDpol"] == 5, "Exposure"] = 1e-6  # tiny but positive: kept, not floored
    r = clean(f, s, rules)
    assert r.report["exposure_quarantine"]["policies"] == 2
    assert set(r.freq["IDpol"]) == {2, 4, 5}
    assert r.freq.set_index("IDpol").loc[5, "Exposure"] == 1e-6
    assert r.orphans["IDpol"].tolist() == [99]  # quarantined policies' rows are not orphans


def test_exposure_cap_unchanged_and_not_a_rating_factor():
    f, s = tiny()
    f.loc[f["IDpol"] == 2, "Exposure"] = 1.7
    assert clean(f, s, rules).freq.set_index("IDpol").loc[2, "Exposure"] == 1.0
    assert "Exposure" not in make_cfg()["rating_factors"]


def test_whole_policy_high_count_exclusion_applies_to_all_targets():
    f, s = tiny()
    r = clean_sensitivity(f, s, rules, "exclude_high_count")
    assert 4 not in set(r.freq["IDpol"]) and 4 not in set(r.sev["IDpol"])
    assert r.report["high_count"]["policies"] == 1 and r.report["high_count"]["sev_rows"] == 7
    for col in ("N", "S", "S_cap", "excess", "freq_count"):
        assert 4 not in r.freq.loc[r.freq[col] > 0, "IDpol"].tolist()
    assert r.freq["S_cap"].sum() == 330.0 and r.sev["ClaimAmount_capped"].sum() == 330.0
    rec = clean_sensitivity(f, s, rules, "frequency_recorded_claim_nb")
    assert 4 in set(rec.freq["IDpol"])  # only the named sensitivity excludes


def test_sensitivity_selection_by_name():
    f, s = tiny()
    assert sensitivity_rules(rules, None)["large_loss"]["cap"] == 1000.0
    assert sensitivity_rules(rules, "primary") == sensitivity_rules(rules, None)
    assert sensitivity_rules(rules, "cap_uncapped")["large_loss"]["cap"] is None
    assert "sensitivities" not in sensitivity_rules(rules, "cap_500")
    assert rules["large_loss"]["cap"] == 1000.0  # base rules not mutated
    assert clean_sensitivity(f, s, rules, "cap_500").freq["S_cap"].sum() == 30 + 3500 + 300
    assert clean_sensitivity(f, s, rules, "cap_uncapped").freq["excess"].sum() == 0.0
    f2 = f.copy()
    f2.loc[0, "Exposure"] = 1.6
    assert clean_sensitivity(f2, s, rules, "raw_exposure").freq["Exposure"].max() == 1.6
    short = clean_sensitivity(f, s, rules, "short_records")
    assert (short.freq["Exposure"] >= 0.05).all() and len(short.freq) == 5
    with pytest.raises(ValueError, match="unknown sensitivity"):
        clean_sensitivity(f, s, rules, "nope")


def test_excess_loading_uses_train_only():
    f, s = tiny()
    t = clean(f, s, rules).freq
    train = t[t["IDpol"].isin([2, 4, 5])]
    assert excess_loading(train, rules) == pytest.approx(7000.0 / 2.5)
    assert excess_loading(t, {"excess_loading": {"mode": "none"}}) == 0.0
    with pytest.raises(ValueError):
        excess_loading(train, {"excess_loading": {"mode": "bogus"}})


def test_exclusion_actions_drop_policies_and_their_severity():
    rr = copy.deepcopy(rules)
    rr["claim_nb"] = {"max": 4, "action": "exclude"}
    rr["exposure_min"] = 0.05
    rr["sev_consistency"]["claims_without_sev"] = "drop"
    r = clean(freq, sev, rr)
    assert (r.freq["ClaimNb"] <= 4).all() and (r.freq["Exposure"] >= 0.05).all()
    assert (r.freq["sev_status"] != "claims_without_sev").all()
    assert r.sev["IDpol"].isin(r.freq["IDpol"]).all()


def test_duplicate_idpol_error():
    with pytest.raises(ValueError):
        clean(pd.concat([freq, freq.head(1)]), sev, rules)


def test_clean_identical_regardless_of_set():
    cfg = make_cfg()
    s = assign_split(freq, cfg)
    full = clean(freq, sev, rules)
    test_ids = set(s.loc[s["primary_set"] == "test", "IDpol"])
    sub_f = freq[freq["IDpol"].isin(test_ids)]
    sub_s = sev[sev["IDpol"].isin(test_ids)]
    alone = clean(sub_f, sub_s, rules)
    restricted = full.freq[full.freq["IDpol"].isin(test_ids)].reset_index(drop=True)
    pd.testing.assert_frame_equal(restricted, alone.freq)
    rs = full.sev[full.sev["IDpol"].isin(test_ids)].reset_index(drop=True)
    pd.testing.assert_frame_equal(rs, alone.sev)


def test_sev_status_categories():
    nb = pd.Series([0, 0, 2, 2, 1, 3])
    rows = pd.Series([0, 1, 0, 2, 2, 1])
    assert sev_status(nb, rows).tolist() == [
        "no_claims",
        "sev_without_claims",
        "claims_without_sev",
        "ok",
        "count_mismatch",
        "count_mismatch",
    ]
