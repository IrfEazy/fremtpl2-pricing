import copy

import pandas as pd
import pytest

from fremtpl2_pricing.clean import clean, sev_status
from fremtpl2_pricing.split import assign_split
from synth import make_cfg, make_freq, make_sev

freq = make_freq()
sev = make_sev(freq)
rules = make_cfg()["clean"]


def test_rules_applied():
    r = clean(freq, sev, rules)
    assert r.freq["Exposure"].max() <= 1.0
    assert r.freq["ClaimNb"].max() <= 4
    assert r.sev["ClaimAmount"].max() <= 1000.0
    assert (r.sev["ClaimAmount_uncapped"] >= r.sev["ClaimAmount"]).all()
    assert r.sev["IDpol"].isin(r.freq["IDpol"]).all()
    assert r.report["sev_without_policy"]["rows"] == 2
    assert r.report["claim_nb"]["policies"] == int((freq["ClaimNb"] > 4).sum())
    assert r.report["exposure_cap"]["policies"] == int((freq["Exposure"] > 1).sum())
    assert len(r.freq) == len(freq)


def test_exclusion_actions_drop_policies_and_their_severity():
    rr = copy.deepcopy(rules)
    rr["claim_nb"]["action"] = "exclude"
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


def _tiny_targets_inputs():
    base = {"Exposure": 1.0}
    f = pd.DataFrame(
        [
            {"IDpol": 1, "ClaimNb": 0, **base},
            {"IDpol": 2, "ClaimNb": 2, **base},  # 2 sev rows
            {"IDpol": 3, "ClaimNb": 1, **base},  # no sev rows
            {"IDpol": 4, "ClaimNb": 6, **base},  # capped to 4, 5 sev rows kept
        ]
    )
    s = pd.DataFrame({"IDpol": [2, 2, 4, 4, 4, 4, 4], "ClaimAmount": [10.0] * 2 + [2000.0] * 5})
    return f, s


def test_targets_defaults_and_options():
    f, s = _tiny_targets_inputs()
    r = clean(f, s, rules).freq.set_index("IDpol")
    assert r["cost"].to_dict() == {1: 0.0, 2: 20.0, 3: 0.0, 4: 5000.0}  # capped at 1000
    assert r["sev_weight"].to_dict() == {1: 0, 2: 2, 3: 0, 4: 5}  # sev rows, not ClaimNb
    assert r["freq_count"].to_dict() == {1: 0, 2: 2, 3: 1, 4: 4}  # ClaimNb after cap
    assert r["in_severity_sample"].to_dict() == {1: False, 2: True, 3: False, 4: True}
    assert r["in_pure_premium_sample"].all()
    rr = copy.deepcopy(rules)
    rr["targets"] = {
        "frequency_count": "sev_rows",
        "severity_weight": "claim_nb",
        "pure_premium_claims_without_sev": "exclude",
    }
    o = clean(f, s, rr).freq.set_index("IDpol")
    assert o["freq_count"].to_dict() == {1: 0, 2: 2, 3: 0, 4: 5}
    assert o["sev_weight"].to_dict() == {1: 0, 2: 2, 3: 0, 4: 4}
    assert o["in_pure_premium_sample"].to_dict() == {1: True, 2: True, 3: False, 4: True}


def test_claim_nb_cap_does_not_touch_severity_rows():
    f, s = _tiny_targets_inputs()
    r = clean(f, s, rules)
    assert (r.sev["IDpol"] == 4).sum() == 5
    rep = r.report["claim_nb"]
    assert rep["sev_rows"] == 5 and rep["sev_rows_kept"] and rep["excess_claims"] == 2
    t = r.report["targets"]
    assert t["claims_without_sev_rows"] == 1 + 0  # policy 3 only; policy 4 has more rows than cap
    assert t["cost_total"] == 5020.0


def test_invalid_target_option_rejected():
    f, s = _tiny_targets_inputs()
    rr = copy.deepcopy(rules)
    rr["targets"]["frequency_count"] = "bogus"
    with pytest.raises(ValueError):
        clean(f, s, rr)
