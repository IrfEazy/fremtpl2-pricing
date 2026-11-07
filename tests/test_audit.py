import copy
import json

import pandas as pd
import pytest

from fremtpl2_pricing.audit import compute_audit, render_markdown
from synth import make_cfg

CFG = make_cfg()


def tiny():
    base = dict(
        Area="A",
        VehPower=5,
        VehAge=1,
        DrivAge=30,
        BonusMalus=50,
        VehBrand="B1",
        VehGas="Diesel",
        Density=10,
        Region="R1",
    )
    rows = [
        # id, claimnb, exposure, overrides
        (1, 0, 0.5, {}),
        (2, 1, 1.5, {}),  # same tuple as 1 -> group of 3 with id 3
        (3, 0, 0.02, {}),
        (4, 2, 0.1, {"Area": "B"}),
        (5, 1, 1.0, {"VehPower": 6}),  # claim, no sev
        (6, 6, 0.1, {"VehPower": 7}),  # extreme ClaimNb
        (7, 0, 0.9, {"VehPower": 8}),  # sev rows but ClaimNb = 0
    ]
    freq = pd.DataFrame(
        [{"IDpol": i, "ClaimNb": nb, "Exposure": e, **{**base, **o}} for i, nb, e, o in rows]
    )
    sev = pd.DataFrame(
        {
            "IDpol": [2, 4, 4, 6, 7, 99],  # policy 4: 2 rows ok; 6: 1 row vs ClaimNb 6; 99 orphan
            "ClaimAmount": [100.0, 200.0, 3000.0, 400.0, 500.0, 600.0],
        }
    )
    return freq, sev


def test_counts_on_hand_built_frame():
    freq, sev = tiny()
    au = compute_audit(freq, sev, CFG)
    assert au["exposure"]["gt1"]["rows"] == 1 and au["exposure"]["gt1"]["claims"] == 1
    assert au["exposure"]["small"][0]["rows"] == 1  # only exposure 0.02 <= 0.05
    assert au["exposure"]["small"][1]["rows"] == 3  # 0.02, 0.1, 0.1
    assert au["claim_nb"]["extreme_values"] == [6] and au["claim_nb"]["extreme_claims"] == 6
    lk = au["linkage"]
    assert lk["sev_without_freq"]["rows"] == 1 and lk["sev_without_freq"]["amount"] == 600.0
    st = lk["status"]
    assert st["claims_without_sev"]["policies"] == 1 and st["claims_without_sev"]["claims"] == 1
    assert st["sev_without_claims"]["policies"] == 1
    assert (
        st["count_mismatch"]["policies"] == 1
        and st["count_mismatch"]["sev_fewer_than_claimnb"] == 1
    )
    assert st["ok"]["policies"] == 2  # policy 2 (ClaimNb 1, 1 row) and policy 4 (2 and 2)
    assert lk["duplicate_idpol_freq"] == 0
    sv = au["severity"]
    assert sv["rows"] == 6 and sv["total"] == 4800.0
    assert sv["top"][0]["amount"] == 3000.0 and sv["top"][0]["share"] == 3000.0 / 4800.0
    assert sv["cap_candidates"][0]["claims_above"] == 1
    lg = au["leakage"]
    assert lg["distinct_groups"] == 5 and lg["largest_group"] == 3
    assert lg["policies_in_multi_groups"] == 3
    assert lg["claims_in_multi_groups"] == 1
    assert au["cleaning"]["before"]["policies"] == 7


def test_cleaning_totals_and_render():
    freq, sev = tiny()
    au = compute_audit(freq, sev, CFG)
    b, a = au["cleaning"]["before"], au["cleaning"]["after"]
    assert b["claims"] == 10 and a["claims"] == 8  # ClaimNb 6 capped to 4
    assert b["sev_rows"] == 6 and a["sev_rows"] == 5  # orphan dropped
    assert abs(b["exposure"] - a["exposure"] - 0.5) < 1e-9  # exposure 1.5 capped at 1
    md = render_markdown(au)
    assert "for human review" in md and "pre-registered" in md


def test_targets_report_on_hand_built_frame():
    freq, sev = tiny()
    tg = compute_audit(freq, sev, CFG)["cleaning"]["impact"]["targets"]
    # policies with sev rows after cleaning: 2, 4, 6, 7; claims without sev rows: policy 5 (1)
    # and the shortfall of policy 6 (ClaimNb capped to 4, 1 row -> 3)
    assert tg["severity_sample_policies"] == 4 and tg["severity_sample_sev_rows"] == 5
    assert tg["claims_without_sev_rows"] == 1 + 3
    assert tg["cost_total"] == 100.0 + 200.0 + 1000.0 + 400.0 + 500.0  # 3000 capped at 1000
    assert tg["severity_weight_total"] == 5
    assert tg["frequency_count_total"] == tg["claims_total"] == 8
    nb = compute_audit(freq, sev, CFG)["cleaning"]["impact"]["claim_nb"]
    assert nb["sev_rows"] == 1 and nb["sev_amount"] == 400.0 and nb["sev_rows_kept"]


def test_markdown_is_rendered_from_json():
    freq, sev = tiny()
    au = compute_audit(freq, sev, CFG)
    # round trip through JSON, as results/audit.json is stored
    au = json.loads(json.dumps(au))
    md = render_markdown(au)
    assert render_markdown(au) == md
    changed = copy.deepcopy(au)
    changed["dataset"]["freq_rows"] = 123457
    changed["exposure"]["gt1"]["rows"] = 987654
    changed["leakage"]["distinct_groups"] = 424242
    changed["cleaning"]["impact"]["targets"]["claims_without_sev_rows"] = 777777
    md2 = render_markdown(changed)
    for number in ("123,457", "987,654", "424,242", "777,777"):
        assert number in md2 and number not in md
    changed["linkage"]["status"]["claims_without_sev"]["policies"] = 1
    assert "1 policy" in render_markdown(changed) and "1 policies" not in render_markdown(changed)


@pytest.mark.parametrize(
    ("cap", "phrase"),
    [
        (10.0, "below the lowest tabulated quantile (50%: 200"),
        (250.0, "between the 50% quantile (200) and the 99% quantile (2,970)"),
        (5000.0, "at or above the highest tabulated quantile (99%: 2,970"),
    ],
)
def test_cap_position_is_computed_from_quantiles(cap, phrase):
    freq, sev = tiny()
    au = compute_audit(freq, sev, CFG)
    au["severity"]["quantiles"] = {"0.5": 200.0, "0.99": 2970.0}
    au["cleaning"]["impact"]["large_loss"]["cap"] = cap
    assert phrase in render_markdown(au)


def test_missing_quantile_is_not_hidden():
    freq, sev = tiny()
    au = compute_audit(freq, sev, CFG)
    au["severity"]["quantiles"] = {}
    with pytest.raises((KeyError, IndexError)):
        render_markdown(au)


def test_no_unbacked_qualitative_claims_in_markdown():
    freq, sev = tiny()
    md = render_markdown(compute_audit(freq, sev, CFG))
    for phrase in ("roughly flat", "most likely data errors", "no group dominates", "implausible"):
        assert phrase not in md


def test_target_text_follows_configured_targets():
    freq, sev = tiny()
    au = json.loads(json.dumps(compute_audit(freq, sev, CFG)))
    md = render_markdown(au)
    assert "`frequency_count: claim_nb`, configured" in md
    assert "`frequency_count: sev_rows`, alternative" in md
    assert "treats them as zero cost" in md
    changed = copy.deepcopy(au)
    tg = changed["cleaning"]["impact"]["targets"]
    tg["frequency_count"] = "sev_rows"
    tg["pure_premium_claims_without_sev"] = "exclude"
    md2 = render_markdown(changed)
    assert "`frequency_count: sev_rows`, configured" in md2
    assert "`frequency_count: claim_nb`, alternative" in md2
    assert "treats them as zero cost" not in md2
    assert "drops those policies from the pure-premium sample" in md2
