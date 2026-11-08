import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fremtpl2_pricing import features
from fremtpl2_pricing.clean import clean
from synth import make_cfg, make_freq, make_sev

FEAT = yaml.safe_load((Path(__file__).parents[1] / "configs" / "default.yaml").read_text())[
    "features"
]


def cfg_with(min_exposure=1500):
    c = copy.deepcopy(FEAT)
    c["rare_levels"]["min_exposure_years"] = min_exposure
    return c


def row(**kw):
    base = dict(
        Exposure=1.0, Area="A", VehPower=6, VehAge=2, DrivAge=30, BonusMalus=50,
        VehBrand="B1", VehGas="Diesel", Density=100, Region="R1",
    )  # fmt: skip
    base.update(kw)
    return base


def frame(rows):
    return pd.DataFrame(rows)


def cleaned():
    rules = make_cfg()["clean"]
    return clean(make_freq(), make_sev(make_freq()), rules).freq


def test_config_section_complete():
    assert FEAT["definitions"]["DrivAge"]["lower_edges"] == [18, 21, 26, 31, 41, 51, 61, 71, 81]
    assert FEAT["definitions"]["VehAge"]["lower_edges"] == [0, 1, 3, 6, 11, 16]
    assert FEAT["rare_levels"]["min_exposure_years"] == 1500
    assert set(FEAT["feature_sets"]) == {"full", "bonus_malus_only", "severity_simplified"}


def test_band_edges_at_boundaries():
    df = frame(
        [
            row(DrivAge=a, VehAge=v)
            for a, v in [(18, 0), (20, 1), (21, 2), (25, 3), (80, 10), (81, 11), (99, 100)]
        ]
    )
    out = features.transform(df, features.fit(df, cfg_with(0)))
    assert list(out["DrivAge"].astype(str)) == [
        "18-20",
        "18-20",
        "21-25",
        "21-25",
        "71-80",
        "81+",
        "81+",
    ]
    assert list(out["VehAge"].astype(str)) == ["0-0", "1-2", "1-2", "3-5", "6-10", "11-15", "16+"]
    # only bands with fitting exposure are categories; the reference (here a tie: first) leads
    assert out["DrivAge"].cat.categories.tolist() == ["18-20", "21-25", "71-80", "81+"]


def test_bonus_malus_cap_log_density_and_vehpower_cap():
    df = frame(
        [
            row(BonusMalus=b, Density=d, VehPower=p)
            for b, d, p in [(50, 1, 4), (150, 100, 12), (230, 27000, 15)]
        ]
    )
    out = features.transform(df, features.fit(df, cfg_with(0)))
    assert out["BonusMalus"].tolist() == [50, 150, 150]
    np.testing.assert_allclose(out["Density"], np.log([1, 100, 27000]))
    assert list(out["VehPower"].astype(str)) == ["4", "12", "12"]
    assert out["VehPower"].cat.categories.tolist() == ["12", "4"]  # reference (most exposure) first


def test_nan_bonus_malus_and_density_raise():
    fmap = features.fit(frame([row()]), cfg_with(0))
    for col in ("BonusMalus", "Density"):
        with pytest.raises(ValueError):
            features.transform(frame([row(**{col: np.nan})]), fmap)


def test_nonpositive_density_raises():
    df = frame([row(), row(Density=0)])
    fmap = features.fit(frame([row()]), cfg_with(0))
    with pytest.raises(ValueError):
        features.transform(df, fmap)


def train_valid():
    train = frame(
        [row(VehBrand="B1", Exposure=1.0)] * 10
        + [row(VehBrand="B2", Exposure=1.0)] * 3
        + [row(VehBrand="B3", Exposure=0.5)] * 2
        + [row(VehBrand="B1", Region="R2", Exposure=0.2)] * 2
    )
    valid = frame([row(VehBrand="B9"), row(VehBrand="B3"), row(Region="R7")])
    return train, valid


def test_rare_levels_from_fit_only_and_unseen_maps_to_other():
    train, valid = train_valid()
    cfg = cfg_with(2)
    fmap = features.fit(train, cfg)
    info = fmap.levels["VehBrand"]
    assert info["rare"] == ["B3"] and info["levels"] == ["B1", "B2", "OTHER"]
    assert info["reference"] == "B1"
    assert fmap.levels["Region"]["rare"] == ["R2"]
    out = features.transform(valid, fmap)
    assert list(out["VehBrand"].astype(str)) == ["OTHER", "OTHER", "B1"]
    assert list(out["Region"].astype(str)) == ["R1", "R1", "OTHER"]
    assert not out.isna().any().any()


def test_validation_rows_do_not_change_fit_or_train_output():
    train, valid = train_valid()
    cfg = cfg_with(2)
    fmap = features.fit(train, cfg)
    before = features.transform(train, fmap)
    # "fitting" is only on train: altering or adding validation rows cannot enter the map
    big_valid = frame([row(VehBrand="B3", Exposure=1.0)] * 50 + valid.to_dict("records"))
    fmap2 = features.fit(train, cfg)
    assert fmap2.to_dict() == fmap.to_dict()
    pd.testing.assert_frame_equal(before, features.transform(train, fmap2))
    # fitting on train plus valid would change it: the guard is that fit is given train only
    assert "B3" not in features.fit(pd.concat([train, big_valid]), cfg).levels["VehBrand"]["rare"]


def test_output_columns_no_leakage_and_dtypes():
    df = cleaned()
    fmap = features.fit(df, cfg_with(0))
    out = features.transform(df, fmap)
    assert list(out.columns) == features.feature_names(FEAT, "full")
    for c in ["Area", "VehPower", "VehAge", "DrivAge", "VehBrand", "VehGas", "Region"]:
        assert isinstance(out[c].dtype, pd.CategoricalDtype)
    assert out["BonusMalus"].dtype == float and out["Density"].dtype == float
    assert not out.isna().any().any()
    assert not {"Exposure", "IDpol", "N", "S", "S_cap", "ClaimNb"} & set(out.columns)


def test_json_roundtrip_identical_transform():
    train, valid = train_valid()
    fmap = features.fit(train, cfg_with(2))
    back = features.FeatureMap.from_json(fmap.to_json())
    assert back.to_dict() == fmap.to_dict()
    pd.testing.assert_frame_equal(features.transform(valid, fmap), features.transform(valid, back))
    pd.testing.assert_frame_equal(features.transform(train, fmap), features.transform(train, back))


def test_feature_sets():
    assert features.feature_names(FEAT, "bonus_malus_only") == ["BonusMalus"]
    assert len(features.feature_names(FEAT, "full")) == 9
    with pytest.raises(KeyError):
        features.feature_names(FEAT, "nope")


def test_row_output_independent_of_other_rows():
    train, valid = train_valid()
    fmap = features.fit(train, cfg_with(2))
    both = pd.concat([train, valid], ignore_index=True)
    full = features.transform(both, fmap)
    alone = features.transform(valid, fmap)
    pd.testing.assert_frame_equal(
        full.iloc[len(train) :].reset_index(drop=True), alone.reset_index(drop=True)
    )
    one = features.transform(both.iloc[[3]], fmap)
    pd.testing.assert_frame_equal(one, full.iloc[[3]])


def ref_not_first_frame():
    rows = []
    # reference is never the natural first level: B2/R2/B/Petrol/7/band 31-40/VehAge 3-5
    for _ in range(5):
        rows.append(row(VehBrand="B2", Region="R2", Area="B", VehGas="Petrol", VehPower=7,
                        DrivAge=35, VehAge=4))  # fmt: skip
    rows.append(row(VehBrand="B1", Region="R1", Area="A", VehGas="Diesel", VehPower=6,
                    DrivAge=19, VehAge=0))  # fmt: skip
    rows.append(row(VehBrand="B3", Region="R3", Area="C", VehGas="Diesel", VehPower=9,
                    DrivAge=45, VehAge=8))  # fmt: skip
    return frame(rows)


def test_reference_is_first_category_everywhere():
    df = ref_not_first_frame()
    fmap = features.fit(df, cfg_with(2))
    out = features.transform(df, fmap)
    expected = {"Area": "B", "VehGas": "Petrol", "VehPower": "7", "DrivAge": "31-40",
                "VehAge": "3-5", "VehBrand": "B2", "Region": "R2"}  # fmt: skip
    for name, ref in expected.items():
        info = fmap.levels[name]
        assert info["reference"] == ref and info["levels"][0] == ref
        assert out[name].cat.categories[0] == ref
        assert out[name].cat.categories.tolist() == info["levels"]
    assert fmap.levels["DrivAge"]["levels"] == ["31-40", "18-20", "41-50"]
    assert fmap.levels["VehBrand"]["levels"] == ["B2", "OTHER"]  # B1, B3 rare, OTHER last


SEEN = dict(Area="B", VehGas="Petrol", VehPower=7, DrivAge=35, VehAge=4, VehBrand="B2", Region="R2")


def test_unseen_non_grouped_level_raises():
    fmap = features.fit(ref_not_first_frame(), cfg_with(2))
    for kw, name in [({"Area": "F"}, "Area"), ({"VehGas": "Electric"}, "VehGas"),
                     ({"VehPower": 3}, "VehPower")]:  # fmt: skip
        with pytest.raises(ValueError, match=name):
            features.transform(frame([row(**{**SEEN, **kw})]), fmap)
    with pytest.raises(ValueError, match="3"):
        features.transform(frame([row(**{**SEEN, "VehPower": 3})]), fmap)


def test_empty_levels_are_dropped():
    # no rare level: no OTHER; an unseen grouped level then raises
    df = frame([row(VehBrand="B1")] * 3 + [row(VehBrand="B2")] * 3)
    fmap = features.fit(df, cfg_with(1))
    assert fmap.levels["VehBrand"]["levels"] == ["B1", "B2"]
    assert fmap.levels["VehBrand"]["rare"] == []
    with pytest.raises(ValueError, match="VehBrand"):
        features.transform(frame([row(VehBrand="B9")]), fmap)
    # a band without fitting exposure is dropped and a value in it raises
    assert "81+" not in fmap.levels["DrivAge"]["levels"]
    with pytest.raises(ValueError, match="DrivAge"):
        features.transform(frame([row(DrivAge=90)]), fmap)
    # with a rare level OTHER exists and absorbs unseen grouped levels
    train, _ = train_valid()
    fm2 = features.fit(train, cfg_with(2))
    assert "OTHER" in fm2.levels["VehBrand"]["levels"]
    out = features.transform(frame([row(VehBrand="B9")]), fm2)
    assert out["VehBrand"].astype(str).tolist() == ["OTHER"]


def test_weight_column_drives_rare_levels_and_reference():
    df = frame(
        [row(VehBrand="B1", Exposure=5.0, N=1)] * 4
        + [row(VehBrand="B2", Exposure=0.1, N=5)] * 4
        + [row(VehBrand="B3", Exposure=0.1, N=1)] * 4
    )
    by_exposure = features.fit(df, cfg_with(1)).levels["VehBrand"]
    assert by_exposure["reference"] == "B1" and by_exposure["rare"] == ["B2", "B3"]
    by_claims = features.fit(df, cfg_with(1), weight_column="N", min_weight=10).levels["VehBrand"]
    assert by_claims["reference"] == "B2" and by_claims["rare"] == ["B1", "B3"]


# ------------------------------------------------------------------ severity map (policy-derived)


def sev_map(policies, min_claims=3, min_exposure=2, ordinal=("DrivAge", "VehAge", "VehPower")):
    cfg = cfg_with(min_exposure)
    pmap = features.fit(policies, cfg)
    claims = policies[policies["N"] > 0]
    return pmap, features.fit_severity(claims, cfg, pmap, min_claims, list(ordinal))


def brand_frame(counts, zero_claim=()):
    rows = [row(VehBrand=b, N=1) for b, n in counts.items() for _ in range(n)]
    rows += [row(VehBrand=b, N=0) for b in zero_claim for _ in range(3)]
    return frame(rows)


def test_severity_level_with_zero_claims_maps_to_reference():
    pol = brand_frame({"B1": 10, "B2": 5}, zero_claim=["B3"])
    pmap, smap = sev_map(pol)
    assert "B3" in pmap.levels["VehBrand"]["levels"]  # in the policy schema, no claims
    info = smap.levels["VehBrand"]
    assert info["merge"]["B3"] == "B1" and info["levels"] == ["B1", "B2"]
    out = features.transform(frame([row(VehBrand="B3")]), smap)
    assert out["VehBrand"].astype(str).tolist() == ["B1"]


def test_severity_nominal_merge_into_other_or_reference():
    # small levels together below the threshold -> reference
    pol = brand_frame({"B1": 10, "B2": 4, "B3": 1, "B4": 1})
    _, smap = sev_map(pol, min_claims=3, min_exposure=1)
    info = smap.levels["VehBrand"]
    assert info["merge"] == {"B1": "B1", "B2": "B2", "B3": "B1", "B4": "B1"}
    assert "OTHER" not in info["levels"]
    # together at least the threshold -> OTHER (new level, last)
    pol = brand_frame({"B1": 10, "B2": 4, "B3": 2, "B4": 2})
    _, smap = sev_map(pol, min_claims=3, min_exposure=1)
    info = smap.levels["VehBrand"]
    assert info["merge"]["B3"] == info["merge"]["B4"] == "OTHER"
    assert info["levels"] == ["B1", "B2", "OTHER"] and info["reference"] == "B1"


def test_severity_policy_other_level_below_threshold_joins_reference():
    # B3 is rare for the policy map (exposure < 2) and becomes the policy OTHER level
    pol = brand_frame({"B1": 10, "B2": 4})
    pol = pd.concat([pol, frame([row(VehBrand="B3", Exposure=0.5, N=1)])], ignore_index=True)
    pmap, smap = sev_map(pol, min_claims=3, min_exposure=2)
    assert pmap.levels["VehBrand"]["levels"] == ["B1", "B2", "OTHER"]
    info = smap.levels["VehBrand"]
    assert info["merge"] == {"B1": "B1", "B2": "B2", "OTHER": "B1"}
    out = features.transform(frame([row(VehBrand="B3"), row(VehBrand="B9")]), smap)
    assert out["VehBrand"].astype(str).tolist() == ["B1", "B1"]  # unseen -> OTHER -> reference


def test_severity_ordinal_merges_into_adjacent_level_with_more_claims():
    ages = {20: 1, 23: 10, 28: 2, 35: 8}  # bands 18-20, 21-25, 26-30, 31-40
    pol = frame([row(DrivAge=a, N=1) for a, n in ages.items() for _ in range(n)])
    _, smap = sev_map(pol, min_claims=3, min_exposure=1)
    m = smap.levels["DrivAge"]["merge"]
    # smallest first: 18-20 -> only neighbour; then 26-30 -> the neighbour with more claims
    assert m == {"18-20": "21-25", "21-25": "21-25", "26-30": "21-25", "31-40": "31-40"}
    assert smap.levels["DrivAge"]["reference"] == "21-25"


def test_severity_ordinal_tie_goes_toward_the_reference():
    ages = {20: 20, 23: 6, 28: 1, 35: 6}  # 26-30 sits between two equal neighbours
    pol = frame([row(DrivAge=a, N=1) for a, n in ages.items() for _ in range(n)])
    _, smap = sev_map(pol, min_claims=3, min_exposure=1)
    m = smap.levels["DrivAge"]["merge"]
    assert smap.levels["DrivAge"]["reference"] == "18-20"
    assert m["26-30"] == "21-25" and m["31-40"] == "31-40"  # toward the reference side


def test_severity_value_outside_policy_schema_still_raises():
    pol = brand_frame({"B1": 10, "B2": 5})
    _, smap = sev_map(pol)
    with pytest.raises(ValueError, match="VehGas"):
        features.transform(frame([row(VehBrand="B1", VehGas="Electric")]), smap)
    with pytest.raises(ValueError, match="DrivAge"):
        features.transform(frame([row(DrivAge=90)]), smap)


def test_severity_map_json_roundtrip_and_every_schema_level_predicts_finitely():
    from fremtpl2_pricing import glm

    pol = cleaned()
    cfg = cfg_with(20)
    pmap = features.fit(pol, cfg)
    claims = pol[pol["N"] > 0]
    smap = features.fit_severity(claims, cfg, pmap, 40, ["DrivAge", "VehAge", "VehPower"])
    assert any(
        len(smap.levels[f]["levels"]) < len(pmap.levels[f]["levels"]) for f in smap.levels
    )  # the threshold actually merges something
    assert features.FeatureMap.from_json(smap.to_json()).levels == smap.levels
    cols = features.feature_names(cfg, "full")
    Xc = features.transform(claims, smap)[cols]
    avg = (claims["S_cap"] / claims["N"]).to_numpy()
    model = glm.fit_severity(Xc, avg, claims["N"].to_numpy())  # no empty column
    # every level of the policy schema gets a finite severity prediction
    grid = []
    for f in pmap.levels:
        for lv in pmap.levels[f]["levels"]:
            if lv != "OTHER" or lv in pmap.levels[f]["levels"]:
                grid.append((f, lv))
    base = pol.iloc[0:1]
    for f, lv in grid:
        r = base.copy()
        raw = pmap.levels[f]["rare"][0] if lv == "OTHER" else lv
        spec = cfg["definitions"][f]
        if spec["kind"] == "band":
            r[f] = int(lv.split("-")[0].rstrip("+"))
        else:
            r[f] = type(base[f].iloc[0])(raw) if spec.get("cap") is None else int(raw)
        pred = model.predict_rate(features.transform(r, smap)[cols])
        assert np.isfinite(pred).all() and (pred > 0).all(), (f, lv)


def test_severity_ordinal_feature_with_group_rare_is_rejected_at_fit():
    pol = brand_frame({"B1": 10, "B2": 4})
    pol = pd.concat([pol, frame([row(VehBrand="B3", Exposure=0.5, N=1)])], ignore_index=True)
    with pytest.raises(ValueError, match="VehBrand.*ordinal.*group_rare"):
        sev_map(pol, min_claims=3, min_exposure=2, ordinal=("VehBrand", "DrivAge"))
    # without a rare level grouped there is no OTHER level and the configuration fits
    ok = brand_frame({"B1": 10, "B2": 4})
    sev_map(ok, min_claims=3, min_exposure=2, ordinal=("VehBrand",))
