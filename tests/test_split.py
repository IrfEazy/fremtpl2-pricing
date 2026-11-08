import numpy as np
import pandas as pd
import pytest

from fremtpl2_pricing.clean import clean
from fremtpl2_pricing.split import (
    assign_split,
    attach_severity,
    split_summary,
    train_excess_loading,
    verify_frozen_split,
)
from synth import make_cfg, make_freq, make_sev

freq = make_freq()


def test_deterministic_and_seed_sensitive():
    a = assign_split(freq, make_cfg(seed=7))
    b = assign_split(freq, make_cfg(seed=7))
    c = assign_split(freq, make_cfg(seed=8))
    pd.testing.assert_frame_equal(a, b)
    assert not a["primary_set"].equals(c["primary_set"])
    assert not a["cv_fold"].equals(c["cv_fold"])
    assert not a["random_set"].equals(c["random_set"])


def test_grouped_invariants():
    s = assign_split(freq, make_cfg())
    assert s["IDpol"].is_unique and len(s) == len(freq)
    train, test = s[s["primary_set"] == "train"], s[s["primary_set"] == "test"]
    assert not set(train["IDpol"]) & set(test["IDpol"])
    assert not set(train["group_id"]) & set(test["group_id"])
    assert 0.15 < len(test) / len(s) < 0.25
    # test has no fold, every train policy has exactly one validation fold in 0..4
    assert (test["cv_fold"] == -1).all()
    assert (train["cv_fold"] >= 0).all() and set(train["cv_fold"]) == {0, 1, 2, 3, 4}
    assert train.groupby("IDpol")["cv_fold"].nunique().eq(1).all()
    # no group key shared between CV folds
    assert (train.groupby("group_id")["cv_fold"].nunique() == 1).all()


def test_group_id_matches_factor_tuple():
    s = assign_split(freq, make_cfg())
    cols = [
        "Area",
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "VehBrand",
        "VehGas",
        "Density",
        "Region",
    ]
    d = freq.merge(s, on="IDpol")
    assert (
        d.groupby("group_id", observed=True)
        .apply(lambda g: len(g[cols].drop_duplicates()))
        .eq(1)
        .all()
    )
    assert d["group_id"].nunique() == len(d[cols].drop_duplicates())


def test_random_sensitivity_split_leaks_groups_and_primary_selectable():
    s = assign_split(freq, make_cfg())
    rt, rr = s[s["random_set"] == "train"], s[s["random_set"] == "test"]
    assert abs(len(rr) / len(s) - 0.2) < 0.01
    assert set(rt["group_id"]) & set(rr["group_id"])  # random split shares tuples across sides
    r = assign_split(freq, make_cfg(primary="random"))
    pd.testing.assert_series_equal(r["primary_set"], r["random_set"], check_names=False)
    assert set(r.loc[r["primary_set"] == "train", "cv_fold"]) == {0, 1, 2, 3, 4}


def test_stratification_claim_share_within_tolerance():
    s = assign_split(freq, make_cfg())
    d = freq.merge(s, on="IDpol")
    d["has_claim"] = d["ClaimNb"] > 0
    overall = d["has_claim"].mean()
    tol = 0.01
    for col in ("primary_set", "random_set"):
        for _, g in d.groupby(col):
            assert abs(g["has_claim"].mean() - overall) < tol
    train = d[d["primary_set"] == "train"]
    for _, g in train.groupby("cv_fold"):
        assert abs(g["has_claim"].mean() - train["has_claim"].mean()) < tol
    res = clean(freq, make_sev(freq), make_cfg()["clean"])
    sm = split_summary(res.freq, res.sev, s, make_cfg())
    shares = [sm["primary_sets"][k]["share_with_claim"] for k in ("train", "test")]
    assert all(abs(x - overall) < tol for x in shares)
    assert 0 < sm["primary_sets"]["test"]["largest_group_share_of_policies"] < 0.5


def test_attach_severity_after_split_keeps_policy_rows_together():
    idpol = freq["IDpol"].to_numpy()
    split = assign_split(freq, make_cfg())
    # three severity rows for every 10th policy, one for every 7th, two orphans, shuffled order
    rows = [i for k, i in enumerate(idpol) if k % 10 == 0 for _ in range(3)]
    rows += [i for k, i in enumerate(idpol) if k % 7 == 0]
    rows += [-1, -2, -1]
    sev = pd.DataFrame({"IDpol": rows, "ClaimAmount": np.arange(len(rows), dtype=float)})
    sev = sev.sample(frac=1.0, random_state=3).reset_index(drop=True)
    out = attach_severity(sev, split)
    # orphans excluded, every other row kept exactly once with its original amount
    assert not out["IDpol"].isin([-1, -2]).any()
    assert len(out) == len(sev) - 3
    assert sorted(out["ClaimAmount"]) == sorted(sev.loc[sev["IDpol"] > 0, "ClaimAmount"])
    # independent check: each policy's rows are all on the policy's side and none moved elsewhere
    side = {}
    for i, p, f, r in zip(
        split["IDpol"], split["primary_set"], split["cv_fold"], split["random_set"], strict=True
    ):
        side[int(i)] = (p, int(f), r)
    multi = sev["IDpol"].value_counts()
    assert (multi[multi.index > 0] > 1).any()  # policies with several rows are exercised
    for row in out.itertuples(index=False):
        assert (row.primary_set, row.cv_fold, row.random_set) == side[int(row.IDpol)]
    per_policy = out.groupby("IDpol").agg(
        sides=("primary_set", "nunique"), n=("ClaimAmount", "size")
    )
    assert (per_policy["sides"] == 1).all()
    assert per_policy["n"].to_dict() == {
        k: v for k, v in sev["IDpol"].value_counts().to_dict().items() if k > 0
    }
    # train and test severity policies are disjoint
    tr = set(out.loc[out["primary_set"] == "train", "IDpol"])
    te = set(out.loc[out["primary_set"] == "test", "IDpol"])
    assert tr and te and not tr & te


def test_attach_severity_rejects_duplicate_policy_in_split():
    split = assign_split(freq, make_cfg())
    with pytest.raises(ValueError):
        attach_severity(make_sev(freq), pd.concat([split, split.head(1)]))


def test_summary_counts_and_orphans_excluded():
    cfg = make_cfg()
    res = clean(freq, make_sev(freq), cfg["clean"])  # two orphan rows are removed by clean
    sev = make_sev(freq)  # raw frame: orphans must be excluded by attach_severity
    s = assign_split(freq, cfg)
    sm = split_summary(res.freq, sev, s, cfg)
    assert sm["overlap_primary_train_test"] == {"idpol": 0, "group_id": 0}
    assert sm["groups_spanning_multiple_cv_folds"] == 0
    assert sm["severity_rows_by_primary_set"]["excluded_without_policy"] == 2
    by = sm["severity_rows_by_primary_set"]
    assert by["train"] + by["test"] == len(res.sev)
    assert sm["severity_rows_inherit_policy_set"] is True
    tr, te = sm["primary_sets"]["train"], sm["primary_sets"]["test"]
    assert tr["claims_n"] + te["claims_n"] == int(res.freq["N"].sum())
    assert tr["excess_cost"] + te["excess_cost"] == pytest.approx(res.freq["excess"].sum())
    assert tr["limited_cost"] + te["limited_cost"] == pytest.approx(res.freq["S_cap"].sum())
    assert tr["policies_with_claim_rows"] + te["policies_with_claim_rows"] == int(
        (res.freq["N"] > 0).sum()
    )
    assert set(sm["primary_rating_factor_mix"]["train"]) == set(cfg["rating_factors"])
    assert sm["primary_sets"]["train"]["policies"] + sm["primary_sets"]["test"]["policies"] == len(
        freq
    )
    assert np.isclose(
        sum(f["exposure"] for f in sm["cv_folds"].values()), sm["primary_sets"]["train"]["exposure"]
    )


def test_excess_loading_from_train_set_only():
    cfg = make_cfg()
    res = clean(freq, make_sev(freq), cfg["clean"])
    s = assign_split(res.freq, cfg)
    d = res.freq.merge(s, on="IDpol")
    train = d[d["primary_set"] == "train"]
    base = train_excess_loading(d, cfg["clean"])
    assert base == pytest.approx(train["excess"].sum() / train["Exposure"].sum())
    assert base > 0
    changed = d.copy()
    test_rows = changed["primary_set"] == "test"
    changed.loc[test_rows, "excess"] = 1e9
    changed.loc[test_rows, "Exposure"] = 123.0
    assert train_excess_loading(changed, cfg["clean"]) == base


def test_frozen_split_verification_detects_difference(tmp_path):
    cfg = make_cfg()
    s = assign_split(freq, cfg)
    path = tmp_path / "split.parquet"
    s.to_parquet(path, index=False)
    before = path.read_bytes()
    digest = verify_frozen_split(assign_split(freq, cfg), path)
    assert len(digest) == 64 and path.read_bytes() == before
    other = assign_split(freq, make_cfg(seed=8))
    with pytest.raises(ValueError, match="never overwritten"):
        verify_frozen_split(other, path)
    assert path.read_bytes() == before


def test_split_summary_is_identical_on_first_and_verify_runs(tmp_path, monkeypatch, capsys):
    from fremtpl2_pricing import split as split_mod

    cfg = make_cfg()
    cfg["paths"] = {"processed": str(tmp_path / "proc"), "results": str(tmp_path / "res")}
    monkeypatch.setattr(split_mod, "load_raw", lambda c: (freq, make_sev(freq)))
    summary = tmp_path / "res/split_summary.json"
    split_mod.run_split(cfg)
    first, out1 = summary.read_bytes(), capsys.readouterr().out
    stored = (tmp_path / "proc/split.parquet").read_bytes()
    split_mod.run_split(cfg)
    second, out2 = summary.read_bytes(), capsys.readouterr().out
    assert first == second  # no run-state text in the file
    assert "written (first run)" in out1 and "verified identical" in out2
    assert b"first run" not in first and b"verified" not in first
    assert (tmp_path / "proc/split.parquet").read_bytes() == stored  # never overwritten
