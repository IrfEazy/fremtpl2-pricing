import numpy as np
import pytest
from sklearn.metrics import (
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_tweedie_deviance,
)

from fremtpl2_pricing import metrics as m

rng = np.random.default_rng(0)


def test_gini_toy_hand_computed():
    loss = [1.0, 0.0, 3.0]
    pred = [1.0, 2.0, 3.0]
    g = m.gini(loss, pred, np.ones(3))
    # Lorenz y = 0, .25, .25, 1 at x = 0, 1/3, 2/3, 1: area = 1/3, Gini = 1/3; oracle = 1/2
    assert g["gini"] == pytest.approx(1 / 3)
    assert g["gini_normalized"] == pytest.approx(2 / 3)


def test_gini_ties_reversal_perfect_and_nan():
    loss = np.array([0.0, 1.0, 2.0, 5.0])
    expo = np.array([1.0, 1.0, 2.0, 1.0])
    assert m.gini(loss, np.full(4, 0.3), expo)["gini"] == 0.0
    pred = loss / expo
    g = m.gini(loss, pred, expo)
    assert g["gini_normalized"] == pytest.approx(1.0)
    rev = m.gini(loss, -pred, expo)
    assert rev["gini"] < 0
    # sign flip for a 2-point ordering
    two = m.gini([1.0, 3.0], [1.0, 2.0], [1.0, 1.0])["gini"]
    two_rev = m.gini([1.0, 3.0], [2.0, 1.0], [1.0, 1.0])["gini"]
    assert two == pytest.approx(-two_rev)
    assert np.isnan(m.gini(np.zeros(4), pred, expo)["gini_normalized"])


def test_lorenz_curve_shape_and_weights():
    lc = m.lorenz_curve([1.0, 0.0, 3.0], [1.0, 2.0, 3.0], np.ones(3))
    assert lc.iloc[0].tolist() == [0.0, 0.0]
    assert lc.iloc[-1].tolist() == [1.0, 1.0]
    assert len(lc) == 4
    tied = m.lorenz_curve([1.0, 2.0], [1.0, 1.0], [1.0, 1.0])
    assert len(tied) == 2
    # integer weights equal repeated rows
    g_w = m.gini([1.0, 3.0], [1.0, 2.0], [1.0, 1.0], weight=[2, 1])
    g_r = m.gini([1.0, 1.0, 3.0], [1.0, 1.0, 2.0], [1.0, 1.0, 1.0])
    assert g_w["gini"] == pytest.approx(g_r["gini"])


def _data(n=500):
    mu = rng.uniform(0.2, 3.0, n)
    y = rng.poisson(mu).astype(float)
    w = rng.uniform(0.5, 2.0, n)
    return y, mu, w


def test_deviances_match_sklearn():
    y, mu, w = _data()
    tot = w.sum()
    assert m.poisson_deviance(y, mu, w) == pytest.approx(
        mean_poisson_deviance(y, mu, sample_weight=w) * tot
    )
    ypos = rng.gamma(2.0, 1.0, len(mu))
    assert m.gamma_deviance(ypos, mu, w) == pytest.approx(
        mean_gamma_deviance(ypos, mu, sample_weight=w) * tot
    )
    for p in (1.0, 1.3, 1.7, 2.0):
        yy = ypos if p == 2.0 else y
        assert m.tweedie_deviance(yy, mu, p, w) == pytest.approx(
            mean_tweedie_deviance(yy, mu, sample_weight=w, power=p) * tot
        )
    with pytest.raises(ValueError):
        m.gamma_deviance(y, mu)  # contains zeros
    with pytest.raises(ValueError):
        m.tweedie_deviance(y, mu, 2.5)


def test_poisson_count_vs_rate_exposure_once():
    y, mu_rate, expo = _data()
    counts = y
    # count deviance with mu = rate * exposure equals exposure-weighted rate deviance
    dev_counts = m.poisson_deviance(counts, mu_rate * expo)
    dev_rate = m.tweedie_deviance(counts / expo, mu_rate, 1.0, expo)
    assert dev_counts == pytest.approx(dev_rate)
    assert m.poisson_deviance_per_exposure(counts, mu_rate * expo, expo) == pytest.approx(
        dev_counts / expo.sum()
    )
    # bootstrap multiplicities weight numerator and denominator
    mult = rng.integers(0, 3, len(y)).astype(float)
    assert m.poisson_deviance_per_exposure(counts, mu_rate * expo, expo, mult) == pytest.approx(
        m.poisson_deviance(counts, mu_rate * expo, mult) / (mult * expo).sum()
    )


def test_null_deviance_with_supplied_mean():
    y, _, w = _data()
    expo = w
    null_rate = 0.7
    nd = m.null_deviance(m.poisson_deviance, y, null_rate, scale=expo)
    assert nd == pytest.approx(m.poisson_deviance(y, null_rate * expo))
    # the supplied mean is used, not the mean of y
    nd2 = m.null_deviance(m.poisson_deviance, y, y.sum() / expo.sum(), scale=expo)
    assert nd2 != pytest.approx(nd)
    nt = m.null_deviance(m.tweedie_deviance, y / expo, null_rate, weight=expo, power=1.5)
    assert nt == pytest.approx(m.tweedie_deviance(y / expo, np.full(len(y), null_rate), 1.5, expo))
    assert m.deviance_explained(60.0, 80.0) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        m.null_deviance(m.poisson_deviance, y, 0.0)


def test_lift_table_bins_and_reconciliation():
    n = 1000
    expo = rng.uniform(0.1, 1.0, n)
    pred = rng.uniform(0.05, 0.5, n) * expo
    obs = rng.poisson(pred).astype(float)
    lt = m.lift_table(obs, pred, expo, n_bins=10)
    assert len(lt) == 10
    share = lt["exposure"] / expo.sum()
    assert share.between(0.08, 0.12).all()
    tot = m.calibration_totals(obs, pred)
    assert lt["observed"].sum() == pytest.approx(tot["observed"])
    assert lt["predicted"].sum() == pytest.approx(tot["predicted"])
    assert lt["observed"].sum() / lt["predicted"].sum() == pytest.approx(tot["ratio"])
    assert lt["predicted_rate"].is_monotonic_increasing


def test_double_lift_reconciles():
    n = 800
    expo = rng.uniform(0.1, 1.0, n)
    a = rng.uniform(0.05, 0.5, n) * expo
    b = rng.uniform(0.05, 0.5, n) * expo
    obs = rng.poisson(a).astype(float)
    dl = m.double_lift_table(obs, a, b, expo, n_bins=5)
    assert len(dl) == 5
    assert dl["observed"].sum() == pytest.approx(obs.sum())
    assert dl["pred_a"].sum() == pytest.approx(a.sum())
    assert dl["pred_b"].sum() == pytest.approx(b.sum())
    assert (dl["ratio_a"] == dl["observed"] / dl["pred_a"]).all()
    assert (dl["exposure"] / expo.sum()).between(0.15, 0.25).all()


def test_calibration_by_level():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([2.0, 2.0, 2.0, 2.0])
    cal = m.calibration_by_level(obs, pred, ["a", "a", "b", "b"])
    assert cal.loc["a", "observed"] == 3.0
    assert cal.loc["b", "ratio"] == pytest.approx(7 / 4)
    assert cal.loc["a", "weight"] == 2.0
    assert cal["observed"].sum() == m.calibration_totals(obs, pred)["observed"]


def _fn_factory(y, mu_a, mu_b):
    def fn(w):
        a = m.poisson_deviance(y, mu_a, w)
        b = m.poisson_deviance(y, mu_b, w)
        return {"a": a, "b": b, "diff": a - b}

    return fn


def test_bootstrap_deterministic_paired_and_units():
    n = 200
    y, mu_a, _ = _data(n)
    mu_b = mu_a * 1.1
    fn = _fn_factory(y, mu_a, mu_b)
    units = np.arange(n)
    r1 = m.paired_bootstrap(fn, n, units, 50, seed=3)
    r2 = m.paired_bootstrap(fn, n, units, 50, seed=3)
    r3 = m.paired_bootstrap(fn, n, units, 50, seed=4)
    assert r1["replicates"].equals(r2["replicates"])
    assert not r1["replicates"].equals(r3["replicates"])
    reps = r1["replicates"]
    lo, hi = m.percentile_ci(reps["a"] - reps["b"], 0.95)
    assert r1["ci"].loc["diff", "lower"] == pytest.approx(lo)
    assert r1["ci"].loc["diff", "upper"] == pytest.approx(hi)
    assert r1["point"]["diff"] == pytest.approx(fn(np.ones(n))["diff"])


def test_bootstrap_weights_policy_and_group_units():
    n = 60
    seen = []

    def fn(w):
        seen.append(w.copy())
        return {"s": float(w.sum())}

    m.paired_bootstrap(fn, n, np.arange(n), 20, seed=1)
    assert all(w.sum() == n for w in seen[:-1])  # last call is the point estimate
    seen.clear()
    groups = np.repeat(np.arange(12), 5)
    m.paired_bootstrap(fn, n, groups, 20, seed=1)
    for w in seen[:-1]:
        by_group = w.reshape(12, 5)
        assert (by_group == by_group[:, [0]]).all()  # rows of a group stay together
        assert by_group[:, 0].sum() == 12
        assert np.all(w == np.round(w))
    with pytest.raises(ValueError):
        m.paired_bootstrap(fn, n, groups[:-1], 5, seed=1)
