"""Offline tests for the GLM layer (synthetic data, statsmodels cross-check)."""

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from fremtpl2_pricing import glm
from synth import model_cfg

# the tight library default and the solver settings of the pipeline (configs/default.yaml)
SOLVERS = pytest.mark.parametrize(
    "solver", [None, model_cfg()["models"]["glm"]["solver"]], ids=["default", "pipeline"]
)


def make_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "area": pd.Categorical(rng.choice(["A", "B", "C"], n), categories=["A", "B", "C"]),
            "age": rng.uniform(-1, 1, n),
        }
    )
    expo = rng.uniform(0.2, 1.0, n)
    eta = -1.0 + 0.4 * (X["area"] == "B") - 0.3 * (X["area"] == "C") + 0.5 * X["age"]
    count = rng.poisson(np.exp(eta) * expo).astype(float)
    ncl = rng.integers(1, 5, n).astype(float)
    cost = rng.gamma(4.0, np.exp(6 + 0.3 * X["age"] + 0.2 * (X["area"] == "B")) / 4.0)
    pp = rng.gamma(2.0, np.exp(eta + 5) / 2.0) * (rng.uniform(size=n) < 0.7)
    return X, count, expo, ncl, cost, pp


def sm_design(X):
    D = pd.get_dummies(X["area"], drop_first=True, dtype=float)
    D["age"] = X["age"]
    return sm.add_constant(D)


def params(m):
    return np.r_[m.intercept, m.coef]


@SOLVERS
def test_poisson_matches_statsmodels(solver):
    X, count, expo, *_ = make_data()
    m = glm.fit_frequency(X, count, expo, solver)
    D = sm_design(X)
    ref = sm.GLM(count, D, family=sm.families.Poisson(), offset=np.log(expo)).fit()
    np.testing.assert_allclose(params(m), ref.params.to_numpy(), rtol=1e-5, atol=1e-6)
    expected = ref.predict(D, offset=np.log(expo))
    np.testing.assert_allclose(m.predict_count(X, expo), expected, rtol=1e-5)


@SOLVERS
def test_gamma_matches_statsmodels(solver):
    X, _, _, ncl, cost, _ = make_data()
    m = glm.fit_severity(X, cost, ncl, solver)
    fam = sm.families.Gamma(sm.families.links.Log())
    ref = sm.GLM(cost, sm_design(X), family=fam, var_weights=ncl).fit()
    np.testing.assert_allclose(params(m), ref.params.to_numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(m.predict_rate(X), ref.predict(sm_design(X)), rtol=1e-5)


@SOLVERS
def test_tweedie_matches_statsmodels(solver):
    X, _, expo, _, _, pp = make_data()
    m = glm.fit_tweedie(X, pp, expo, 1.5, solver)
    fam = sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log())
    ref = sm.GLM(pp, sm_design(X), family=fam, var_weights=expo).fit()
    np.testing.assert_allclose(params(m), ref.params.to_numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(m.predict_rate(X), ref.predict(sm_design(X)), rtol=1e-4)


def test_doubling_exposure():
    X, count, expo, *_ = make_data()
    m = glm.fit_frequency(X, count, expo)
    np.testing.assert_allclose(m.predict_count(X, 2 * expo), 2 * m.predict_count(X, expo))
    np.testing.assert_allclose(m.predict_rate(X), m.predict_count(X, np.ones(len(X))))


def test_exposure_applied_once():
    X, count, expo, *_ = make_data()
    a = glm.fit_frequency(X, count, expo)
    D, _ = glm._design(X, a.features)
    est = glm._estimator("poisson", None, True)
    est.fit(D, count / expo, sample_weight=expo)
    np.testing.assert_allclose(params(a), np.r_[est.intercept_, est.coef_], rtol=1e-5, atol=1e-6)
    # guard: exposure used as both offset and weight is a different (wrong) model
    bad = glm._estimator("poisson", None, True)
    bad.fit(D, count, sample_weight=expo, offset=np.log(expo))
    assert np.abs(np.r_[bad.intercept_, bad.coef_] - params(a)).max() > 1e-3


def test_intercept_only_weighted_means():
    X, count, expo, ncl, cost, pp = make_data()
    E = X.iloc[:, :0]
    f = glm.fit_frequency(E, count, expo)
    np.testing.assert_allclose(f.predict_rate(E), count.sum() / expo.sum(), rtol=1e-6)
    s = glm.fit_severity(E, cost, ncl)
    np.testing.assert_allclose(s.predict_rate(E), (cost * ncl).sum() / ncl.sum(), rtol=1e-6)
    t = glm.fit_tweedie(E, pp, expo, 1.5)
    np.testing.assert_allclose(t.predict_rate(E), (pp * expo).sum() / expo.sum(), rtol=1e-5)


@pytest.mark.parametrize("robust", [True, False])
def test_coefficient_table(robust):
    X, count, expo, *_ = make_data()
    m = glm.fit_frequency(X, count, expo)
    tab = glm.coefficient_table(m, X, count, np.ones(len(X)), np.log(expo), robust=robust)
    ref_row = tab[(tab.feature == "area") & (tab.level == "A")].iloc[0]
    assert ref_row.relativity == 1 and np.isnan(ref_row.se)
    est = tab[tab.term != "area[A]"]
    assert np.isfinite(est.se).all() and (est.se > 0).all()
    assert set(tab.term) == {glm.INTERCEPT, "area[A]", "area[B]", "area[C]", "age"}
    assert (tab.covariance == ("robust" if robust else "model-based")).all()
    assert (est.rel_lo < est.relativity).all() and (est.relativity < est.rel_hi).all()


def test_model_based_se_matches_statsmodels():
    X, count, expo, *_ = make_data()
    m = glm.fit_frequency(X, count, expo)
    tab = glm.coefficient_table(m, X, count, np.ones(len(X)), np.log(expo), robust=False)
    ref = sm.GLM(count, sm_design(X), family=sm.families.Poisson(), offset=np.log(expo)).fit()
    se = tab[tab.term != "area[A]"].se.to_numpy()
    np.testing.assert_allclose(se, ref.bse[["const", "B", "C", "age"]].to_numpy(), rtol=1e-2)


def test_gamma_model_based_se_close_to_statsmodels():
    X, _, _, ncl, cost, _ = make_data()
    m = glm.fit_severity(X, cost, ncl)
    tab = glm.coefficient_table(m, X, cost, ncl, None, robust=False)
    fam = sm.families.Gamma(sm.families.links.Log())
    ref = sm.GLM(cost, sm_design(X), family=fam, var_weights=ncl).fit()
    se = tab[tab.term != "area[A]"].se.to_numpy()
    np.testing.assert_allclose(se, ref.bse.to_numpy(), rtol=1e-2)


def test_intercept_only_table():
    X, count, expo, *_ = make_data()
    E = X.iloc[:, :0]
    m = glm.fit_frequency(E, count, expo)
    tab = glm.coefficient_table(m, E, count, np.ones(len(E)), np.log(expo), robust=False)
    assert len(tab) == 1 and tab.se.iloc[0] > 0


def test_round_trip():
    X, count, expo, ncl, cost, pp = make_data()
    for m in (
        glm.fit_frequency(X, count, expo),
        glm.fit_severity(X, cost, ncl),
        glm.fit_tweedie(X, pp, expo, 1.6),
        glm.fit_frequency(X.iloc[:, :0], count, expo),
    ):
        m2 = glm.GLMModel.from_dict(json.loads(json.dumps(m.to_dict())))
        Xs = X[[s["name"] for s in m.features]]
        np.testing.assert_array_equal(m.predict_rate(Xs), m2.predict_rate(Xs))


def test_unseen_category_raises():
    X, count, expo, *_ = make_data()
    m = glm.fit_frequency(X, count, expo)
    X2 = X.copy()
    X2["area"] = pd.Categorical(["Z"] * len(X), categories=["Z"])
    with pytest.raises(ValueError, match="not in training categories"):
        m.predict_rate(X2)


def test_severity_rejects_nonpositive():
    X, _, _, ncl, cost, _ = make_data()
    bad = ncl.copy()
    bad[0] = 0
    with pytest.raises(ValueError):
        glm.fit_severity(X, cost, bad)
    bad_cost = cost.copy()
    bad_cost[1] = 0
    with pytest.raises(ValueError):
        glm.fit_severity(X, bad_cost, ncl)


def test_tweedie_power_range():
    X, _, expo, _, _, pp = make_data()
    with pytest.raises(ValueError):
        glm.fit_tweedie(X, pp, expo, 2.0)


def test_empty_category_raises_named_error():
    X, count, expo, *_ = make_data()
    X["area"] = pd.Categorical(X["area"].astype(str), categories=["A", "B", "C", "Z"])
    with pytest.raises(ValueError, match=r"area\[Z\]"):
        glm.fit_frequency(X, count, expo)


def test_non_finite_numeric_raises():
    X, count, expo, *_ = make_data()
    X.loc[3, "age"] = np.nan
    with pytest.raises(ValueError, match="age"):
        glm.fit_frequency(X, count, expo)
    X.loc[3, "age"] = np.inf
    with pytest.raises(ValueError, match="age"):
        glm.fit_frequency(X, count, expo)


def test_coefficient_table_length_guard():
    X, count, expo, *_ = make_data()
    m = glm.fit_frequency(X, count, expo)
    with pytest.raises(ValueError):
        glm.coefficient_table(m, X, count, np.ones(len(X) - 1), offset=np.log(expo))


def test_convergence_flag_and_solver_override():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.normal(size=300)})
    y = rng.poisson(np.exp(0.3 * X["x"]))
    ok = glm.fit_frequency(X, y, np.ones(300))
    assert ok.converged is True and ok.n_iter >= 1
    hard = glm.fit_frequency(X, y, np.ones(300), {"max_iter": 1, "gradient_tol": 1e-14})
    assert hard.converged is False and hard.n_iter == 1


def test_solver_by_family_overrides_only_that_family():
    solver = {
        "max_iter": 50,
        "gradient_tol": 1e-7,
        "by_family": {"tweedie": {"gradient_tol": 1e-5}},
    }
    assert glm._estimator("tweedie", 1.5, True, solver).gradient_tol == 1e-5
    assert glm._estimator("poisson", None, True, solver).gradient_tol == 1e-7
    assert glm._estimator("gamma", None, True, solver).gradient_tol == 1e-7
    assert glm._estimator("tweedie", 1.5, True, solver).max_iter == 50


def test_l2_penalty_shrinks_and_is_weight_scale_invariant():
    X, _, _, ncl, _, _ = make_data()
    rng = np.random.default_rng(3)
    avg = np.exp(0.5 * X["age"].to_numpy() + 1) * rng.gamma(5, 0.2, len(X))
    plain = glm.fit_severity(X, avg, ncl)
    small = glm.fit_severity(X, avg, ncl, l2=1e-3)
    big = glm.fit_severity(X, avg, ncl, l2=1.0)
    norm = lambda m: float(np.linalg.norm(m.coef))  # noqa: E731
    assert norm(big) < norm(small) < norm(plain)
    scaled = glm.fit_severity(X, avg, ncl * 10, l2=1e-3)
    assert np.allclose(scaled.coef, small.coef, atol=1e-6)
    assert np.isclose(scaled.intercept, small.intercept, atol=1e-6)
    d = small.to_dict()
    assert d["l2"] == 1e-3 and glm.GLMModel.from_dict(json.loads(json.dumps(d))).l2 == 1e-3
    assert "l2" not in plain.to_dict()


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_l2_must_be_positive(bad):
    X, _, _, ncl, _, _ = make_data()
    with pytest.raises(ValueError, match="l2"):
        glm.fit_severity(X, np.ones(len(X)), ncl, l2=bad)
