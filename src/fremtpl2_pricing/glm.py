"""Thin unpenalised GLM layer on glum for the three pricing models.

No configuration is read here: every hyperparameter is an argument.

Encoding: ``pandas.Categorical`` columns are dummy-coded dropping the FIRST category, so the
caller must order the reference level first in each Categorical. All other columns are numeric
and passed through. An empty feature frame (no columns) gives the intercept-only (null) model.

Exposure handling (applied exactly once):
- frequency: Poisson, log link, offset = log(exposure), unit sample weight, y = claim count;
- severity: Gamma, log link, sample weight = number of claims, no offset, y = average cost;
- Tweedie: log link, sample weight = exposure, no offset, y = pure premium (cost / exposure).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor, TweedieDistribution
from scipy import stats
from sklearn.exceptions import ConvergenceWarning

INTERCEPT = "(Intercept)"
# Solver settings used when the caller passes none (the pipeline passes models.glm.solver).
DEFAULT_SOLVER = {"max_iter": 1000, "gradient_tol": 1e-10}


def _feature_specs(X: pd.DataFrame) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for col in X.columns:
        if isinstance(X[col].dtype, pd.CategoricalDtype):
            specs.append(
                {"name": str(col), "kind": "cat", "categories": list(X[col].cat.categories)}
            )
        else:
            specs.append({"name": str(col), "kind": "num", "categories": None})
    return specs


def _design(
    X: pd.DataFrame, specs: list[dict[str, Any]]
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Dense design matrix and (feature, level) per column, using the training categories."""
    if [s["name"] for s in specs] != [str(c) for c in X.columns]:
        raise ValueError("feature columns differ from the fitted model")
    cols: list[np.ndarray] = []
    terms: list[tuple[str, str]] = []
    for spec in specs:
        values = X[spec["name"]]
        if spec["kind"] == "num":
            arr = values.to_numpy(dtype=float)
            if not np.isfinite(arr).all():
                raise ValueError(f"{spec['name']}: non-finite numeric values")
            cols.append(arr)
            terms.append((spec["name"], ""))
            continue
        cats = spec["categories"]
        raw = values.astype(object).to_numpy()
        unseen = set(pd.unique(raw[~pd.isna(raw)])) - set(cats)
        if unseen or pd.isna(raw).any():
            raise ValueError(
                f"{spec['name']}: values not in training categories: {sorted(map(str, unseen))}"
            )
        for level in cats[1:]:
            cols.append((raw == level).astype(float))
            terms.append((spec["name"], str(level)))
    if not cols:
        return np.ones((len(X), 1)), [(INTERCEPT, "")]
    return np.column_stack(cols), terms


@dataclass
class GLMModel:
    """A fitted unpenalised log-link GLM (``kind`` is frequency, severity or tweedie)."""

    kind: str
    family: str
    power: float | None
    link: str
    intercept: float
    coef: list[float]
    features: list[dict[str, Any]]
    estimator: Any = field(default=None, repr=False, compare=False)
    converged: bool | None = field(default=None, compare=False)  # False if glum warned
    n_iter: int | None = field(default=None, compare=False)

    def _eta(self, X: pd.DataFrame) -> np.ndarray:
        D, _ = _design(X, self.features)
        if not self.features:
            return np.full(len(X), self.intercept)
        return self.intercept + D @ np.asarray(self.coef)

    def predict_rate(self, X: pd.DataFrame) -> np.ndarray:
        """Expected cost per unit: claims per exposure-year (frequency), cost per claim
        (severity) or pure premium per exposure-year (tweedie)."""
        return np.exp(self._eta(X))

    def predict_count(self, X: pd.DataFrame, exposure) -> np.ndarray:
        """Predicted claim counts = rate * exposure (frequency models only)."""
        if self.kind != "frequency":
            raise ValueError("predict_count is only defined for frequency models")
        return self.predict_rate(X) * np.asarray(exposure, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "family": self.family,
            "power": self.power,
            "link": self.link,
            "intercept": float(self.intercept),
            "coef": [float(c) for c in self.coef],
            "features": [
                {**s, "categories": None if s["categories"] is None else list(s["categories"])}
                for s in self.features
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GLMModel:
        return cls(
            kind=d["kind"],
            family=d["family"],
            power=d["power"],
            link=d["link"],
            intercept=float(d["intercept"]),
            coef=[float(c) for c in d["coef"]],
            features=[dict(s) for s in d["features"]],
        )


def _fit(kind, X, y, weight, offset, family, power, solver=None) -> GLMModel:
    y = np.asarray(y, dtype=float)
    weight = np.asarray(weight, dtype=float)
    if not (len(X) == len(y) == len(weight)):
        raise ValueError("X, y and weights must have the same length")
    specs = _feature_specs(X)
    D, _ = _design(X, specs)
    empty = [
        f"{s['name']}[{lev}]"
        for s in specs
        if s["kind"] == "cat"
        for lev in s["categories"]
        if not (X[s["name"]].astype(object) == lev).any()
    ]
    if empty:
        raise ValueError(f"categories without training rows (empty levels): {empty}")
    est = _estimator(family, power, fit_intercept=bool(specs), solver=solver)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est.fit(D, y, sample_weight=weight, offset=offset)
    for w in caught:  # re-emit everything except the convergence warning, which is recorded
        if not issubclass(w.category, ConvergenceWarning):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    if specs:
        intercept, coef = float(est.intercept_), [float(c) for c in est.coef_]
    else:  # ones column carries the intercept
        intercept, coef = float(est.coef_[0]), []
    return GLMModel(
        kind, family, power, "log", intercept, coef, specs, est, converged, int(est.n_iter_)
    )


def _estimator(family, power, fit_intercept, solver=None) -> GeneralizedLinearRegressor:
    opts = {**DEFAULT_SOLVER, **(solver or {})}
    # optional per-family overrides, e.g. {"tweedie": {"gradient_tol": 1e-5}}
    opts.update((opts.get("by_family") or {}).get("tweedie" if power is not None else family, {}))
    if power is not None:
        family = TweedieDistribution(power)
    return GeneralizedLinearRegressor(
        family=family,
        link="log",
        alpha=0,
        fit_intercept=fit_intercept,
        max_iter=int(opts["max_iter"]),
        gradient_tol=float(opts["gradient_tol"]),
    )


def fit_frequency(X: pd.DataFrame, count, exposure, solver=None) -> GLMModel:
    """Poisson log link, offset log(exposure), unit weight."""
    exposure = np.asarray(exposure, dtype=float)
    if (exposure <= 0).any():
        raise ValueError("exposure must be > 0")
    return _fit(
        "frequency", X, count, np.ones(len(exposure)), np.log(exposure), "poisson", None, solver
    )


def fit_severity(X: pd.DataFrame, avg_cost, n_claims, solver=None) -> GLMModel:
    """Gamma log link, weight = number of claims; rows with n_claims <= 0 or cost <= 0 rejected."""
    avg_cost = np.asarray(avg_cost, dtype=float)
    n_claims = np.asarray(n_claims, dtype=float)
    if (n_claims <= 0).any() or (avg_cost <= 0).any():
        raise ValueError("severity needs n_claims > 0 and avg_cost > 0 on every row")
    return _fit("severity", X, avg_cost, n_claims, None, "gamma", None, solver)


def fit_tweedie(X: pd.DataFrame, pure_premium, exposure, power: float, solver=None) -> GLMModel:
    """Tweedie log link on pure premium per exposure-year, weight = exposure."""
    if not 1 < power < 2:
        raise ValueError("Tweedie power must be in (1, 2)")
    exposure = np.asarray(exposure, dtype=float)
    if (exposure <= 0).any():
        raise ValueError("exposure must be > 0")
    return _fit("tweedie", X, pure_premium, exposure, None, "tweedie", float(power), solver)


def _dispersion(model: GLMModel, X, y, weights) -> float:
    """Model-based dispersion: 1 for Poisson, else Pearson chi2 / (n - p) with prior weights.

    Passed explicitly because glum's own estimate does not match the weighted Pearson
    statistic used by statsmodels for Gamma/Tweedie with prior weights."""
    if model.family == "poisson":
        return 1.0
    mu = model.predict_rate(X)
    var_power = 2.0 if model.family == "gamma" else model.power
    n_params = 1 + sum(
        1 if s["kind"] == "num" else len(s["categories"]) - 1 for s in model.features
    )
    return float((weights * (y - mu) ** 2 / mu**var_power).sum() / (len(y) - n_params))


def coefficient_table(
    model: GLMModel, X, y, weights, offset=None, robust: bool = True, level: float = 0.95
) -> pd.DataFrame:
    """Coefficients with SE, z, p, relativities and CI. Needs the freshly fitted model."""
    if model.estimator is None:
        raise ValueError("coefficient_table needs a model fitted in this session (no estimator)")
    D, terms = _design(X, model.features)
    est = model.estimator
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not (len(y) == len(weights) == len(X)) or (
        offset is not None and len(np.asarray(offset)) != len(X)
    ):
        raise ValueError("X, y, weights and offset must have the same length")
    kwargs: dict[str, Any] = {"robust": robust}
    if not robust:
        kwargs["dispersion"] = _dispersion(model, X, y, weights)
    cov = np.asarray(est.covariance_matrix(D, y, weights, offset=offset, **kwargs))
    # glum orders the intercept first when fit_intercept=True
    se_all = np.sqrt(np.diag(cov))
    if model.features:
        se_int, se_coef = se_all[0], se_all[1:]
    else:
        se_int, se_coef = se_all[0], np.array([])
    zc = stats.norm.ppf(0.5 + level / 2)
    rows = [(INTERCEPT, INTERCEPT, "", model.intercept, se_int)]
    k = 0
    for spec in model.features:
        if spec["kind"] == "num":
            rows.append((spec["name"], spec["name"], "", model.coef[k], se_coef[k]))
            k += 1
            continue
        rows.append(
            (
                f"{spec['name']}[{spec['categories'][0]}]",
                spec["name"],
                str(spec["categories"][0]),
                0.0,
                np.nan,
            )
        )
        for lev in spec["categories"][1:]:
            rows.append(
                (f"{spec['name']}[{lev}]", spec["name"], str(lev), model.coef[k], se_coef[k])
            )
            k += 1
    df = pd.DataFrame(rows, columns=["term", "feature", "level", "coef", "se"])
    df["z"] = df["coef"] / df["se"]
    df["p_value"] = 2 * stats.norm.sf(np.abs(df["z"]))
    df["relativity"] = np.exp(df["coef"])
    df["rel_lo"] = np.exp(df["coef"] - zc * df["se"])
    df["rel_hi"] = np.exp(df["coef"] + zc * df["se"])
    df["covariance"] = "robust" if robust else "model-based"
    return df
