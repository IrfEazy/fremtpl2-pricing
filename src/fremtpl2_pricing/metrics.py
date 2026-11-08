"""Pure actuarial evaluation metrics.

Config-free: every parameter (power, n_bins, seed, replicates, level) is an argument and nothing
here reads files. All functions take array-likes plus an optional non-negative sample weight
(default 1). A weight is a multiplicity: it multiplies counts, losses and exposure alike, which is
what the bootstrap resampling needs.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp


def _vec(x, name: str = "array") -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return a


def _weights(weight, n: int) -> np.ndarray:
    if weight is None:
        return np.ones(n)
    w = _vec(weight, "weight")
    if len(w) != n:
        raise ValueError("weight length differs from data length")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weight must be finite and non-negative")
    return w


def _check_len(n: int, *arrays: np.ndarray) -> None:
    if any(len(a) != n for a in arrays):
        raise ValueError("inputs must have the same length")


# --------------------------------------------------------------------------- deviances


def _xlogy_ratio(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """y * log(y / mu) with 0 where y == 0."""
    out = np.zeros_like(y)
    pos = y > 0
    out[pos] = y[pos] * np.log(y[pos] / mu[pos])
    return out


def _prep(y, mu, weight) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _vec(y, "y")
    mu = _vec(mu, "mu")
    _check_len(len(y), mu)
    if np.any(y < 0):
        raise ValueError("y must be non-negative")
    if not np.all(np.isfinite(mu)) or np.any(mu <= 0):
        raise ValueError("mu must be finite and > 0")
    return y, mu, _weights(weight, len(y))


def poisson_unit_deviance(y, mu) -> np.ndarray:
    """2 * (y log(y/mu) - (y - mu)); equals 2 mu when y == 0."""
    y, mu, _ = _prep(y, mu, None)
    return 2.0 * (_xlogy_ratio(y, mu) - (y - mu))


def gamma_unit_deviance(y, mu) -> np.ndarray:
    """2 * (-log(y/mu) + (y - mu)/mu); requires y > 0."""
    y, mu, _ = _prep(y, mu, None)
    if np.any(y <= 0):
        raise ValueError("gamma deviance requires y > 0")
    return 2.0 * (-np.log(y / mu) + (y - mu) / mu)


def tweedie_unit_deviance(y, mu, power: float) -> np.ndarray:
    """Tweedie unit deviance for power in [1, 2] (1 = Poisson, 2 = Gamma limit)."""
    if not 1.0 <= power <= 2.0:
        raise ValueError("power must be in [1, 2]")
    if power == 1.0:
        return poisson_unit_deviance(y, mu)
    if power == 2.0:
        return gamma_unit_deviance(y, mu)
    y, mu, _ = _prep(y, mu, None)
    a, b = 1.0 - power, 2.0 - power
    return 2.0 * (y**b / (a * b) - y * mu**a / a + mu**b / b)


def poisson_deviance(y, mu, weight=None) -> float:
    """Total weighted Poisson deviance of COUNTS y; mu is the predicted count incl. exposure.

    Exposure is part of mu only; do not also pass it as weight.
    """
    _, _, w = _prep(y, mu, weight)
    return float(np.sum(w * poisson_unit_deviance(y, mu)))


def poisson_deviance_per_exposure(y, mu, exposure, weight=None) -> float:
    """Total Poisson count deviance divided by total exposure (exposure applied once).

    `mu` already contains the exposure; `exposure` only forms the denominator. With bootstrap
    multiplicities `weight`, both numerator and denominator are weighted.
    """
    exposure = _vec(exposure, "exposure")
    w = _weights(weight, len(exposure))
    denom = float(np.sum(w * exposure))
    if denom <= 0:
        raise ValueError("total exposure must be > 0")
    return poisson_deviance(y, mu, w) / denom


def gamma_deviance(y, mu, weight=None) -> float:
    """Total weighted Gamma deviance of average severity y; weight = number of claims.

    Requires y > 0 and raises ValueError otherwise.
    """
    _, _, w = _prep(y, mu, weight)
    return float(np.sum(w * gamma_unit_deviance(y, mu)))


def tweedie_deviance(y, mu, power: float, weight=None) -> float:
    """Total weighted Tweedie deviance of pure premium y (rate); weight = exposure.

    power in [1, 2]; power 1 equals Poisson, power 2 equals Gamma (needs y > 0).
    """
    _, _, w = _prep(y, mu, weight)
    return float(np.sum(w * tweedie_unit_deviance(y, mu, power)))


def _tweedie_log_w(y: np.ndarray, phi: np.ndarray, power: float, chunk_terms: int) -> np.ndarray:
    """log W(y, phi, p) of the Dunn & Smyth (2005) series for y > 0, 1 < p < 2.

    W = sum_{j>=1} W_j with log W_j = j z - log Gamma(j + 1) - log Gamma(-j alpha), where
    alpha = (2 - p) / (1 - p) and z = -alpha log y + alpha log(p - 1) - (1 - alpha) log phi
    - log(2 - p). The terms peak at j_max = y^(2-p) / (phi (2-p)) with spread about
    sqrt(j_max / (1 - alpha)); the sum runs over j_max +- 12 spreads (terms below e^-70 of the
    peak are dropped) and is evaluated with a log-sum-exp.
    """
    alpha = (2.0 - power) / (1.0 - power)
    z = -alpha * np.log(y) + alpha * np.log(power - 1.0) - (1.0 - alpha) * np.log(phi)
    z -= np.log(2.0 - power)
    jmax = y ** (2.0 - power) / (phi * (2.0 - power))
    half = 12.0 * np.sqrt(jmax / (1.0 - alpha)) + 8.0
    lo = np.maximum(1.0, np.floor(jmax - half))
    hi = np.maximum(lo + 1.0, np.ceil(jmax + half))
    out = np.empty(len(y))
    order = np.argsort(hi - lo)
    start = 0
    while start < len(order):  # rows of similar width share a padded block
        width = int((hi - lo)[order[start]]) + 1
        n_rows = max(1, chunk_terms // max(width, 1))
        end = min(len(order), start + n_rows)
        idx = order[start:end]
        width = int((hi - lo)[idx].max()) + 1
        while len(idx) > 1 and len(idx) * width > 4 * chunk_terms:
            end = start + (end - start) // 2
            idx = order[start:end]
            width = int((hi - lo)[idx].max()) + 1
        j = lo[idx, None] + np.arange(width)[None, :]
        valid = j <= hi[idx, None]
        lw = j * z[idx, None] - gammaln(j + 1.0) - gammaln(-j * alpha)
        out[idx] = logsumexp(np.where(valid, lw, -np.inf), axis=1)
        start = end
    return out


def tweedie_logpdf(y, mu, phi: float, power: float, weight=None, chunk_terms: int = 2_000_000):
    """Per-observation log density of a compound Poisson-gamma (Tweedie) response, 1 < p < 2.

    Reproductive form with prior weights: Y_i ~ Tw_p(mu_i, phi / w_i), i.e.
    f(y) = a(y, phi/w, p) exp{ w [y mu^(1-p)/(1-p) - mu^(2-p)/(2-p)] / phi }, with the
    Dunn & Smyth (2005) series for a(y, ., p) when y > 0. The point mass at zero is
    log P(Y = 0) = -w mu^(2-p) / (phi (2-p)). Observations with weight 0 return 0 (no
    information). The response is treated as continuous: any point mass above zero (for example
    values sitting exactly at a cap) is not modelled. Returns an array of the length of ``y``.
    """
    if not 1.0 < power < 2.0:
        raise ValueError("power must be in (1, 2)")
    if not np.isfinite(phi) or phi <= 0:
        raise ValueError("phi must be finite and > 0")
    y, mu, w = _prep(y, mu, weight)
    out = np.zeros(len(y))
    live = w > 0
    pos = live & (y > 0)
    zero = live & (y == 0)
    out[zero] = -w[zero] * mu[zero] ** (2.0 - power) / (phi * (2.0 - power))
    if pos.any():
        yp, mp, wp = y[pos], mu[pos], w[pos]
        phi_i = phi / wp
        kernel = yp * mp ** (1.0 - power) / (1.0 - power) - mp ** (2.0 - power) / (2.0 - power)
        out[pos] = -np.log(yp) + _tweedie_log_w(yp, phi_i, power, chunk_terms) + kernel / phi_i
    return out


def tweedie_loglik(y, mu, phi: float, power: float, weight=None) -> float:
    """Total log-likelihood: the sum of :func:`tweedie_logpdf` (see there)."""
    return float(np.sum(tweedie_logpdf(y, mu, phi, power, weight)))


def tweedie_mle_dispersion_fit(
    y,
    mu,
    power: float,
    weight=None,
    log_range: float = 3.0,
    xatol: float = 1e-3,
    edge_rtol: float = 0.01,
    max_expansions: int = 5,
) -> tuple[float, int]:
    """Maximum-likelihood dispersion phi for fixed means: bounded 1-D search over log phi.

    The first bracket is the weighted Pearson estimate times exp(+-log_range). An optimum within
    ``edge_rtol`` (relative to the half-width ``log_range``) of a bound is not trusted: the
    bracket is extended by ``log_range`` on that side and the search repeated, up to
    ``max_expansions`` times; if the optimum is still on a bound a ValueError is raised.
    Returns (phi, number of expansions)."""
    y, mu, w = _prep(y, mu, weight)
    live = w > 0
    pearson = float(np.sum(w[live] * (y[live] - mu[live]) ** 2 / mu[live] ** power) / live.sum())
    centre = np.log(max(pearson, 1e-12))

    def neg(log_phi: float) -> float:
        return -tweedie_loglik(y, mu, float(np.exp(log_phi)), power, w)

    lo, hi = centre - log_range, centre + log_range
    margin = edge_rtol * log_range
    for expansions in range(max_expansions + 1):
        res = minimize_scalar(neg, bounds=(lo, hi), method="bounded", options={"xatol": xatol})
        at_lo, at_hi = res.x - lo < margin, hi - res.x < margin
        if not (at_lo or at_hi):
            return float(np.exp(res.x)), expansions
        if expansions == max_expansions:
            break
        lo -= log_range if at_lo else 0.0
        hi += log_range if at_hi else 0.0
    raise ValueError(
        f"dispersion MLE (power {power}) is on the bracket bound after {max_expansions} "
        f"expansions: log phi in [{lo:.3f}, {hi:.3f}], optimum {res.x:.3f}"
    )


def tweedie_mle_dispersion(y, mu, power: float, weight=None, **kwargs) -> float:
    """Maximum-likelihood dispersion (see :func:`tweedie_mle_dispersion_fit`)."""
    return tweedie_mle_dispersion_fit(y, mu, power, weight, **kwargs)[0]


def deviance_explained(dev_model: float, dev_null: float) -> float:
    """1 - dev_model / dev_null."""
    if dev_null == 0:
        return float("nan")
    return 1.0 - dev_model / dev_null


def null_deviance(
    deviance_fn: Callable[..., float], y, null_mean: float, scale=None, weight=None, **kwargs
) -> float:
    """Deviance of a constant null model, with the null mean supplied by the caller.

    The caller passes the TRAINING-part mean (in the same unit as the evaluated y); this function
    never estimates it from the evaluated data. mu = null_mean * scale, where `scale` is e.g. the
    exposure for Poisson counts and None (1) for rates and severities. `kwargs` (e.g. power) go to
    `deviance_fn(y, mu, weight=..., **kwargs)`.
    """
    if not np.isfinite(null_mean) or null_mean <= 0:
        raise ValueError("null_mean must be finite and > 0")
    y = _vec(y, "y")
    mu = np.full(len(y), float(null_mean))
    if scale is not None:
        scale = _vec(scale, "scale")
        _check_len(len(y), scale)
        mu = mu * scale
    return deviance_fn(y, mu, weight=weight, **kwargs)


# --------------------------------------------------------------------------- Gini / Lorenz


def _lorenz_points(sort_key: np.ndarray, exposure: np.ndarray, loss: np.ndarray):
    """Cumulative exposure/loss shares ordered ascending by sort_key, ties aggregated."""
    keep = exposure > 0
    sort_key, exposure, loss = sort_key[keep], exposure[keep], loss[keep]
    if len(exposure) == 0:
        raise ValueError("total exposure must be > 0")
    uniq, inv = np.unique(sort_key, return_inverse=True)
    e = np.bincount(inv, weights=exposure, minlength=len(uniq))
    lo = np.bincount(inv, weights=loss, minlength=len(uniq))
    cum_e = np.concatenate([[0.0], np.cumsum(e)]) / e.sum()
    total_loss = lo.sum()
    cum_l = np.concatenate([[0.0], np.cumsum(lo)])
    cum_l = cum_l / total_loss if total_loss > 0 else np.full(len(cum_l), np.nan)
    return cum_e, cum_l


def _prep_gini(observed_loss, predicted_rate, exposure, weight):
    loss = _vec(observed_loss, "observed_loss")
    rate = _vec(predicted_rate, "predicted_rate")
    expo = _vec(exposure, "exposure")
    _check_len(len(loss), rate, expo)
    w = _weights(weight, len(loss))
    return loss * w, rate, expo * w


def lorenz_curve(observed_loss, predicted_rate, exposure, weight=None) -> pd.DataFrame:
    """Lorenz points (cum_exposure, cum_loss) ordered by predicted rate ascending.

    Starts at (0, 0); tied predictions form a single segment. Observed loss is a total (not a
    rate); with weights, loss and exposure are both multiplied by the weight. cum_loss is NaN when
    total observed loss is 0.
    """
    loss, rate, expo = _prep_gini(observed_loss, predicted_rate, exposure, weight)
    x, y = _lorenz_points(rate, expo, loss)
    return pd.DataFrame({"cum_exposure": x, "cum_loss": y})


def _gini_from_points(x: np.ndarray, y: np.ndarray) -> float:
    if np.isnan(y).any():
        return float("nan")
    return float(1.0 - 2.0 * np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0))


def gini(observed_loss, predicted_rate, exposure, weight=None) -> dict[str, float]:
    """Raw and normalised exposure-weighted Gini.

    Policies are sorted by predicted rate ascending (ties aggregated into one segment, so
    constant predictions give exactly 0); Gini = 1 - 2 * area under the Lorenz curve of cumulative
    exposure share vs cumulative observed loss share (trapezoid). Positive when higher predicted
    rate goes with higher observed rate. `normalized` = raw / oracle Gini, the oracle sorting by
    the observed rate loss / exposure; it is NaN when total observed loss is 0 or the oracle is 0.
    Returns {"gini": raw, "gini_normalized": normalised}.
    """
    loss, rate, expo = _prep_gini(observed_loss, predicted_rate, exposure, weight)
    raw = _gini_from_points(*_lorenz_points(rate, expo, loss))
    with np.errstate(divide="ignore", invalid="ignore"):
        obs_rate = np.where(expo > 0, loss / np.where(expo > 0, expo, 1.0), 0.0)
    oracle = _gini_from_points(*_lorenz_points(obs_rate, expo, loss))
    norm = float("nan") if not np.isfinite(oracle) or oracle == 0 else raw / oracle
    return {"gini": raw, "gini_normalized": float(norm)}


# --------------------------------------------------------------------------- lift tables


def _equal_exposure_bins(order_key: np.ndarray, expo: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin index per row: sorted ascending by order_key, bins of ~equal exposure."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    order = np.argsort(order_key, kind="stable")
    cum = np.cumsum(expo[order])
    total = cum[-1]
    if total <= 0:
        raise ValueError("total exposure must be > 0")
    mid = (cum - expo[order] / 2.0) / total
    sorted_bins = np.minimum((mid * n_bins).astype(int), n_bins - 1)
    bins = np.empty(len(order), dtype=int)
    bins[order] = sorted_bins
    return bins


def lift_table(observed, predicted, exposure, n_bins: int = 10, weight=None) -> pd.DataFrame:
    """Lift table by predicted rate, in `n_bins` bins of approximately equal exposure.

    `observed` and `predicted` are totals per row (same unit, predicted incl. exposure); the
    ranking key is predicted / exposure. Per bin: exposure, observed, predicted, observed_rate,
    predicted_rate, ratio (observed / predicted). Rows with zero effective exposure are dropped.
    """
    obs, pred, expo = (
        _vec(a, n)
        for a, n in ((observed, "observed"), (predicted, "predicted"), (exposure, "exposure"))
    )
    _check_len(len(obs), pred, expo)
    w = _weights(weight, len(obs))
    keep = expo * w > 0
    obs, pred, expo, w = obs[keep] * w[keep], pred[keep] * w[keep], expo[keep] * w[keep], w[keep]
    rate = pred / expo
    bins = _equal_exposure_bins(rate, expo, n_bins)
    df = pd.DataFrame({"bin": bins, "exposure": expo, "observed": obs, "predicted": pred})
    out = df.groupby("bin").sum()
    out["observed_rate"] = out["observed"] / out["exposure"]
    out["predicted_rate"] = out["predicted"] / out["exposure"]
    out["ratio"] = out["observed"] / out["predicted"]
    return out


def double_lift_table(
    observed, pred_a, pred_b, exposure, n_bins: int = 10, weight=None
) -> pd.DataFrame:
    """Double lift: bins by pred_a / pred_b (ascending) with ~equal exposure.

    Per bin: exposure, observed, pred_a, pred_b totals, ratio_a = observed / pred_a and
    ratio_b = observed / pred_b. Inputs are totals per row (predicted incl. exposure).
    """
    obs, a, b, expo = (
        _vec(x, n)
        for x, n in (
            (observed, "observed"),
            (pred_a, "pred_a"),
            (pred_b, "pred_b"),
            (exposure, "exposure"),
        )
    )
    _check_len(len(obs), a, b, expo)
    w = _weights(weight, len(obs))
    keep = expo * w > 0
    obs, a, b, expo = (
        obs[keep] * w[keep],
        a[keep] * w[keep],
        b[keep] * w[keep],
        expo[keep] * w[keep],
    )
    bins = _equal_exposure_bins(a / b, expo, n_bins)
    df = pd.DataFrame({"bin": bins, "exposure": expo, "observed": obs, "pred_a": a, "pred_b": b})
    out = df.groupby("bin").sum()
    out["ratio_a"] = out["observed"] / out["pred_a"]
    out["ratio_b"] = out["observed"] / out["pred_b"]
    return out


# --------------------------------------------------------------------------- calibration


def calibration_by_level(observed, predicted, levels, weight=None) -> pd.DataFrame:
    """Per level: weight (sum), observed and predicted totals, ratio observed / predicted.

    `observed` and `predicted` are totals per row; with weight w they are multiplied by w.
    """
    obs, pred = _vec(observed, "observed"), _vec(predicted, "predicted")
    lev = np.asarray(levels)
    _check_len(len(obs), pred, lev)
    w = _weights(weight, len(obs))
    df = pd.DataFrame({"level": lev, "weight": w, "observed": obs * w, "predicted": pred * w})
    out = df.groupby("level").sum()
    out["ratio"] = out["observed"] / out["predicted"]
    return out


def calibration_totals(observed, predicted, weight=None) -> dict[str, float]:
    """Total observed, total predicted and their ratio (observed / predicted)."""
    obs, pred = _vec(observed, "observed"), _vec(predicted, "predicted")
    _check_len(len(obs), pred)
    w = _weights(weight, len(obs))
    o, p = float(np.sum(obs * w)), float(np.sum(pred * w))
    return {"observed": o, "predicted": p, "ratio": o / p if p != 0 else float("nan")}


# --------------------------------------------------------------------------- bootstrap


def percentile_ci(values, level: float = 0.95) -> tuple[float, float]:
    """Two-sided percentile interval of replicate values."""
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    lo, hi = np.nanquantile(np.asarray(values, dtype=float), [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def paired_bootstrap(
    metric_fn: Callable[[np.ndarray], dict[str, float]],
    n: int,
    unit_ids,
    replicates: int,
    seed: int,
    level: float = 0.95,
) -> dict:
    """Paired bootstrap by multiplicity weights over resampling units.

    Units are the distinct values of `unit_ids` (groups, or policies when unit_ids =
    arange(n)). Each replicate draws as many units as there are, with replacement, and turns the
    draw into integer per-row weights (the unit's multiplicity); `metric_fn(weights)` returns a
    dict of floats. The same draws feed every metric in the dict, so evaluate all models and their
    differences inside one `metric_fn` to keep the comparison paired. Deterministic for a seed.

    Returns {"point": metric_fn(ones), "replicates": DataFrame (replicates x metric),
    "ci": DataFrame (metric x [lower, upper]), "level": level}.
    """
    unit_ids = np.asarray(unit_ids)
    if len(unit_ids) != n:
        raise ValueError("unit_ids length must equal n")
    if replicates < 1:
        raise ValueError("replicates must be >= 1")
    _, inverse = np.unique(unit_ids, return_inverse=True)
    k = int(inverse.max()) + 1
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(replicates):
        counts = np.bincount(rng.integers(0, k, size=k), minlength=k)
        rows.append(metric_fn(counts[inverse].astype(float)))
    reps = pd.DataFrame(rows)
    ci = pd.DataFrame(
        {c: percentile_ci(reps[c].to_numpy(), level) for c in reps.columns},
        index=["lower", "upper"],
    ).T
    point = {k_: float(v) for k_, v in metric_fn(np.ones(n)).items()}
    return {"point": point, "replicates": reps, "ci": ci, "level": level}
