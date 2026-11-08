"""CV selection rules of ``reports/evaluation_protocol.md`` section 4 (pure functions).

Both rules work on per-fold primary losses (lower is better) of the K registered folds and return
a machine-readable trace. The mean difference is compared with ``sd(d, ddof=1) / sqrt(K)``: a
complexity heuristic, not a significance test and not a valid standard error (training sets
overlap across folds and K is small).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

RULE_VERSION = "protocol-2026-11-08-s4-v1"


def _losses(name: str, values: Sequence[float], n_folds: int | None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError(f"{name}: per-fold losses must be a non-empty 1-d sequence")
    if n_folds is not None and len(arr) != n_folds:
        raise ValueError(f"{name}: {len(arr)} fold losses, expected {n_folds} (missing folds)")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: non-finite fold losses")
    return arr


def _paired(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """d = a - b with its mean, sd (ddof=1) and the heuristic threshold sd / sqrt(K)."""
    d = a - b
    k = len(d)
    if k < 2:
        raise ValueError("at least 2 folds are needed for the sd of the fold differences")
    sd = float(np.std(d, ddof=1))
    return {
        "d": [float(x) for x in d],
        "mean": float(np.mean(d)),
        "sd": sd,
        "threshold": sd / math.sqrt(k),
        "all_zero": bool(np.all(d == 0)),
    }


def ladder(per_fold_losses: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Complexity ladder (section 4, steps 3-5).

    ``per_fold_losses``: candidate -> per-fold primary loss, ordered simplest to most complex.
    The first candidate is the incumbent; each next rung replaces the CURRENT incumbent iff
    mean(d) > sd(d, ddof=1) / sqrt(K) with d_k = loss_incumbent,k - loss_next,k. Exact ties (all
    d_k equal to 0) and every mean(d) <= threshold keep the simpler candidate."""
    names = list(per_fold_losses)
    if len(names) < 2:
        raise ValueError("the ladder needs at least 2 candidates")
    first = _losses(names[0], per_fold_losses[names[0]], None)
    n_folds = len(first)
    losses = {names[0]: first} | {n: _losses(n, per_fold_losses[n], n_folds) for n in names[1:]}
    incumbent = names[0]
    steps = []
    for nxt in names[1:]:
        st = _paired(losses[incumbent], losses[nxt])
        replace = st["mean"] > st["threshold"]
        if replace:
            reason = "mean(d) > sd/sqrt(K): the next rung improves on the incumbent"
        elif st["all_zero"]:
            reason = "exact tie: the simpler candidate is kept"
        elif st["mean"] > 0:
            reason = "mean(d) > 0 but <= sd/sqrt(K): inconclusive, the simpler candidate is kept"
        else:
            reason = "mean(d) <= 0: no improvement, the simpler candidate is kept"
        steps.append(
            {
                "incumbent": incumbent,
                "next": nxt,
                "d_k": st["d"],
                "mean_d": st["mean"],
                "sd_d": st["sd"],
                "threshold": st["threshold"],
                "decision": "replace" if replace else "keep",
                "reason": reason,
                "inconclusive": bool(not st["all_zero"] and 0 < st["mean"] <= st["threshold"]),
            }
        )
        if replace:
            incumbent = nxt
    return {
        "rule": "ladder",
        "rule_version": RULE_VERSION,
        "n_folds": n_folds,
        "order": names,
        "steps": steps,
        "selected": incumbent,
    }


def freq_x_sev_vs_tweedie(
    freq_x_sev: Sequence[float],
    tweedie: Sequence[float],
    names: tuple[str, str] = ("freq_x_sev", "tweedie"),
) -> dict[str, Any]:
    """Frequency x severity versus Tweedie (section 4, step 6).

    d_k = loss_tweedie,k - loss_freqxsev,k. The lower mean loss wins, but if
    |mean(d)| <= sd(d, ddof=1) / sqrt(K) frequency x severity is kept (the difference is
    inconclusive)."""
    fxs = _losses(names[0], freq_x_sev, None)
    tw = _losses(names[1], tweedie, len(fxs))
    st = _paired(tw, fxs)
    inside = abs(st["mean"]) <= st["threshold"]
    if inside:
        winner = names[0]
        reason = (
            "exact tie: frequency x severity is kept"
            if st["all_zero"]
            else "|mean(d)| <= sd/sqrt(K): inconclusive, frequency x severity is kept"
        )
    elif st["mean"] < 0:
        winner = names[1]
        reason = "Tweedie has the lower mean loss and |mean(d)| > sd/sqrt(K)"
    else:
        winner = names[0]
        reason = "frequency x severity has the lower mean loss and |mean(d)| > sd/sqrt(K)"
    return {
        "rule": "freq_x_sev_vs_tweedie",
        "rule_version": RULE_VERSION,
        "n_folds": len(fxs),
        "steps": [
            {
                "candidates": list(names),
                "d_k": st["d"],
                "mean_d": st["mean"],
                "sd_d": st["sd"],
                "threshold": st["threshold"],
                "decision": winner,
                "reason": reason,
                "inconclusive": bool(inside),
            }
        ],
        "selected": winner,
    }
