"""Synthetic freq/sev frames and config for offline tests."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_DEFAULTS = yaml.safe_load((Path(__file__).parents[1] / "configs" / "default.yaml").read_text())

FACTORS = [
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


def model_cfg() -> dict:
    """features + models sections of the default config, shrunk for small synthetic data."""
    feats = copy.deepcopy(_DEFAULTS["features"])
    feats["rare_levels"]["min_exposure_years"] = 20
    models = copy.deepcopy(_DEFAULTS["models"])
    models["tweedie"]["power_grid"] = [1.4, 1.5, 1.6]
    models["severity_features"]["min_claims"] = 5
    models["parallel"]["n_jobs"] = 1
    return {"features": feats, "models": models}


def make_cfg(seed: int = 7, primary: str = "grouped") -> dict:
    return {
        **model_cfg(),
        "seed": seed,
        "rating_factors": FACTORS,
        "audit": {
            "exposure_small_thresholds": [0.05, 0.2],
            "exposure_quantiles": [0.5],
            "exposure_bands": [0.01, 0.05, 1.0],
            "claim_nb_extreme_above": 4,
            "high_count_n_above": 5,
            "amount_quantiles": [0.5, 0.99],
            "top_n": 2,
            "large_loss_candidates": [1000, 10000],
            "cap_table": [500, 1000, None],
            "idpol_band_edges": [3, 6],
            "discrepancy_numeric_bins": 3,
            "external_reference": {"source": "doc", "policies": 6, "severity_rows": 4},
            "small_amount_threshold": 100,
            "group_size_bins": [1, 2, 3, 4, 5, 10],
            "epsilon": 1e-12,
        },
        "split": {"primary": primary, "test_folds": 5, "test_fold": 0, "cv_folds": 5},
        "clean": {
            "duplicate_idpol": "error",
            "exposure_cap": 1.0,
            "exposure_min": None,
            "claim_nb": {"max": None, "action": "cap"},
            "high_count": {"exclude_n_above": None},
            "sev_without_policy": "drop",
            "sev_consistency": {
                "claims_without_sev": "flag",
                "sev_without_claims": "flag",
                "count_mismatch": "flag",
            },
            "large_loss": {"cap": 1000.0},
            "epsilon": 1e-12,
            "targets": {"frequency_count": "sev_rows"},
            "excess_loading": {"mode": "flat_per_exposure_year"},
            "sensitivities": {
                "exclude_high_count": {"high_count": {"exclude_n_above": 5}},
                "frequency_recorded_claim_nb": {"targets": {"frequency_count": "claim_nb"}},
                "cap_500": {"large_loss": {"cap": 500}},
                "cap_uncapped": {"large_loss": {"cap": None}},
                "raw_exposure": {"exposure_cap": None},
                "short_records": {"exposure_min": 0.05},
            },
        },
    }


def make_freq(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # small domains so many policies share a rating-factor tuple
    return pd.DataFrame(
        {
            "IDpol": np.arange(1, n + 1) * 3,
            "ClaimNb": rng.choice([0, 1, 2, 7], size=n, p=[0.93, 0.05, 0.015, 0.005]),
            "Exposure": rng.uniform(0.01, 1.4, size=n),
            "Area": pd.Categorical(rng.choice(list("AB"), size=n)),
            "VehPower": rng.integers(4, 7, size=n),
            "VehAge": rng.integers(0, 4, size=n),
            "DrivAge": rng.integers(20, 30, size=n),
            "BonusMalus": rng.choice([50, 60], size=n),
            "VehBrand": pd.Categorical(rng.choice(["B1", "B2", "B3"], size=n)),
            "VehGas": rng.choice(["Diesel", "Regular"], size=n),
            "Density": rng.choice([10, 100, 1000], size=n),
            "Region": pd.Categorical(rng.choice(["R1", "R2"], size=n)),
        }
    )


def make_sev(freq: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.repeat(freq["IDpol"].to_numpy(), freq["ClaimNb"].clip(upper=2).to_numpy())
    ids = ids[rng.random(len(ids)) < 0.8]  # some claims lack severity rows
    amount = rng.lognormal(6, 1.5, size=len(ids))
    orphans = np.array([-1, -2])
    return pd.DataFrame(
        {
            "IDpol": np.concatenate([ids, orphans]),
            "ClaimAmount": np.concatenate([amount, [50.0, 5000.0]]),
        }
    )


def tiny_gbm_block():
    comp = lambda obj, metric, mono: {  # noqa: E731
        "objective": obj,
        "metric": metric,
        "monotone": mono,
        "params": {},
        "grid": {"num_leaves": [3, 5], "min_child_samples": [20], "learning_rate": [0.2]},
    }
    return {
        "seed": 11,
        "features": {
            "numeric": ["DrivAge", "VehAge", "VehPower", "BonusMalus", "Density"],
            "categorical": ["Area", "VehGas", "VehBrand", "Region"],
        },
        "unseen_level": "missing",
        "deterministic": {"deterministic": True, "force_row_wise": True, "num_threads": 1},
        "base_params": {
            "verbosity": -1,
            "max_bin": 63,
            "min_data_per_group": 10,
            "cat_smooth": 5.0,
            "monotone_constraints_method": "basic",
        },
        "early_stopping": {
            "round_cap": 25,
            "patience": 5,
            "min_improvement": 0.0,
            "round_rounding": "nearest",
            "at_cap": "record_use_cap",
        },
        "components": {
            "frequency": comp("poisson", "poisson", {"BonusMalus": 1}),
            "severity": comp("gamma", "gamma_deviance", {}),
            "tweedie": comp("tweedie", "tweedie", {"BonusMalus": 1}),
        },
        "diagnostic": {"enabled": True, "components": ["frequency", "tweedie"], "ablation": True},
        "checks": {
            "reload_rows": 500,
            "reload_tolerance": 1e-9,
            "sweep_rows": 50,
            "sweep_points": 8,
            "monotone_tolerance": 1e-12,
        },
        "outputs": {
            "cv_json": "cv_gbm.json",
            "models_json": "gbm_models.json",
            "report": "gbm_cv.md",
            "model_dir": "models",
            "staging_dir": "gbm_staging",
            "failure_diagnostic": "gbm_FAILED.json",
        },
    }
