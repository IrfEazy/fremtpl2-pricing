"""Pre-registered evaluation split: grouped (primary) and random (sensitivity), plus CV folds.

Groups are tuples of all rating factors. Splits are stratified on claim presence. Severity rows are
attached by IDpol after assignment, so all claims of a policy stay on the same side.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from .clean import clean
from .data import load_raw
from .groups import group_ids


def _grouped_test_mask(
    y: np.ndarray, groups: np.ndarray, n_folds: int, test_fold: int, seed: int
) -> np.ndarray:
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for i, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        if i == test_fold:
            mask = np.zeros(len(y), dtype=bool)
            mask[test_idx] = True
            return mask
    raise ValueError(f"test_fold {test_fold} outside 0..{n_folds - 1}")


def assign_split(freq: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Return IDpol, group_id, primary_set, cv_fold (-1 for test), random_set."""
    sc = cfg["split"]
    if sc["primary"] not in ("grouped", "random"):
        raise ValueError("split.primary must be 'grouped' or 'random'")
    seed = cfg["seed"]
    y = (freq["ClaimNb"].to_numpy() > 0).astype(int)
    gid = group_ids(freq, cfg["rating_factors"]).to_numpy()
    n = len(freq)
    idx = np.arange(n)

    grouped_test = _grouped_test_mask(y, gid, sc["test_folds"], sc["test_fold"], seed)
    _, rnd_test_idx = train_test_split(
        idx, test_size=1 / sc["test_folds"], stratify=y, random_state=seed
    )
    random_test = np.zeros(n, dtype=bool)
    random_test[rnd_test_idx] = True
    test = grouped_test if sc["primary"] == "grouped" else random_test

    cv_fold = np.full(n, -1, dtype="int64")
    tr = np.flatnonzero(~test)
    if sc["primary"] == "grouped":
        cv = StratifiedGroupKFold(n_splits=sc["cv_folds"], shuffle=True, random_state=seed)
        splits = cv.split(np.zeros(len(tr)), y[tr], gid[tr])
    else:
        cv = StratifiedKFold(n_splits=sc["cv_folds"], shuffle=True, random_state=seed)
        splits = cv.split(np.zeros(len(tr)), y[tr])
    for k, (_, val) in enumerate(splits):
        cv_fold[tr[val]] = k
    return pd.DataFrame(
        {
            "IDpol": freq["IDpol"].to_numpy(),
            "group_id": gid,
            "primary_set": np.where(test, "test", "train"),
            "cv_fold": cv_fold,
            "random_set": np.where(random_test, "test", "train"),
        }
    )


def attach_severity(sev: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """Attach set labels to severity rows by IDpol, after the split was assigned.

    Every severity row of a policy gets that policy's labels, so all claims of a policy are on its
    side. Rows whose IDpol has no split entry (orphans) are excluded (inner join).
    """
    if not split["IDpol"].is_unique:
        raise ValueError("split must have one row per IDpol")
    cols = ["IDpol", "primary_set", "cv_fold", "random_set"]
    return sev.merge(split[cols], on="IDpol", how="inner", validate="many_to_one")


def _stats(freq: pd.DataFrame) -> dict[str, Any]:
    claims = int(freq["ClaimNb"].sum())
    expo = float(freq["Exposure"].sum())
    return {
        "policies": int(len(freq)),
        "exposure": expo,
        "claims": claims,
        "policies_with_claim": int((freq["ClaimNb"] > 0).sum()),
        "claim_rate": claims / expo if expo else None,
        "share_with_claim": float((freq["ClaimNb"] > 0).mean()) if len(freq) else None,
        "largest_group_policies": int(freq["group_id"].value_counts().max()) if len(freq) else 0,
        "largest_group_share_of_policies": (
            float(freq["group_id"].value_counts().max() / len(freq)) if len(freq) else None
        ),
        "largest_group_share_of_claims": (
            float(freq.groupby("group_id")["ClaimNb"].sum().max() / max(claims, 1))
            if len(freq)
            else None
        ),
    }


def split_summary(
    freq: pd.DataFrame, sev: pd.DataFrame, split: pd.DataFrame, cfg: dict[str, Any]
) -> dict[str, Any]:
    d = freq.merge(split, on="IDpol", validate="one_to_one")
    train, test = d[d["primary_set"] == "train"], d[d["primary_set"] == "test"]
    rtrain, rtest = d[d["random_set"] == "train"], d[d["random_set"] == "test"]

    def overlap(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, int]:
        return {
            "idpol": int(len(set(a["IDpol"]) & set(b["IDpol"]))),
            "group_id": int(len(set(a["group_id"]) & set(b["group_id"]))),
        }

    folds = {
        str(k): _stats(train[train["cv_fold"] == k]) for k in sorted(train["cv_fold"].unique())
    }
    fold_group_overlap = int((train.groupby("group_id")["cv_fold"].nunique() > 1).sum())
    sev_attached = attach_severity(sev, split)
    return {
        "primary": cfg["split"]["primary"],
        "seed": cfg["seed"],
        "settings": cfg["split"],
        "policies": int(len(d)),
        "primary_sets": {"train": _stats(train), "test": _stats(test)},
        "test_share_of_policies": float(len(test) / len(d)),
        "cv_folds": folds,
        "random_sets": {"train": _stats(rtrain), "test": _stats(rtest)},
        "random_test_share_of_policies": float(len(rtest) / len(d)),
        "overlap_primary_train_test": overlap(train, test),
        "overlap_random_train_test": overlap(rtrain, rtest),
        "groups_spanning_multiple_cv_folds": fold_group_overlap,
        "policies_without_cv_fold_in_train": int((train["cv_fold"] < 0).sum()),
        "test_policies_with_cv_fold": int((test["cv_fold"] >= 0).sum()),
        "severity_rows_by_primary_set": {
            "train": int((sev_attached["primary_set"] == "train").sum()),
            "test": int((sev_attached["primary_set"] == "test").sum()),
            "excluded_without_policy": int(len(sev) - len(sev_attached)),
        },
        "group_sizes_largest": int(d["group_id"].value_counts().max()),
    }


def run_split(cfg: dict[str, Any]) -> dict[str, Any]:
    freq, sev = load_raw(cfg)
    res = clean(freq, sev, cfg["clean"])
    split = assign_split(res.freq, cfg)
    summary = split_summary(res.freq, res.sev, split, cfg)
    processed = Path(cfg["paths"]["processed"])
    results = Path(cfg["paths"]["results"])
    processed.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    split.to_parquet(processed / "split.parquet", index=False)
    (results / "split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
