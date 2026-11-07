"""Config-driven pricing features: ``fit`` learns level sets, ``transform`` applies them.

All definitions (band edges, caps, rare-level threshold, feature sets) come from the ``features``
config section. ``fit`` learns, from the fitting rows only, per categorical feature the ordered
level list (reference first; only levels with fitting exposure; the other level only if a rare
level was grouped), the reference level (largest exposure) and the rare levels merged into the
other label.
``transform`` depends only on a row and the ``FeatureMap``; it never uses exposure or targets.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

EXPOSURE = "Exposure"


@dataclass
class FeatureMap:
    """JSON-serialisable result of ``fit``: the feature config plus the learned levels."""

    config: dict
    levels: dict  # feature -> {"levels": [...], "reference": str, "rare": [...]}

    def to_dict(self) -> dict:
        return {"config": copy.deepcopy(self.config), "levels": copy.deepcopy(self.levels)}

    @classmethod
    def from_dict(cls, d: dict) -> FeatureMap:
        return cls(config=copy.deepcopy(d["config"]), levels=copy.deepcopy(d["levels"]))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> FeatureMap:
        return cls.from_dict(json.loads(s))


def _band_labels(edges: list) -> list[str]:
    labels = [f"{lo}-{hi - 1}" for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    return labels + [f"{edges[-1]}+"]


def _band(s: pd.Series, edges: list) -> pd.Series:
    if s.isna().any():
        raise ValueError(f"{s.name}: missing values cannot be banded")
    idx = np.searchsorted(np.asarray(edges), s.to_numpy(), side="right") - 1
    labels = np.asarray(_band_labels(edges), dtype=object)
    return pd.Series(labels[np.clip(idx, 0, len(edges) - 1)], index=s.index, name=s.name)


def _categorical_values(s: pd.Series, spec: dict) -> pd.Series:
    """Raw string labels of a categorical feature (after the optional numeric cap)."""
    if s.isna().any():
        raise ValueError(f"{s.name}: missing values are not allowed")
    if spec.get("cap") is not None:
        s = s.clip(upper=spec["cap"]).astype(int)
    return s.astype(str)


def _labels(df: pd.DataFrame, name: str, spec: dict) -> pd.Series:
    if spec["kind"] == "band":
        return _band(df[name], spec["lower_edges"])
    return _categorical_values(df[name], spec)


def _level_sort_key(v: str):
    try:
        return (0, float(v), v)
    except ValueError:
        return (1, 0.0, v)


def fit(
    train_df: pd.DataFrame,
    feat_cfg: dict,
    weight_column: str = EXPOSURE,
    min_weight: float | None = None,
) -> FeatureMap:
    """Learn level sets, reference levels and rare levels from ``train_df`` only.

    ``weight_column`` is the weight of the rare-level and reference choice (default Exposure;
    the severity map uses the claim count N on claims-only rows) and ``min_weight`` the rare
    threshold (default ``rare_levels.min_exposure_years``).
    """
    defs = feat_cfg["definitions"]
    other = feat_cfg["rare_levels"]["other_label"]
    min_exp = feat_cfg["rare_levels"]["min_exposure_years"] if min_weight is None else min_weight
    if feat_cfg.get("reference_level") != "max_exposure":
        raise ValueError("features.reference_level: only 'max_exposure' is supported")
    exposure = train_df[weight_column]
    levels: dict = {}
    for name, spec in defs.items():
        kind = spec["kind"]
        if kind in ("numeric_cap", "log"):
            continue
        if kind not in ("band", "categorical"):
            raise ValueError(f"features.definitions.{name}: unknown kind {kind!r}")
        labels = _labels(train_df, name, spec)
        by_level = exposure.groupby(labels.to_numpy()).sum()
        by_level = by_level[by_level > 0]  # only levels with positive fitting exposure
        if kind == "band":
            natural = [lv for lv in _band_labels(spec["lower_edges"]) if lv in by_level.index]
        else:
            natural = sorted(by_level.index, key=_level_sort_key)
        rare: list[str] = []
        if kind == "categorical" and spec.get("group_rare"):
            rare = [lv for lv in natural if by_level[lv] < min_exp]
            natural = [lv for lv in natural if lv not in rare]
        if not natural:
            raise ValueError(
                f"{name}: no non-rare level with positive exposure in the fitting data"
            )
        # reference = largest exposure (ties: natural order); stored first, as the GLM drops it
        reference = max(natural, key=lambda lv: (by_level[lv], -natural.index(lv)))
        ordered = [reference] + [lv for lv in natural if lv != reference]
        if rare:  # the other level exists only if a rare level was grouped in the fitting data
            ordered.append(other)
        levels[name] = {"levels": ordered, "reference": reference, "rare": rare}
    return FeatureMap(config=copy.deepcopy(feat_cfg), levels=levels)


def feature_names(feat_cfg: dict, feature_set: str = "full") -> list[str]:
    """Feature names of a feature set declared in config."""
    sets = feat_cfg["feature_sets"]
    if feature_set not in sets:
        raise KeyError(f"unknown feature set {feature_set!r}; declared: {sorted(sets)}")
    return list(sets[feature_set])


def transform(df: pd.DataFrame, fmap: FeatureMap) -> pd.DataFrame:
    """Model features for every row of ``df``; row-wise, depends only on the row and ``fmap``."""
    cfg = fmap.config
    defs = cfg["definitions"]
    other = cfg["rare_levels"]["other_label"]
    out: dict = {}
    for name in feature_names(cfg, "full"):
        spec = defs[name]
        kind = spec["kind"]
        if kind == "numeric_cap":
            if df[name].isna().any():
                raise ValueError(f"{name}: missing values are not allowed")
            out[name] = df[name].clip(upper=spec["cap"]).astype(float)
        elif kind == "log":
            if (df[name] <= 0).any() or df[name].isna().any():
                raise ValueError(f"{name}: log transform needs positive values")
            out[name] = np.log(df[name].astype(float))
        else:
            info = fmap.levels[name]
            known = set(info["levels"])
            rare = set(info["rare"])
            labels = _labels(df, name, spec)
            mapped = labels.where(~labels.isin(rare), other)
            unseen = ~mapped.isin(known)
            if unseen.any():
                if spec.get("group_rare") and other in known:
                    mapped = mapped.where(~unseen, other)
                else:
                    vals = sorted(set(mapped[unseen]))
                    raise ValueError(f"{name}: levels unseen in the fitting data: {vals}")
            out[name] = pd.Categorical(mapped, categories=info["levels"])
    return pd.DataFrame(out, index=df.index)
