"""Config-driven pricing features: ``fit`` learns level sets, ``transform`` applies them.

All definitions (band edges, caps, rare-level threshold, feature sets) come from the ``features``
config section. ``fit`` learns, from the fitting rows only, per categorical feature the ordered
level list (reference first; only levels with fitting exposure; the other level only if a rare
level was grouped), the reference level (largest exposure) and the rare levels merged into the
other label.
``fit_severity`` derives the severity map from the policy map of the same fitting part: the set of
allowed levels is the policy schema; levels with too few claims are merged (see there), and the
mapping policy level -> severity level is stored in the map.
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
    # feature -> {"levels": [...], "reference": str, "rare": [...]}; a severity map adds
    # "merge" (policy level -> severity level, one key per level of the policy schema) and "claims"
    levels: dict

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


def _natural_order(name: str, spec: dict, policy_levels: list[str], other: str) -> list[str]:
    if spec["kind"] == "band":
        return [lv for lv in _band_labels(spec["lower_edges"]) if lv in policy_levels]
    return sorted((lv for lv in policy_levels if lv != other), key=_level_sort_key)


def _merge_ordinal(natural: list[str], claims: dict, min_claims: float) -> dict[str, str]:
    """Merge levels with fewer than ``min_claims`` claims into the adjacent level with more claims
    (ties toward the reference = level with the most claims), smallest level first.
    The merged level keeps the label of the absorbing level. Returns level -> merged label."""
    groups = [([lv], float(claims[lv])) for lv in natural]  # (members, claims); first = label
    while len(groups) > 1:
        i = min(range(len(groups)), key=lambda g: (groups[g][1], g))
        if groups[i][1] >= min_claims:
            break
        ref = max(range(len(groups)), key=lambda g: (groups[g][1], -g))
        neigh = [j for j in (i - 1, i + 1) if 0 <= j < len(groups)]
        # larger claims first; tie -> the neighbour on the side of the reference
        j = max(neigh, key=lambda g: (groups[g][1], -abs(g - ref), -g))
        members = (groups[j][0] + groups[i][0]) if j < i else (groups[i][0] + groups[j][0])
        groups[min(i, j)] = (members, groups[i][1] + groups[j][1])
        del groups[max(i, j)]
    # the label of a group is that of its member with the most claims (first on ties)
    out: dict[str, str] = {}
    for members, _ in groups:
        label = max(members, key=lambda lv: (claims[lv], -members.index(lv)))
        out.update({lv: label for lv in members})
    return out


def _merge_nominal(
    order: list[str], claims: dict, min_claims: float, other: str
) -> tuple[dict[str, str], str]:
    """Nominal merge: levels (other than the reference = most claims) with fewer than
    ``min_claims`` claims form the ``other`` level; if that is still below the threshold it joins
    the reference. An existing policy ``other`` level above the threshold absorbs them.
    Returns (level -> merged label, reference)."""
    cand = [lv for lv in order if lv != other]
    ref = max(cand, key=lambda lv: (claims[lv], -cand.index(lv))) if cand else other
    small = [lv for lv in order if lv != ref and claims[lv] < min_claims]
    mapping = {lv: lv for lv in order}
    if small:
        total = sum(claims[lv] for lv in small)
        if other in order and other not in small:
            total += claims[other]
        target = ref if total < min_claims else other
        for lv in small:
            mapping[lv] = target
    return mapping, ref


def fit_severity(
    claims_df: pd.DataFrame, feat_cfg: dict, policy_map: FeatureMap, min_claims: float,
    ordinal: list[str], weight_column: str = "N",
) -> FeatureMap:  # fmt: skip
    """Severity feature map derived from ``policy_map`` and the claims-only fitting rows (the claim
    count of a level is the sum of ``weight_column`` over its rows).

    The allowed levels per feature are those of the policy schema (a value outside it still
    raises in ``transform``). Per feature the claim count of every policy level (zero when a
    level has no claim) decides the merge: nominal features (not in ``ordinal``): levels with
    fewer than ``min_claims`` claims form the other level, which joins the reference level if
    still below the threshold; ordinal features: such a level joins the adjacent level with more
    claims (ties toward the reference). Every policy level maps to a severity level that has
    claims, so no severity design column is empty. Numeric features are not mapped. An ordinal
    feature whose policy map has the other level (``group_rare``) is rejected.
    """
    defs = feat_cfg["definitions"]
    other = feat_cfg["rare_levels"]["other_label"]
    levels: dict = {}
    for name, info in policy_map.levels.items():
        spec = defs[name]
        plevels = list(info["levels"])
        labels = _labels(claims_df, name, spec)
        rare = set(info["rare"])
        labels = labels.where(~labels.isin(rare), other)
        outside = ~labels.isin(plevels)
        if outside.any():
            if spec.get("group_rare") and other in plevels:
                labels = labels.where(~outside, other)
            else:
                raise ValueError(
                    f"{name}: levels unseen in the policy schema: {sorted(set(labels[outside]))}"
                )
        counts = claims_df[weight_column].groupby(labels.to_numpy()).sum()
        claims = {lv: float(counts.get(lv, 0.0)) for lv in plevels}
        if name in ordinal:
            if other in plevels:
                raise ValueError(
                    f"{name}: an ordinal severity feature cannot be combined with group_rare "
                    f"(the policy map has the level {other!r}); declare it nominal"
                )
            order = _natural_order(name, spec, plevels, other)
            mapping = _merge_ordinal(order, claims, min_claims)
            total: dict[str, float] = {}
            for lv, target in mapping.items():
                total[target] = total.get(target, 0.0) + claims[lv]
            ref = max(order, key=lambda lv: (total.get(lv, -1.0), -order.index(lv)))
        else:
            order = plevels
            mapping, ref = _merge_nominal(plevels, claims, min_claims, other)
        used = set(mapping.values())
        ordered = [ref] + [lv for lv in order if lv in used and lv not in (ref, other)]
        if other in used and other != ref:
            ordered.append(other)
        levels[name] = {
            "levels": ordered,
            "reference": ref,
            "rare": list(info["rare"]),
            "merge": mapping,
            "claims": claims,
        }
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
            merge = info.get("merge")
            known = set(info["levels"]) if merge is None else set(merge)  # policy schema
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
            if merge is not None:  # severity map: policy level -> estimable severity level
                mapped = mapped.map(merge)
            out[name] = pd.Categorical(mapped, categories=info["levels"])
    return pd.DataFrame(out, index=df.index)
