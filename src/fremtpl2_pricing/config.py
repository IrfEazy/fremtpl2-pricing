"""YAML configuration loading."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = "configs/default.yaml"
# Post-test verification policy (`freeze --check`, `reproduce`); kept out of the frozen main config.
DEFAULT_REPRODUCE_CONFIG = "configs/reproduce.yaml"
REPRODUCE_KEYS = ("check", "reproduce")
REQUIRED_KEYS = ("seed", "paths", "download")
REQUIRED_PATHS = ("raw", "processed", "results", "reports")


class ConfigError(ValueError):
    """Raised when the configuration file is missing or malformed."""


def validate_models(models: dict[str, Any]) -> None:
    """Check the severity ladder and the L2 grid of ``models.glm`` (strictly positive penalties)."""
    glm_cfg = models.get("glm", {})
    grid = glm_cfg.get("l2_grid")
    if not isinstance(grid, list) or not grid:
        raise ConfigError("models.glm.l2_grid must be a non-empty list")
    for v in grid:
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) or v <= 0:
            raise ConfigError(f"models.glm.l2_grid must contain finite values > 0, got {v!r}")
    if len(set(grid)) != len(grid):
        raise ConfigError("models.glm.l2_grid must not contain duplicates")
    ladder = glm_cfg.get("severity_ladder")
    if not isinstance(ladder, list) or len(ladder) < 2:
        raise ConfigError("models.glm.severity_ladder must list at least 2 candidates")
    ids = [e.get("id") for e in ladder]
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        raise ConfigError("models.glm.severity_ladder ids must be unique non-empty strings")
    for e in ladder:
        if e.get("kind") not in ("constant", "glm", "glm_l2"):
            raise ConfigError(f"severity_ladder {e.get('id')!r}: unknown kind {e.get('kind')!r}")
        if e["kind"] != "constant" and not e.get("feature_set"):
            raise ConfigError(f"severity_ladder {e['id']!r}: feature_set is required")
    if sum(e["kind"] == "glm_l2" for e in ladder) > 1:
        raise ConfigError("severity_ladder: at most one glm_l2 candidate is supported")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ConfigError(f"config must be a mapping: {path}")
    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise ConfigError(f"missing config key '{key}' in {path}")
    for key in REQUIRED_PATHS:
        if key not in cfg["paths"]:
            raise ConfigError(f"missing config key 'paths.{key}' in {path}")
    if not isinstance(cfg["seed"], int):
        raise ConfigError("seed must be an integer")
    if not cfg["download"].get("datasets"):
        raise ConfigError("download.datasets must list at least one dataset")
    if "models" in cfg:
        validate_models(cfg["models"])
    return cfg


def load_reproduce_config(path: str | Path = DEFAULT_REPRODUCE_CONFIG) -> dict[str, Any]:
    """Load the verification policy: top-level ``check`` (``freeze --check``) and ``reproduce``."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"reproduce config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        rcfg = yaml.safe_load(fh)
    if not isinstance(rcfg, dict):
        raise ConfigError(f"reproduce config must be a mapping: {path}")
    for key in REPRODUCE_KEYS:
        if not isinstance(rcfg.get(key), dict):
            raise ConfigError(f"missing config mapping '{key}' in {path}")
    return rcfg


def with_reproduce(cfg: dict[str, Any], rcfg: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``cfg`` with ``freeze.check`` and ``reproduce`` taken from the reproduce config.

    Raises ``ConfigError`` if ``cfg`` already has either key (never silently overrides).
    """
    if "check" in (cfg.get("freeze") or {}):
        raise ConfigError(
            "main config already defines freeze.check; it belongs in the reproduce config"
        )
    if "reproduce" in cfg:
        raise ConfigError(
            "main config already defines reproduce; it belongs in the reproduce config"
        )
    out = copy.deepcopy(cfg)
    out.setdefault("freeze", {})["check"] = copy.deepcopy(rcfg["check"])
    out["reproduce"] = copy.deepcopy(rcfg["reproduce"])
    return out
