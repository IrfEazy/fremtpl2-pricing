"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = "configs/default.yaml"
REQUIRED_KEYS = ("seed", "paths", "download")
REQUIRED_PATHS = ("raw", "processed", "results", "reports")


class ConfigError(ValueError):
    """Raised when the configuration file is missing or malformed."""


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
    return cfg
