"""Load the raw parquet files after verifying the SHA-256 manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .manifest import read_manifest, verify_manifest


class DataError(RuntimeError):
    """Raised when raw data are missing or fail verification."""


def load_raw(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (freq, sev) after hash-checking the raw ARFF files against the manifest."""
    raw = Path(cfg["paths"]["raw"])
    manifest = read_manifest(raw / "manifest.json")
    if manifest is None:
        raise DataError(f"no manifest in {raw}; run `fremtpl2 download` first")
    problems = verify_manifest(raw, manifest)
    if problems:
        raise DataError("raw data failed verification: " + "; ".join(problems))
    try:
        freq = pd.read_parquet(raw / "freMTPL2freq.parquet")
        sev = pd.read_parquet(raw / "freMTPL2sev.parquet")
    except (OSError, ValueError) as exc:
        raise DataError(f"cannot read parquet in {raw}: {exc}") from exc
    return freq, sev
