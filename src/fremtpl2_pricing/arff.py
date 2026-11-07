"""Minimal ARFF reader for the dense numeric/nominal/string files published on OpenML.

Limitations (acceptable for the freMTPL2 files, which contain none of these): backslash
escapes inside quoted values are not handled, and a bare ``?`` is always read as missing,
even inside a quoted string.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

_ATTR = re.compile(r"@attribute\s+(?:'([^']*)'|(\S+))\s+(.+)$", re.IGNORECASE)


def _nominal_values(spec: str) -> list[str]:
    return next(csv.reader([spec.strip()[1:-1]], quotechar="'", skipinitialspace=True))


def _to_int64(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series)
    if values.isna().any():
        raise ValueError(f"column {name}: missing values in an integer column")
    if not (values == values.round()).all():
        raise ValueError(f"column {name}: non-integral value in an integer column")
    return values.astype("int64")


def read_arff(path: str | Path, int_columns: Iterable[str] = ()) -> pd.DataFrame:
    """Read an ARFF file. ``integer`` attributes and ``int_columns`` become int64 or fail."""
    int_columns = set(int_columns)
    names: list[str] = []
    kinds: list[str | list[str]] = []
    data_lines: list[str] = []
    in_data = False
    with Path(path).open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\r\n")
            if in_data:
                if line and not line.startswith("%"):
                    data_lines.append(line)
            elif line.lower().startswith("@attribute"):
                m = _ATTR.match(line)
                if not m:
                    raise ValueError(f"cannot parse ARFF attribute line: {line!r}")
                names.append(m.group(1) or m.group(2))
                spec = m.group(3).strip()
                kinds.append(_nominal_values(spec) if spec.startswith("{") else spec.lower())
            elif line.lower().startswith("@data"):
                in_data = True
    rows = list(csv.reader(data_lines, quotechar="'", skipinitialspace=True))
    if any(len(r) != len(names) for r in rows):
        raise ValueError(f"{path}: data row width differs from attribute count")
    df = pd.DataFrame(rows, columns=names).replace("?", None)
    unknown = int_columns - set(names)
    if unknown:
        raise ValueError(f"{path}: int_columns not in file: {sorted(unknown)}")
    for name, kind in zip(names, kinds, strict=True):
        if isinstance(kind, list):
            df[name] = pd.Categorical(df[name], categories=kind)
        elif kind == "integer" or (name in int_columns and kind in ("numeric", "real")):
            df[name] = _to_int64(df[name], name)
        elif kind in ("numeric", "real"):
            df[name] = pd.to_numeric(df[name])
        elif kind != "string":
            raise ValueError(f"unsupported ARFF type {kind!r} for column {name}")
    return df
