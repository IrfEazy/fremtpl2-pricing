"""SHA-256 manifest helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path, **extra: Any) -> dict[str, Any]:
    path = Path(path)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path), **extra}


def verify_record(directory: str | Path, record: dict[str, Any]) -> str | None:
    """Return None if the file matches the record, else a description of the problem."""
    path = Path(directory) / record["path"]
    if not path.is_file():
        return f"missing file: {path}"
    size = path.stat().st_size
    if size != record["bytes"]:
        return f"size mismatch for {path}: expected {record['bytes']}, got {size}"
    actual = sha256_file(path)
    if actual != record["sha256"]:
        return f"sha256 mismatch for {path}: expected {record['sha256']}, got {actual}"
    return None


def is_derived(record: dict[str, Any]) -> bool:
    return "derived_from" in record


def parquet_summary(path: str | Path) -> dict[str, Any]:
    """Rows, columns and Arrow schema of a parquet file (read from its metadata only)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    return {
        "rows": pf.metadata.num_rows,
        "columns": len(schema),
        "schema": [[f.name, str(f.type)] for f in schema],
    }


def verify_derived(directory: str | Path, record: dict[str, Any]) -> str | None:
    """Check a derived parquet record by rows/columns/schema (never by hash)."""
    path = Path(directory) / record["path"]
    if not path.is_file():
        return f"missing file: {path}"
    try:
        actual = parquet_summary(path)
    except Exception as exc:  # unreadable parquet is a mismatch, not a crash
        return f"unreadable parquet {path}: {exc}"
    for key in ("rows", "columns", "schema"):
        if actual[key] != record.get(key):
            return f"{key} mismatch for {path}"
    return None


def verify_manifest(directory: str | Path, manifest: dict[str, Any]) -> list[str]:
    """Hash-check authoritative (non-derived) files only."""
    problems = []
    for dataset in manifest.get("datasets", []):
        for record in dataset.get("files", []):
            if is_derived(record):
                continue
            problem = verify_record(directory, record)
            if problem:
                problems.append(problem)
    return problems


def read_manifest(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
