"""Download freMTPL2 datasets from OpenML with a SHA-256 manifest."""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path
from typing import Any

import requests

from .arff import read_arff
from .manifest import (
    file_record,
    is_derived,
    parquet_summary,
    read_manifest,
    verify_derived,
    verify_manifest,
    write_manifest,
)

MANIFEST_NAME = "manifest.json"


class DownloadError(RuntimeError):
    """Raised when a dataset cannot be fetched or verified."""


def fetch_description(session: requests.Session, cfg: dict[str, Any], openml_id: int) -> dict:
    dl = cfg["download"]
    url = dl["description_url"].format(id=openml_id)
    resp = session.get(url, timeout=dl["timeout_seconds"])
    resp.raise_for_status()
    return resp.json()["data_set_description"]


def check_description(desc: dict, spec: dict) -> None:
    if str(desc.get("id")) != str(spec["openml_id"]):
        raise DownloadError(
            f"OpenML id mismatch: expected {spec['openml_id']}, got {desc.get('id')}"
        )
    expected_version = spec.get("expected_version")
    if expected_version is not None and str(desc.get("version")) != str(expected_version):
        raise DownloadError(
            f"OpenML dataset {spec['openml_id']} is version {desc.get('version')!r}, "
            f"expected {expected_version!r}"
        )
    if desc.get("name") != spec["expected_name"]:
        raise DownloadError(
            f"OpenML dataset {spec['openml_id']} is named {desc.get('name')!r}, "
            f"expected {spec['expected_name']!r}"
        )


def download_file(session: requests.Session, url: str, dest: Path, timeout: int) -> None:
    if not url.startswith("https://"):
        raise DownloadError(f"refusing non-HTTPS url: {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with session.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
    tmp.replace(dest)


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet(raw: Path, arff_name: str, parquet_name: str, spec: dict) -> dict[str, Any]:
    """Derive the parquet copy from a verified ARFF file and return its manifest record."""
    frame = read_arff(raw / arff_name, int_columns=spec.get("int_columns", ()))
    parquet = raw / parquet_name
    frame.to_parquet(parquet, index=False)
    return {
        "path": parquet_name,
        "format": "parquet",
        "derived_from": arff_name,
        **parquet_summary(parquet),
    }


def manifest_matches_config(existing: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """True if the manifest covers exactly the configured datasets (id, key, name, version)."""
    have = {d["openml_id"]: d for d in existing.get("datasets", [])}
    specs = cfg["download"]["datasets"]
    if set(have) != {s["openml_id"] for s in specs} or len(have) != len(specs):
        return False
    for spec in specs:
        entry = have[spec["openml_id"]]
        if entry.get("key") != spec["key"] or entry.get("name") != spec["expected_name"]:
            return False
        version = spec.get("expected_version")
        if version is not None and str(entry.get("version")) != str(version):
            return False
    return True


def _refresh_derived(raw: Path, existing: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Regenerate missing or mismatching parquet copies from the verified ARFF files."""
    specs = {s["openml_id"]: s for s in cfg["download"]["datasets"]}
    changed = False
    for dataset in existing["datasets"]:
        for i, record in enumerate(dataset["files"]):
            if is_derived(record) and (problem := verify_derived(raw, record)):
                print(f"regenerating {record['path']} from {record['derived_from']}: {problem}")
                dataset["files"][i] = write_parquet(
                    raw, record["derived_from"], record["path"], specs[dataset["openml_id"]]
                )
                changed = True
    return changed


def run_download(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    raw = Path(cfg["paths"]["raw"])
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / MANIFEST_NAME
    existing = read_manifest(manifest_path)
    if existing and not force and manifest_matches_config(existing, cfg):
        problems = verify_manifest(raw, existing)
        if problems:
            raise DownloadError(
                "existing raw files do not match manifest (use --force to re-download):\n"
                + "\n".join(problems)
            )
        if _refresh_derived(raw, existing, cfg):
            write_manifest(manifest_path, existing)
        print(
            f"verified {sum(len(d['files']) for d in existing['datasets'])} files against manifest"
        )
        return existing

    timeout = cfg["download"]["timeout_seconds"]
    datasets = []
    with requests.Session() as session:
        for spec in cfg["download"]["datasets"]:
            desc = fetch_description(session, cfg, spec["openml_id"])
            check_description(desc, spec)
            name = desc["name"]
            url = desc["url"]
            dest = raw / f"{name}.arff"
            print(f"downloading {url}")
            download_file(session, url, dest, timeout)
            expected_md5 = desc.get("md5_checksum")
            if not expected_md5:
                print(f"warning: OpenML gives no md5_checksum for {dest.name}; md5 check skipped")
            elif _md5(dest) != expected_md5:
                raise DownloadError(f"md5 mismatch for {dest} against OpenML checksum")
            files = [file_record(dest, format="arff", source_url=url, openml_md5=expected_md5)]
            files.append(write_parquet(raw, dest.name, f"{name}.parquet", spec))
            datasets.append(
                {
                    "key": spec["key"],
                    "openml_id": int(desc["id"]),
                    "name": name,
                    "version": desc.get("version"),
                    "licence": desc.get("licence"),
                    "original_data_url": desc.get("original_data_url"),
                    "description_url": cfg["download"]["description_url"].format(
                        id=spec["openml_id"]
                    ),
                    "retrieved": dt.date.today().isoformat(),
                    "files": files,
                }
            )
    manifest = {"datasets": datasets}
    write_manifest(manifest_path, manifest)
    return manifest
