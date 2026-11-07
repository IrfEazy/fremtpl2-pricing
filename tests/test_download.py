import copy

import pandas as pd
import pytest

from fremtpl2_pricing.download import (
    DownloadError,
    check_description,
    manifest_matches_config,
    run_download,
    write_parquet,
)
from fremtpl2_pricing.manifest import (
    file_record,
    parquet_summary,
    verify_derived,
    verify_manifest,
    write_manifest,
)

ARFF = "@relation t\n@attribute IDpol numeric\n@attribute x numeric\n@data\n1,2.5\n2,3.5\n"
SPEC = {
    "key": "freq",
    "openml_id": 1,
    "expected_name": "t",
    "expected_version": "1",
    "int_columns": ["IDpol"],
}


def _cfg(tmp_path):
    return {
        "paths": {"raw": str(tmp_path)},
        "download": {"timeout_seconds": 1, "description_url": "x/{id}", "datasets": [SPEC]},
    }


def _setup(tmp_path):
    (tmp_path / "t.arff").write_text(ARFF)
    pq = write_parquet(tmp_path, "t.arff", "t.parquet", SPEC)
    manifest = {
        "datasets": [
            {
                "key": "freq",
                "openml_id": 1,
                "name": "t",
                "version": "1",
                "files": [file_record(tmp_path / "t.arff", format="arff"), pq],
            }
        ]
    }
    write_manifest(tmp_path / "manifest.json", manifest)
    return manifest


def test_parquet_record_has_no_hash_and_verifies(tmp_path):
    manifest = _setup(tmp_path)
    pq = manifest["datasets"][0]["files"][1]
    assert "sha256" not in pq and pq["rows"] == 2 and pq["columns"] == 2
    assert pq["schema"][0] == ["IDpol", "int64"]
    assert verify_derived(tmp_path, pq) is None
    assert verify_manifest(tmp_path, manifest) == []


def test_changed_parquet_bytes_do_not_fail_verification(tmp_path):
    manifest = _setup(tmp_path)
    pq = tmp_path / "t.parquet"
    pd.read_parquet(pq).to_parquet(pq, index=False, compression="gzip")
    assert verify_manifest(tmp_path, manifest) == []
    assert verify_derived(tmp_path, manifest["datasets"][0]["files"][1]) is None


def test_derived_mismatch_detected(tmp_path):
    manifest = _setup(tmp_path)
    rec = manifest["datasets"][0]["files"][1]
    pd.DataFrame({"IDpol": [1]}).to_parquet(tmp_path / "t.parquet", index=False)
    assert "rows mismatch" in verify_derived(tmp_path, rec)
    (tmp_path / "t.parquet").unlink()
    assert "missing" in verify_derived(tmp_path, rec)
    regen = write_parquet(tmp_path, "t.arff", "t.parquet", SPEC)
    assert parquet_summary(tmp_path / "t.parquet")["rows"] == regen["rows"] == 2


def test_rerun_regenerates_missing_or_bad_parquet_offline(tmp_path):
    _setup(tmp_path)
    (tmp_path / "t.parquet").write_bytes(b"garbage")
    result = run_download(_cfg(tmp_path))
    assert verify_derived(tmp_path, result["datasets"][0]["files"][1]) is None
    (tmp_path / "t.parquet").unlink()
    run_download(_cfg(tmp_path))
    assert (tmp_path / "t.parquet").is_file()


def test_rerun_fails_when_arff_tampered(tmp_path):
    _setup(tmp_path)
    (tmp_path / "t.arff").write_text(ARFF + "3,4\n")
    with pytest.raises(DownloadError, match="do not match"):
        run_download(_cfg(tmp_path))


def test_manifest_matches_config_compares_name_key_version(tmp_path):
    manifest = _setup(tmp_path)
    cfg = _cfg(tmp_path)
    assert manifest_matches_config(manifest, cfg)
    for field, value in (("expected_name", "u"), ("key", "sev"), ("expected_version", "2")):
        bad = copy.deepcopy(cfg)
        bad["download"]["datasets"][0][field] = value
        assert not manifest_matches_config(manifest, bad)
    extra = copy.deepcopy(cfg)
    extra["download"]["datasets"].append({**SPEC, "openml_id": 2})
    assert not manifest_matches_config(manifest, extra)


def test_check_description_version():
    desc = {"id": "1", "name": "t", "version": "2"}
    with pytest.raises(DownloadError, match="version"):
        check_description(desc, SPEC)
    check_description({**desc, "version": "1"}, SPEC)
