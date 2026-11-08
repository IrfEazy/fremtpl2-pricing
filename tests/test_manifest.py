import hashlib
import json
from pathlib import Path

from fremtpl2_pricing.manifest import (
    file_record,
    read_manifest,
    sha256_file,
    verify_manifest,
    verify_record,
    write_manifest,
)


def test_sha256_matches_hashlib(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    assert sha256_file(f) == hashlib.sha256(b"hello").hexdigest()


def test_record_verify_and_tamper(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    rec = file_record(f, format="x")
    assert rec["bytes"] == 5 and rec["path"] == "a.bin"
    assert verify_record(tmp_path, rec) is None
    f.write_bytes(b"hellO")
    assert "sha256 mismatch" in verify_record(tmp_path, rec)
    f.write_bytes(b"hello!")
    assert "size mismatch" in verify_record(tmp_path, rec)
    f.unlink()
    assert "missing" in verify_record(tmp_path, rec)


def test_manifest_roundtrip(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"data")
    manifest = {"datasets": [{"files": [file_record(f)]}]}
    write_manifest(tmp_path / "manifest.json", manifest)
    loaded = read_manifest(tmp_path / "manifest.json")
    assert loaded == manifest
    assert verify_manifest(tmp_path, loaded) == []
    f.write_bytes(b"dat4")
    assert len(verify_manifest(tmp_path, loaded)) == 1
    assert read_manifest(tmp_path / "none.json") is None


def test_default_config_is_the_frozen_config():
    """Guards drift: the main config stays byte-identical to the hash in the freeze manifest
    (post-test settings live in configs/reproduce.yaml)."""
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "results/freeze_manifest.json").read_text(encoding="utf-8"))
    entries = [e for e in manifest["files"] if e["role"] == "config"]
    assert [e["path"] for e in entries] == ["configs/default.yaml"]
    data = (root / entries[0]["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entries[0]["sha256"]
    assert len(data) == entries[0]["bytes"]
