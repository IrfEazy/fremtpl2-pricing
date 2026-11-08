from pathlib import Path

import pytest
import yaml

from fremtpl2_pricing.config import (
    ConfigError,
    load_config,
    load_reproduce_config,
    with_reproduce,
)

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    ids = {d["openml_id"]: d["expected_name"] for d in cfg["download"]["datasets"]}
    assert ids == {41214: "freMTPL2freq", 41215: "freMTPL2sev"}
    assert isinstance(cfg["seed"], int)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_missing_key(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"seed": 1, "paths": {"raw": "x"}, "download": {}}))
    with pytest.raises(ConfigError):
        load_config(p)


@pytest.mark.parametrize("bad", [[0.0, 0.1], [-0.1], [], [float("nan")], [0.1, 0.1]])
def test_l2_grid_must_be_strictly_positive(bad):
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["models"]["glm"]["l2_grid"] = bad
    from fremtpl2_pricing.config import validate_models

    with pytest.raises(ConfigError):
        validate_models(cfg["models"])


def test_reproduce_config_is_separate_and_merged_on_a_copy():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert "check" not in cfg["freeze"] and "reproduce" not in cfg
    rcfg = load_reproduce_config(ROOT / "configs" / "reproduce.yaml")
    assert rcfg["reproduce"]["exact_keys"] == rcfg["check"]["exact_keys"]
    merged = with_reproduce(cfg, rcfg)
    assert merged["freeze"]["check"] == rcfg["check"] and merged["reproduce"] == rcfg["reproduce"]
    assert "check" not in cfg["freeze"] and "reproduce" not in cfg  # the input is untouched


def test_with_reproduce_rejects_conflicting_main_config():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    rcfg = load_reproduce_config(ROOT / "configs" / "reproduce.yaml")
    with pytest.raises(ConfigError, match="freeze.check"):
        with_reproduce({**cfg, "freeze": {**cfg["freeze"], "check": {}}}, rcfg)
    with pytest.raises(ConfigError, match="reproduce"):
        with_reproduce({**cfg, "reproduce": {}}, rcfg)


def test_reproduce_config_errors(tmp_path):
    with pytest.raises(ConfigError):
        load_reproduce_config(tmp_path / "nope.yaml")
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump({"check": {}}))
    with pytest.raises(ConfigError):
        load_reproduce_config(p)
