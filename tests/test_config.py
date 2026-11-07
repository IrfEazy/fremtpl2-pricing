from pathlib import Path

import pytest
import yaml

from fremtpl2_pricing.cli import main
from fremtpl2_pricing.config import ConfigError, load_config

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


@pytest.mark.parametrize("cmd", ["train", "evaluate", "report"])
def test_stubs_exit_2(cmd):
    assert main([cmd, "--config", str(ROOT / "configs" / "default.yaml")]) == 2
