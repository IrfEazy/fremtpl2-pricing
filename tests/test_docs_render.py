"""`fremtpl2 docs`: model report and README / model card blocks, rendered from pipeline files.

Runs on the synthetic project of ``test_freeze_evaluate`` (real `train` / `gbm` / `interpret` /
`freeze` / `evaluate` stages on synthetic data, its own git repository); the real project is never
touched. The inputs that stage cannot produce on synthetic data (audit, split and target summaries,
CV sensitivities) are copied from the tracked ``results/`` of this repository.
"""

import json
import shutil
from pathlib import Path

import pytest

from fremtpl2_pricing import cli, docs_render, evaluate, freeze, interpret
from fremtpl2_pricing.config import (
    ConfigError,
    load_config,
    load_docs_config,
    with_docs,
)
from fremtpl2_pricing.report import ReportError
from test_freeze_evaluate import CONFIG, _commit, template  # noqa: F401  (fixture)
from test_interpret_report import _with_sections

REPO = Path(__file__).resolve().parent.parent
BORROWED = ["audit.json", "split_summary.json", "targets_summary.json", "cv_sensitivities.json"]
README = (
    "# Title\r\nintro é\n\n<!-- docs:results:begin -->\nold\n<!-- docs:results:end -->\n"
    "middle\n<!-- docs:reproduction:begin -->\n<!-- docs:reproduction:end -->\n"
    "<!-- docs:drivers:begin --><!-- docs:drivers:end -->\nlast line without newline"
)
CARD = "card\n<!-- docs:metrics:begin -->\n<!-- docs:metrics:end -->\n"


@pytest.fixture(scope="module")
def docs_root(template, tmp_path_factory):  # noqa: F811
    root = tmp_path_factory.mktemp("docs") / "p"
    shutil.copytree(template, root, symlinks=True)
    mp = pytest.MonkeyPatch()
    mp.chdir(root)
    try:
        cfg = _with_sections(load_config(CONFIG))
        interpret.run_interpret(cfg, root=root)
        _commit(root, "interpret")
        freeze.run_freeze(cfg, CONFIG)
        _commit(root, "freeze")
        evaluate.run_evaluate(cfg, CONFIG)
        for name in BORROWED:
            shutil.copy(REPO / "results" / name, root / "results" / name)
        (root / "README.md").write_bytes(README.encode("utf-8"))
        (root / "MODEL_CARD.md").write_bytes(CARD.encode("utf-8"))
    finally:
        mp.undo()
    return root


@pytest.fixture
def proj(docs_root, tmp_path, monkeypatch):
    root = tmp_path / "p"
    shutil.copytree(docs_root, root, symlinks=True)
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def dcfg(proj):
    cfg = _with_sections(load_config(CONFIG))
    return with_docs(cfg, load_docs_config(REPO / "configs/docs.yaml"))


def _files(root: Path) -> dict[str, bytes]:
    skip = {".git", "__pycache__"}
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and not skip & set(p.parts)
    }


def _blocks(text: str, names: list[str]) -> str:
    """The text with every marker block emptied: what must survive byte for byte."""
    for n in names:
        a, b = f"<!-- docs:{n}:begin -->", f"<!-- docs:{n}:end -->"
        text = text[: text.index(a) + len(a)] + text[text.index(b) :]
    return text


# --------------------------------------------------------------------------- markers


def test_replace_block_keeps_everything_outside_the_markers():
    mk = {"begin": "<!-- b:{name} -->", "end": "<!-- e:{name} -->"}
    text = "pre\r\n<!-- b:x -->\nold\n<!-- e:x -->\r\npost é"
    out = docs_render.replace_block(text, "x", "new\nbody\n", mk, "f")
    assert out == "pre\r\n<!-- b:x -->\nnew\nbody\n<!-- e:x -->\r\npost é"
    assert docs_render.replace_block(out, "x", "new\nbody", mk, "f") == out  # idempotent


@pytest.mark.parametrize(
    "text",
    [
        "nothing",
        "<!-- b:x -->\nonly begin",
        "<!-- e:x --><!-- b:x -->",
        "<!-- b:x --><!-- e:x --><!-- b:x -->",
    ],
)
def test_replace_block_missing_misordered_or_duplicate_marker_errors(text):
    mk = {"begin": "<!-- b:{name} -->", "end": "<!-- e:{name} -->"}
    with pytest.raises(docs_render.DocsError, match="marker"):
        docs_render.replace_block(text, "x", "body", mk, "f")


def test_missing_marker_in_a_target_errors_and_writes_nothing(proj, dcfg):
    (proj / "README.md").write_text("no markers here\n")
    before = _files(proj)
    with pytest.raises(docs_render.DocsError, match="README.md: needs exactly one"):
        docs_render.run_docs(dcfg, root=proj)
    assert _files(proj) == before


def test_unknown_block_errors(proj, dcfg):
    dcfg["docs"]["blocks"]["README.md"].append("nope")
    with pytest.raises(docs_render.DocsError, match="unknown block `nope`"):
        docs_render.run_docs(dcfg, root=proj)


def test_text_outside_markers_is_preserved_byte_for_byte(proj, dcfg):
    docs_render.run_docs(dcfg, root=proj)
    out = (proj / "README.md").read_bytes().decode("utf-8")
    names = ["results", "reproduction", "drivers"]
    assert _blocks(out, names) == _blocks(README, names)
    assert out.startswith("# Title\r\nintro é\n") and out.endswith("last line without newline")
    assert (proj / "MODEL_CARD.md").read_text().startswith("card\n")


# --------------------------------------------------------------------------- determinism, check


def test_two_runs_are_byte_identical_and_check_passes(proj, dcfg):
    first = docs_render.run_docs(dcfg, root=proj)
    assert sorted(first["written"]) == ["MODEL_CARD.md", "README.md", "reports/model_report.md"]
    snap = _files(proj)
    second = docs_render.run_docs(dcfg, root=proj)
    assert second["written"] == [] and _files(proj) == snap
    assert docs_render.run_docs(dcfg, root=proj, check=True)["drift"] == []


def test_check_detects_drift_and_writes_nothing(proj, dcfg):
    docs_render.run_docs(dcfg, root=proj)
    card = proj / "MODEL_CARD.md"
    card.write_text(
        card.read_text().replace("<!-- docs:metrics:end -->", "edited\n<!-- docs:metrics:end -->")
    )
    (proj / "reports/model_report.md").unlink()
    before = _files(proj)
    res = docs_render.run_docs(dcfg, root=proj, check=True)
    assert sorted(res["drift"]) == ["MODEL_CARD.md", "reports/model_report.md"]
    assert res["written"] == [] and _files(proj) == before


# --------------------------------------------------------------------------- numbers


def test_numbers_equal_the_input_values(proj, dcfg):
    docs_render.run_docs(dcfg, root=proj)
    m = json.loads((proj / "results/metrics.json").read_text())
    g = m["bootstrap"]["group"]
    d, gd, iv = g["delta"], g["gini_difference"], g["intervals"]
    readme = (proj / "README.md").read_text()
    card = (proj / "MODEL_CARD.md").read_text()
    report = (proj / "reports/model_report.md").read_text()
    for text in (readme, card, report):
        assert f"{d['point']:.5f}" in text and f"{d['lower']:.5f}" in text
        assert f"**{d['decision']}**" in text
        assert f"{gd['point']:.5f}" in text and f"{gd['upper']:.5f}" in text
        for cid, c in m["candidates"].items():
            assert f"`{cid}`" in text
            assert f"{c['primary_loss']:.3f}" in text
            assert f"{c['calibration_totals']['ratio']:.4f}" in text
            lo, hi = iv[f"loss:{cid}"]["lower"], iv[f"loss:{cid}"]["upper"]
            assert f"[{lo:.3f}, {hi:.3f}]" in text
    gbm_gini = iv["gini_normalized:gbm"]
    assert f"[{gbm_gini['lower']:.4f}, {gbm_gini['upper']:.4f}]" in readme
    assert f"{m['rows']['policies']:,}".replace(",", " ") in readme
    # model report: counts, selections and hyperparameters read from their files
    audit = json.loads((proj / "results/audit.json").read_text())
    gbm = json.loads((proj / "results/cv_gbm.json").read_text())
    interp = json.loads((proj / "results/interpretation.json").read_text())
    assert f"{audit['reconciliation']['orphan_rows']}" in report
    for comp in gbm["components"].values():
        assert f"num_leaves={comp['chosen_hp']['num_leaves']}" in report
    top = interp["shap"]["frequency"]["ranking"][0]
    assert f"{top} ({interp['shap']['frequency']['mean_abs'][top]:.4f}" in readme
    for heading in (
        "## 1. Data",
        "## 5. Selections",
        "## 7. Monotone",
        "## 10. The single test",
        "## 11. Limits",
    ):
        assert heading in report
    # the loaded premium is reported separately from the primary loss
    assert f"{m['loaded_premium']['loading_per_exposure_year']:.2f}" in report


# --------------------------------------------------------------------------- reproduction


def test_missing_reproduction_renders_the_placeholder(proj, dcfg):
    assert not (proj / "results/reproduction.json").exists()
    docs_render.run_docs(dcfg, root=proj)
    assert "not yet recorded" in (proj / "README.md").read_text()


def test_reproduction_block_reads_the_record(proj, dcfg):
    rec = {
        "tested_commit": "0123456789abcdef", "verdict": "EXACT_PASS", "total_seconds": 90.0,
        "platform": "Linux-x", "machine": "x86_64", "cpu_count": 8, "python": "3.13.1",
        "steps": [{"name": "train", "seconds": 30.0}, {"name": "gbm", "seconds": 60.0}],
    }  # fmt: skip
    (proj / "results/reproduction.json").write_text(json.dumps(rec))
    docs_render.run_docs(dcfg, root=proj)
    text = (proj / "README.md").read_text()
    assert "**EXACT_PASS**" in text and "`01234567`" in text and "0123456789" not in text
    assert "total 1.5 minutes" in text and "| train | 0.5 |" in text and "| gbm | 1.0 |" in text
    assert "Linux-x (x86_64, 8 CPUs), Python 3.13.1" in text
    assert "not yet recorded" not in text


# --------------------------------------------------------------------------- verification, config


def test_tampered_attempt_is_refused_and_nothing_is_written(proj, dcfg):
    tables = proj / "results/test/attempt-1/tables.json"
    tables.write_text(tables.read_text().replace("0.", "1.", 1))
    before = _files(proj)
    with pytest.raises(ReportError, match="sha256 differs"):
        docs_render.run_docs(dcfg, root=proj)
    assert _files(proj) == before


def test_missing_input_file_errors(proj, dcfg):
    (proj / "results/interpretation.json").unlink()
    with pytest.raises(docs_render.DocsError, match="interpretation.json"):
        docs_render.run_docs(dcfg, root=proj)


def test_docs_config_conflict_and_missing_file_raise():
    dcfg = load_docs_config(REPO / "configs/docs.yaml")
    with pytest.raises(ConfigError, match="already defines docs"):
        with_docs({"docs": {}}, dcfg)
    with pytest.raises(ConfigError, match="not found"):
        load_docs_config(REPO / "configs/none.yaml")


def test_default_config_is_untouched_by_docs():
    assert "docs" not in load_config(REPO / "configs/default.yaml")


def test_cli_wires_docs_and_check_exit_code(monkeypatch):
    calls = []

    def fake(cfg, check=False):
        calls.append(check)
        return {"drift": ["README.md"] if check else [], "written": []}

    monkeypatch.setattr(cli, "run_docs", fake)
    assert cli.main(["docs"]) == 0
    assert cli.main(["docs", "--check"]) == 1
    assert calls == [False, True]
