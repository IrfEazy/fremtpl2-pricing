"""Report of the single test evaluation (``fremtpl2 report``).

Reads only ``results/metrics.json`` (the pointer written for the completed attempt) and the attempt
it points to (``results/test/attempt-<n>/``: ``metrics.json``, ``tables.json``, the bootstrap
replicates), plus the journal ``results/test_access.json`` to prove that the attempt is COMPLETE and
to list every attempt. Every file is checked against the sha256 recorded by the COMPLETE event, and
the pointer must equal the attempt's ``metrics.json``. If there is no completed attempt (or anything
disagrees) the command raises before it writes anything.

Renders ``reports/test_evaluation.md`` and figures (Lorenz curves of all candidates, lift charts,
double lift of the selected GBM against the selected GLM, calibration by decile and by factor, the
bootstrap distribution of Delta). No number is typed by hand; no model is loaded or refitted and no
data row is read. Wording follows protocol section 6: the primary contrast and the decision rule are
quoted, other pairs are descriptive, the Gini difference is secondary, intervals are conditional on
the frozen models, and the policy bootstrap is a labelled sensitivity.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from . import figures
from .evaluate import (
    OUTPUT_FILES,
    TEST_POPULATION,
    access_log,
    decision,
    read_events,
    real_sink,
    validate_events,
)
from .evaluate import _parse_journal as parse_journal
from .freeze import project_paths
from .manifest import sha256_file

PRIMARY_CONTRAST = (
    "Single primary contrast: Delta = L_GBM - L_GLM, the selected GBM and GLM pure-premium "
    "candidates (chosen in CV by section 4) on the primary loss. The Gini difference is "
    "secondary; all other pairs are descriptive."
)
DECISION_RULE = (
    "Decision: improvement is claimed iff the upper interval endpoint of Delta < 0; "
    "deterioration iff the lower endpoint > 0; otherwise no demonstrated difference."
)


JOURNAL_CHECK = (
    "the journal (results/test_access.json) is authoritative, the attempt files were checked "
    "against the COMPLETE hashes and the pointer against the COMPLETE event"
)


class ReportError(ValueError):
    """Raised when there is no completed attempt or the evaluation outputs are inconsistent."""


# --------------------------------------------------------------------------- loading


def _journal_events(sink: dict[str, Path]) -> tuple[list[dict[str, Any]], str]:
    """Journal events; checked against the git-ignored mirror when that exists (the rendered
    report never depends on the mirror being present)."""
    if not sink["state"].exists():
        raise ReportError(
            "no test-access journal (results/test_access.json): the test evaluation was not run "
            "(`fremtpl2 evaluate`); nothing was written"
        )
    if sink["mirror"].exists():
        return read_events(sink), JOURNAL_CHECK
    events = parse_journal(sink["state"])
    validate_events(events)
    return events, JOURNAL_CHECK


def load_completed(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    """The completed attempt, verified: journal COMPLETE, pointer, attempt files and hashes."""
    paths = project_paths(cfg, root)
    sink = real_sink(root, paths)
    events, _ = _journal_events(sink)
    done = [e for e in events if e["event"] == "COMPLETE"]
    if not done:
        listing = "; ".join(f"attempt {a['attempt']}: {a['status']}" for a in access_log(events))
        raise ReportError(
            "no completed attempt in results/test_access.json "
            f"({listing or 'no attempt'}); nothing was written"
        )
    complete = done[-1]
    pointer_path = sink["pointer"]
    if not pointer_path.is_file():
        raise ReportError(f"{paths['pointer']} is missing; `fremtpl2 evaluate --publish-only`")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    src = pointer.get("source", {})
    for key, want in (
        ("attempt_id", complete["attempt_id"]),
        ("attempt", complete["attempt"]),
        ("attempt_dir", complete["publish_dir"]),
        ("outputs_sha256", complete["outputs_sha256"]),
    ):
        if src.get(key) != want:
            raise ReportError(f"{paths['pointer']}: source.{key} differs from the COMPLETE event")
    adir = root / complete["publish_dir"]
    for name in OUTPUT_FILES:
        f = adir / name
        if not f.is_file():
            raise ReportError(f"{complete['publish_dir']}/{name} is missing")
        if sha256_file(f) != complete["staged"][name]:
            raise ReportError(f"{complete['publish_dir']}/{name}: sha256 differs from COMPLETE")
    metrics = json.loads((adir / "metrics.json").read_text(encoding="utf-8"))
    if {k: v for k, v in pointer.items() if k != "source"} != metrics:
        raise ReportError(f"{paths['pointer']} differs from {complete['publish_dir']}/metrics.json")
    if metrics["attempt_id"] != complete["attempt_id"] or metrics["population"] != TEST_POPULATION:
        raise ReportError("the completed attempt is not a test-set attempt of this journal")
    for unit in ("group", "policy"):
        d = metrics["bootstrap"][unit]["delta"]
        if d["decision"] != decision(d["lower"], d["upper"]):
            raise ReportError(f"bootstrap {unit}: the recorded decision contradicts the interval")
    return {
        "metrics": metrics,
        "tables": json.loads((adir / metrics["tables_file"]).read_text(encoding="utf-8")),
        "replicates": {
            u: pd.read_csv(adir / f"bootstrap_replicates_{u}.csv") for u in ("group", "policy")
        },
        "complete": complete,
        "attempts": access_log(events),
        "pointer_sha256": sha256_file(pointer_path),
    }


# --------------------------------------------------------------------------- figures


def make_figures(
    data: dict[str, Any], fig_dir: Path, fcfg: dict[str, Any], rcfg: dict[str, Any]
) -> dict[str, str]:
    """Draw every figure from the tables and replicates; returns stem -> file name."""
    m, t = data["metrics"], data["tables"]
    prefix = rcfg["outputs"]["figure_prefix"]
    cands = list(m["candidates_order"])
    files: dict[str, str] = {}

    def path(stem: str) -> Path:
        return fig_dir / f"{prefix}{stem}.png"

    series = [
        {
            "label": cid,
            "x": t["lorenz"][cid]["cum_exposure"],
            "y": t["lorenz"][cid]["cum_loss"],
            "style": "-",
        }
        for cid in cands
    ]
    files["lorenz"] = figures.line_panels(
        fcfg,
        path("lorenz"),
        [
            {
                "name": "Lorenz curves of the pure premium (test set)",
                "xlabel": "cumulative exposure share",
                "ylabel": "cumulative loss share",
                "series": series,
            }
        ],
        "Lorenz curves: all candidates",
        ncol=1,
        diagonal=True,
    )
    panels = []
    for cid in cands:
        rows = t["lift"][cid]["pure_premium"]
        x = [r["bin"] + 1 for r in rows]
        panels.append(
            {
                "name": cid,
                "xlabel": "predicted-rate decile (equal exposure)",
                "ylabel": "rate per exposure-year",
                "series": [
                    {"label": "observed", "x": x, "y": [r["observed_rate"] for r in rows]},
                    {
                        "label": "predicted",
                        "x": x,
                        "y": [r["predicted_rate"] for r in rows],
                        "style": "s--",
                    },
                ],
            }
        )
    files["lift"] = figures.line_panels(
        fcfg, path("lift"), panels, "Lift charts of the pure premium (test set)", ncol=3
    )
    dl = t["double_lift"]
    rows = dl["table"]
    x = [r["bin"] + 1 for r in rows]
    files["double_lift"] = figures.line_panels(
        fcfg,
        path("double_lift"),
        [
            {
                "name": f"Double lift: {dl['a']} vs {dl['b']}",
                "xlabel": "decile of predicted ratio GBM / GLM",
                "ylabel": "observed / predicted",
                "hline": 1.0,
                "series": [
                    {"label": f"observed / {dl['a']}", "x": x, "y": [r["ratio_a"] for r in rows]},
                    {
                        "label": f"observed / {dl['b']}",
                        "x": x,
                        "y": [r["ratio_b"] for r in rows],
                        "style": "s--",
                    },
                ],
            }
        ],
        "Double lift: selected GBM against selected GLM",
        ncol=1,
    )
    files["calibration_decile"] = figures.line_panels(
        fcfg,
        path("calibration_decile"),
        [
            {
                "name": "Calibration by predicted-rate decile",
                "xlabel": "decile (equal exposure)",
                "ylabel": "observed / predicted",
                "hline": 1.0,
                "series": [
                    {
                        "label": cid,
                        "x": [r["bin"] + 1 for r in t["lift"][cid]["pure_premium"]],
                        "y": [r["ratio"] for r in t["lift"][cid]["pure_premium"]],
                    }
                    for cid in cands
                ],
            }
        ],
        "Calibration by decile: all candidates",
        ncol=1,
    )
    factors = list(t["calibration_by_factor"][cands[0]])
    panels = []
    for f in factors:
        order = [r["level"] for r in t["calibration_by_factor"][cands[0]][f]["pure_premium"]]
        panels.append(
            {
                "name": f,
                "xlabel": f,
                "ylabel": "observed / predicted",
                "hline": 1.0,
                "xticklabels": order,
                "series": [
                    {
                        "label": cid,
                        "x": list(range(len(order))),
                        "y": [
                            r["ratio"] for r in t["calibration_by_factor"][cid][f]["pure_premium"]
                        ],
                    }
                    for cid in cands
                ],
            }
        )
    files["calibration_factor"] = figures.line_panels(
        fcfg, path("calibration_factor"), panels, "Calibration by factor: all candidates", ncol=2
    )
    for unit, stem in (("group", "bootstrap_delta"), ("policy", "bootstrap_delta_policy")):
        d = m["bootstrap"][unit]["delta"]
        files[stem] = figures.histogram(
            fcfg,
            path(stem),
            data["replicates"][unit]["delta"].to_numpy(),
            int(rcfg["bootstrap_histogram_bins"]),
            {"point": d["point"], "lower": d["lower"], "upper": d["upper"], "zero": 0.0},
            m["bootstrap"][unit]["label"],
            "Delta = L_GBM - L_GLM",
        )
    return files


# --------------------------------------------------------------------------- markdown


def _f(x: Any, nd: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _row(cells: list[Any]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    return [_row(header), _row(["---"] * len(header)), *(_row(r) for r in rows), ""]


def _boot_rows(b: dict[str, Any]) -> list[list[Any]]:
    return [
        [k, _f(v["point"], 5), _f(v["lower"], 5), _f(v["upper"], 5)]
        for k, v in b["intervals"].items()
    ]


def render_report(
    data: dict[str, Any], files: dict[str, str], cfg: dict[str, Any], fig_dir: str
) -> str:
    m, c = data["metrics"], data["complete"]
    g, p = m["bootstrap"]["group"], m["bootstrap"]["policy"]
    d = g["delta"]
    gbm_id, glm_id = m["selected_gbm_id"], m["selected_glm_id"]
    ref = cfg["models"]["tweedie"]["reference_power"]
    gd, lp = g["gini_difference"], m["loaded_premium"]
    rows, gr = m["rows"], m["groups"]
    out = [
        "# Test evaluation report",
        "",
        "Generated by `fremtpl2 report` from `results/metrics.json` and the completed attempt it "
        f"points to (`{c['publish_dir']}`, attempt {m['attempt']}, id `{m['attempt_id']}`). The "
        "attempt files were checked against the sha256 values of the journal COMPLETE event "
        f"(`{c['outputs_sha256'][:16]}...`); the pointer file has sha256 "
        f"`{data['pointer_sha256'][:16]}...`; {JOURNAL_CHECK}. No number in this report "
        "is typed by hand.",
        "",
        f"Population: the primary grouped test fold ({rows['policies']} policies, "
        f"{_f(rows['exposure'], 1)} exposure-years, {rows['claims']} claims on "
        f"{rows['claim_policies']} policies; {gr['groups']} rating-profile groups). Frozen "
        f"manifest `{m['manifest_sha256'][:16]}...`, evaluated commit `{m['commit'][:12]}`, "
        f"training commit `{m['training_commit'][:12]}`. Models are fitted on the training set "
        "only and are frozen; this report conditions on them.",
        "",
        "## 1. Primary contrast and decision (protocol section 6)",
        "",
        PRIMARY_CONTRAST,
        "",
        f"Selected GBM: `{gbm_id}`. Selected GLM: `{glm_id}`. Primary loss L: exposure-weighted "
        f"Tweedie deviance of S_cap / Exposure at power {ref} (`models.tweedie.reference_power`), "
        "per exposure-year.",
        "",
        DECISION_RULE,
        "",
    ]
    out += _table(
        ["quantity", "point", "lower", "upper", "level", "replicates", "units"],
        [
            [
                "Delta = L_GBM - L_GLM",
                _f(d["point"], 5),
                _f(d["lower"], 5),
                _f(d["upper"], 5),
                g["level"],
                g["replicates"],
                f"{g['n_units']} {g['unit_column']}",
            ]
        ],
    )
    out += [
        f"**Result: {d['decision']}.** The upper endpoint of the Delta interval is "
        f"{_f(d['upper'], 5)} and the lower endpoint is {_f(d['lower'], 5)}.",
        "",
        "The interval is a percentile interval of a paired bootstrap that resamples whole "
        f"rating-profile groups ({g['quantile_method']} quantiles, seed {g['seed']}); it is "
        "conditional on the frozen fitted models. Retraining, tuning and selection variability "
        "are not included, and treating profiles as independent units is an assumption. The "
        "inference concerns held-out rating profiles of this dataset only.",
        "",
        "Secondary: normalised Gini difference "
        f"(`{gd['definition']}`): point "
        f"{_f(gd['point'], 5)}, interval [{_f(gd['lower'], 5)}, {_f(gd['upper'], 5)}]. "
        "It is secondary and carries no decision rule.",
        "",
        f"![bootstrap distribution of Delta]({fig_dir}/{files['bootstrap_delta']})",
        "",
        "## 2. All prespecified candidates (descriptive)",
        "",
        "Every prespecified candidate is reported regardless of outcome. Pairs other than the "
        "primary contrast are descriptive only: no interval or test is attached to them.",
        "",
    ]
    body = []
    for cid, cm in m["candidates"].items():
        tag = " (selected GBM)" if cid == gbm_id else " (selected GLM)" if cid == glm_id else ""
        cal = cm["calibration_totals"]
        body.append(
            [
                cid + tag,
                _f(cm["primary_loss"], 4),
                _f(cm["tweedie_deviance_per_exposure"], 4),
                _f(cm["tweedie_power"], 2),
                _f(cm["poisson_deviance_per_exposure"], 4),
                _f(cm["gini_normalized"], 4),
                _f(cal["ratio"], 4),
            ]
        )
    out += _table(
        [
            "candidate",
            "primary loss L",
            "Tweedie deviance / exposure (own power)",
            "power",
            "Poisson deviance / exposure",
            "normalised Gini",
            "observed / predicted total",
        ],
        body,
    )
    comp = []
    for cid, cm in m["candidates"].items():
        fr, sv = cm.get("frequency"), cm.get("severity")
        if fr is None and sv is None:
            continue
        comp.append(
            [
                cid,
                _f(fr["poisson_deviance_per_exposure"], 4) if fr else "-",
                _f(fr["gini_normalized"], 4) if fr else "-",
                _f(fr["calibration_totals"]["ratio"], 4) if fr else "-",
                _f(sv["gamma_deviance_per_claim"], 4) if sv else "-",
                _f(sv["calibration_totals"]["ratio"], 4) if sv else "-",
            ]
        )
    out += [
        "Components of the frequency x severity candidates (descriptive):",
        "",
        *_table(
            [
                "candidate",
                "frequency Poisson deviance / exposure",
                "frequency normalised Gini",
                "frequency observed / predicted",
                "severity Gamma deviance / claim",
                "severity observed / predicted",
            ],
            comp,
        ),
        f"![Lorenz curves]({fig_dir}/{files['lorenz']})",
        "",
        f"![lift charts]({fig_dir}/{files['lift']})",
        "",
        "## 3. Double lift: selected GBM against selected GLM",
        "",
        f"![double lift]({fig_dir}/{files['double_lift']})",
        "",
    ]
    dl = data["tables"]["double_lift"]["table"]
    out += _table(
        [
            "decile",
            "exposure",
            "observed",
            f"predicted {gbm_id}",
            f"predicted {glm_id}",
            "observed / GBM",
            "observed / GLM",
        ],
        [
            [
                r["bin"] + 1,
                _f(r["exposure"], 1),
                _f(r["observed"], 0),
                _f(r["pred_a"], 0),
                _f(r["pred_b"], 0),
                _f(r["ratio_a"], 4),
                _f(r["ratio_b"], 4),
            ]
            for r in dl
        ],
    )
    out += [
        "## 4. Calibration",
        "",
        "By predicted-rate decile (equal exposure) and by factor; values are observed capped "
        "cost divided by predicted capped cost.",
        "",
        f"![calibration by decile]({fig_dir}/{files['calibration_decile']})",
        "",
        f"![calibration by factor]({fig_dir}/{files['calibration_factor']})",
        "",
        "## 5. Loaded premium (reported separately)",
        "",
        m["loaded_premium"]["label"] + ".",
        "",
        f"Loading per exposure-year (training only): {_f(lp['loading_per_exposure_year'], 4)}; "
        f"observed uncapped total: {_f(lp['observed_uncapped_total'], 0)}.",
        "",
        *_table(
            ["candidate", "predicted loaded total", "observed uncapped / loaded"],
            [
                [
                    cid,
                    _f(v["predicted_loaded_total"], 0),
                    _f(v["ratio_observed_uncapped_to_loaded"], 4),
                ]
                for cid, v in m["loaded_premium"]["by_candidate"].items()
            ],
        ),
        "## 6. Bootstrap intervals",
        "",
        f"Primary ({g['label']}); conditional on the frozen models; {g['replicates']} replicates, "
        f"level {g['level']}:",
        "",
        *_table(["quantity", "point", "lower", "upper"], _boot_rows(g)),
        f"Sensitivity ({p['label']}): this is a labelled sensitivity and never a confirmatory "
        f"claim. Delta point {_f(p['delta']['point'], 5)}, interval "
        f"[{_f(p['delta']['lower'], 5)}, {_f(p['delta']['upper'], 5)}] "
        f"(rule applied to this interval: {p['delta']['decision']}); {p['n_units']} units, "
        f"{p['replicates']} replicates, seed {p['seed']}.",
        "",
        f"![policy bootstrap distribution of Delta]({fig_dir}/{files['bootstrap_delta_policy']})",
        "",
        "## 7. Rating-profile groups (the bootstrap unit)",
        "",
        *_table(
            ["quantity", "value"],
            [
                ["groups", gr["groups"]],
                ["policies", gr["policies"]],
                ["exposure-years", _f(gr["exposure"], 1)],
                [
                    "group size quantiles",
                    ", ".join(f"q{k}: {_f(v, 1)}" for k, v in gr["size_quantiles"].items()),
                ],
                ["largest group (policies)", gr["max_group_size"]],
                [
                    f"top {gr['top_groups']} groups ({gr['top_share']} of the groups)",
                    f"{_f(100 * gr['top_policy_share'], 1)}% of the policies, "
                    f"{_f(100 * gr['top_exposure_share'], 1)}% of the exposure",
                ],
            ],
        ),
        "## 8. Attempts in the journal",
        "",
        "Every attempt recorded in `results/test_access.json` is listed, failed or interrupted "
        "ones included (error type only, no metric).",
        "",
        *_table(
            ["attempt", "id", "opened (UTC)", "technical rerun", "status", "error type"],
            [
                [
                    a["attempt"],
                    a["attempt_id"],
                    a["opened_utc"],
                    a["technical_rerun"],
                    a["status"],
                    a["error_type"] or "-",
                ]
                for a in data["attempts"]
            ],
        ),
        "Self-check before test access: "
        f"{m['self_check']['rows']} stored train predictions reproduced, max relative difference "
        f"{m['self_check']['max_rel_diff']:.2e} (tolerance {m['self_check']['rtol']:.0e}).",
        "",
        f"Conditional on: {m['conditional_on']}.",
    ]
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------- run


def run_report(cfg: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """``fremtpl2 report``: verifies the completed attempt, then renders the report and figures
    (staged, then moved into place). Raises :class:`ReportError` before writing anything if no
    completed attempt exists or the outputs disagree."""
    root = Path.cwd() if root is None else Path(root)
    data = load_completed(cfg, root)
    rcfg, fcfg = cfg["report"], cfg["figures"]
    stage = root / cfg["paths"]["processed"] / rcfg["outputs"]["staging_dir"]
    if stage.exists():
        shutil.rmtree(stage)
    try:
        files = make_figures(data, stage / "figures", fcfg, rcfg)
        md = render_report(data, files, cfg, fcfg["dir"])
        (stage / rcfg["outputs"]["markdown"]).write_text(md, encoding="utf-8")
        reports = root / cfg["paths"]["reports"]
        fig_dir = reports / fcfg["dir"]
        fig_dir.mkdir(parents=True, exist_ok=True)
        for name in files.values():
            (stage / "figures" / name).replace(fig_dir / name)
        (stage / rcfg["outputs"]["markdown"]).replace(reports / rcfg["outputs"]["markdown"])
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {
        "attempt": data["metrics"]["attempt"],
        "figures": sorted(files.values()),
        "delta": data["metrics"]["bootstrap"]["group"]["delta"],
    }
