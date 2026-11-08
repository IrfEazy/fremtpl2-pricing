"""Deterministic matplotlib figures (Agg backend, fixed size and dpi, no PNG metadata).

Only drawing helpers live here: every number plotted comes from a table the caller computed from
pipeline files. Size, dpi and directory are config keys (``figures``). Figures are written as PNG
without the Software / date metadata so that the same data and library versions give the same file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Okabe-Ito colour-blind safe palette
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
RC = {
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.constrained_layout.use": True,
    "savefig.facecolor": "white",
}


def color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def new_figure(fcfg: dict[str, Any], rows: int = 1, cols: int = 1, **kw):
    """(fig, axes array) of the configured size (``width`` x ``height`` inches per panel row/col
    scaled to the grid: the figure is ``width`` wide and ``height * rows`` high)."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(
            rows, cols, figsize=(float(fcfg["width"]), float(fcfg["height"]) * rows), **kw
        )
    return fig, np.atleast_1d(axes).reshape(-1)


def save(fig, path: Path, fcfg: dict[str, Any]) -> str:
    """Write ``fig`` as PNG (fixed dpi, no metadata) and close it; returns the file name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path, format="png", dpi=int(fcfg["dpi"]), metadata={"Software": None, "Creation Time": None}
    )
    plt.close(fig)
    return path.name


def hide_unused(axes, used: int) -> None:
    for ax in axes[used:]:
        ax.set_visible(False)


def bar_importance(
    fcfg: dict[str, Any], path: Path, panels: dict[str, dict[str, float]], title: str, xlabel: str
) -> str:
    """One horizontal bar panel per component: feature -> value (sorted descending)."""
    fig, axes = new_figure(fcfg, 1, len(panels))
    for ax, (name, vals) in zip(axes, panels.items(), strict=True):
        items = sorted(vals.items(), key=lambda kv: kv[1])
        ax.barh([k for k, _ in items], [v for _, v in items], color=color(0))
        ax.set_title(name)
        ax.set_xlabel(xlabel)
    fig.suptitle(title)
    return save(fig, path, fcfg)


def dependence_grid(
    fcfg: dict[str, Any], path: Path, panels: list[dict[str, Any]], title: str
) -> str:
    """Grid of dependence panels. Each panel dict: ``name``, ``kind`` ('numeric'|'categorical'),
    ``x`` / ``y`` (point cloud, numeric only), ``bins`` (list of {x, label, mean}), ``xlabel``,
    ``log_x`` (optional)."""
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = new_figure(fcfg, nrow, ncol)
    for ax, p in zip(axes, panels, strict=False):
        b = p["bins"]
        if p["kind"] == "numeric":
            ax.scatter(p["x"], p["y"], s=2, alpha=0.25, color=color(5), rasterized=True)
            ax.plot([r["x"] for r in b], [r["mean"] for r in b], color=color(1), lw=1.5)
            if p.get("log_x"):
                ax.set_xscale("log")
        else:
            xs = np.arange(len(b))
            ax.bar(xs, [r["mean"] for r in b], color=color(0))
            ax.set_xticks(xs, [r["label"] for r in b], rotation=90)
        ax.set_title(p["name"])
        ax.set_xlabel(p["xlabel"])
        ax.set_ylabel("contribution (log scale)")
    hide_unused(axes, len(panels))
    fig.suptitle(title)
    return save(fig, path, fcfg)


def compare_grid(fcfg: dict[str, Any], path: Path, panels: list[dict[str, Any]], title: str) -> str:
    """GBM mean contribution vs log GLM relativity per factor. Each panel: ``name``, ``labels``,
    ``gbm``, ``glms`` (list of {label, role, values}), ``ylabel``."""
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = new_figure(fcfg, nrow, ncol)
    for ax, p in zip(axes, panels, strict=False):
        xs = np.arange(len(p["labels"]))
        for i, g in enumerate(p["glms"]):
            ax.plot(xs, g["values"], "o-" if g["role"] == "selected" else "^:",
                    color=color(1 + i), label=g["label"], ms=3)  # fmt: skip
        ax.plot(xs, p["gbm"], "s--", color=color(0), label="GBM mean contribution", ms=3)
        ax.axhline(0.0, color="black", lw=0.5)
        ax.set_xticks(xs, p["labels"], rotation=90)
        ax.set_title(p["name"])
        ax.set_ylabel(p["ylabel"])
    axes[0].legend()
    hide_unused(axes, len(panels))
    fig.suptitle(title)
    return save(fig, path, fcfg)


def interaction_figure(
    fcfg: dict[str, Any], path: Path, cloud: dict[str, Any], lines: list[dict[str, Any]], title: str
) -> str:
    """Left: DrivAge contribution cloud coloured by VehPower; right: band means per power band."""
    fig, axes = new_figure(fcfg, 1, 2)
    sc = axes[0].scatter(
        cloud["x"], cloud["y"], c=cloud["c"], s=3, alpha=0.5, cmap="viridis", rasterized=True
    )
    fig.colorbar(sc, ax=axes[0], label=cloud["clabel"])
    axes[0].set_xlabel(cloud["xlabel"])
    axes[0].set_ylabel("contribution (log scale)")
    for i, ln in enumerate(lines):
        axes[1].plot(range(len(ln["y"])), ln["y"], "o-", color=color(i), ms=3, label=ln["label"])
    axes[1].set_xticks(range(len(lines[0]["y"])), lines[0]["x"], rotation=90)
    axes[1].set_xlabel("DrivAge band")
    axes[1].set_ylabel("mean contribution (log scale)")
    axes[1].legend(title="VehPower band")
    fig.suptitle(title)
    return save(fig, path, fcfg)


def line_panels(
    fcfg: dict[str, Any],
    path: Path,
    panels: list[dict[str, Any]],
    title: str,
    ncol: int = 3,
    diagonal: bool = False,
) -> str:
    """Generic grid of line panels. Panel: ``name``, ``xlabel``, ``ylabel``, ``series`` (list of
    {label, x, y, style}), optional ``hline`` and ``xticklabels`` (categorical x)."""
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = new_figure(fcfg, nrow, ncol)
    for ax, p in zip(axes, panels, strict=False):
        for i, s in enumerate(p["series"]):
            ax.plot(s["x"], s["y"], s.get("style", "o-"), color=color(i), ms=3, label=s["label"])
        if diagonal:
            ax.plot([0, 1], [0, 1], color="black", lw=0.6, ls=":")
        if p.get("hline") is not None:
            ax.axhline(p["hline"], color="black", lw=0.6, ls=":")
        if p.get("xticklabels") is not None:
            ax.set_xticks(range(len(p["xticklabels"])), p["xticklabels"], rotation=90)
        ax.set_title(p["name"])
        ax.set_xlabel(p["xlabel"])
        ax.set_ylabel(p["ylabel"])
        ax.legend()
    hide_unused(axes, len(panels))
    fig.suptitle(title)
    return save(fig, path, fcfg)


def histogram(
    fcfg: dict[str, Any],
    path: Path,
    values,
    bins: int,
    marks: dict[str, float],
    title: str,
    xlabel: str,
) -> str:
    fig, axes = new_figure(fcfg)
    ax = axes[0]
    ax.hist(values, bins=bins, color=color(5), edgecolor="white")
    for i, (label, x) in enumerate(marks.items()):
        ax.axvline(x, color=color(1 + i), lw=1.2, label=f"{label}: {x:.4g}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("replicates")
    ax.legend()
    ax.set_title(title)
    return save(fig, path, fcfg)
