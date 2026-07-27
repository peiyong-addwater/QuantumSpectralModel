"""Shared Matplotlib/SciencePlots style helpers for paper figures."""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def ensure_matplotlib_config_dir() -> None:
    """Use a writable Matplotlib config directory on restricted systems."""

    if "MPLCONFIGDIR" in os.environ:
        return
    mpl_config_dir = Path("/tmp") / "qfm_matplotlib_config"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)


ensure_matplotlib_config_dir()

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

DEFAULT_STYLE = ("science", "nature")
NO_LATEX_STYLE = "no-latex"
LATEX_POLICY_TOKEN = "latex"
LATEX_STYLE_NAMES = frozenset({LATEX_POLICY_TOKEN, "latex-sans"})
LINE_PLOT_FONT_SIZE_INCREMENT = 1.0
LINE_PLOT_FONT_KEYS = (
    "font.size",
    "axes.titlesize",
    "axes.labelsize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
)


def add_style_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--style",
        nargs="+",
        default=list(DEFAULT_STYLE),
        help=(
            "Matplotlib/SciencePlots style names. Use repeated values, '+', or ','. "
            "The default applies SciencePlots science+nature with local non-LaTeX "
            "rendering while preserving Nature's sans-serif font override."
        ),
    )
    parser.add_argument(
        "--list-styles",
        action="store_true",
        help="List available Matplotlib/SciencePlots style names and exit.",
    )


def normalize_style_names(styles: str | list[str] | tuple[str, ...]) -> list[str]:
    """Normalize CLI style values into an ordered Matplotlib style cascade."""

    raw_styles = [styles] if isinstance(styles, str) else list(styles)
    names: list[str] = []
    for raw_style in raw_styles:
        for name in re.split(r"[+,]", str(raw_style)):
            stripped = name.strip()
            if stripped:
                names.append(stripped)

    if not names:
        return ["default"]
    if names == ["default"]:
        return names

    wants_latex = any(name in LATEX_STYLE_NAMES for name in names)
    names = [name for name in names if name != LATEX_POLICY_TOKEN]

    if not wants_latex and NO_LATEX_STYLE not in names:
        if "nature" in names:
            nature_index = names.index("nature")
            names.insert(nature_index, NO_LATEX_STYLE)
        else:
            names.append(NO_LATEX_STYLE)
    elif not wants_latex and "nature" in names and NO_LATEX_STYLE in names:
        names = _move_style_before(names, NO_LATEX_STYLE, "nature")
    return names


def _move_style_before(names: list[str], style: str, anchor: str) -> list[str]:
    """Move one style before another while preserving other style order."""

    without_style = [name for name in names if name != style]
    anchor_index = without_style.index(anchor)
    return without_style[:anchor_index] + [style] + without_style[anchor_index:]


def import_scienceplots_styles() -> None:
    try:
        import scienceplots  # noqa: F401
    except Exception:
        pass


def available_style_names() -> list[str]:
    import_scienceplots_styles()
    return sorted(set(plt.style.available) | {"default"})


def validate_style_names(style_names: list[str]) -> None:
    available = set(available_style_names())
    missing = [name for name in style_names if name not in available]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"unavailable plot style(s): {missing_text}. "
            "Use --list-styles to inspect available styles."
        )


def print_available_styles() -> None:
    for style_name in available_style_names():
        print(style_name)


def use_style(styles: str | list[str] | tuple[str, ...]) -> None:
    style_names = normalize_style_names(styles)
    validate_style_names(style_names)
    if style_names == ["default"]:
        plt.style.use("default")
    else:
        plt.style.use(style_names)


def font_size_in_points(size: Any) -> float:
    """Resolve a Matplotlib font-size value to points."""

    return float(FontProperties(size=size).get_size_in_points())


def bumped_font_size(size: Any, increment: float = LINE_PLOT_FONT_SIZE_INCREMENT) -> float:
    """Return a font size increased by ``increment`` points."""

    return font_size_in_points(size) + increment


@contextmanager
def line_plot_font_context(increment: float = LINE_PLOT_FONT_SIZE_INCREMENT):
    """Temporarily increase text sizes for paper line plots."""

    updates = {
        key: bumped_font_size(mpl.rcParams[key], increment)
        for key in LINE_PLOT_FONT_KEYS
    }
    with mpl.rc_context(updates):
        yield


def add_bottom_legend(
    fig: Any,
    handles: Sequence[Any],
    labels: Sequence[str],
    *,
    ncol: int,
    top: float = 1.0,
    target_gap_lines: float = 2.0,
    min_clearance_lines: float = 0.25,
    bottom_anchor: float = 0.015,
    font_size: Any | None = None,
    frameon: bool = False,
) -> Any | None:
    """Add a bottom legend and move axes near it without overlap.

    The target gap is measured from the legend top to the bottom of the plotted
    axes region. If tick or axis labels need more room, non-overlap takes
    priority and the gap grows just enough to preserve readable text.
    """

    if not handles:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
        return None

    legend = fig.legend(
        handles=list(handles),
        labels=list(labels),
        loc="lower center",
        bbox_to_anchor=(0.5, bottom_anchor),
        ncol=ncol,
        fontsize=font_size,
        frameon=frameon,
        borderaxespad=0.0,
    )
    _expand_figure_for_bottom_legend(fig, legend, target_gap_lines)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    line_height = _legend_text_line_height(legend, renderer) / fig.bbox.height
    bottom = min(legend_box.y1 + target_gap_lines * line_height, top - 0.05)
    fig.tight_layout(rect=(0.0, bottom, 1.0, top))
    _move_axes_for_bottom_legend_gap(
        fig,
        legend,
        target_gap_lines=target_gap_lines,
        min_clearance_lines=min_clearance_lines,
        top=top,
    )
    return legend


def _expand_figure_for_bottom_legend(
    fig: Any,
    legend: Any,
    target_gap_lines: float,
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    line_height = _legend_text_line_height(legend, renderer)
    dpi = fig.dpi
    reserved_inches = (
        legend_box.height / dpi
        + target_gap_lines * line_height / dpi
        + 0.25
    )
    width, height = fig.get_size_inches()
    baseline_reserved_inches = 0.75
    if reserved_inches > baseline_reserved_inches:
        fig.set_size_inches(
            width,
            height + reserved_inches - baseline_reserved_inches,
            forward=True,
        )


def _move_axes_for_bottom_legend_gap(
    fig: Any,
    legend: Any,
    *,
    target_gap_lines: float,
    min_clearance_lines: float,
    top: float,
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    line_height = _legend_text_line_height(legend, renderer) / fig.bbox.height
    axes = list(_visible_axes(fig.axes))
    if not axes:
        return

    axes_bottom = min(axis.get_position().y0 for axis in axes)
    tight_bottom = min(
        axis.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0
        for axis in axes
    )
    desired_axes_bottom = legend_box.y1 + target_gap_lines * line_height
    desired_tight_bottom = legend_box.y1 + min_clearance_lines * line_height
    delta = max(desired_axes_bottom - axes_bottom, desired_tight_bottom - tight_bottom)
    if abs(delta) < 1e-4:
        return

    if delta > 0.0:
        max_top = max(axis.get_position().y1 for axis in axes)
        delta = min(delta, max(0.0, top - max_top - 0.005))
    if abs(delta) < 1e-4:
        return

    for axis in axes:
        box = axis.get_position()
        axis.set_position([box.x0, box.y0 + delta, box.width, box.height])


def _legend_text_line_height(legend: Any, renderer: Any) -> float:
    text_heights = [
        text.get_window_extent(renderer).height
        for text in legend.get_texts()
        if text.get_text()
    ]
    if text_heights:
        return max(text_heights)
    return font_size_in_points(mpl.rcParams["legend.fontsize"]) * legend.figure.dpi / 72.0


def _visible_axes(axes: Iterable[Any]) -> Iterable[Any]:
    return (axis for axis in axes if axis.get_visible())


def style_audit() -> dict[str, Any]:
    """Return rcParams relevant to journal font/style checks."""

    return {
        "font.family": list(mpl.rcParams["font.family"]),
        "font.sans-serif": list(mpl.rcParams["font.sans-serif"]),
        "mathtext.fontset": mpl.rcParams["mathtext.fontset"],
        "text.usetex": bool(mpl.rcParams["text.usetex"]),
    }
