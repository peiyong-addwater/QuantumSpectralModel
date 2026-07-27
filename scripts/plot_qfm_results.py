#!/usr/bin/env python3
"""Generate static QFM result figures from aggregated run tables."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path("/tmp") / "qfm_matplotlib_config"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ham_embed_spectral.naming import canonical_encoder_name  # noqa: E402
from ham_embed_spectral.plotting import (  # noqa: E402
    add_bottom_legend,
    add_style_arguments,
    available_style_names,
    bumped_font_size,
    import_scienceplots_styles,
    line_plot_font_context,
    normalize_style_names,
    print_available_styles,
    style_audit,
    use_style,
    validate_style_names,
)

__all__ = [
    "add_style_arguments",
    "available_style_names",
    "import_scienceplots_styles",
    "normalize_style_names",
    "print_available_styles",
    "style_audit",
    "use_style",
    "validate_style_names",
]

ENCODER_STYLES = {
    "fixed-ry": {"marker": "o", "color": "#1f77b4"},
    "fixed-ry-rz": {"marker": "s", "color": "#17becf"},
    "trainable-frequency-ry": {"marker": "^", "color": "#2ca02c"},
    "patch-su4": {"marker": "D", "color": "#9467bd"},
    "trainable-patch-su4": {"marker": "P", "color": "#ff7f0e"},
    "non-overlap-patch-block-hamiltonian": {"marker": "v", "color": "#8c564b"},
    "symmetric-hamiltonian": {"marker": "X", "color": "#d62728"},
    "block-hamiltonian": {"marker": "h", "color": "#222222"},
}
DEFAULT_ENCODER_STYLE = {"marker": "o", "color": None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        default=None,
        help=(
            "Aggregated CSV from scripts/aggregate_runs.py. "
            "Required unless --list-styles is used."
        ),
    )
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--metric", default="final_test_accuracy")
    add_style_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_styles:
        print_available_styles()
        return
    if args.table is None:
        raise SystemExit("--table is required unless --list-styles is used")
    rows = load_complete_rows(Path(args.table), metric=args.metric)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_style(args.style)
    figure_path = output_dir / f"{Path(args.table).stem}__{args.metric}_vs_depth"
    plot_metric_vs_depth(rows, args.metric, figure_path)
    print(f"{figure_path}.pdf")
    print(f"{figure_path}.png")


def load_complete_rows(path: Path, *, metric: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("status") == "complete" and row.get(metric) not in {"", None}
        ]
    if not rows:
        raise ValueError(f"no complete rows with metric {metric!r} found in {path}")
    return rows


def panel_key(row: dict[str, str]) -> str:
    """Return the dataset-like facet key for a plotted row."""
    representation = row["representation"]
    if representation == "synthetic":
        return row["dataset"]
    return representation


def panel_title(panel: str) -> str:
    return panel.replace("_", " ").replace("-", " ").upper()


def metric_series_by_panel(
    rows: list[dict[str, str]],
    metric: str,
) -> dict[tuple[str, str], dict[int, tuple[float, float]]]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (panel_key(row), canonical_encoder_name(row["encoder"]), int(row["depth"]))
        grouped[key].append(float(row[metric]))

    series: dict[tuple[str, str], dict[int, tuple[float, float]]] = defaultdict(dict)
    for (panel, encoder, depth), values in grouped.items():
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
        series[(panel, encoder)][depth] = (mean, variance**0.5)
    return series


def encoder_plot_style(encoder: str) -> dict[str, str]:
    """Return stable marker/color settings for a canonical encoder name."""

    style = ENCODER_STYLES.get(encoder, DEFAULT_ENCODER_STYLE)
    return {key: value for key, value in style.items() if value is not None}


def plot_metric_vs_depth(rows: list[dict[str, str]], metric: str, output_base: Path) -> None:
    with line_plot_font_context():
        series = metric_series_by_panel(rows, metric)
        panels = sorted({key[0] for key in series})

        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(5.2 * len(panels), 4.1),
            squeeze=False,
        )
        legend_handles: dict[str, object] = {}
        for axis, panel in zip(axes[0], panels, strict=True):
            for rep, encoder in sorted(series):
                if rep != panel:
                    continue
                depths = sorted(series[(rep, encoder)])
                means = [series[(rep, encoder)][depth][0] for depth in depths]
                stds = [series[(rep, encoder)][depth][1] for depth in depths]
                handle = axis.errorbar(
                    depths,
                    means,
                    yerr=stds,
                    capsize=2,
                    label=encoder,
                    linewidth=1.25,
                    markersize=3.0,
                    markerfacecolor="white",
                    markeredgewidth=0.8,
                    **encoder_plot_style(encoder),
                )
                legend_handles.setdefault(encoder, handle)
            axis.set_xscale("log", base=2)
            axis.set_xlabel("Re-upload depth")
            axis.set_ylabel(metric.replace("_", " "))
            axis.set_title(panel_title(panel))
            axis.grid(True, alpha=0.25)
        add_bottom_legend(
            fig,
            list(legend_handles.values()),
            list(legend_handles),
            ncol=min(4, max(1, len(legend_handles))),
            font_size=bumped_font_size("small"),
        )
        fig.savefig(f"{output_base}.pdf")
        fig.savefig(f"{output_base}.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
