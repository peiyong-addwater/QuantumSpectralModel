#!/usr/bin/env python3
"""Plot classical baseline aggregate summaries as point/error-bar facets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path("/tmp") / "qfm_matplotlib_config"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ham_embed_spectral.plotting import (  # noqa: E402
    add_style_arguments,
    print_available_styles,
    use_style,
)

PANEL_ORDER = (
    ("pendigits", "dyn", "Pendigits DYN"),
    ("pendigits", "sta4", "Pendigits STA4"),
    ("synthetic-eigengap", "synthetic", "Synthetic eigengap"),
    ("synthetic-singular", "synthetic", "Synthetic singular"),
)
PANEL_INDEX = {
    (dataset, representation): index
    for index, (dataset, representation, _) in enumerate(PANEL_ORDER)
}
PANEL_LABELS = {(dataset, representation): label for dataset, representation, label in PANEL_ORDER}

BASELINE_ORDER = (
    ("raw-data", "raw", "Raw values", "#2a9d8f"),
    ("symmetric-hamiltonian", "values", "H_sym eigenvalues", "#4575b4"),
    ("block-hamiltonian", "values", "H_block singular values", "#d73027"),
)
BASELINE_INDEX = {
    (descriptor, feature_set): index
    for index, (descriptor, feature_set, _, _) in enumerate(BASELINE_ORDER)
}
BASELINE_LABELS = {
    (descriptor, feature_set): label
    for descriptor, feature_set, label, _ in BASELINE_ORDER
}
BASELINE_COLORS = {
    (descriptor, feature_set): color
    for descriptor, feature_set, _, color in BASELINE_ORDER
}


@dataclass(frozen=True)
class PlotRecord:
    panel_key: tuple[str, str]
    panel_label: str
    baseline_key: tuple[str, str]
    baseline_label: str
    color: str
    mean: float
    std: float
    n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--summary", default="results/tables/classical_baseline_summary.json")
    parser.add_argument("--output-dir", default="results/figures/classical_baseline")
    parser.add_argument("--name", default="classical_baseline")
    parser.add_argument("--metric", default="test_accuracy")
    parser.add_argument("--classifier", default="mlp")
    add_style_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_styles:
        print_available_styles()
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(Path(args.summary), metric=args.metric, classifier=args.classifier)
    use_style(args.style)
    output_base = output_dir / f"{args.name}__{args.metric}_point_error"
    plot_point_error(records, output_base, metric=args.metric)
    print(f"{output_base}.pdf")
    print(f"{output_base}.png")


def load_records(
    path: Path, *, metric: str = "test_accuracy", classifier: str = "mlp"
) -> list[PlotRecord]:
    payload = json.loads(path.read_text())
    return normalize_groups(payload.get("groups", []), metric=metric, classifier=classifier)


def normalize_groups(
    groups: list[dict[str, Any]],
    *,
    metric: str = "test_accuracy",
    classifier: str = "mlp",
) -> list[PlotRecord]:
    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"
    records: list[PlotRecord] = []
    for group in groups:
        if group.get("classifier") != classifier:
            continue
        panel_key = (str(group.get("dataset", "")), str(group.get("representation", "")))
        baseline_key = (str(group.get("descriptor", "")), str(group.get("feature_set", "")))
        if panel_key not in PANEL_INDEX or baseline_key not in BASELINE_INDEX:
            continue
        if group.get(mean_key, "") in {"", None}:
            continue
        records.append(
            PlotRecord(
                panel_key=panel_key,
                panel_label=PANEL_LABELS[panel_key],
                baseline_key=baseline_key,
                baseline_label=BASELINE_LABELS[baseline_key],
                color=BASELINE_COLORS[baseline_key],
                mean=float(group[mean_key]),
                std=float(group.get(std_key) or 0.0),
                n=int(group.get("n") or 0),
            )
        )
    if not records:
        raise ValueError("no plottable classical baseline groups found in summary")
    return sorted(
        records,
        key=lambda record: (
            PANEL_INDEX[record.panel_key],
            BASELINE_INDEX[record.baseline_key],
        ),
    )


def panel_ylim(records: list[PlotRecord], *, min_span: float = 0.06) -> tuple[float, float]:
    if not records:
        return 0.0, 1.0

    lower = min(record.mean - record.std for record in records)
    upper = max(record.mean + record.std for record in records)
    span = max(upper - lower, 0.0)
    padding = max(0.01, 0.12 * span)
    lower -= padding
    upper += padding

    if upper - lower < min_span:
        center = 0.5 * (lower + upper)
        lower = center - 0.5 * min_span
        upper = center + 0.5 * min_span

    if lower < 0.0:
        upper = min(1.0, upper - lower)
        lower = 0.0
    if upper > 1.0:
        lower = max(0.0, lower - (upper - 1.0))
        upper = 1.0

    if upper - lower < min_span:
        if lower <= 0.0:
            upper = min(1.0, min_span)
        elif upper >= 1.0:
            lower = max(0.0, 1.0 - min_span)

    return lower, upper


def plot_point_error(records: list[PlotRecord], output_base: Path, *, metric: str) -> None:
    by_panel: dict[tuple[str, str], list[PlotRecord]] = {key[:2]: [] for key in PANEL_ORDER}
    for record in records:
        by_panel.setdefault(record.panel_key, []).append(record)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.8), sharey=False)
    for axis, (dataset, representation, panel_label) in zip(axes.ravel(), PANEL_ORDER, strict=True):
        panel_key = (dataset, representation)
        panel_records = by_panel.get(panel_key, [])
        for record in panel_records:
            x_position = BASELINE_INDEX[record.baseline_key]
            axis.plot(
                [x_position],
                [record.mean],
                marker="o",
                linestyle="none",
                markersize=2.0,
                markerfacecolor="white",
                markeredgecolor=record.color,
                markeredgewidth=0.8,
                zorder=3,
            )
            axis.errorbar(
                [x_position],
                [record.mean],
                yerr=[record.std],
                fmt="none",
                capsize=4,
                capthick=1.2,
                elinewidth=1.2,
                color=record.color,
                zorder=4,
            )
        axis.set_title(panel_label)
        axis.set_ylim(*panel_ylim(panel_records))
        axis.set_xticks(range(len(BASELINE_ORDER)))
        axis.set_xticklabels([label for _, _, label, _ in BASELINE_ORDER], rotation=25, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
        if not panel_records:
            axis.text(0.5, 0.5, "No data", transform=axis.transAxes, ha="center", va="center")

    for axis in axes[:, 0]:
        axis.set_ylabel(metric.replace("_", " "))

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=color,
            label=label,
        )
        for _, _, label, color in BASELINE_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(f"{output_base}.pdf")
    fig.savefig(f"{output_base}.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
