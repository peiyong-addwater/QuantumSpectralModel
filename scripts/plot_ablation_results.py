#!/usr/bin/env python3
"""Plot QFM ablation aggregate results and compare them with main runs."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path("/tmp") / "qfm_matplotlib_config"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ham_embed_spectral.naming import (  # noqa: E402
    ENCODER_CLI_CHOICES,
    canonical_encoder_name,
)
from ham_embed_spectral.plotting import (  # noqa: E402
    add_bottom_legend,
    add_style_arguments,
    bumped_font_size,
    line_plot_font_context,
    print_available_styles,
    use_style,
)
from scripts.plot_qfm_results import (  # noqa: E402
    encoder_plot_style,
    load_complete_rows,
    panel_key,
    panel_title,
    plot_metric_vs_depth,
)
from scripts.plot_qfm_results import (  # noqa: E402
    metric_series_by_panel as _metric_series_by_panel,
)

metric_series_by_panel = _metric_series_by_panel

DEFAULT_ABLATION_ROOT = Path("results/tables/ablation_aggregates")
DEFAULT_OUTPUT_DIR = Path("results/figures/ablation_studies")
DEFAULT_METRICS = (
    "final_test_accuracy",
    "best_validation_accuracy",
    "validation_accuracy_auc",
)
DATASET_FAMILIES = ("pendigits", "synthetic")
DEFAULT_MAIN_TABLES = {
    "pendigits": Path("results/tables/pendigits_runs.csv"),
    "synthetic": Path("results/tables/synthetic_runs.csv"),
}
COMPARISON_KEY_FIELDS = (
    "dataset",
    "representation",
    "encoder",
    "depth",
    "seed",
    "learning_rate",
    "batch_size",
    "steps",
    "standardize",
    "initial_state",
    "projector_renormalize",
    "track_readout_leakage",
)


@dataclass(frozen=True)
class AblationTable:
    """One aggregated ablation run table and its path-derived identity."""

    dataset_family: str
    ablation: str
    path: Path


@dataclass(frozen=True)
class PairedMetricPoint:
    """One matched original/ablated metric value for a run-grid point."""

    panel: str
    encoder: str
    depth: int
    seed: int
    original: float
    ablated: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ablation-root", default=str(DEFAULT_ABLATION_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        default=None,
        help="Metric column to plot. May be repeated.",
    )
    parser.add_argument(
        "--ablation",
        action="append",
        default=None,
        help="Ablation slug to include, such as entry-permutation. May be repeated.",
    )
    parser.add_argument(
        "--encoder",
        action="append",
        choices=ENCODER_CLI_CHOICES,
        default=None,
        help="Encoder slug to include. Legacy aliases are canonicalized. May be repeated.",
    )
    parser.add_argument(
        "--dataset-family",
        nargs="+",
        choices=DATASET_FAMILIES,
        default=None,
        help="Dataset family/families to include.",
    )
    parser.add_argument(
        "--comparison",
        choices=("overlay", "delta", "both"),
        default="both",
        help="Original-vs-ablation comparison view to generate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned input and output paths without writing figures.",
    )
    add_style_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_styles:
        print_available_styles()
        return

    metrics = tuple(args.metrics or DEFAULT_METRICS)
    dataset_families = set(args.dataset_family or DATASET_FAMILIES)
    ablation_filter = set(args.ablation) if args.ablation else None
    encoder_filter = canonical_encoder_filter(args.encoder)
    ablation_tables = discover_ablation_tables(
        Path(args.ablation_root),
        dataset_families=dataset_families,
        ablations=ablation_filter,
    )
    if not ablation_tables:
        raise SystemExit("no ablation aggregate tables matched the requested filters")

    main_tables = selected_main_tables(dataset_families)
    if args.dry_run:
        print(f"DRY-RUN ablation root: {Path(args.ablation_root)}")
        for family, table_path in sorted(main_tables.items()):
            print(f"DRY-RUN main table [{family}]: {table_path}")
    else:
        metric_output_dir(args).mkdir(parents=True, exist_ok=True)
        comparison_output_dir(args).mkdir(parents=True, exist_ok=True)
        use_style(args.style)

    planned_or_written = 0
    for table in ablation_tables:
        main_table_path = main_tables[table.dataset_family]
        for metric in metrics:
            planned_or_written += write_or_plan_metric_plot(
                table,
                metric,
                args,
                encoder_filter=encoder_filter,
            )
            planned_or_written += write_or_plan_comparison_plots(
                table,
                main_table_path,
                metric,
                args,
                encoder_filter=encoder_filter,
            )
    if planned_or_written == 0:
        raise SystemExit("no figures were planned or written for the requested filters")


def canonical_encoder_filter(encoders: list[str] | None) -> set[str] | None:
    if encoders is None:
        return None
    return {canonical_encoder_name(encoder) for encoder in encoders}


def selected_main_tables(dataset_families: set[str]) -> dict[str, Path]:
    main_tables = {
        family: DEFAULT_MAIN_TABLES[family]
        for family in DATASET_FAMILIES
        if family in dataset_families
    }
    missing = [str(path) for path in main_tables.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing main aggregate table(s): {', '.join(missing)}")
    return main_tables


def parse_ablation_table_path(path: Path) -> AblationTable | None:
    parts = path.parent.name.split("__", maxsplit=1)
    if len(parts) != 2:
        return None
    dataset_family, ablation = parts
    if dataset_family not in DATASET_FAMILIES:
        return None
    return AblationTable(dataset_family=dataset_family, ablation=ablation, path=path)


def discover_ablation_tables(
    ablation_root: Path,
    *,
    dataset_families: set[str] | None = None,
    ablations: set[str] | None = None,
) -> list[AblationTable]:
    tables: list[AblationTable] = []
    for path in sorted(ablation_root.glob("*/*_runs.csv")):
        table = parse_ablation_table_path(path)
        if table is None:
            continue
        if dataset_families is not None and table.dataset_family not in dataset_families:
            continue
        if ablations is not None and table.ablation not in ablations:
            continue
        tables.append(table)
    return tables


def filtered_rows(
    path: Path,
    *,
    metric: str,
    encoder_filter: set[str] | None = None,
) -> list[dict[str, str]]:
    rows = load_complete_rows(path, metric=metric)
    if encoder_filter is None:
        return rows
    return [
        row
        for row in rows
        if canonical_encoder_name(str(row.get("encoder", ""))) in encoder_filter
    ]


def write_or_plan_metric_plot(
    table: AblationTable,
    metric: str,
    args: argparse.Namespace,
    *,
    encoder_filter: set[str] | None,
) -> int:
    output_base = metric_plot_base(Path(args.output_dir), table, metric)
    rows = filtered_rows(table.path, metric=metric, encoder_filter=encoder_filter)
    if not rows:
        return 0
    if args.dry_run:
        print(f"DRY-RUN metric-vs-depth input: {table.path}")
        print_figure_paths(output_base)
        return 2
    plot_metric_vs_depth(rows, metric, output_base)
    print_figure_paths(output_base)
    return 2


def write_or_plan_comparison_plots(
    table: AblationTable,
    main_table_path: Path,
    metric: str,
    args: argparse.Namespace,
    *,
    encoder_filter: set[str] | None,
) -> int:
    main_rows = filtered_rows(main_table_path, metric=metric, encoder_filter=encoder_filter)
    ablation_rows = filtered_rows(table.path, metric=metric, encoder_filter=encoder_filter)
    if not main_rows or not ablation_rows:
        return 0
    points, n_missing = paired_metric_points(main_rows, ablation_rows, metric)
    if n_missing:
        print(
            f"warning: skipped {n_missing} unpaired ablation row(s) for {table.path}",
            file=sys.stderr,
        )
    if not points:
        return 0

    count = 0
    for encoder in sorted({point.encoder for point in points}):
        encoder_points = [point for point in points if point.encoder == encoder]
        for view in comparison_views(args.comparison):
            output_base = comparison_plot_base(Path(args.output_dir), table, encoder, metric, view)
            if args.dry_run:
                print(
                    "DRY-RUN original-vs-ablation input: "
                    f"main={main_table_path} ablation={table.path} "
                    f"encoder={encoder} view={view} n_pairs={len(encoder_points)}"
                )
                print_figure_paths(output_base)
            elif view == "overlay":
                plot_overlay_comparison(
                    encoder_points,
                    metric=metric,
                    ablation=table.ablation,
                    encoder=encoder,
                    output_base=output_base,
                )
                print_figure_paths(output_base)
            else:
                plot_delta_comparison(
                    encoder_points,
                    metric=metric,
                    encoder=encoder,
                    output_base=output_base,
                )
                print_figure_paths(output_base)
            count += 2
    return count


def comparison_views(selection: str) -> tuple[str, ...]:
    if selection == "both":
        return ("overlay", "delta")
    return (selection,)


def metric_output_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / "metric_vs_depth"


def comparison_output_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / "original_vs_ablation"


def metric_plot_base(output_dir: Path, table: AblationTable, metric: str) -> Path:
    return (
        output_dir
        / "metric_vs_depth"
        / f"{table.dataset_family}__{table.ablation}__{metric}_vs_depth"
    )


def comparison_plot_base(
    output_dir: Path,
    table: AblationTable,
    encoder: str,
    metric: str,
    view: str,
) -> Path:
    return (
        output_dir
        / "original_vs_ablation"
        / f"{table.dataset_family}__{table.ablation}__{encoder}__{metric}__{view}"
    )


def print_figure_paths(output_base: Path) -> None:
    print(f"{output_base}.pdf")
    print(f"{output_base}.png")


def comparison_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(normalized_key_value(row, field) for field in COMPARISON_KEY_FIELDS)


def normalized_key_value(row: dict[str, str], field: str) -> str:
    value = row.get(field, "")
    if value in {"", None}:
        return ""
    if field == "encoder":
        return canonical_encoder_name(str(value))
    if field in {"depth", "seed", "batch_size", "steps"}:
        return str(int(float(value)))
    if field == "learning_rate":
        return f"{float(value):.12g}"
    return str(value)


def rows_by_comparison_key(rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, str]]:
    indexed: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = comparison_key(row)
        if key in indexed:
            raise ValueError(f"duplicate row for comparison key {key!r}")
        indexed[key] = row
    return indexed


def paired_metric_points(
    main_rows: list[dict[str, str]],
    ablation_rows: list[dict[str, str]],
    metric: str,
) -> tuple[list[PairedMetricPoint], int]:
    main_index = rows_by_comparison_key(main_rows)
    points: list[PairedMetricPoint] = []
    n_missing = 0
    for ablation_row in ablation_rows:
        main_row = main_index.get(comparison_key(ablation_row))
        if main_row is None:
            n_missing += 1
            continue
        points.append(
            PairedMetricPoint(
                panel=panel_key(ablation_row),
                encoder=canonical_encoder_name(str(ablation_row["encoder"])),
                depth=int(float(ablation_row["depth"])),
                seed=int(float(ablation_row["seed"])),
                original=float(main_row[metric]),
                ablated=float(ablation_row[metric]),
            )
        )
    return points, n_missing


def summarize_points(
    points: list[PairedMetricPoint],
    value_fn: Callable[[PairedMetricPoint], float],
) -> dict[str, dict[int, tuple[float, float]]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for point in points:
        grouped[(point.panel, point.depth)].append(value_fn(point))

    series: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    for (panel, depth), values in grouped.items():
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
        series[panel][depth] = (mean, variance**0.5)
    return series


def plot_overlay_comparison(
    points: list[PairedMetricPoint],
    *,
    metric: str,
    ablation: str,
    encoder: str,
    output_base: Path,
) -> None:
    with line_plot_font_context():
        panels = sorted({point.panel for point in points})
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(5.2 * len(panels), 4.1),
            squeeze=False,
        )
        style = encoder_plot_style(encoder)
        original_series = summarize_points(points, lambda point: point.original)
        ablation_series = summarize_points(points, lambda point: point.ablated)
        legend_handles: dict[str, object] = {}
        for axis, panel in zip(axes[0], panels, strict=True):
            for label, series, linestyle in (
                ("Original", original_series, "-"),
                (f"Ablated: {human_label(ablation)}", ablation_series, "--"),
            ):
                depths = sorted(series[panel])
                means = [series[panel][depth][0] for depth in depths]
                stds = [series[panel][depth][1] for depth in depths]
                handle = axis.errorbar(
                    depths,
                    means,
                    yerr=stds,
                    capsize=2,
                    label=label,
                    linewidth=1.25,
                    linestyle=linestyle,
                    markersize=3.0,
                    markerfacecolor="white",
                    markeredgewidth=0.8,
                    **style,
                )
                legend_handles.setdefault(label, handle)
            finish_axis(axis, panel, ylabel=metric_label(metric))
        add_bottom_legend(
            fig,
            list(legend_handles.values()),
            list(legend_handles),
            ncol=2,
            font_size=bumped_font_size("small"),
        )
        save_figure(fig, output_base)


def plot_delta_comparison(
    points: list[PairedMetricPoint],
    *,
    metric: str,
    encoder: str,
    output_base: Path,
) -> None:
    panels = sorted({point.panel for point in points})
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.2 * len(panels), 4.1),
        squeeze=False,
    )
    style = encoder_plot_style(encoder)
    delta_series = summarize_points(points, lambda point: point.original - point.ablated)
    for axis, panel in zip(axes[0], panels, strict=True):
        depths = sorted(delta_series[panel])
        means = [delta_series[panel][depth][0] for depth in depths]
        stds = [delta_series[panel][depth][1] for depth in depths]
        axis.errorbar(
            depths,
            means,
            yerr=stds,
            capsize=2,
            linewidth=1.25,
            markersize=3.0,
            markerfacecolor="white",
            markeredgewidth=0.8,
            **style,
        )
        axis.axhline(0.0, color="0.4", linewidth=0.8, linestyle=":")
        finish_axis(axis, panel, ylabel=delta_label(metric))
    fig.tight_layout()
    save_figure(fig, output_base)


def finish_axis(axis: plt.Axes, panel: str, *, ylabel: str) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Re-upload depth")
    axis.set_ylabel(ylabel)
    axis.set_title(panel_title(panel))
    axis.grid(True, alpha=0.25)


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_base}.pdf")
    fig.savefig(f"{output_base}.png", dpi=300)
    plt.close(fig)


def metric_label(metric: str) -> str:
    return metric.replace("_", " ")


def delta_label(metric: str) -> str:
    label = metric_label(metric)
    if "accuracy" in metric or metric.endswith("auc"):
        return f"{label} drop (original - ablated)"
    return f"{label} difference (original - ablated)"


def human_label(value: str) -> str:
    return value.replace("_", " ").replace("-", " ")


if __name__ == "__main__":
    main()
