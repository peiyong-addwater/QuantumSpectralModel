#!/usr/bin/env python3
"""Plot merged gradient diagnostic summaries as static paper figures."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ham_embed_spectral.naming import canonical_encoder_name  # noqa: E402
from ham_embed_spectral.plotting import (  # noqa: E402
    add_bottom_legend,
    add_style_arguments,
    line_plot_font_context,
    print_available_styles,
    use_style,
)
from scripts.plot_qfm_results import ENCODER_STYLES, encoder_plot_style  # noqa: E402

DEFAULT_INPUT_DIR = "results/tables/gradient_diagnostics"
DEFAULT_OUTPUT_DIR = "results/figures/gradient_diagnostics"
MAIN_GRADIENT_FONT_SIZES = {
    "font.size": 18.0,
    "axes.titlesize": 18.0,
    "axes.labelsize": 18.0,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "legend.fontsize": 15.0,
}
MAIN_GRADIENT_ROW_HEIGHT = 2.9
MAIN_GRADIENT_RIGHT = 0.965

PLOT_CHOICES = (
    "init-variance",
    "near-zero",
    "rms-depth",
    "layerwise-flow",
    "checkpoint-evolution",
    "fisher",
    "group-balance",
    "all",
)
ALL_PLOTS = tuple(choice for choice in PLOT_CHOICES if choice != "all")
MODES = ("init", "final", "checkpoints")
PANEL_ORDER = (
    (("pendigits", "dyn"), "Pendigits DYN"),
    (("pendigits", "sta4"), "Pendigits STA4"),
    (("synthetic-eigengap", "synthetic"), "Synthetic eigengap"),
    (("synthetic-singular", "synthetic"), "Synthetic singular"),
)
PANEL_INDEX = {key: index for index, (key, _) in enumerate(PANEL_ORDER)}
PANEL_LABELS = dict(PANEL_ORDER)
GROUP_ORDER = ("theta_su", "gamma", "t_layers", "patch_map")
GROUP_LABELS = {
    "theta_su": "SU(4) mixer",
    "gamma": "Trainable frequencies",
    "t_layers": "Upload times",
    "patch_map": "Patch map",
}
GROUP_INDEX = {group: index for index, group in enumerate(GROUP_ORDER)}
NEAR_ZERO_Y_LIMITS = (-0.1, 1.0)
FISHER_METRICS = (
    ("trace", "Fisher trace", True),
    ("effective_rank", "Fisher effective rank", False),
    ("condition_number", "Fisher condition number", True),
)
BALANCE_RATIOS = (
    ("gamma", "theta_su", "gamma / theta_su"),
    ("t_layers", "theta_su", "t_layers / theta_su"),
    ("patch_map", "theta_su", "patch_map / theta_su"),
)
DEPTH_LINESTYLES = ("-", "--", ":", "-.")


@dataclass(frozen=True)
class DiagnosticRecord:
    """One complete diagnostic record from a merged gradient diagnostic JSON."""

    mode: str
    dataset: str
    representation: str
    panel_key: tuple[str, str]
    panel_label: str
    encoder: str
    depth: int
    seed: int | None
    checkpoint: str | None
    checkpoint_step: int | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class GroupMetricRow:
    """One scalar group metric ready for aggregation and plotting."""

    mode: str
    panel_key: tuple[str, str]
    panel_label: str
    encoder: str
    depth: int
    group: str
    metric: str
    value: float
    seed: int | None = None
    checkpoint_step: int | None = None


@dataclass(frozen=True)
class FisherMetricRow:
    """One scalar empirical Fisher metric ready for aggregation and plotting."""

    mode: str
    panel_key: tuple[str, str]
    panel_label: str
    encoder: str
    depth: int
    metric: str
    value: float
    seed: int | None = None
    checkpoint_step: int | None = None


@dataclass(frozen=True)
class LayerwiseFlowRow:
    """One layerwise flow point ready for aggregation and plotting."""

    mode: str
    panel_key: tuple[str, str]
    panel_label: str
    encoder: str
    depth: int
    group: str
    layer_fraction: float
    n_layers: int
    value: float
    seed: int | None = None
    checkpoint_step: int | None = None


@dataclass(frozen=True)
class BalanceRatioRow:
    """One group-to-mixer RMS-gradient ratio ready for aggregation and plotting."""

    mode: str
    panel_key: tuple[str, str]
    panel_label: str
    encoder: str
    depth: int
    ratio: str
    value: float
    seed: int | None = None
    checkpoint_step: int | None = None


@dataclass(frozen=True)
class Summary:
    """Mean/std/count summary for repeated diagnostic rows."""

    mean: float
    std: float
    n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot", choices=PLOT_CHOICES, default="all")
    add_style_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_styles:
        print_available_styles()
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(Path(args.input_dir))
    use_style(args.style)

    written = plot_requested(records, args.plot, output_dir)
    if not written:
        raise ValueError(f"no figures were produced for --plot {args.plot!r}")
    for path in written:
        print(path)


def plot_requested(records: list[DiagnosticRecord], plot: str, output_dir: Path) -> list[str]:
    requested = ALL_PLOTS if plot == "all" else (plot,)
    is_all = plot == "all"
    written: list[str] = []
    with line_plot_font_context():
        for name in requested:
            if name == "init-variance":
                written.extend(plot_init_variance(records, output_dir))
            elif name == "near-zero":
                written.extend(plot_near_zero_depth(records, output_dir))
            elif name == "rms-depth":
                written.extend(plot_rms_depth(records, output_dir))
            elif name == "layerwise-flow":
                written.extend(plot_layerwise_flow(records, output_dir))
            elif name == "checkpoint-evolution":
                written.extend(plot_checkpoint_evolution(records, output_dir))
            elif name == "fisher":
                written.extend(plot_fisher(records, output_dir, include_checkpoint=not is_all))
            elif name == "group-balance":
                written.extend(plot_group_balance(records, output_dir))
    return written


def load_records(input_dir: Path) -> list[DiagnosticRecord]:
    payloads = load_payloads(input_dir)
    records = complete_records(payloads)
    if not records:
        raise ValueError(f"no complete gradient diagnostic records found in {input_dir}")
    return records


def load_payloads(input_dir: Path) -> list[dict[str, Any]]:
    payloads = []
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            payloads.append(payload)
    if not payloads:
        raise ValueError(f"no merged gradient diagnostic JSON files found in {input_dir}")
    return payloads


def complete_records(payloads: list[dict[str, Any]]) -> list[DiagnosticRecord]:
    records = []
    for payload in payloads:
        payload_mode = str(payload.get("mode", ""))
        for raw in payload.get("records", []):
            if not isinstance(raw, dict):
                continue
            if not str(raw.get("status", "")).startswith("complete"):
                continue
            dataset = str(raw.get("dataset", ""))
            representation = str(raw.get("representation", ""))
            key = (dataset, representation)
            records.append(
                DiagnosticRecord(
                    mode=str(raw.get("diagnostic_mode") or payload_mode),
                    dataset=dataset,
                    representation=representation,
                    panel_key=key,
                    panel_label=panel_label(key),
                    encoder=canonical_encoder_name(str(raw.get("encoder", ""))),
                    depth=int(raw["depth"]),
                    seed=optional_int(raw.get("seed")),
                    checkpoint=optional_str(raw.get("checkpoint")),
                    checkpoint_step=optional_int(raw.get("checkpoint_step")),
                    raw=raw,
                )
            )
    return records


def optional_int(value: Any) -> int | None:
    if value in {"", None}:
        return None
    return int(value)


def optional_str(value: Any) -> str | None:
    if value in {"", None}:
        return None
    return str(value)


def panel_label(panel_key: tuple[str, str]) -> str:
    if panel_key in PANEL_LABELS:
        return PANEL_LABELS[panel_key]
    dataset, representation = panel_key
    return f"{dataset} {representation}".replace("_", " ").replace("-", " ").strip().title()


def finite_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    current = float(value)
    if not math.isfinite(current):
        return None
    return current


def group_metric_rows(
    records: list[DiagnosticRecord],
    metric: str,
    *,
    modes: set[str] | None = None,
) -> list[GroupMetricRow]:
    rows = []
    for record in records:
        if modes is not None and record.mode not in modes:
            continue
        groups = record.raw.get("groups", {})
        if not isinstance(groups, dict):
            continue
        for group, summary in groups.items():
            if not isinstance(summary, dict):
                continue
            value = finite_float(summary.get(metric))
            if value is None:
                continue
            rows.append(
                GroupMetricRow(
                    mode=record.mode,
                    panel_key=record.panel_key,
                    panel_label=record.panel_label,
                    encoder=record.encoder,
                    depth=record.depth,
                    group=str(group),
                    metric=metric,
                    value=value,
                    seed=record.seed,
                    checkpoint_step=record.checkpoint_step,
                )
            )
    return rows


def fisher_metric_rows(
    records: list[DiagnosticRecord],
    metric: str,
    *,
    modes: set[str] | None = None,
) -> list[FisherMetricRow]:
    rows = []
    for record in records:
        if modes is not None and record.mode not in modes:
            continue
        fisher = record.raw.get("fisher", {})
        summary = fisher.get("summary", {}) if isinstance(fisher, dict) else {}
        value = finite_float(summary.get(metric))
        if value is None:
            continue
        rows.append(
            FisherMetricRow(
                mode=record.mode,
                panel_key=record.panel_key,
                panel_label=record.panel_label,
                encoder=record.encoder,
                depth=record.depth,
                metric=metric,
                value=value,
                seed=record.seed,
                checkpoint_step=record.checkpoint_step,
            )
        )
    return rows


def layerwise_flow_rows(
    records: list[DiagnosticRecord],
    *,
    modes: set[str] | None = None,
) -> list[LayerwiseFlowRow]:
    rows = []
    for record in records:
        if modes is not None and record.mode not in modes:
            continue
        flow = record.raw.get("layerwise_flow", {})
        if not isinstance(flow, dict):
            continue
        for group, values in flow.items():
            if not isinstance(values, list) or not values:
                continue
            n_layers = len(values)
            for index, raw_value in enumerate(values):
                value = finite_float(raw_value)
                if value is None:
                    continue
                layer_fraction = 0.0 if n_layers == 1 else index / (n_layers - 1)
                rows.append(
                    LayerwiseFlowRow(
                        mode=record.mode,
                        panel_key=record.panel_key,
                        panel_label=record.panel_label,
                        encoder=record.encoder,
                        depth=record.depth,
                        group=str(group),
                        layer_fraction=layer_fraction,
                        n_layers=n_layers,
                        value=value,
                        seed=record.seed,
                        checkpoint_step=record.checkpoint_step,
                    )
                )
    return rows


def balance_ratio_rows(
    records: list[DiagnosticRecord],
    *,
    modes: set[str] | None = None,
) -> list[BalanceRatioRow]:
    rows = []
    for record in records:
        if modes is not None and record.mode not in modes:
            continue
        groups = record.raw.get("groups", {})
        if not isinstance(groups, dict):
            continue
        for numerator, denominator, label in BALANCE_RATIOS:
            numerator_value = group_rms(groups, numerator)
            denominator_value = group_rms(groups, denominator)
            if numerator_value is None or denominator_value is None or denominator_value <= 0.0:
                continue
            rows.append(
                BalanceRatioRow(
                    mode=record.mode,
                    panel_key=record.panel_key,
                    panel_label=record.panel_label,
                    encoder=record.encoder,
                    depth=record.depth,
                    ratio=label,
                    value=numerator_value / denominator_value,
                    seed=record.seed,
                    checkpoint_step=record.checkpoint_step,
                )
            )
    return rows


def group_rms(groups: dict[str, Any], group: str) -> float | None:
    summary = groups.get(group)
    if not isinstance(summary, dict):
        return None
    return finite_float(summary.get("mean_rms_gradient"))


def aggregate_by_key[T](rows: list[T], key_fn) -> dict[Any, Summary]:
    grouped: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row.value)
    return {key: summarize(values) for key, values in grouped.items()}


def summarize(values: list[float]) -> Summary:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return Summary(mean=mean, std=0.0, n=len(values))
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return Summary(mean=mean, std=variance**0.5, n=len(values))


def ordered_panels(rows) -> list[tuple[str, str]]:
    present = {row.panel_key for row in rows}
    ordered = [key for key, _ in PANEL_ORDER if key in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def ordered_groups(rows) -> list[str]:
    present = {row.group for row in rows}
    ordered = [group for group in GROUP_ORDER if group in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def ordered_ratios(rows: list[BalanceRatioRow]) -> list[str]:
    defined = [label for _, _, label in BALANCE_RATIOS]
    present = {row.ratio for row in rows}
    ordered = [ratio for ratio in defined if ratio in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def ordered_encoders(rows) -> list[str]:
    present = {row.encoder for row in rows}
    ordered = [encoder for encoder in ENCODER_STYLES if encoder in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def ordered_depths(rows) -> list[int]:
    return sorted({row.depth for row in rows})


def line_style_for_depth(depth: int, depths: list[int]) -> str:
    try:
        index = depths.index(depth)
    except ValueError:
        index = 0
    return DEPTH_LINESTYLES[index % len(DEPTH_LINESTYLES)]


def plot_init_variance(records: list[DiagnosticRecord], output_dir: Path) -> list[str]:
    rows = group_metric_rows(records, "median_log10_variance", modes={"init"})
    output_base = output_dir / "gradient_diagnostics__init_variance_vs_depth"
    with matplotlib.rc_context(MAIN_GRADIENT_FONT_SIZES):
        return plot_group_metric_depth_grid(
            rows,
            output_base,
            y_label="median log10\ngradient variance",
            y_log=False,
            paper_layout=True,
        )


def plot_near_zero_depth(records: list[DiagnosticRecord], output_dir: Path) -> list[str]:
    written = []
    for mode in MODES:
        rows = group_metric_rows(records, "near_zero_fraction", modes={mode})
        output_base = output_dir / f"gradient_diagnostics__{mode}_near_zero_vs_depth"
        written.extend(
            plot_group_metric_depth_grid(
                rows,
                output_base,
                y_label="near-zero gradient fraction",
                y_log=False,
                y_limits=NEAR_ZERO_Y_LIMITS,
            )
        )
    return written


def plot_rms_depth(records: list[DiagnosticRecord], output_dir: Path) -> list[str]:
    written = []
    with matplotlib.rc_context(MAIN_GRADIENT_FONT_SIZES):
        for mode in MODES:
            rows = group_metric_rows(records, "mean_rms_gradient", modes={mode})
            output_base = output_dir / f"gradient_diagnostics__{mode}_rms_gradient_vs_depth"
            written.extend(
                plot_group_metric_depth_grid(
                    rows,
                    output_base,
                    y_label="mean RMS\ngradient",
                    y_log=True,
                    paper_layout=True,
                )
            )
    return written


def plot_group_metric_depth_grid(
    rows: list[GroupMetricRow],
    output_base: Path,
    *,
    y_label: str,
    y_log: bool,
    y_limits: tuple[float, float] | None = None,
    paper_layout: bool = False,
) -> list[str]:
    if not rows:
        return []
    panels = ordered_panels(rows)
    groups = ordered_groups(rows)
    summaries = aggregate_by_key(
        rows,
        lambda row: (row.panel_key, row.group, row.encoder, row.depth),
    )

    fig, axes = plt.subplots(
        len(groups),
        len(panels),
        figsize=(
            4.1 * len(panels),
            (MAIN_GRADIENT_ROW_HEIGHT if paper_layout else 2.7) * len(groups),
        ),
        squeeze=False,
        sharex=True,
    )
    legend_handles: dict[str, Any] = {}
    for row_index, group in enumerate(groups):
        for col_index, panel in enumerate(panels):
            axis = axes[row_index, col_index]
            panel_group_rows = [
                row for row in rows if row.panel_key == panel and row.group == group
            ]
            for encoder in ordered_encoders(panel_group_rows):
                depths = [
                    depth
                    for depth in ordered_depths(panel_group_rows)
                    if (panel, group, encoder, depth) in summaries
                ]
                if not depths:
                    continue
                means = [summaries[(panel, group, encoder, depth)].mean for depth in depths]
                stds = [summaries[(panel, group, encoder, depth)].std for depth in depths]
                plot_means = positive_or_nan(means) if y_log else means
                handle = axis.errorbar(
                    depths,
                    plot_means,
                    yerr=stds,
                    capsize=2,
                    label=encoder,
                    linewidth=1.15,
                    markersize=2.8,
                    markerfacecolor="white",
                    markeredgewidth=0.8,
                    **encoder_plot_style(encoder),
                )
                legend_handles.setdefault(encoder, handle)
            style_depth_axis(axis, y_log=y_log, y_limits=y_limits)
            if row_index == 0:
                axis.set_title(panel_label(panel))
            if col_index == 0:
                group_text = group_label(group)
                if paper_layout and group == "gamma":
                    group_text = "Trainable\nfrequencies"
                axis.set_ylabel(f"{group_text}\n{y_label}")
            if not panel_group_rows:
                empty_axis_text(axis)

    finish_grid_figure(fig, legend_handles)
    if paper_layout:
        fig.subplots_adjust(right=MAIN_GRADIENT_RIGHT)
        shift_grid_axes_inside_figure(fig)
    return save_figure(fig, output_base)


def shift_grid_axes_inside_figure(fig) -> None:
    """Shift axes horizontally so their tight bounds stay inside the figure."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = [axis for axis in fig.axes if axis.get_visible()]
    tight_boxes = [
        axis.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        for axis in axes
    ]
    if not tight_boxes:
        return

    margin = 0.002
    min_shift = margin - min(box.x0 for box in tight_boxes)
    max_shift = 1.0 - margin - max(box.x1 for box in tight_boxes)
    if min_shift > max_shift:
        return
    shift = min(max(0.0, min_shift), max_shift)
    if abs(shift) < 1e-6:
        return

    for axis in axes:
        position = axis.get_position()
        axis.set_position(
            [position.x0 + shift, position.y0, position.width, position.height]
        )


def positive_or_nan(values: list[float]) -> list[float]:
    return [value if value > 0.0 else math.nan for value in values]


def style_depth_axis(
    axis,
    *,
    y_log: bool,
    y_limits: tuple[float, float] | None = None,
) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Re-upload depth")
    if y_log:
        axis.set_yscale("log")
    if y_limits is not None:
        axis.set_ylim(*y_limits)
    axis.grid(True, alpha=0.25)


def group_label(group: str) -> str:
    return GROUP_LABELS.get(group, group.replace("_", " "))


def plot_layerwise_flow(records: list[DiagnosticRecord], output_dir: Path) -> list[str]:
    written = []
    for mode in MODES:
        rows = layerwise_flow_rows(records, modes={mode})
        output_base = output_dir / f"gradient_diagnostics__{mode}_layerwise_flow"
        written.extend(plot_layerwise_flow_grid(rows, output_base))
    return written


def plot_layerwise_flow_grid(
    rows: list[LayerwiseFlowRow],
    output_base: Path,
) -> list[str]:
    rows = rows_with_consistent_layer_counts(rows)
    if not rows:
        return []
    panels = ordered_panels(rows)
    groups = ordered_groups(rows)
    depths = ordered_depths(rows)
    summaries = aggregate_by_key(
        rows,
        lambda row: (
            row.panel_key,
            row.group,
            row.encoder,
            row.depth,
            row.layer_fraction,
        ),
    )

    fig, axes = plt.subplots(
        len(groups),
        len(panels),
        figsize=(4.1 * len(panels), 2.7 * len(groups)),
        squeeze=False,
        sharex=True,
        sharey=False,
    )
    legend_handles: dict[str, Any] = {}
    for row_index, group in enumerate(groups):
        for col_index, panel in enumerate(panels):
            axis = axes[row_index, col_index]
            panel_group_rows = [
                row for row in rows if row.panel_key == panel and row.group == group
            ]
            for encoder in ordered_encoders(panel_group_rows):
                for depth in ordered_depths(panel_group_rows):
                    points = sorted(
                        {
                            row.layer_fraction
                            for row in panel_group_rows
                            if row.encoder == encoder and row.depth == depth
                        }
                    )
                    if not points:
                        continue
                    means = [
                        summaries[(panel, group, encoder, depth, point)].mean for point in points
                    ]
                    style = encoder_plot_style(encoder)
                    handle = axis.plot(
                        points,
                        positive_or_nan(means),
                        label=f"{encoder} L{depth}",
                        linewidth=1.05,
                        markersize=2.2,
                        markerfacecolor="white",
                        linestyle=line_style_for_depth(depth, depths),
                        **style,
                    )[0]
                    legend_handles.setdefault(f"{encoder} L{depth}", handle)
            axis.set_yscale("log")
            axis.set_xlabel("Normalized layer index")
            axis.grid(True, alpha=0.25)
            if row_index == 0:
                axis.set_title(panel_label(panel))
            if col_index == 0:
                axis.set_ylabel(f"{group_label(group)}\nlayerwise RMS gradient")
            if not panel_group_rows:
                empty_axis_text(axis)

    finish_grid_figure(fig, legend_handles, max_legend_columns=5)
    return save_figure(fig, output_base)


def rows_with_consistent_layer_counts(
    rows: list[LayerwiseFlowRow],
) -> list[LayerwiseFlowRow]:
    counts: dict[tuple[str, tuple[str, str], str, str, int], set[int]] = defaultdict(set)
    for row in rows:
        counts[(row.mode, row.panel_key, row.group, row.encoder, row.depth)].add(row.n_layers)
    compatible_keys = {key for key, values in counts.items() if len(values) == 1}
    return [
        row
        for row in rows
        if (row.mode, row.panel_key, row.group, row.encoder, row.depth) in compatible_keys
    ]


def plot_checkpoint_evolution(records: list[DiagnosticRecord], output_dir: Path) -> list[str]:
    written = []
    metric_specs = (
        ("mean_rms_gradient", "mean RMS gradient", True),
        ("near_zero_fraction", "near-zero gradient fraction", False),
    )
    for metric, label, y_log in metric_specs:
        rows = [
            row
            for row in group_metric_rows(records, metric, modes={"checkpoints"})
            if row.checkpoint_step is not None
        ]
        output_base = output_dir / f"gradient_diagnostics__checkpoint_{metric}"
        written.extend(
            plot_checkpoint_group_grid(
                rows,
                output_base,
                y_label=label,
                y_log=y_log,
                y_limits=NEAR_ZERO_Y_LIMITS if metric == "near_zero_fraction" else None,
            )
        )
    written.extend(plot_checkpoint_fisher_grid(records, output_dir))
    return written


def plot_checkpoint_group_grid(
    rows: list[GroupMetricRow],
    output_base: Path,
    *,
    y_label: str,
    y_log: bool,
    y_limits: tuple[float, float] | None = None,
) -> list[str]:
    if not rows:
        return []
    panels = ordered_panels(rows)
    groups = ordered_groups(rows)
    depths = ordered_depths(rows)
    summaries = aggregate_by_key(
        rows,
        lambda row: (
            row.panel_key,
            row.group,
            row.encoder,
            row.depth,
            row.checkpoint_step,
        ),
    )

    fig, axes = plt.subplots(
        len(groups),
        len(panels),
        figsize=(4.1 * len(panels), 2.7 * len(groups)),
        squeeze=False,
        sharex=True,
    )
    legend_handles: dict[str, Any] = {}
    for row_index, group in enumerate(groups):
        for col_index, panel in enumerate(panels):
            axis = axes[row_index, col_index]
            panel_group_rows = [
                row for row in rows if row.panel_key == panel and row.group == group
            ]
            for encoder in ordered_encoders(panel_group_rows):
                for depth in ordered_depths(panel_group_rows):
                    steps = sorted(
                        {
                            row.checkpoint_step
                            for row in panel_group_rows
                            if row.encoder == encoder
                            and row.depth == depth
                            and row.checkpoint_step is not None
                        }
                    )
                    if not steps:
                        continue
                    means = [
                        summaries[(panel, group, encoder, depth, step)].mean for step in steps
                    ]
                    style = encoder_plot_style(encoder)
                    handle = axis.plot(
                        steps,
                        positive_or_nan(means) if y_log else means,
                        label=f"{encoder} L{depth}",
                        linewidth=1.05,
                        markersize=2.2,
                        markerfacecolor="white",
                        linestyle=line_style_for_depth(depth, depths),
                        **style,
                    )[0]
                    legend_handles.setdefault(f"{encoder} L{depth}", handle)
            style_checkpoint_axis(axis, y_log=y_log, y_limits=y_limits)
            if row_index == 0:
                axis.set_title(panel_label(panel))
            if col_index == 0:
                axis.set_ylabel(f"{group_label(group)}\n{y_label}")
            if not panel_group_rows:
                empty_axis_text(axis)

    finish_grid_figure(fig, legend_handles, max_legend_columns=5)
    return save_figure(fig, output_base)


def style_checkpoint_axis(
    axis,
    *,
    y_log: bool,
    y_limits: tuple[float, float] | None = None,
) -> None:
    axis.set_xlabel("Checkpoint step")
    if y_log:
        axis.set_yscale("log")
    if y_limits is not None:
        axis.set_ylim(*y_limits)
    axis.grid(True, alpha=0.25)


def plot_checkpoint_fisher_grid(
    records: list[DiagnosticRecord],
    output_dir: Path,
) -> list[str]:
    rows_by_metric = {
        metric: [
            row
            for row in fisher_metric_rows(records, metric, modes={"checkpoints"})
            if row.checkpoint_step is not None
        ]
        for metric, _, _ in FISHER_METRICS
    }
    rows = [row for metric_rows in rows_by_metric.values() for row in metric_rows]
    if not rows:
        return []
    output_base = output_dir / "gradient_diagnostics__checkpoint_fisher"
    return plot_fisher_checkpoint_grid(rows_by_metric, output_base)


def plot_fisher_checkpoint_grid(
    rows_by_metric: dict[str, list[FisherMetricRow]],
    output_base: Path,
) -> list[str]:
    rows = [row for metric_rows in rows_by_metric.values() for row in metric_rows]
    if not rows:
        return []
    panels = ordered_panels(rows)
    depths = ordered_depths(rows)
    metrics = [spec for spec in FISHER_METRICS if rows_by_metric.get(spec[0])]
    summaries = aggregate_by_key(
        rows,
        lambda row: (
            row.metric,
            row.panel_key,
            row.encoder,
            row.depth,
            row.checkpoint_step,
        ),
    )

    fig, axes = plt.subplots(
        len(metrics),
        len(panels),
        figsize=(4.1 * len(panels), 2.7 * len(metrics)),
        squeeze=False,
        sharex=True,
    )
    legend_handles: dict[str, Any] = {}
    for row_index, (metric, label, y_log) in enumerate(metrics):
        for col_index, panel in enumerate(panels):
            axis = axes[row_index, col_index]
            metric_panel_rows = [
                row for row in rows_by_metric[metric] if row.panel_key == panel
            ]
            for encoder in ordered_encoders(metric_panel_rows):
                for depth in ordered_depths(metric_panel_rows):
                    steps = sorted(
                        {
                            row.checkpoint_step
                            for row in metric_panel_rows
                            if row.encoder == encoder
                            and row.depth == depth
                            and row.checkpoint_step is not None
                        }
                    )
                    if not steps:
                        continue
                    means = [
                        summaries[(metric, panel, encoder, depth, step)].mean for step in steps
                    ]
                    style = encoder_plot_style(encoder)
                    handle = axis.plot(
                        steps,
                        positive_or_nan(means) if y_log else means,
                        label=f"{encoder} L{depth}",
                        linewidth=1.05,
                        markersize=2.2,
                        markerfacecolor="white",
                        linestyle=line_style_for_depth(depth, depths),
                        **style,
                    )[0]
                    legend_handles.setdefault(f"{encoder} L{depth}", handle)
            style_checkpoint_axis(axis, y_log=y_log)
            if row_index == 0:
                axis.set_title(panel_label(panel))
            if col_index == 0:
                axis.set_ylabel(label)
            if not metric_panel_rows:
                empty_axis_text(axis)

    finish_grid_figure(fig, legend_handles, max_legend_columns=5)
    return save_figure(fig, output_base)


def plot_fisher(
    records: list[DiagnosticRecord],
    output_dir: Path,
    *,
    include_checkpoint: bool = True,
) -> list[str]:
    written = []
    with matplotlib.rc_context(MAIN_GRADIENT_FONT_SIZES):
        for mode in ("init", "final"):
            output_base = output_dir / f"gradient_diagnostics__{mode}_fisher_vs_depth"
            written.extend(plot_fisher_depth_grid(records, mode, output_base))
    if include_checkpoint:
        written.extend(plot_checkpoint_fisher_grid(records, output_dir))
    return written


def plot_fisher_depth_grid(
    records: list[DiagnosticRecord],
    mode: str,
    output_base: Path,
) -> list[str]:
    rows_by_metric = {
        metric: fisher_metric_rows(records, metric, modes={mode}) for metric, _, _ in FISHER_METRICS
    }
    rows = [row for metric_rows in rows_by_metric.values() for row in metric_rows]
    if not rows:
        return []
    panels = ordered_panels(rows)
    metrics = [spec for spec in FISHER_METRICS if rows_by_metric.get(spec[0])]
    summaries = aggregate_by_key(
        rows,
        lambda row: (row.metric, row.panel_key, row.encoder, row.depth),
    )

    fig, axes = plt.subplots(
        len(metrics),
        len(panels),
        figsize=(4.1 * len(panels), MAIN_GRADIENT_ROW_HEIGHT * len(metrics)),
        squeeze=False,
        sharex=True,
    )
    legend_handles: dict[str, Any] = {}
    for row_index, (metric, label, y_log) in enumerate(metrics):
        for col_index, panel in enumerate(panels):
            axis = axes[row_index, col_index]
            metric_panel_rows = [
                row for row in rows_by_metric[metric] if row.panel_key == panel
            ]
            for encoder in ordered_encoders(metric_panel_rows):
                depths = [
                    depth
                    for depth in ordered_depths(metric_panel_rows)
                    if (metric, panel, encoder, depth) in summaries
                ]
                if not depths:
                    continue
                means = [summaries[(metric, panel, encoder, depth)].mean for depth in depths]
                stds = [summaries[(metric, panel, encoder, depth)].std for depth in depths]
                handle = axis.errorbar(
                    depths,
                    positive_or_nan(means) if y_log else means,
                    yerr=stds,
                    capsize=2,
                    label=encoder,
                    linewidth=1.15,
                    markersize=2.8,
                    markerfacecolor="white",
                    markeredgewidth=0.8,
                    **encoder_plot_style(encoder),
                )
                legend_handles.setdefault(encoder, handle)
            style_depth_axis(axis, y_log=y_log)
            if row_index == 0:
                axis.set_title(panel_label(panel))
            if col_index == 0:
                axis.set_ylabel(label.replace("Fisher ", "Fisher\n", 1))
            if not metric_panel_rows:
                empty_axis_text(axis)

    finish_grid_figure(fig, legend_handles)
    return save_figure(fig, output_base)


def plot_group_balance(records: list[DiagnosticRecord], output_dir: Path) -> list[str]:
    written = []
    for mode in MODES:
        rows = balance_ratio_rows(records, modes={mode})
        output_base = output_dir / f"gradient_diagnostics__{mode}_group_balance_vs_depth"
        written.extend(plot_balance_depth_grid(rows, output_base))
    return written


def plot_balance_depth_grid(
    rows: list[BalanceRatioRow],
    output_base: Path,
) -> list[str]:
    if not rows:
        return []
    panels = ordered_panels(rows)
    ratios = ordered_ratios(rows)
    summaries = aggregate_by_key(
        rows,
        lambda row: (row.panel_key, row.ratio, row.encoder, row.depth),
    )

    fig, axes = plt.subplots(
        len(ratios),
        len(panels),
        figsize=(4.1 * len(panels), 2.7 * len(ratios)),
        squeeze=False,
        sharex=True,
    )
    legend_handles: dict[str, Any] = {}
    for row_index, ratio in enumerate(ratios):
        for col_index, panel in enumerate(panels):
            axis = axes[row_index, col_index]
            panel_ratio_rows = [
                row for row in rows if row.panel_key == panel and row.ratio == ratio
            ]
            for encoder in ordered_encoders(panel_ratio_rows):
                depths = [
                    depth
                    for depth in ordered_depths(panel_ratio_rows)
                    if (panel, ratio, encoder, depth) in summaries
                ]
                if not depths:
                    continue
                means = [summaries[(panel, ratio, encoder, depth)].mean for depth in depths]
                stds = [summaries[(panel, ratio, encoder, depth)].std for depth in depths]
                handle = axis.errorbar(
                    depths,
                    positive_or_nan(means),
                    yerr=stds,
                    capsize=2,
                    label=encoder,
                    linewidth=1.15,
                    markersize=2.8,
                    markerfacecolor="white",
                    markeredgewidth=0.8,
                    **encoder_plot_style(encoder),
                )
                legend_handles.setdefault(encoder, handle)
            style_depth_axis(axis, y_log=True)
            if row_index == 0:
                axis.set_title(panel_label(panel))
            if col_index == 0:
                axis.set_ylabel(f"{ratio}\nRMS-gradient ratio")
            if not panel_ratio_rows:
                empty_axis_text(axis)

    finish_grid_figure(fig, legend_handles)
    return save_figure(fig, output_base)


def empty_axis_text(axis) -> None:
    axis.text(0.5, 0.5, "No data", transform=axis.transAxes, ha="center", va="center")


def finish_grid_figure(
    fig,
    legend_handles: dict[str, Any],
    *,
    max_legend_columns: int = 4,
) -> Any | None:
    if legend_handles:
        n_entries = len(legend_handles)
        n_columns = min(max_legend_columns, max(1, n_entries))
        return add_bottom_legend(
            fig,
            list(legend_handles.values()),
            list(legend_handles),
            ncol=n_columns,
            font_size=matplotlib.rcParams["legend.fontsize"],
        )
    fig.tight_layout()
    return None


def save_figure(fig, output_base: Path) -> list[str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = f"{output_base}.pdf"
    png_path = f"{output_base}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return [pdf_path, png_path]


if __name__ == "__main__":
    main()
