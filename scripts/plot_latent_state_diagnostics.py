#!/usr/bin/env python3
"""Plot aggregated latent-state diagnostics as static paper figures."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path("/tmp") / "qfm_matplotlib_config"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ham_embed_spectral.naming import canonical_encoder_name  # noqa: E402
from ham_embed_spectral.plotting import (  # noqa: E402
    add_bottom_legend,
    add_style_arguments,
    bumped_font_size,
    line_plot_font_context,
    print_available_styles,
    use_style,
)
from scripts.plot_qfm_results import ENCODER_STYLES, encoder_plot_style  # noqa: E402

DEFAULT_RECORDS_CSV = "results/tables/latent_state_diagnostics/batched_summary_records.csv"
DEFAULT_LAYERWISE_CSV = "results/tables/latent_state_diagnostics/batched_summary_layerwise.csv"
DEFAULT_CKA_CSV = "results/tables/latent_state_diagnostics/batched_summary_cka.csv"
DEFAULT_SPECTRAL_STATE_CSV = (
    "results/tables/latent_state_diagnostics/batched_summary_spectral_state.csv"
)
DEFAULT_OUTPUT_DIR = "results/figures/latent_state_diagnostics"
X_AXIS_LABEL_WRAP_WIDTH = 22
Y_AXIS_LABEL_WRAP_WIDTH = 18
SAVEFIG_PAD_INCHES = 0.05

PLOT_CHOICES = (
    "summary",
    "trajectories",
    "layerwise",
    "cka",
    "spectral-state",
    "scatter",
    "all",
)
ALL_PLOTS = tuple(choice for choice in PLOT_CHOICES if choice != "all")

PANEL_ORDER = (
    (("pendigits", "dyn"), "Pendigits DYN"),
    (("pendigits", "sta4"), "Pendigits STA4"),
    (("synthetic-eigengap", "synthetic"), "Synthetic eigengap"),
    (("synthetic-singular", "synthetic"), "Synthetic singular"),
)
PANEL_INDEX = {key: index for index, (key, _) in enumerate(PANEL_ORDER)}
PANEL_LABELS = dict(PANEL_ORDER)

ENCODER_ORDER = tuple(ENCODER_STYLES)
ENCODER_INDEX = {encoder: index for index, encoder in enumerate(ENCODER_ORDER)}
ENCODER_LABELS = {
    "fixed-ry": "Fixed Ry",
    "fixed-ry-rz": "Fixed Ry/Rz",
    "trainable-frequency-ry": "Trainable-frequency Ry",
    "patch-su4": "Patch SU(4)",
    "trainable-patch-su4": "Trainable patch SU(4)",
    "non-overlap-patch-block-hamiltonian": "Patch H_block",
    "symmetric-hamiltonian": "H_sym",
    "block-hamiltonian": "H_block",
}
DEPTH_MARKERS = ("o", "s", "^", "D", "v")


@dataclass(frozen=True)
class MetricSpec:
    """One scalar metric that can be plotted from an aggregate CSV table."""

    slug: str
    column: str
    label: str
    vmin: float | None = None
    vmax: float | None = None


@dataclass(frozen=True)
class LayerwiseMetricSpec:
    """One layerwise metric and the row family/stage where it lives."""

    slug: str
    family: str
    column: str
    label: str
    stage: str = "post_mixer"


@dataclass(frozen=True)
class ScatterSpec:
    """One final-record scatter diagnostic against logit accuracy."""

    slug: str
    x_column: str
    x_label: str


RECORD_METRICS = (
    MetricSpec(
        "projector_accuracy",
        "final_projector_accuracy_on_diagnostic_batch",
        "Diagnostic projector accuracy",
        0.0,
        1.0,
    ),
    MetricSpec("logit_accuracy", "logit_final_accuracy", "Logit final accuracy", 0.0, 1.0),
    MetricSpec(
        "kernel_target_alignment",
        "final_kernel_target_alignment",
        "Final kernel target alignment",
    ),
    MetricSpec(
        "kernel_effective_rank",
        "final_kernel_effective_rank",
        "Final kernel effective rank",
    ),
    MetricSpec(
        "kernel_centered_effective_rank",
        "final_kernel_centered_effective_rank",
        "Final centered kernel effective rank",
    ),
    MetricSpec(
        "kernel_gap",
        "final_kernel_mean_gap_within_minus_between",
        "Final within-between fidelity gap",
    ),
    MetricSpec(
        "logit_path_length",
        "logit_total_path_length_mean",
        "Mean logit path length",
    ),
)

LAYERWISE_METRICS = (
    LayerwiseMetricSpec(
        "projector_accuracy",
        "projector_probe",
        "projector_accuracy",
        "Layerwise projector accuracy",
    ),
    LayerwiseMetricSpec(
        "projector_top_score",
        "projector_probe",
        "projector_mean_top_score",
        "Layerwise mean top projector score",
    ),
    LayerwiseMetricSpec(
        "kernel_target_alignment",
        "fidelity_kernel",
        "target_alignment",
        "Layerwise kernel target alignment",
    ),
    LayerwiseMetricSpec(
        "kernel_effective_rank",
        "fidelity_kernel",
        "effective_rank",
        "Layerwise kernel effective rank",
    ),
    LayerwiseMetricSpec(
        "kernel_gap",
        "fidelity_kernel",
        "mean_gap_within_minus_between",
        "Layerwise within-between fidelity gap",
    ),
)

SPECTRAL_STATE_METRICS = (
    MetricSpec("occupation_l1_change", "value_mean", "Mean occupation L1 change"),
    MetricSpec("phase_increment_abs_mean", "value_mean", "Mean absolute phase increment"),
    MetricSpec("phase_increment_abs_max", "value_mean", "Max absolute phase increment"),
)

SCATTER_SPECS = (
    ScatterSpec(
        "kernel_target_alignment",
        "final_kernel_target_alignment",
        "Final kernel target alignment",
    ),
    ScatterSpec(
        "kernel_effective_rank",
        "final_kernel_effective_rank",
        "Final kernel effective rank",
    ),
    ScatterSpec(
        "kernel_gap",
        "final_kernel_mean_gap_within_minus_between",
        "Final within-between fidelity gap",
    ),
    ScatterSpec(
        "logit_path_length",
        "logit_total_path_length_mean",
        "Mean logit path length",
    ),
)
SCATTER_Y_COLUMN = "logit_final_accuracy"
SCATTER_Y_LABEL = "Logit final accuracy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--records-csv", default=DEFAULT_RECORDS_CSV)
    parser.add_argument("--layerwise-csv", default=DEFAULT_LAYERWISE_CSV)
    parser.add_argument("--cka-csv", default=DEFAULT_CKA_CSV)
    parser.add_argument("--spectral-state-csv", default=DEFAULT_SPECTRAL_STATE_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot", choices=PLOT_CHOICES, default="all")
    add_style_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_styles:
        print_available_styles()
        return
    use_style(args.style)
    written = plot_requested(args)
    if not written:
        raise ValueError(f"no latent diagnostic figures were produced for --plot {args.plot!r}")
    for path in written:
        print(path)


def plot_requested(args: argparse.Namespace) -> list[str]:
    requested = ALL_PLOTS if args.plot == "all" else (args.plot,)
    output_dir = Path(args.output_dir)
    written: list[str] = []

    records: list[dict[str, str]] | None = None
    layerwise_rows: list[dict[str, str]] | None = None
    cka_rows: list[dict[str, str]] | None = None
    spectral_rows: list[dict[str, str]] | None = None

    for plot_name in requested:
        if plot_name in {"summary", "trajectories", "scatter"}:
            if records is None:
                records = load_complete_rows(Path(args.records_csv))
            if plot_name == "summary":
                written.extend(plot_summary_heatmaps(records, output_dir))
            elif plot_name == "trajectories":
                with line_plot_font_context():
                    written.extend(plot_checkpoint_trajectories(records, output_dir))
            elif plot_name == "scatter":
                written.extend(plot_final_scatters(records, output_dir))
        elif plot_name == "layerwise":
            if layerwise_rows is None:
                layerwise_rows = load_complete_rows(Path(args.layerwise_csv))
            with line_plot_font_context():
                written.extend(plot_layerwise_profiles(layerwise_rows, output_dir))
        elif plot_name == "cka":
            if cka_rows is None:
                cka_rows = load_complete_rows(Path(args.cka_csv))
            with line_plot_font_context():
                written.extend(plot_cka_profiles(cka_rows, output_dir))
        elif plot_name == "spectral-state":
            if spectral_rows is None:
                spectral_rows = load_complete_rows(Path(args.spectral_state_csv))
            with line_plot_font_context():
                written.extend(plot_spectral_state_profiles(spectral_rows, output_dir))
    return written


def load_complete_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing latent diagnostic CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if is_complete_status(row)]
    if not rows:
        raise ValueError(f"no complete latent diagnostic rows found in {path}")
    return rows


def is_complete_status(row: dict[str, str]) -> bool:
    return str(row.get("status", "")).startswith("complete")


def is_final_record(row: dict[str, str]) -> bool:
    return row.get("mode") == "final" or row.get("diagnostic_mode") == "final"


def is_checkpoint_record(row: dict[str, str]) -> bool:
    return row.get("mode") == "checkpoints" or row.get("diagnostic_mode") == "checkpoints"


def float_value(row: dict[str, str], column: str) -> float | None:
    raw = str(row.get(column, "")).strip()
    if raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def int_value(row: dict[str, str], column: str) -> int | None:
    value = float_value(row, column)
    if value is None:
        return None
    return int(value)


def panel_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("dataset", ""), row.get("representation", ""))


def panel_label(key: tuple[str, str]) -> str:
    if key in PANEL_LABELS:
        return PANEL_LABELS[key]
    dataset, representation = key
    if dataset == "pendigits":
        return f"Pendigits {representation.upper()}"
    return dataset.replace("-", " ").replace("_", " ").title()


def row_encoder(row: dict[str, str]) -> str:
    return canonical_encoder_name(row.get("encoder", ""))


def encoder_label(encoder: str) -> str:
    return ENCODER_LABELS.get(encoder, encoder.replace("-", " "))


def encoder_color(encoder: str) -> str:
    return str(encoder_plot_style(encoder).get("color", "#333333"))


def ordered_panels(rows: list[dict[str, str]]) -> list[tuple[str, str]]:
    keys = {panel_key(row) for row in rows}
    return sorted(keys, key=lambda key: (PANEL_INDEX.get(key, 999), key))


def ordered_depths(rows: list[dict[str, str]]) -> list[int]:
    depths = {depth for row in rows if (depth := int_value(row, "depth")) is not None}
    return sorted(depths)


def ordered_encoders(rows: list[dict[str, str]]) -> list[str]:
    encoders = {row_encoder(row) for row in rows}
    return sorted(encoders, key=lambda encoder: (ENCODER_INDEX.get(encoder, 999), encoder))


def depth_marker(depth: int, depths: list[int]) -> str:
    try:
        index = depths.index(depth)
    except ValueError:
        index = len(depths)
    return DEPTH_MARKERS[index % len(DEPTH_MARKERS)]


def wrap_axis_label(label: str, *, width: int) -> str:
    if len(label) <= width:
        return label
    return fill(label, width=width, break_long_words=False, break_on_hyphens=False)


def set_wrapped_xlabel(axis, label: str) -> None:
    axis.set_xlabel(wrap_axis_label(label, width=X_AXIS_LABEL_WRAP_WIDTH))


def set_wrapped_ylabel(axis, label: str) -> None:
    axis.set_ylabel(wrap_axis_label(label, width=Y_AXIS_LABEL_WRAP_WIDTH))


def mean_by_key(
    rows: list[dict[str, str]],
    metric: str,
    key_columns: tuple[str, ...],
) -> dict[tuple[object, ...], float]:
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = float_value(row, metric)
        if value is None:
            continue
        key: list[object] = []
        for column in key_columns:
            if column == "panel":
                key.append(panel_key(row))
            elif column == "encoder":
                key.append(row_encoder(row))
            elif column in {"depth", "checkpoint_step", "layer_index"}:
                parsed = int_value(row, column)
                if parsed is None:
                    key = []
                    break
                key.append(parsed)
            else:
                key.append(row.get(column, ""))
        if key:
            grouped[tuple(key)].append(value)
    return {key: float(np.mean(values)) for key, values in grouped.items() if values}


def plot_summary_heatmaps(
    records: list[dict[str, str]],
    output_dir: Path,
) -> list[str]:
    final_rows = [row for row in records if is_final_record(row)]
    written: list[str] = []
    for spec in RECORD_METRICS:
        rows = [row for row in final_rows if float_value(row, spec.column) is not None]
        if not rows:
            continue
        panels = ordered_panels(rows)
        encoders = ordered_encoders(rows)
        depths = ordered_depths(rows)
        if not panels or not encoders or not depths:
            continue

        values = mean_by_key(rows, spec.column, ("panel", "encoder", "depth"))
        if not values:
            continue
        data_values = list(values.values())
        vmin = spec.vmin if spec.vmin is not None else min(data_values)
        vmax = spec.vmax if spec.vmax is not None else max(data_values)
        if math.isclose(vmin, vmax):
            vmin -= 0.5
            vmax += 0.5

        panel_columns = min(2, max(1, len(panels)))
        fig, axes = panel_figure(len(panels), width=3.8, height=3.0)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#f1f1f1")
        image = None
        for axis_index, (axis, panel) in enumerate(zip(axes, panels, strict=False)):
            matrix = np.full((len(encoders), len(depths)), np.nan, dtype=float)
            for row_index, encoder in enumerate(encoders):
                for col_index, depth in enumerate(depths):
                    matrix[row_index, col_index] = values.get((panel, encoder, depth), np.nan)
            image = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(panel_label(panel))
            axis.set_xticks(range(len(depths)), [str(depth) for depth in depths])
            axis.set_yticks(range(len(encoders)))
            if axis_index % panel_columns == 0:
                axis.set_yticklabels([encoder_label(encoder) for encoder in encoders])
            else:
                axis.set_yticklabels([])
            set_wrapped_xlabel(axis, "Re-upload depth")
            axis.tick_params(axis="both", labelsize=7)
            for row_index in range(len(encoders)):
                for col_index in range(len(depths)):
                    value = matrix[row_index, col_index]
                    if math.isfinite(value):
                        axis.text(
                            col_index,
                            row_index,
                            format_cell(value),
                            ha="center",
                            va="center",
                            fontsize=5.5,
                            color=text_color(value, vmin, vmax),
                        )
        hide_unused_axes(axes, len(panels))
        fig.subplots_adjust(
            left=0.2,
            right=0.87,
            top=0.94,
            bottom=0.1,
            wspace=0.28,
            hspace=0.45,
        )
        if image is not None:
            cax = fig.add_axes((0.91, 0.18, 0.018, 0.62))
            fig.colorbar(image, cax=cax)
        written.extend(save_figure(fig, output_dir / "summary" / f"latent_summary__{spec.slug}"))
    return written


def plot_checkpoint_trajectories(
    records: list[dict[str, str]],
    output_dir: Path,
) -> list[str]:
    checkpoint_rows = [
        row
        for row in records
        if is_checkpoint_record(row) and int_value(row, "checkpoint_step") is not None
    ]
    written: list[str] = []
    for spec in RECORD_METRICS:
        rows = [row for row in checkpoint_rows if float_value(row, spec.column) is not None]
        if not rows:
            continue
        panels = ordered_panels(rows)
        depths = ordered_depths(rows)
        encoders = ordered_encoders(rows)
        if not panels or not depths or not encoders:
            continue

        fig, axes = panel_depth_figure(panels, depths, width=2.45, height=1.8)
        legend_handles: dict[str, object] = {}
        for panel in panels:
            for depth in depths:
                axis = axes[(panel, depth)]
                axis_rows = [
                    row
                    for row in rows
                    if panel_key(row) == panel and int_value(row, "depth") == depth
                ]
                for encoder in encoders:
                    series = aggregate_series(
                        [row for row in axis_rows if row_encoder(row) == encoder],
                        x_column="checkpoint_step",
                        y_column=spec.column,
                    )
                    if not series:
                        continue
                    steps, means = zip(*series, strict=True)
                    handle = axis.plot(
                        steps,
                        means,
                        linewidth=1.05,
                        markersize=2.4,
                        markerfacecolor="white",
                        markeredgewidth=0.7,
                        label=encoder_label(encoder),
                        **encoder_plot_style(encoder),
                    )[0]
                    legend_handles.setdefault(encoder_label(encoder), handle)
                style_line_axis(axis)
                axis.set_title(f"{panel_label(panel)}, L={depth}", fontsize=bumped_font_size(8))
                set_wrapped_xlabel(axis, "Checkpoint step")
                set_wrapped_ylabel(axis, spec.label if depth == depths[0] else "")
        finish_grid_figure(fig, legend_handles)
        written.extend(
            save_figure(fig, output_dir / "trajectories" / f"latent_trajectory__{spec.slug}")
        )
    return written


def plot_layerwise_profiles(
    layerwise_rows: list[dict[str, str]],
    output_dir: Path,
) -> list[str]:
    final_rows = [row for row in layerwise_rows if is_final_record(row)]
    written: list[str] = []
    for spec in LAYERWISE_METRICS:
        rows = [
            row
            for row in final_rows
            if row.get("metric_family") == spec.family
            and row.get("stage") == spec.stage
            and int_value(row, "layer_index") is not None
            and float_value(row, spec.column) is not None
        ]
        written.extend(
            plot_layer_profile_grid(
                rows,
                metric_column=spec.column,
                ylabel=spec.label,
                output_base=output_dir / "layerwise" / f"latent_layerwise__{spec.slug}",
            )
        )
    return written


def plot_cka_profiles(
    cka_rows: list[dict[str, str]],
    output_dir: Path,
) -> list[str]:
    final_rows = [
        row
        for row in cka_rows
        if is_final_record(row)
        and int_value(row, "layer_index") is not None
        and float_value(row, "cka_value") is not None
    ]
    written: list[str] = []
    families = sorted(
        {
            row.get("comparison_family", "")
            for row in final_rows
            if row.get("comparison_family")
        }
    )
    for family in families:
        rows = [row for row in final_rows if row.get("comparison_family") == family]
        written.extend(
            plot_layer_profile_grid(
                rows,
                metric_column="cka_value",
                ylabel="CKA",
                output_base=output_dir / "cka" / f"latent_cka__{sanitize_slug(family)}",
            )
        )
    return written


def plot_spectral_state_profiles(
    spectral_rows: list[dict[str, str]],
    output_dir: Path,
) -> list[str]:
    final_rows = [
        row
        for row in spectral_rows
        if is_final_record(row)
        and int_value(row, "layer_index") is not None
        and float_value(row, "value_mean") is not None
    ]
    written: list[str] = []
    for metric in SPECTRAL_STATE_METRICS:
        rows = [row for row in final_rows if row.get("metric") == metric.slug]
        written.extend(
            plot_layer_profile_grid(
                rows,
                metric_column=metric.column,
                ylabel=metric.label,
                output_base=(
                    output_dir / "spectral_state" / f"latent_spectral_state__{metric.slug}"
                ),
            )
        )
    return written


def plot_layer_profile_grid(
    rows: list[dict[str, str]],
    *,
    metric_column: str,
    ylabel: str,
    output_base: Path,
) -> list[str]:
    if not rows:
        return []
    panels = ordered_panels(rows)
    depths = ordered_depths(rows)
    encoders = ordered_encoders(rows)
    if not panels or not depths or not encoders:
        return []

    fig, axes = panel_depth_figure(panels, depths, width=2.35, height=1.75)
    legend_handles: dict[str, object] = {}
    for panel in panels:
        for depth in depths:
            axis = axes[(panel, depth)]
            axis_rows = [
                row
                for row in rows
                if panel_key(row) == panel and int_value(row, "depth") == depth
            ]
            for encoder in encoders:
                series = aggregate_series(
                    [row for row in axis_rows if row_encoder(row) == encoder],
                    x_column="layer_index",
                    y_column=metric_column,
                )
                if not series:
                    continue
                layers, means = zip(*series, strict=True)
                handle = axis.plot(
                    layers,
                    means,
                    linewidth=1.05,
                    markersize=2.4,
                    markerfacecolor="white",
                    markeredgewidth=0.7,
                    label=encoder_label(encoder),
                    **encoder_plot_style(encoder),
                )[0]
                legend_handles.setdefault(encoder_label(encoder), handle)
            style_line_axis(axis)
            axis.set_title(f"{panel_label(panel)}, L={depth}", fontsize=bumped_font_size(8))
            set_wrapped_xlabel(axis, "Layer index")
            set_wrapped_ylabel(axis, ylabel if depth == depths[0] else "")
    finish_grid_figure(fig, legend_handles)
    return save_figure(fig, output_base)


def plot_final_scatters(
    records: list[dict[str, str]],
    output_dir: Path,
) -> list[str]:
    final_rows = [row for row in records if is_final_record(row)]
    written: list[str] = []
    for spec in SCATTER_SPECS:
        rows = [
            row
            for row in final_rows
            if float_value(row, spec.x_column) is not None
            and float_value(row, SCATTER_Y_COLUMN) is not None
            and int_value(row, "depth") is not None
        ]
        if not rows:
            continue
        panels = ordered_panels(rows)
        depths = ordered_depths(rows)
        encoders = ordered_encoders(rows)
        fig, axes = panel_figure(len(panels), width=3.2, height=2.6)
        for axis, panel in zip(axes, panels, strict=False):
            panel_rows = [row for row in rows if panel_key(row) == panel]
            for row in panel_rows:
                depth = int_value(row, "depth")
                encoder = row_encoder(row)
                if depth is None:
                    continue
                axis.scatter(
                    float_value(row, spec.x_column),
                    float_value(row, SCATTER_Y_COLUMN),
                    marker=depth_marker(depth, depths),
                    s=30,
                    color=encoder_color(encoder),
                    edgecolor="white",
                    linewidth=0.4,
                    alpha=0.9,
                )
            axis.set_title(panel_label(panel))
            set_wrapped_xlabel(axis, spec.x_label)
            set_wrapped_ylabel(axis, SCATTER_Y_LABEL)
            style_small_axis(axis)
        hide_unused_axes(axes, len(panels))
        add_scatter_legend(fig, encoders, depths)
        fig.tight_layout(rect=(0.0, 0.16, 1.0, 0.99))
        written.extend(save_figure(fig, output_dir / "scatter" / f"latent_scatter__{spec.slug}"))
    return written


def aggregate_series(
    rows: list[dict[str, str]],
    *,
    x_column: str,
    y_column: str,
) -> list[tuple[int, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        x_value = int_value(row, x_column)
        y_value = float_value(row, y_column)
        if x_value is None or y_value is None:
            continue
        grouped[x_value].append(y_value)
    return [(step, float(np.mean(values))) for step, values in sorted(grouped.items()) if values]


def panel_figure(
    n_panels: int,
    *,
    width: float,
    height: float,
    columns: int = 2,
):
    ncols = min(columns, max(1, n_panels))
    nrows = int(math.ceil(max(1, n_panels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width * ncols, height * nrows),
        squeeze=False,
    )
    return fig, list(axes.ravel())


def panel_depth_figure(
    panels: list[tuple[str, str]],
    depths: list[int],
    *,
    width: float,
    height: float,
) -> tuple[object, dict[tuple[tuple[str, str], int], object]]:
    fig, axes_array = plt.subplots(
        len(panels),
        len(depths),
        figsize=(width * len(depths), height * len(panels) + 0.8),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    axes: dict[tuple[tuple[str, str], int], object] = {}
    for row_index, panel in enumerate(panels):
        for col_index, depth in enumerate(depths):
            axes[(panel, depth)] = axes_array[row_index, col_index]
    return fig, axes


def hide_unused_axes(axes: list[object], n_used: int) -> None:
    for axis in axes[n_used:]:
        axis.set_visible(False)


def style_small_axis(axis, *, labelsize: float = 7) -> None:
    axis.grid(True, alpha=0.25, linewidth=0.5)
    axis.tick_params(axis="both", labelsize=labelsize)


def style_line_axis(axis) -> None:
    style_small_axis(axis, labelsize=bumped_font_size(7))


def finish_grid_figure(
    fig,
    legend_handles: dict[str, object],
) -> None:
    if legend_handles:
        add_bottom_legend(
            fig,
            list(legend_handles.values()),
            list(legend_handles),
            ncol=min(4, max(1, len(legend_handles))),
            top=0.99,
            font_size=bumped_font_size("x-small"),
        )
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))


def add_scatter_legend(fig, encoders: list[str], depths: list[int]) -> None:
    encoder_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=encoder_color(encoder),
            markeredgecolor="white",
            label=encoder_label(encoder),
        )
        for encoder in encoders
    ]
    depth_handles = [
        Line2D(
            [0],
            [0],
            marker=depth_marker(depth, depths),
            linestyle="None",
            color="#333333",
            label=f"L={depth}",
        )
        for depth in depths
    ]
    handles = encoder_handles + depth_handles
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(4, len(handles)),
            fontsize="x-small",
            frameon=False,
        )


def text_color(value: float, vmin: float, vmax: float) -> str:
    if math.isclose(vmin, vmax):
        return "black"
    normalized = (value - vmin) / (vmax - vmin)
    return "white" if normalized > 0.55 else "black"


def format_cell(value: float) -> str:
    if abs(value) < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def sanitize_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def save_figure(fig, output_base: Path) -> list[str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = f"{output_base}.pdf"
    png_path = f"{output_base}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=SAVEFIG_PAD_INCHES)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=SAVEFIG_PAD_INCHES)
    plt.close(fig)
    return [pdf_path, png_path]


if __name__ == "__main__":
    main()
