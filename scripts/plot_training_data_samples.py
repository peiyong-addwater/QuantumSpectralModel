#!/usr/bin/env python3
"""Plot representative training-data samples for paper figures."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path("/tmp") / "qfm_matplotlib_config"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ham_embed_spectral._jax_config import enable_x64  # noqa: E402

enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from ham_embed_spectral.data.pendigits import load_pendigits_split  # noqa: E402
from ham_embed_spectral.data.synthetic import (  # noqa: E402
    eigengap_classification,
    singular_value_classification,
)
from ham_embed_spectral.plotting import (  # noqa: E402
    add_style_arguments,
    print_available_styles,
    use_style,
)

DEFAULT_SYNTHETIC_MANIFEST = "configs/experiments/synthetic.json"
DEFAULT_OUTPUT_DIR = "results/figures/training_data"
PENDIGITS_CLASS_LABELS = tuple(range(1, 11))
SYNTHETIC_CLASS_LABELS = (0, 1)
SYNTHETIC_DATASET_ORDER = ("synthetic-eigengap", "synthetic-singular")
SYNTHETIC_DISPLAY_NAMES = {
    "synthetic-eigengap": "Eigengap",
    "synthetic-singular": "Singular",
}


@dataclass(frozen=True)
class SelectedSampleSet:
    """Samples selected by class label from a source array."""

    labels: tuple[int, ...]
    source_row_indices: tuple[int, ...]
    values: np.ndarray


@dataclass(frozen=True)
class PendigitsSampleSet:
    """Paired raw Pendigits DYN and STA4 samples from the same train rows."""

    labels: tuple[int, ...]
    source_row_indices: tuple[int, ...]
    dyn: np.ndarray
    sta4: np.ndarray


@dataclass(frozen=True)
class SyntheticSampleSet:
    """One synthetic task's class-representative training samples."""

    dataset: str
    display_name: str
    labels: tuple[int, ...]
    source_row_indices: tuple[int, ...]
    values: np.ndarray
    settings: dict[str, Any]


@dataclass(frozen=True)
class SyntheticSplitIndices:
    """Source-row indices for the synthetic train/validation/test split."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default="data/raw/pendigits")
    parser.add_argument("--synthetic-manifest", default=DEFAULT_SYNTHETIC_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pendigits-name", default="pendigits_training_samples")
    parser.add_argument("--synthetic-name", default="synthetic_training_samples")
    parser.add_argument("--metadata-name", default="training_data_samples_metadata.json")
    add_style_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_styles:
        print_available_styles()
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_style(args.style)

    pendigits_samples = load_paired_pendigits_samples(Path(args.data_root))
    synthetic_manifest = load_json(Path(args.synthetic_manifest))
    synthetic_samples = load_synthetic_sample_sets(
        Path(args.synthetic_manifest),
        synthetic_manifest,
    )

    pendigits_outputs = save_pendigits_figure(
        pendigits_samples,
        output_dir / args.pendigits_name,
    )
    synthetic_outputs = save_synthetic_figure(
        synthetic_samples,
        output_dir / args.synthetic_name,
    )
    metadata_path = output_dir / args.metadata_name
    metadata = build_metadata(
        data_root=Path(args.data_root),
        synthetic_manifest_path=Path(args.synthetic_manifest),
        synthetic_manifest=synthetic_manifest,
        pendigits_samples=pendigits_samples,
        synthetic_samples=synthetic_samples,
        pendigits_outputs=pendigits_outputs,
        synthetic_outputs=synthetic_outputs,
        metadata_path=metadata_path,
    )
    write_metadata(metadata_path, metadata)

    for path in (*pendigits_outputs, *synthetic_outputs, str(metadata_path)):
        print(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_paired_pendigits_samples(
    data_root: Path,
    *,
    required_labels: tuple[int, ...] = PENDIGITS_CLASS_LABELS,
) -> PendigitsSampleSet:
    """Load raw DYN/STA4 train rows and select the same row per class."""

    dyn_split = load_pendigits_split(data_root, representation="dyn", split="train")
    sta4_split = load_pendigits_split(data_root, representation="sta4", split="train")
    dyn_labels = np.asarray(dyn_split.y, dtype=np.int64)
    sta4_labels = np.asarray(sta4_split.y, dtype=np.int64)
    if dyn_labels.shape != sta4_labels.shape or not np.array_equal(dyn_labels, sta4_labels):
        raise ValueError("DYN and STA4 Pendigits train labels do not align row-by-row")

    selected = select_samples_by_label(
        np.asarray(dyn_split.x),
        dyn_labels,
        required_labels=required_labels,
    )
    indices = np.asarray(selected.source_row_indices, dtype=np.int64)
    return PendigitsSampleSet(
        labels=selected.labels,
        source_row_indices=selected.source_row_indices,
        dyn=selected.values,
        sta4=np.asarray(sta4_split.x)[indices],
    )


def load_synthetic_sample_sets(
    manifest_path: Path,
    manifest: dict[str, Any] | None = None,
    *,
    required_labels: tuple[int, ...] = SYNTHETIC_CLASS_LABELS,
) -> tuple[SyntheticSampleSet, ...]:
    """Generate manifest-defined synthetic tasks and select train-split samples."""

    manifest = manifest if manifest is not None else load_json(manifest_path)
    data_settings = dict(manifest.get("data", {}))
    synthetic_settings = dict(manifest.get("synthetic", {}))
    datasets = dict(synthetic_settings.get("datasets", {}))
    data_seed = int(data_settings.get("data_seed", 0))
    validation_fraction = float(data_settings.get("validation_fraction", 0.1))
    n_samples = int(synthetic_settings.get("n_samples", 128))
    noise_epsilon = float(synthetic_settings.get("synthetic_noise_epsilon", 0.0))
    dtype = dtype_from_manifest(manifest)

    sample_sets: list[SyntheticSampleSet] = []
    for dataset in SYNTHETIC_DATASET_ORDER:
        task_settings = dict(datasets.get(dataset, {}))
        generated, effective_settings = generate_synthetic_dataset(
            dataset,
            data_seed=data_seed,
            n_samples=n_samples,
            noise_epsilon=noise_epsilon,
            task_settings=task_settings,
            dtype=dtype,
        )
        split_indices = synthetic_split_indices(
            int(generated.y.shape[0]),
            validation_fraction=validation_fraction,
            seed=data_seed,
        )
        selected = select_samples_by_label(
            np.asarray(generated.x),
            np.asarray(generated.y, dtype=np.int64),
            required_labels=required_labels,
            candidate_indices=split_indices.train,
        )
        sample_sets.append(
            SyntheticSampleSet(
                dataset=dataset,
                display_name=SYNTHETIC_DISPLAY_NAMES[dataset],
                labels=selected.labels,
                source_row_indices=selected.source_row_indices,
                values=selected.values,
                settings={
                    "data_seed": data_seed,
                    "validation_fraction": validation_fraction,
                    **effective_settings,
                    "train_split_size": int(split_indices.train.shape[0]),
                    "validation_split_size": int(split_indices.validation.shape[0]),
                    "test_split_size": int(split_indices.test.shape[0]),
                },
            )
        )
    return tuple(sample_sets)


def dtype_from_manifest(manifest: dict[str, Any]) -> jnp.dtype:
    dtype_name = str(manifest.get("model", {}).get("dtype", "float64"))
    if dtype_name == "float32":
        return jnp.float32
    if dtype_name == "float64":
        return jnp.float64
    raise ValueError(f"unsupported manifest model.dtype for plotting: {dtype_name!r}")


def generate_synthetic_dataset(
    dataset: str,
    *,
    data_seed: int,
    n_samples: int,
    noise_epsilon: float,
    task_settings: dict[str, Any],
    dtype: jnp.dtype,
):
    key = jax.random.PRNGKey(data_seed)
    if dataset == "synthetic-eigengap":
        dim = int(task_settings.get("synthetic_dim", 4))
        threshold = float(task_settings.get("synthetic_threshold", 0.75))
        generated = eigengap_classification(
            key,
            n_samples=n_samples,
            dim=dim,
            threshold=threshold,
            noise_epsilon=noise_epsilon,
        )
        settings = {
            "n_samples": n_samples,
            "synthetic_dim": dim,
            "synthetic_threshold": threshold,
            "synthetic_noise_epsilon": noise_epsilon,
        }
    elif dataset == "synthetic-singular":
        rows = int(task_settings.get("synthetic_rows", 4))
        cols = int(task_settings.get("synthetic_cols", 2))
        threshold = float(task_settings.get("synthetic_threshold", 1.5))
        generated = singular_value_classification(
            key,
            n_samples=n_samples,
            shape=(rows, cols),
            threshold=threshold,
            noise_epsilon=noise_epsilon,
        )
        settings = {
            "n_samples": n_samples,
            "synthetic_rows": rows,
            "synthetic_cols": cols,
            "synthetic_threshold": threshold,
            "synthetic_noise_epsilon": noise_epsilon,
        }
    else:
        raise ValueError(f"unsupported synthetic dataset for sample plotting: {dataset!r}")
    return type(generated)(x=generated.x.astype(dtype), y=generated.y), settings


def synthetic_split_indices(
    n_examples: int,
    *,
    validation_fraction: float,
    seed: int,
) -> SyntheticSplitIndices:
    """Return split indices matching scripts/train.py split_synthetic."""

    if n_examples <= 0:
        raise ValueError("synthetic split requires at least one example")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(np.arange(n_examples))
    n_test = max(1, int(round(0.2 * n_examples)))
    n_validation = max(1, int(round(validation_fraction * (n_examples - n_test))))
    test_indices = indices[:n_test]
    validation_indices = indices[n_test : n_test + n_validation]
    train_indices = indices[n_test + n_validation :]
    return SyntheticSplitIndices(
        train=train_indices,
        validation=validation_indices,
        test=test_indices,
    )


def select_samples_by_label(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    required_labels: tuple[int, ...],
    candidate_indices: np.ndarray | None = None,
) -> SelectedSampleSet:
    """Select the first candidate row for each required class label."""

    labels = np.ravel(np.asarray(labels, dtype=np.int64))
    if values.shape[0] != labels.shape[0]:
        raise ValueError(
            f"values/labels row mismatch: {values.shape[0]} values vs {labels.shape[0]} labels"
        )
    if candidate_indices is None:
        candidate_indices = np.arange(labels.shape[0], dtype=np.int64)
    else:
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if candidate_indices.size == 0:
        raise ValueError("cannot select class samples from an empty candidate split")

    selected_indices: list[int] = []
    missing: list[int] = []
    candidate_labels = labels[candidate_indices]
    for class_label in required_labels:
        matches = np.flatnonzero(candidate_labels == int(class_label))
        if matches.size == 0:
            missing.append(int(class_label))
            continue
        selected_indices.append(int(candidate_indices[int(matches[0])]))
    if missing:
        missing_text = ", ".join(str(label) for label in missing)
        required_text = ", ".join(str(label) for label in required_labels)
        raise ValueError(
            "missing required class label(s) in candidate split: "
            f"{missing_text}; required labels were {required_text}"
        )
    source_indices = tuple(selected_indices)
    return SelectedSampleSet(
        labels=tuple(int(label) for label in required_labels),
        source_row_indices=source_indices,
        values=np.asarray(values)[np.asarray(source_indices, dtype=np.int64)],
    )


def save_pendigits_figure(samples: PendigitsSampleSet, output_base: Path) -> list[str]:
    fig = plot_pendigits_samples(samples)
    return save_figure(fig, output_base)


def plot_pendigits_samples(samples: PendigitsSampleSet) -> plt.Figure:
    n_classes = len(samples.labels)
    fig = plt.figure(figsize=(1.1 * n_classes + 0.45, 3.0), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        n_classes + 1,
        width_ratios=[1.0] * n_classes + [0.08],
        height_ratios=[1.0, 1.0],
    )
    dyn_limits = padded_xy_limits(samples.dyn)
    sta4_vmin = float(np.nanmin(samples.sta4))
    sta4_vmax = float(np.nanmax(samples.sta4))
    if np.isclose(sta4_vmin, sta4_vmax):
        sta4_vmin -= 0.5
        sta4_vmax += 0.5

    last_image = None
    for column, (label, dyn_sample, sta4_sample) in enumerate(
        zip(samples.labels, samples.dyn, samples.sta4, strict=True)
    ):
        dyn_axis = fig.add_subplot(grid[0, column])
        dyn_axis.plot(
            dyn_sample[:, 0],
            dyn_sample[:, 1],
            color="#1f77b4",
            linewidth=1.0,
            marker="o",
            markersize=2.2,
        )
        dyn_axis.set_xlim(*dyn_limits[0])
        dyn_axis.set_ylim(*dyn_limits[1])
        dyn_axis.invert_yaxis()
        dyn_axis.set_aspect("equal", adjustable="box")
        dyn_axis.set_xticks([])
        dyn_axis.set_yticks([])
        dyn_axis.set_title(f"Class {label}", fontsize="small")
        if column == 0:
            dyn_axis.set_ylabel("DYN")

        sta4_axis = fig.add_subplot(grid[1, column])
        last_image = sta4_axis.imshow(
            sta4_sample,
            cmap="viridis",
            origin="upper",
            vmin=sta4_vmin,
            vmax=sta4_vmax,
            interpolation="nearest",
        )
        sta4_axis.set_xticks([])
        sta4_axis.set_yticks([])
        if column == 0:
            sta4_axis.set_ylabel("STA4")

    blank_axis = fig.add_subplot(grid[0, -1])
    blank_axis.axis("off")
    colorbar_axis = fig.add_subplot(grid[1, -1])
    if last_image is not None:
        colorbar = fig.colorbar(last_image, cax=colorbar_axis)
        colorbar.set_label("Raw value", fontsize="small")
    return fig


def save_synthetic_figure(samples: tuple[SyntheticSampleSet, ...], output_base: Path) -> list[str]:
    fig = plot_synthetic_samples(samples)
    return save_figure(fig, output_base)


def plot_synthetic_samples(samples: tuple[SyntheticSampleSet, ...]) -> plt.Figure:
    if not samples:
        raise ValueError("at least one synthetic sample set is required")
    n_rows = len(samples)
    n_classes = max(len(sample_set.labels) for sample_set in samples)
    fig = plt.figure(figsize=(2.1 * n_classes + 0.55, 2.1 * n_rows), constrained_layout=True)
    grid = fig.add_gridspec(
        n_rows,
        n_classes + 1,
        width_ratios=[1.0] * n_classes + [0.08],
    )
    for row, sample_set in enumerate(samples):
        matrix_limit = symmetric_abs_limit(sample_set.values)
        last_image = None
        for column, label in enumerate(sample_set.labels):
            axis = fig.add_subplot(grid[row, column])
            last_image = axis.imshow(
                sample_set.values[column],
                cmap="coolwarm",
                origin="upper",
                vmin=-matrix_limit,
                vmax=matrix_limit,
                interpolation="nearest",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"Class {label}", fontsize="small")
            if column == 0:
                axis.set_ylabel(sample_set.display_name)
        colorbar_axis = fig.add_subplot(grid[row, -1])
        if last_image is not None:
            colorbar = fig.colorbar(last_image, cax=colorbar_axis)
            colorbar.set_label("Entry", fontsize="small")
    return fig


def padded_xy_limits(values: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    flattened = np.asarray(values).reshape((-1, 2))
    x_limits = padded_limits(flattened[:, 0])
    y_limits = padded_limits(flattened[:, 1])
    return x_limits, y_limits


def padded_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot determine plot limits from non-finite values")
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    span = upper - lower
    pad = 0.05 * span if span > 0 else 0.5
    return lower - pad, upper + pad


def symmetric_abs_limit(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot determine color limits from non-finite values")
    limit = float(np.max(np.abs(finite)))
    return limit if limit > 0 else 1.0


def save_figure(fig: plt.Figure, output_base: Path) -> list[str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [str(pdf_path), str(png_path)]


def build_metadata(
    *,
    data_root: Path,
    synthetic_manifest_path: Path,
    synthetic_manifest: dict[str, Any],
    pendigits_samples: PendigitsSampleSet,
    synthetic_samples: tuple[SyntheticSampleSet, ...],
    pendigits_outputs: list[str],
    synthetic_outputs: list[str],
    metadata_path: Path,
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pendigits": {
            "data_root": str(data_root),
            "split": "train",
            "source": "raw Pendigits DYN and STA4 train CSV rows",
            "preprocessing": (
                "Raw values are plotted for visual interpretation; model training "
                "standardizes Pendigits features separately."
            ),
            "selected_samples": sample_metadata(
                pendigits_samples.labels,
                pendigits_samples.source_row_indices,
            ),
            "outputs": pendigits_outputs,
        },
        "synthetic": {
            "manifest": str(synthetic_manifest_path),
            "manifest_id": synthetic_manifest.get("manifest_id"),
            "split": "train",
            "selected_samples_by_dataset": {
                sample_set.dataset: {
                    "settings": sample_set.settings,
                    "selected_samples": sample_metadata(
                        sample_set.labels,
                        sample_set.source_row_indices,
                    ),
                }
                for sample_set in synthetic_samples
            },
            "outputs": synthetic_outputs,
        },
        "metadata_path": str(metadata_path),
    }


def sample_metadata(
    labels: tuple[int, ...],
    source_row_indices: tuple[int, ...],
) -> list[dict[str, int]]:
    return [
        {"class_label": int(label), "source_row_index": int(source_row_index)}
        for label, source_row_index in zip(labels, source_row_indices, strict=True)
    ]


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
