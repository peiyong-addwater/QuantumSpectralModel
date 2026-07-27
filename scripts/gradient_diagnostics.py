#!/usr/bin/env python3
"""Compute grouped gradient and Fisher diagnostics for manifest jobs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from ham_embed_spectral.experiments.ablations import apply_dataset_ablation  # noqa: E402
from ham_embed_spectral.experiments.fisher import empirical_fisher_summary  # noqa: E402
from ham_embed_spectral.experiments.gradients import (  # noqa: E402
    flatten_gradient_groups,
    gradient_group_summary,
    layerwise_gradient_flow,
    per_sample_gradient_matrix,
)
from ham_embed_spectral.experiments.manifest import (  # noqa: E402
    REPRESENTATION_CHOICES,
    jobs_from_manifest,
    load_manifest,
    manifest_section,
)
from ham_embed_spectral.models.reuploading import (  # noqa: E402
    cross_entropy_loss,
    init_reuploading_params,
)
from ham_embed_spectral.naming import ENCODER_CLI_CHOICES, canonical_encoder_name  # noqa: E402
from ham_embed_spectral.utils.checkpointing import (  # noqa: E402
    list_hdf5_checkpoint_labels,
    write_json,
)
from scripts import train as train_script  # noqa: E402


@dataclass(frozen=True)
class DiagnosticBundle:
    """One JSON record plus unpersisted vectors used for aggregate summaries."""

    record: dict[str, Any]
    group_vectors: dict[str, jnp.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", default="configs/experiments/Legacy/smoke_tiny.json")
    parser.add_argument("--output", default="results/tables/gradient_diagnostics.json")
    parser.add_argument("--mode", choices=("init", "checkpoints", "final"), default="init")
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--encoders", nargs="+", choices=ENCODER_CLI_CHOICES, default=None)
    parser.add_argument("--reupload-depths", nargs="+", type=int, default=None)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATION_CHOICES,
        default=None,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--diagnostic-batch-size", type=int, default=32)
    parser.add_argument("--diagnostic-seed", type=int, default=0)
    parser.add_argument("--near-zero-tol", type=float, default=1e-10)
    parser.add_argument("--n-init-seeds", type=int, default=8)
    parser.add_argument("--fisher-batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    jobs = selected_jobs_from_manifest(manifest, args)

    bundles: list[DiagnosticBundle] = []
    for job in jobs:
        try:
            bundles.extend(diagnose_job(job, manifest, args))
        except Exception as exc:  # noqa: BLE001 - diagnostics must record failures per job.
            bundles.append(
                DiagnosticBundle(
                    record={
                        "status": "error",
                        "job_slug": job.slug,
                        "dataset": job.dataset,
                        "representation": job.representation,
                        "encoder": job.encoder,
                        "depth": job.reupload_depth,
                        "seed": job.seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    group_vectors={},
                )
            )

    records = [bundle.record for bundle in bundles]
    payload = {
        "manifest_id": manifest["manifest_id"],
        "mode": args.mode,
        "diagnostic_batch_size": args.diagnostic_batch_size,
        "diagnostic_seed": args.diagnostic_seed,
        "near_zero_tol": args.near_zero_tol,
        "n_init_seeds": args.n_init_seeds,
        "fisher_batch_size": args.fisher_batch_size,
        "filters": filter_metadata(args),
        "n_records": len(records),
        "n_complete": sum(
            str(record.get("status", "")).startswith("complete") for record in records
        ),
        "records": records,
        "aggregates": aggregate_gradient_vectors(bundles, args.near_zero_tol),
    }
    write_json(args.output, payload)
    print(args.output)


def selected_jobs_from_manifest(manifest: dict[str, Any], args: argparse.Namespace):
    jobs = filter_jobs(jobs_from_manifest(manifest, encoders=args.encoders), args)
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    return jobs


def filter_jobs(jobs, args: argparse.Namespace):
    encoders = set(canonical_encoder_filter(args.encoders) or [])
    depths = set(args.reupload_depths or [])
    representations = set(args.representations or [])
    seeds = set(args.seeds or [])
    return [
        job
        for job in jobs
        if (not encoders or job.encoder in encoders)
        and (not depths or job.reupload_depth in depths)
        and (not representations or job.representation in representations)
        and (not seeds or job.seed in seeds)
    ]


def canonical_encoder_filter(encoders: list[str] | None) -> list[str] | None:
    if encoders is None:
        return None
    return [canonical_encoder_name(value) for value in encoders]


def filter_metadata(args: argparse.Namespace) -> dict[str, list[int] | list[str] | None]:
    return {
        "encoders": canonical_encoder_filter(args.encoders),
        "reupload_depths": args.reupload_depths,
        "representations": args.representations,
        "seeds": args.seeds,
    }


def diagnose_job(job, manifest: dict[str, Any], args: argparse.Namespace) -> list[DiagnosticBundle]:
    if args.mode == "init":
        return [diagnose_initialization(job, manifest, args)]
    return diagnose_restored_checkpoints(job, manifest, args)


def diagnose_initialization(
    job,
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> DiagnosticBundle:
    train_args, dataset, encoder, model_config = context_for_job(job, manifest)
    x, y = diagnostic_batch(dataset.train, args.diagnostic_batch_size, args.diagnostic_seed)
    group_samples: dict[str, list[jnp.ndarray]] = {}
    flow_samples: dict[str, list[jnp.ndarray]] = {}
    losses = []

    def batch_loss(current_params, batch_x=x, batch_y=y):
        return cross_entropy_loss(current_params, encoder, batch_x, batch_y, model_config)

    first_params = None
    for offset in range(args.n_init_seeds):
        params = init_reuploading_params(
            jax.random.PRNGKey(job.seed + offset),
            encoder,
            model_config,
            dtype=train_script.dtype_from_name(train_args.dtype),
        )
        first_params = params if first_params is None else first_params
        loss, grads = jax.value_and_grad(batch_loss)(params)
        losses.append(float(loss))
        for group, values in flatten_gradient_groups(grads).items():
            group_samples.setdefault(group, []).append(values)
        for group, values in layerwise_gradient_flow(grads).items():
            flow_samples.setdefault(group, []).append(values)

    group_summaries = {
        group: gradient_group_summary(jnp.stack(values), near_zero_tol=args.near_zero_tol)
        for group, values in group_samples.items()
    }
    flow_summaries = {
        group: jnp.mean(jnp.stack(values), axis=0) for group, values in flow_samples.items()
    }
    record = base_record(job, train_args, dataset, "complete")
    record.update(
        {
            "diagnostic_mode": "init",
            "loss_mean": sum(losses) / len(losses),
            "n_gradient_vectors": args.n_init_seeds,
            "groups": group_summaries,
            "layerwise_flow": flow_summaries,
        }
    )
    if first_params is not None and args.fisher_batch_size:
        record["fisher"] = fisher_for_params(
            first_params,
            encoder,
            model_config,
            x,
            y,
            args.fisher_batch_size,
        )
    return DiagnosticBundle(
        record=record,
        group_vectors={group: jnp.stack(values) for group, values in group_samples.items()},
    )


def diagnose_restored_checkpoints(
    job,
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> list[DiagnosticBundle]:
    train_args, dataset, encoder, model_config = context_for_job(job, manifest)
    x, y = diagnostic_batch(dataset.train, args.diagnostic_batch_size, args.diagnostic_seed)
    run_dir = run_directory(job, manifest, args)
    if not run_dir.exists():
        return [DiagnosticBundle(base_record(job, train_args, dataset, "missing_run_dir"), {})]
    config = load_run_config(run_dir)
    checkpoint_paths = checkpoint_paths_for_mode(run_dir, args.mode)
    if not checkpoint_paths:
        return [DiagnosticBundle(base_record(job, train_args, dataset, "missing_checkpoint"), {})]

    bundles = []
    for checkpoint_label, checkpoint_path in checkpoint_paths:
        try:
            state = train_script.restored_train_state(checkpoint_path)
            params = state.params
            loss, grads = jax.value_and_grad(
                lambda current_params: cross_entropy_loss(
                    current_params,
                    encoder,
                    x,
                    y,
                    model_config,
                )
            )(params)
            groups = flatten_gradient_groups(grads)
            record = base_record(
                job,
                train_args,
                dataset,
                "complete_dirty_git" if config.get("git", {}).get("dirty") else "complete",
            )
            record.update(
                {
                    "diagnostic_mode": args.mode,
                    "run_dir": str(run_dir),
                    "checkpoint": checkpoint_label,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_step": int(getattr(state, "step", -1)),
                    "loss": float(loss),
                    "groups": {
                        group: gradient_group_summary(
                            values[jnp.newaxis, :],
                            near_zero_tol=args.near_zero_tol,
                        )
                        for group, values in groups.items()
                    },
                    "layerwise_flow": layerwise_gradient_flow(grads),
                    "git_dirty": bool(config.get("git", {}).get("dirty", False)),
                }
            )
            if args.fisher_batch_size:
                record["fisher"] = fisher_for_params(
                    params,
                    encoder,
                    model_config,
                    x,
                    y,
                    args.fisher_batch_size,
                )
            bundles.append(DiagnosticBundle(record=record, group_vectors=groups))
        except Exception as exc:  # noqa: BLE001 - record checkpoint-level failures.
            record = base_record(job, train_args, dataset, "error")
            record.update(
                {
                    "diagnostic_mode": args.mode,
                    "run_dir": str(run_dir),
                    "checkpoint": checkpoint_label,
                    "checkpoint_path": str(checkpoint_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            bundles.append(DiagnosticBundle(record=record, group_vectors={}))
    return bundles


def context_for_job(job, manifest: dict[str, Any]):
    train_args = args_for_job(job, manifest)
    data_seed = job.seed if train_args.data_seed is None else train_args.data_seed
    dataset = train_script.load_dataset(train_args, data_seed)
    encoder = train_script.build_encoder(train_args)
    ablated = apply_dataset_ablation(
        dataset,
        ablation=train_args.ablation,
        encoder_name=train_args.encoder,
        seed=train_args.ablation_seed,
    )
    dataset = ablated.dataset
    model_config = train_script.build_model_config(train_args, encoder, dataset)
    return train_args, dataset, encoder, model_config


def diagnostic_batch(split, batch_size: int, seed: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    n_examples = int(split.y.shape[0])
    if n_examples == 0:
        raise ValueError("cannot build a diagnostic batch from an empty split")
    if batch_size >= n_examples:
        return split.x, split.y
    rng = np.random.default_rng(seed)
    labels = np.asarray(split.y)
    classes = np.unique(labels)
    pools = {label: list(rng.permutation(np.flatnonzero(labels == label))) for label in classes}
    selected: list[int] = []
    while len(selected) < batch_size:
        before = len(selected)
        for label in classes:
            if pools[label] and len(selected) < batch_size:
                selected.append(int(pools[label].pop()))
        if len(selected) == before:
            break
    rng.shuffle(selected)
    indices = np.asarray(selected, dtype=np.int64)
    return split.x[indices], split.y[indices]


def fisher_for_params(
    params,
    encoder,
    model_config,
    x: jnp.ndarray,
    y: jnp.ndarray,
    max_batch: int,
) -> dict[str, jnp.ndarray]:
    batch_x = x[:max_batch]
    batch_y = y[:max_batch]

    def loss_fn(current_params, samples, labels):
        return cross_entropy_loss(current_params, encoder, samples, labels, model_config)

    matrix = per_sample_gradient_matrix(loss_fn, params, batch_x, batch_y)
    summary = empirical_fisher_summary(matrix)
    return {"batch_size": int(batch_y.shape[0]), "gradient_shape": matrix.shape, "summary": summary}


def aggregate_gradient_vectors(
    bundles: list[DiagnosticBundle],
    near_zero_tol: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[jnp.ndarray]] = {}
    for bundle in bundles:
        if not str(bundle.record.get("status", "")).startswith("complete"):
            continue
        for group, vector in bundle.group_vectors.items():
            key = aggregate_key(bundle.record, group)
            grouped.setdefault(key, []).append(jnp.ravel(vector))

    records = []
    for key, vectors in sorted(grouped.items(), key=lambda item: str(item[0])):
        dataset, representation, encoder, depth, checkpoint, group = key
        shapes = {tuple(vector.shape) for vector in vectors}
        record = {
            "dataset": dataset,
            "representation": representation,
            "encoder": encoder,
            "depth": depth,
            "checkpoint": checkpoint,
            "group": group,
            "n_vectors": len(vectors),
        }
        if len(shapes) != 1:
            record.update({"status": "parameter_shape_mismatch", "shapes": sorted(shapes)})
        else:
            record.update(
                {
                    "status": "complete",
                    "summary": gradient_group_summary(
                        jnp.stack(vectors),
                        near_zero_tol=near_zero_tol,
                    ),
                }
            )
        records.append(record)
    return records


def aggregate_key(record: dict[str, Any], group: str) -> tuple[Any, ...]:
    return (
        record.get("dataset"),
        record.get("representation"),
        record.get("encoder"),
        record.get("depth"),
        record.get("checkpoint", record.get("diagnostic_mode")),
        group,
    )


def checkpoint_paths_for_mode(run_dir: Path, mode: str) -> list[tuple[str, str | Path]]:
    hdf5_path = run_dir / "checkpoints.h5"
    checkpoints = run_dir / "checkpoints"
    if mode == "final":
        if "final" in list_hdf5_checkpoint_labels(hdf5_path):
            return [("final", f"{hdf5_path}::final")]
        path = checkpoints / "final"
        return [("final", path)] if path.exists() else []

    hdf5_labels = list_hdf5_checkpoint_labels(hdf5_path, prefix="step_")
    if hdf5_labels:
        return [(label, f"{hdf5_path}::{label}") for label in hdf5_labels]
    paths = []
    for path in sorted(checkpoints.glob("step_*")):
        paths.append((path.name, path))
    return paths


def run_directory(job, manifest: dict[str, Any], args: argparse.Namespace) -> Path:
    outputs = manifest_section(manifest, "outputs")
    runs_root = Path(args.runs_root or outputs.get("output_root", "results/runs"))
    experiment_name = args.experiment_name or outputs.get(
        "experiment_name",
        manifest["manifest_id"],
    )
    return runs_root / experiment_name / f"{job.slug}_seed{job.seed}"


def load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def base_record(job, train_args: SimpleNamespace, dataset, status: str) -> dict[str, Any]:
    return {
        "status": status,
        "job_slug": job.slug,
        "dataset": job.dataset,
        "representation": dataset.representation,
        "encoder": job.encoder,
        "depth": job.reupload_depth,
        "seed": job.seed,
        "data_seed": train_args.data_seed if train_args.data_seed is not None else job.seed,
        "ablation": train_args.ablation,
        "input_shape": dataset.input_shape,
        "n_train": int(dataset.train.y.shape[0]),
        "n_classes": dataset.n_classes,
    }


def args_for_job(job, manifest: dict[str, Any]) -> SimpleNamespace:
    data = manifest_section(manifest, "data")
    synthetic = manifest_section(manifest, "synthetic")
    training = manifest_section(manifest, "training")
    model = manifest_section(manifest, "model")
    ablations = manifest_section(manifest, "ablations")
    synthetic_settings = synthetic.get("datasets", {}).get(job.dataset, {})
    return SimpleNamespace(
        dataset=job.dataset,
        data_root=data.get("data_root", "data/raw/pendigits"),
        data_seed=data.get("data_seed"),
        representation=job.representation,
        download_data=data.get("download_data", False),
        validation_fraction=data.get("validation_fraction", 0.1),
        class_subset=job.class_subset,
        standardize=data.get("standardize", True),
        max_train_examples=None,
        max_eval_examples=None,
        n_samples=synthetic_settings.get(
            "n_samples",
            synthetic.get("n_samples", training.get("diagnostic_n_samples", 128)),
        ),
        synthetic_dim=synthetic_settings.get("synthetic_dim", synthetic.get("synthetic_dim", 4)),
        synthetic_rows=synthetic_settings.get("synthetic_rows", synthetic.get("synthetic_rows", 4)),
        synthetic_cols=synthetic_settings.get("synthetic_cols", synthetic.get("synthetic_cols", 2)),
        synthetic_threshold=synthetic_settings.get(
            "synthetic_threshold",
            synthetic.get("synthetic_threshold"),
        ),
        synthetic_noise_epsilon=synthetic_settings.get(
            "synthetic_noise_epsilon",
            synthetic.get("synthetic_noise_epsilon", 0.0),
        ),
        encoder=job.encoder,
        reupload_depth=job.reupload_depth,
        mixer_scale=model.get("mixer_scale", 0.01),
        projector_renormalize=model.get("projector_renormalize", True),
        ry_alpha=model.get("ry_alpha", 1.0),
        rz_beta=model.get("rz_beta", 1.0),
        tf_init_scale=model.get("tf_init_scale", 1.0),
        tf_init_noise=model.get("tf_init_noise", 0.01),
        patch_scale=model.get("patch_scale", 1.0),
        patch_map_init_noise=model.get("patch_map_init_noise", 0.01),
        trainable_times=model.get("trainable_times", True),
        fixed_time=model.get("fixed_time"),
        initial_state=model.get("initial_state", "plus"),
        dtype=model.get("dtype", "float64"),
        dry_run=False,
        ablation=ablations.get("ablation", "none"),
        ablation_seed=ablations.get("ablation_seed", 0),
    )


if __name__ == "__main__":
    main()
