#!/usr/bin/env python3
"""Compute feature and latent-state diagnostics for re-uploading checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from ham_embed_spectral.experiments.latent_states import (  # noqa: E402
    adjacent_layer_cka,
    aggregate_layer_summaries,
    fidelity_kernel,
    hamiltonian_spectral_state_summary,
    kernel_geometry_summary,
    layerwise_kernel_summaries,
    logit_trajectory_summary,
    paired_stage_cka,
    projector_probe_summary,
    projector_score_trace,
    value_summary,
)
from ham_embed_spectral.experiments.manifest import (  # noqa: E402
    DATASET_CHOICES,
    REPRESENTATION_CHOICES,
    load_manifest,
)
from ham_embed_spectral.models.reuploading import (  # noqa: E402
    init_reuploading_params,
    layerwise_states,
    probabilities,
)
from ham_embed_spectral.naming import ENCODER_CLI_CHOICES, canonical_encoder_name  # noqa: E402
from ham_embed_spectral.quantum.encoders import (  # noqa: E402
    BlockHamiltonianEncoder,
    SymmetricHamiltonianEncoder,
)
from ham_embed_spectral.utils.checkpointing import write_json  # noqa: E402
from scripts import gradient_diagnostics as gradient_script  # noqa: E402
from scripts import train as train_script  # noqa: E402


@dataclass(frozen=True)
class LatentDiagnosticBundle:
    """One diagnostic record plus optional arrays for HDF5 persistence."""

    record: dict[str, Any]
    artifact: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", default="configs/experiments/Legacy/smoke_tiny.json")
    parser.add_argument("--output", default="results/tables/latent_state_diagnostics.json")
    parser.add_argument("--hdf5-output", default=None)
    parser.add_argument(
        "--mode",
        choices=("init-reference", "checkpoints", "final"),
        default="final",
    )
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, default=None)
    parser.add_argument("--encoders", nargs="+", choices=ENCODER_CLI_CHOICES, default=None)
    parser.add_argument("--reupload-depths", nargs="+", type=int, default=None)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATION_CHOICES,
        default=None,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--diagnostic-batch-size", type=int, default=64)
    parser.add_argument("--diagnostic-seed", type=int, default=0)
    parser.add_argument("--spectral-state-max-samples", type=int, default=8)
    parser.add_argument("--store-state-traces", action="store_true")
    args = parser.parse_args()
    if args.max_jobs is not None and args.max_jobs < 0:
        raise ValueError("--max-jobs must be nonnegative")
    if args.diagnostic_batch_size < 1:
        raise ValueError("--diagnostic-batch-size must be positive")
    if args.spectral_state_max_samples < 0:
        raise ValueError("--spectral-state-max-samples must be nonnegative")
    if args.encoders is not None:
        args.encoders = [canonical_encoder_name(value) for value in args.encoders]
    return args


def main() -> None:
    args = parse_args()
    started = time.time()
    manifest = load_manifest(args.manifest)
    jobs = selected_jobs_from_manifest(manifest, args)

    bundles: list[LatentDiagnosticBundle] = []
    for job in jobs:
        try:
            bundles.extend(diagnose_job(job, manifest, args))
        except Exception as exc:  # noqa: BLE001 - record job-level diagnostics failures.
            bundles.append(
                LatentDiagnosticBundle(
                    record={
                        "status": "error",
                        "job_slug": job.slug,
                        "dataset": job.dataset,
                        "representation": job.representation,
                        "encoder": job.encoder,
                        "depth": job.reupload_depth,
                        "seed": job.seed,
                        "diagnostic_mode": args.mode,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    artifact={},
                )
            )

    if args.hdf5_output is not None:
        write_hdf5_artifacts(args.hdf5_output, bundles)

    records = [bundle.record for bundle in bundles]
    payload = {
        "schema_version": "latent_state_diagnostics_v1",
        "manifest_id": manifest["manifest_id"],
        "mode": args.mode,
        "split": args.split,
        "diagnostic_batch_size": args.diagnostic_batch_size,
        "diagnostic_seed": args.diagnostic_seed,
        "spectral_state_max_samples": args.spectral_state_max_samples,
        "filters": filter_metadata(args),
        "hdf5_output": args.hdf5_output,
        "hdf5_artifact_policy": (
            "state traces stored only when --hdf5-output and --store-state-traces are set"
        ),
        "n_records": len(records),
        "n_complete": sum(
            str(record.get("status", "")).startswith("complete") for record in records
        ),
        "elapsed_seconds": time.time() - started,
        "records": records,
    }
    write_json(args.output, payload)
    print(args.output)


def selected_jobs_from_manifest(manifest: dict[str, Any], args: argparse.Namespace):
    selection_args = argparse.Namespace(**vars(args))
    selection_args.max_jobs = None
    jobs = gradient_script.selected_jobs_from_manifest(manifest, selection_args)
    wanted_datasets = set(args.datasets or [])
    if wanted_datasets:
        jobs = [job for job in jobs if job.dataset in wanted_datasets]
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    return jobs


def filter_metadata(args: argparse.Namespace) -> dict[str, list[int] | list[str] | None]:
    return {
        "datasets": args.datasets,
        "encoders": args.encoders,
        "reupload_depths": args.reupload_depths,
        "representations": args.representations,
        "seeds": args.seeds,
    }


def diagnose_job(
    job,
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> list[LatentDiagnosticBundle]:
    train_args, dataset, encoder, model_config = gradient_script.context_for_job(job, manifest)
    split = split_for_name(dataset, args.split)
    x, y = gradient_script.diagnostic_batch(
        split,
        args.diagnostic_batch_size,
        args.diagnostic_seed,
    )
    if args.mode == "init-reference":
        params = init_reuploading_params(
            jax.random.PRNGKey(job.seed),
            encoder,
            model_config,
            dtype=train_script.dtype_from_name(train_args.dtype),
        )
        record = gradient_script.base_record(job, train_args, dataset, "complete")
        record.update({"diagnostic_mode": args.mode})
        return [diagnose_params(record, params, encoder, model_config, x, y, args)]

    run_dir = gradient_script.run_directory(job, manifest, args)
    if not run_dir.exists():
        record = gradient_script.base_record(job, train_args, dataset, "missing_run_dir")
        record.update({"diagnostic_mode": args.mode, "run_dir": str(run_dir)})
        return [LatentDiagnosticBundle(record=record, artifact={})]
    checkpoint_paths = gradient_script.checkpoint_paths_for_mode(run_dir, args.mode)
    if not checkpoint_paths:
        record = gradient_script.base_record(job, train_args, dataset, "missing_checkpoint")
        record.update({"diagnostic_mode": args.mode, "run_dir": str(run_dir)})
        return [LatentDiagnosticBundle(record=record, artifact={})]

    config = gradient_script.load_run_config(run_dir)
    bundles = []
    for checkpoint_label, checkpoint_path in checkpoint_paths:
        try:
            state = train_script.restored_train_state(checkpoint_path)
            params = state.params
            record = gradient_script.base_record(
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
                    "git_dirty": bool(config.get("git", {}).get("dirty", False)),
                }
            )
            bundles.append(diagnose_params(record, params, encoder, model_config, x, y, args))
        except Exception as exc:  # noqa: BLE001 - record checkpoint-level failures.
            record = gradient_script.base_record(job, train_args, dataset, "error")
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
            bundles.append(LatentDiagnosticBundle(record=record, artifact={}))
    return bundles


def diagnose_params(
    base_record: dict[str, Any],
    params: Mapping[str, Any],
    encoder: object,
    model_config,
    x: jnp.ndarray,
    y: jnp.ndarray,
    args: argparse.Namespace,
) -> LatentDiagnosticBundle:
    trace = batched_layerwise_states(params, encoder, x, model_config)
    initial_states = trace["initial"][:, None, :]
    post_upload = trace["post_upload"]
    post_mixer = trace["post_mixer"]
    all_states, stage_names = ordered_state_path(initial_states, post_upload, post_mixer)
    all_scores = projector_score_trace(
        all_states,
        n_classes=model_config.n_classes,
        renormalize=model_config.projector_renormalize,
    )
    final_probs = probabilities(params, encoder, x, model_config)
    final_predictions = jnp.argmax(final_probs, axis=-1)

    record = dict(base_record)
    record.update(
        {
            "split": args.split,
            "n_diagnostic_examples": int(y.shape[0]),
            "stage_names": stage_names,
            "final_projector_accuracy_on_diagnostic_batch": jnp.mean(final_predictions == y),
            "projector_probes": {
                "initial": projector_probe_summary(
                    projector_score_trace(
                        initial_states,
                        n_classes=model_config.n_classes,
                        renormalize=model_config.projector_renormalize,
                    ),
                    y,
                ),
                "post_upload": projector_probe_summary(
                    projector_score_trace(
                        post_upload,
                        n_classes=model_config.n_classes,
                        renormalize=model_config.projector_renormalize,
                    ),
                    y,
                ),
                "post_mixer": projector_probe_summary(
                    projector_score_trace(
                        post_mixer,
                        n_classes=model_config.n_classes,
                        renormalize=model_config.projector_renormalize,
                    ),
                    y,
                ),
            },
            "fidelity_kernels": {
                "initial": layerwise_kernel_summaries(initial_states, y),
                "post_upload": layerwise_kernel_summaries(post_upload, y),
                "post_mixer": layerwise_kernel_summaries(post_mixer, y),
                "final": kernel_geometry_summary(fidelity_kernel(post_mixer[:, -1, :]), y),
            },
            "cka": {
                "post_mixer_adjacent_layers": adjacent_layer_cka(post_mixer),
                "post_upload_to_post_mixer_same_layer": paired_stage_cka(post_upload, post_mixer),
                "initial_to_final_post_mixer": paired_stage_cka(
                    jnp.broadcast_to(initial_states, post_mixer.shape),
                    post_mixer,
                ),
            },
            "logit_trajectories": logit_trajectory_summary(all_scores, y, stage_names=stage_names),
            "readout_score_scale": "projector probabilities",
        }
    )
    spectral_state = spectral_state_diagnostics(
        params,
        encoder,
        model_config,
        x,
        trace,
        args.spectral_state_max_samples,
    )
    if spectral_state is not None:
        record["hamiltonian_spectral_state"] = spectral_state

    artifact = {
        "labels": np.asarray(jax.device_get(y)),
        "stage_scores": np.asarray(jax.device_get(all_scores)),
    }
    if args.store_state_traces:
        artifact.update(
            {
                "initial_states": np.asarray(jax.device_get(trace["initial"])),
                "post_upload_states": np.asarray(jax.device_get(post_upload)),
                "post_mixer_states": np.asarray(jax.device_get(post_mixer)),
            }
        )
    return LatentDiagnosticBundle(record=record, artifact=artifact)


def batched_layerwise_states(
    params,
    encoder,
    samples: jnp.ndarray,
    model_config,
) -> dict[str, jnp.ndarray]:
    return jax.vmap(lambda sample: layerwise_states(params, encoder, sample, model_config))(samples)


def ordered_state_path(
    initial_states: jnp.ndarray,
    post_upload: jnp.ndarray,
    post_mixer: jnp.ndarray,
) -> tuple[jnp.ndarray, list[str]]:
    states = [initial_states]
    names = ["initial"]
    for layer_index in range(post_upload.shape[1]):
        states.append(post_upload[:, layer_index : layer_index + 1, :])
        names.append(f"layer_{layer_index}_post_upload")
        states.append(post_mixer[:, layer_index : layer_index + 1, :])
        names.append(f"layer_{layer_index}_post_mixer")
    return jnp.concatenate(states, axis=1), names


def spectral_state_diagnostics(
    params: Mapping[str, Any],
    encoder: object,
    model_config,
    x: jnp.ndarray,
    trace: dict[str, jnp.ndarray],
    max_samples: int,
) -> dict[str, Any] | None:
    if not isinstance(encoder, (SymmetricHamiltonianEncoder, BlockHamiltonianEncoder)):
        return None
    if max_samples <= 0:
        return {"status": "skipped", "reason": "spectral_state_max_samples is 0"}

    n_samples = min(max_samples, int(x.shape[0]))
    post_mixer = trace["post_mixer"][:n_samples]
    pre_upload = jnp.concatenate(
        [trace["initial"][:n_samples, None, :], post_mixer[:, :-1, :]],
        axis=1,
    )
    post_upload = trace["post_upload"][:n_samples]
    n_qubits = model_config.n_qubits or encoder.n_qubits(model_config.input_shape)
    sample_summaries = []
    for sample_index in range(n_samples):
        sample_summaries.append(
            hamiltonian_spectral_state_summary(
                encoder,
                params["encoder"],
                x[sample_index],
                pre_upload[sample_index],
                post_upload[sample_index],
                n_qubits=n_qubits,
                reupload_depth=model_config.reupload_depth,
            )
        )
    return {
        "status": "complete",
        "n_samples": n_samples,
        "occupation_l1_change_by_layer": aggregate_layer_summaries(
            sample_summaries,
            "occupation_l1_change",
        ),
        "phase_abs_max_by_layer": aggregate_nested_layer_summary(
            sample_summaries,
            "phase_increment_abs",
            "max",
        ),
        "phase_abs_mean_by_layer": aggregate_nested_layer_summary(
            sample_summaries,
            "phase_increment_abs",
            "mean",
        ),
    }


def aggregate_nested_layer_summary(
    sample_summaries: list[list[dict[str, Any]]],
    field: str,
    nested_field: str,
) -> list[dict[str, Any]]:
    if not sample_summaries:
        return []
    n_layers = len(sample_summaries[0])
    records = []
    for layer_index in range(n_layers):
        values = jnp.asarray(
            [sample[layer_index][field][nested_field] for sample in sample_summaries]
        )
        records.append(
            {
                "layer_index": layer_index,
                f"{field}_{nested_field}": value_summary(values),
            }
        )
    return records


def split_for_name(dataset, split_name: str):
    if split_name == "train":
        return dataset.train
    if split_name == "validation":
        return dataset.validation
    if split_name == "test":
        return dataset.test
    raise ValueError(f"unknown split {split_name!r}")


def write_hdf5_artifacts(path: str | Path, bundles: list[LatentDiagnosticBundle]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "a") as handle:
        handle.attrs["schema_version"] = "latent_state_diagnostics_hdf5_v1"
        handle.attrs["updated_at_utc"] = datetime.now(UTC).isoformat()
        root = handle.require_group("records")
        for index, bundle in enumerate(bundles):
            name = hdf5_record_name(index, bundle.record)
            if name in root:
                del root[name]
            group = root.create_group(name)
            group.create_dataset(
                "record_json",
                data=json.dumps(json_safe(bundle.record), sort_keys=True),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            arrays = group.create_group("arrays")
            for key, value in bundle.artifact.items():
                arrays.create_dataset(key, data=value)


def hdf5_record_name(index: int, record: Mapping[str, Any]) -> str:
    parts = [
        f"{index:06d}",
        str(record.get("job_slug", "job")),
        str(record.get("checkpoint", record.get("diagnostic_mode", "mode"))),
    ]
    raw = "__".join(parts)
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in raw)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, jnp.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, SimpleNamespace):
        return vars(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    main()
