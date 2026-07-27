#!/usr/bin/env python3
"""Train re-uploading models from CLI arguments."""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ham_embed_spectral._jax_config import enable_x64  # noqa: E402

enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ham_embed_spectral.config import ReuploadingModelConfig  # noqa: E402
from ham_embed_spectral.data.pendigits import (  # noqa: E402
    PendigitsDataset,
    PendigitsSplit,
    prepare_pendigits,
)
from ham_embed_spectral.data.synthetic import (  # noqa: E402
    eigengap_classification,
    singular_value_classification,
)
from ham_embed_spectral.experiments.ablations import (  # noqa: E402
    ABLATION_CHOICES,
    apply_dataset_ablation,
)
from ham_embed_spectral.experiments.train_loop import (  # noqa: E402
    TrainingLoopConfig,
    TrainState,
    create_train_state,
    epoch_minibatches,
    evaluate,
    limit_split,
    make_optimizer,
    make_predict_step,
    make_readout_diagnostics_step,
    make_train_step,
)
from ham_embed_spectral.models.reuploading import init_reuploading_params  # noqa: E402
from ham_embed_spectral.naming import (  # noqa: E402
    CANONICAL_BLOCK_HAMILTONIAN,
    CANONICAL_NON_OVERLAP_PATCH_BLOCK_HAMILTONIAN,
    CANONICAL_SYMMETRIC_HAMILTONIAN,
    CANONICAL_TRAINABLE_PATCH_SU4,
    ENCODER_CLI_CHOICES,
    canonical_encoder_name,
)
from ham_embed_spectral.quantum.encoders import (  # noqa: E402
    BlockHamiltonianEncoder,
    FixedRyEncoder,
    FixedRyRzEncoder,
    NonOverlapPatchBlockHamiltonianEncoder,
    PatchSU4Encoder,
    SymmetricHamiltonianEncoder,
    TrainableFrequencyRyEncoder,
    TrainablePatchSU4Encoder,
)
from ham_embed_spectral.quantum.readout import ceil_log2  # noqa: E402
from ham_embed_spectral.utils.checkpointing import (  # noqa: E402
    append_jsonl,
    reset_jsonl,
    restore_checkpoint_pytree,
    save_hdf5_pytree,
    save_orbax_pytree,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    data = parser.add_argument_group("data")
    data.add_argument(
        "--dataset",
        choices=("pendigits", "synthetic-eigengap", "synthetic-singular"),
        default="pendigits",
    )
    data.add_argument("--data-root", default="data/raw/pendigits")
    data.add_argument(
        "--representation",
        choices=("dyn", "sta4", "sta8", "sta16", "synthetic"),
        default="sta4",
    )
    data.add_argument("--download-data", action="store_true")
    data.add_argument("--validation-fraction", type=float, default=0.1)
    data.add_argument(
        "--class-subset",
        default=None,
        help="Comma-separated original labels, e.g. 3,8",
    )
    data.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    data.add_argument("--max-train-examples", type=int, default=None)
    data.add_argument("--max-eval-examples", type=int, default=None)

    synthetic = parser.add_argument_group("synthetic data")
    synthetic.add_argument("--n-samples", type=int, default=128)
    synthetic.add_argument("--synthetic-dim", type=int, default=4)
    synthetic.add_argument("--synthetic-rows", type=int, default=4)
    synthetic.add_argument("--synthetic-cols", type=int, default=2)
    synthetic.add_argument("--synthetic-threshold", type=float, default=None)
    synthetic.add_argument("--synthetic-noise-epsilon", type=float, default=0.0)

    model = parser.add_argument_group("model")
    model.add_argument(
        "--encoder",
        choices=ENCODER_CLI_CHOICES,
        default=CANONICAL_SYMMETRIC_HAMILTONIAN,
    )
    model.add_argument("--reupload-depth", type=int, default=1)
    model.add_argument("--mixer-scale", type=float, default=0.01)
    model.add_argument(
        "--projector-renormalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    model.add_argument("--ry-alpha", type=float, default=1.0)
    model.add_argument("--rz-beta", type=float, default=1.0)
    model.add_argument("--tf-init-scale", type=float, default=1.0)
    model.add_argument("--tf-init-noise", type=float, default=0.01)
    model.add_argument("--patch-scale", type=float, default=1.0)
    model.add_argument("--patch-map-init-noise", type=float, default=0.01)
    model.add_argument("--trainable-times", action=argparse.BooleanOptionalAction, default=True)
    model.add_argument("--fixed-time", type=float, default=None)
    model.add_argument(
        "--initial-state",
        choices=("plus", "zero"),
        default="plus",
        help="Initial state for all encoders; plus is |+>^{otimes n}.",
    )
    model.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    model.add_argument(
        "--track-readout-leakage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record unused projector-readout mass diagnostics for leakage-problem studies.",
    )

    train = parser.add_argument_group("training")
    train.add_argument("--seed", type=int, action="append", default=None)
    train.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds; merged with repeated --seed",
    )
    train.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Dataset split/generation seed. Defaults to the model initialization seed.",
    )
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--eval-batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=1e-2)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--log-every", type=int, default=10)
    train.add_argument("--eval-every", type=int, default=50)
    train.add_argument("--dry-run", action="store_true")

    output = parser.add_argument_group("outputs")
    output.add_argument("--output-root", default="results/runs")
    output.add_argument("--experiment-name", default="train")
    output.add_argument("--run-id", default=None)
    output.add_argument("--manifest-id", default=None)
    output.add_argument("--job-slug", default=None)
    output.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=False)
    output.add_argument("--checkpoint-format", choices=("hdf5", "orbax"), default="hdf5")
    output.add_argument("--checkpoint-every", type=int, default=0)
    output.add_argument(
        "--checkpoint-steps",
        default=None,
        help="Comma-separated exact training steps to checkpoint, e.g. 0,50,100,500.",
    )
    output.add_argument("--resume", default=None, help="Checkpoint reference to restore.")

    ablation = parser.add_argument_group("ablations")
    ablation.add_argument("--ablation", choices=ABLATION_CHOICES, default="none")
    ablation.add_argument("--ablation-seed", type=int, default=0)

    args = parser.parse_args()
    args.encoder = canonical_encoder_name(args.encoder)
    return args


def main() -> None:
    args = parse_args()
    if args.run_id is None:
        args.run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    seeds = resolve_seeds(args)
    summaries = []

    for seed in seeds:
        started = time.time()
        data_seed = seed if args.data_seed is None else args.data_seed
        dataset = load_dataset(args, data_seed)
        dataset = PendigitsDataset(
            train=limit_split(dataset.train, args.max_train_examples),
            validation=limit_split(dataset.validation, args.max_eval_examples),
            test=limit_split(dataset.test, args.max_eval_examples),
            representation=dataset.representation,
            input_shape=dataset.input_shape,
            class_values=dataset.class_values,
            feature_mean=dataset.feature_mean,
            feature_std=dataset.feature_std,
        )

        encoder = build_encoder(args)
        ablation_result = apply_dataset_ablation(
            dataset,
            ablation=args.ablation,
            encoder_name=args.encoder,
            seed=args.ablation_seed,
        )
        dataset = ablation_result.dataset
        model_config = build_model_config(args, encoder, dataset)
        loop_config = build_loop_config(args)
        checkpoint_steps = resolve_checkpoint_steps(args.checkpoint_steps)

        run_dir = make_run_dir(args, seed)
        metrics_jsonl_path = run_dir / "metrics.jsonl"
        reset_jsonl(metrics_jsonl_path)
        append_jsonl(
            metrics_jsonl_path,
            {
                "event": "run_start",
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "seed": seed,
                "manifest_id": args.manifest_id,
                "job_slug": args.job_slug,
                "run_dir": str(run_dir),
            },
        )

        optimizer = make_optimizer(loop_config)
        params = init_reuploading_params(
            jax.random.PRNGKey(seed),
            encoder,
            model_config,
            dtype=dtype_from_name(args.dtype),
        )
        if args.resume:
            state = restored_train_state(args.resume)
            params = state.params
        else:
            state = create_train_state(params, optimizer)
        write_json(
            run_dir / "config.json",
            config_payload(
                args,
                seed,
                data_seed,
                model_config,
                loop_config,
                dataset,
                params,
                checkpoint_steps,
                ablation_result.metadata,
            ),
        )
        train_step = make_train_step(encoder, model_config, optimizer)
        predict_step = make_predict_step(encoder, model_config)
        diagnostics_step = (
            make_readout_diagnostics_step(encoder, model_config)
            if args.track_readout_leakage
            else None
        )

        history = []
        train_batches = epoch_minibatches(
            dataset.train,
            loop_config.batch_size,
            seed=seed,
            shuffle=True,
            drop_remainder=False,
        )
        if args.checkpoint and 0 in checkpoint_steps and int(state.step) == 0:
            save_checkpoint(args, run_dir, "step_00000000", {"state": state})

        for step_index in range(loop_config.steps):
            batch_x, batch_y = next(train_batches)
            state, train_metrics = train_step(state, batch_x, batch_y)

            should_log = (step_index + 1) % loop_config.log_every == 0
            should_eval = (step_index + 1) % loop_config.eval_every == 0
            is_final = step_index + 1 == loop_config.steps
            if should_log or should_eval or is_final:
                record: dict[str, Any] = {
                    "elapsed_seconds": time.time() - started,
                    "step": int(state.step),
                    "train_loss": float(train_metrics["loss"]),
                    "train_accuracy": float(train_metrics["accuracy"]),
                    "grad_norm": float(train_metrics["grad_norm"]),
                }
                if diagnostics_step is not None:
                    train_diagnostics = diagnostics_step(state.params, batch_x)
                    record.update(
                        {
                            "train_mean_valid_readout_mass": float(
                                jnp.mean(train_diagnostics["valid_mass"])
                            ),
                            "train_mean_readout_leakage_mass": float(
                                jnp.mean(train_diagnostics["leakage_mass"])
                            ),
                        }
                    )
                if should_eval or is_final:
                    validation_metrics = evaluate(
                        state.params,
                        dataset.validation,
                        predict_step=predict_step,
                        diagnostics_step=diagnostics_step,
                        batch_size=loop_config.eval_batch_size,
                    )
                    record.update(
                        {
                            "validation_loss": validation_metrics["loss"],
                            "validation_accuracy": validation_metrics["accuracy"],
                        }
                    )
                    if diagnostics_step is not None:
                        record.update(
                            {
                                "validation_mean_valid_readout_mass": validation_metrics[
                                    "mean_valid_readout_mass"
                                ],
                                "validation_mean_readout_leakage_mass": validation_metrics[
                                    "mean_readout_leakage_mass"
                                ],
                            }
                        )
                history.append(record)
                append_jsonl(
                    metrics_jsonl_path,
                    {
                        "event": "history",
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                        "seed": seed,
                        "manifest_id": args.manifest_id,
                        "job_slug": args.job_slug,
                        **record,
                    },
                )
                print(format_record(run_dir, record), flush=True)

            should_checkpoint = args.checkpoint and (
                (args.checkpoint_every > 0 and (step_index + 1) % args.checkpoint_every == 0)
                or (step_index + 1) in checkpoint_steps
            )
            if should_checkpoint:
                save_checkpoint(args, run_dir, f"step_{int(state.step):08d}", {"state": state})

        final_validation = evaluate(
            state.params,
            dataset.validation,
            predict_step=predict_step,
            diagnostics_step=diagnostics_step,
            batch_size=loop_config.eval_batch_size,
        )
        final_test = evaluate(
            state.params,
            dataset.test,
            predict_step=predict_step,
            diagnostics_step=diagnostics_step,
            batch_size=loop_config.eval_batch_size,
        )
        summary = {
            "seed": seed,
            "manifest_id": args.manifest_id,
            "job_slug": args.job_slug,
            "run_dir": str(run_dir),
            "wall_time_seconds": time.time() - started,
            "final_validation": final_validation,
            "final_test": final_test,
            "completed": True,
        }
        append_jsonl(
            metrics_jsonl_path,
            {
                "event": "summary",
                "timestamp_utc": datetime.now(UTC).isoformat(),
                **summary,
            },
        )
        write_json(run_dir / "metrics.json", {"history": history, "summary": summary})
        if args.checkpoint:
            save_checkpoint(args, run_dir, "final", {"state": state})
        summaries.append(summary)

    write_json(
        Path(args.output_root) / args.experiment_name / f"{args.run_id}_summary.json",
        summaries,
    )


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    seeds = list(args.seed or [])
    if args.seeds:
        seeds.extend(int(part) for part in args.seeds.split(",") if part.strip())
    return seeds or [0]


def resolve_checkpoint_steps(raw: str | None) -> tuple[int, ...]:
    """Parse comma-separated explicit checkpoint steps."""

    if raw is None or not raw.strip():
        return ()
    steps = tuple(sorted({int(part) for part in raw.split(",") if part.strip()}))
    if any(step < 0 for step in steps):
        raise ValueError(f"checkpoint steps must be nonnegative, got {steps}")
    return steps


def load_dataset(args: argparse.Namespace, seed: int) -> PendigitsDataset:
    if args.dry_run:
        args.steps = min(args.steps, 3)
        args.max_train_examples = args.max_train_examples or 32
        args.max_eval_examples = args.max_eval_examples or 32

    class_subset = parse_class_subset(args.class_subset)
    dtype = dtype_from_name(args.dtype)
    if args.dataset == "pendigits":
        return prepare_pendigits(
            args.data_root,
            representation=args.representation,
            validation_fraction=args.validation_fraction,
            seed=seed,
            class_subset=class_subset,
            standardize=args.standardize,
            download=args.download_data,
            dtype=dtype,
        )

    key = jax.random.PRNGKey(seed)
    if args.dataset == "synthetic-eigengap":
        threshold = 0.75 if args.synthetic_threshold is None else args.synthetic_threshold
        raw = eigengap_classification(
            key,
            n_samples=args.n_samples,
            dim=args.synthetic_dim,
            threshold=threshold,
            noise_epsilon=args.synthetic_noise_epsilon,
        )
    else:
        threshold = 1.5 if args.synthetic_threshold is None else args.synthetic_threshold
        raw = singular_value_classification(
            key,
            n_samples=args.n_samples,
            shape=(args.synthetic_rows, args.synthetic_cols),
            threshold=threshold,
            noise_epsilon=args.synthetic_noise_epsilon,
        )
    return split_synthetic(raw.x.astype(dtype), raw.y, args.validation_fraction, seed)


def split_synthetic(
    x: jnp.ndarray,
    y: jnp.ndarray,
    validation_fraction: float,
    seed: int,
) -> PendigitsDataset:
    n_examples = int(y.shape[0])
    rng = np.random.default_rng(seed)
    indices = rng.permutation(np.arange(n_examples))
    n_test = max(1, int(round(0.2 * n_examples)))
    n_validation = max(1, int(round(validation_fraction * (n_examples - n_test))))
    test_indices = indices[:n_test]
    validation_indices = indices[n_test : n_test + n_validation]
    train_indices = indices[n_test + n_validation :]
    class_values = tuple(int(value) for value in np.unique(np.asarray(y)))
    return PendigitsDataset(
        train=PendigitsSplit(x=x[train_indices], y=y[train_indices]),
        validation=PendigitsSplit(x=x[validation_indices], y=y[validation_indices]),
        test=PendigitsSplit(x=x[test_indices], y=y[test_indices]),
        representation="synthetic",
        input_shape=tuple(x.shape[1:]),
        class_values=class_values,
        feature_mean=jnp.zeros(x.shape[1:], dtype=x.dtype),
        feature_std=jnp.ones(x.shape[1:], dtype=x.dtype),
    )


def build_encoder(args: argparse.Namespace):
    args.encoder = canonical_encoder_name(args.encoder)
    if args.encoder == "fixed-ry":
        return FixedRyEncoder(alpha=args.ry_alpha)
    if args.encoder == "fixed-ry-rz":
        return FixedRyRzEncoder(alpha=args.ry_alpha, beta=args.rz_beta)
    if args.encoder == "trainable-frequency-ry":
        return TrainableFrequencyRyEncoder(
            init_scale=args.tf_init_scale,
            init_noise=args.tf_init_noise,
        )
    if args.encoder == "patch-su4":
        return PatchSU4Encoder(scale=args.patch_scale)
    if args.encoder == CANONICAL_TRAINABLE_PATCH_SU4:
        return TrainablePatchSU4Encoder(
            scale=args.patch_scale,
            init_noise=args.patch_map_init_noise,
        )
    if args.encoder == CANONICAL_NON_OVERLAP_PATCH_BLOCK_HAMILTONIAN:
        return NonOverlapPatchBlockHamiltonianEncoder(
            trainable_times=args.trainable_times,
            fixed_time=args.fixed_time,
        )
    if args.encoder == CANONICAL_SYMMETRIC_HAMILTONIAN:
        return SymmetricHamiltonianEncoder(
            trainable_times=args.trainable_times,
            fixed_time=args.fixed_time,
        )
    if args.encoder == CANONICAL_BLOCK_HAMILTONIAN:
        return BlockHamiltonianEncoder(
            trainable_times=args.trainable_times,
            fixed_time=args.fixed_time,
        )
    raise ValueError(f"unknown encoder {args.encoder!r}")


def build_model_config(
    args: argparse.Namespace,
    encoder,
    dataset: PendigitsDataset,
) -> ReuploadingModelConfig:
    natural_n_qubits = encoder.n_qubits(dataset.input_shape)
    label_n_qubits = ceil_log2(dataset.n_classes)
    n_qubits = natural_n_qubits
    if isinstance(encoder, (SymmetricHamiltonianEncoder, BlockHamiltonianEncoder)):
        n_qubits = max(natural_n_qubits, label_n_qubits)
    elif label_n_qubits > natural_n_qubits:
        raise ValueError(
            f"encoder {args.encoder!r} has {natural_n_qubits} qubits, "
            f"but {dataset.n_classes} classes need {label_n_qubits} label qubits"
        )

    return ReuploadingModelConfig(
        input_shape=dataset.input_shape,
        n_classes=dataset.n_classes,
        reupload_depth=args.reupload_depth,
        n_qubits=n_qubits,
        initial_state=args.initial_state,
        mixer_scale=args.mixer_scale,
        projector_renormalize=args.projector_renormalize,
    )


def build_loop_config(args: argparse.Namespace) -> TrainingLoopConfig:
    return TrainingLoopConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        log_every=max(1, args.log_every),
        eval_every=max(1, args.eval_every),
    )


def make_run_dir(args: argparse.Namespace, seed: int) -> Path:
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path(args.output_root) / args.experiment_name / f"{run_id}_seed{seed}"


def save_checkpoint(args: argparse.Namespace, run_dir: Path, label: str, item: Any) -> Path:
    """Save a checkpoint label using the selected backend."""

    if args.checkpoint_format == "hdf5":
        return save_hdf5_pytree(run_dir / "checkpoints.h5", label, item)
    if args.checkpoint_format == "orbax":
        return save_orbax_pytree(run_dir / "checkpoints" / label, item)
    raise ValueError(f"unsupported checkpoint format: {args.checkpoint_format}")


def config_payload(
    args: argparse.Namespace,
    seed: int,
    data_seed: int,
    model_config: ReuploadingModelConfig,
    loop_config: TrainingLoopConfig,
    dataset: PendigitsDataset,
    params: Any,
    checkpoint_steps: tuple[int, ...],
    ablation_metadata: dict[str, Any],
) -> dict[str, Any]:
    git = git_info()
    return {
        "args": vars(args),
        "seed": seed,
        "data_seed": data_seed,
        "manifest_id": args.manifest_id,
        "job_slug": args.job_slug,
        "command_line": sys.argv,
        "git": git,
        "model_config": {
            "input_shape": model_config.input_shape,
            "n_classes": model_config.n_classes,
            "reupload_depth": model_config.reupload_depth,
            "n_qubits": model_config.n_qubits,
            "hilbert_dim": 2 ** int(model_config.n_qubits or 0),
            "initial_state": model_config.initial_state,
            "mixer_scale": model_config.mixer_scale,
            "projector_renormalize": model_config.projector_renormalize,
            "track_readout_leakage": args.track_readout_leakage,
            "patch_map_init_noise": args.patch_map_init_noise,
            "parameter_count": parameter_count(params),
            "requested_dtype": args.dtype,
            "dtype_policy": "x64_enabled_parameter_dtype",
        },
        "training": {
            "steps": loop_config.steps,
            "batch_size": loop_config.batch_size,
            "eval_batch_size": loop_config.eval_batch_size,
            "learning_rate": loop_config.learning_rate,
            "weight_decay": loop_config.weight_decay,
            "log_every": loop_config.log_every,
            "eval_every": loop_config.eval_every,
            "checkpoint": args.checkpoint,
            "checkpoint_format": args.checkpoint_format,
            "checkpoint_every": args.checkpoint_every,
            "checkpoint_steps": checkpoint_steps,
        },
        "dataset": {
            "representation": dataset.representation,
            "input_shape": dataset.input_shape,
            "class_values": dataset.class_values,
            "n_train": int(dataset.train.y.shape[0]),
            "n_validation": int(dataset.validation.y.shape[0]),
            "n_test": int(dataset.test.y.shape[0]),
            "data_seed": data_seed,
        },
        "ablation": ablation_metadata,
        "jax": {
            "enable_x64": jax.config.read("jax_enable_x64"),
            "devices": [str(device) for device in jax.devices()],
        },
        "packages": package_versions(),
    }


def parse_class_subset(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or not raw.strip():
        return None
    return tuple(int(part) for part in raw.split(",") if part.strip())


def dtype_from_name(name: str):
    return jnp.float64 if name == "float64" else jnp.float32


def restored_train_state(path: str | Path):
    """Restore a training state saved as ``{\"state\": state}``."""

    payload = restore_checkpoint_pytree(path)
    state = payload["state"] if isinstance(payload, Mapping) and "state" in payload else payload
    if isinstance(state, Mapping):
        return TrainState(
            step=jnp.asarray(state["step"]),
            params=state["params"],
            opt_state=state["opt_state"],
        )
    return state


def parameter_count(params: Any) -> int:
    """Count scalar leaves in a parameter pytree."""

    total = 0
    for leaf in jax.tree.leaves(params):
        if hasattr(leaf, "size"):
            total += int(leaf.size)
    return total


def git_info() -> dict[str, Any]:
    """Return current git commit and dirty status when available."""

    commit = _run_git(["git", "rev-parse", "HEAD"])
    status = _run_git(["git", "status", "--short"])
    return {
        "commit": commit or None,
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def _run_git(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def package_versions() -> dict[str, str | None]:
    """Record key package versions for reproducibility."""

    packages = (
        "jax",
        "flax",
        "optax",
        "orbax",
        "h5py",
        "numpy",
        "matplotlib",
        "plotly",
        "scienceplots",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def format_record(run_dir: Path, record: dict[str, Any]) -> str:
    parts = [f"run={run_dir}", f"step={record['step']}"]
    for key in (
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "grad_norm",
        "train_mean_valid_readout_mass",
        "train_mean_readout_leakage_mass",
        "validation_mean_valid_readout_mass",
        "validation_mean_readout_leakage_mass",
    ):
        if key in record:
            parts.append(f"{key}={record[key]:.6g}")
    return " ".join(parts)


if __name__ == "__main__":
    main()
