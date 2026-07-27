#!/usr/bin/env python3
"""Generate and optionally submit one-training-run SLURM scripts."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ham_embed_spectral.experiments.manifest import (  # noqa: E402
    DATASET_CHOICES,
    REPRESENTATION_CHOICES,
    TrainJob,
    jobs_from_manifest,
    load_manifest,
    manifest_section,
    normalize_optional_value,
    sanitize_slug,
)
from ham_embed_spectral.naming import (  # noqa: E402
    CANONICAL_SYMMETRIC_HAMILTONIAN,
    ENCODER_CLI_CHOICES,
    canonical_encoder_name,
)


@dataclass(frozen=True)
class TimingKey:
    """Key identifying timing rows, excluding model seed and depth."""

    manifest_id: str
    dataset: str
    representation: str
    encoder: str
    learning_rate: float
    batch_size: int
    steps: int
    ablation: str = "none"
    ablation_seed: int = 0


@dataclass(frozen=True)
class TimedTrainJob:
    """One training job with raw and safety-padded wall-time estimates."""

    job: TrainJob
    raw_seconds: float
    padded_seconds: float


@dataclass(frozen=True)
class TrainJobBatch:
    """A sequential SLURM batch of training jobs."""

    slug: str
    jobs: tuple[TimedTrainJob, ...]
    raw_seconds: float
    padded_seconds: float

    @property
    def member_job_slugs(self) -> list[str]:
        return [timed.job.slug for timed in self.jobs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    manifest = parser.add_argument_group("manifest")
    manifest.add_argument(
        "--manifest",
        default=None,
        help="JSON experiment manifest. When set, manifest grid values replace CLI grid values.",
    )
    manifest.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print planned job slugs and active SLURM defaults without writing "
            "or submitting scripts."
        ),
    )

    slurm = parser.add_argument_group(
        "slurm resources",
        "Cluster resource requests written directly to #SBATCH lines.",
    )
    slurm.add_argument(
        "--project-root",
        default=str(Path.cwd()),
        help="Absolute project path used inside each submitted job.",
    )
    slurm.add_argument(
        "--venv-dir",
        default=".venv",
        help="Virtual environment directory relative to --project-root.",
    )
    slurm.add_argument(
        "--slurm-dir",
        default="slurm/train",
        help="Directory where generated .slurm files are written.",
    )
    slurm.add_argument(
        "--log-dir",
        default="sbatch_log",
        help="SLURM stdout/stderr directory relative to --project-root.",
    )
    slurm.add_argument("--job-name-prefix", default="qfm_train", help="Prefix for SLURM job names.")
    slurm.add_argument("--account", default=None, help="Optional SLURM account charged by sbatch.")
    slurm.add_argument("--partition", default=None, help="Optional SLURM partition.")
    slurm.add_argument("--qos", default=None, help="Optional SLURM QoS.")
    slurm.add_argument("--reservation", default=None, help="Optional SLURM reservation.")
    slurm.add_argument("--constraint", default=None, help="Optional SLURM node constraint.")
    slurm.add_argument("--exclude", default=None, help="Optional comma-separated nodes to exclude.")
    slurm.add_argument("--time", default="48:00:00", help="Wall-clock time limit.")
    slurm.add_argument("--mem", default="32GB", help="Memory requested per node.")
    slurm.add_argument("--gpus-per-node", type=int, default=1, help="GPUs requested per node.")
    slurm.add_argument("--cpus-per-gpu", type=int, default=16, help="CPU cores requested per GPU.")
    slurm.add_argument("--nodes", type=int, default=1, help="Number of nodes.")
    slurm.add_argument("--ntasks-per-node", type=int, default=1, help="Tasks per node.")
    slurm.add_argument("--mail-type", default="ALL", help="SLURM mail notification type.")
    slurm.add_argument("--mail-user", default=None, help="Optional email address for SLURM mail.")
    slurm.add_argument(
        "--module",
        action="append",
        default=None,
        help="Module to load. Repeatable. Defaults to uv/0.9.5, cuda/12.9.1, openmpi.",
    )
    slurm.add_argument(
        "--submit",
        action="store_true",
        help="Call sbatch for every generated script.",
    )
    slurm.add_argument(
        "--submit-log",
        default=None,
        help=(
            "JSONL ledger of sbatch attempts. Defaults to "
            "<slurm-dir>/submissions.jsonl when submitting or reporting."
        ),
    )
    slurm.add_argument(
        "--skip-submitted",
        action="store_true",
        help="With --submit, skip job slugs already marked submitted in --submit-log.",
    )
    slurm.add_argument(
        "--submission-report",
        action="store_true",
        help=(
            "Print a local submission report from the manifest, optional "
            "--first-failed-script, and --submit-log, then exit."
        ),
    )
    slurm.add_argument(
        "--first-failed-script",
        default=None,
        help=(
            "Script path from a failed sbatch traceback. In --submission-report, this "
            "script and every later manifest-order script are reported as not submitted "
            "by that aborted invocation. With --submit, only this script and later "
            "manifest-order scripts are submitted."
        ),
    )
    slurm.add_argument(
        "--print-slurm-defaults",
        action="store_true",
        help="Print the SLURM resource defaults as JSON and exit.",
    )

    batch = parser.add_argument_group(
        "batching",
        "Optional sequential batching of many training commands into fewer SLURM scripts.",
    )
    batch.add_argument(
        "--batch-from-timing",
        default=None,
        help=(
            "Timing benchmark CSV used to pack jobs into sequential SLURM batches. "
            "When unset, one script is generated per training job."
        ),
    )
    batch.add_argument(
        "--batch-safety-margin",
        type=float,
        default=2.0,
        help="Multiplier applied to each raw timing estimate before packing.",
    )
    batch.add_argument(
        "--batch-time",
        default="24:00:00",
        help="Wall-clock time limit and padded packing capacity for batch scripts.",
    )
    batch.add_argument(
        "--batch-mem",
        default="16GB",
        help="Memory requested per batched SLURM script.",
    )
    batch.add_argument(
        "--batch-slurm-dir",
        default=None,
        help="Directory for generated batch scripts. Defaults to '<slurm-dir>_batched'.",
    )

    grid = parser.add_argument_group(
        "training grid",
        "Cartesian product of these values. Every combination becomes one SLURM job.",
    )
    grid.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, default=["pendigits"])
    grid.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATION_CHOICES,
        default=["sta4"],
    )
    grid.add_argument(
        "--encoders",
        nargs="+",
        choices=ENCODER_CLI_CHOICES,
        default=None,
    )
    grid.add_argument("--reupload-depths", nargs="+", type=int, default=[1])
    grid.add_argument("--seeds", nargs="+", type=int, default=[0])
    grid.add_argument("--learning-rates", nargs="+", type=float, default=[1e-2])
    grid.add_argument("--batch-sizes", nargs="+", type=int, default=[32])
    grid.add_argument(
        "--class-subsets",
        nargs="*",
        default=[None],
        help="Optional comma-separated class subsets. Use 'none' for full 10-class task.",
    )

    train = parser.add_argument_group(
        "scripts/train.py arguments",
        "Arguments forwarded into the single training command in each SLURM script.",
    )
    train.add_argument("--data-root", default="data/raw/pendigits")
    train.add_argument("--data-seed", type=int, default=None)
    train.add_argument("--download-data", action="store_true")
    train.add_argument("--validation-fraction", type=float, default=0.1)
    train.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--n-samples", type=int, default=128)
    train.add_argument("--synthetic-dim", type=int, default=4)
    train.add_argument("--synthetic-rows", type=int, default=4)
    train.add_argument("--synthetic-cols", type=int, default=2)
    train.add_argument("--synthetic-threshold", type=float, default=None)
    train.add_argument("--synthetic-noise-epsilon", type=float, default=0.0)
    train.add_argument("--train-steps", type=int, default=1000)
    train.add_argument("--eval-batch-size", type=int, default=128)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--log-every", type=int, default=50)
    train.add_argument("--eval-every", type=int, default=100)
    train.add_argument("--output-root", default="results/runs")
    train.add_argument("--experiment-name", default="train_slurm")
    train.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--checkpoint-format", choices=("hdf5", "orbax"), default="hdf5")
    train.add_argument("--checkpoint-every", type=int, default=0)
    train.add_argument("--checkpoint-steps", default=None)
    train.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    train.add_argument("--mixer-scale", type=float, default=0.01)
    train.add_argument(
        "--projector-renormalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument("--ry-alpha", type=float, default=1.0)
    train.add_argument("--rz-beta", type=float, default=1.0)
    train.add_argument("--tf-init-scale", type=float, default=1.0)
    train.add_argument("--tf-init-noise", type=float, default=0.01)
    train.add_argument("--patch-scale", type=float, default=1.0)
    train.add_argument("--patch-map-init-noise", type=float, default=0.01)
    train.add_argument("--trainable-times", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--fixed-time", type=float, default=None)
    train.add_argument(
        "--initial-state",
        choices=("plus", "zero"),
        default="plus",
        help="Initial state forwarded to scripts/train.py.",
    )
    train.add_argument(
        "--track-readout-leakage",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    train.add_argument(
        "--ablation",
        choices=(
            "none",
            "entry-permutation",
            "row-column-permutation",
            "spectrum-only",
            "eigenvector-only",
            "singular-spectrum-only",
            "singular-vector-only",
        ),
        default="none",
    )
    train.add_argument("--ablation-seed", type=int, default=0)
    train.add_argument(
        "--extra-train-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments appended verbatim after '--'.",
    )

    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> list[TrainJob]:
    """Expand the CLI grid into one job per model training run."""

    if getattr(args, "manifest", None):
        manifest = getattr(args, "_manifest_payload", None) or load_manifest(args.manifest)
        return jobs_from_manifest(manifest, encoders=args.encoders)

    class_subsets = [normalize_optional_value(value) for value in args.class_subsets]
    encoders = args.encoders or [CANONICAL_SYMMETRIC_HAMILTONIAN]
    jobs = []
    for combo in itertools.product(
        args.datasets,
        args.representations,
        encoders,
        args.reupload_depths,
        args.seeds,
        args.learning_rates,
        args.batch_sizes,
        class_subsets,
    ):
        (
            dataset,
            representation,
            encoder,
            depth,
            seed,
            learning_rate,
            batch_size,
            class_subset,
        ) = combo
        jobs.append(
            TrainJob(
                dataset=dataset,
                representation=representation,
                encoder=encoder,
                reupload_depth=depth,
                seed=seed,
                learning_rate=learning_rate,
                batch_size=batch_size,
                class_subset=class_subset,
            )
        )
    return jobs


def apply_manifest_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Apply manifest defaults to argparse values.

    Manifest values are the source of truth for experiment definitions. CLI
    resource flags remain available for adapting jobs to a particular SLURM
    server; values that are not present in the manifest keep parser defaults.
    """

    if not args.manifest:
        return args

    manifest = load_manifest(args.manifest)
    args._manifest_payload = manifest

    grid = manifest["grid"]
    requested_encoders = args.encoders
    args.datasets = grid["datasets"]
    args.representations = grid["representations"]
    manifest_encoders = [canonical_encoder_name(value) for value in grid["encoders"]]
    if requested_encoders is None:
        args.encoders = manifest_encoders
    else:
        requested = [canonical_encoder_name(value) for value in requested_encoders]
        missing = [value for value in requested if value not in manifest_encoders]
        if missing:
            raise ValueError(
                f"manifest {manifest['manifest_id']!r} does not include requested encoder(s): "
                f"{', '.join(missing)}"
            )
        requested_set = set(requested)
        args.encoders = [value for value in manifest_encoders if value in requested_set]
    args.reupload_depths = [int(value) for value in grid["reupload_depths"]]
    args.seeds = [int(value) for value in grid["seeds"]]
    args.learning_rates = [float(value) for value in grid["learning_rates"]]
    args.batch_sizes = [int(value) for value in grid["batch_sizes"]]
    args.class_subsets = grid.get("class_subsets", [None])

    data = manifest_section(manifest, "data")
    synthetic = manifest_section(manifest, "synthetic")
    training = manifest_section(manifest, "training")
    model = manifest_section(manifest, "model")
    outputs = manifest_section(manifest, "outputs")
    slurm = manifest_section(manifest, "slurm")

    _set_present(args, data, "data_root")
    _set_present(args, data, "data_seed")
    _set_present(args, data, "download_data")
    _set_present(args, data, "validation_fraction")
    _set_present(args, data, "standardize")
    _set_present(args, synthetic, "n_samples")
    _set_present(args, synthetic, "synthetic_dim")
    _set_present(args, synthetic, "synthetic_rows")
    _set_present(args, synthetic, "synthetic_cols")
    _set_present(args, synthetic, "synthetic_threshold")
    _set_present(args, synthetic, "synthetic_noise_epsilon")
    args._synthetic_dataset_settings = synthetic.get("datasets", {})
    _set_present(args, training, "steps", arg_name="train_steps")
    _set_present(args, training, "eval_batch_size")
    _set_present(args, training, "weight_decay")
    _set_present(args, training, "log_every")
    _set_present(args, training, "eval_every")
    _set_present(args, model, "mixer_scale")
    _set_present(args, model, "projector_renormalize")
    _set_present(args, model, "ry_alpha")
    _set_present(args, model, "rz_beta")
    _set_present(args, model, "tf_init_scale")
    _set_present(args, model, "tf_init_noise")
    _set_present(args, model, "patch_scale")
    _set_present(args, model, "patch_map_init_noise")
    _set_present(args, model, "trainable_times")
    _set_present(args, model, "fixed_time")
    _set_present(args, model, "initial_state")
    _set_present(args, model, "dtype")
    _set_present(args, model, "track_readout_leakage")
    _set_present(args, outputs, "output_root")
    _set_present(args, outputs, "experiment_name")
    _set_present(args, outputs, "checkpoint")
    _set_present(args, outputs, "checkpoint_format")
    _set_present(args, outputs, "checkpoint_every")
    _set_present(args, outputs, "checkpoint_steps")
    ablations = manifest_section(manifest, "ablations")
    _set_present(args, ablations, "ablation")
    _set_present(args, ablations, "ablation_seed")

    for key in (
        "project_root",
        "venv_dir",
        "slurm_dir",
        "log_dir",
        "job_name_prefix",
        "account",
        "partition",
        "qos",
        "reservation",
        "constraint",
        "exclude",
        "time",
        "mem",
        "gpus_per_node",
        "cpus_per_gpu",
        "nodes",
        "ntasks_per_node",
        "mail_type",
        "mail_user",
    ):
        _set_present(args, slurm, key)
    if "modules" in slurm:
        args.module = list(slurm["modules"])

    return args


def _set_present(
    args: argparse.Namespace,
    section: dict[str, object],
    key: str,
    *,
    arg_name: str | None = None,
) -> None:
    if key in section:
        setattr(args, arg_name or key, section[key])


def render_slurm_script(job: TrainJob, args: argparse.Namespace) -> str:
    """Render one SLURM script for one training job."""

    modules = args.module or ["uv/0.9.5", "cuda/12.9.1", "openmpi"]
    job_name = slurm_job_name(job, args)
    output_pattern = f"{args.log_dir}/%x-%j.out"
    sbatch_lines = "\n".join(sbatch_directives(args, job_name, output_pattern))
    module_lines = "\n".join(f"module load {shlex.quote(module)}" for module in modules)
    train_command = format_shell_command(build_train_command(job, args))

    return f"""#!/bin/bash
{sbatch_lines}

set -euo pipefail

PROJECT_ROOT={shlex.quote(str(args.project_root))}
VENV_DIR={shlex.quote(args.venv_dir)}

cd "$PROJECT_ROOT"
{module_lines}
export JAX_PLATFORMS=cuda

source "$PROJECT_ROOT/$VENV_DIR/bin/activate"

{train_command}
"""


def sbatch_directives(
    args: argparse.Namespace,
    job_name: str,
    output_pattern: str,
    *,
    time: str | None = None,
    mem: str | None = None,
) -> list[str]:
    """Return explicit SBATCH lines for the selected SLURM configuration."""

    directives = [
        ("job-name", job_name),
        ("time", args.time if time is None else time),
        ("mem", args.mem if mem is None else mem),
        ("gpus-per-node", args.gpus_per_node),
        ("cpus-per-gpu", args.cpus_per_gpu),
        ("nodes", args.nodes),
        ("account", args.account),
        ("ntasks-per-node", args.ntasks_per_node),
        ("mail-type", args.mail_type),
        ("output", f'"{output_pattern}"'),
        ("partition", args.partition),
        ("qos", args.qos),
        ("reservation", args.reservation),
        ("constraint", args.constraint),
        ("exclude", args.exclude),
        ("mail-user", args.mail_user),
    ]
    return [f"#SBATCH --{name}={value}" for name, value in directives if value is not None]


def slurm_job_name(job: TrainJob, args: argparse.Namespace) -> str:
    """Return the job name written to ``#SBATCH --job-name``."""

    return sanitize_slug(f"{args.job_name_prefix}__{job.slug}")[:128]


def build_train_command(job: TrainJob, args: argparse.Namespace) -> list[str]:
    """Build the single scripts/train.py command for a job."""

    command = [
        "uv",
        "run",
        "scripts/train.py",
        "--dataset",
        job.dataset,
        "--representation",
        job.representation,
        "--encoder",
        job.encoder,
        "--reupload-depth",
        str(job.reupload_depth),
        "--seed",
        str(job.seed),
        "--steps",
        str(args.train_steps),
        "--batch-size",
        str(job.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--learning-rate",
        f"{job.learning_rate:g}",
        "--weight-decay",
        f"{args.weight_decay:g}",
        "--log-every",
        str(args.log_every),
        "--eval-every",
        str(args.eval_every),
        "--data-root",
        args.data_root,
        "--validation-fraction",
        f"{args.validation_fraction:g}",
        "--output-root",
        train_output_root(args),
        "--experiment-name",
        args.experiment_name,
        "--run-id",
        job.slug,
        "--dtype",
        args.dtype,
        "--manifest-id",
        manifest_id(args),
        "--job-slug",
        job.slug,
        "--mixer-scale",
        f"{args.mixer_scale:g}",
        "--ry-alpha",
        f"{args.ry_alpha:g}",
        "--rz-beta",
        f"{args.rz_beta:g}",
        "--tf-init-scale",
        f"{args.tf_init_scale:g}",
        "--tf-init-noise",
        f"{args.tf_init_noise:g}",
        "--patch-scale",
        f"{args.patch_scale:g}",
        "--patch-map-init-noise",
        f"{args.patch_map_init_noise:g}",
        "--initial-state",
        args.initial_state,
    ]
    if args.data_seed is not None:
        command.extend(["--data-seed", str(args.data_seed)])
    if job.dataset.startswith("synthetic-"):
        command.extend(synthetic_train_args(job, args))
    command.append(
        "--projector-renormalize" if args.projector_renormalize else "--no-projector-renormalize"
    )
    command.append("--trainable-times" if args.trainable_times else "--no-trainable-times")
    if args.fixed_time is not None:
        command.extend(["--fixed-time", f"{args.fixed_time:g}"])
    command.append(
        "--track-readout-leakage" if args.track_readout_leakage else "--no-track-readout-leakage"
    )
    if job.class_subset is not None:
        command.extend(["--class-subset", job.class_subset])
    if args.download_data:
        command.append("--download-data")
    if not args.standardize:
        command.append("--no-standardize")
    command.append("--checkpoint" if args.checkpoint else "--no-checkpoint")
    if args.checkpoint:
        command.extend(["--checkpoint-format", args.checkpoint_format])
    if args.checkpoint_every:
        command.extend(["--checkpoint-every", str(args.checkpoint_every)])
    if args.checkpoint_steps:
        command.extend(["--checkpoint-steps", str(args.checkpoint_steps)])
    if args.ablation != "none":
        command.extend(["--ablation", args.ablation, "--ablation-seed", str(args.ablation_seed)])
    command.extend(strip_remainder_separator(args.extra_train_args))
    return command


def train_output_root(args: argparse.Namespace) -> str:
    """Return the output root path used inside the generated SLURM job."""

    output_root = Path(args.output_root).expanduser()
    if output_root.is_absolute():
        return str(output_root)
    project_root = Path(args.project_root).expanduser().resolve(strict=False)
    return str(project_root / output_root)


def synthetic_train_args(job: TrainJob, args: argparse.Namespace) -> list[str]:
    """Return synthetic-data arguments, including manifest per-dataset overrides."""

    settings = dict(getattr(args, "_synthetic_dataset_settings", {}).get(job.dataset, {}))
    n_samples = settings.get("n_samples", args.n_samples)
    synthetic_dim = settings.get("synthetic_dim", args.synthetic_dim)
    synthetic_rows = settings.get("synthetic_rows", args.synthetic_rows)
    synthetic_cols = settings.get("synthetic_cols", args.synthetic_cols)
    synthetic_threshold = settings.get("synthetic_threshold", args.synthetic_threshold)
    synthetic_noise_epsilon = settings.get(
        "synthetic_noise_epsilon",
        args.synthetic_noise_epsilon,
    )
    command = [
        "--n-samples",
        str(n_samples),
        "--synthetic-dim",
        str(synthetic_dim),
        "--synthetic-rows",
        str(synthetic_rows),
        "--synthetic-cols",
        str(synthetic_cols),
        "--synthetic-noise-epsilon",
        f"{synthetic_noise_epsilon:g}",
    ]
    if synthetic_threshold is not None:
        command.extend(["--synthetic-threshold", f"{synthetic_threshold:g}"])
    return command


def manifest_id(args: argparse.Namespace) -> str:
    """Return the active manifest id or a CLI-grid sentinel."""

    if getattr(args, "manifest", None):
        manifest = getattr(args, "_manifest_payload", None) or load_manifest(args.manifest)
        return str(manifest["manifest_id"])
    return "cli_grid"


def load_timing_estimates(path: Path) -> dict[TimingKey, dict[int, float]]:
    """Load representative wall-time estimates keyed by job attributes and depth."""

    required = {
        "status",
        "manifest_id",
        "dataset",
        "representation",
        "encoder",
        "reupload_depth",
        "learning_rate",
        "batch_size",
        "steps",
        "returncode",
        "subprocess_wall_time_seconds",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            missing_fields = ", ".join(sorted(missing))
            raise ValueError(f"timing CSV {path} is missing required fields: {missing_fields}")
        grouped: dict[TimingKey, dict[int, list[float]]] = {}
        for row in reader:
            if row.get("status") != "complete" or str(row.get("returncode")) != "0":
                continue
            wall_time = parse_float_field(row, "subprocess_wall_time_seconds")
            key = TimingKey(
                manifest_id=str(row["manifest_id"]),
                dataset=str(row["dataset"]),
                representation=str(row["representation"]),
                encoder=str(row["encoder"]),
                learning_rate=parse_float_field(row, "learning_rate"),
                batch_size=parse_int_field(row, "batch_size"),
                steps=parse_int_field(row, "steps"),
                ablation=str(row.get("ablation") or "none"),
                ablation_seed=parse_optional_int_field(row, "ablation_seed", default=0),
            )
            depth = parse_int_field(row, "reupload_depth")
            grouped.setdefault(key, {}).setdefault(depth, []).append(wall_time)

    estimates = {
        key: {depth: mean(values) for depth, values in sorted(depths.items())}
        for key, depths in grouped.items()
    }
    if not estimates:
        raise ValueError(f"timing CSV {path} has no complete successful timing rows")
    return estimates


def parse_float_field(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"invalid float in timing field {field!r}: {value!r}") from exc


def parse_int_field(row: dict[str, str], field: str) -> int:
    value = row.get(field, "")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"invalid integer in timing field {field!r}: {value!r}") from exc


def parse_optional_int_field(row: dict[str, str], field: str, *, default: int) -> int:
    value = row.get(field, "")
    if value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"invalid integer in timing field {field!r}: {value!r}") from exc


def timing_key_for_job(job: TrainJob, args: argparse.Namespace) -> TimingKey:
    """Return the timing lookup key for a generated training job."""

    return TimingKey(
        manifest_id=manifest_id(args),
        dataset=job.dataset,
        representation=job.representation,
        encoder=job.encoder,
        learning_rate=float(job.learning_rate),
        batch_size=int(job.batch_size),
        steps=int(args.train_steps),
        ablation=str(args.ablation),
        ablation_seed=int(args.ablation_seed),
    )


def estimate_job_seconds(
    job: TrainJob,
    args: argparse.Namespace,
    estimates: dict[TimingKey, dict[int, float]],
) -> float:
    """Return a raw wall-time estimate for one job, interpolating missing depths."""

    key = timing_key_for_job(job, args)
    depth_estimates = estimates.get(key)
    if not depth_estimates:
        raise ValueError(f"no timing estimates found for {key}")
    depth = int(job.reupload_depth)
    if depth in depth_estimates:
        return depth_estimates[depth]

    measured_depths = sorted(depth_estimates)
    lower = [value for value in measured_depths if value < depth]
    upper = [value for value in measured_depths if value > depth]
    if not lower or not upper:
        raise ValueError(
            f"cannot interpolate timing for {job.slug}: depth {depth} is not bracketed by "
            f"measured depths {measured_depths}"
        )
    lo = lower[-1]
    hi = upper[0]
    lo_seconds = depth_estimates[lo]
    hi_seconds = depth_estimates[hi]
    fraction = (depth - lo) / (hi - lo)
    return lo_seconds + (hi_seconds - lo_seconds) * fraction


def build_train_job_batches(jobs: list[TrainJob], args: argparse.Namespace) -> list[TrainJobBatch]:
    """Build first-fit-decreasing sequential batches from timing estimates."""

    if args.batch_safety_margin <= 0:
        raise ValueError("--batch-safety-margin must be positive")
    capacity_seconds = parse_slurm_time_seconds(args.batch_time)
    estimates = load_timing_estimates(Path(args.batch_from_timing))
    timed_jobs = [
        TimedTrainJob(
            job=job,
            raw_seconds=raw_seconds,
            padded_seconds=raw_seconds * args.batch_safety_margin,
        )
        for job in jobs
        for raw_seconds in [estimate_job_seconds(job, args, estimates)]
    ]
    return pack_timed_jobs(timed_jobs, capacity_seconds)


def pack_timed_jobs(
    timed_jobs: list[TimedTrainJob],
    capacity_seconds: float,
) -> list[TrainJobBatch]:
    """Pack jobs with first-fit decreasing using padded wall-time estimates."""

    bins: list[list[TimedTrainJob]] = []
    used_seconds: list[float] = []
    ordered_jobs = sorted(
        timed_jobs,
        key=lambda timed: (-timed.padded_seconds, timed.job.slug),
    )
    for timed in ordered_jobs:
        if timed.padded_seconds > capacity_seconds:
            raise ValueError(
                f"{timed.job.slug} padded estimate {timed.padded_seconds:.1f}s exceeds "
                f"batch capacity {capacity_seconds:.1f}s"
            )
        for index, used in enumerate(used_seconds):
            if used + timed.padded_seconds <= capacity_seconds + 1e-9:
                bins[index].append(timed)
                used_seconds[index] += timed.padded_seconds
                break
        else:
            bins.append([timed])
            used_seconds.append(timed.padded_seconds)

    batches = []
    for index, members in enumerate(bins, start=1):
        batches.append(
            TrainJobBatch(
                slug=f"batch_{index:04d}",
                jobs=tuple(members),
                raw_seconds=sum(timed.raw_seconds for timed in members),
                padded_seconds=sum(timed.padded_seconds for timed in members),
            )
        )
    return batches


def parse_slurm_time_seconds(value: str) -> float:
    """Parse a SLURM time string into seconds."""

    day_count = 0
    time_part = value
    if "-" in value:
        days, time_part = value.split("-", 1)
        day_count = int(days)
    pieces = [int(piece) for piece in time_part.split(":")]
    if len(pieces) == 3:
        hours, minutes, seconds = pieces
    elif len(pieces) == 2:
        hours = 0
        minutes, seconds = pieces
    elif len(pieces) == 1:
        hours = 0
        minutes = pieces[0]
        seconds = 0
    else:
        raise ValueError(f"invalid SLURM time value: {value!r}")
    total = day_count * 86400 + hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise ValueError(f"SLURM time value must be positive: {value!r}")
    return float(total)


def batch_slurm_dir(args: argparse.Namespace) -> Path:
    """Return the output directory for generated batch scripts."""

    if args.batch_slurm_dir:
        return Path(args.batch_slurm_dir)
    return Path(f"{args.slurm_dir}_batched")


def batch_job_name(batch: TrainJobBatch, args: argparse.Namespace) -> str:
    """Return the SLURM job name for a sequential batch script."""

    return sanitize_slug(f"{args.job_name_prefix}__{batch.slug}")[:128]


def render_batch_slurm_script(batch: TrainJobBatch, args: argparse.Namespace) -> str:
    """Render one SLURM script that runs multiple training commands sequentially."""

    modules = args.module or ["uv/0.9.5", "cuda/12.9.1", "openmpi"]
    output_pattern = f"{args.log_dir}/%x-%j.out"
    sbatch_lines = "\n".join(
        sbatch_directives(
            args,
            batch_job_name(batch, args),
            output_pattern,
            time=args.batch_time,
            mem=args.batch_mem,
        )
    )
    module_lines = "\n".join(f"module load {shlex.quote(module)}" for module in modules)
    train_commands = "\n\n".join(
        format_shell_command(build_train_command(timed.job, args)) for timed in batch.jobs
    )

    return f"""#!/bin/bash
{sbatch_lines}

set -euo pipefail

PROJECT_ROOT={shlex.quote(str(args.project_root))}
VENV_DIR={shlex.quote(args.venv_dir)}

cd "$PROJECT_ROOT"
{module_lines}
export JAX_PLATFORMS=cuda

source "$PROJECT_ROOT/$VENV_DIR/bin/activate"

{train_commands}
"""


def write_slurm_scripts(jobs: list[TrainJob], args: argparse.Namespace) -> list[Path]:
    """Write generated SLURM scripts and ensure log directories exist."""

    slurm_dir = Path(args.slurm_dir)
    slurm_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(args.project_root)
    (project_root / args.log_dir).mkdir(parents=True, exist_ok=True)

    paths = []
    for job in jobs:
        path = slurm_dir / f"{job.slug}.slurm"
        path.write_text(render_slurm_script(job, args))
        paths.append(path)
    return paths


def write_batch_slurm_scripts(batches: list[TrainJobBatch], args: argparse.Namespace) -> list[Path]:
    """Write generated sequential batch SLURM scripts."""

    slurm_dir = batch_slurm_dir(args)
    slurm_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(args.project_root)
    (project_root / args.log_dir).mkdir(parents=True, exist_ok=True)

    paths = []
    for batch in batches:
        path = slurm_dir / f"{batch.slug}.slurm"
        path.write_text(render_batch_slurm_script(batch, args))
        paths.append(path)
    return paths


def slurm_script_paths(jobs: list[TrainJob], args: argparse.Namespace) -> list[Path]:
    """Return generated script paths without writing them."""

    slurm_dir = Path(args.slurm_dir)
    return [slurm_dir / f"{job.slug}.slurm" for job in jobs]


def batch_slurm_script_paths(batches: list[TrainJobBatch], args: argparse.Namespace) -> list[Path]:
    """Return generated batch script paths without writing them."""

    slurm_dir = batch_slurm_dir(args)
    return [slurm_dir / f"{batch.slug}.slurm" for batch in batches]


SBATCH_JOB_ID_RE = re.compile(r"Submitted batch job (?P<job_id>\d+)")


def parse_sbatch_job_id(stdout: str) -> str | None:
    """Parse the job id printed by a successful ``sbatch`` call."""

    match = SBATCH_JOB_ID_RE.search(stdout)
    if match is None:
        return None
    return match.group("job_id")


def submission_log_path(args: argparse.Namespace) -> Path:
    """Return the JSONL ledger path for submission attempts."""

    if args.submit_log:
        return Path(args.submit_log)
    if getattr(args, "batch_from_timing", None):
        return batch_slurm_dir(args) / "submissions.jsonl"
    return Path(args.slurm_dir) / "submissions.jsonl"


def read_submission_records(path: Path) -> list[dict[str, object]]:
    """Read submission ledger records, ignoring blank lines."""

    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"submission record in {path}:{line_number} is not an object")
        records.append(record)
    return records


def submitted_slugs_from_log(path: Path) -> set[str]:
    """Return job slugs marked as successfully submitted in the ledger."""

    return {
        str(record["job_slug"])
        for record in read_submission_records(path)
        if record.get("status") == "submitted" and record.get("job_slug")
    }


def append_submission_record(path: Path, record: dict[str, object]) -> None:
    """Append one JSON submission record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def submission_record(
    job: TrainJob,
    args: argparse.Namespace,
    path: Path,
    *,
    status: str,
    returncode: int,
    stdout: str,
    stderr: str,
    job_id: str | None = None,
) -> dict[str, object]:
    """Build a JSON-serializable sbatch attempt record."""

    record: dict[str, object] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "returncode": returncode,
        "script": str(path),
        "job_slug": job.slug,
        "job_name": slurm_job_name(job, args),
        "stdout": stdout,
        "stderr": stderr,
    }
    if job_id is not None:
        record["job_id"] = job_id
    return record


def batch_submission_record(
    batch: TrainJobBatch,
    args: argparse.Namespace,
    path: Path,
    *,
    status: str,
    returncode: int,
    stdout: str,
    stderr: str,
    job_id: str | None = None,
) -> dict[str, object]:
    """Build a JSON-serializable sbatch attempt record for a batch script."""

    record: dict[str, object] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "returncode": returncode,
        "script": str(path),
        "job_slug": batch.slug,
        "job_name": batch_job_name(batch, args),
        "member_job_slugs": batch.member_job_slugs,
        "n_train_jobs": len(batch.jobs),
        "estimated_raw_seconds": batch.raw_seconds,
        "estimated_padded_seconds": batch.padded_seconds,
        "stdout": stdout,
        "stderr": stderr,
    }
    if job_id is not None:
        record["job_id"] = job_id
    return record


def submit_scripts(jobs: list[TrainJob], paths: list[Path], args: argparse.Namespace) -> None:
    """Submit generated scripts with ``sbatch``."""

    log_path = submission_log_path(args)
    submitted_slugs = submitted_slugs_from_log(log_path) if args.skip_submitted else set()
    for job, path in zip(jobs, paths, strict=True):
        if job.slug in submitted_slugs:
            print(f"Skipping previously submitted job slug: {job.slug}")
            continue

        try:
            result = subprocess.run(
                ["sbatch", str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="", file=sys.stderr)
            append_submission_record(
                log_path,
                submission_record(
                    job,
                    args,
                    path,
                    status="failed",
                    returncode=exc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                ),
            )
            raise

        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        append_submission_record(
            log_path,
            submission_record(
                job,
                args,
                path,
                status="submitted",
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                job_id=parse_sbatch_job_id(result.stdout),
            ),
        )


def submit_batch_scripts(
    batches: list[TrainJobBatch],
    paths: list[Path],
    args: argparse.Namespace,
) -> None:
    """Submit generated sequential batch scripts with ``sbatch``."""

    log_path = submission_log_path(args)
    submitted_slugs = submitted_slugs_from_log(log_path) if args.skip_submitted else set()
    for batch, path in zip(batches, paths, strict=True):
        if batch.slug in submitted_slugs:
            print(f"Skipping previously submitted batch slug: {batch.slug}")
            continue

        try:
            result = subprocess.run(
                ["sbatch", str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="", file=sys.stderr)
            append_submission_record(
                log_path,
                batch_submission_record(
                    batch,
                    args,
                    path,
                    status="failed",
                    returncode=exc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                ),
            )
            raise

        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        append_submission_record(
            log_path,
            batch_submission_record(
                batch,
                args,
                path,
                status="submitted",
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                job_id=parse_sbatch_job_id(result.stdout),
            ),
        )


def first_failed_script_index(paths: list[Path], args: argparse.Namespace) -> int | None:
    """Return the manifest-order index of ``--first-failed-script`` if provided."""

    if not args.first_failed_script:
        return None
    first_failed = Path(args.first_failed_script)
    try:
        return paths.index(first_failed)
    except ValueError as exc:
        raise ValueError(
            f"--first-failed-script {first_failed} is not in the manifest-generated path list"
        ) from exc


def submit_target_range(
    jobs: list[TrainJob],
    paths: list[Path],
    args: argparse.Namespace,
) -> tuple[list[TrainJob], list[Path]]:
    """Restrict submission targets to ``--first-failed-script`` and later if set."""

    first_failed_index = first_failed_script_index(paths, args)
    if first_failed_index is None:
        return jobs, paths
    return jobs[first_failed_index:], paths[first_failed_index:]


def submit_batch_target_range(
    batches: list[TrainJobBatch],
    paths: list[Path],
    args: argparse.Namespace,
) -> tuple[list[TrainJobBatch], list[Path]]:
    """Restrict batch submission targets to ``--first-failed-script`` and later if set."""

    first_failed_index = first_failed_script_index(paths, args)
    if first_failed_index is None:
        return batches, paths
    return batches[first_failed_index:], paths[first_failed_index:]


def print_submission_report(
    jobs: list[TrainJob],
    paths: list[Path],
    args: argparse.Namespace,
) -> None:
    """Print a local report of generated, logged, and known-aborted submissions."""

    first_failed_index = first_failed_script_index(paths, args)
    log_path = submission_log_path(args)
    records = read_submission_records(log_path)
    logged_submitted = {
        str(record["job_slug"]): record
        for record in records
        if record.get("status") == "submitted" and record.get("job_slug")
    }
    logged_failed = {
        str(record["job_slug"]): record
        for record in records
        if record.get("status") == "failed" and record.get("job_slug")
    }

    unsubmitted_from_abort = paths[first_failed_index:] if first_failed_index is not None else []
    summary = {
        "total": len(paths),
        "submit_log": str(log_path),
        "logged_submitted": len(logged_submitted),
        "logged_failed": len(logged_failed),
        "first_failed_index_zero_based": first_failed_index,
        "submitted_before_first_failed": first_failed_index,
        "not_submitted_by_aborted_invocation": len(unsubmitted_from_abort),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if first_failed_index is not None:
        print("not_submitted_by_aborted_invocation:")
        for path in unsubmitted_from_abort:
            print(path)

    if records:
        logged_missing = [
            path for job, path in zip(jobs, paths, strict=True) if job.slug not in logged_submitted
        ]
        print("not_marked_submitted_in_log:")
        for path in logged_missing:
            print(path)


def print_batch_submission_report(
    batches: list[TrainJobBatch],
    paths: list[Path],
    args: argparse.Namespace,
) -> None:
    """Print a local submission report for generated batch scripts."""

    first_failed_index = first_failed_script_index(paths, args)
    log_path = submission_log_path(args)
    records = read_submission_records(log_path)
    logged_submitted = {
        str(record["job_slug"]): record
        for record in records
        if record.get("status") == "submitted" and record.get("job_slug")
    }
    logged_failed = {
        str(record["job_slug"]): record
        for record in records
        if record.get("status") == "failed" and record.get("job_slug")
    }

    unsubmitted_from_abort = paths[first_failed_index:] if first_failed_index is not None else []
    summary = {
        "total_batches": len(paths),
        "total_train_jobs": sum(len(batch.jobs) for batch in batches),
        "submit_log": str(log_path),
        "logged_submitted": len(logged_submitted),
        "logged_failed": len(logged_failed),
        "first_failed_index_zero_based": first_failed_index,
        "submitted_before_first_failed": first_failed_index,
        "not_submitted_by_aborted_invocation": len(unsubmitted_from_abort),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if first_failed_index is not None:
        print("not_submitted_by_aborted_invocation:")
        for path in unsubmitted_from_abort:
            print(path)

    if records:
        logged_missing = [
            path
            for batch, path in zip(batches, paths, strict=True)
            if batch.slug not in logged_submitted
        ]
        print("not_marked_submitted_in_log:")
        for path in logged_missing:
            print(path)


def batch_dry_run_payload(
    jobs: list[TrainJob],
    batches: list[TrainJobBatch],
    args: argparse.Namespace,
) -> dict[str, object]:
    """Return a JSON-serializable batch dry-run report."""

    return {
        "n_jobs": len(jobs),
        "n_batches": len(batches),
        "batch_from_timing": args.batch_from_timing,
        "batch_safety_margin": args.batch_safety_margin,
        "batch_time": args.batch_time,
        "batch_mem": args.batch_mem,
        "batch_slurm_dir": str(batch_slurm_dir(args)),
        "estimated_raw_hours": sum(batch.raw_seconds for batch in batches) / 3600.0,
        "estimated_padded_hours": sum(batch.padded_seconds for batch in batches) / 3600.0,
        "slurm": slurm_defaults(args),
        "batches": [
            {
                "slug": batch.slug,
                "n_train_jobs": len(batch.jobs),
                "estimated_raw_seconds": batch.raw_seconds,
                "estimated_padded_seconds": batch.padded_seconds,
                "member_job_slugs": batch.member_job_slugs,
            }
            for batch in batches
        ],
    }


def format_shell_command(command: list[str]) -> str:
    """Format a shell command as a readable continuation block."""

    quoted = [shlex.quote(part) for part in command]
    if len(quoted) <= 4:
        return " ".join(quoted)
    first, rest = quoted[:3], quoted[3:]
    lines = [" ".join(first) + " \\"]
    for index, part in enumerate(rest):
        suffix = " \\" if index < len(rest) - 1 else ""
        lines.append(f"  {part}{suffix}")
    return "\n".join(lines)


def strip_remainder_separator(values: list[str]) -> list[str]:
    """Drop a leading ``--`` from argparse.REMAINDER values."""

    if values and values[0] == "--":
        return values[1:]
    return values


def slurm_defaults(args: argparse.Namespace) -> dict[str, object]:
    """Return the active SLURM resource settings for auditing."""

    batch_from_timing = getattr(args, "batch_from_timing", None)
    return {
        "account": args.account,
        "partition": args.partition,
        "qos": args.qos,
        "reservation": args.reservation,
        "constraint": args.constraint,
        "exclude": args.exclude,
        "time": args.time,
        "mem": args.mem,
        "gpus_per_node": args.gpus_per_node,
        "cpus_per_gpu": args.cpus_per_gpu,
        "nodes": args.nodes,
        "ntasks_per_node": args.ntasks_per_node,
        "mail_type": args.mail_type,
        "mail_user": args.mail_user,
        "modules": args.module or ["uv/0.9.5", "cuda/12.9.1", "openmpi"],
        "slurm_dir": args.slurm_dir,
        "log_dir": args.log_dir,
        "job_name_prefix": args.job_name_prefix,
        "submit_log": str(submission_log_path(args)),
        "batch_mode": bool(batch_from_timing),
        "batch_from_timing": batch_from_timing,
        "batch_safety_margin": getattr(args, "batch_safety_margin", None),
        "batch_time": getattr(args, "batch_time", None),
        "batch_mem": getattr(args, "batch_mem", None),
        "batch_slurm_dir": str(batch_slurm_dir(args)) if batch_from_timing else None,
    }


def main() -> None:
    args = apply_manifest_defaults(parse_args())
    if args.print_slurm_defaults:
        print(json.dumps(slurm_defaults(args), indent=2, sort_keys=True))
        return
    jobs = build_jobs(args)
    if args.batch_from_timing:
        batches = build_train_job_batches(jobs, args)
        paths = batch_slurm_script_paths(batches, args)
        if args.submission_report:
            print_batch_submission_report(batches, paths, args)
            return
        if args.dry_run:
            print(json.dumps(batch_dry_run_payload(jobs, batches, args), indent=2, sort_keys=True))
            return
        paths = write_batch_slurm_scripts(batches, args)
        for path in paths:
            print(path)
        if args.submit:
            submit_batches, submit_paths = submit_batch_target_range(batches, paths, args)
            if submit_paths != paths:
                print(f"Submitting {len(submit_paths)} batch scripts starting at {submit_paths[0]}")
            submit_batch_scripts(submit_batches, submit_paths, args)
        return

    paths = slurm_script_paths(jobs, args)
    if args.submission_report:
        print_submission_report(jobs, paths, args)
        return
    if args.dry_run:
        print(
            json.dumps(
                {"n_jobs": len(jobs), "slurm": slurm_defaults(args)},
                indent=2,
                sort_keys=True,
            )
        )
        for job in jobs:
            print(job.slug)
        return
    paths = write_slurm_scripts(jobs, args)
    for path in paths:
        print(path)
    if args.submit:
        submit_jobs, submit_paths = submit_target_range(jobs, paths, args)
        if submit_paths != paths:
            print(f"Submitting {len(submit_paths)} scripts starting at {submit_paths[0]}")
        submit_scripts(submit_jobs, submit_paths, args)


if __name__ == "__main__":
    main()
