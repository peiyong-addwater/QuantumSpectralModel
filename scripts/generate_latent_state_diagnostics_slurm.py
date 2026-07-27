#!/usr/bin/env python3
"""Generate batched SLURM scripts for latent-state diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_timing_benchmark import representation_matches_dataset  # noqa: E402

from ham_embed_spectral.experiments.manifest import (  # noqa: E402
    DATASET_CHOICES,
    REPRESENTATION_CHOICES,
    TrainJob,
    jobs_from_manifest,
    load_manifest,
    sanitize_slug,
)
from ham_embed_spectral.naming import ENCODER_CLI_CHOICES, canonical_encoder_name  # noqa: E402
from ham_embed_spectral.utils.checkpointing import write_json  # noqa: E402

DEFAULT_MANIFESTS = (
    "configs/experiments/pendigits.json",
    "configs/experiments/synthetic.json",
)
DEFAULT_MODES = ("final",)
DIAGNOSTIC_MODES = ("init-reference", "checkpoints", "final")
DEFAULT_DEPTHS = (1, 4, 16)
DEFAULT_SEEDS = (0,)
DEFAULT_OUTPUT_ROOT = "results/tables/latent_state_diagnostics/batched"
DEFAULT_SLURM_DIR = "slurm/latent_state_diagnostics_batched"
DEFAULT_MODULES = ("uv/0.9.5", "cuda/12.9.1", "openmpi")


@dataclass(frozen=True)
class LatentDiagnosticItem:
    """One latent-state diagnostics item for one manifest job and mode."""

    manifest_path: Path
    manifest: dict[str, Any]
    train_job: TrainJob
    mode: str

    @property
    def manifest_id(self) -> str:
        return str(self.manifest["manifest_id"])

    @property
    def slug(self) -> str:
        return sanitize_slug(f"{self.manifest_id}__{self.train_job.slug}__{self.mode}")


@dataclass(frozen=True)
class TimingKey:
    """Benchmark timing key excluding seed."""

    manifest_id: str
    dataset: str
    representation: str
    encoder: str
    mode: str
    split: str
    diagnostic_batch_size: int
    spectral_state_max_samples: int | None
    reupload_depth: int

    @property
    def base(self) -> tuple[object, ...]:
        return (
            self.manifest_id,
            self.dataset,
            self.representation,
            self.encoder,
            self.mode,
            self.split,
            self.diagnostic_batch_size,
            self.spectral_state_max_samples,
        )

    @property
    def legacy_spectral_base(self) -> tuple[object, ...]:
        return (
            self.manifest_id,
            self.dataset,
            self.representation,
            self.encoder,
            self.mode,
            self.split,
            self.diagnostic_batch_size,
            None,
        )


@dataclass(frozen=True)
class TimedLatentDiagnosticItem:
    """One diagnostics item with raw and safety-padded timing estimates."""

    item: LatentDiagnosticItem
    raw_seconds: float
    padded_seconds: float


@dataclass(frozen=True)
class LatentDiagnosticBatch:
    """A sequential SLURM batch of latent-state diagnostics items."""

    slug: str
    items: tuple[TimedLatentDiagnosticItem, ...]
    raw_seconds: float
    padded_seconds: float

    @property
    def member_slugs(self) -> list[str]:
        return [timed.item.slug for timed in self.items]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    diagnostics = parser.add_argument_group("diagnostics")
    diagnostics.add_argument("--manifests", nargs="+", default=list(DEFAULT_MANIFESTS))
    diagnostics.add_argument(
        "--modes",
        nargs="+",
        choices=DIAGNOSTIC_MODES,
        default=list(DEFAULT_MODES),
    )
    diagnostics.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    diagnostics.add_argument("--runs-root", default=None)
    diagnostics.add_argument("--experiment-name", default=None)
    diagnostics.add_argument("--encoders", nargs="+", choices=ENCODER_CLI_CHOICES, default=None)
    diagnostics.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, default=None)
    diagnostics.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATION_CHOICES,
        default=None,
    )
    diagnostics.add_argument("--reupload-depths", nargs="+", type=int, default=list(DEFAULT_DEPTHS))
    diagnostics.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    diagnostics.add_argument("--job-slugs", nargs="+", default=None)
    diagnostics.add_argument("--limit", type=int, default=None)
    diagnostics.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
    )
    diagnostics.add_argument("--diagnostic-batch-size", type=int, default=32)
    diagnostics.add_argument("--diagnostic-seed", type=int, default=0)
    diagnostics.add_argument("--spectral-state-max-samples", type=int, default=8)
    diagnostics.add_argument(
        "--write-hdf5",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write one HDF5 artifact path per item and forward it to latent_state_diagnostics.py.",
    )
    diagnostics.add_argument(
        "--store-state-traces",
        action="store_true",
        help="Forward --store-state-traces when HDF5 artifacts are enabled.",
    )

    slurm = parser.add_argument_group("slurm")
    slurm.add_argument("--project-root", default=str(ROOT))
    slurm.add_argument("--venv-dir", default=".venv")
    slurm.add_argument("--slurm-dir", default=DEFAULT_SLURM_DIR)
    slurm.add_argument("--log-dir", default="sbatch_log")
    slurm.add_argument("--job-name-prefix", default="qfm_latent_diag")
    slurm.add_argument("--account", default=None)
    slurm.add_argument("--partition", default=None)
    slurm.add_argument("--qos", default=None)
    slurm.add_argument("--reservation", default=None)
    slurm.add_argument("--constraint", default=None)
    slurm.add_argument("--exclude", default=None)
    slurm.add_argument("--gpus-per-node", type=int, default=1)
    slurm.add_argument("--cpus-per-gpu", type=int, default=16)
    slurm.add_argument("--nodes", type=int, default=1)
    slurm.add_argument("--ntasks-per-node", type=int, default=1)
    slurm.add_argument("--mail-type", default="ALL")
    slurm.add_argument("--mail-user", default=None)
    slurm.add_argument("--module", action="append", default=None)
    slurm.add_argument("--submit", action="store_true")
    slurm.add_argument("--submit-log", default=None)
    slurm.add_argument("--skip-submitted", action="store_true")
    slurm.add_argument("--dry-run", action="store_true")

    batch = parser.add_argument_group("batching")
    batch.add_argument(
        "--batch-from-benchmark",
        default=None,
        help="Combined latent-state benchmark CSV used to estimate and pack diagnostics items.",
    )
    batch.add_argument("--batch-time", default="24:00:00")
    batch.add_argument("--batch-mem", default="32GB")
    batch.add_argument("--batch-safety-margin", type=float, default=2.0)
    return normalize_args(parser.parse_args())


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.modes:
        raise ValueError("--modes must contain at least one mode")
    if not args.reupload_depths:
        raise ValueError("--reupload-depths must contain at least one depth")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    if args.diagnostic_batch_size < 1:
        raise ValueError("--diagnostic-batch-size must be positive")
    if args.spectral_state_max_samples < 0:
        raise ValueError("--spectral-state-max-samples must be nonnegative")
    if args.store_state_traces and not args.write_hdf5:
        raise ValueError("--store-state-traces requires --write-hdf5")
    if args.batch_safety_margin <= 0:
        raise ValueError("--batch-safety-margin must be positive")
    if args.encoders is not None:
        args.encoders = [canonical_encoder_name(value) for value in args.encoders]
    return args


def main() -> None:
    args = parse_args()
    if args.batch_from_benchmark is None:
        raise SystemExit("--batch-from-benchmark is required for batched latent diagnostics")

    items = select_diagnostic_items(args)
    batches = build_diagnostic_batches(items, args)
    script_paths = batch_slurm_script_paths(batches, args)

    if args.dry_run:
        print(
            json.dumps(
                dry_run_payload(items, batches, script_paths, args),
                indent=2,
                sort_keys=True,
            )
        )
        return

    write_item_manifests(items, args)
    for batch, script_path in zip(batches, script_paths, strict=True):
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(render_batch_slurm_script(batch, args))
        print(script_path)

    if args.submit:
        submit_batch_scripts(batches, script_paths, args)


def select_diagnostic_items(args: argparse.Namespace) -> list[LatentDiagnosticItem]:
    """Expand manifests and filters into latent-state diagnostics items."""

    wanted_encoders = set(args.encoders or [])
    wanted_datasets = set(args.datasets or [])
    wanted_representations = set(args.representations or [])
    wanted_depths = {int(value) for value in args.reupload_depths or []}
    wanted_seeds = {int(value) for value in args.seeds or []}
    wanted_job_slugs = set(args.job_slugs or [])
    selected: list[LatentDiagnosticItem] = []
    seen: set[tuple[str, str, str]] = set()

    for manifest_raw in args.manifests:
        manifest_path = Path(manifest_raw)
        manifest = load_manifest(manifest_path)
        manifest_id = str(manifest["manifest_id"])
        for train_job in jobs_from_manifest(manifest, encoders=args.encoders):
            if wanted_datasets and train_job.dataset not in wanted_datasets:
                continue
            if wanted_representations and train_job.representation not in wanted_representations:
                continue
            if not representation_matches_dataset(train_job.dataset, train_job.representation):
                continue
            if wanted_encoders and train_job.encoder not in wanted_encoders:
                continue
            if wanted_depths and train_job.reupload_depth not in wanted_depths:
                continue
            if wanted_seeds and train_job.seed not in wanted_seeds:
                continue
            if wanted_job_slugs and train_job.slug not in wanted_job_slugs:
                continue
            for mode in args.modes:
                key = (manifest_id, train_job.slug, mode)
                if key in seen:
                    continue
                seen.add(key)
                selected.append(
                    LatentDiagnosticItem(
                        manifest_path=manifest_path,
                        manifest=manifest,
                        train_job=train_job,
                        mode=mode,
                    )
                )

    if args.limit is not None:
        return selected[: args.limit]
    return selected


def load_benchmark_timings(path: Path) -> dict[TimingKey, float]:
    """Load complete benchmark wall times keyed by latent diagnostic workload."""

    grouped: dict[TimingKey, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") != "complete":
                continue
            value = row.get("subprocess_wall_time_seconds")
            if value in {None, ""}:
                continue
            mode = row.get("mode") or row.get("diagnostic_mode")
            if mode in {None, ""}:
                continue
            key = TimingKey(
                manifest_id=str(row["manifest_id"]),
                dataset=str(row["dataset"]),
                representation=str(row["representation"]),
                encoder=canonical_encoder_name(str(row["encoder"])),
                mode=str(mode),
                split=str(row.get("split") or "validation"),
                diagnostic_batch_size=int(row.get("diagnostic_batch_size") or 32),
                spectral_state_max_samples=optional_int(row.get("spectral_state_max_samples")),
                reupload_depth=int(row["reupload_depth"]),
            )
            grouped.setdefault(key, []).append(float(value))
    return {key: mean(values) for key, values in grouped.items()}


def optional_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    return int(float(str(value)))


def timing_key(
    item: LatentDiagnosticItem,
    args: argparse.Namespace,
    depth: int | None = None,
    *,
    spectral_state_max_samples: int | None | object = Ellipsis,
) -> TimingKey:
    train_job = item.train_job
    spectral_samples = (
        args.spectral_state_max_samples
        if spectral_state_max_samples is Ellipsis
        else spectral_state_max_samples
    )
    return TimingKey(
        manifest_id=item.manifest_id,
        dataset=train_job.dataset,
        representation=train_job.representation,
        encoder=train_job.encoder,
        mode=item.mode,
        split=args.split,
        diagnostic_batch_size=int(args.diagnostic_batch_size),
        spectral_state_max_samples=spectral_samples,
        reupload_depth=train_job.reupload_depth if depth is None else int(depth),
    )


def estimate_item_seconds(
    item: LatentDiagnosticItem,
    timings: dict[TimingKey, float],
    args: argparse.Namespace,
) -> float:
    """Estimate diagnostics wall time, interpolating unbenchmarked depths linearly."""

    exact_key = timing_key(item, args)
    if exact_key in timings:
        return timings[exact_key]

    legacy_key = timing_key(item, args, spectral_state_max_samples=None)
    if legacy_key in timings:
        return timings[legacy_key]

    available: dict[tuple[object, ...], list[int]] = {}
    for base in (exact_key.base, exact_key.legacy_spectral_base):
        measured = sorted(
            (key.reupload_depth, value)
            for key, value in timings.items()
            if key.base == base
        )
        available[base] = [measured_depth for measured_depth, _ in measured]
        estimate = interpolate_depth_seconds(measured, exact_key.reupload_depth)
        if estimate is not None:
            return estimate

    raise ValueError(
        f"no bracketing benchmark timing for {item.slug} at depth "
        f"{exact_key.reupload_depth}; available depths by base: {available}"
    )


def interpolate_depth_seconds(
    measured: list[tuple[int, float]],
    depth: int,
) -> float | None:
    if not measured:
        return None
    exact = [value for measured_depth, value in measured if measured_depth == depth]
    if exact:
        return exact[0]
    lower = [
        (measured_depth, value)
        for measured_depth, value in measured
        if measured_depth < depth
    ]
    upper = [
        (measured_depth, value)
        for measured_depth, value in measured
        if measured_depth > depth
    ]
    if not lower or not upper:
        return None
    lower_depth, lower_seconds = lower[-1]
    upper_depth, upper_seconds = upper[0]
    slope = (upper_seconds - lower_seconds) / float(upper_depth - lower_depth)
    return lower_seconds + slope * float(depth - lower_depth)


def build_diagnostic_batches(
    items: list[LatentDiagnosticItem],
    args: argparse.Namespace,
) -> list[LatentDiagnosticBatch]:
    """Build first-fit-decreasing sequential batches from benchmark timings."""

    capacity_seconds = parse_slurm_time_seconds(args.batch_time)
    timings = load_benchmark_timings(Path(args.batch_from_benchmark))
    timed_items = [
        TimedLatentDiagnosticItem(
            item=item,
            raw_seconds=raw_seconds,
            padded_seconds=raw_seconds * args.batch_safety_margin,
        )
        for item in items
        for raw_seconds in [estimate_item_seconds(item, timings, args)]
    ]
    return pack_timed_diagnostic_items(timed_items, capacity_seconds)


def pack_timed_diagnostic_items(
    timed_items: list[TimedLatentDiagnosticItem],
    capacity_seconds: float,
) -> list[LatentDiagnosticBatch]:
    """Pack diagnostics items using first-fit decreasing by padded wall time."""

    bins: list[list[TimedLatentDiagnosticItem]] = []
    used_seconds: list[float] = []
    ordered_items = sorted(
        timed_items,
        key=lambda timed: (-timed.padded_seconds, timed.item.slug),
    )
    for timed in ordered_items:
        if timed.padded_seconds > capacity_seconds:
            raise ValueError(
                f"{timed.item.slug} padded estimate {timed.padded_seconds:.1f}s exceeds "
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
            LatentDiagnosticBatch(
                slug=f"batch_{index:04d}",
                items=tuple(members),
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


def output_root(args: argparse.Namespace) -> Path:
    return Path(args.output_root)


def single_manifest_path(args: argparse.Namespace, item: LatentDiagnosticItem) -> Path:
    return output_root(args) / "manifests" / f"{item.slug}.json"


def item_output_path(args: argparse.Namespace, item: LatentDiagnosticItem) -> Path:
    return output_root(args) / "items" / f"{item.slug}.json"


def item_hdf5_path(args: argparse.Namespace, item: LatentDiagnosticItem) -> Path:
    return output_root(args) / "items" / f"{item.slug}.h5"


def single_job_manifest_payload(item: LatentDiagnosticItem) -> dict[str, Any]:
    """Return a manifest payload containing exactly one TrainJob."""

    train_job = item.train_job
    payload = {key: value for key, value in item.manifest.items() if not key.startswith("_")}
    payload["manifest_id"] = item.manifest_id
    payload["grid"] = {
        "datasets": [train_job.dataset],
        "representations": [train_job.representation],
        "encoders": [train_job.encoder],
        "reupload_depths": [train_job.reupload_depth],
        "seeds": [train_job.seed],
        "learning_rates": [train_job.learning_rate],
        "batch_sizes": [train_job.batch_size],
        "class_subsets": [train_job.class_subset],
    }
    return payload


def write_item_manifests(items: list[LatentDiagnosticItem], args: argparse.Namespace) -> None:
    for item in items:
        write_json(single_manifest_path(args, item), single_job_manifest_payload(item))


def batch_slurm_script_paths(
    batches: list[LatentDiagnosticBatch],
    args: argparse.Namespace,
) -> list[Path]:
    return [Path(args.slurm_dir) / f"{batch.slug}.slurm" for batch in batches]


def render_batch_slurm_script(batch: LatentDiagnosticBatch, args: argparse.Namespace) -> str:
    modules = args.module or list(DEFAULT_MODULES)
    output_pattern = f"{args.log_dir}/%x-%j.out"
    sbatch_lines = "\n".join(
        sbatch_directives(args, batch_job_name(batch, args), output_pattern)
    )
    module_lines = "\n".join(f"module load {shlex.quote(module)}" for module in modules)
    commands = "\n\n".join(
        format_shell_command(diagnostic_command(timed.item, args)) for timed in batch.items
    )
    return f"""#!/bin/bash
{sbatch_lines}

set -euo pipefail

PROJECT_ROOT={shlex.quote(str(args.project_root))}
VENV_DIR={shlex.quote(args.venv_dir)}

cd "$PROJECT_ROOT"
mkdir -p {shlex.quote(args.log_dir)}
{module_lines}

source "$PROJECT_ROOT/$VENV_DIR/bin/activate"
export JAX_PLATFORMS=cuda

{commands}
"""


def sbatch_directives(
    args: argparse.Namespace,
    job_name: str,
    output_pattern: str,
) -> list[str]:
    directives = [
        ("job-name", job_name),
        ("time", args.batch_time),
        ("mem", args.batch_mem),
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


def batch_job_name(batch: LatentDiagnosticBatch, args: argparse.Namespace) -> str:
    return sanitize_slug(f"{args.job_name_prefix}__{batch.slug}")[:128]


def diagnostic_command(item: LatentDiagnosticItem, args: argparse.Namespace) -> list[str]:
    command = [
        "uv",
        "run",
        "scripts/latent_state_diagnostics.py",
        "--manifest",
        str(single_manifest_path(args, item)),
        "--mode",
        item.mode,
        "--split",
        args.split,
        "--diagnostic-batch-size",
        str(args.diagnostic_batch_size),
        "--diagnostic-seed",
        str(args.diagnostic_seed),
        "--spectral-state-max-samples",
        str(args.spectral_state_max_samples),
        "--max-jobs",
        "1",
        "--output",
        str(item_output_path(args, item)),
    ]
    if args.runs_root is not None:
        command.extend(["--runs-root", args.runs_root])
    if args.experiment_name is not None:
        command.extend(["--experiment-name", args.experiment_name])
    if args.write_hdf5:
        command.extend(["--hdf5-output", str(item_hdf5_path(args, item))])
    if args.store_state_traces:
        command.append("--store-state-traces")
    return command


def format_shell_command(command: list[str]) -> str:
    quoted = [shlex.quote(part) for part in command]
    if len(quoted) <= 4:
        return " ".join(quoted)
    first, rest = quoted[:3], quoted[3:]
    lines = [" ".join(first) + " \\"]
    for index, part in enumerate(rest):
        suffix = " \\" if index < len(rest) - 1 else ""
        lines.append(f"  {part}{suffix}")
    return "\n".join(lines)


def dry_run_payload(
    items: list[LatentDiagnosticItem],
    batches: list[LatentDiagnosticBatch],
    script_paths: list[Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "n_items": len(items),
        "n_batches": len(batches),
        "batch_from_benchmark": args.batch_from_benchmark,
        "output_root": args.output_root,
        "runs_root_override": args.runs_root,
        "experiment_name_override": args.experiment_name,
        "write_hdf5": args.write_hdf5,
        "store_state_traces": args.store_state_traces,
        "slurm": slurm_defaults(args),
        "estimated_raw_hours": sum(batch.raw_seconds for batch in batches) / 3600.0,
        "estimated_padded_hours": sum(batch.padded_seconds for batch in batches) / 3600.0,
        "items_preview": [item_record(item, args) for item in items[:20]],
        "batches": [
            {
                "batch_slug": batch.slug,
                "script_path": str(path),
                "n_items": len(batch.items),
                "estimated_raw_seconds": batch.raw_seconds,
                "estimated_padded_seconds": batch.padded_seconds,
                "member_slugs": batch.member_slugs,
            }
            for batch, path in zip(batches, script_paths, strict=True)
        ],
    }


def item_record(item: LatentDiagnosticItem, args: argparse.Namespace) -> dict[str, Any]:
    train_job = item.train_job
    return {
        "slug": item.slug,
        "manifest_id": item.manifest_id,
        "manifest_path": str(item.manifest_path),
        "mode": item.mode,
        "job_slug": train_job.slug,
        "dataset": train_job.dataset,
        "representation": train_job.representation,
        "encoder": train_job.encoder,
        "reupload_depth": train_job.reupload_depth,
        "seed": train_job.seed,
        "single_manifest_path": str(single_manifest_path(args, item)),
        "output_path": str(item_output_path(args, item)),
        "hdf5_path": str(item_hdf5_path(args, item)) if args.write_hdf5 else None,
        "command": diagnostic_command(item, args),
    }


def slurm_defaults(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "account": args.account,
        "partition": args.partition,
        "qos": args.qos,
        "reservation": args.reservation,
        "constraint": args.constraint,
        "exclude": args.exclude,
        "time": args.batch_time,
        "mem": args.batch_mem,
        "gpus_per_node": args.gpus_per_node,
        "cpus_per_gpu": args.cpus_per_gpu,
        "nodes": args.nodes,
        "ntasks_per_node": args.ntasks_per_node,
        "mail_type": args.mail_type,
        "mail_user": args.mail_user,
        "modules": args.module or list(DEFAULT_MODULES),
        "slurm_dir": args.slurm_dir,
        "log_dir": args.log_dir,
        "job_name_prefix": args.job_name_prefix,
        "submit_log": str(submission_log_path(args)),
    }


def submission_log_path(args: argparse.Namespace) -> Path:
    if args.submit_log:
        return Path(args.submit_log)
    return Path(args.slurm_dir) / "submissions.jsonl"


def read_submission_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"submission record in {path}:{line_number} is not an object")
        records.append(payload)
    return records


def submitted_batch_slugs_from_log(path: Path) -> set[str]:
    return {
        str(record["batch_slug"])
        for record in read_submission_records(path)
        if record.get("status") == "submitted" and "batch_slug" in record
    }


def append_submission_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def submit_batch_scripts(
    batches: list[LatentDiagnosticBatch],
    script_paths: list[Path],
    args: argparse.Namespace,
) -> None:
    log_path = submission_log_path(args)
    submitted_slugs = submitted_batch_slugs_from_log(log_path) if args.skip_submitted else set()
    for batch, path in zip(batches, script_paths, strict=True):
        if batch.slug in submitted_slugs:
            print(f"skip-submitted {batch.slug}")
            continue
        result = subprocess.run(["sbatch", str(path)], check=False, capture_output=True, text=True)
        status = "submitted" if result.returncode == 0 else "failed"
        record = {
            "status": status,
            "batch_slug": batch.slug,
            "script_path": str(path),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "job_id": parse_sbatch_job_id(result.stdout),
            "n_items": len(batch.items),
            "member_slugs": batch.member_slugs,
            "submitted_at_utc": datetime.now(UTC).isoformat(),
        }
        append_submission_record(log_path, record)
        print(f"{status} {batch.slug} {record['job_id'] or ''}".rstrip())
        if result.returncode != 0:
            raise SystemExit(result.returncode)


def parse_sbatch_job_id(stdout: str) -> str | None:
    words = stdout.strip().split()
    if len(words) >= 4 and words[:3] == ["Submitted", "batch", "job"]:
        return words[3]
    return None


if __name__ == "__main__":
    main()
