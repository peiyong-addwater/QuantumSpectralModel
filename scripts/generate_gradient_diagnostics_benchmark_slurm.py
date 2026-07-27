#!/usr/bin/env python3
"""Generate and optionally submit SLURM scripts for gradient diagnostic benchmarks."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_gradient_diagnostics_resource_benchmark import (  # noqa: E402
    DEFAULT_DEPTHS,
    DEFAULT_MANIFESTS,
    DEFAULT_MODES,
    DEFAULT_OUTPUT_DIR,
    DIAGNOSTIC_MODES,
    GradientDiagnosticBenchmarkJob,
    select_benchmark_jobs,
)

from ham_embed_spectral.experiments.manifest import (  # noqa: E402
    DATASET_CHOICES,
    REPRESENTATION_CHOICES,
    sanitize_slug,
)
from ham_embed_spectral.naming import ENCODER_CLI_CHOICES  # noqa: E402

DEFAULT_MODULES = ("uv/0.9.5", "cuda/12.9.1", "openmpi")


@dataclass(frozen=True)
class TimedBenchmarkJob:
    """One gradient diagnostics benchmark job with wall-time estimates."""

    job: GradientDiagnosticBenchmarkJob
    raw_seconds: float
    padded_seconds: float


@dataclass(frozen=True)
class BenchmarkBatch:
    """A sequential SLURM batch of gradient diagnostics benchmark jobs."""

    slug: str
    jobs: tuple[TimedBenchmarkJob, ...]
    raw_seconds: float
    padded_seconds: float

    @property
    def member_slugs(self) -> list[str]:
        return [timed.job.slug for timed in self.jobs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    benchmark = parser.add_argument_group("benchmark")
    benchmark.add_argument("--manifests", nargs="+", default=list(DEFAULT_MANIFESTS))
    benchmark.add_argument(
        "--modes",
        nargs="+",
        choices=DIAGNOSTIC_MODES,
        default=list(DEFAULT_MODES),
    )
    benchmark.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    benchmark.add_argument("--name", default="gradient_diagnostics_resource")
    benchmark.add_argument("--runs-root", default="results/runs")
    benchmark.add_argument("--experiment-name", default=None)
    benchmark.add_argument("--encoders", nargs="+", choices=ENCODER_CLI_CHOICES, default=None)
    benchmark.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, default=None)
    benchmark.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATION_CHOICES,
        default=None,
    )
    benchmark.add_argument("--reupload-depths", nargs="+", type=int, default=list(DEFAULT_DEPTHS))
    benchmark.add_argument("--seeds", nargs="+", type=int, default=[0])
    benchmark.add_argument("--job-slugs", nargs="+", default=None)
    benchmark.add_argument("--limit", type=int, default=None)
    benchmark.add_argument("--diagnostic-batch-size", type=int, default=32)
    benchmark.add_argument("--diagnostic-seed", type=int, default=0)
    benchmark.add_argument("--near-zero-tol", type=float, default=1e-10)
    benchmark.add_argument("--n-init-seeds", type=int, default=20)
    benchmark.add_argument("--fisher-batch-size", type=int, default=32)
    benchmark.add_argument("--sample-interval", type=float, default=2.0)
    benchmark.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    benchmark.add_argument("--skip-jax-preflight", action="store_true")

    slurm = parser.add_argument_group("slurm")
    slurm.add_argument("--project-root", default=str(ROOT))
    slurm.add_argument("--venv-dir", default=".venv")
    slurm.add_argument("--slurm-dir", default="slurm/gradient_diagnostics_benchmark")
    slurm.add_argument("--log-dir", default="sbatch_log")
    slurm.add_argument("--job-name-prefix", default="qfm_grad_diag_bench")
    slurm.add_argument("--account", default=None)
    slurm.add_argument("--partition", default=None)
    slurm.add_argument("--qos", default=None)
    slurm.add_argument("--reservation", default=None)
    slurm.add_argument("--constraint", default=None)
    slurm.add_argument("--exclude", default=None)
    slurm.add_argument("--time", default="24:00:00")
    slurm.add_argument("--mem", default="32GB")
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
        help="Combined gradient diagnostics benchmark CSV used to pack jobs into batches.",
    )
    batch.add_argument("--batch-time", default="24:00:00")
    batch.add_argument("--batch-mem", default="16GB")
    batch.add_argument("--batch-safety-margin", type=float, default=2.0)
    batch.add_argument(
        "--batch-slurm-dir",
        default=None,
        help="Directory for generated batch scripts. Defaults to '<slurm-dir>_batched'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = select_benchmark_jobs(args)

    if args.batch_from_benchmark:
        batches = build_benchmark_batches(jobs, args)
        script_paths = batch_slurm_script_paths(batches, args)
        if args.dry_run:
            print(
                json.dumps(
                    batch_dry_run_payload(jobs, batches, script_paths, args),
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        for batch, script_path in zip(batches, script_paths, strict=True):
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(render_batch_slurm_script(batch, args))
            print(script_path)
        if args.submit:
            submit_batch_scripts(batches, script_paths, args)
        return

    script_paths = slurm_script_paths(jobs, args)

    if args.dry_run:
        print(json.dumps(dry_run_payload(jobs, script_paths, args), indent=2, sort_keys=True))
        return

    for job, script_path in zip(jobs, script_paths, strict=True):
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(render_slurm_script(job, args))
        print(script_path)

    if args.submit:
        submit_scripts(jobs, script_paths, args)


def slurm_script_paths(
    jobs: list[GradientDiagnosticBenchmarkJob],
    args: argparse.Namespace,
) -> list[Path]:
    return [Path(args.slurm_dir) / f"{job.slug}.slurm" for job in jobs]


def render_slurm_script(job: GradientDiagnosticBenchmarkJob, args: argparse.Namespace) -> str:
    modules = args.module or list(DEFAULT_MODULES)
    job_name = slurm_job_name(job, args)
    output_pattern = f"{args.log_dir}/%x-%j.out"
    sbatch_lines = "\n".join(sbatch_directives(args, job_name, output_pattern))
    module_lines = "\n".join(f"module load {shlex.quote(module)}" for module in modules)
    command = format_shell_command(benchmark_command(job, args))
    return f"""#!/bin/bash
{sbatch_lines}

set -euo pipefail

PROJECT_ROOT={shlex.quote(str(args.project_root))}
VENV_DIR={shlex.quote(args.venv_dir)}

cd "$PROJECT_ROOT"
mkdir -p {shlex.quote(args.log_dir)}
{module_lines}

source "$PROJECT_ROOT/$VENV_DIR/bin/activate"
export JAX_PLATFORMS="${{JAX_PLATFORMS:-cuda}}"

{command}
"""


def sbatch_directives(
    args: argparse.Namespace,
    job_name: str,
    output_pattern: str,
    *,
    time: str | None = None,
    mem: str | None = None,
) -> list[str]:
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


def slurm_job_name(job: GradientDiagnosticBenchmarkJob, args: argparse.Namespace) -> str:
    return sanitize_slug(f"{args.job_name_prefix}__{job.slug}")[:128]


def batch_job_name(batch: BenchmarkBatch, args: argparse.Namespace) -> str:
    return sanitize_slug(f"{args.job_name_prefix}__{batch.slug}")[:128]


def benchmark_command(job: GradientDiagnosticBenchmarkJob, args: argparse.Namespace) -> list[str]:
    train_job = job.train_job
    command = [
        "uv",
        "run",
        "scripts/run_gradient_diagnostics_resource_benchmark.py",
        "--manifests",
        str(job.manifest_path),
        "--modes",
        job.mode,
        "--output-dir",
        args.output_dir,
        "--name",
        job.slug,
        "--runs-root",
        args.runs_root,
        "--datasets",
        train_job.dataset,
        "--representations",
        train_job.representation,
        "--encoders",
        train_job.encoder,
        "--reupload-depths",
        str(train_job.reupload_depth),
        "--seeds",
        str(train_job.seed),
        "--job-slugs",
        train_job.slug,
        "--limit",
        "1",
        "--diagnostic-batch-size",
        str(args.diagnostic_batch_size),
        "--diagnostic-seed",
        str(args.diagnostic_seed),
        "--near-zero-tol",
        f"{args.near_zero_tol:g}",
        "--n-init-seeds",
        str(args.n_init_seeds),
        "--fisher-batch-size",
        str(args.fisher_batch_size),
        "--sample-interval",
        f"{args.sample_interval:g}",
    ]
    if args.experiment_name is not None:
        command.extend(["--experiment-name", args.experiment_name])
    command.append("--require-gpu" if args.require_gpu else "--no-require-gpu")
    if args.skip_jax_preflight:
        command.append("--skip-jax-preflight")
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


def load_benchmark_estimates(path: Path) -> dict[str, float]:
    """Load mean wall-time estimates keyed by benchmark slug."""

    grouped: dict[str, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") != "complete":
                continue
            slug = row.get("benchmark_slug")
            value = row.get("subprocess_wall_time_seconds")
            if not slug or value in {None, ""}:
                continue
            grouped.setdefault(slug, []).append(float(value))
    return {slug: mean(values) for slug, values in grouped.items()}


def estimate_benchmark_seconds(
    job: GradientDiagnosticBenchmarkJob,
    estimates: dict[str, float],
) -> float:
    try:
        return estimates[job.slug]
    except KeyError as exc:
        raise ValueError(f"no benchmark timing estimate found for {job.slug}") from exc


def build_benchmark_batches(
    jobs: list[GradientDiagnosticBenchmarkJob],
    args: argparse.Namespace,
) -> list[BenchmarkBatch]:
    """Build first-fit-decreasing sequential batches from benchmark timings."""

    if args.batch_safety_margin <= 0:
        raise ValueError("--batch-safety-margin must be positive")
    capacity_seconds = parse_slurm_time_seconds(args.batch_time)
    estimates = load_benchmark_estimates(Path(args.batch_from_benchmark))
    timed_jobs = [
        TimedBenchmarkJob(
            job=job,
            raw_seconds=raw_seconds,
            padded_seconds=raw_seconds * args.batch_safety_margin,
        )
        for job in jobs
        for raw_seconds in [estimate_benchmark_seconds(job, estimates)]
    ]
    return pack_timed_benchmark_jobs(timed_jobs, capacity_seconds)


def pack_timed_benchmark_jobs(
    timed_jobs: list[TimedBenchmarkJob],
    capacity_seconds: float,
) -> list[BenchmarkBatch]:
    """Pack benchmark jobs using first-fit decreasing by padded wall time."""

    bins: list[list[TimedBenchmarkJob]] = []
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
            BenchmarkBatch(
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
    if args.batch_slurm_dir:
        return Path(args.batch_slurm_dir)
    return Path(f"{args.slurm_dir}_batched")


def batch_slurm_script_paths(
    batches: list[BenchmarkBatch],
    args: argparse.Namespace,
) -> list[Path]:
    return [batch_slurm_dir(args) / f"{batch.slug}.slurm" for batch in batches]


def render_batch_slurm_script(batch: BenchmarkBatch, args: argparse.Namespace) -> str:
    modules = args.module or list(DEFAULT_MODULES)
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
    commands = "\n\n".join(
        format_shell_command(benchmark_command(timed.job, args)) for timed in batch.jobs
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
export JAX_PLATFORMS="${{JAX_PLATFORMS:-cuda}}"

{commands}
"""


def dry_run_payload(
    jobs: list[GradientDiagnosticBenchmarkJob],
    script_paths: list[Path],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "n_scripts": len(jobs),
        "slurm": slurm_defaults(args),
        "benchmarks": [
            {
                "benchmark_slug": job.slug,
                "script_path": str(path),
                "command": benchmark_command(job, args),
            }
            for job, path in zip(jobs, script_paths, strict=True)
        ],
    }


def batch_dry_run_payload(
    jobs: list[GradientDiagnosticBenchmarkJob],
    batches: list[BenchmarkBatch],
    script_paths: list[Path],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "n_benchmark_jobs": len(jobs),
        "n_batches": len(batches),
        "batch_from_benchmark": args.batch_from_benchmark,
        "batch_time": args.batch_time,
        "batch_mem": args.batch_mem,
        "batch_safety_margin": args.batch_safety_margin,
        "batch_slurm_dir": str(batch_slurm_dir(args)),
        "estimated_raw_hours": sum(batch.raw_seconds for batch in batches) / 3600.0,
        "estimated_padded_hours": sum(batch.padded_seconds for batch in batches) / 3600.0,
        "slurm": slurm_defaults(args),
        "batches": [
            {
                "batch_slug": batch.slug,
                "script_path": str(path),
                "n_benchmark_jobs": len(batch.jobs),
                "estimated_raw_seconds": batch.raw_seconds,
                "estimated_padded_seconds": batch.padded_seconds,
                "member_benchmark_slugs": batch.member_slugs,
            }
            for batch, path in zip(batches, script_paths, strict=True)
        ],
    }


def slurm_defaults(args: argparse.Namespace) -> dict[str, object]:
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
        "modules": args.module or list(DEFAULT_MODULES),
        "slurm_dir": args.slurm_dir,
        "log_dir": args.log_dir,
        "job_name_prefix": args.job_name_prefix,
        "submit_log": str(submission_log_path(args)),
        "batch_mode": bool(args.batch_from_benchmark),
        "batch_from_benchmark": args.batch_from_benchmark,
        "batch_time": args.batch_time,
        "batch_mem": args.batch_mem,
        "batch_safety_margin": args.batch_safety_margin,
        "batch_slurm_dir": str(batch_slurm_dir(args)) if args.batch_from_benchmark else None,
    }


def submission_log_path(args: argparse.Namespace) -> Path:
    if args.submit_log:
        return Path(args.submit_log)
    if args.batch_from_benchmark:
        return batch_slurm_dir(args) / "submissions.jsonl"
    return Path(args.slurm_dir) / "submissions.jsonl"


def read_submission_records(path: Path) -> list[dict[str, object]]:
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


def submitted_slugs_from_log(path: Path) -> set[str]:
    return {
        str(record["benchmark_slug"])
        for record in read_submission_records(path)
        if record.get("status") == "submitted" and "benchmark_slug" in record
    }


def append_submission_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def submit_scripts(
    jobs: list[GradientDiagnosticBenchmarkJob],
    script_paths: list[Path],
    args: argparse.Namespace,
) -> None:
    log_path = submission_log_path(args)
    submitted_slugs = submitted_slugs_from_log(log_path) if args.skip_submitted else set()
    for job, path in zip(jobs, script_paths, strict=True):
        if job.slug in submitted_slugs:
            print(f"skip-submitted {job.slug}")
            continue
        result = subprocess.run(["sbatch", str(path)], check=False, capture_output=True, text=True)
        status = "submitted" if result.returncode == 0 else "failed"
        record = {
            "status": status,
            "benchmark_slug": job.slug,
            "script_path": str(path),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "job_id": parse_sbatch_job_id(result.stdout),
            "submitted_at_utc": datetime.now(UTC).isoformat(),
        }
        append_submission_record(log_path, record)
        print(f"{status} {job.slug} {record['job_id'] or ''}".rstrip())
        if result.returncode != 0:
            raise SystemExit(result.returncode)


def submit_batch_scripts(
    batches: list[BenchmarkBatch],
    script_paths: list[Path],
    args: argparse.Namespace,
) -> None:
    log_path = submission_log_path(args)
    submitted_slugs = submitted_slugs_from_log(log_path) if args.skip_submitted else set()
    for batch, path in zip(batches, script_paths, strict=True):
        if batch.slug in submitted_slugs:
            print(f"skip-submitted {batch.slug}")
            continue
        result = subprocess.run(["sbatch", str(path)], check=False, capture_output=True, text=True)
        status = "submitted" if result.returncode == 0 else "failed"
        record = {
            "status": status,
            "benchmark_slug": batch.slug,
            "script_path": str(path),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "job_id": parse_sbatch_job_id(result.stdout),
            "n_benchmark_jobs": len(batch.jobs),
            "member_benchmark_slugs": batch.member_slugs,
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
