#!/usr/bin/env python3
"""Generate and optionally submit SLURM scripts for latent-state benchmarks."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_latent_state_diagnostics_resource_benchmark import (  # noqa: E402
    BenchmarkJob,
    select_benchmark_jobs,
)

from ham_embed_spectral.experiments.manifest import (  # noqa: E402
    DATASET_CHOICES,
    REPRESENTATION_CHOICES,
    sanitize_slug,
)
from ham_embed_spectral.naming import ENCODER_CLI_CHOICES, canonical_encoder_name  # noqa: E402

DEFAULT_MANIFESTS = (
    "configs/experiments/pendigits.json",
    "configs/experiments/synthetic.json",
)
DEFAULT_DEPTHS = (1, 8, 32)
DEFAULT_SEEDS = (0,)
DEFAULT_OUTPUT_DIR = "results/tables/latent_state_diagnostics/resource_benchmarks"
DEFAULT_MODULES = ("uv/0.9.5", "cuda/12.9.1", "openmpi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    benchmark = parser.add_argument_group("benchmark")
    benchmark.add_argument("--manifests", nargs="+", default=list(DEFAULT_MANIFESTS))
    benchmark.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    benchmark.add_argument("--name", default="latent_state_resource")
    benchmark.add_argument(
        "--mode",
        choices=("init-reference", "checkpoints", "final"),
        default="final",
    )
    benchmark.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    benchmark.add_argument("--diagnostic-batch-size", type=int, default=32)
    benchmark.add_argument("--spectral-state-max-samples", type=int, default=8)
    benchmark.add_argument("--runs-root", default=None)
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
    benchmark.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    benchmark.add_argument("--limit", type=int, default=None)
    benchmark.add_argument("--work-root", default=None)
    benchmark.add_argument("--keep-temp-outputs", action="store_true")
    benchmark.add_argument("--fail-fast", action="store_true")

    slurm = parser.add_argument_group("slurm")
    slurm.add_argument("--project-root", default=str(ROOT))
    slurm.add_argument("--venv-dir", default=".venv")
    slurm.add_argument("--slurm-dir", default="slurm/latent_state_diagnostics_benchmark")
    slurm.add_argument("--log-dir", default="sbatch_log")
    slurm.add_argument("--job-name-prefix", default="qfm_latent_bench")
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
    return normalize_args(parser.parse_args())


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.encoders is not None:
        args.encoders = [canonical_encoder_name(value) for value in args.encoders]
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    if args.diagnostic_batch_size < 1:
        raise ValueError("--diagnostic-batch-size must be positive")
    if args.spectral_state_max_samples < 0:
        raise ValueError("--spectral-state-max-samples must be nonnegative")
    if not args.reupload_depths:
        raise ValueError("--reupload-depths must contain at least one depth")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed")
    return args


def main() -> None:
    args = parse_args()
    jobs = select_benchmark_jobs(args)
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
    jobs: list[BenchmarkJob],
    args: argparse.Namespace,
) -> list[Path]:
    return [Path(args.slurm_dir) / f"{job_script_slug(job, args)}.slurm" for job in jobs]


def job_script_slug(job: BenchmarkJob, args: argparse.Namespace) -> str:
    return sanitize_slug(f"{job.manifest_id}__{job.slug}__{args.mode}")


def render_slurm_script(job: BenchmarkJob, args: argparse.Namespace) -> str:
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
export JAX_PLATFORMS=cuda

{command}
"""


def sbatch_directives(
    args: argparse.Namespace,
    job_name: str,
    output_pattern: str,
) -> list[str]:
    directives = [
        ("job-name", job_name),
        ("time", args.time),
        ("mem", args.mem),
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


def slurm_job_name(job: BenchmarkJob, args: argparse.Namespace) -> str:
    return sanitize_slug(f"{args.job_name_prefix}__{job_script_slug(job, args)}")[:128]


def benchmark_command(job: BenchmarkJob, args: argparse.Namespace) -> list[str]:
    train_job = job.train_job
    command = [
        "uv",
        "run",
        "python",
        "scripts/run_latent_state_diagnostics_resource_benchmark.py",
        "--manifests",
        str(job.manifest_path),
        "--mode",
        args.mode,
        "--split",
        args.split,
        "--output-dir",
        args.output_dir,
        "--name",
        job_script_slug(job, args),
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
        "--limit",
        "1",
        "--diagnostic-batch-size",
        str(args.diagnostic_batch_size),
        "--spectral-state-max-samples",
        str(args.spectral_state_max_samples),
    ]
    if args.runs_root is not None:
        command.extend(["--runs-root", args.runs_root])
    if args.experiment_name is not None:
        command.extend(["--experiment-name", args.experiment_name])
    if args.work_root is not None:
        command.extend(["--work-root", args.work_root])
    if args.keep_temp_outputs:
        command.append("--keep-temp-outputs")
    if args.fail_fast:
        command.append("--fail-fast")
    return command


def format_shell_command(command: list[str]) -> str:
    quoted = [shlex.quote(part) for part in command]
    if len(quoted) <= 4:
        return " ".join(quoted)
    first, rest = quoted[:4], quoted[4:]
    lines = [" ".join(first) + " \\"]
    for index, part in enumerate(rest):
        suffix = " \\" if index < len(rest) - 1 else ""
        lines.append(f"  {part}{suffix}")
    return "\n".join(lines)


def dry_run_payload(
    jobs: list[BenchmarkJob],
    script_paths: list[Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "n_scripts": len(jobs),
        "slurm": slurm_defaults(args),
        "benchmarks": [
            {
                "benchmark_slug": job_script_slug(job, args),
                "script_path": str(path),
                "command": benchmark_command(job, args),
            }
            for job, path in zip(jobs, script_paths, strict=True)
        ],
    }


def slurm_defaults(args: argparse.Namespace) -> dict[str, Any]:
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


def submitted_slugs_from_log(path: Path) -> set[str]:
    return {
        str(record["benchmark_slug"])
        for record in read_submission_records(path)
        if record.get("status") == "submitted" and "benchmark_slug" in record
    }


def append_submission_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def submit_scripts(
    jobs: list[BenchmarkJob],
    script_paths: list[Path],
    args: argparse.Namespace,
) -> None:
    log_path = submission_log_path(args)
    submitted_slugs = submitted_slugs_from_log(log_path) if args.skip_submitted else set()
    for job, path in zip(jobs, script_paths, strict=True):
        slug = job_script_slug(job, args)
        if slug in submitted_slugs:
            print(f"skip-submitted {slug}")
            continue
        result = subprocess.run(["sbatch", str(path)], check=False, capture_output=True, text=True)
        status = "submitted" if result.returncode == 0 else "failed"
        record = {
            "status": status,
            "benchmark_slug": slug,
            "script_path": str(path),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "job_id": parse_sbatch_job_id(result.stdout),
            "submitted_at_utc": datetime.now(UTC).isoformat(),
        }
        append_submission_record(log_path, record)
        print(f"{status} {slug} {record['job_id'] or ''}".rstrip())
        if result.returncode != 0:
            raise SystemExit(result.returncode)


def parse_sbatch_job_id(stdout: str) -> str | None:
    words = stdout.strip().split()
    if len(words) >= 4 and words[:3] == ["Submitted", "batch", "job"]:
        return words[3]
    return None


if __name__ == "__main__":
    main()
