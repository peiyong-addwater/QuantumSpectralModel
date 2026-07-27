#!/usr/bin/env python3
"""Generate and optionally submit one SLURM job for timing benchmarks."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ham_embed_spectral.experiments.manifest import (  # noqa: E402
    DATASET_CHOICES,
    REPRESENTATION_CHOICES,
    sanitize_slug,
)
from ham_embed_spectral.naming import ENCODER_CLI_CHOICES  # noqa: E402

DEFAULT_MANIFESTS = (
    "configs/experiments/pendigits.json",
    "configs/experiments/synthetic.json",
)
DEFAULT_MODULES = ("uv/0.9.5", "cuda/12.9.1", "openmpi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    benchmark = parser.add_argument_group("benchmark")
    benchmark.add_argument("--manifests", nargs="+", default=list(DEFAULT_MANIFESTS))
    benchmark.add_argument("--name", default="representative_timing")
    benchmark.add_argument("--output-dir", default="results/timing_benchmarks")
    benchmark.add_argument("--work-root", default=None)
    benchmark.add_argument("--encoders", nargs="+", choices=ENCODER_CLI_CHOICES, default=None)
    benchmark.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, default=None)
    benchmark.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATION_CHOICES,
        default=None,
    )
    benchmark.add_argument("--reupload-depths", nargs="+", type=int, default=[1, 8, 32])
    benchmark.add_argument("--seeds", nargs="+", type=int, default=[0])
    benchmark.add_argument("--steps", type=int, default=None)
    benchmark.add_argument("--max-train-examples", type=int, default=None)
    benchmark.add_argument("--max-eval-examples", type=int, default=None)
    benchmark.add_argument("--limit", type=int, default=None)
    benchmark.add_argument("--keep-temp-runs", action="store_true")
    benchmark.add_argument("--fail-fast", action="store_true")
    benchmark.add_argument(
        "--extra-train-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments appended verbatim after '--'.",
    )

    slurm = parser.add_argument_group("slurm")
    slurm.add_argument("--project-root", default=str(ROOT))
    slurm.add_argument("--venv-dir", default=".venv")
    slurm.add_argument("--slurm-script", default=None)
    slurm.add_argument("--log-dir", default="sbatch_log")
    slurm.add_argument("--job-name", default="qfm_timing")
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
    slurm.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = slurm_script_path(args)
    script = render_slurm_script(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "slurm_script": str(script_path),
                    "slurm": slurm_defaults(args),
                    "benchmark_command": benchmark_command(args),
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(script)
        return

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)
    print(script_path)
    if args.submit:
        subprocess.run(["sbatch", str(script_path)], check=True)


def slurm_script_path(args: argparse.Namespace) -> Path:
    if args.slurm_script:
        return Path(args.slurm_script)
    return Path("slurm") / "timing_benchmark" / f"{sanitize_slug(args.name)}.slurm"


def render_slurm_script(args: argparse.Namespace) -> str:
    modules = args.module or list(DEFAULT_MODULES)
    output_pattern = f"{args.log_dir}/%x-%j.out"
    sbatch_lines = "\n".join(sbatch_directives(args, output_pattern))
    module_lines = "\n".join(f"module load {shlex.quote(module)}" for module in modules)
    command = format_shell_command(benchmark_command(args))
    return f"""#!/bin/bash
{sbatch_lines}

set -euo pipefail

PROJECT_ROOT={shlex.quote(str(args.project_root))}
VENV_DIR={shlex.quote(args.venv_dir)}

cd "$PROJECT_ROOT"
mkdir -p {shlex.quote(args.log_dir)}
{module_lines}
export JAX_PLATFORMS=cuda

source "$PROJECT_ROOT/$VENV_DIR/bin/activate"

{command}
"""


def sbatch_directives(args: argparse.Namespace, output_pattern: str) -> list[str]:
    directives = [
        ("job-name", args.job_name),
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


def benchmark_command(args: argparse.Namespace) -> list[str]:
    command = [
        "uv",
        "run",
        "scripts/run_timing_benchmark.py",
        "--manifests",
        *args.manifests,
        "--output-dir",
        args.output_dir,
        "--name",
        args.name,
        "--reupload-depths",
        *(str(value) for value in args.reupload_depths),
        "--seeds",
        *(str(value) for value in args.seeds),
    ]
    append_optional_values(command, "--work-root", [args.work_root] if args.work_root else [])
    append_optional_values(command, "--encoders", args.encoders or [])
    append_optional_values(command, "--datasets", args.datasets or [])
    append_optional_values(command, "--representations", args.representations or [])
    if args.steps is not None:
        command.extend(["--steps", str(args.steps)])
    if args.max_train_examples is not None:
        command.extend(["--max-train-examples", str(args.max_train_examples)])
    if args.max_eval_examples is not None:
        command.extend(["--max-eval-examples", str(args.max_eval_examples)])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.keep_temp_runs:
        command.append("--keep-temp-runs")
    if args.fail_fast:
        command.append("--fail-fast")
    command.extend(strip_remainder_separator(args.extra_train_args))
    return command


def append_optional_values(command: list[str], flag: str, values: list[str]) -> None:
    if values:
        command.append(flag)
        command.extend(str(value) for value in values)


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


def strip_remainder_separator(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


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
        "log_dir": args.log_dir,
        "job_name": args.job_name,
    }


if __name__ == "__main__":
    main()
