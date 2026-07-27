#!/usr/bin/env python3
"""Generate a GPU SLURM timing benchmark for matrix-ablation training jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import generate_ablation_train_slurm as ablation_train_slurm  # noqa: E402
import generate_timing_benchmark_slurm as timing_slurm  # noqa: E402
import run_timing_benchmark as timing_runner  # noqa: E402

from ham_embed_spectral.naming import (  # noqa: E402
    ENCODER_CLI_CHOICES,
    canonical_encoder_name,
)

DEFAULT_NAME = "ablation_timing_memory"
DEFAULT_OUTPUT_DIR = ROOT / "results/timing_benchmarks"


@dataclass(frozen=True)
class AblationTimingBenchmark:
    """Generated ablation timing benchmark artifacts."""

    manifest_paths: tuple[Path, ...]
    slurm_script_path: Path
    slurm_script: str
    n_benchmark_jobs: int
    timing_csv: Path
    timing_report: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    benchmark = parser.add_argument_group("ablation benchmark")
    benchmark.add_argument(
        "--manifests",
        nargs="+",
        default=list(ablation_train_slurm.DEFAULT_MANIFESTS),
        help="Base manifest aliases ('pendigits', 'synthetic') or JSON manifest paths.",
    )
    benchmark.add_argument(
        "--ablations",
        nargs="+",
        choices=ablation_train_slurm.DEFAULT_ABLATIONS,
        default=list(ablation_train_slurm.DEFAULT_ABLATIONS),
    )
    benchmark.add_argument(
        "--manifest-output-dir",
        default=str(ablation_train_slurm.DEFAULT_MANIFEST_OUTPUT_DIR),
    )
    benchmark.add_argument(
        "--ablation-seed",
        type=int,
        default=ablation_train_slurm.DEFAULT_ABLATION_SEED,
    )
    benchmark.add_argument("--name", default=DEFAULT_NAME)
    benchmark.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    benchmark.add_argument("--work-root", default=None)
    benchmark.add_argument("--encoders", nargs="+", choices=ENCODER_CLI_CHOICES, default=None)
    benchmark.add_argument(
        "--datasets",
        nargs="+",
        choices=timing_runner.DATASET_CHOICES,
        default=None,
    )
    benchmark.add_argument(
        "--representations",
        nargs="+",
        choices=timing_runner.REPRESENTATION_CHOICES,
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
    slurm.add_argument("--job-name", default="qfm_ablation_timing")
    slurm.add_argument("--account", default=None)
    slurm.add_argument("--partition", default=None)
    slurm.add_argument("--qos", default=None)
    slurm.add_argument("--reservation", default=None)
    slurm.add_argument("--constraint", default=None)
    slurm.add_argument("--exclude", default=None)
    slurm.add_argument("--time", default="48:00:00")
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
    args = parser.parse_args(argv)
    if args.dry_run and args.submit:
        raise ValueError("--dry-run cannot be combined with --submit")
    if not args.reupload_depths:
        raise ValueError("--reupload-depths must contain at least one depth")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed")
    if args.encoders is not None:
        args.encoders = [canonical_encoder_name(value) for value in args.encoders]
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    benchmark = generate_ablation_timing_benchmark(args)
    print(json.dumps(summary_payload(benchmark, args), indent=2, sort_keys=True))
    if args.dry_run:
        print(benchmark.slurm_script)


def generate_ablation_timing_benchmark(args: argparse.Namespace) -> AblationTimingBenchmark:
    """Write derived ablation manifests and optionally write/submit the timing SLURM script."""

    manifest_paths = write_ablation_timing_manifests(args)
    timing_args = timing_generator_args(args, manifest_paths)
    slurm_script = timing_slurm.render_slurm_script(timing_args)
    slurm_script_path = timing_slurm.slurm_script_path(timing_args)
    report_paths = timing_runner.benchmark_report_paths(Path(args.output_dir), args.name)
    n_benchmark_jobs = len(timing_runner.select_benchmark_jobs(timing_args))

    if not args.dry_run:
        slurm_script_path.parent.mkdir(parents=True, exist_ok=True)
        slurm_script_path.write_text(slurm_script, encoding="utf-8")
        print(slurm_script_path)
        if args.submit:
            subprocess.run(["sbatch", str(slurm_script_path)], check=True)

    return AblationTimingBenchmark(
        manifest_paths=tuple(manifest_paths),
        slurm_script_path=slurm_script_path,
        slurm_script=slurm_script,
        n_benchmark_jobs=n_benchmark_jobs,
        timing_csv=report_paths["csv"],
        timing_report=report_paths["json"],
    )


def write_ablation_timing_manifests(args: argparse.Namespace) -> list[Path]:
    """Write the derived ablation manifests used by the timing benchmark."""

    output_dir = Path(args.manifest_output_dir)
    manifest_paths: list[Path] = []
    for base_path in ablation_train_slurm.resolve_manifest_paths(args.manifests):
        base_manifest = ablation_train_slurm.load_manifest(base_path)
        for ablation in args.ablations:
            payload = ablation_train_slurm.derived_manifest_payload(
                base_manifest,
                ablation=ablation,
                ablation_seed=args.ablation_seed,
            )
            manifest_path = ablation_train_slurm.derived_manifest_path(
                payload,
                ablation,
                output_dir,
            )
            ablation_train_slurm.write_manifest(manifest_path, payload)
            manifest_paths.append(manifest_path)
    return manifest_paths


def timing_generator_args(
    args: argparse.Namespace,
    manifest_paths: list[Path],
) -> argparse.Namespace:
    """Return an argparse namespace compatible with generate_timing_benchmark_slurm.py."""

    return argparse.Namespace(
        manifests=[str(path) for path in manifest_paths],
        name=args.name,
        output_dir=args.output_dir,
        work_root=args.work_root,
        encoders=args.encoders,
        datasets=args.datasets,
        representations=args.representations,
        reupload_depths=args.reupload_depths,
        seeds=args.seeds,
        steps=args.steps,
        max_train_examples=args.max_train_examples,
        max_eval_examples=args.max_eval_examples,
        limit=args.limit,
        keep_temp_runs=args.keep_temp_runs,
        fail_fast=args.fail_fast,
        extra_train_args=args.extra_train_args,
        project_root=args.project_root,
        venv_dir=args.venv_dir,
        slurm_script=args.slurm_script,
        log_dir=args.log_dir,
        job_name=args.job_name,
        account=args.account,
        partition=args.partition,
        qos=args.qos,
        reservation=args.reservation,
        constraint=args.constraint,
        exclude=args.exclude,
        time=args.time,
        mem=args.mem,
        gpus_per_node=args.gpus_per_node,
        cpus_per_gpu=args.cpus_per_gpu,
        nodes=args.nodes,
        ntasks_per_node=args.ntasks_per_node,
        mail_type=args.mail_type,
        mail_user=args.mail_user,
        module=args.module,
        submit=False,
        dry_run=args.dry_run,
    )


def summary_payload(
    benchmark: AblationTimingBenchmark,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Return a concise summary of the generated ablation timing benchmark."""

    return {
        "dry_run": bool(args.dry_run),
        "submit": bool(args.submit),
        "manifest_output_dir": str(args.manifest_output_dir),
        "n_derived_manifests": len(benchmark.manifest_paths),
        "n_benchmark_jobs": benchmark.n_benchmark_jobs,
        "derived_manifests": [str(path) for path in benchmark.manifest_paths],
        "slurm_script": str(benchmark.slurm_script_path),
        "timing_csv": str(benchmark.timing_csv),
        "timing_report": str(benchmark.timing_report),
    }


if __name__ == "__main__":
    main()
