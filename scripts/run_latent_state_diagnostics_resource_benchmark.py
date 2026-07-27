#!/usr/bin/env python3
"""Benchmark local latent-state diagnostic wall time and memory usage."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ham_embed_spectral.experiments.manifest import (  # noqa: E402
    DATASET_CHOICES,
    REPRESENTATION_CHOICES,
)
from ham_embed_spectral.naming import ENCODER_CLI_CHOICES, canonical_encoder_name  # noqa: E402
from scripts import run_timing_benchmark as timing_runner  # noqa: E402

DEFAULT_MANIFESTS = ("configs/experiments/Legacy/smoke_tiny.json",)
CSV_FIELDS = (
    "status",
    "manifest_id",
    "job_slug",
    "dataset",
    "representation",
    "encoder",
    "reupload_depth",
    "seed",
    "mode",
    "split",
    "diagnostic_batch_size",
    "spectral_state_max_samples",
    "returncode",
    "subprocess_wall_time_seconds",
    "memory_report_available",
    "memory_report_source",
    "subprocess_peak_rss_kb",
    "subprocess_peak_rss_mb",
    "gpu_memory_report_available",
    "gpu_memory_report_source",
    "gpu_memory_used_mb_before",
    "gpu_memory_used_mb_after",
    "gpu_memory_used_mb_max_observed",
    "started_at_utc",
    "finished_at_utc",
)


@dataclass(frozen=True)
class BenchmarkJob:
    """One latent-diagnostic benchmark job selected from a manifest."""

    manifest_path: Path
    manifest_id: str
    train_job: Any

    @property
    def slug(self) -> str:
        return self.train_job.slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifests", nargs="+", default=list(DEFAULT_MANIFESTS))
    parser.add_argument(
        "--output-dir",
        default="results/tables/latent_state_diagnostics/resource_benchmarks",
    )
    parser.add_argument("--name", default="local_latent_state_diagnostics")
    parser.add_argument("--work-root", default=None)
    parser.add_argument(
        "--mode",
        choices=("init-reference", "checkpoints", "final"),
        default="init-reference",
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--diagnostic-batch-size", type=int, default=8)
    parser.add_argument("--spectral-state-max-samples", type=int, default=2)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--encoders", nargs="+", choices=ENCODER_CLI_CHOICES, default=None)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, default=None)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATION_CHOICES,
        default=None,
    )
    parser.add_argument("--reupload-depths", nargs="+", type=int, default=[1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-temp-outputs", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--extra-diagnostic-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional latent_state_diagnostics.py arguments appended after '--'.",
    )
    args = parser.parse_args()
    if args.encoders is not None:
        args.encoders = [canonical_encoder_name(value) for value in args.encoders]
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    if args.diagnostic_batch_size < 1:
        raise ValueError("--diagnostic-batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    jobs = select_benchmark_jobs(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_paths = benchmark_report_paths(output_dir, args.name)

    if args.dry_run:
        print(json.dumps(dry_run_payload(args, jobs, report_paths), indent=2, sort_keys=True))
        for job in jobs:
            print(job.slug)
        return

    temp_workspace = Path(tempfile.mkdtemp(prefix=f"{args.name}_", dir=str(work_parent(args))))
    rows: list[dict[str, Any]] = []
    write_reports(report_paths, args, jobs, rows, temp_workspace)
    try:
        for index, job in enumerate(jobs, start=1):
            print(f"[{index}/{len(jobs)}] latent diagnostics {job.slug}", flush=True)
            row = run_one_job(job, args, temp_workspace)
            rows.append(row)
            write_reports(report_paths, args, jobs, rows, temp_workspace)
            if row["returncode"] != 0 and args.fail_fast:
                break
    finally:
        if not args.keep_temp_outputs:
            shutil.rmtree(temp_workspace, ignore_errors=True)
            write_reports(report_paths, args, jobs, rows, temp_workspace)

    print(f"benchmark_report_json={report_paths['json']}", flush=True)
    print(f"benchmark_report_csv={report_paths['csv']}", flush=True)
    if any(row["returncode"] != 0 for row in rows):
        raise SystemExit(1)


def select_benchmark_jobs(args: argparse.Namespace) -> list[BenchmarkJob]:
    timing_args = argparse.Namespace(
        manifests=args.manifests,
        output_dir=args.output_dir,
        name=args.name,
        work_root=args.work_root,
        encoders=args.encoders,
        datasets=args.datasets,
        representations=args.representations,
        reupload_depths=args.reupload_depths,
        seeds=args.seeds,
        steps=0,
        max_train_examples=None,
        max_eval_examples=None,
        limit=args.limit,
        dry_run=False,
        keep_temp_runs=False,
        fail_fast=False,
        extra_train_args=[],
    )
    selected = timing_runner.select_benchmark_jobs(timing_args)
    return [
        BenchmarkJob(
            manifest_path=job.manifest_path,
            manifest_id=job.manifest_id,
            train_job=job.train_job,
        )
        for job in selected
    ]


def run_one_job(
    job: BenchmarkJob,
    args: argparse.Namespace,
    temp_workspace: Path,
) -> dict[str, Any]:
    output_path = temp_workspace / "items" / f"{job.slug}_latent.json"
    hdf5_path = temp_workspace / "items" / f"{job.slug}_latent.h5"
    command = build_diagnostic_command(job, args, output_path, hdf5_path)
    usage_path = temp_workspace / "resource_usage" / f"{job.slug}.txt"

    started_at = datetime.now(UTC)
    started = time.perf_counter()
    gpu_before = query_gpu_memory_mb()
    result, resource_usage = run_with_resource_report(command, ROOT, usage_path)
    gpu_after = query_gpu_memory_mb()
    subprocess_wall = time.perf_counter() - started
    finished_at = datetime.now(UTC)

    row = {
        "status": "complete" if result.returncode == 0 else "failed",
        "manifest_id": job.manifest_id,
        "manifest_path": str(job.manifest_path),
        "job_slug": job.slug,
        "dataset": job.train_job.dataset,
        "representation": job.train_job.representation,
        "encoder": job.train_job.encoder,
        "reupload_depth": job.train_job.reupload_depth,
        "seed": job.train_job.seed,
        "mode": args.mode,
        "split": args.split,
        "diagnostic_batch_size": args.diagnostic_batch_size,
        "spectral_state_max_samples": args.spectral_state_max_samples,
        "returncode": result.returncode,
        "subprocess_wall_time_seconds": subprocess_wall,
        **resource_usage,
        **gpu_memory_report(gpu_before, gpu_after),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "command": timing_runner.redact_project_paths(command),
        "temporary_output_path": str(output_path),
        "temporary_hdf5_path": str(hdf5_path),
        "temporary_artifacts_deleted": not args.keep_temp_outputs,
    }
    return row


def build_diagnostic_command(
    job: BenchmarkJob,
    args: argparse.Namespace,
    output_path: Path,
    hdf5_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "latent_state_diagnostics.py"),
        "--manifest",
        str(job.manifest_path),
        "--mode",
        args.mode,
        "--output",
        str(output_path),
        "--hdf5-output",
        str(hdf5_path),
        "--datasets",
        job.train_job.dataset,
        "--representations",
        job.train_job.representation,
        "--encoders",
        job.train_job.encoder,
        "--reupload-depths",
        str(job.train_job.reupload_depth),
        "--seeds",
        str(job.train_job.seed),
        "--split",
        args.split,
        "--diagnostic-batch-size",
        str(args.diagnostic_batch_size),
        "--spectral-state-max-samples",
        str(args.spectral_state_max_samples),
        "--max-jobs",
        "1",
    ]
    if args.runs_root is not None:
        command.extend(["--runs-root", args.runs_root])
    if args.experiment_name is not None:
        command.extend(["--experiment-name", args.experiment_name])
    command.extend(strip_remainder_separator(args.extra_diagnostic_args))
    return command


def run_with_resource_report(
    command: list[str],
    cwd: Path,
    usage_path: Path,
) -> tuple[subprocess.CompletedProcess[Any], dict[str, Any]]:
    time_executable = timing_runner.gnu_time_executable()
    if time_executable is None:
        result = subprocess.run(command, cwd=cwd, check=False)
        return result, {
            "memory_report_available": False,
            "memory_report_source": None,
            "subprocess_peak_rss_kb": None,
            "subprocess_peak_rss_mb": None,
        }
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [time_executable, "-v", "-o", str(usage_path), *command],
        cwd=cwd,
        check=False,
    )
    return result, timing_runner.parse_gnu_time_report(usage_path)


def query_gpu_memory_mb() -> list[float] | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    result = subprocess.run(
        [
            executable,
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    values = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            values.append(float(stripped))
        except ValueError:
            return None
    return values or None


def gpu_memory_report(before: list[float] | None, after: list[float] | None) -> dict[str, Any]:
    if before is None and after is None:
        return {
            "gpu_memory_report_available": False,
            "gpu_memory_report_source": None,
            "gpu_memory_used_mb_before": None,
            "gpu_memory_used_mb_after": None,
            "gpu_memory_used_mb_max_observed": None,
        }
    before_max = max(before) if before else None
    after_max = max(after) if after else None
    observed = [value for value in (before_max, after_max) if value is not None]
    return {
        "gpu_memory_report_available": True,
        "gpu_memory_report_source": "nvidia-smi memory.used before/after subprocess",
        "gpu_memory_used_mb_before": before_max,
        "gpu_memory_used_mb_after": after_max,
        "gpu_memory_used_mb_max_observed": max(observed) if observed else None,
    }


def write_reports(
    report_paths: dict[str, Path],
    args: argparse.Namespace,
    jobs: list[BenchmarkJob],
    rows: list[dict[str, Any]],
    temp_workspace: Path,
) -> None:
    report_paths["json"].parent.mkdir(parents=True, exist_ok=True)
    report = {
        "name": args.name,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "project_root": ".",
        "manifests": list(args.manifests),
        "mode": args.mode,
        "split": args.split,
        "diagnostic_batch_size": args.diagnostic_batch_size,
        "spectral_state_max_samples": args.spectral_state_max_samples,
        "n_planned": len(jobs),
        "n_recorded": len(rows),
        "n_complete": sum(row["status"] == "complete" for row in rows),
        "n_failed": sum(row["status"] == "failed" for row in rows),
        "temporary_workspace": str(temp_workspace),
        "temporary_artifact_policy": (
            "kept because --keep-temp-outputs was set"
            if args.keep_temp_outputs
            else "temporary latent diagnostic outputs are deleted after the benchmark"
        ),
        "memory_reporting": {
            "host_method": "GNU time -v maximum resident set size",
            "gpu_method": "optional nvidia-smi memory.used before/after subprocess",
        },
        "planned_jobs": [planned_job_record(job) for job in jobs],
        "runs": rows,
        "groups": group_summaries(rows),
        "csv_path": str(report_paths["csv"]),
    }
    report_paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_csv(report_paths["csv"], rows)


def group_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wall_groups: dict[tuple[object, ...], list[float]] = {}
    rss_groups: dict[tuple[object, ...], list[float]] = {}
    for row in rows:
        if row.get("status") != "complete":
            continue
        key = (
            row.get("dataset"),
            row.get("representation"),
            row.get("encoder"),
            row.get("reupload_depth"),
            row.get("mode"),
        )
        wall_groups.setdefault(key, []).append(float(row["subprocess_wall_time_seconds"]))
        if row.get("subprocess_peak_rss_mb") is not None:
            rss_groups.setdefault(key, []).append(float(row["subprocess_peak_rss_mb"]))
    summaries = []
    for key, values in sorted(wall_groups.items()):
        dataset, representation, encoder, depth, mode = key
        summary = {
            "dataset": dataset,
            "representation": representation,
            "encoder": encoder,
            "reupload_depth": depth,
            "mode": mode,
            "n": len(values),
            "subprocess_wall_time_seconds_mean": mean(values),
            "subprocess_wall_time_seconds_median": median(values),
            "subprocess_wall_time_seconds_min": min(values),
            "subprocess_wall_time_seconds_max": max(values),
        }
        rss_values = rss_groups.get(key, [])
        if rss_values:
            summary.update(
                {
                    "subprocess_peak_rss_mb_mean": mean(rss_values),
                    "subprocess_peak_rss_mb_median": median(rss_values),
                    "subprocess_peak_rss_mb_min": min(rss_values),
                    "subprocess_peak_rss_mb_max": max(rss_values),
                }
            )
        summaries.append(summary)
    return summaries


def planned_job_record(job: BenchmarkJob) -> dict[str, Any]:
    train_job = job.train_job
    return {
        "manifest_id": job.manifest_id,
        "manifest_path": str(job.manifest_path),
        "job_slug": train_job.slug,
        "dataset": train_job.dataset,
        "representation": train_job.representation,
        "encoder": train_job.encoder,
        "reupload_depth": train_job.reupload_depth,
        "seed": train_job.seed,
    }


def dry_run_payload(
    args,
    jobs: list[BenchmarkJob],
    report_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "n_jobs": len(jobs),
        "manifests": list(args.manifests),
        "mode": args.mode,
        "split": args.split,
        "diagnostic_batch_size": args.diagnostic_batch_size,
        "spectral_state_max_samples": args.spectral_state_max_samples,
        "report_json": str(report_paths["json"]),
        "report_csv": str(report_paths["csv"]),
        "temporary_artifact_policy": "delete after benchmark unless --keep-temp-outputs is set",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def benchmark_report_paths(output_dir: Path, name: str) -> dict[str, Path]:
    safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    return {
        "json": output_dir / f"{safe_name}_report.json",
        "csv": output_dir / f"{safe_name}_runs.csv",
    }


def work_parent(args: argparse.Namespace) -> Path:
    parent = Path(args.work_root) if args.work_root else Path(tempfile.gettempdir())
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def strip_remainder_separator(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


if __name__ == "__main__":
    main()
