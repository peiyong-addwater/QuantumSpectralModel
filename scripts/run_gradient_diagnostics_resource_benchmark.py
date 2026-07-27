#!/usr/bin/env python3
"""Benchmark resource use for gradient diagnostics jobs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ham_embed_spectral.experiments.manifest import (  # noqa: E402
    DATASET_CHOICES,
    REPRESENTATION_CHOICES,
    TrainJob,
    jobs_from_manifest,
    load_manifest,
    sanitize_slug,
)
from ham_embed_spectral.naming import (  # noqa: E402
    ENCODER_CLI_CHOICES,
    canonical_encoder_name,
)

DEFAULT_MANIFESTS = (
    "configs/experiments/pendigits.json",
    "configs/experiments/synthetic.json",
)
DEFAULT_MODES = ("init", "checkpoints", "final")
DEFAULT_DEPTHS = (1, 8, 32)
DEFAULT_SEEDS = (0,)
DEFAULT_OUTPUT_DIR = "results/tables/gradient_diagnostics/resource_benchmarks"
DIAGNOSTIC_MODES = ("init", "checkpoints", "final")

CSV_FIELDS = (
    "status",
    "manifest_id",
    "manifest_path",
    "benchmark_slug",
    "diagnostic_mode",
    "job_slug",
    "dataset",
    "representation",
    "encoder",
    "reupload_depth",
    "seed",
    "learning_rate",
    "batch_size",
    "returncode",
    "subprocess_wall_time_seconds",
    "diagnostic_json_path",
    "diagnostic_n_records",
    "diagnostic_n_complete",
    "diagnostic_status_counts_json",
    "process_sample_count",
    "process_tree_peak_rss_bytes",
    "process_tree_peak_rss_mb",
    "process_tree_peak_cpu_percent",
    "process_tree_mean_cpu_percent",
    "child_user_time_seconds",
    "child_system_time_seconds",
    "child_total_cpu_time_seconds",
    "child_max_rss_bytes",
    "child_max_rss_mb",
    "nvidia_smi_available",
    "gpu_sample_count",
    "peak_gpu_utilization_percent",
    "mean_gpu_utilization_percent",
    "peak_gpu_memory_used_mb",
    "peak_gpu_process_memory_used_mb",
    "started_at_utc",
    "finished_at_utc",
    "single_manifest_path",
    "samples_path",
    "command",
)


@dataclass(frozen=True)
class GradientDiagnosticBenchmarkJob:
    """One benchmarked gradient-diagnostics subprocess."""

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


@dataclass
class MonitorStats:
    sample_count: int = 0
    process_cpu_percent_sum: float = 0.0
    process_cpu_percent_count: int = 0
    peak_process_cpu_percent: float | None = None
    peak_process_tree_rss_bytes: int | None = None
    nvidia_smi_available: bool = False
    gpu_sample_count: int = 0
    gpu_utilization_sum: float = 0.0
    gpu_utilization_count: int = 0
    peak_gpu_utilization_percent: float | None = None
    peak_gpu_memory_used_mb: float | None = None
    peak_gpu_process_memory_used_mb: float | None = None


class ResourceMonitor:
    """Background sampler for process-tree host memory and GPU memory."""

    def __init__(self, pid: int, interval_seconds: float, samples_path: Path) -> None:
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.samples_path = samples_path
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._stats = MonitorStats(nvidia_smi_available=shutil.which("nvidia-smi") is not None)
        self._started = time.perf_counter()
        self._last_process_ticks: int | None = None
        self._last_sample_time: float | None = None
        self._clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])

    def start(self) -> None:
        self.samples_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_seconds + 1.0))

    def summary(self) -> dict[str, Any]:
        stats = self._stats
        mean_cpu = (
            stats.process_cpu_percent_sum / stats.process_cpu_percent_count
            if stats.process_cpu_percent_count
            else None
        )
        mean_gpu_utilization = (
            stats.gpu_utilization_sum / stats.gpu_utilization_count
            if stats.gpu_utilization_count
            else None
        )
        return {
            "process_sample_count": stats.sample_count,
            "process_tree_peak_rss_bytes": stats.peak_process_tree_rss_bytes,
            "process_tree_peak_rss_mb": bytes_to_mb(stats.peak_process_tree_rss_bytes),
            "process_tree_peak_cpu_percent": stats.peak_process_cpu_percent,
            "process_tree_mean_cpu_percent": mean_cpu,
            "nvidia_smi_available": stats.nvidia_smi_available,
            "gpu_sample_count": stats.gpu_sample_count,
            "peak_gpu_utilization_percent": stats.peak_gpu_utilization_percent,
            "mean_gpu_utilization_percent": mean_gpu_utilization,
            "peak_gpu_memory_used_mb": stats.peak_gpu_memory_used_mb,
            "peak_gpu_process_memory_used_mb": stats.peak_gpu_process_memory_used_mb,
        }

    def _run(self) -> None:
        with self.samples_path.open("a", encoding="utf-8") as handle:
            while not self._stop.is_set():
                sample = self._sample()
                handle.write(json.dumps(sample, sort_keys=True) + "\n")
                handle.flush()
                self._update_stats(sample)
                self._stop.wait(self.interval_seconds)

    def _sample(self) -> dict[str, Any]:
        now = time.perf_counter()
        process_tree = read_process_tree_sample(self.pid)
        if process_tree is not None:
            process_ticks = int(process_tree["cpu_ticks"])
            if self._last_process_ticks is not None and self._last_sample_time is not None:
                delta_ticks = process_ticks - self._last_process_ticks
                delta_time = max(now - self._last_sample_time, 1e-12)
                process_tree["cpu_percent"] = (
                    100.0 * delta_ticks / self._clock_ticks / delta_time
                )
            self._last_process_ticks = process_ticks
            self._last_sample_time = now
        pids = process_tree.get("pids", []) if process_tree is not None else []
        return {
            "elapsed_seconds": now - self._started,
            "process_tree": process_tree,
            "gpus": query_gpu_samples(),
            "gpu_processes": query_gpu_process_samples(set(pids)),
        }

    def _update_stats(self, sample: dict[str, Any]) -> None:
        stats = self._stats
        stats.sample_count += 1
        process_tree = sample.get("process_tree")
        if process_tree:
            rss = process_tree.get("rss_bytes")
            if rss is not None:
                stats.peak_process_tree_rss_bytes = max_none(
                    stats.peak_process_tree_rss_bytes,
                    int(rss),
                )
            cpu_percent = process_tree.get("cpu_percent")
            if cpu_percent is not None:
                cpu_percent = float(cpu_percent)
                stats.process_cpu_percent_sum += cpu_percent
                stats.process_cpu_percent_count += 1
                stats.peak_process_cpu_percent = max_none(
                    stats.peak_process_cpu_percent,
                    cpu_percent,
                )

        gpus = sample.get("gpus") or []
        if gpus:
            stats.gpu_sample_count += 1
        for gpu in gpus:
            utilization = gpu.get("utilization_gpu_percent")
            if utilization is not None:
                utilization = float(utilization)
                stats.gpu_utilization_sum += utilization
                stats.gpu_utilization_count += 1
                stats.peak_gpu_utilization_percent = max_none(
                    stats.peak_gpu_utilization_percent,
                    utilization,
                )
            memory_used = gpu.get("memory_used_mb")
            if memory_used is not None:
                stats.peak_gpu_memory_used_mb = max_none(
                    stats.peak_gpu_memory_used_mb,
                    float(memory_used),
                )

        for gpu_process in sample.get("gpu_processes") or []:
            used_memory = gpu_process.get("used_memory_mb")
            if used_memory is not None:
                stats.peak_gpu_process_memory_used_mb = max_none(
                    stats.peak_gpu_process_memory_used_mb,
                    float(used_memory),
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifests", nargs="+", default=list(DEFAULT_MANIFESTS))
    parser.add_argument("--modes", nargs="+", choices=DIAGNOSTIC_MODES, default=list(DEFAULT_MODES))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="gradient_diagnostics_resource")
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--encoders", nargs="+", choices=ENCODER_CLI_CHOICES, default=None)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, default=None)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATION_CHOICES,
        default=None,
    )
    parser.add_argument("--reupload-depths", nargs="+", type=int, default=list(DEFAULT_DEPTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--job-slugs", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--diagnostic-batch-size", type=int, default=32)
    parser.add_argument("--diagnostic-seed", type=int, default=0)
    parser.add_argument("--near-zero-tol", type=float, default=1e-10)
    parser.add_argument("--n-init-seeds", type=int, default=20)
    parser.add_argument("--fisher-batch-size", type=int, default=32)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-jax-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    if not args.reupload_depths:
        raise ValueError("--reupload-depths must contain at least one depth")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed")
    if args.sample_interval <= 0:
        raise ValueError("--sample-interval must be positive")
    if args.encoders is not None:
        args.encoders = [canonical_encoder_name(value) for value in args.encoders]
    return args


def main() -> None:
    args = parse_args()
    jobs = select_benchmark_jobs(args)
    output_dir = Path(args.output_dir)
    report_paths = benchmark_report_paths(output_dir, args.name)

    if args.dry_run:
        print(json.dumps(dry_run_payload(args, jobs, report_paths), indent=2, sort_keys=True))
        for job in jobs:
            print(job.slug)
        return

    if args.require_gpu and not args.skip_jax_preflight:
        run_jax_preflight()

    rows: list[dict[str, Any]] = []
    write_reports(report_paths, args, jobs, rows)
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] benchmarking {job.slug}", flush=True)
        row = run_one_job(job, args, output_dir)
        rows.append(row)
        write_reports(report_paths, args, jobs, rows)
        if row["returncode"] != 0 and args.fail_fast:
            break

    print(f"gradient_resource_report_json={report_paths['json']}", flush=True)
    print(f"gradient_resource_report_csv={report_paths['csv']}", flush=True)
    if any(row["returncode"] != 0 for row in rows):
        raise SystemExit(1)


def select_benchmark_jobs(args: argparse.Namespace) -> list[GradientDiagnosticBenchmarkJob]:
    """Select representative gradient diagnostic jobs."""

    wanted_encoders = set(args.encoders or [])
    wanted_datasets = set(args.datasets or [])
    wanted_representations = set(args.representations or [])
    wanted_depths = {int(value) for value in args.reupload_depths}
    wanted_seeds = {int(value) for value in args.seeds}
    wanted_job_slugs = set(args.job_slugs or [])
    selected: list[GradientDiagnosticBenchmarkJob] = []
    seen: set[tuple[object, ...]] = set()

    for manifest_raw in args.manifests:
        manifest_path = Path(manifest_raw)
        manifest = load_manifest(manifest_path)
        for train_job in jobs_from_manifest(manifest, encoders=args.encoders):
            if wanted_datasets and train_job.dataset not in wanted_datasets:
                continue
            if wanted_representations and train_job.representation not in wanted_representations:
                continue
            if not representation_matches_dataset(train_job.dataset, train_job.representation):
                continue
            if wanted_encoders and train_job.encoder not in wanted_encoders:
                continue
            if train_job.reupload_depth not in wanted_depths:
                continue
            if train_job.seed not in wanted_seeds:
                continue
            if wanted_job_slugs and train_job.slug not in wanted_job_slugs:
                continue
            for mode in args.modes:
                key = (
                    manifest["manifest_id"],
                    train_job.slug,
                    mode,
                )
                if key in seen:
                    continue
                seen.add(key)
                selected.append(
                    GradientDiagnosticBenchmarkJob(
                        manifest_path=manifest_path,
                        manifest=manifest,
                        train_job=train_job,
                        mode=mode,
                    )
                )
    if args.limit is not None:
        return selected[: args.limit]
    return selected


def representation_matches_dataset(dataset: str, representation: str) -> bool:
    if dataset == "pendigits":
        return representation != "synthetic"
    return representation == "synthetic"


def run_one_job(
    job: GradientDiagnosticBenchmarkJob,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    single_manifest_path = single_manifest_output_path(output_dir, args.name, job)
    diagnostic_path = diagnostic_output_path(output_dir, args.name, job)
    samples_path = resource_samples_path(output_dir, args.name, job)
    write_json(single_manifest_path, single_job_manifest_payload(job))

    command = build_gradient_diagnostics_command(job, args, single_manifest_path, diagnostic_path)
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    process = subprocess.Popen(command, cwd=ROOT)
    monitor = ResourceMonitor(process.pid, args.sample_interval, samples_path)
    monitor.start()
    try:
        returncode = process.wait()
    finally:
        monitor.stop()
    subprocess_wall = time.perf_counter() - started
    finished_at = datetime.now(UTC)
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)

    diagnostic_summary = summarize_diagnostic_output(diagnostic_path)
    monitor_summary = monitor.summary()
    user_seconds = usage_after.ru_utime - usage_before.ru_utime
    system_seconds = usage_after.ru_stime - usage_before.ru_stime
    child_max_rss_bytes = (
        maxrss_to_bytes(usage_after.ru_maxrss)
        if usage_after.ru_maxrss > usage_before.ru_maxrss
        else None
    )

    return {
        "status": "complete" if returncode == 0 else "failed",
        "manifest_id": job.manifest_id,
        "manifest_path": str(job.manifest_path),
        "benchmark_slug": job.slug,
        "diagnostic_mode": job.mode,
        "job_slug": job.train_job.slug,
        "dataset": job.train_job.dataset,
        "representation": job.train_job.representation,
        "encoder": job.train_job.encoder,
        "reupload_depth": job.train_job.reupload_depth,
        "seed": job.train_job.seed,
        "learning_rate": job.train_job.learning_rate,
        "batch_size": job.train_job.batch_size,
        "returncode": returncode,
        "subprocess_wall_time_seconds": subprocess_wall,
        "diagnostic_json_path": str(diagnostic_path),
        **diagnostic_summary,
        **monitor_summary,
        "child_user_time_seconds": user_seconds,
        "child_system_time_seconds": system_seconds,
        "child_total_cpu_time_seconds": user_seconds + system_seconds,
        "child_max_rss_bytes": child_max_rss_bytes,
        "child_max_rss_mb": bytes_to_mb(child_max_rss_bytes),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "single_manifest_path": str(single_manifest_path),
        "samples_path": str(samples_path),
        "command": redact_project_paths(command),
    }


def build_gradient_diagnostics_command(
    job: GradientDiagnosticBenchmarkJob,
    args: argparse.Namespace,
    manifest_path: Path,
    diagnostic_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "gradient_diagnostics.py"),
        "--manifest",
        str(manifest_path),
        "--mode",
        job.mode,
        "--runs-root",
        args.runs_root,
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
        "--output",
        str(diagnostic_path),
    ]
    if args.experiment_name is not None:
        command.extend(["--experiment-name", args.experiment_name])
    return command


def single_job_manifest_payload(job: GradientDiagnosticBenchmarkJob) -> dict[str, Any]:
    """Return a manifest payload containing exactly one TrainJob."""

    train_job = job.train_job
    payload = {key: value for key, value in job.manifest.items() if not key.startswith("_")}
    payload["manifest_id"] = job.manifest_id
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


def summarize_diagnostic_output(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    records = payload.get("records", []) if isinstance(payload, dict) else []
    status_counts = Counter(
        str(record.get("status", "missing_status"))
        for record in records
        if isinstance(record, dict)
    )
    return {
        "diagnostic_n_records": int(payload.get("n_records", len(records)))
        if isinstance(payload, dict)
        else 0,
        "diagnostic_n_complete": int(payload.get("n_complete", 0))
        if isinstance(payload, dict)
        else 0,
        "diagnostic_status_counts_json": json.dumps(dict(sorted(status_counts.items()))),
    }


def single_manifest_output_path(
    output_dir: Path,
    name: str,
    job: GradientDiagnosticBenchmarkJob,
) -> Path:
    return output_dir / "manifests" / f"{sanitize_slug(name)}__{job.slug}.json"


def diagnostic_output_path(
    output_dir: Path,
    name: str,
    job: GradientDiagnosticBenchmarkJob,
) -> Path:
    return output_dir / "diagnostics" / f"{sanitize_slug(name)}__{job.slug}.json"


def resource_samples_path(
    output_dir: Path,
    name: str,
    job: GradientDiagnosticBenchmarkJob,
) -> Path:
    return output_dir / "resource_samples" / f"{sanitize_slug(name)}__{job.slug}.jsonl"


def benchmark_report_paths(output_dir: Path, name: str) -> dict[str, Path]:
    safe_name = sanitize_slug(name)
    return {
        "json": output_dir / f"{safe_name}_report.json",
        "csv": output_dir / f"{safe_name}_runs.csv",
    }


def dry_run_payload(
    args: argparse.Namespace,
    jobs: list[GradientDiagnosticBenchmarkJob],
    report_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "n_jobs": len(jobs),
        "manifests": list(args.manifests),
        "modes": list(args.modes),
        "encoders": sorted({job.train_job.encoder for job in jobs}),
        "datasets": sorted({job.train_job.dataset for job in jobs}),
        "representations": sorted({job.train_job.representation for job in jobs}),
        "reupload_depths": sorted({job.train_job.reupload_depth for job in jobs}),
        "seeds": sorted({job.train_job.seed for job in jobs}),
        "report_json": str(report_paths["json"]),
        "report_csv": str(report_paths["csv"]),
        "require_gpu": args.require_gpu,
        "sample_interval": args.sample_interval,
    }


def write_reports(
    report_paths: dict[str, Path],
    args: argparse.Namespace,
    jobs: list[GradientDiagnosticBenchmarkJob],
    rows: list[dict[str, Any]],
) -> None:
    report_paths["json"].parent.mkdir(parents=True, exist_ok=True)
    report = {
        "name": args.name,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "project_root": ".",
        "manifests": list(args.manifests),
        "modes": list(args.modes),
        "n_planned": len(jobs),
        "n_recorded": len(rows),
        "n_complete": sum(row.get("status") == "complete" for row in rows),
        "n_failed": sum(row.get("status") == "failed" for row in rows),
        "resource_reporting": {
            "host_memory_method": "sampled /proc process-tree RSS",
            "child_max_rss_note": (
                "child ru_maxrss is recorded only when the subprocess raises the "
                "runner's cumulative child max RSS; process_tree_peak_rss_mb is "
                "the primary per-command host memory field"
            ),
            "gpu_method": "sampled nvidia-smi whole-GPU and compute-app process memory",
            "sample_interval_seconds": args.sample_interval,
        },
        "slurm": slurm_environment(),
        "planned_jobs": [planned_job_record(job) for job in jobs],
        "runs": rows,
        "groups": group_summaries(rows),
        "csv_path": str(report_paths["csv"]),
    }
    report_paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_csv(report_paths["csv"], rows)


def planned_job_record(job: GradientDiagnosticBenchmarkJob) -> dict[str, Any]:
    train_job = job.train_job
    return {
        "manifest_id": job.manifest_id,
        "manifest_path": str(job.manifest_path),
        "benchmark_slug": job.slug,
        "diagnostic_mode": job.mode,
        "job_slug": train_job.slug,
        "dataset": train_job.dataset,
        "representation": train_job.representation,
        "encoder": train_job.encoder,
        "reupload_depth": train_job.reupload_depth,
        "seed": train_job.seed,
        "learning_rate": train_job.learning_rate,
        "batch_size": train_job.batch_size,
    }


def group_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_keys = (
        "dataset",
        "representation",
        "encoder",
        "reupload_depth",
        "diagnostic_mode",
    )
    value_fields = (
        "subprocess_wall_time_seconds",
        "process_tree_peak_rss_mb",
        "child_max_rss_mb",
        "peak_gpu_utilization_percent",
        "mean_gpu_utilization_percent",
        "peak_gpu_memory_used_mb",
        "peak_gpu_process_memory_used_mb",
    )
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    counts: Counter[tuple[Any, ...]] = Counter()
    for row in rows:
        if row.get("status") != "complete":
            continue
        key = tuple(row.get(field) for field in group_keys)
        counts[key] += 1
        value_map = grouped.setdefault(key, {field: [] for field in value_fields})
        for field in value_fields:
            value = numeric_value(row.get(field))
            if value is not None:
                value_map[field].append(value)

    summaries = []
    for key in sorted(counts, key=lambda item: str(item)):
        summary = dict(zip(group_keys, key, strict=True))
        summary["n"] = counts[key]
        for field, values in grouped[key].items():
            if values:
                summary.update(prefixed_stats(field, values))
        summaries.append(summary)
    return summaries


def prefixed_stats(field: str, values: list[float]) -> dict[str, float]:
    return {
        f"{field}_mean": mean(values),
        f"{field}_median": median(values),
        f"{field}_min": min(values),
        f"{field}_max": max(values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json_cell(row.get(field)) for field in CSV_FIELDS})


def json_cell(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return "" if value is None else value


def read_process_tree_sample(pid: int) -> dict[str, Any] | None:
    proc_table = read_proc_table()
    if pid not in proc_table:
        return None
    children: dict[int, list[int]] = {}
    for child_pid, sample in proc_table.items():
        children.setdefault(int(sample["ppid"]), []).append(child_pid)
    tree_pids: list[int] = []
    queue: deque[int] = deque([pid])
    while queue:
        current = queue.popleft()
        if current not in proc_table:
            continue
        tree_pids.append(current)
        queue.extend(children.get(current, []))
    if not tree_pids:
        return None
    cpu_ticks = sum(int(proc_table[tree_pid]["cpu_ticks"]) for tree_pid in tree_pids)
    rss_bytes = sum(int(proc_table[tree_pid]["rss_bytes"]) for tree_pid in tree_pids)
    return {
        "pid": pid,
        "pids": tree_pids,
        "n_processes": len(tree_pids),
        "cpu_ticks": cpu_ticks,
        "rss_bytes": rss_bytes,
        "cpu_percent": None,
    }


def read_proc_table() -> dict[int, dict[str, int]]:
    table: dict[int, dict[str, int]] = {}
    proc_root = Path("/proc")
    for stat_path in proc_root.glob("[0-9]*/stat"):
        try:
            stat = stat_path.read_text()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        close_paren = stat.rfind(")")
        if close_paren < 0:
            continue
        try:
            pid = int(stat_path.parent.name)
            fields = stat[close_paren + 2 :].split()
            table[pid] = {
                "ppid": int(fields[1]),
                "cpu_ticks": int(fields[11]) + int(fields[12]),
                "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
            }
        except (IndexError, ValueError, OSError):
            continue
    return table


def query_gpu_samples() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    rows = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) < 6:
            continue
        rows.append(
            {
                "index": parse_int(row[0]),
                "uuid": row[1].strip(),
                "name": row[2].strip(),
                "utilization_gpu_percent": parse_float(row[3]),
                "memory_used_mb": parse_float(row[4]),
                "memory_total_mb": parse_float(row[5]),
            }
        )
    return rows


def query_gpu_process_samples(pids: set[int]) -> list[dict[str, Any]]:
    if not pids or shutil.which("nvidia-smi") is None:
        return []
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    rows = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) < 3:
            continue
        row_pid = parse_int(row[0])
        if row_pid not in pids:
            continue
        rows.append(
            {
                "pid": row_pid,
                "gpu_uuid": row[1].strip(),
                "used_memory_mb": parse_float(row[2]),
            }
        )
    return rows


def run_jax_preflight() -> None:
    code = """
import jax
import jax.numpy as jnp

devices = jax.devices()
gpu_devices = [device for device in devices if device.platform == "gpu"]
if not gpu_devices:
    raise SystemExit("ERROR: --require-gpu is set, but JAX reports no GPU devices")
target = gpu_devices[0]
x = jax.device_put(jnp.ones((64, 64), dtype=jnp.float32), target)
(x @ x).block_until_ready()
print("jax_preflight_device", target)
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def maxrss_to_bytes(value: int) -> int:
    if sys.platform == "darwin":
        return int(value)
    return int(value) * 1024


def bytes_to_mb(value: int | None) -> float | None:
    if value is None:
        return None
    return value / (1024.0 * 1024.0)


def parse_int(value: str) -> int | None:
    value = value.strip()
    if not value or value.lower() in {"n/a", "not supported"}:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_float(value: str) -> float | None:
    value = value.strip()
    if not value or value.lower() in {"n/a", "not supported"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def numeric_value(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def max_none(current: Any, candidate: Any) -> Any:
    return candidate if current is None else max(current, candidate)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def slurm_environment() -> dict[str, str | None]:
    keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "CUDA_VISIBLE_DEVICES",
    )
    return {key: os.environ.get(key) for key in keys}


def redact_project_paths(command: list[str]) -> list[str]:
    """Replace repository-absolute command arguments with relative paths."""

    root = str(ROOT)
    prefix = f"{root}{os.sep}"
    redacted = []
    for argument in command:
        if argument == root:
            redacted.append(".")
        elif argument.startswith(prefix):
            redacted.append(argument.removeprefix(prefix))
        else:
            redacted.append(argument)
    return redacted


if __name__ == "__main__":
    main()
