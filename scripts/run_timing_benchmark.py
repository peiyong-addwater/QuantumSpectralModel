#!/usr/bin/env python3
"""Run representative training jobs and write a wall-time benchmark report."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ham_embed_spectral.experiments.manifest import (  # noqa: E402
    DATASET_CHOICES,
    REPRESENTATION_CHOICES,
    TrainJob,
    load_manifest,
    manifest_section,
    normalize_optional_value,
)
from ham_embed_spectral.naming import (  # noqa: E402
    ENCODER_CLI_CHOICES,
    canonical_encoder_name,
)

DEFAULT_MANIFESTS = (
    "configs/experiments/pendigits.json",
    "configs/experiments/synthetic.json",
)
DEFAULT_DEPTHS = (1, 8, 32)
DEFAULT_SEEDS = (0,)
TEMP_EXPERIMENT_NAME = "timing_benchmark_tmp"

CSV_FIELDS = (
    "status",
    "manifest_id",
    "job_slug",
    "dataset",
    "representation",
    "encoder",
    "reupload_depth",
    "seed",
    "learning_rate",
    "batch_size",
    "steps",
    "ablation",
    "ablation_seed",
    "returncode",
    "subprocess_wall_time_seconds",
    "train_reported_wall_time_seconds",
    "memory_report_available",
    "memory_report_source",
    "subprocess_peak_rss_kb",
    "subprocess_peak_rss_mb",
    "parameter_count",
    "n_qubits",
    "hilbert_dim",
    "final_validation_accuracy",
    "final_test_accuracy",
    "started_at_utc",
    "finished_at_utc",
)


@dataclass(frozen=True)
class BenchmarkJob:
    """One representative training job selected from a manifest."""

    manifest_path: Path
    manifest: dict[str, Any]
    train_job: TrainJob
    steps: int

    @property
    def manifest_id(self) -> str:
        return str(self.manifest["manifest_id"])

    @property
    def slug(self) -> str:
        return self.train_job.slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifests", nargs="+", default=list(DEFAULT_MANIFESTS))
    parser.add_argument("--output-dir", default="results/timing_benchmarks")
    parser.add_argument("--name", default="representative_timing")
    parser.add_argument(
        "--work-root",
        default=None,
        help=(
            "Parent directory for temporary training outputs. Defaults to "
            "$SLURM_TMPDIR when set, otherwise the system temp directory."
        ),
    )
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
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override manifest training steps for timing calibration.",
    )
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-temp-runs", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--extra-train-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments appended verbatim after '--'.",
    )
    args = parser.parse_args()
    if not args.reupload_depths:
        raise ValueError("--reupload-depths must contain at least one depth")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    if args.steps is not None and args.steps < 0:
        raise ValueError("--steps must be nonnegative")
    if args.encoders is not None:
        args.encoders = [canonical_encoder_name(value) for value in args.encoders]
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
            print(f"[{index}/{len(jobs)}] timing {job.slug}", flush=True)
            row = run_one_job(job, args, temp_workspace)
            rows.append(row)
            write_reports(report_paths, args, jobs, rows, temp_workspace)
            if row["returncode"] != 0 and args.fail_fast:
                break
    finally:
        if not args.keep_temp_runs:
            shutil.rmtree(temp_workspace, ignore_errors=True)
            write_reports(report_paths, args, jobs, rows, temp_workspace)

    print(f"timing_report_json={report_paths['json']}", flush=True)
    print(f"timing_report_csv={report_paths['csv']}", flush=True)
    if any(row["returncode"] != 0 for row in rows):
        raise SystemExit(1)


def select_benchmark_jobs(args: argparse.Namespace) -> list[BenchmarkJob]:
    """Select a representative Cartesian subset from the supplied manifests."""

    wanted_encoders = set(args.encoders or [])
    wanted_datasets = set(args.datasets or [])
    wanted_representations = set(args.representations or [])
    wanted_depths = {int(value) for value in args.reupload_depths}
    jobs: list[BenchmarkJob] = []
    seen: set[tuple[object, ...]] = set()

    for manifest_path_raw in args.manifests:
        manifest_path = Path(manifest_path_raw)
        manifest = load_manifest(manifest_path)
        grid = manifest["grid"]
        learning_rate = float(grid["learning_rates"][0])
        batch_size = int(grid["batch_sizes"][0])
        class_subsets = [
            normalize_optional_value(value) for value in grid.get("class_subsets", [None])
        ]
        class_subset = class_subsets[0] if class_subsets else None
        manifest_steps = int(manifest_section(manifest, "training").get("steps", 100))
        steps = int(args.steps if args.steps is not None else manifest_steps)
        ablation, ablation_seed = ablation_settings(manifest)

        for dataset in grid["datasets"]:
            if wanted_datasets and dataset not in wanted_datasets:
                continue
            for representation in grid["representations"]:
                if wanted_representations and representation not in wanted_representations:
                    continue
                if not representation_matches_dataset(dataset, representation):
                    continue
                for encoder_raw in grid["encoders"]:
                    encoder = canonical_encoder_name(encoder_raw)
                    if wanted_encoders and encoder not in wanted_encoders:
                        continue
                    for depth_raw in grid["reupload_depths"]:
                        depth = int(depth_raw)
                        if depth not in wanted_depths:
                            continue
                        for seed in args.seeds:
                            train_job = TrainJob(
                                dataset=dataset,
                                representation=representation,
                                encoder=encoder,
                                reupload_depth=depth,
                                seed=int(seed),
                                learning_rate=learning_rate,
                                batch_size=batch_size,
                                class_subset=class_subset,
                            )
                            key = (
                                manifest["manifest_id"],
                                train_job.dataset,
                                train_job.representation,
                                train_job.encoder,
                                train_job.reupload_depth,
                                train_job.seed,
                                train_job.learning_rate,
                                train_job.batch_size,
                                train_job.class_subset,
                                ablation,
                                ablation_seed,
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            jobs.append(
                                BenchmarkJob(
                                    manifest_path=manifest_path,
                                    manifest=manifest,
                                    train_job=train_job,
                                    steps=steps,
                                )
                            )
    if args.limit is not None:
        return jobs[: args.limit]
    return jobs


def representation_matches_dataset(dataset: str, representation: str) -> bool:
    if dataset == "pendigits":
        return representation != "synthetic"
    return representation == "synthetic"


def ablation_settings(manifest: dict[str, Any]) -> tuple[str, int]:
    """Return the manifest ablation name and deterministic ablation seed."""

    ablations = manifest_section(manifest, "ablations")
    ablation = str(ablations.get("ablation", "none"))
    ablation_seed = int(ablations.get("ablation_seed", 0))
    return ablation, ablation_seed


def run_one_job(
    job: BenchmarkJob,
    args: argparse.Namespace,
    temp_workspace: Path,
) -> dict[str, Any]:
    temp_output_root = temp_workspace / "runs"
    command = build_train_command(job, args, temp_output_root)
    run_dir = temp_output_root / TEMP_EXPERIMENT_NAME / f"{job.slug}_seed{job.train_job.seed}"
    summary_path = temp_output_root / TEMP_EXPERIMENT_NAME / f"{job.slug}_summary.json"

    started_at = datetime.now(UTC)
    started = time.perf_counter()
    usage_path = temp_workspace / "resource_usage" / f"{job.slug}_seed{job.train_job.seed}.txt"
    result, resource_usage = run_with_resource_report(command, ROOT, usage_path)
    subprocess_wall = time.perf_counter() - started
    finished_at = datetime.now(UTC)

    metrics = load_json(run_dir / "metrics.json")
    config = load_json(run_dir / "config.json")
    summary = metrics.get("summary", {}) if isinstance(metrics, dict) else {}
    final_validation = summary.get("final_validation", {})
    final_test = summary.get("final_test", {})
    model_config = config.get("model_config", {}) if isinstance(config, dict) else {}
    ablation, ablation_seed = ablation_settings(job.manifest)

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
        "learning_rate": job.train_job.learning_rate,
        "batch_size": job.train_job.batch_size,
        "steps": job.steps,
        "ablation": ablation,
        "ablation_seed": ablation_seed,
        "returncode": result.returncode,
        "subprocess_wall_time_seconds": subprocess_wall,
        "train_reported_wall_time_seconds": summary.get("wall_time_seconds"),
        **resource_usage,
        "parameter_count": model_config.get("parameter_count"),
        "n_qubits": model_config.get("n_qubits"),
        "hilbert_dim": model_config.get("hilbert_dim"),
        "final_validation_accuracy": final_validation.get("accuracy"),
        "final_test_accuracy": final_test.get("accuracy"),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "command": redact_project_paths(command),
        "temporary_run_dir": str(run_dir),
        "temporary_artifacts_deleted": not args.keep_temp_runs,
    }

    if not args.keep_temp_runs:
        shutil.rmtree(run_dir, ignore_errors=True)
        summary_path.unlink(missing_ok=True)
    return row


def run_with_resource_report(
    command: list[str],
    cwd: Path,
    usage_path: Path,
) -> tuple[subprocess.CompletedProcess[Any], dict[str, Any]]:
    """Run a command and collect peak RSS from GNU time when available.

    The SLURM ``--mem`` limit accounts for host RAM, so this records the
    subprocess maximum resident set size rather than GPU memory. The benchmark
    still runs if GNU time is unavailable; those rows are marked explicitly.
    """

    time_executable = gnu_time_executable()
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
    return result, parse_gnu_time_report(usage_path)


@cache
def gnu_time_executable() -> str | None:
    """Return a GNU time executable supporting ``-v``, if one is available."""

    candidates = [
        os.environ.get("QFM_GNU_TIME"),
        "/usr/bin/time",
        shutil.which("gtime"),
        shutil.which("time"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_absolute() and not path.exists():
            continue
        probe = subprocess.run(
            [candidate, "-v", "true"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            return str(candidate)
    return None


def parse_gnu_time_report(path: Path) -> dict[str, Any]:
    """Parse the memory fields from a GNU ``/usr/bin/time -v`` report."""

    row: dict[str, Any] = {
        "memory_report_available": False,
        "memory_report_source": "gnu_time_v",
        "subprocess_peak_rss_kb": None,
        "subprocess_peak_rss_mb": None,
    }
    if not path.exists():
        return row

    peak_rss_kb = None
    for line in path.read_text(errors="replace").splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        if key.strip() == "Maximum resident set size (kbytes)":
            peak_rss_kb = parse_int(value.strip())
            break

    if peak_rss_kb is None:
        return row

    row["memory_report_available"] = True
    row["subprocess_peak_rss_kb"] = peak_rss_kb
    row["subprocess_peak_rss_mb"] = peak_rss_kb / 1024.0
    return row


def parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def build_train_command(
    job: BenchmarkJob,
    args: argparse.Namespace,
    temp_output_root: Path,
) -> list[str]:
    manifest = job.manifest
    train_job = job.train_job
    data = manifest_section(manifest, "data")
    synthetic = manifest_section(manifest, "synthetic")
    training = manifest_section(manifest, "training")
    model = manifest_section(manifest, "model")
    ablations = manifest_section(manifest, "ablations")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--dataset",
        train_job.dataset,
        "--representation",
        train_job.representation,
        "--encoder",
        train_job.encoder,
        "--reupload-depth",
        str(train_job.reupload_depth),
        "--seed",
        str(train_job.seed),
        "--steps",
        str(job.steps),
        "--batch-size",
        str(train_job.batch_size),
        "--eval-batch-size",
        str(int(training.get("eval_batch_size", 128))),
        "--learning-rate",
        f"{train_job.learning_rate:g}",
        "--weight-decay",
        f"{float(training.get('weight_decay', 0.0)):g}",
        "--log-every",
        str(int(training.get("log_every", 50))),
        "--eval-every",
        str(int(training.get("eval_every", 100))),
        "--data-root",
        str(data.get("data_root", "data/raw/pendigits")),
        "--validation-fraction",
        f"{float(data.get('validation_fraction', 0.1)):g}",
        "--output-root",
        str(temp_output_root),
        "--experiment-name",
        TEMP_EXPERIMENT_NAME,
        "--run-id",
        train_job.slug,
        "--dtype",
        str(model.get("dtype", "float64")),
        "--manifest-id",
        job.manifest_id,
        "--job-slug",
        train_job.slug,
        "--mixer-scale",
        f"{float(model.get('mixer_scale', 0.01)):g}",
        "--ry-alpha",
        f"{float(model.get('ry_alpha', 1.0)):g}",
        "--rz-beta",
        f"{float(model.get('rz_beta', 1.0)):g}",
        "--tf-init-scale",
        f"{float(model.get('tf_init_scale', 1.0)):g}",
        "--tf-init-noise",
        f"{float(model.get('tf_init_noise', 0.01)):g}",
        "--patch-scale",
        f"{float(model.get('patch_scale', 1.0)):g}",
        "--patch-map-init-noise",
        f"{float(model.get('patch_map_init_noise', 0.01)):g}",
        "--initial-state",
        str(model.get("initial_state", "plus")),
    ]
    data_seed = data.get("data_seed")
    if data_seed is not None:
        command.extend(["--data-seed", str(data_seed)])
    if train_job.dataset.startswith("synthetic-"):
        command.extend(synthetic_train_args(train_job.dataset, synthetic))
    command.append(
        "--projector-renormalize"
        if bool(model.get("projector_renormalize", True))
        else "--no-projector-renormalize"
    )
    command.append(
        "--trainable-times"
        if bool(model.get("trainable_times", True))
        else "--no-trainable-times"
    )
    fixed_time = model.get("fixed_time")
    if fixed_time is not None:
        command.extend(["--fixed-time", f"{float(fixed_time):g}"])
    command.append(
        "--track-readout-leakage"
        if bool(model.get("track_readout_leakage", False))
        else "--no-track-readout-leakage"
    )
    if train_job.class_subset is not None:
        command.extend(["--class-subset", train_job.class_subset])
    if bool(data.get("download_data", False)):
        command.append("--download-data")
    command.append("--standardize" if bool(data.get("standardize", True)) else "--no-standardize")
    command.append("--no-checkpoint")
    ablation = ablations.get("ablation", "none")
    if ablation != "none":
        command.extend(["--ablation", str(ablation)])
        command.extend(["--ablation-seed", str(int(ablations.get("ablation_seed", 0)))])
    if args.max_train_examples is not None:
        command.extend(["--max-train-examples", str(args.max_train_examples)])
    if args.max_eval_examples is not None:
        command.extend(["--max-eval-examples", str(args.max_eval_examples)])
    command.extend(strip_remainder_separator(args.extra_train_args))
    return command


def synthetic_train_args(dataset: str, synthetic: dict[str, Any]) -> list[str]:
    settings = dict(synthetic.get("datasets", {}).get(dataset, {}))
    command = [
        "--n-samples",
        str(int(synthetic.get("n_samples", 128))),
        "--synthetic-dim",
        str(int(settings.get("synthetic_dim", 4))),
        "--synthetic-rows",
        str(int(settings.get("synthetic_rows", 4))),
        "--synthetic-cols",
        str(int(settings.get("synthetic_cols", 2))),
        "--synthetic-noise-epsilon",
        f"{float(synthetic.get('synthetic_noise_epsilon', 0.0)):g}",
    ]
    threshold = settings.get("synthetic_threshold")
    if threshold is not None:
        command.extend(["--synthetic-threshold", f"{float(threshold):g}"])
    return command


def dry_run_payload(
    args: argparse.Namespace,
    jobs: list[BenchmarkJob],
    report_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "n_jobs": len(jobs),
        "manifests": list(args.manifests),
        "encoders": sorted({job.train_job.encoder for job in jobs}),
        "datasets": sorted({job.train_job.dataset for job in jobs}),
        "representations": sorted({job.train_job.representation for job in jobs}),
        "reupload_depths": sorted({job.train_job.reupload_depth for job in jobs}),
        "seeds": sorted({job.train_job.seed for job in jobs}),
        "ablations": sorted({ablation_settings(job.manifest)[0] for job in jobs}),
        "report_json": str(report_paths["json"]),
        "report_csv": str(report_paths["csv"]),
        "temporary_artifact_policy": "delete after each job unless --keep-temp-runs is set",
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
        "n_planned": len(jobs),
        "n_recorded": len(rows),
        "n_complete": sum(row["status"] == "complete" for row in rows),
        "n_failed": sum(row["status"] == "failed" for row in rows),
        "temporary_workspace": str(temp_workspace),
        "temporary_artifact_policy": (
            "kept because --keep-temp-runs was set"
            if args.keep_temp_runs
            else "temporary train.py run directories are deleted after parsing metrics"
        ),
        "memory_reporting": {
            "method": "GNU time -v maximum resident set size",
            "scope": "host RAM for each train.py subprocess",
            "csv_fields": [
                "memory_report_available",
                "memory_report_source",
                "subprocess_peak_rss_kb",
                "subprocess_peak_rss_mb",
            ],
        },
        "slurm": slurm_environment(),
        "planned_jobs": [planned_job_record(job) for job in jobs],
        "runs": rows,
        "groups": group_summaries(rows),
        "csv_path": str(report_paths["csv"]),
    }
    report_paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_csv(report_paths["csv"], rows)


def planned_job_record(job: BenchmarkJob) -> dict[str, Any]:
    train_job = job.train_job
    ablation, ablation_seed = ablation_settings(job.manifest)
    return {
        "manifest_id": job.manifest_id,
        "manifest_path": str(job.manifest_path),
        "job_slug": job.slug,
        "dataset": train_job.dataset,
        "representation": train_job.representation,
        "encoder": train_job.encoder,
        "reupload_depth": train_job.reupload_depth,
        "seed": train_job.seed,
        "learning_rate": train_job.learning_rate,
        "batch_size": train_job.batch_size,
        "steps": job.steps,
        "ablation": ablation,
        "ablation_seed": ablation_seed,
    }


def group_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subprocess_groups: dict[tuple[object, ...], list[float]] = {}
    train_reported_groups: dict[tuple[object, ...], list[float]] = {}
    peak_rss_groups: dict[tuple[object, ...], list[float]] = {}
    for row in rows:
        if row.get("status") != "complete":
            continue
        subprocess_wall_time = row.get("subprocess_wall_time_seconds")
        if subprocess_wall_time is None:
            continue
        key = (
            row.get("dataset"),
            row.get("representation"),
            row.get("encoder"),
            row.get("reupload_depth"),
            row.get("ablation", "none"),
            row.get("ablation_seed", 0),
        )
        subprocess_groups.setdefault(key, []).append(float(subprocess_wall_time))
        train_reported_wall_time = row.get("train_reported_wall_time_seconds")
        if train_reported_wall_time is not None:
            train_reported_groups.setdefault(key, []).append(float(train_reported_wall_time))
        peak_rss_mb = row.get("subprocess_peak_rss_mb")
        if peak_rss_mb is not None:
            peak_rss_groups.setdefault(key, []).append(float(peak_rss_mb))

    summaries = []
    for key, values in sorted(subprocess_groups.items()):
        dataset, representation, encoder, depth, ablation, ablation_seed = key
        summary = {
            "dataset": dataset,
            "representation": representation,
            "encoder": encoder,
            "reupload_depth": depth,
            "ablation": ablation,
            "ablation_seed": ablation_seed,
            "n": len(values),
            "subprocess_wall_time_seconds_mean": mean(values),
            "subprocess_wall_time_seconds_median": median(values),
            "subprocess_wall_time_seconds_min": min(values),
            "subprocess_wall_time_seconds_max": max(values),
        }
        train_values = train_reported_groups.get(key, [])
        if train_values:
            summary.update(
                {
                    "train_reported_wall_time_seconds_mean": mean(train_values),
                    "train_reported_wall_time_seconds_median": median(train_values),
                    "train_reported_wall_time_seconds_min": min(train_values),
                    "train_reported_wall_time_seconds_max": max(train_values),
                }
            )
        peak_rss_values = peak_rss_groups.get(key, [])
        if peak_rss_values:
            summary.update(
                {
                    "subprocess_peak_rss_mb_mean": mean(peak_rss_values),
                    "subprocess_peak_rss_mb_median": median(peak_rss_values),
                    "subprocess_peak_rss_mb_min": min(peak_rss_values),
                    "subprocess_peak_rss_mb_max": max(peak_rss_values),
                }
            )
        summaries.append(summary)
    return summaries


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
    if args.work_root:
        parent = Path(args.work_root)
    else:
        parent = Path(os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir())
    parent.mkdir(parents=True, exist_ok=True)
    return parent


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def strip_remainder_separator(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


if __name__ == "__main__":
    main()
