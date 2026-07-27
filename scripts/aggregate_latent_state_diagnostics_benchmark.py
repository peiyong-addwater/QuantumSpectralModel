#!/usr/bin/env python3
"""Aggregate latent-state diagnostics resource benchmark CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_latent_state_diagnostics_resource_benchmark import (  # noqa: E402
    CSV_FIELDS,
    group_summaries,
    select_benchmark_jobs,
)

from ham_embed_spectral.naming import canonical_encoder_name  # noqa: E402

DEFAULT_INPUT_DIR = "results/tables/latent_state_diagnostics/resource_benchmarks"
DEFAULT_CHECKPOINT_INPUT_DIR = (
    "results/tables/latent_state_diagnostics/resource_benchmarks_checkpoints"
)
DEFAULT_INPUT_DIRS = (DEFAULT_INPUT_DIR, DEFAULT_CHECKPOINT_INPUT_DIR)
DEFAULT_OUTPUT_PREFIX = "results/tables/latent_state_diagnostics/resource_benchmarks/combined"
DEFAULT_EXPECTED_MANIFESTS = (
    "configs/experiments/pendigits.json",
    "configs/experiments/synthetic.json",
)
DEFAULT_MODES = ("final", "checkpoints")
DEFAULT_EXCLUDED_MANIFEST_IDS = ("smoke_tiny",)
DEFAULT_EXPECTED_DEPTHS = (1, 8, 32)
DEFAULT_EXPECTED_SEEDS = (0,)
WORKLOAD_KEY_FIELDS = (
    "manifest_id",
    "dataset",
    "representation",
    "encoder",
    "reupload_depth",
    "seed",
    "mode",
    "split",
    "diagnostic_batch_size",
    "spectral_state_max_samples",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Legacy single input directory. Use --input-dirs for mixed-mode aggregation.",
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=None,
        help=(
            "Input directories to search for per-job benchmark *_runs.csv files. "
            f"Defaults to: {', '.join(DEFAULT_INPUT_DIRS)}."
        ),
    )
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--pattern", default="*_runs.csv")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("init-reference", "checkpoints", "final"),
        default=list(DEFAULT_MODES),
        help="Keep only rows from these latent diagnostic modes.",
    )
    parser.add_argument(
        "--include-manifest-ids",
        nargs="+",
        default=None,
        help="Keep only these manifest ids after discovery. This explicit include list "
        "takes precedence over the default smoke manifest exclusion.",
    )
    parser.add_argument(
        "--exclude-manifest-ids",
        nargs="*",
        default=list(DEFAULT_EXCLUDED_MANIFEST_IDS),
        help="Drop rows from these manifest ids. Pass the flag with no values to disable.",
    )
    parser.add_argument(
        "--expected-manifests",
        nargs="+",
        default=list(DEFAULT_EXPECTED_MANIFESTS),
        help="Manifest files used to build the expected benchmark coverage grid.",
    )
    parser.add_argument(
        "--expected-depths",
        nargs="+",
        type=int,
        default=list(DEFAULT_EXPECTED_DEPTHS),
    )
    parser.add_argument(
        "--expected-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_EXPECTED_SEEDS),
    )
    parser.add_argument("--expected-split", default="validation")
    parser.add_argument("--expected-diagnostic-batch-size", type=int, default=32)
    parser.add_argument("--expected-spectral-state-max-samples", type=int, default=8)
    parser.add_argument(
        "--no-coverage-check",
        action="store_false",
        dest="coverage_check",
        help="Skip expected-grid coverage checks and only validate rows that were found.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if sanity checks find failed rows, duplicates, missing timings, "
        "or missing expected benchmark coverage.",
    )
    parser.set_defaults(coverage_check=True)
    return normalize_args(parser.parse_args())


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.input_dir is not None and args.input_dirs is not None:
        raise ValueError("--input-dir and --input-dirs cannot be used together")
    args.input_dirs = normalized_input_dirs(args)
    args.input_dir = args.input_dirs[0]
    if not args.modes:
        raise ValueError("--modes must contain at least one mode")
    if args.include_manifest_ids is not None and not args.include_manifest_ids:
        raise ValueError("--include-manifest-ids must contain at least one manifest id")
    if not args.expected_depths:
        raise ValueError("--expected-depths must contain at least one depth")
    if not args.expected_seeds:
        raise ValueError("--expected-seeds must contain at least one seed")
    if args.expected_diagnostic_batch_size < 1:
        raise ValueError("--expected-diagnostic-batch-size must be positive")
    if args.expected_spectral_state_max_samples < 0:
        raise ValueError("--expected-spectral-state-max-samples must be nonnegative")
    return args


def normalized_input_dirs(args: argparse.Namespace) -> list[str]:
    input_dirs = getattr(args, "input_dirs", None)
    input_dir = getattr(args, "input_dir", None)
    if input_dirs is not None:
        normalized = list(input_dirs)
    elif input_dir is not None:
        normalized = [input_dir]
    else:
        normalized = list(DEFAULT_INPUT_DIRS)
    if not normalized:
        raise ValueError("--input-dirs must contain at least one input directory")
    return normalized


def main() -> None:
    args = parse_args()
    report = aggregate(args)
    print(f"combined_latent_resource_csv={report['csv_path']}")
    print(f"combined_latent_resource_json={report['json_path']}")
    print(
        "latent_resource_sanity="
        f"{'ok' if report['sanity_checks']['ok'] else 'issues_found'} "
        f"complete={report['n_complete']} failed={report['n_failed']} "
        f"missing_expected={report['sanity_checks']['n_missing_expected_workloads']} "
        f"duplicates={report['sanity_checks']['n_duplicate_workload_keys']}"
    )
    if args.strict and not report["sanity_checks"]["ok"]:
        raise SystemExit(1)


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    args.input_dirs = normalized_input_dirs(args)
    args.input_dir = args.input_dirs[0]
    input_dirs = [Path(path) for path in args.input_dirs]
    output_prefix = Path(args.output_prefix)
    csv_path, json_path = output_paths(output_prefix)
    paths = discover_csvs(input_dirs, args.pattern, csv_path)
    all_rows = read_csv_rows(paths)
    kept_rows, skipped_rows = filter_rows(all_rows, args)
    report = write_combined_reports(
        csv_path=csv_path,
        json_path=json_path,
        input_dirs=input_dirs,
        paths=paths,
        all_rows=all_rows,
        kept_rows=kept_rows,
        skipped_rows=skipped_rows,
        args=args,
    )
    return report


def output_paths(output_prefix: Path) -> tuple[Path, Path]:
    return (
        output_prefix.with_name(f"{output_prefix.name}_runs.csv"),
        output_prefix.with_name(f"{output_prefix.name}_report.json"),
    )


def discover_csvs(input_dirs: list[Path], pattern: str, output_csv: Path) -> list[Path]:
    paths = []
    output_csv_resolved = output_csv.resolve()
    seen: set[Path] = set()
    for input_dir in input_dirs:
        for path in sorted(input_dir.rglob(pattern)):
            resolved = path.resolve()
            if resolved == output_csv_resolved:
                continue
            if resolved in seen:
                continue
            if is_combined_runs_csv(path):
                continue
            seen.add(resolved)
            paths.append(path)
    return paths


def is_combined_runs_csv(path: Path) -> bool:
    return path.name.startswith("combined") and path.name.endswith("_runs.csv")


def read_csv_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["source_csv_path"] = str(path)
                rows.append(row)
    return rows


def filter_rows(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    include_manifest_ids = set(args.include_manifest_ids or [])
    exclude_manifest_ids = set(args.exclude_manifest_ids or [])
    modes = set(args.modes or [])
    kept: list[dict[str, str]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        reason = filter_skip_reason(row, include_manifest_ids, exclude_manifest_ids, modes)
        if reason is None:
            kept.append(row)
        else:
            skipped.append(
                {
                    "reason": reason,
                    "manifest_id": row.get("manifest_id", ""),
                    "mode": row_mode(row),
                    "source_csv_path": row.get("source_csv_path", ""),
                }
            )
    return kept, skipped


def filter_skip_reason(
    row: dict[str, str],
    include_manifest_ids: set[str],
    exclude_manifest_ids: set[str],
    modes: set[str],
) -> str | None:
    manifest_id = row.get("manifest_id", "")
    mode = row_mode(row)
    if include_manifest_ids and manifest_id not in include_manifest_ids:
        return "manifest_id_not_in_include_filter"
    if manifest_id in exclude_manifest_ids and manifest_id not in include_manifest_ids:
        return "manifest_id_excluded"
    if modes and mode not in modes:
        return "mode_not_selected"
    return None


def write_combined_reports(
    csv_path: Path,
    json_path: Path,
    input_dirs: list[Path],
    paths: list[Path],
    all_rows: list[dict[str, str]],
    kept_rows: list[dict[str, str]],
    skipped_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_with_extra_fields(csv_path, kept_rows)
    sanity = build_sanity_checks(kept_rows, args)
    report = {
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "input_dir": str(input_dirs[0]) if input_dirs else "",
        "input_dirs": [str(path) for path in input_dirs],
        "pattern": args.pattern,
        "n_input_csvs": len(paths),
        "input_csvs": [str(path) for path in paths],
        "n_discovered_rows": len(all_rows),
        "n_rows": len(kept_rows),
        "n_skipped_rows": len(skipped_rows),
        "skipped_rows_by_reason": dict(Counter(row["reason"] for row in skipped_rows)),
        "n_complete": sum(row.get("status") == "complete" for row in kept_rows),
        "n_failed": sum(row.get("status") == "failed" for row in kept_rows),
        "groups": group_summaries_safe(kept_rows),
        "sanity_checks": sanity,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_sanity_checks(rows: list[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    missing_fields = missing_required_fields(rows)
    noncomplete_rows = [row_summary(row) for row in rows if row.get("status") != "complete"]
    missing_timing_rows = [
        row_summary(row)
        for row in rows
        if row.get("status") == "complete"
        and parse_positive_float(row.get("subprocess_wall_time_seconds")) is None
    ]
    duplicate_keys = duplicate_workload_keys(rows)
    missing_expected = missing_expected_workloads(rows, args) if args.coverage_check else []
    coverage = coverage_summary(rows)
    blocking_issue_counts = {
        "missing_required_fields": len(missing_fields),
        "noncomplete_rows": len(noncomplete_rows),
        "missing_timing_rows": len(missing_timing_rows),
        "duplicate_workload_keys": len(duplicate_keys),
        "missing_expected_workloads": len(missing_expected),
    }
    return {
        "ok": not any(blocking_issue_counts.values()),
        "blocking_issue_counts": blocking_issue_counts,
        "missing_required_fields": missing_fields,
        "n_noncomplete_rows": len(noncomplete_rows),
        "noncomplete_rows": noncomplete_rows,
        "n_missing_timing_rows": len(missing_timing_rows),
        "missing_timing_rows": missing_timing_rows,
        "n_duplicate_workload_keys": len(duplicate_keys),
        "duplicate_workload_keys": duplicate_keys,
        "coverage_check_enabled": bool(args.coverage_check),
        "coverage": coverage,
        "expected_coverage": expected_coverage_config(args) if args.coverage_check else None,
        "n_missing_expected_workloads": len(missing_expected),
        "missing_expected_workloads": missing_expected,
    }


def missing_required_fields(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    missing = []
    required_fields = set(CSV_FIELDS)
    for row in rows:
        row_missing = sorted(field for field in required_fields if field not in row)
        if row_missing:
            missing.append(
                {
                    "source_csv_path": row.get("source_csv_path", ""),
                    "missing_fields": row_missing,
                }
            )
    return missing


def duplicate_workload_keys(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[workload_key(row)].append(row)
    duplicates = []
    for key, members in sorted(grouped.items()):
        if len(members) <= 1:
            continue
        duplicates.append(
            {
                "key": workload_key_record(key),
                "n": len(members),
                "source_csv_paths": sorted(
                    {member.get("source_csv_path", "") for member in members}
                ),
            }
        )
    return duplicates


def missing_expected_workloads(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    observed = {
        workload_key(row)
        for row in rows
        if row.get("status") == "complete"
        and parse_positive_float(row.get("subprocess_wall_time_seconds")) is not None
    }
    expected = expected_workload_keys(args)
    missing = sorted(expected - observed)
    return [workload_key_record(key) for key in missing]


def expected_workload_keys(args: argparse.Namespace) -> set[tuple[str, ...]]:
    expected_args = argparse.Namespace(
        manifests=args.expected_manifests,
        output_dir=args.input_dir,
        name="latent_state_resource_expected_coverage",
        work_root=None,
        encoders=None,
        datasets=None,
        representations=None,
        reupload_depths=args.expected_depths,
        seeds=args.expected_seeds,
        limit=None,
        dry_run=False,
        keep_temp_runs=False,
        fail_fast=False,
        extra_train_args=[],
    )
    keys: set[tuple[str, ...]] = set()
    for job in select_benchmark_jobs(expected_args):
        train_job = job.train_job
        for mode in args.modes:
            record = {
                "manifest_id": str(job.manifest_id),
                "dataset": str(train_job.dataset),
                "representation": str(train_job.representation),
                "encoder": canonical_encoder_or_raw(str(train_job.encoder)),
                "reupload_depth": normalize_optional_int(train_job.reupload_depth),
                "seed": normalize_optional_int(train_job.seed),
                "mode": str(mode),
                "split": str(args.expected_split),
                "diagnostic_batch_size": normalize_optional_int(
                    args.expected_diagnostic_batch_size
                ),
                "spectral_state_max_samples": normalize_optional_int(
                    args.expected_spectral_state_max_samples
                ),
            }
            keys.add(tuple(record[field] for field in WORKLOAD_KEY_FIELDS))
    return keys


def expected_coverage_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "manifests": list(args.expected_manifests),
        "modes": list(args.modes),
        "depths": list(args.expected_depths),
        "seeds": list(args.expected_seeds),
        "split": args.expected_split,
        "diagnostic_batch_size": args.expected_diagnostic_batch_size,
        "spectral_state_max_samples": args.expected_spectral_state_max_samples,
    }


def coverage_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        if row.get("status") != "complete":
            continue
        if parse_positive_float(row.get("subprocess_wall_time_seconds")) is None:
            continue
        counts[
            (
                row.get("manifest_id", ""),
                row_mode(row),
                normalize_optional_int(row.get("reupload_depth")),
            )
        ] += 1
    return [
        {
            "manifest_id": manifest_id,
            "mode": mode,
            "reupload_depth": reupload_depth,
            "n_complete_with_timing": count,
        }
        for (manifest_id, mode, reupload_depth), count in sorted(counts.items())
    ]


def workload_key(row: dict[str, str]) -> tuple[str, ...]:
    record = {
        "manifest_id": row.get("manifest_id", ""),
        "dataset": row.get("dataset", ""),
        "representation": row.get("representation", ""),
        "encoder": canonical_encoder_or_raw(row.get("encoder", "")),
        "reupload_depth": normalize_optional_int(row.get("reupload_depth")),
        "seed": normalize_optional_int(row.get("seed")),
        "mode": row_mode(row),
        "split": row.get("split") or "validation",
        "diagnostic_batch_size": normalize_optional_int(row.get("diagnostic_batch_size")),
        "spectral_state_max_samples": normalize_optional_int(row.get("spectral_state_max_samples")),
    }
    return tuple(record[field] for field in WORKLOAD_KEY_FIELDS)


def workload_key_record(key: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(WORKLOAD_KEY_FIELDS, key, strict=True))


def row_summary(row: dict[str, str]) -> dict[str, Any]:
    summary = workload_key_record(workload_key(row))
    summary.update(
        {
            "status": row.get("status", ""),
            "returncode": row.get("returncode", ""),
            "subprocess_wall_time_seconds": row.get("subprocess_wall_time_seconds", ""),
            "source_csv_path": row.get("source_csv_path", ""),
        }
    )
    return summary


def group_summaries_safe(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    sanitized_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "complete":
            continue
        wall_time = parse_positive_float(row.get("subprocess_wall_time_seconds"))
        if wall_time is None:
            continue
        sanitized = dict(row)
        sanitized["encoder"] = canonical_encoder_or_raw(row.get("encoder", ""))
        sanitized["subprocess_wall_time_seconds"] = wall_time
        rss = parse_positive_float(row.get("subprocess_peak_rss_mb"))
        if rss is not None:
            sanitized["subprocess_peak_rss_mb"] = rss
        else:
            sanitized.pop("subprocess_peak_rss_mb", None)
        sanitized_rows.append(sanitized)
    return group_summaries(sanitized_rows)


def write_csv_with_extra_fields(path: Path, rows: list[dict[str, str]]) -> None:
    extra_fields = sorted({field for row in rows for field in row} - set(CSV_FIELDS))
    fieldnames = [*CSV_FIELDS, *extra_fields]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def row_mode(row: dict[str, str]) -> str:
    return row.get("mode") or row.get("diagnostic_mode") or ""


def normalize_optional_int(value: object) -> str:
    if value in {None, ""}:
        return ""
    try:
        return str(int(float(str(value))))
    except ValueError:
        return str(value)


def canonical_encoder_or_raw(value: str) -> str:
    if not value:
        return ""
    try:
        return canonical_encoder_name(value)
    except ValueError:
        return value


def parse_positive_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(str(value))
    except ValueError:
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


if __name__ == "__main__":
    main()
