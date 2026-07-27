#!/usr/bin/env python3
"""Aggregate latent-state diagnostic JSON item summaries into flat tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SUMMARY_FIELDS = ("count", "mean", "median", "std", "min", "max", "q25", "q75")

IDENTITY_FIELDS = (
    "source_json_path",
    "record_index",
    "record_key",
    "schema_version",
    "manifest_id",
    "mode",
    "split",
    "status",
    "git_dirty",
    "dataset",
    "representation",
    "encoder",
    "depth",
    "seed",
    "diagnostic_mode",
    "checkpoint",
    "checkpoint_step",
    "job_slug",
)

RECORD_FIELDS = (
    *IDENTITY_FIELDS,
    "source_hdf5_path",
    "hdf5_status",
    "hdf5_schema_version",
    "hdf5_record_count",
    "diagnostic_batch_size",
    "diagnostic_seed",
    "spectral_state_max_samples",
    "filters_json",
    "n_diagnostic_examples",
    "final_projector_accuracy_on_diagnostic_batch",
    "readout_score_scale",
    "n_classes",
    "input_shape_json",
    "ablation_json",
    "data_seed",
    "n_train",
    "run_dir",
    "checkpoint_path",
    "stage_names_json",
    "final_kernel_target_alignment",
    "final_kernel_effective_rank",
    "final_kernel_centered_effective_rank",
    "final_kernel_within_count",
    "final_kernel_within_mean",
    "final_kernel_within_median",
    "final_kernel_within_std",
    "final_kernel_within_min",
    "final_kernel_within_max",
    "final_kernel_within_q25",
    "final_kernel_within_q75",
    "final_kernel_between_count",
    "final_kernel_between_mean",
    "final_kernel_between_median",
    "final_kernel_between_std",
    "final_kernel_between_min",
    "final_kernel_between_max",
    "final_kernel_between_q25",
    "final_kernel_between_q75",
    "final_kernel_mean_gap_within_minus_between",
    "logit_final_accuracy",
    "logit_total_path_length_count",
    "logit_total_path_length_mean",
    "logit_total_path_length_median",
    "logit_total_path_length_std",
    "logit_total_path_length_min",
    "logit_total_path_length_max",
    "logit_total_path_length_q25",
    "logit_total_path_length_q75",
    "logit_correct_total_path_length_count",
    "logit_correct_total_path_length_mean",
    "logit_correct_total_path_length_median",
    "logit_correct_total_path_length_std",
    "logit_correct_total_path_length_min",
    "logit_correct_total_path_length_max",
    "logit_correct_total_path_length_q25",
    "logit_correct_total_path_length_q75",
    "logit_incorrect_total_path_length_count",
    "logit_incorrect_total_path_length_mean",
    "logit_incorrect_total_path_length_median",
    "logit_incorrect_total_path_length_std",
    "logit_incorrect_total_path_length_min",
    "logit_incorrect_total_path_length_max",
    "logit_incorrect_total_path_length_q25",
    "logit_incorrect_total_path_length_q75",
    "logit_transition_names_json",
    "logit_mean_step_movement_json",
    "spectral_state_status",
    "spectral_state_n_samples",
    "error_type",
    "error",
)

LAYERWISE_FIELDS = (
    *IDENTITY_FIELDS,
    "metric_family",
    "stage",
    "layer_index",
    "stage_name",
    "projector_accuracy",
    "projector_mean_top_score",
    "top_score_count",
    "top_score_mean",
    "top_score_median",
    "top_score_std",
    "top_score_min",
    "top_score_max",
    "top_score_q25",
    "top_score_q75",
    "target_alignment",
    "effective_rank",
    "centered_effective_rank",
    "within_count",
    "within_mean",
    "within_median",
    "within_std",
    "within_min",
    "within_max",
    "within_q25",
    "within_q75",
    "between_count",
    "between_mean",
    "between_median",
    "between_std",
    "between_min",
    "between_max",
    "between_q25",
    "between_q75",
    "mean_gap_within_minus_between",
)

CKA_FIELDS = (
    *IDENTITY_FIELDS,
    "comparison_family",
    "comparison_index",
    "layer_index",
    "cka_value",
)

SPECTRAL_STATE_FIELDS = (
    *IDENTITY_FIELDS,
    "spectral_state_status",
    "spectral_state_n_samples",
    "metric",
    "layer_index",
    "value_count",
    "value_mean",
    "value_median",
    "value_std",
    "value_min",
    "value_max",
    "value_q25",
    "value_q75",
)

HDF5_OK_STATUS = "ok"
HDF5_SKIPPED_STATUS = "skipped"


@dataclass(frozen=True)
class ItemPayload:
    """One per-item latent diagnostics JSON payload."""

    path: Path
    payload: dict[str, Any]

    @property
    def records(self) -> list[dict[str, Any]]:
        records = self.payload.get("records", [])
        if not isinstance(records, list):
            raise ValueError("field 'records' must be a list")
        if not all(isinstance(record, dict) for record in records):
            raise ValueError("field 'records' must contain only objects")
        return records


@dataclass(frozen=True)
class HDF5Check:
    """Readability status for one optional HDF5 sidecar."""

    path: Path
    status: str
    schema_version: str = ""
    record_count: int | None = None
    error_type: str = ""
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default="results/tables/latent_state_diagnostics/batched/items",
    )
    parser.add_argument(
        "--output-prefix",
        default="results/tables/latent_state_diagnostics/batched_summary",
    )
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when malformed items, duplicate keys, incomplete rows, "
        "HDF5 sidecar issues, mixed schemas, or expected-value mismatches are found.",
    )
    parser.add_argument(
        "--skip-hdf5-check",
        action="store_true",
        help="Do not open HDF5 sidecars; record their expected paths only.",
    )
    parser.add_argument("--expected-split")
    parser.add_argument("--expected-diagnostic-batch-size", type=int)
    parser.add_argument("--expected-spectral-state-max-samples", type=int)
    parser.add_argument("--expected-depths", nargs="+", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = aggregate(args)
    print(f"latent_records_csv={report['output_paths']['records_csv']}")
    print(f"latent_records_json={report['output_paths']['records_json']}")
    print(f"latent_layerwise_csv={report['output_paths']['layerwise_csv']}")
    print(f"latent_cka_csv={report['output_paths']['cka_csv']}")
    print(f"latent_spectral_state_csv={report['output_paths']['spectral_state_csv']}")
    print(f"latent_report_json={report['output_paths']['report_json']}")
    print(
        "latent_item_aggregation="
        f"{'ok' if report['ok'] else 'issues_found'} "
        f"records={report['n_records']} complete={report['n_complete']} "
        f"errors={len(report['errors'])}"
    )
    if args.strict and not report["ok"]:
        raise SystemExit(1)


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir)
    output_prefix = Path(args.output_prefix)
    paths = discover_json_files(input_dir, args.pattern)
    items, item_errors, consistency_issues = load_item_payloads(paths)
    hdf5_checks = collect_hdf5_checks(items, skip=args.skip_hdf5_check)

    record_rows: list[dict[str, Any]] = []
    layerwise_rows: list[dict[str, Any]] = []
    cka_rows: list[dict[str, Any]] = []
    spectral_state_rows: list[dict[str, Any]] = []
    seen_keys: dict[str, str] = {}
    duplicate_keys: list[dict[str, Any]] = []

    for item in items:
        hdf5_check = hdf5_checks[str(item.path)]
        for index, record in enumerate(item.records):
            context = base_context(item, index, record)
            key = str(context["record_key"])
            if key in seen_keys:
                duplicate_keys.append(
                    {
                        "record_key": key,
                        "first_source_json_path": seen_keys[key],
                        "duplicate_source_json_path": str(item.path),
                    }
                )
            else:
                seen_keys[key] = str(item.path)
            record_rows.append(flatten_record_row(item, record, context, hdf5_check))
            layerwise_rows.extend(flatten_layerwise_rows(record, context))
            cka_rows.extend(flatten_cka_rows(record, context))
            spectral_state_rows.extend(flatten_spectral_state_rows(record, context))

    paths_by_name = output_paths(output_prefix)
    report = build_report(
        args=args,
        input_dir=input_dir,
        json_paths=paths,
        items=items,
        item_errors=item_errors,
        consistency_issues=consistency_issues,
        hdf5_checks=hdf5_checks,
        record_rows=record_rows,
        layerwise_rows=layerwise_rows,
        cka_rows=cka_rows,
        spectral_state_rows=spectral_state_rows,
        duplicate_keys=duplicate_keys,
        paths_by_name=paths_by_name,
    )

    write_csv(paths_by_name["records_csv"], RECORD_FIELDS, record_rows)
    write_json(
        paths_by_name["records_json"],
        {
            "updated_at_utc": report["updated_at_utc"],
            "input_dir": str(input_dir),
            "n_records": len(record_rows),
            "records": record_rows,
        },
    )
    write_csv(paths_by_name["layerwise_csv"], LAYERWISE_FIELDS, layerwise_rows)
    write_csv(paths_by_name["cka_csv"], CKA_FIELDS, cka_rows)
    write_csv(paths_by_name["spectral_state_csv"], SPECTRAL_STATE_FIELDS, spectral_state_rows)
    write_json(paths_by_name["report_json"], report)
    return report


def output_paths(output_prefix: Path) -> dict[str, Path]:
    parent = output_prefix.parent
    stem = output_prefix.name
    return {
        "records_csv": parent / f"{stem}_records.csv",
        "records_json": parent / f"{stem}_records.json",
        "layerwise_csv": parent / f"{stem}_layerwise.csv",
        "cka_csv": parent / f"{stem}_cka.csv",
        "spectral_state_csv": parent / f"{stem}_spectral_state.csv",
        "report_json": parent / f"{stem}_report.json",
    }


def discover_json_files(input_dir: Path, pattern: str) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(path for path in input_dir.rglob(pattern) if path.is_file())


def load_item_payloads(
    paths: list[Path],
) -> tuple[list[ItemPayload], list[dict[str, Any]], list[dict[str, Any]]]:
    items = []
    errors = []
    consistency_issues = []
    for path in paths:
        try:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("top-level JSON value must be an object")
            item = ItemPayload(path=path, payload=payload)
            records = item.records
        except Exception as exc:  # noqa: BLE001 - aggregation should report bad items.
            errors.append(
                {
                    "source_json_path": str(path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        consistency_issues.extend(payload_consistency_issues(item, records))
        items.append(item)
    return items, errors, consistency_issues


def payload_consistency_issues(
    item: ItemPayload,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    issues = []
    expected_records = item.payload.get("n_records")
    if expected_records is not None and expected_records != len(records):
        issues.append(
            {
                "source_json_path": str(item.path),
                "field": "n_records",
                "declared": expected_records,
                "observed": len(records),
            }
        )
    expected_complete = item.payload.get("n_complete")
    observed_complete = sum(is_complete_status(record.get("status")) for record in records)
    if expected_complete is not None and expected_complete != observed_complete:
        issues.append(
            {
                "source_json_path": str(item.path),
                "field": "n_complete",
                "declared": expected_complete,
                "observed": observed_complete,
            }
        )
    return issues


def collect_hdf5_checks(items: list[ItemPayload], *, skip: bool) -> dict[str, HDF5Check]:
    checks: dict[str, HDF5Check] = {}
    for item in items:
        hdf5_path = hdf5_path_for_item(item)
        checks[str(item.path)] = check_hdf5_sidecar(hdf5_path, skip=skip)
    return checks


def hdf5_path_for_item(item: ItemPayload) -> Path:
    raw = item.payload.get("hdf5_output")
    if not raw:
        return item.path.with_suffix(".h5")
    candidate = Path(str(raw))
    if candidate.is_absolute() or candidate.exists():
        return candidate
    local_candidate = item.path.parent / candidate
    if local_candidate.exists():
        return local_candidate
    return candidate


def check_hdf5_sidecar(path: Path, *, skip: bool) -> HDF5Check:
    if skip:
        return HDF5Check(path=path, status=HDF5_SKIPPED_STATUS)
    if not path.exists():
        return HDF5Check(path=path, status="missing")
    try:
        with h5py.File(path, "r") as handle:
            schema_version = str(to_jsonable(handle.attrs.get("schema_version", "")))
            if "records" not in handle:
                return HDF5Check(
                    path=path,
                    status="missing_records_group",
                    schema_version=schema_version,
                )
            records_group = handle["records"]
            if not isinstance(records_group, h5py.Group):
                return HDF5Check(
                    path=path,
                    status="records_not_group",
                    schema_version=schema_version,
                )
            return HDF5Check(
                path=path,
                status=HDF5_OK_STATUS,
                schema_version=schema_version,
                record_count=len(records_group),
            )
    except Exception as exc:  # noqa: BLE001 - aggregation should report bad sidecars.
        return HDF5Check(
            path=path,
            status="unreadable",
            error_type=type(exc).__name__,
            error=str(exc),
        )


def base_context(item: ItemPayload, index: int, record: dict[str, Any]) -> dict[str, Any]:
    context = {
        "source_json_path": str(item.path),
        "record_index": index,
        "schema_version": item.payload.get("schema_version", ""),
        "manifest_id": item.payload.get("manifest_id", ""),
        "mode": item.payload.get("mode", record.get("diagnostic_mode", "")),
        "split": record.get("split", item.payload.get("split", "")),
        "status": record.get("status", ""),
        "git_dirty": bool(record.get("git_dirty", False))
        or str(record.get("status", "")) == "complete_dirty_git",
        "dataset": record.get("dataset", ""),
        "representation": record.get("representation", ""),
        "encoder": record.get("encoder", ""),
        "depth": record.get("depth", ""),
        "seed": record.get("seed", ""),
        "diagnostic_mode": record.get("diagnostic_mode", item.payload.get("mode", "")),
        "checkpoint": record.get("checkpoint", ""),
        "checkpoint_step": record.get("checkpoint_step", ""),
        "job_slug": record.get("job_slug", ""),
    }
    context["record_key"] = record_key(context)
    return context


def record_key(context: dict[str, Any]) -> str:
    parts = [
        context.get("manifest_id", ""),
        context.get("mode", ""),
        context.get("dataset", ""),
        context.get("representation", ""),
        context.get("encoder", ""),
        context.get("depth", ""),
        context.get("seed", ""),
        context.get("diagnostic_mode", ""),
        context.get("checkpoint", ""),
        context.get("checkpoint_step", ""),
        context.get("job_slug", ""),
    ]
    return "|".join(str(part) for part in parts)


def flatten_record_row(
    item: ItemPayload,
    record: dict[str, Any],
    context: dict[str, Any],
    hdf5_check: HDF5Check,
) -> dict[str, Any]:
    row = {field: "" for field in RECORD_FIELDS}
    row.update(context)
    row.update(
        {
            "source_hdf5_path": str(hdf5_check.path),
            "hdf5_status": hdf5_check.status,
            "hdf5_schema_version": hdf5_check.schema_version,
            "hdf5_record_count": hdf5_check.record_count,
            "diagnostic_batch_size": item.payload.get("diagnostic_batch_size", ""),
            "diagnostic_seed": item.payload.get("diagnostic_seed", ""),
            "spectral_state_max_samples": item.payload.get("spectral_state_max_samples", ""),
            "filters_json": item.payload.get("filters", ""),
            "n_diagnostic_examples": record.get("n_diagnostic_examples", ""),
            "final_projector_accuracy_on_diagnostic_batch": record.get(
                "final_projector_accuracy_on_diagnostic_batch",
                "",
            ),
            "readout_score_scale": record.get("readout_score_scale", ""),
            "n_classes": record.get("n_classes", ""),
            "input_shape_json": record.get("input_shape", ""),
            "ablation_json": record.get("ablation", ""),
            "data_seed": record.get("data_seed", ""),
            "n_train": record.get("n_train", ""),
            "run_dir": record.get("run_dir", ""),
            "checkpoint_path": record.get("checkpoint_path", ""),
            "stage_names_json": record.get("stage_names", ""),
            "error_type": record.get("error_type", ""),
            "error": record.get("error", ""),
        }
    )
    kernels = as_mapping(record.get("fidelity_kernels"))
    row.update(prefixed_kernel_columns("final_kernel", as_mapping(kernels.get("final"))))
    trajectories = as_mapping(record.get("logit_trajectories"))
    row["logit_final_accuracy"] = trajectories.get("final_accuracy", "")
    row.update(
        prefixed_summary_columns(
            "logit_total_path_length",
            as_mapping(trajectories.get("total_path_length")),
        )
    )
    row.update(
        prefixed_summary_columns(
            "logit_correct_total_path_length",
            as_mapping(trajectories.get("correct_total_path_length")),
        )
    )
    row.update(
        prefixed_summary_columns(
            "logit_incorrect_total_path_length",
            as_mapping(trajectories.get("incorrect_total_path_length")),
        )
    )
    row["logit_transition_names_json"] = trajectories.get("transition_names", "")
    row["logit_mean_step_movement_json"] = trajectories.get("mean_step_movement", "")
    spectral_state = as_mapping(record.get("hamiltonian_spectral_state"))
    row["spectral_state_status"] = spectral_state.get("status", "")
    row["spectral_state_n_samples"] = spectral_state.get("n_samples", "")
    return {field: row.get(field, "") for field in RECORD_FIELDS}


def flatten_layerwise_rows(record: dict[str, Any], context: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    probes = as_mapping(record.get("projector_probes"))
    for stage, summary in probes.items():
        rows.extend(projector_probe_rows(context, str(stage), as_mapping(summary)))
    kernels = as_mapping(record.get("fidelity_kernels"))
    for stage, summaries in kernels.items():
        rows.extend(fidelity_kernel_rows(context, str(stage), summaries))
    return rows


def projector_probe_rows(
    context: dict[str, Any],
    stage: str,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    accuracies = as_list(summary.get("accuracy_by_layer"))
    top_scores = as_list(summary.get("mean_top_score_by_layer"))
    top_score_summaries = as_list(summary.get("top_score_summary"))
    n_layers = max(len(accuracies), len(top_scores), len(top_score_summaries))
    rows = []
    for layer_index in range(n_layers):
        row = {field: "" for field in LAYERWISE_FIELDS}
        row.update(identity_columns(context))
        row.update(
            {
                "metric_family": "projector_probe",
                "stage": stage,
                "layer_index": layer_index,
                "stage_name": stage_name(stage, layer_index),
                "projector_accuracy": value_at(accuracies, layer_index),
                "projector_mean_top_score": value_at(top_scores, layer_index),
            }
        )
        top_score_summary = as_mapping(value_at(top_score_summaries, layer_index))
        row.update(prefixed_summary_columns("top_score", top_score_summary))
        rows.append({field: row.get(field, "") for field in LAYERWISE_FIELDS})
    return rows


def fidelity_kernel_rows(
    context: dict[str, Any],
    stage: str,
    summaries: Any,
) -> list[dict[str, Any]]:
    if isinstance(summaries, dict):
        iterable = [summaries]
    elif isinstance(summaries, list):
        iterable = summaries
    else:
        return []
    rows = []
    for index, raw_summary in enumerate(iterable):
        summary = as_mapping(raw_summary)
        row = {field: "" for field in LAYERWISE_FIELDS}
        layer_index = summary.get("layer_index", "" if stage == "final" else index)
        row.update(identity_columns(context))
        row.update(
            {
                "metric_family": "fidelity_kernel",
                "stage": stage,
                "layer_index": layer_index,
                "stage_name": stage_name(stage, layer_index),
            }
        )
        row.update(prefixed_kernel_columns("", summary))
        rows.append({field: row.get(field, "") for field in LAYERWISE_FIELDS})
    return rows


def flatten_cka_rows(record: dict[str, Any], context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for family, values in as_mapping(record.get("cka")).items():
        for index, value in enumerate(as_list(values)):
            row = {field: "" for field in CKA_FIELDS}
            row.update(identity_columns(context))
            row.update(
                {
                    "comparison_family": family,
                    "comparison_index": index,
                    "layer_index": index,
                    "cka_value": value,
                }
            )
            rows.append({field: row.get(field, "") for field in CKA_FIELDS})
    return rows


def flatten_spectral_state_rows(
    record: dict[str, Any],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    spectral_state = as_mapping(record.get("hamiltonian_spectral_state"))
    if not spectral_state:
        return []
    specs = (
        ("occupation_l1_change", "occupation_l1_change_by_layer", "occupation_l1_change"),
        (
            "phase_increment_abs_mean",
            "phase_abs_mean_by_layer",
            "phase_increment_abs_mean",
        ),
        ("phase_increment_abs_max", "phase_abs_max_by_layer", "phase_increment_abs_max"),
    )
    rows = []
    for metric_name, list_key, summary_key in specs:
        for index, entry in enumerate(as_list(spectral_state.get(list_key))):
            entry_mapping = as_mapping(entry)
            row = {field: "" for field in SPECTRAL_STATE_FIELDS}
            row.update(identity_columns(context))
            row.update(
                {
                    "spectral_state_status": spectral_state.get("status", ""),
                    "spectral_state_n_samples": spectral_state.get("n_samples", ""),
                    "metric": metric_name,
                    "layer_index": entry_mapping.get("layer_index", index),
                }
            )
            row.update(
                prefixed_summary_columns("value", as_mapping(entry_mapping.get(summary_key)))
            )
            rows.append({field: row.get(field, "") for field in SPECTRAL_STATE_FIELDS})
    return rows


def build_report(
    *,
    args: argparse.Namespace,
    input_dir: Path,
    json_paths: list[Path],
    items: list[ItemPayload],
    item_errors: list[dict[str, Any]],
    consistency_issues: list[dict[str, Any]],
    hdf5_checks: dict[str, HDF5Check],
    record_rows: list[dict[str, Any]],
    layerwise_rows: list[dict[str, Any]],
    cka_rows: list[dict[str, Any]],
    spectral_state_rows: list[dict[str, Any]],
    duplicate_keys: list[dict[str, Any]],
    paths_by_name: dict[str, Path],
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []

    if not json_paths:
        add_issue(errors, warnings, "no JSON item files discovered")
    for issue in item_errors:
        add_issue(errors, warnings, f"malformed JSON item: {issue['source_json_path']}", issue)
    for issue in consistency_issues:
        add_issue(
            errors,
            warnings,
            f"payload count mismatch in {issue['source_json_path']} field {issue['field']}",
            issue,
        )
    for duplicate in duplicate_keys:
        add_issue(errors, warnings, f"duplicate record key: {duplicate['record_key']}", duplicate)

    status_counts = Counter(str(row.get("status", "")) for row in record_rows)
    incomplete_rows = [
        {
            "record_key": row.get("record_key", ""),
            "source_json_path": row.get("source_json_path", ""),
            "status": row.get("status", ""),
        }
        for row in record_rows
        if not is_complete_status(row.get("status"))
    ]
    for row in incomplete_rows:
        add_issue(errors, warnings, f"incomplete diagnostic row: {row['record_key']}", row)

    schema_versions = sorted_json_values(
        {item.payload.get("schema_version", "") for item in items}
    )
    if len(schema_versions) > 1:
        add_issue(
            errors,
            warnings,
            f"mixed schema versions: {', '.join(schema_versions)}",
            {"schema_versions": schema_versions},
        )

    hdf5_status_counts = Counter(check.status for check in hdf5_checks.values())
    if not args.skip_hdf5_check:
        for check in hdf5_checks.values():
            if check.status != HDF5_OK_STATUS:
                add_issue(
                    errors,
                    warnings,
                    f"HDF5 sidecar issue {check.status}: {check.path}",
                    hdf5_check_payload(check),
                )

    expected_checks = expected_value_checks(args, record_rows)
    for check in expected_checks:
        if not check["ok"]:
            add_issue(errors, warnings, str(check["message"]), check)

    report = {
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "input_dir": str(input_dir),
        "pattern": args.pattern,
        "skip_hdf5_check": bool(args.skip_hdf5_check),
        "output_paths": {key: str(path) for key, path in paths_by_name.items()},
        "n_json_files": len(json_paths),
        "n_payloads_loaded": len(items),
        "n_item_errors": len(item_errors),
        "n_records": len(record_rows),
        "n_complete": sum(is_complete_status(row.get("status")) for row in record_rows),
        "n_layerwise_rows": len(layerwise_rows),
        "n_cka_rows": len(cka_rows),
        "n_spectral_state_rows": len(spectral_state_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "dirty_git_counts": dict(
            sorted(Counter(str(bool(row.get("git_dirty"))) for row in record_rows).items())
        ),
        "schema_versions": schema_versions,
        "mode_counts": count_field(record_rows, "mode"),
        "split_counts": count_field(record_rows, "split"),
        "diagnostic_batch_size_counts": count_payload_field(items, "diagnostic_batch_size"),
        "spectral_state_max_samples_counts": count_payload_field(
            items,
            "spectral_state_max_samples",
        ),
        "hdf5_status_counts": dict(sorted(hdf5_status_counts.items())),
        "hdf5_sidecars": [hdf5_check_payload(check) for check in hdf5_checks.values()],
        "coverage": {
            "datasets": sorted_unique(record_rows, "dataset"),
            "representations": sorted_unique(record_rows, "representation"),
            "encoders": sorted_unique(record_rows, "encoder"),
            "depths": sorted_unique(record_rows, "depth"),
            "seeds": sorted_unique(record_rows, "seed"),
            "modes": sorted_unique(record_rows, "mode"),
            "splits": sorted_unique(record_rows, "split"),
        },
        "spectral_state_presence_by_encoder": spectral_state_presence_by_encoder(record_rows),
        "item_errors": item_errors,
        "consistency_issues": consistency_issues,
        "duplicate_record_keys": duplicate_keys,
        "expected_checks": expected_checks,
        "warnings": warnings,
        "errors": errors,
        "ok": not errors,
    }
    return to_jsonable(report)


def expected_value_checks(
    args: argparse.Namespace,
    record_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    checks = []
    if args.expected_split is not None:
        checks.append(
            expected_set_check(
                "split",
                observed=sorted_unique(record_rows, "split"),
                expected=[args.expected_split],
            )
        )
    if args.expected_diagnostic_batch_size is not None:
        observed = sorted_unique(record_rows, "diagnostic_batch_size")
        checks.append(
            expected_set_check(
                "diagnostic_batch_size",
                observed=observed,
                expected=[args.expected_diagnostic_batch_size],
            )
        )
    if args.expected_spectral_state_max_samples is not None:
        observed = sorted_unique(record_rows, "spectral_state_max_samples")
        checks.append(
            expected_set_check(
                "spectral_state_max_samples",
                observed=observed,
                expected=[args.expected_spectral_state_max_samples],
            )
        )
    if args.expected_depths is not None:
        checks.append(
            expected_set_check(
                "depths",
                observed=sorted_unique(record_rows, "depth"),
                expected=args.expected_depths,
            )
        )
    return checks


def expected_set_check(name: str, *, observed: list[Any], expected: list[Any]) -> dict[str, Any]:
    observed_strings = {str(value) for value in observed if value != ""}
    expected_strings = {str(value) for value in expected}
    missing = sorted(expected_strings - observed_strings)
    unexpected = sorted(observed_strings - expected_strings)
    ok = not missing and not unexpected
    return {
        "name": name,
        "ok": ok,
        "observed": observed,
        "expected": expected,
        "missing": missing,
        "unexpected": unexpected,
        "message": (
            f"expected {name} mismatch: missing={missing} unexpected={unexpected}"
            if not ok
            else ""
        ),
    }


def add_issue(
    errors: list[dict[str, Any]],
    warnings: list[str],
    message: str,
    payload: dict[str, Any] | None = None,
) -> None:
    warnings.append(message)
    errors.append({"message": message, **(payload or {})})


def identity_columns(context: dict[str, Any]) -> dict[str, Any]:
    return {field: context.get(field, "") for field in IDENTITY_FIELDS}


def prefixed_kernel_columns(prefix: str, summary: dict[str, Any]) -> dict[str, Any]:
    columns = {
        column_name(prefix, "target_alignment"): summary.get("target_alignment", ""),
        column_name(prefix, "effective_rank"): summary.get("effective_rank", ""),
        column_name(prefix, "centered_effective_rank"): summary.get(
            "centered_effective_rank",
            "",
        ),
    }
    distributions = as_mapping(summary.get("fidelity_distributions"))
    columns.update(
        prefixed_summary_columns(
            column_name(prefix, "within"),
            as_mapping(distributions.get("within")),
        )
    )
    columns.update(
        prefixed_summary_columns(
            column_name(prefix, "between"),
            as_mapping(distributions.get("between")),
        )
    )
    columns[column_name(prefix, "mean_gap_within_minus_between")] = distributions.get(
        "mean_gap_within_minus_between",
        "",
    )
    return columns


def prefixed_summary_columns(prefix: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{field}": summary.get(field, "") for field in SUMMARY_FIELDS}


def column_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}" if prefix else suffix


def stage_name(stage: str, layer_index: Any) -> str:
    if stage == "initial":
        return "initial"
    if stage == "post_upload":
        return f"layer_{layer_index}_post_upload"
    if stage == "post_mixer":
        return f"layer_{layer_index}_post_mixer"
    return stage


def as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def value_at(values: list[Any], index: int) -> Any:
    return values[index] if index < len(values) else ""


def count_field(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))


def count_payload_field(items: list[ItemPayload], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(item.payload.get(field, "")) for item in items).items()))


def sorted_unique(rows: list[dict[str, Any]], field: str) -> list[Any]:
    values = {row.get(field, "") for row in rows}
    return sorted_json_values(values)


def sorted_json_values(values: set[Any]) -> list[Any]:
    return sorted((to_jsonable(value) for value in values), key=sort_key)


def sort_key(value: Any) -> tuple[int, float | str, str]:
    if value == "":
        return (2, "", "")
    try:
        return (0, float(value), str(value))
    except (TypeError, ValueError):
        return (1, str(value), str(value))


def spectral_state_presence_by_encoder(
    record_rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for row in record_rows:
        encoder = str(row.get("encoder", ""))
        present = "present" if row.get("spectral_state_status") else "absent"
        counts.setdefault(encoder, Counter())[present] += 1
    return {encoder: dict(sorted(counter.items())) for encoder, counter in sorted(counts.items())}


def hdf5_check_payload(check: HDF5Check) -> dict[str, Any]:
    return {
        "path": str(check.path),
        "status": check.status,
        "schema_version": check.schema_version,
        "record_count": check.record_count,
        "error_type": check.error_type,
        "error": check.error,
    }


def is_complete_status(status: Any) -> bool:
    return str(status or "").startswith("complete")


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fields})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def csv_value(value: Any) -> Any:
    value = to_jsonable(value)
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, allow_nan=False)
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(inner) for inner in value]
    return value


if __name__ == "__main__":
    main()
