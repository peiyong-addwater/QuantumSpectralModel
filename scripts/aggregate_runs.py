#!/usr/bin/env python3
"""Aggregate training run JSON files into paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ham_embed_spectral.experiments.manifest import jobs_from_manifest, load_manifest  # noqa: E402
from ham_embed_spectral.naming import (  # noqa: E402
    ENCODER_CLI_CHOICES,
    canonical_encoder_name,
    canonical_slug_aliases,
)
from ham_embed_spectral.utils.checkpointing import write_json  # noqa: E402

CSV_FIELDS = (
    "status",
    "manifest_id",
    "job_slug",
    "run_dir",
    "dataset",
    "representation",
    "encoder",
    "depth",
    "seed",
    "learning_rate",
    "batch_size",
    "steps",
    "standardize",
    "initial_state",
    "projector_renormalize",
    "track_readout_leakage",
    "final_validation_loss",
    "final_validation_accuracy",
    "final_test_loss",
    "final_test_accuracy",
    "best_validation_accuracy",
    "validation_accuracy_auc",
    "time_to_target_step",
    "wall_time_seconds",
    "mean_readout_leakage_mass",
    "parameter_count",
    "n_qubits",
    "hilbert_dim",
    "git_commit",
    "git_dirty",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=None,
        help="JSON experiment manifest for expected jobs.",
    )
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Root containing experiment run directories.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment directory under --runs-root.",
    )
    parser.add_argument("--output-dir", default="results/tables")
    parser.add_argument("--encoders", nargs="+", choices=ENCODER_CLI_CHOICES, default=None)
    parser.add_argument("--target-validation-accuracy", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest) if args.manifest else None
    manifest_id = manifest["manifest_id"] if manifest else "all_runs"
    runs_root = Path(
        args.runs_root or manifest_value(manifest, "outputs", "output_root", "results/runs")
    )
    experiment_name = args.experiment_name or manifest_value(
        manifest, "outputs", "experiment_name", None
    )
    scan_root = runs_root / experiment_name if experiment_name else runs_root
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_filter = canonical_encoder_filter(args.encoders)
    rows = [
        row
        for row in scan_runs(scan_root, args, manifest_id)
        if row_matches_encoder(row, encoder_filter)
    ]
    rows_by_slug = {
        canonical_slug_aliases(row["job_slug"]): row for row in rows if row["job_slug"]
    }

    missing = []
    if manifest is not None:
        for job in jobs_from_manifest(manifest, encoders=args.encoders):
            if job.slug not in rows_by_slug:
                missing.append(job.slug)
                rows.append(missing_row(job, manifest_id, manifest))

    csv_path = output_dir / f"{manifest_id}_runs.csv"
    json_path = output_dir / f"{manifest_id}_summary.json"
    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "manifest_id": manifest_id,
            "scan_root": str(scan_root),
            "n_rows": len(rows),
            "n_complete": sum(row["status"] == "complete" for row in rows),
            "n_missing": len(missing),
            "selected_encoders": sorted(encoder_filter) if encoder_filter is not None else None,
            "missing_job_slugs": missing,
            "csv_path": str(csv_path),
        },
    )
    print(csv_path)
    print(json_path)


def canonical_encoder_filter(encoders: list[str] | None) -> set[str] | None:
    if encoders is None:
        return None
    return {canonical_encoder_name(value) for value in encoders}


def row_matches_encoder(row: dict[str, Any], encoder_filter: set[str] | None) -> bool:
    if encoder_filter is None:
        return True
    return row.get("encoder") in encoder_filter


def manifest_value(
    manifest: dict[str, Any] | None,
    section: str,
    key: str,
    default: Any,
) -> Any:
    if manifest is None:
        return default
    value = manifest.get(section, {})
    return value.get(key, default) if isinstance(value, dict) else default


def scan_runs(scan_root: Path, args: argparse.Namespace, manifest_id: str):
    if not scan_root.exists():
        return
    for config_path in sorted(scan_root.glob("*/config.json")):
        run_dir = config_path.parent
        try:
            config = json.loads(config_path.read_text())
            metrics = load_metrics(run_dir)
        except json.JSONDecodeError:
            yield {"status": "invalid_json", "run_dir": str(run_dir), "manifest_id": manifest_id}
            continue
        yield row_from_run(run_dir, config, metrics, args, manifest_id)


def load_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        metrics["_source"] = "metrics.json"
        return metrics

    jsonl_path = run_dir / "metrics.jsonl"
    if not jsonl_path.exists():
        return {}

    history = []
    summary = {}
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        event_type = event.get("event")
        if event_type == "history":
            history.append(
                {
                    key: value
                    for key, value in event.items()
                    if key not in {"event", "timestamp_utc", "seed", "manifest_id", "job_slug"}
                }
            )
        elif event_type == "summary":
            summary = {
                key: value
                for key, value in event.items()
                if key not in {"event", "timestamp_utc"}
            }
    return {"history": history, "summary": summary, "_source": "metrics.jsonl"}


def row_from_run(
    run_dir: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    args: argparse.Namespace,
    manifest_id: str,
) -> dict[str, Any]:
    run_args = config.get("args", {})
    model = config.get("model_config", {})
    training = config.get("training", {})
    dataset = config.get("dataset", {})
    git = config.get("git", {})
    summary = metrics.get("summary", {})
    history = metrics.get("history", [])
    final_validation = summary.get("final_validation", {})
    final_test = summary.get("final_test", {})
    target = args.target_validation_accuracy

    if summary.get("completed") and final_test:
        status = "complete"
    elif history:
        status = "partial"
    else:
        status = "incomplete"

    row = {
        "status": status,
        "manifest_id": config.get("manifest_id") or summary.get("manifest_id") or manifest_id,
        "job_slug": config.get("job_slug") or summary.get("job_slug") or run_args.get("job_slug"),
        "run_dir": str(run_dir),
        "dataset": run_args.get("dataset"),
        "representation": dataset.get("representation") or run_args.get("representation"),
        "encoder": canonical_encoder_name(run_args.get("encoder", "")),
        "depth": model.get("reupload_depth"),
        "seed": config.get("seed") or summary.get("seed"),
        "learning_rate": training.get("learning_rate") or run_args.get("learning_rate"),
        "batch_size": training.get("batch_size") or run_args.get("batch_size"),
        "steps": training.get("steps") or run_args.get("steps"),
        "standardize": run_args.get("standardize"),
        "initial_state": model.get("initial_state"),
        "projector_renormalize": model.get("projector_renormalize"),
        "track_readout_leakage": model.get("track_readout_leakage"),
        "final_validation_loss": final_validation.get("loss"),
        "final_validation_accuracy": final_validation.get("accuracy"),
        "final_test_loss": final_test.get("loss"),
        "final_test_accuracy": final_test.get("accuracy"),
        "best_validation_accuracy": best_metric(history, "validation_accuracy"),
        "validation_accuracy_auc": auc_metric(history, "validation_accuracy"),
        "time_to_target_step": time_to_target(history, "validation_accuracy", target),
        "wall_time_seconds": summary.get("wall_time_seconds"),
        "mean_readout_leakage_mass": final_test.get("mean_readout_leakage_mass"),
        "parameter_count": model.get("parameter_count"),
        "n_qubits": model.get("n_qubits"),
        "hilbert_dim": model.get("hilbert_dim"),
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
    }
    return {field: row.get(field, "") for field in CSV_FIELDS}


def missing_row(job, manifest_id: str, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "status": "missing",
            "manifest_id": manifest_id,
            "job_slug": job.slug,
            "dataset": job.dataset,
            "representation": job.representation,
            "encoder": job.encoder,
            "depth": job.reupload_depth,
            "seed": job.seed,
            "learning_rate": job.learning_rate,
            "batch_size": job.batch_size,
            "initial_state": manifest_value(manifest, "model", "initial_state", ""),
        }
    )
    return row


def best_metric(history: list[dict[str, Any]], key: str) -> float | str:
    values = [record[key] for record in history if key in record]
    return max(values) if values else ""


def auc_metric(history: list[dict[str, Any]], key: str) -> float | str:
    points = [
        (record["step"], record[key])
        for record in history
        if "step" in record and key in record
    ]
    if len(points) < 2:
        return ""
    points.sort()
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        area += 0.5 * (float(y0) + float(y1)) * (float(x1) - float(x0))
    return area


def time_to_target(
    history: list[dict[str, Any]],
    key: str,
    target: float | None,
) -> int | str:
    if target is None:
        return ""
    for record in sorted(history, key=lambda item: item.get("step", 0)):
        if record.get(key, float("-inf")) >= target:
            return int(record["step"])
    return ""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
