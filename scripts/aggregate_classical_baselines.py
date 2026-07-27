#!/usr/bin/env python3
"""Aggregate classical baseline JSON files into CSV and summary JSON."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ham_embed_spectral.naming import canonical_descriptor_name  # noqa: E402
from ham_embed_spectral.utils.checkpointing import write_json  # noqa: E402

CSV_FIELDS = (
    "status",
    "manifest_id",
    "job_slug",
    "path",
    "dataset",
    "representation",
    "descriptor",
    "classifier",
    "feature_set",
    "feature_kind",
    "seed",
    "data_seed",
    "learning_rate",
    "steps",
    "bins",
    "standardize",
    "n_train",
    "n_validation",
    "n_test",
    "feature_dim",
    "mlp_hidden_width",
    "classifier_parameter_count",
    "target_parameter_count",
    "target_parameter_source",
    "target_parameter_relative_error",
    "classifier_n_iter",
    "validation_loss",
    "validation_accuracy",
    "test_loss",
    "test_accuracy",
    "wall_time_seconds",
    "git_commit",
    "git_dirty",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", default="results/runs/classical_baseline")
    parser.add_argument("--output-dir", default="results/tables")
    parser.add_argument("--name", default="classical_baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(scan_outputs(input_root))
    csv_path = output_dir / f"{args.name}_runs.csv"
    summary_path = output_dir / f"{args.name}_summary.json"
    write_csv(csv_path, rows)
    write_json(
        summary_path,
        {
            "input_root": str(input_root),
            "n_rows": len(rows),
            "n_complete": sum(row["status"] == "complete" for row in rows),
            "groups": group_summary(rows),
            "csv_path": str(csv_path),
        },
    )
    print(csv_path)
    print(summary_path)


def scan_outputs(input_root: Path):
    if not input_root.exists():
        return
    for path in sorted(input_root.glob("**/*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            row = {field: "" for field in CSV_FIELDS}
            row.update({"status": "invalid_json", "path": str(path)})
            yield row
            continue
        yield row_from_payload(path, payload)


def row_from_payload(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    args = payload.get("args", {})
    dataset = payload.get("dataset", {})
    validation = payload.get("validation", {})
    test = payload.get("test", {})
    git = payload.get("git", {})
    descriptor = canonical_descriptor_name(
        payload.get("descriptor") or args.get("descriptor", "")
    )
    row = {
        "status": payload.get("status", ""),
        "manifest_id": payload.get("manifest_id") or args.get("manifest_id", ""),
        "job_slug": payload.get("job_slug") or args.get("job_slug", path.stem),
        "path": str(path),
        "dataset": args.get("dataset", ""),
        "representation": dataset.get("representation") or args.get("representation", ""),
        "descriptor": descriptor,
        "classifier": payload.get("classifier") or args.get("classifier", "linear-softmax"),
        "feature_set": payload.get("feature_set") or args.get("feature_set", "values-plus-gaps"),
        "feature_kind": payload.get("feature_kind") or args.get("feature_kind", ""),
        "seed": args.get("seed", ""),
        "data_seed": payload.get("data_seed", ""),
        "learning_rate": args.get("learning_rate", ""),
        "steps": args.get("steps", ""),
        "bins": args.get("bins", ""),
        "standardize": args.get("standardize", ""),
        "n_train": dataset.get("n_train", ""),
        "n_validation": dataset.get("n_validation", ""),
        "n_test": dataset.get("n_test", ""),
        "feature_dim": payload.get("feature_dim", ""),
        "mlp_hidden_width": payload.get("mlp_hidden_width", ""),
        "classifier_parameter_count": payload.get("classifier_parameter_count", ""),
        "target_parameter_count": payload.get("target_parameter_count", ""),
        "target_parameter_source": payload.get("target_parameter_source", ""),
        "target_parameter_relative_error": payload.get("target_parameter_relative_error", ""),
        "classifier_n_iter": payload.get("classifier_n_iter", ""),
        "validation_loss": validation.get("loss", ""),
        "validation_accuracy": validation.get("accuracy", ""),
        "test_loss": test.get("loss", ""),
        "test_accuracy": test.get("accuracy", ""),
        "wall_time_seconds": payload.get("wall_time_seconds", ""),
        "git_commit": git.get("commit", ""),
        "git_dirty": git.get("dirty", ""),
    }
    return {field: clean_cell(row.get(field, "")) for field in CSV_FIELDS}


def clean_cell(value: Any) -> Any:
    return "" if value is None else value


def group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "complete":
            grouped[
                (
                    row["dataset"],
                    row["representation"],
                    row["descriptor"],
                    row["classifier"],
                    row["feature_set"],
                )
            ].append(row)

    summary = []
    for (dataset, representation, descriptor, classifier, feature_set), group in sorted(
        grouped.items()
    ):
        summary.append(
            {
                "dataset": dataset,
                "representation": representation,
                "descriptor": descriptor,
                "classifier": classifier,
                "feature_set": feature_set,
                "n": len(group),
                "feature_dim": constant_of(group, "feature_dim"),
                "mlp_hidden_width": constant_of(group, "mlp_hidden_width"),
                "classifier_parameter_count": constant_of(
                    group, "classifier_parameter_count"
                ),
                "target_parameter_count": constant_of(group, "target_parameter_count"),
                "target_parameter_source": constant_of(group, "target_parameter_source"),
                "target_parameter_relative_error": constant_of(
                    group, "target_parameter_relative_error"
                ),
                "validation_accuracy_mean": mean_of(group, "validation_accuracy"),
                "validation_accuracy_std": stdev_of(group, "validation_accuracy"),
                "test_accuracy_mean": mean_of(group, "test_accuracy"),
                "test_accuracy_std": stdev_of(group, "test_accuracy"),
                "test_loss_mean": mean_of(group, "test_loss"),
                "test_loss_std": stdev_of(group, "test_loss"),
            }
        )
    return summary


def constant_of(rows: list[dict[str, Any]], key: str) -> Any:
    values = {row.get(key, "") for row in rows if row.get(key, "") != ""}
    if len(values) == 1:
        return next(iter(values))
    return ""


def mean_of(rows: list[dict[str, Any]], key: str) -> float | str:
    values = numeric_values(rows, key)
    return statistics.fmean(values) if values else ""


def stdev_of(rows: list[dict[str, Any]], key: str) -> float | str:
    values = numeric_values(rows, key)
    return statistics.stdev(values) if len(values) >= 2 else ""


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        values.append(float(value))
    return values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
