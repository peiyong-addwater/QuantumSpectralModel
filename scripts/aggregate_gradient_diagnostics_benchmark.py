#!/usr/bin/env python3
"""Aggregate gradient diagnostics resource benchmark CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gradient_diagnostics_resource_benchmark import (  # noqa: E402
    CSV_FIELDS,
    group_summaries,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default="results/tables/gradient_diagnostics/resource_benchmarks",
    )
    parser.add_argument(
        "--output-prefix",
        default="results/tables/gradient_diagnostics/resource_benchmarks/combined",
    )
    parser.add_argument("--pattern", default="*_runs.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_prefix = Path(args.output_prefix)
    csv_path = output_prefix.with_name(f"{output_prefix.name}_runs.csv")
    json_path = output_prefix.with_name(f"{output_prefix.name}_report.json")
    paths = discover_csvs(input_dir, args.pattern, csv_path)
    rows = read_csv_rows(paths)
    write_combined_reports(csv_path, json_path, input_dir, paths, rows)
    print(f"combined_gradient_resource_csv={csv_path}")
    print(f"combined_gradient_resource_json={json_path}")


def discover_csvs(input_dir: Path, pattern: str, output_csv: Path) -> list[Path]:
    paths = []
    output_csv_resolved = output_csv.resolve()
    for path in sorted(input_dir.rglob(pattern)):
        if path.resolve() == output_csv_resolved:
            continue
        paths.append(path)
    return paths


def read_csv_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row.setdefault("source_csv_path", str(path))
                rows.append(row)
    return rows


def write_combined_reports(
    csv_path: Path,
    json_path: Path,
    input_dir: Path,
    paths: list[Path],
    rows: list[dict[str, str]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_with_extra_fields(csv_path, rows)
    report = {
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "input_dir": str(input_dir),
        "n_input_csvs": len(paths),
        "input_csvs": [str(path) for path in paths],
        "n_rows": len(rows),
        "n_complete": sum(row.get("status") == "complete" for row in rows),
        "n_failed": sum(row.get("status") == "failed" for row in rows),
        "groups": group_summaries(rows),
        "csv_path": str(csv_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def write_csv_with_extra_fields(path: Path, rows: list[dict[str, str]]) -> None:
    extra_fields = sorted({field for row in rows for field in row} - set(CSV_FIELDS))
    if extra_fields:
        fieldnames = [*CSV_FIELDS, *extra_fields]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
    else:
        write_csv(path, rows)


if __name__ == "__main__":
    main()
