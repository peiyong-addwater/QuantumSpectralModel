#!/usr/bin/env python3
"""Merge per-item gradient diagnostics JSON files into paper-facing suites."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ham_embed_spectral.experiments.manifest import sanitize_slug  # noqa: E402
from ham_embed_spectral.utils.checkpointing import write_json  # noqa: E402

DEFAULT_INPUT_DIR = "results/tables/gradient_diagnostics/batched/items"
DEFAULT_OUTPUT_DIR = "results/tables/gradient_diagnostics"
DEFAULT_MANIFEST_IDS = ("pendigits", "synthetic")
DEFAULT_MODES = ("init", "checkpoints", "final")
SETTING_FIELDS = (
    "diagnostic_batch_size",
    "diagnostic_seed",
    "near_zero_tol",
    "n_init_seeds",
    "fisher_batch_size",
    "filters",
)


@dataclass(frozen=True)
class ItemDiagnostics:
    """One per-item diagnostics payload loaded from disk."""

    path: Path
    payload: dict[str, Any]

    @property
    def manifest_id(self) -> str:
        return str(self.payload["manifest_id"])

    @property
    def mode(self) -> str:
        return str(self.payload["mode"])

    @property
    def records(self) -> list[dict[str, Any]]:
        records = self.payload.get("records", [])
        if not isinstance(records, list):
            raise ValueError(f"{self.path} field 'records' must be a list")
        if not all(isinstance(record, dict) for record in records):
            raise ValueError(f"{self.path} field 'records' must contain only objects")
        return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-ids", nargs="+", default=list(DEFAULT_MANIFEST_IDS))
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=list(DEFAULT_MODES))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned merged files without writing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    item_payloads = load_item_diagnostics(Path(args.input_dir))
    groups = group_item_diagnostics(item_payloads)
    outputs = merged_outputs(groups, args)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "input_dir": args.input_dir,
                    "output_dir": args.output_dir,
                    "n_input_files": len(item_payloads),
                    "outputs": {
                        str(path): {
                            "manifest_id": payload["manifest_id"],
                            "mode": payload["mode"],
                            "n_records": payload["n_records"],
                            "n_complete": payload["n_complete"],
                        }
                        for path, payload in outputs.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    for path, payload in outputs.items():
        write_json(path, payload)
        print(path)


def load_item_diagnostics(input_dir: Path) -> list[ItemDiagnostics]:
    """Load per-item diagnostics JSON files from a flat items directory."""

    if not input_dir.exists():
        return []
    items = []
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object")
        if "manifest_id" not in payload or "mode" not in payload:
            raise ValueError(f"{path} missing required manifest_id/mode fields")
        items.append(ItemDiagnostics(path=path, payload=payload))
    return items


def group_item_diagnostics(
    items: Iterable[ItemDiagnostics],
) -> dict[tuple[str, str], list[ItemDiagnostics]]:
    groups: dict[tuple[str, str], list[ItemDiagnostics]] = {}
    for item in items:
        groups.setdefault((item.manifest_id, item.mode), []).append(item)
    return groups


def merged_outputs(
    groups: dict[tuple[str, str], list[ItemDiagnostics]],
    args: argparse.Namespace,
) -> dict[Path, dict[str, Any]]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    validate_output_dir(input_dir, output_dir)
    manifest_ids = ordered_union(
        args.manifest_ids,
        (manifest_id for manifest_id, _ in groups),
    )
    modes = list(args.modes)
    outputs: dict[Path, dict[str, Any]] = {}
    for manifest_id in manifest_ids:
        for mode in modes:
            payload = merge_group(manifest_id, mode, groups.get((manifest_id, mode), []), args)
            outputs[output_path(output_dir, manifest_id, mode)] = payload
    return outputs


def validate_output_dir(input_dir: Path, output_dir: Path) -> None:
    """Reject aggregation outputs that would be written into per-item inputs."""

    input_path = input_dir.expanduser().resolve(strict=False)
    output_path = output_dir.expanduser().resolve(strict=False)
    if output_path == input_path:
        raise ValueError("--output-dir must be outside --input-dir")
    try:
        output_path.relative_to(input_path)
    except ValueError:
        return
    raise ValueError("--output-dir must be outside --input-dir")


def ordered_union(first: Iterable[str], second: Iterable[str]) -> list[str]:
    seen = set()
    values = []
    for value in [*first, *sorted(second)]:
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def output_path(output_dir: Path, manifest_id: str, mode: str) -> Path:
    return output_dir / f"{sanitize_slug(manifest_id)}_{mode}.json"


def merge_group(
    manifest_id: str,
    mode: str,
    items: list[ItemDiagnostics],
    args: argparse.Namespace,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    settings, conflicts = collect_settings(items)
    for item in sorted(items, key=lambda current: str(current.path)):
        records.extend(item.records)

    payload: dict[str, Any] = {
        "manifest_id": manifest_id,
        "mode": mode,
        "source_input_dir": str(args.input_dir),
        "n_items": len(items),
        "n_records": len(records),
        "n_complete": sum(
            str(record.get("status", "")).startswith("complete") for record in records
        ),
        "records": records,
        "aggregates": [],
        "aggregate_note": (
            "Intentionally empty: per-item diagnostics JSON files do not persist the "
            "raw gradient vectors needed to recompute exact cross-item aggregates."
        ),
    }
    for field in SETTING_FIELDS:
        if field in settings:
            payload[field] = settings[field]
    if conflicts:
        payload["setting_conflicts"] = conflicts
    return payload


def collect_settings(
    items: list[ItemDiagnostics],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    settings: dict[str, Any] = {}
    conflicts = []
    for item in sorted(items, key=lambda current: str(current.path)):
        for field in SETTING_FIELDS:
            if field not in item.payload:
                continue
            value = item.payload[field]
            if field not in settings:
                settings[field] = value
            elif settings[field] != value:
                conflicts.append(
                    {
                        "field": field,
                        "first_value": settings[field],
                        "conflicting_value": value,
                        "path": str(item.path),
                    }
                )
    return settings, conflicts


if __name__ == "__main__":
    main()
