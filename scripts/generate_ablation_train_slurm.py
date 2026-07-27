#!/usr/bin/env python3
"""Generate batched SLURM scripts for training matrix-ablation grids."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ham_embed_spectral.experiments.manifest import (  # noqa: E402
    jobs_from_manifest,
    load_manifest,
    sanitize_slug,
    validate_manifest,
)
from ham_embed_spectral.naming import (  # noqa: E402
    CANONICAL_BLOCK_HAMILTONIAN,
    CANONICAL_SYMMETRIC_HAMILTONIAN,
    canonical_encoder_name,
)

DEFAULT_MANIFESTS = {
    "pendigits": ROOT / "configs/experiments/pendigits.json",
    "synthetic": ROOT / "configs/experiments/synthetic.json",
}
DEFAULT_TIMING_CSV = ROOT / "results/timing_benchmarks/ablation_timing_memory_runs.csv"
DEFAULT_MANIFEST_OUTPUT_DIR = ROOT / "results/tables/ablation_manifests"
DEFAULT_ABLATION_SEED = 0

PERMUTATION_ABLATIONS = ("entry-permutation", "row-column-permutation")
SYMMETRIC_ABLATIONS = ("spectrum-only", "eigenvector-only")
BLOCK_ABLATIONS = ("singular-spectrum-only", "singular-vector-only")
DEFAULT_ABLATIONS = PERMUTATION_ABLATIONS + SYMMETRIC_ABLATIONS + BLOCK_ABLATIONS

ABLATION_JOB_PREFIXES = {
    "entry-permutation": "entry_perm",
    "row-column-permutation": "rowcol_perm",
    "spectrum-only": "spec_only",
    "eigenvector-only": "eigvec_only",
    "singular-spectrum-only": "sing_spec",
    "singular-vector-only": "sing_vec",
}


@dataclass(frozen=True)
class DerivedManifest:
    """One generated ablation manifest and the command delegated to train SLURM generation."""

    base_manifest_path: Path
    ablation: str
    path: Path
    payload: dict[str, Any]
    n_jobs: int
    command: list[str]


@dataclass(frozen=True)
class GenerationResult:
    """Result summary for one delegated train-SLURM invocation."""

    derived: DerivedManifest
    stdout: str
    dry_run_payload: dict[str, Any] | None
    generated_slurm_paths: tuple[str, ...]

    @property
    def n_batches(self) -> int:
        if self.dry_run_payload is not None:
            return int(self.dry_run_payload.get("n_batches", 0))
        return len(self.generated_slurm_paths)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifests",
        nargs="+",
        default=list(DEFAULT_MANIFESTS),
        help="Base manifest aliases ('pendigits', 'synthetic') or JSON manifest paths.",
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=DEFAULT_ABLATIONS,
        default=list(DEFAULT_ABLATIONS),
        help="Ablations to generate. Encoder filtering is applied per ablation.",
    )
    parser.add_argument("--timing-csv", default=str(DEFAULT_TIMING_CSV))
    parser.add_argument("--manifest-output-dir", default=str(DEFAULT_MANIFEST_OUTPUT_DIR))
    parser.add_argument("--ablation-seed", type=int, default=DEFAULT_ABLATION_SEED)
    parser.add_argument("--batch-time", default="24:00:00")
    parser.add_argument("--batch-mem", default="16GB")
    parser.add_argument("--batch-safety-margin", type=float, default=2.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ask generate_train_slurm.py for batch dry-run payloads; manifests are still written.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Forward --submit to generate_train_slurm.py after scripts are generated.",
    )
    parser.add_argument(
        "--skip-submitted",
        action="store_true",
        help="Forward --skip-submitted to generate_train_slurm.py.",
    )
    return normalize_args(parser.parse_args(argv))


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.dry_run and args.submit:
        raise ValueError("--dry-run cannot be combined with --submit")
    if args.batch_safety_margin <= 0:
        raise ValueError("--batch-safety-margin must be positive")
    invalid = [value for value in args.ablations if value not in DEFAULT_ABLATIONS]
    if invalid:
        expected = ", ".join(DEFAULT_ABLATIONS)
        raise ValueError(f"invalid ablation(s) {invalid}; expected one of {expected}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results = generate_ablation_jobs(args)
    print(json.dumps(summary_payload(results, args), indent=2, sort_keys=True))


def generate_ablation_jobs(args: argparse.Namespace) -> list[GenerationResult]:
    """Write derived manifests and delegate script generation for each one."""

    output_dir = Path(args.manifest_output_dir)
    results: list[GenerationResult] = []
    for base_path in resolve_manifest_paths(args.manifests):
        base_manifest = load_manifest(base_path)
        for ablation in args.ablations:
            payload = derived_manifest_payload(
                base_manifest,
                ablation=ablation,
                ablation_seed=args.ablation_seed,
            )
            manifest_path = derived_manifest_path(payload, ablation, output_dir)
            write_manifest(manifest_path, payload)
            command = build_generator_command(manifest_path, args)
            derived = DerivedManifest(
                base_manifest_path=base_path,
                ablation=ablation,
                path=manifest_path,
                payload=payload,
                n_jobs=len(jobs_from_manifest(payload)),
                command=command,
            )
            stdout = run_generator(command)
            results.append(
                GenerationResult(
                    derived=derived,
                    stdout=stdout,
                    dry_run_payload=parse_dry_run_payload(stdout) if args.dry_run else None,
                    generated_slurm_paths=parse_generated_slurm_paths(stdout),
                )
            )
    return results


def resolve_manifest_paths(values: list[str]) -> list[Path]:
    """Resolve default aliases and user-provided paths to existing manifest files."""

    paths = []
    for value in values:
        if value in DEFAULT_MANIFESTS:
            path = DEFAULT_MANIFESTS[value]
        else:
            path = Path(value)
            if not path.is_absolute() and not path.exists():
                path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"manifest not found: {value}")
        paths.append(path)
    return paths


def derived_manifest_payload(
    base_manifest: dict[str, Any],
    *,
    ablation: str,
    ablation_seed: int,
) -> dict[str, Any]:
    """Return a valid ablation-specific manifest derived from a base manifest."""

    payload = copy.deepcopy({key: value for key, value in base_manifest.items() if key != "_path"})
    grid = payload["grid"]
    grid["encoders"] = valid_encoders_for_ablation(grid["encoders"], ablation)
    if not grid["encoders"]:
        raise ValueError(
            f"manifest {payload['manifest_id']!r} has no encoders compatible with {ablation!r}"
        )

    base_experiment = str(payload.get("outputs", {}).get("experiment_name", payload["manifest_id"]))
    ablation_slug = sanitize_slug(ablation)
    experiment_name = f"{base_experiment}_ablation_{ablation_slug}"

    outputs = payload.setdefault("outputs", {})
    outputs["experiment_name"] = experiment_name

    slurm = payload.setdefault("slurm", {})
    slurm["slurm_dir"] = f"slurm/{experiment_name}"
    slurm["job_name_prefix"] = ablation_job_name_prefix(base_experiment, ablation)

    ablations = payload.setdefault("ablations", {})
    ablations["ablation"] = ablation
    ablations["ablation_seed"] = int(ablation_seed)

    validate_manifest(payload)
    return payload


def valid_encoders_for_ablation(encoders: list[str], ablation: str) -> list[str]:
    """Return the manifest encoder subset compatible with a named ablation."""

    canonical = [canonical_encoder_name(value) for value in encoders]
    if ablation in PERMUTATION_ABLATIONS:
        return canonical
    if ablation in SYMMETRIC_ABLATIONS:
        return [value for value in canonical if value == CANONICAL_SYMMETRIC_HAMILTONIAN]
    if ablation in BLOCK_ABLATIONS:
        return [value for value in canonical if value == CANONICAL_BLOCK_HAMILTONIAN]
    raise ValueError(f"unsupported ablation {ablation!r}")


def ablation_job_name_prefix(base_experiment: str, ablation: str) -> str:
    """Return a compact SLURM job-name prefix for an ablation grid."""

    base = sanitize_slug(base_experiment)[:14].strip("_") or "manifest"
    ablation_part = ABLATION_JOB_PREFIXES[ablation]
    return sanitize_slug(f"qfm_{base}_abl_{ablation_part}")[:64]


def derived_manifest_path(
    payload: dict[str, Any],
    ablation: str,
    output_dir: Path,
) -> Path:
    manifest_id = sanitize_slug(str(payload["manifest_id"]))
    return output_dir / f"{manifest_id}__{sanitize_slug(ablation)}.json"


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_generator_command(manifest_path: Path, args: argparse.Namespace) -> list[str]:
    """Build the delegated generate_train_slurm.py command."""

    command = [
        "uv",
        "run",
        "scripts/generate_train_slurm.py",
        "--manifest",
        str(manifest_path),
        "--batch-from-timing",
        str(args.timing_csv),
        "--batch-time",
        args.batch_time,
        "--batch-mem",
        args.batch_mem,
        "--batch-safety-margin",
        f"{args.batch_safety_margin:g}",
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.submit:
        command.append("--submit")
    if args.skip_submitted:
        command.append("--skip-submitted")
    return command


def run_generator(command: list[str]) -> str:
    """Run the delegated train SLURM generator and return captured stdout."""

    env = os.environ.copy()
    platforms = [value.strip().lower() for value in env.get("JAX_PLATFORMS", "").split(",")]
    if platforms and platforms[0] == "cpu":
        env.pop("JAX_PLATFORMS")
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.stdout


def parse_dry_run_payload(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("generate_train_slurm.py did not print a JSON dry-run payload") from exc
    if not isinstance(payload, dict):
        raise ValueError("generate_train_slurm.py dry-run payload must be a JSON object")
    return payload


def parse_generated_slurm_paths(stdout: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in stdout.splitlines() if line.strip().endswith(".slurm"))


def summary_payload(results: list[GenerationResult], args: argparse.Namespace) -> dict[str, Any]:
    """Return a concise JSON-serializable generation summary."""

    return {
        "dry_run": bool(args.dry_run),
        "submit": bool(args.submit),
        "skip_submitted": bool(args.skip_submitted),
        "timing_csv": str(args.timing_csv),
        "manifest_output_dir": str(args.manifest_output_dir),
        "n_derived_manifests": len(results),
        "total_jobs": sum(result.derived.n_jobs for result in results),
        "estimated_batches": sum(result.n_batches for result in results),
        "derived_manifests": [manifest_summary(result) for result in results],
    }


def manifest_summary(result: GenerationResult) -> dict[str, Any]:
    payload = result.derived.payload
    slurm = payload.get("slurm", {})
    outputs = payload.get("outputs", {})
    summary: dict[str, Any] = {
        "base_manifest": str(result.derived.base_manifest_path),
        "manifest": str(result.derived.path),
        "manifest_id": payload["manifest_id"],
        "ablation": result.derived.ablation,
        "encoders": payload["grid"]["encoders"],
        "experiment_name": outputs.get("experiment_name"),
        "slurm_dir": slurm.get("slurm_dir"),
        "job_name_prefix": slurm.get("job_name_prefix"),
        "n_jobs": result.derived.n_jobs,
        "n_batches": result.n_batches,
        "command": result.derived.command,
    }
    if result.dry_run_payload is not None:
        summary["batch_slurm_dir"] = result.dry_run_payload.get("batch_slurm_dir")
        summary["estimated_raw_hours"] = result.dry_run_payload.get("estimated_raw_hours")
        summary["estimated_padded_hours"] = result.dry_run_payload.get("estimated_padded_hours")
    else:
        summary["n_slurm_paths"] = len(result.generated_slurm_paths)
        if len(result.generated_slurm_paths) <= 20:
            summary["generated_slurm_paths"] = list(result.generated_slurm_paths)
        else:
            summary["generated_slurm_paths_preview"] = list(result.generated_slurm_paths[:20])
    return summary


if __name__ == "__main__":
    main()
