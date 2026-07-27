#!/usr/bin/env python3
"""Run classical spectral/raw baselines over manifest seed grids."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ham_embed_spectral.experiments.manifest import (  # noqa: E402
    load_manifest,
    normalize_optional_value,
    sanitize_slug,
)
from ham_embed_spectral.naming import (  # noqa: E402
    CANONICAL_BLOCK_HAMILTONIAN,
    CANONICAL_DESCRIPTOR_CHOICES,
    CANONICAL_SYMMETRIC_HAMILTONIAN,
    DESCRIPTOR_CLI_CHOICES,
    canonical_descriptor_name,
)

BASELINE_SCRIPT = ROOT / "scripts" / "classical_baseline.py"
DEFAULT_MANIFESTS = (
    "configs/experiments/pendigits.json",
    "configs/experiments/synthetic.json",
)
DESCRIPTOR_CHOICES = CANONICAL_DESCRIPTOR_CHOICES
FEATURE_SET_CHOICES = ("values", "raw")
RAW_DATA_DESCRIPTOR = "raw-data"


@dataclass(frozen=True)
class ClassicalBaselineJob:
    """One classical baseline run."""

    manifest_id: str
    dataset: str
    representation: str
    descriptor: str
    classifier: str
    feature_set: str
    seed: int
    data_seed: int | None
    learning_rate: float
    steps: int
    output_path: Path
    class_subset: str | None = None
    standardize: bool = True
    download_data: bool = False
    data_root: str = "data/raw/pendigits"
    validation_fraction: float = 0.1
    n_samples: int = 128
    synthetic_dim: int = 4
    synthetic_rows: int = 4
    synthetic_cols: int = 2
    synthetic_threshold: float | None = None
    synthetic_noise_epsilon: float = 0.0
    mlp_hidden_width: int | None = None

    @property
    def slug(self) -> str:
        class_part = "" if self.class_subset is None else f"__classes_{self.class_subset}"
        raw = (
            f"{self.dataset}__{self.representation}__classical-{self.descriptor}"
            f"__{self.classifier}__{self.feature_set}__lr{self.learning_rate:g}"
            f"{class_part}__seed{self.seed}"
        )
        return sanitize_slug(raw)

    def command(self) -> list[str]:
        command = [
            sys.executable,
            str(BASELINE_SCRIPT),
            "--dataset",
            self.dataset,
            "--representation",
            self.representation,
            "--data-root",
            self.data_root,
            "--validation-fraction",
            f"{self.validation_fraction:g}",
            "--seed",
            str(self.seed),
            "--n-samples",
            str(self.n_samples),
            "--synthetic-dim",
            str(self.synthetic_dim),
            "--synthetic-rows",
            str(self.synthetic_rows),
            "--synthetic-cols",
            str(self.synthetic_cols),
            "--synthetic-noise-epsilon",
            f"{self.synthetic_noise_epsilon:g}",
            "--descriptor",
            self.descriptor,
            "--classifier",
            self.classifier,
            "--feature-set",
            self.feature_set,
            "--steps",
            str(self.steps),
            "--learning-rate",
            f"{self.learning_rate:g}",
            "--output",
            str(self.output_path),
            "--manifest-id",
            self.manifest_id,
            "--job-slug",
            self.slug,
        ]
        if self.mlp_hidden_width is not None:
            command.extend(["--mlp-hidden-width", str(self.mlp_hidden_width)])
        if self.data_seed is not None:
            command.extend(["--data-seed", str(self.data_seed)])
        if self.class_subset is not None:
            command.extend(["--class-subset", self.class_subset])
        if self.download_data:
            command.append("--download-data")
        command.append("--standardize" if self.standardize else "--no-standardize")
        if self.synthetic_threshold is not None:
            command.extend(["--synthetic-threshold", f"{self.synthetic_threshold:g}"])
        return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifests", nargs="+", default=list(DEFAULT_MANIFESTS))
    parser.add_argument("--output-root", default="results/runs/classical_baseline")
    parser.add_argument(
        "--descriptor-policy",
        choices=("matched", "all", *DESCRIPTOR_CLI_CHOICES),
        default="matched",
        help="Which spectral descriptors to run for each dataset/representation.",
    )
    parser.add_argument("--classifier", choices=("mlp", "linear-svc"), default="mlp")
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SET_CHOICES,
        default=None,
        help="Backward-compatible singular form. Prefer --feature-sets for new runs.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=FEATURE_SET_CHOICES,
        default=None,
        help=(
            "Feature sets to run. Raw emits one job per dataset/seed "
            "independent of descriptor policy."
        ),
    )
    parser.add_argument("--mlp-hidden-width", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--bins", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--jax-platform",
        default="cpu",
        help="Set JAX_PLATFORMS for child baseline processes; use empty string to leave unset.",
    )
    args = parser.parse_args()
    if args.descriptor_policy not in {"matched", "all"}:
        args.descriptor_policy = canonical_descriptor_name(args.descriptor_policy)
    if args.feature_sets is None:
        args.feature_sets = [args.feature_set or "values"]
    if args.mlp_hidden_width is not None and args.mlp_hidden_width < 1:
        parser.error("--mlp-hidden-width must be positive")
    return args


def main() -> None:
    args = parse_args()
    jobs = list(iter_classical_baseline_jobs(args))
    if args.limit is not None:
        jobs = jobs[: args.limit]

    if args.dry_run:
        for job in jobs:
            print(shlex.join(job.command()))
        print(f"# {len(jobs)} classical baseline job(s)")
        return

    failures: list[tuple[ClassicalBaselineJob, int]] = []
    env = os.environ.copy()
    if args.jax_platform:
        env["JAX_PLATFORMS"] = args.jax_platform

    for index, job in enumerate(jobs, start=1):
        if job.output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(jobs)}] skip existing {job.output_path}")
            continue
        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(jobs)}] run {job.slug}")
        result = subprocess.run(job.command(), cwd=ROOT, check=False, env=env)
        if result.returncode != 0:
            failures.append((job, result.returncode))
            print(f"[{index}/{len(jobs)}] failed {job.slug}: exit {result.returncode}")
            if args.fail_fast:
                break

    if failures:
        print(f"{len(failures)} classical baseline job(s) failed", file=sys.stderr)
        raise SystemExit(1)


def iter_classical_baseline_jobs(args: argparse.Namespace):
    for manifest_path in args.manifests:
        manifest = load_manifest(manifest_path)
        manifest_id = str(manifest["manifest_id"])
        grid = manifest["grid"]
        data = manifest.get("data", {})
        training = manifest.get("training", {})
        synthetic = manifest.get("synthetic", {})
        class_subsets = [
            normalize_optional_value(value) for value in grid.get("class_subsets", [None])
        ]
        learning_rates = (
            [float(args.learning_rate)]
            if args.learning_rate is not None
            else [float(value) for value in grid["learning_rates"]]
        )
        steps = int(args.steps if args.steps is not None else training.get("steps", 500))
        data_seed = data.get("data_seed")
        output_dir = Path(args.output_root) / manifest_id

        for dataset in grid["datasets"]:
            dataset_synthetic = synthetic_options_for_dataset(synthetic, dataset)
            for representation in grid["representations"]:
                if not representation_matches_dataset(dataset, representation):
                    continue
                for feature_set in args.feature_sets:
                    descriptors = descriptors_for_feature_set(
                        dataset,
                        representation,
                        args.descriptor_policy,
                        feature_set,
                    )
                    for descriptor in descriptors:
                        for learning_rate in learning_rates:
                            for class_subset in class_subsets:
                                for seed in grid["seeds"]:
                                    seed_int = int(seed)
                                    preview = ClassicalBaselineJob(
                                        manifest_id=manifest_id,
                                        dataset=dataset,
                                        representation=representation,
                                        descriptor=descriptor,
                                        classifier=args.classifier,
                                        feature_set=feature_set,
                                        seed=seed_int,
                                        data_seed=int(data_seed) if data_seed is not None else None,
                                        learning_rate=learning_rate,
                                        steps=steps,
                                        output_path=Path(),
                                        class_subset=class_subset,
                                        standardize=bool(data.get("standardize", True)),
                                        download_data=bool(data.get("download_data", False)),
                                        data_root=str(data.get("data_root", "data/raw/pendigits")),
                                        validation_fraction=float(
                                            data.get("validation_fraction", 0.1)
                                        ),
                                        mlp_hidden_width=args.mlp_hidden_width,
                                        **dataset_synthetic,
                                    )
                                    output_path = output_dir / f"{preview.slug}.json"
                                    yield replace_output_path(preview, output_path)


def synthetic_options_for_dataset(synthetic: dict[str, Any], dataset: str) -> dict[str, Any]:
    dataset_options = synthetic.get("datasets", {}).get(dataset, {})
    return {
        "n_samples": int(synthetic.get("n_samples", 128)),
        "synthetic_dim": int(dataset_options.get("synthetic_dim", 4)),
        "synthetic_rows": int(dataset_options.get("synthetic_rows", 4)),
        "synthetic_cols": int(dataset_options.get("synthetic_cols", 2)),
        "synthetic_threshold": dataset_options.get("synthetic_threshold"),
        "synthetic_noise_epsilon": float(synthetic.get("synthetic_noise_epsilon", 0.0)),
    }


def representation_matches_dataset(dataset: str, representation: str) -> bool:
    if dataset == "pendigits":
        return representation != "synthetic"
    return representation == "synthetic"


def descriptors_for_feature_set(
    dataset: str,
    representation: str,
    policy: str,
    feature_set: str,
) -> tuple[str, ...]:
    if feature_set == "raw":
        return (RAW_DATA_DESCRIPTOR,)
    if feature_set == "values":
        return descriptors_for(dataset, representation, policy)
    expected = ", ".join(FEATURE_SET_CHOICES)
    raise ValueError(f"unknown feature_set {feature_set!r}; expected one of {expected}")


def descriptors_for(dataset: str, representation: str, policy: str) -> tuple[str, ...]:
    if policy not in {"matched", "all"}:
        policy = canonical_descriptor_name(policy)
    if policy in DESCRIPTOR_CHOICES:
        return (policy,)
    if policy == "all":
        return DESCRIPTOR_CHOICES
    if dataset == "synthetic-eigengap":
        return (CANONICAL_SYMMETRIC_HAMILTONIAN,)
    if dataset == "synthetic-singular":
        return (CANONICAL_BLOCK_HAMILTONIAN,)
    if representation == "dyn":
        return (CANONICAL_BLOCK_HAMILTONIAN,)
    return (CANONICAL_SYMMETRIC_HAMILTONIAN,)


def replace_output_path(job: ClassicalBaselineJob, output_path: Path) -> ClassicalBaselineJob:
    return ClassicalBaselineJob(
        manifest_id=job.manifest_id,
        dataset=job.dataset,
        representation=job.representation,
        descriptor=job.descriptor,
        classifier=job.classifier,
        feature_set=job.feature_set,
        seed=job.seed,
        data_seed=job.data_seed,
        learning_rate=job.learning_rate,
        steps=job.steps,
        output_path=output_path,
        class_subset=job.class_subset,
        standardize=job.standardize,
        download_data=job.download_data,
        data_root=job.data_root,
        validation_fraction=job.validation_fraction,
        n_samples=job.n_samples,
        synthetic_dim=job.synthetic_dim,
        synthetic_rows=job.synthetic_rows,
        synthetic_cols=job.synthetic_cols,
        synthetic_threshold=job.synthetic_threshold,
        synthetic_noise_epsilon=job.synthetic_noise_epsilon,
        mlp_hidden_width=job.mlp_hidden_width,
    )


if __name__ == "__main__":
    main()
