"""Experiment manifest loading and training-job expansion."""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ham_embed_spectral.naming import (
    CANONICAL_ENCODER_CHOICES,
    ENCODER_CLI_CHOICES,
    canonical_encoder_name,
)

ENCODER_CHOICES = CANONICAL_ENCODER_CHOICES
DATASET_CHOICES = ("pendigits", "synthetic-eigengap", "synthetic-singular")
REPRESENTATION_CHOICES = ("dyn", "sta4", "sta8", "sta16", "synthetic")
INITIAL_STATE_CHOICES = ("plus", "zero")


@dataclass(frozen=True)
class TrainJob:
    """One model-training job in an experiment grid."""

    dataset: str
    representation: str
    encoder: str
    reupload_depth: int
    seed: int
    learning_rate: float
    batch_size: int
    class_subset: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "encoder", canonical_encoder_name(self.encoder))

    @property
    def slug(self) -> str:
        """Filesystem/job-name-safe identifier for this training run."""

        class_part = "" if self.class_subset is None else f"__classes_{self.class_subset}"
        raw = (
            f"{self.dataset}__{self.representation}__{self.encoder}"
            f"__L{self.reupload_depth}__lr{self.learning_rate:g}"
            f"__bs{self.batch_size}{class_part}__seed{self.seed}"
        )
        return sanitize_slug(raw)


def sanitize_slug(value: str) -> str:
    """Convert a string to a conservative filename/job-name slug."""

    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return "".join(char if char in allowed else "_" for char in value).strip("_")


def normalize_optional_value(value: str | None) -> str | None:
    """Normalize optional grid values such as class subsets."""

    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped or stripped.lower() in {"none", "null", "full", "-"}:
        return None
    return stripped


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a JSON experiment manifest."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    payload.setdefault("manifest_id", manifest_path.stem)
    payload.setdefault("_path", str(manifest_path))
    validate_manifest(payload)
    return payload


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate fields required for training job expansion."""

    grid = manifest.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("manifest requires a 'grid' object")
    required = (
        "datasets",
        "representations",
        "encoders",
        "reupload_depths",
        "seeds",
        "learning_rates",
        "batch_sizes",
    )
    missing = [key for key in required if key not in grid]
    if missing:
        raise ValueError(f"manifest grid missing required field(s): {', '.join(missing)}")

    _validate_choices("datasets", grid["datasets"], DATASET_CHOICES)
    _validate_choices("representations", grid["representations"], REPRESENTATION_CHOICES)
    _validate_choices("encoders", grid["encoders"], ENCODER_CLI_CHOICES)
    model = manifest.get("model", {})
    if isinstance(model, dict) and "initial_state" in model:
        initial_state = model["initial_state"]
        if initial_state not in INITIAL_STATE_CHOICES:
            expected = ", ".join(INITIAL_STATE_CHOICES)
            raise ValueError(f"manifest model initial_state={initial_state!r}; expected {expected}")


def _validate_choices(name: str, values: list[str], choices: tuple[str, ...]) -> None:
    invalid = [value for value in values if value not in choices]
    if invalid:
        expected = ", ".join(choices)
        raise ValueError(
            f"manifest grid {name} has invalid value(s) {invalid}; expected {expected}"
        )


def jobs_from_manifest(
    manifest: dict[str, Any],
    encoders: Iterable[str] | None = None,
) -> list[TrainJob]:
    """Expand a manifest grid into one job per model training run."""

    grid = manifest["grid"]
    wanted_encoders = None
    if encoders is not None:
        wanted_encoders = {canonical_encoder_name(value) for value in encoders}
    class_subsets = [normalize_optional_value(value) for value in grid.get("class_subsets", [None])]
    jobs: list[TrainJob] = []
    for combo in itertools.product(
        grid["datasets"],
        grid["representations"],
        grid["encoders"],
        grid["reupload_depths"],
        grid["seeds"],
        grid["learning_rates"],
        grid["batch_sizes"],
        class_subsets,
    ):
        dataset, representation, encoder, depth, seed, learning_rate, batch_size, class_subset = (
            combo
        )
        job = TrainJob(
            dataset=dataset,
            representation=representation,
            encoder=encoder,
            reupload_depth=int(depth),
            seed=int(seed),
            learning_rate=float(learning_rate),
            batch_size=int(batch_size),
            class_subset=class_subset,
        )
        if wanted_encoders is None or job.encoder in wanted_encoders:
            jobs.append(job)
    return jobs


def manifest_section(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a manifest section as a dictionary."""

    section = manifest.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"manifest section {name!r} must be an object")
    return section


def manifest_value(manifest: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    """Read a value from a manifest section with a default."""

    return manifest_section(manifest, section).get(key, default)
