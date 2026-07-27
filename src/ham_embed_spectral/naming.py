"""Canonical public names and quiet legacy aliases."""

from __future__ import annotations

CANONICAL_SYMMETRIC_HAMILTONIAN = "symmetric-hamiltonian"
CANONICAL_BLOCK_HAMILTONIAN = "block-hamiltonian"
CANONICAL_NON_OVERLAP_PATCH_BLOCK_HAMILTONIAN = "non-overlap-patch-block-hamiltonian"
CANONICAL_TRAINABLE_PATCH_SU4 = "trainable-patch-su4"

CANONICAL_ENCODER_CHOICES = (
    "fixed-ry",
    "fixed-ry-rz",
    "trainable-frequency-ry",
    "patch-su4",
    CANONICAL_TRAINABLE_PATCH_SU4,
    CANONICAL_NON_OVERLAP_PATCH_BLOCK_HAMILTONIAN,
    CANONICAL_SYMMETRIC_HAMILTONIAN,
    CANONICAL_BLOCK_HAMILTONIAN,
)

ENCODER_ALIASES = {
    "hamiltonian": CANONICAL_SYMMETRIC_HAMILTONIAN,
}

ENCODER_CLI_CHOICES = CANONICAL_ENCODER_CHOICES + tuple(ENCODER_ALIASES)

CANONICAL_DESCRIPTOR_CHOICES = (
    CANONICAL_SYMMETRIC_HAMILTONIAN,
    CANONICAL_BLOCK_HAMILTONIAN,
)

DESCRIPTOR_ALIASES = {
    "old-hamiltonian": CANONICAL_SYMMETRIC_HAMILTONIAN,
}

DESCRIPTOR_CLI_CHOICES = CANONICAL_DESCRIPTOR_CHOICES + tuple(DESCRIPTOR_ALIASES)


def canonical_encoder_name(name: str) -> str:
    """Return the canonical encoder slug, accepting quiet legacy aliases."""

    return ENCODER_ALIASES.get(name, name)


def canonical_descriptor_name(name: str) -> str:
    """Return the canonical spectral descriptor slug, accepting quiet aliases."""

    return DESCRIPTOR_ALIASES.get(name, name)


def canonical_slug_aliases(value: str) -> str:
    """Return a slug-like string with legacy Hamiltonian segments canonicalized."""

    canonical = value
    canonical = canonical.replace("__hamiltonian__", f"__{CANONICAL_SYMMETRIC_HAMILTONIAN}__")
    canonical = canonical.replace(
        "__classical-old-hamiltonian__",
        f"__classical-{CANONICAL_SYMMETRIC_HAMILTONIAN}__",
    )
    canonical = canonical.replace(
        "__spectral-old-hamiltonian__",
        f"__spectral-{CANONICAL_SYMMETRIC_HAMILTONIAN}__",
    )
    return canonical


def validate_choice(name: str, value: str, choices: tuple[str, ...]) -> str:
    """Validate one string value against a tuple of accepted choices."""

    if value not in choices:
        expected = ", ".join(choices)
        raise ValueError(f"invalid {name}={value!r}; expected one of {expected}")
    return value
