"""Pauli matrices and tensor-product utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import product

import jax.numpy as jnp

PAULI_ORDER = ("I", "X", "Y", "Z")


def pauli_matrices(dtype: jnp.dtype = jnp.complex128) -> dict[str, jnp.ndarray]:
    """Return the one-qubit Pauli basis in a stable order."""

    return {
        "I": jnp.asarray([[1, 0], [0, 1]], dtype=dtype),
        "X": jnp.asarray([[0, 1], [1, 0]], dtype=dtype),
        "Y": jnp.asarray([[0, -1j], [1j, 0]], dtype=dtype),
        "Z": jnp.asarray([[1, 0], [0, -1]], dtype=dtype),
    }


def pauli(label: str, dtype: jnp.dtype = jnp.complex128) -> jnp.ndarray:
    """Return a one-qubit Pauli matrix by label."""

    try:
        return pauli_matrices(dtype)[label]
    except KeyError as exc:
        msg = f"unknown Pauli label {label!r}; expected one of {PAULI_ORDER}"
        raise ValueError(msg) from exc


def kron_n(factors: Sequence[jnp.ndarray]) -> jnp.ndarray:
    """Kronecker product of a non-empty matrix sequence."""

    if not factors:
        raise ValueError("kron_n requires at least one factor")
    result = factors[0]
    for factor in factors[1:]:
        result = jnp.kron(result, factor)
    return result


def pauli_string(label: str | Iterable[str], dtype: jnp.dtype = jnp.complex128) -> jnp.ndarray:
    """Construct a tensor-product Pauli string.

    ``label`` may be a compact string such as ``"IXZ"`` or any iterable of
    one-qubit labels.
    """

    labels = tuple(label)
    if not labels:
        raise ValueError("pauli_string requires at least one Pauli label")
    return kron_n([pauli(single, dtype=dtype) for single in labels])


def two_qubit_pauli_labels(include_identity: bool = False) -> tuple[str, ...]:
    """Ordered two-qubit Pauli labels used for the SU(4) parameterization."""

    labels = tuple("".join(parts) for parts in product(PAULI_ORDER, repeat=2))
    if include_identity:
        return labels
    return tuple(label for label in labels if label != "II")


def two_qubit_pauli_generators(dtype: jnp.dtype = jnp.complex128) -> jnp.ndarray:
    """Return the 15 non-identity two-qubit Pauli generators."""

    return jnp.stack([pauli_string(label, dtype=dtype) for label in two_qubit_pauli_labels()])
