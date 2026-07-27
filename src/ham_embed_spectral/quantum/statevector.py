"""Dense statevector utilities for exact small-qubit simulation."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp


def is_power_of_two(value: int) -> bool:
    """Return whether ``value`` is a positive power of two."""

    return value > 0 and value & (value - 1) == 0


def next_power_of_two(value: int) -> int:
    """Smallest power of two greater than or equal to ``value``."""

    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def num_qubits_from_dim(dim: int) -> int:
    """Infer the number of qubits from a Hilbert-space dimension."""

    if not is_power_of_two(dim):
        raise ValueError(f"Hilbert dimension must be a power of two, got {dim}")
    return dim.bit_length() - 1


def zero_state(
    n_qubits: int,
    batch_shape: tuple[int, ...] = (),
    dtype: jnp.dtype = jnp.complex128,
) -> jnp.ndarray:
    """Return ``|0...0>`` with shape ``batch_shape + (2**n_qubits,)``."""

    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    dim = 2**n_qubits
    state = jnp.zeros((*batch_shape, dim), dtype=dtype)
    return state.at[(..., 0)].set(jnp.asarray(1, dtype=dtype))


def plus_state(
    n_qubits: int,
    batch_shape: tuple[int, ...] = (),
    dtype: jnp.dtype = jnp.complex128,
) -> jnp.ndarray:
    """Return ``|+>^{otimes n}`` with shape ``batch_shape + (2**n_qubits,)``."""

    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    dim = 2**n_qubits
    amplitude = jnp.asarray(1.0 / jnp.sqrt(jnp.asarray(dim, dtype=jnp.float64)), dtype=dtype)
    return jnp.ones((*batch_shape, dim), dtype=dtype) * amplitude


def initial_state_vector(
    name: str,
    n_qubits: int,
    batch_shape: tuple[int, ...] = (),
    dtype: jnp.dtype = jnp.complex128,
) -> jnp.ndarray:
    """Return a named initial state used by re-uploading models."""

    if name == "zero":
        return zero_state(n_qubits=n_qubits, batch_shape=batch_shape, dtype=dtype)
    if name == "plus":
        return plus_state(n_qubits=n_qubits, batch_shape=batch_shape, dtype=dtype)
    raise ValueError(f"unknown initial state {name!r}; expected 'zero' or 'plus'")


def normalize_state(state: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Normalize a state or batch of states along the Hilbert dimension."""

    norm = jnp.linalg.norm(state, axis=-1, keepdims=True)
    return state / jnp.maximum(norm, eps)


def apply_unitary(state: jnp.ndarray, unitary: jnp.ndarray) -> jnp.ndarray:
    """Apply a dense unitary to a state or batch of states."""

    if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
        raise ValueError(f"unitary must be square, got shape {unitary.shape}")
    if state.shape[-1] != unitary.shape[1]:
        raise ValueError(
            f"state dimension {state.shape[-1]} does not match unitary dimension {unitary.shape[1]}"
        )
    return jnp.einsum("ij,...j->...i", unitary, state)


def apply_local_unitary(
    state: jnp.ndarray,
    unitary: jnp.ndarray,
    wires: Sequence[int],
    n_qubits: int,
) -> jnp.ndarray:
    """Apply a dense local unitary to selected wires.

    Wire ``0`` is the most significant qubit in the statevector basis.
    """

    wires = tuple(wires)
    if not wires:
        raise ValueError("at least one wire is required")
    if len(set(wires)) != len(wires):
        raise ValueError(f"wires must be unique, got {wires}")
    if any(wire < 0 or wire >= n_qubits for wire in wires):
        raise ValueError(f"wires {wires} out of range for {n_qubits} qubits")

    local_dim = 2 ** len(wires)
    if unitary.shape != (local_dim, local_dim):
        raise ValueError(
            f"unitary for {len(wires)} wires must have shape {(local_dim, local_dim)}, "
            f"got {unitary.shape}"
        )
    if state.shape[-1] != 2**n_qubits:
        raise ValueError(
            f"state dimension {state.shape[-1]} does not match {n_qubits} qubits"
        )

    batch_shape = state.shape[:-1]
    batch_ndim = len(batch_shape)
    tensor = jnp.reshape(state, (*batch_shape, *([2] * n_qubits)))
    remaining = tuple(wire for wire in range(n_qubits) if wire not in wires)
    qubit_perm = (*wires, *remaining)
    perm = (*range(batch_ndim), *(batch_ndim + wire for wire in qubit_perm))
    moved = jnp.transpose(tensor, perm)
    moved = jnp.reshape(moved, (*batch_shape, local_dim, 2 ** (n_qubits - len(wires))))
    updated = jnp.einsum("ij,...jk->...ik", unitary, moved)
    updated = jnp.reshape(updated, (*batch_shape, *([2] * n_qubits)))

    inverse_qubit_perm = tuple(qubit_perm.index(wire) for wire in range(n_qubits))
    inverse_perm = (
        *range(batch_ndim),
        *(batch_ndim + axis for axis in inverse_qubit_perm),
    )
    return jnp.reshape(jnp.transpose(updated, inverse_perm), (*batch_shape, 2**n_qubits))


def pad_square_matrix_to_power_of_two(matrix: jnp.ndarray) -> jnp.ndarray:
    """Zero-pad a square matrix to the next power-of-two dimension."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {matrix.shape}")
    dim = matrix.shape[0]
    padded_dim = next_power_of_two(dim)
    if padded_dim == dim:
        return matrix
    pad_width = ((0, padded_dim - dim), (0, padded_dim - dim))
    return jnp.pad(matrix, pad_width)
