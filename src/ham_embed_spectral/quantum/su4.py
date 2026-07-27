"""SU(4) utilities based on two-qubit Pauli generators."""

from __future__ import annotations

import jax.numpy as jnp

from ham_embed_spectral.quantum.pauli import two_qubit_pauli_generators, two_qubit_pauli_labels
from ham_embed_spectral.quantum.statevector import apply_local_unitary

SU4_PARAM_COUNT = 15


def su4_generator_labels() -> tuple[str, ...]:
    """Return the ordered labels for the 15 SU(4) generators."""

    return two_qubit_pauli_labels(include_identity=False)


def su4_generators(dtype: jnp.dtype = jnp.complex128) -> jnp.ndarray:
    """Return the 15 non-identity two-qubit Pauli generators."""

    return two_qubit_pauli_generators(dtype=dtype)


def su4_from_params(theta: jnp.ndarray) -> jnp.ndarray:
    """Construct ``exp(i sum_r theta_r G_r)`` from a 15-vector."""

    if theta.shape != (SU4_PARAM_COUNT,):
        raise ValueError(f"theta must have shape ({SU4_PARAM_COUNT},), got {theta.shape}")
    dtype = jnp.result_type(theta, jnp.complex128)
    generators = su4_generators(dtype=dtype)
    h_generator = jnp.tensordot(theta, generators, axes=1)
    eigvals, eigvecs = jnp.linalg.eigh(h_generator)
    phases = jnp.exp(1j * eigvals)
    return (eigvecs * phases) @ eigvecs.conj().T


def apply_su4_block(
    state: jnp.ndarray,
    theta: jnp.ndarray,
    wires: tuple[int, int],
    n_qubits: int,
) -> jnp.ndarray:
    """Apply one SU(4) block to a statevector."""

    return apply_local_unitary(state, su4_from_params(theta), wires=wires, n_qubits=n_qubits)
