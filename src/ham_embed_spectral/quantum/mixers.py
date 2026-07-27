"""Local SU(4) brick-wall mixers."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from ham_embed_spectral.quantum.su4 import SU4_PARAM_COUNT, apply_su4_block


def brickwall_pairs(n_qubits: int) -> tuple[tuple[int, int], ...]:
    """Return even then odd nearest-neighbor two-qubit pairs."""

    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    even_pairs = tuple((left, left + 1) for left in range(0, n_qubits - 1, 2))
    odd_pairs = tuple((left, left + 1) for left in range(1, n_qubits - 1, 2))
    return (*even_pairs, *odd_pairs)


def count_su4_mixer_params(n_qubits: int, reupload_depth: int) -> int:
    """Count trainable parameters in the local SU(4) mixer stack."""

    return len(brickwall_pairs(n_qubits)) * reupload_depth * SU4_PARAM_COUNT


def init_su4_mixer_params(
    key: jax.Array,
    n_qubits: int,
    reupload_depth: int,
    scale: float = 0.01,
    dtype: jnp.dtype = jnp.float64,
) -> jnp.ndarray:
    """Initialize local SU(4) mixer parameters."""

    n_blocks = len(brickwall_pairs(n_qubits))
    return scale * jax.random.normal(key, (reupload_depth, n_blocks, SU4_PARAM_COUNT), dtype=dtype)


def apply_su4_mixer(
    state: jnp.ndarray,
    theta_layer: jnp.ndarray,
    n_qubits: int,
    pairs: Sequence[tuple[int, int]] | None = None,
) -> jnp.ndarray:
    """Apply a full even/odd brick-wall mixer layer."""

    pairs = brickwall_pairs(n_qubits) if pairs is None else tuple(pairs)
    if theta_layer.shape != (len(pairs), SU4_PARAM_COUNT):
        raise ValueError(
            f"theta_layer must have shape {(len(pairs), SU4_PARAM_COUNT)}, "
            f"got {theta_layer.shape}"
        )
    for block_index, wires in enumerate(pairs):
        state = apply_su4_block(state, theta_layer[block_index], wires=wires, n_qubits=n_qubits)
    return state
