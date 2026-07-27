"""Readout maps from statevectors to model outputs."""

from __future__ import annotations

import jax.numpy as jnp

from ham_embed_spectral.quantum.statevector import num_qubits_from_dim


def ceil_log2(value: int) -> int:
    """Ceiling base-2 logarithm for positive integers."""

    if value < 1:
        raise ValueError("value must be positive")
    return (value - 1).bit_length()


def projector_probabilities(
    state: jnp.ndarray,
    n_classes: int,
    label_qubits: int | None = None,
    renormalize: bool = False,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Projector readout on the leading label qubits.

    For ``K`` classes, class ``i`` is assigned to computational-basis label
    state ``i`` on the leading ``ceil(log2(K))`` qubits. The returned vector
    contains only those ``K`` class states. If ``K`` is not a power of two,
    probability on unused label states is not included in the returned vector;
    ``renormalize=True`` makes the class probabilities conditional on landing
    in one of the named class states.
    """

    label_probs = projector_label_probabilities(state, n_classes, label_qubits)
    probs = label_probs[..., :n_classes]
    if renormalize:
        probs = probs / jnp.maximum(jnp.sum(probs, axis=-1, keepdims=True), eps)
    return probs


def projector_label_probabilities(
    state: jnp.ndarray,
    n_classes: int,
    label_qubits: int | None = None,
) -> jnp.ndarray:
    """Return probabilities for all computational states of the label register."""

    n_qubits = num_qubits_from_dim(state.shape[-1])
    label_qubits = ceil_log2(n_classes) if label_qubits is None else label_qubits
    if label_qubits > n_qubits:
        raise ValueError(
            f"label_qubits={label_qubits} exceeds available n_qubits={n_qubits}"
        )
    if n_classes > 2**label_qubits:
        raise ValueError("n_classes must fit in the requested label qubits")

    rest_dim = 2 ** (n_qubits - label_qubits)
    reshaped = jnp.reshape(state, (*state.shape[:-1], 2**label_qubits, rest_dim))
    return jnp.sum(jnp.abs(reshaped) ** 2, axis=-1)


def projector_leakage_mass(
    state: jnp.ndarray,
    n_classes: int,
    label_qubits: int | None = None,
) -> jnp.ndarray:
    """Return probability mass assigned to unused projector-readout states.

    This is a diagnostic/API hook for future leakage-problem experiments. It
    does not change the current training target or introduce an additional
    class; it only exposes mass outside the named class projectors.
    """

    label_probs = projector_label_probabilities(state, n_classes, label_qubits)
    return jnp.sum(label_probs[..., n_classes:], axis=-1)


def projector_logits(
    state: jnp.ndarray,
    n_classes: int,
    label_qubits: int | None = None,
    renormalize: bool = True,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Log-probability logits derived from projector probabilities."""

    probs = projector_probabilities(
        state,
        n_classes=n_classes,
        label_qubits=label_qubits,
        renormalize=renormalize,
        eps=eps,
    )
    return jnp.log(jnp.maximum(probs, eps))


def observable_expectations(state: jnp.ndarray, observables: jnp.ndarray) -> jnp.ndarray:
    """Return real expectation values for a stack of dense observables."""

    if observables.ndim != 3:
        raise ValueError(f"observables must have shape (n_obs, dim, dim), got {observables.shape}")
    if observables.shape[-1] != state.shape[-1] or observables.shape[-2] != state.shape[-1]:
        raise ValueError("observable dimensions must match state dimension")
    acted = jnp.einsum("oij,...j->...oi", observables, state)
    return jnp.real(jnp.einsum("...i,...oi->...o", jnp.conj(state), acted))
