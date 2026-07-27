"""Simple metrics used by smoke tests and scripts."""

from __future__ import annotations

import jax.numpy as jnp


def accuracy_from_probs(probs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Classification accuracy from class probabilities."""

    return jnp.mean(jnp.argmax(probs, axis=-1) == labels)


def cross_entropy_from_probs(
    probs: jnp.ndarray,
    labels: jnp.ndarray,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Mean cross-entropy from class probabilities."""

    selected = jnp.take_along_axis(probs, labels[..., None], axis=-1)[..., 0]
    return -jnp.mean(jnp.log(jnp.maximum(selected, eps)))
