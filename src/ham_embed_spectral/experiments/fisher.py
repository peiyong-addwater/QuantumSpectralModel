"""Empirical Fisher / gradient covariance diagnostics."""

from __future__ import annotations

import jax.numpy as jnp


def empirical_fisher_summary(
    gradients: jnp.ndarray,
    *,
    damping: float = 1e-8,
) -> dict[str, jnp.ndarray]:
    """Summarize ``F = mean_i g_i g_i^T`` from per-sample gradient rows."""

    if gradients.ndim != 2:
        raise ValueError(f"gradients must have shape (batch, n_params), got {gradients.shape}")
    if gradients.shape[0] == 0 or gradients.shape[1] == 0:
        return {
            "trace": jnp.asarray(0.0),
            "effective_rank": jnp.asarray(0.0),
            "condition_number": jnp.asarray(0.0),
        }
    # Non-zero eigenvalues of ``G.T @ G / B`` and ``G @ G.T / B`` match.  The
    # sample-space form is much smaller for the diagnostic batches used here.
    gram = gradients @ gradients.T / gradients.shape[0]
    eigvals = jnp.maximum(jnp.linalg.eigvalsh(gram), 0.0)
    trace = jnp.sum(eigvals)
    trace_sq = jnp.sum(eigvals**2)
    effective_rank = trace**2 / jnp.maximum(trace_sq, damping)
    positive = eigvals[eigvals > damping]
    condition = jnp.where(
        positive.size > 0,
        jnp.max(positive) / jnp.maximum(jnp.min(positive), damping),
        0.0,
    )
    return {
        "trace": trace,
        "effective_rank": effective_rank,
        "condition_number": condition,
    }
