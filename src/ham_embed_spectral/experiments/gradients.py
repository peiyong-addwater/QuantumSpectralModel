"""Gradient diagnostic helpers."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def tree_ravel(tree) -> jnp.ndarray:
    """Flatten a JAX pytree into one vector."""

    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray([], dtype=jnp.float64)
    flattened = [jnp.ravel(leaf) for leaf in leaves if jnp.size(leaf) > 0]
    if not flattened:
        return jnp.asarray([], dtype=jnp.float64)
    return jnp.concatenate(flattened)


def gradient_summary(grads, near_zero_tol: float = 1e-10) -> dict[str, jnp.ndarray]:
    """Compute basic finite-gradient diagnostics for a gradient pytree."""

    flat = tree_ravel(grads)
    if flat.size == 0:
        return {
            "rms": jnp.asarray(0.0),
            "near_zero_fraction": jnp.asarray(1.0),
            "finite_fraction": jnp.asarray(1.0),
        }
    abs_flat = jnp.abs(flat)
    return {
        "rms": jnp.sqrt(jnp.mean(abs_flat**2)),
        "near_zero_fraction": jnp.mean(abs_flat < near_zero_tol),
        "finite_fraction": jnp.mean(jnp.isfinite(abs_flat)),
    }


def gradients_for_loss(loss_fn, params):
    """Return ``(loss, grads, summary)`` for a scalar loss function."""

    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads, gradient_summary(grads)


def flatten_gradient_groups(grads: dict[str, Any]) -> dict[str, jnp.ndarray]:
    """Flatten gradients into the paper-facing parameter groups."""

    groups: dict[str, jnp.ndarray] = {}
    theta_su = grads.get("theta_su")
    if theta_su is not None:
        groups["theta_su"] = tree_ravel(theta_su)

    encoder_grads = grads.get("encoder", {})
    if isinstance(encoder_grads, dict):
        for group in ("gamma", "t_layers", "patch_map"):
            if group in encoder_grads:
                groups[group] = tree_ravel(encoder_grads[group])
    return groups


def gradient_group_summary(
    samples: jnp.ndarray,
    *,
    near_zero_tol: float = 1e-10,
    eps: float = 1e-30,
) -> dict[str, jnp.ndarray]:
    """Summarize repeated gradient vectors for one parameter group."""

    if samples.ndim != 2:
        raise ValueError(f"samples must have shape (n_samples, n_params), got {samples.shape}")
    if samples.shape[1] == 0:
        zero = jnp.asarray(0.0)
        return {
            "parameter_count": jnp.asarray(0),
            "median_log10_variance": zero,
            "iqr_log10_variance": zero,
            "mean_rms_gradient": zero,
            "std_rms_gradient": zero,
            "near_zero_fraction": jnp.asarray(1.0),
            "finite_fraction": jnp.asarray(1.0),
            "median_log10_snr": zero,
        }

    variances = jnp.var(samples, axis=0)
    log_variances = jnp.log10(variances + eps)
    rms_by_sample = jnp.sqrt(jnp.mean(jnp.abs(samples) ** 2, axis=1))
    snr = jnp.abs(jnp.mean(samples, axis=0)) / (jnp.sqrt(variances) + jnp.sqrt(eps))
    return {
        "parameter_count": jnp.asarray(samples.shape[1]),
        "median_log10_variance": jnp.median(log_variances),
        "iqr_log10_variance": jnp.percentile(log_variances, 75)
        - jnp.percentile(log_variances, 25),
        "mean_rms_gradient": jnp.mean(rms_by_sample),
        "std_rms_gradient": jnp.std(rms_by_sample),
        "near_zero_fraction": jnp.mean(jnp.abs(samples) < near_zero_tol),
        "finite_fraction": jnp.mean(jnp.isfinite(samples)),
        "median_log10_snr": jnp.median(jnp.log10(snr + jnp.sqrt(eps))),
    }


def layerwise_gradient_flow(grads: dict[str, Any]) -> dict[str, jnp.ndarray]:
    """Return per-layer gradient magnitudes for layer-structured groups."""

    flow: dict[str, jnp.ndarray] = {}
    theta_su = grads.get("theta_su")
    if theta_su is not None and theta_su.ndim >= 2:
        if theta_su.size == 0:
            flow["theta_su"] = jnp.zeros((theta_su.shape[0],), dtype=jnp.float64)
        else:
            flow["theta_su"] = jnp.sqrt(
                jnp.mean(
                    jnp.abs(theta_su) ** 2,
                    axis=tuple(range(1, theta_su.ndim)),
                )
            )

    encoder_grads = grads.get("encoder", {})
    if isinstance(encoder_grads, dict):
        gamma = encoder_grads.get("gamma")
        if gamma is not None and gamma.ndim >= 2:
            flow["gamma"] = jnp.sqrt(
                jnp.mean(jnp.abs(gamma) ** 2, axis=tuple(range(1, gamma.ndim)))
            )
        t_layers = encoder_grads.get("t_layers")
        if t_layers is not None:
            flow["t_layers"] = jnp.abs(jnp.ravel(t_layers))
        patch_map = encoder_grads.get("patch_map")
        if patch_map is not None and patch_map.ndim >= 2:
            flow["patch_map"] = jnp.sqrt(
                jnp.mean(jnp.abs(patch_map) ** 2, axis=tuple(range(1, patch_map.ndim)))
            )
    return flow


def per_sample_gradient_matrix(loss_fn, params, samples: jnp.ndarray, labels: jnp.ndarray):
    """Return a dense ``(batch, n_params)`` per-sample gradient matrix."""

    def one(sample, label):
        def one_loss(current_params):
            return loss_fn(current_params, sample[jnp.newaxis, ...], label[jnp.newaxis])

        return jax.grad(one_loss)(params)

    per_sample_tree = jax.vmap(one)(samples, labels)
    leaves = [
        jnp.reshape(leaf, (leaf.shape[0], -1))
        for leaf in jax.tree.leaves(per_sample_tree)
        if getattr(leaf, "size", 0) > 0
    ]
    if not leaves:
        return jnp.zeros((samples.shape[0], 0), dtype=jnp.float64)
    return jnp.concatenate(leaves, axis=1)
