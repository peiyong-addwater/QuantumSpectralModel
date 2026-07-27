"""Controlled synthetic spectral tasks."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class SyntheticDataset:
    """In-memory synthetic dataset."""

    x: jnp.ndarray
    y: jnp.ndarray


def _relative_noise(key: jax.Array, base: jnp.ndarray, epsilon: float) -> jnp.ndarray:
    noise = jax.random.normal(key, base.shape, dtype=base.dtype)
    noise_norm = jnp.linalg.norm(noise.reshape((noise.shape[0], -1)), axis=-1)
    base_norm = jnp.linalg.norm(base.reshape((base.shape[0], -1)), axis=-1)
    scale = epsilon * base_norm / jnp.maximum(noise_norm, 1e-12)
    return noise * scale.reshape((-1, 1, 1))


def eigengap_classification(
    key: jax.Array,
    n_samples: int = 128,
    dim: int = 4,
    threshold: float = 0.75,
    noise_epsilon: float = 0.0,
) -> SyntheticDataset:
    """Generate ``M = Q Lambda Q.T + epsilon N`` with eigengap labels."""

    key_q, key_lambda, key_noise = jax.random.split(key, 3)
    raw = jax.random.normal(key_q, (n_samples, dim, dim))
    q, _ = jax.vmap(jnp.linalg.qr)(raw)
    eigvals = jnp.sort(jax.random.normal(key_lambda, (n_samples, dim)), axis=-1)
    base = jnp.einsum("nij,nj,nkj->nik", q, eigvals, q)
    if noise_epsilon:
        base = base + _relative_noise(key_noise, base, noise_epsilon)
    labels = (jnp.abs(eigvals[:, -1] - eigvals[:, -2]) > threshold).astype(jnp.int32)
    return SyntheticDataset(x=base, y=labels)


def singular_value_classification(
    key: jax.Array,
    n_samples: int = 128,
    shape: tuple[int, int] = (4, 2),
    threshold: float = 1.5,
    noise_epsilon: float = 0.0,
) -> SyntheticDataset:
    """Generate rectangular matrices with labels from leading singular values."""

    rows, cols = shape
    rank = min(rows, cols)
    key_u, key_v, key_sigma, key_noise = jax.random.split(key, 4)
    u_raw = jax.random.normal(key_u, (n_samples, rows, rows))
    v_raw = jax.random.normal(key_v, (n_samples, cols, cols))
    u, _ = jax.vmap(jnp.linalg.qr)(u_raw)
    v, _ = jax.vmap(jnp.linalg.qr)(v_raw)
    sigma = jnp.sort(jax.random.uniform(key_sigma, (n_samples, rank), minval=0.0, maxval=2.0))[
        :, ::-1
    ]
    diag = jax.vmap(lambda s: jnp.pad(jnp.diag(s), ((0, rows - rank), (0, cols - rank))))(sigma)
    base = jnp.einsum("nij,njk,nlk->nil", u, diag, v)
    if noise_epsilon:
        base = base + _relative_noise(key_noise, base, noise_epsilon)
    second = sigma[:, 1] if rank > 1 else jnp.zeros((n_samples,), dtype=sigma.dtype)
    top_two = sigma[:, 0] + second
    labels = (top_two > threshold).astype(jnp.int32)
    return SyntheticDataset(x=base, y=labels)
