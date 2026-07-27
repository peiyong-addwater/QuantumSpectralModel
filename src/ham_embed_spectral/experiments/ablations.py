"""Matrix-structure ablations for Hamiltonian embedding experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ham_embed_spectral.data.pendigits import PendigitsDataset, PendigitsSplit
from ham_embed_spectral.naming import (
    CANONICAL_BLOCK_HAMILTONIAN,
    CANONICAL_SYMMETRIC_HAMILTONIAN,
    canonical_encoder_name,
)
from ham_embed_spectral.quantum.encoders import build_H_sym

ABLATION_CHOICES = (
    "none",
    "entry-permutation",
    "row-column-permutation",
    "spectrum-only",
    "eigenvector-only",
    "singular-spectrum-only",
    "singular-vector-only",
)


@dataclass(frozen=True)
class AblationApplication:
    """Dataset transformed by one ablation plus audit metadata."""

    dataset: PendigitsDataset
    metadata: dict[str, Any]


def permute_entries(sample: jnp.ndarray, permutation: jnp.ndarray) -> jnp.ndarray:
    """Apply a fixed flattened-entry permutation to one matrix sample."""

    flat = jnp.ravel(sample)
    if permutation.shape[0] != flat.shape[0]:
        raise ValueError("entry permutation length must match sample size")
    return jnp.reshape(flat[permutation], sample.shape)


def permute_rows_cols(
    sample: jnp.ndarray,
    row_permutation: jnp.ndarray,
    col_permutation: jnp.ndarray,
) -> jnp.ndarray:
    """Apply fixed row and column permutations to one matrix sample."""

    return sample[row_permutation, :][:, col_permutation]


def spectrum_only_symmetric(sample: jnp.ndarray, fixed_eigenvectors: jnp.ndarray) -> jnp.ndarray:
    """Use sample eigenvalues with training-set fixed eigenvectors."""

    eigvals = jnp.linalg.eigvalsh((sample + sample.T) / 2)
    return (fixed_eigenvectors * eigvals) @ fixed_eigenvectors.T


def eigenvector_only_symmetric(sample: jnp.ndarray, fixed_eigenvalues: jnp.ndarray) -> jnp.ndarray:
    """Use sample eigenvectors with training-set fixed eigenvalues."""

    _, eigvecs = jnp.linalg.eigh((sample + sample.T) / 2)
    return (eigvecs * fixed_eigenvalues) @ eigvecs.T


def singular_spectrum_only(
    sample: jnp.ndarray,
    fixed_left: jnp.ndarray,
    fixed_right: jnp.ndarray,
) -> jnp.ndarray:
    """Use sample singular values with training-set fixed singular vectors."""

    sigma = jnp.linalg.svd(sample, compute_uv=False)
    return fixed_left[:, : sigma.shape[0]] @ jnp.diag(sigma) @ fixed_right[: sigma.shape[0], :]


def singular_vector_only(sample: jnp.ndarray, fixed_singular_values: jnp.ndarray) -> jnp.ndarray:
    """Use sample singular vectors with training-set fixed singular values."""

    u, _, vt = jnp.linalg.svd(sample, full_matrices=False)
    return u @ jnp.diag(fixed_singular_values[: u.shape[1]]) @ vt


def fixed_entry_permutation(key: jax.Array, sample_shape: tuple[int, ...]) -> jnp.ndarray:
    """Create a deterministic entry permutation for a sample shape."""

    return jax.random.permutation(key, jnp.arange(int(jnp.prod(jnp.asarray(sample_shape)))))


def fixed_row_col_permutations(
    key: jax.Array,
    sample_shape: tuple[int, ...],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create deterministic row and column permutations for a matrix shape."""

    if len(sample_shape) != 2:
        raise ValueError(f"row/column permutation requires matrix samples, got {sample_shape}")
    row_key, col_key = jax.random.split(key)
    rows, cols = sample_shape
    return (
        jax.random.permutation(row_key, jnp.arange(rows)),
        jax.random.permutation(col_key, jnp.arange(cols)),
    )


def apply_dataset_ablation(
    dataset: PendigitsDataset,
    *,
    ablation: str,
    encoder_name: str,
    seed: int = 0,
) -> AblationApplication:
    """Apply a named ablation to all splits using training-set-only state."""

    if ablation not in ABLATION_CHOICES:
        expected = ", ".join(ABLATION_CHOICES)
        raise ValueError(f"unknown ablation {ablation!r}; expected one of {expected}")
    encoder_name = canonical_encoder_name(encoder_name)
    metadata: dict[str, Any] = {
        "ablation": ablation,
        "encoder": encoder_name,
        "seed": seed,
        "fixed_objects_source": "training_split_only",
        "input_shape_before": dataset.input_shape,
        "n_train_for_fixed_objects": int(dataset.train.y.shape[0]),
    }
    if ablation == "none":
        metadata["input_shape_after"] = dataset.input_shape
        return AblationApplication(dataset=dataset, metadata=metadata)

    key = jax.random.PRNGKey(seed)
    if ablation == "entry-permutation":
        permutation = fixed_entry_permutation(key, dataset.input_shape)

        def transform(sample: jnp.ndarray) -> jnp.ndarray:
            return permute_entries(sample, permutation)

        metadata["entry_permutation"] = permutation
    elif ablation == "row-column-permutation":
        row_permutation, col_permutation = fixed_row_col_permutations(key, dataset.input_shape)

        def transform(sample: jnp.ndarray) -> jnp.ndarray:
            return permute_rows_cols(sample, row_permutation, col_permutation)

        metadata.update(
            {
                "row_permutation": row_permutation,
                "col_permutation": col_permutation,
            }
        )
    elif ablation in {"spectrum-only", "eigenvector-only"}:
        if encoder_name != CANONICAL_SYMMETRIC_HAMILTONIAN:
            raise ValueError(
                f"{ablation} is only valid with encoder='{CANONICAL_SYMMETRIC_HAMILTONIAN}'"
            )
        train_h = jax.vmap(build_H_sym)(dataset.train.x)
        mean_h = jnp.mean(train_h, axis=0)
        fixed_eigenvalues = jnp.median(jax.vmap(jnp.linalg.eigvalsh)(train_h), axis=0)
        _, fixed_eigenvectors = jnp.linalg.eigh(mean_h)
        if ablation == "spectrum-only":

            def transform(sample: jnp.ndarray) -> jnp.ndarray:
                return spectrum_only_symmetric(build_H_sym(sample), fixed_eigenvectors)

        else:

            def transform(sample: jnp.ndarray) -> jnp.ndarray:
                return eigenvector_only_symmetric(build_H_sym(sample), fixed_eigenvalues)

        metadata.update(
            {
                "fixed_eigenvalues_shape": fixed_eigenvalues.shape,
                "fixed_eigenvectors_shape": fixed_eigenvectors.shape,
            }
        )
    elif ablation in {"singular-spectrum-only", "singular-vector-only"}:
        if encoder_name != CANONICAL_BLOCK_HAMILTONIAN:
            raise ValueError(
                f"{ablation} is only valid with encoder='{CANONICAL_BLOCK_HAMILTONIAN}'"
            )
        mean_matrix = jnp.mean(dataset.train.x, axis=0)
        fixed_left, _, fixed_right = jnp.linalg.svd(mean_matrix, full_matrices=True)
        train_singular_values = jax.vmap(lambda sample: jnp.linalg.svd(sample, compute_uv=False))(
            dataset.train.x
        )
        fixed_singular_values = jnp.median(train_singular_values, axis=0)
        if ablation == "singular-spectrum-only":

            def transform(sample: jnp.ndarray) -> jnp.ndarray:
                return singular_spectrum_only(sample, fixed_left, fixed_right)

        else:

            def transform(sample: jnp.ndarray) -> jnp.ndarray:
                return singular_vector_only(sample, fixed_singular_values)

        metadata.update(
            {
                "fixed_left_shape": fixed_left.shape,
                "fixed_right_shape": fixed_right.shape,
                "fixed_singular_values_shape": fixed_singular_values.shape,
            }
        )
    else:
        raise AssertionError(f"unhandled ablation {ablation!r}")

    transformed = _transform_dataset(dataset, transform)
    metadata["input_shape_after"] = transformed.input_shape
    return AblationApplication(dataset=transformed, metadata=metadata)


def _transform_dataset(
    dataset: PendigitsDataset,
    transform: Callable[[jnp.ndarray], jnp.ndarray],
) -> PendigitsDataset:
    train = _transform_split(dataset.train, transform)
    validation = _transform_split(dataset.validation, transform)
    test = _transform_split(dataset.test, transform)
    input_shape = tuple(int(dim) for dim in train.x.shape[1:])
    return PendigitsDataset(
        train=train,
        validation=validation,
        test=test,
        representation=dataset.representation,
        input_shape=input_shape,
        class_values=dataset.class_values,
        feature_mean=jnp.zeros(input_shape, dtype=train.x.dtype),
        feature_std=jnp.ones(input_shape, dtype=train.x.dtype),
    )


def _transform_split(
    split: PendigitsSplit,
    transform: Callable[[jnp.ndarray], jnp.ndarray],
) -> PendigitsSplit:
    if int(split.y.shape[0]) == 0:
        return split
    return PendigitsSplit(x=jax.vmap(transform)(split.x), y=split.y)
