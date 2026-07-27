"""Latent-state diagnostics for trained re-uploading models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp

from ham_embed_spectral.quantum.encoders import (
    BlockHamiltonianEncoder,
    SymmetricHamiltonianEncoder,
    build_H_block,
    build_H_sym,
)
from ham_embed_spectral.quantum.readout import projector_probabilities
from ham_embed_spectral.quantum.statevector import pad_square_matrix_to_power_of_two


def projector_score_trace(
    states: jnp.ndarray,
    *,
    n_classes: int,
    renormalize: bool = True,
) -> jnp.ndarray:
    """Return projector-readout scores for a state trace.

    ``states`` may have any leading shape ending in Hilbert dimension, for
    example ``(n_samples, n_layers, dim)``.
    """

    return projector_probabilities(states, n_classes=n_classes, renormalize=renormalize)


def projector_accuracy_by_layer(scores: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Return accuracy for each layer/stage of projector scores."""

    predictions = jnp.argmax(scores, axis=-1)
    return jnp.mean(predictions == labels[:, None], axis=0)


def fidelity_kernel(states: jnp.ndarray) -> jnp.ndarray:
    """Return the pure-state fidelity Gram matrix ``|<psi_i|psi_j>|^2``."""

    if states.ndim != 2:
        raise ValueError(f"states must have shape (n_samples, dim), got {states.shape}")
    inner = jnp.einsum("id,jd->ij", jnp.conj(states), states)
    return jnp.real(jnp.abs(inner) ** 2)


def label_kernel(labels: jnp.ndarray) -> jnp.ndarray:
    """Return the class-equality label kernel."""

    return (labels[:, None] == labels[None, :]).astype(jnp.float64)


def center_gram(kernel: jnp.ndarray) -> jnp.ndarray:
    """Double-center a square Gram matrix."""

    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"kernel must be square, got {kernel.shape}")
    return kernel - jnp.mean(kernel, axis=0, keepdims=True) - jnp.mean(
        kernel,
        axis=1,
        keepdims=True,
    ) + jnp.mean(kernel)


def centered_alignment(
    first: jnp.ndarray,
    second: jnp.ndarray,
    *,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Return centered kernel alignment between two Gram matrices."""

    first_centered = center_gram(first)
    second_centered = center_gram(second)
    numerator = jnp.sum(first_centered * second_centered)
    denominator = jnp.linalg.norm(first_centered) * jnp.linalg.norm(second_centered)
    return jnp.where(denominator > eps, numerator / denominator, jnp.nan)


def kernel_target_alignment(
    kernel: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Return centered alignment between a state kernel and label kernel."""

    return centered_alignment(kernel, label_kernel(labels), eps=eps)


def cka_from_kernels(
    first: jnp.ndarray,
    second: jnp.ndarray,
    *,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Return kernel CKA, i.e. centered alignment for valid kernels."""

    return centered_alignment(first, second, eps=eps)


def effective_rank(
    kernel: jnp.ndarray,
    *,
    centered: bool = False,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Return ``tr(K)^2 / tr(K^2)`` from nonnegative Gram eigenvalues."""

    matrix = center_gram(kernel) if centered else kernel
    matrix = (matrix + matrix.T) / 2
    eigvals = jnp.maximum(jnp.linalg.eigvalsh(matrix), 0.0)
    trace = jnp.sum(eigvals)
    trace_sq = jnp.sum(eigvals**2)
    return jnp.where(trace_sq > eps, trace**2 / trace_sq, 0.0)


def fidelity_distribution_summary(kernel: jnp.ndarray, labels: jnp.ndarray) -> dict[str, Any]:
    """Summarize within-class and between-class off-diagonal fidelities."""

    if kernel.shape[0] != labels.shape[0]:
        raise ValueError("kernel and labels must contain the same number of samples")
    n_samples = int(labels.shape[0])
    upper = jnp.triu(jnp.ones((n_samples, n_samples), dtype=bool), k=1)
    same = labels[:, None] == labels[None, :]
    within = kernel[upper & same]
    between = kernel[upper & ~same]
    return {
        "within": value_summary(within),
        "between": value_summary(between),
        "mean_gap_within_minus_between": _safe_mean(within) - _safe_mean(between),
    }


def kernel_geometry_summary(kernel: jnp.ndarray, labels: jnp.ndarray) -> dict[str, Any]:
    """Return compact fidelity-kernel geometry diagnostics."""

    return {
        "target_alignment": kernel_target_alignment(kernel, labels),
        "effective_rank": effective_rank(kernel),
        "centered_effective_rank": effective_rank(kernel, centered=True),
        "fidelity_distributions": fidelity_distribution_summary(kernel, labels),
    }


def layerwise_kernel_summaries(states: jnp.ndarray, labels: jnp.ndarray) -> list[dict[str, Any]]:
    """Return one fidelity-kernel summary per state layer/stage."""

    if states.ndim != 3:
        raise ValueError(f"states must have shape (n_samples, n_layers, dim), got {states.shape}")
    summaries = []
    for layer_index in range(states.shape[1]):
        kernel = fidelity_kernel(states[:, layer_index, :])
        summary = {"layer_index": layer_index, **kernel_geometry_summary(kernel, labels)}
        summaries.append(summary)
    return summaries


def adjacent_layer_cka(states: jnp.ndarray) -> jnp.ndarray:
    """Return CKA between adjacent layers/stages of one state trace."""

    if states.ndim != 3:
        raise ValueError(f"states must have shape (n_samples, n_layers, dim), got {states.shape}")
    values = []
    for layer_index in range(states.shape[1] - 1):
        first = fidelity_kernel(states[:, layer_index, :])
        second = fidelity_kernel(states[:, layer_index + 1, :])
        values.append(cka_from_kernels(first, second))
    return jnp.asarray(values, dtype=jnp.float64)


def paired_stage_cka(first_states: jnp.ndarray, second_states: jnp.ndarray) -> jnp.ndarray:
    """Return CKA for corresponding layers of two state traces."""

    if first_states.shape != second_states.shape:
        raise ValueError(
            "state traces must have matching shapes, "
            f"got {first_states.shape} and {second_states.shape}"
        )
    values = []
    for layer_index in range(first_states.shape[1]):
        first = fidelity_kernel(first_states[:, layer_index, :])
        second = fidelity_kernel(second_states[:, layer_index, :])
        values.append(cka_from_kernels(first, second))
    return jnp.asarray(values, dtype=jnp.float64)


def logit_trajectory_summary(
    scores: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    stage_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Summarize sample trajectories through projector-score space."""

    if scores.ndim != 3:
        raise ValueError(
            "scores must have shape (n_samples, n_stages, n_classes), "
            f"got {scores.shape}"
        )
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must contain the same number of samples")
    if stage_names is None:
        names = [f"stage_{index}" for index in range(scores.shape[1])]
    else:
        names = list(stage_names)
        if len(names) != scores.shape[1]:
            raise ValueError("stage_names length must match scores.shape[1]")

    if scores.shape[1] < 2:
        movement = jnp.zeros((scores.shape[0], 0), dtype=scores.dtype)
    else:
        movement = jnp.linalg.norm(jnp.diff(scores, axis=1), axis=-1)
    path_length = jnp.sum(movement, axis=-1)
    final_predictions = jnp.argmax(scores[:, -1, :], axis=-1)
    correct = final_predictions == labels
    transitions = [f"{names[index]}->{names[index + 1]}" for index in range(max(0, len(names) - 1))]
    return {
        "stage_names": names,
        "transition_names": transitions,
        "mean_step_movement": jnp.mean(movement, axis=0),
        "total_path_length": value_summary(path_length),
        "correct_total_path_length": value_summary(path_length[correct]),
        "incorrect_total_path_length": value_summary(path_length[~correct]),
        "final_accuracy": jnp.mean(correct),
    }


def projector_probe_summary(scores: jnp.ndarray, labels: jnp.ndarray) -> dict[str, Any]:
    """Return layerwise projector-rule accuracy and confidence summaries."""

    if scores.ndim != 3:
        raise ValueError(
            "scores must have shape (n_samples, n_layers, n_classes), "
            f"got {scores.shape}"
        )
    top_scores = jnp.max(scores, axis=-1)
    return {
        "accuracy_by_layer": projector_accuracy_by_layer(scores, labels),
        "mean_top_score_by_layer": jnp.mean(top_scores, axis=0),
        "top_score_summary": [
            value_summary(top_scores[:, index]) for index in range(scores.shape[1])
        ],
    }


def hamiltonian_matrix_for_encoder(
    encoder: object,
    sample: jnp.ndarray,
    *,
    n_qubits: int,
) -> jnp.ndarray:
    """Return the padded Hamiltonian matrix used by a full-matrix encoder."""

    if isinstance(encoder, SymmetricHamiltonianEncoder):
        H = build_H_sym(sample, rectangular_strategy=encoder.rectangular_strategy)
    elif isinstance(encoder, BlockHamiltonianEncoder):
        H = build_H_block(sample)
    else:
        raise TypeError("Hamiltonian spectral-state diagnostics require H_sym or H_block encoders")
    target_dim = 2**n_qubits
    if H.shape[0] > target_dim:
        raise ValueError(f"Hamiltonian dimension {H.shape[0]} exceeds model dimension {target_dim}")
    if H.shape[0] == target_dim:
        return H
    return pad_square_matrix_to_power_of_two(jnp.pad(H, ((0, target_dim - H.shape[0]),) * 2))


def hamiltonian_upload_time(
    encoder: object,
    encoder_params: dict[str, jnp.ndarray],
    *,
    layer_index: int,
    reupload_depth: int,
) -> jnp.ndarray:
    """Return the upload time used by a full-matrix Hamiltonian encoder."""

    if isinstance(encoder, (SymmetricHamiltonianEncoder, BlockHamiltonianEncoder)):
        return encoder._time(encoder_params, layer_index, reupload_depth)  # noqa: SLF001
    raise TypeError("Hamiltonian upload times require H_sym or H_block encoders")


def hamiltonian_spectral_state_summary(
    encoder: object,
    encoder_params: dict[str, jnp.ndarray],
    sample: jnp.ndarray,
    pre_upload_states: jnp.ndarray,
    post_upload_states: jnp.ndarray,
    *,
    n_qubits: int,
    reupload_depth: int,
) -> list[dict[str, Any]]:
    """Summarize eigenbasis occupation and phase increments per layer."""

    if pre_upload_states.shape != post_upload_states.shape:
        raise ValueError("pre_upload_states and post_upload_states must have matching shapes")
    H = hamiltonian_matrix_for_encoder(encoder, sample, n_qubits=n_qubits)
    eigvals, eigvecs = jnp.linalg.eigh(H)
    summaries = []
    for layer_index in range(pre_upload_states.shape[0]):
        before = eigvecs.conj().T @ pre_upload_states[layer_index]
        after = eigvecs.conj().T @ post_upload_states[layer_index]
        occ_before = jnp.abs(before) ** 2
        occ_after = jnp.abs(after) ** 2
        time = hamiltonian_upload_time(
            encoder,
            encoder_params,
            layer_index=layer_index,
            reupload_depth=reupload_depth,
        )
        phase = 0.5 * time * eigvals
        summaries.append(
            {
                "layer_index": layer_index,
                "occupation_before": value_summary(occ_before),
                "occupation_after": value_summary(occ_after),
                "occupation_l1_change": jnp.sum(jnp.abs(occ_after - occ_before)),
                "phase_increment": value_summary(phase),
                "phase_increment_abs": value_summary(jnp.abs(phase)),
            }
        )
    return summaries


def value_summary(values: jnp.ndarray) -> dict[str, Any]:
    """Return compact scalar summaries for a vector-like array."""

    values = jnp.ravel(values)
    count = int(values.shape[0])
    if count == 0:
        nan = jnp.asarray(jnp.nan)
        return {
            "count": 0,
            "mean": nan,
            "std": nan,
            "median": nan,
            "q25": nan,
            "q75": nan,
            "min": nan,
            "max": nan,
        }
    return {
        "count": count,
        "mean": jnp.mean(values),
        "std": jnp.std(values),
        "median": jnp.median(values),
        "q25": jnp.quantile(values, 0.25),
        "q75": jnp.quantile(values, 0.75),
        "min": jnp.min(values),
        "max": jnp.max(values),
    }


def aggregate_layer_summaries(
    sample_summaries: list[list[dict[str, Any]]],
    field: str,
) -> list[dict[str, Any]]:
    """Aggregate a scalar field from per-sample layer summaries."""

    if not sample_summaries:
        return []
    n_layers = len(sample_summaries[0])
    records = []
    for layer_index in range(n_layers):
        values = jnp.asarray([sample[layer_index][field] for sample in sample_summaries])
        records.append({"layer_index": layer_index, field: value_summary(values)})
    return records


def _safe_mean(values: jnp.ndarray) -> jnp.ndarray:
    if values.shape[0] == 0:
        return jnp.asarray(jnp.nan)
    return jnp.mean(values)
