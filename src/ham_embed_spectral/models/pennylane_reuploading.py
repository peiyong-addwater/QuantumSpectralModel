"""Experimental PennyLane backend for re-uploading models.

This module mirrors :mod:`ham_embed_spectral.models.reuploading` while keeping
the existing dense JAX statevector implementation as the source of truth for
model semantics. SU(4)-coordinate uploads and mixers use
``qml.SpecialUnitary``; sample-conditioned Hamiltonian uploads use PennyLane
matrix evolution gates.
"""

from __future__ import annotations

# ruff: noqa: E402, I001

from collections.abc import Callable, Sequence
from typing import Any

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import optax
import pennylane as qml

from ham_embed_spectral.config import ReuploadingModelConfig
from ham_embed_spectral.experiments.metrics import accuracy_from_probs, cross_entropy_from_probs
from ham_embed_spectral.quantum.encoders import (
    BlockHamiltonianEncoder,
    EncoderProtocol,
    FixedRyEncoder,
    FixedRyRzEncoder,
    NonOverlapPatchBlockHamiltonianEncoder,
    PatchSU4Encoder,
    SymmetricHamiltonianEncoder,
    TrainableFrequencyRyEncoder,
    TrainablePatchSU4Encoder,
    build_H_block,
    build_H_sym,
    extract_four_patches,
)
from ham_embed_spectral.quantum.mixers import brickwall_pairs
from ham_embed_spectral.quantum.readout import projector_probabilities
from ham_embed_spectral.quantum.su4 import SU4_PARAM_COUNT

StateQNode = Callable[[dict[str, Any], jnp.ndarray], jnp.ndarray]


def pennylane_forward_state(
    params: dict[str, Any],
    encoder: EncoderProtocol,
    sample: jnp.ndarray,
    config: ReuploadingModelConfig,
) -> jnp.ndarray:
    """Return the PennyLane final state for one sample.

    The returned state uses PennyLane's default computational basis ordering,
    which matches the repository dense-state convention with wire ``0`` as the
    most significant qubit.
    """

    qnode = _make_state_qnode(encoder, config)
    return qnode(params, sample)


def pennylane_probabilities(
    params: dict[str, Any],
    encoder: EncoderProtocol,
    samples: jnp.ndarray,
    config: ReuploadingModelConfig,
) -> jnp.ndarray:
    """Return projector-readout probabilities from the PennyLane backend."""

    qnode = _make_state_qnode(encoder, config)
    return _probabilities_from_qnode(qnode, params, samples, config)


def make_pennylane_train_step(
    encoder: EncoderProtocol,
    model_config: ReuploadingModelConfig,
    optimizer: optax.GradientTransformation,
):
    """Create a jitted Optax train-step closure for the PennyLane backend."""

    qnode = _make_state_qnode(encoder, model_config)

    @jax.jit
    def train_step(state, batch_x: jnp.ndarray, batch_y: jnp.ndarray):
        def loss_fn(params):
            probs = _probabilities_from_qnode(qnode, params, batch_x, model_config)
            loss = cross_entropy_from_probs(probs, batch_y)
            return loss, probs

        (loss, probs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        metrics = {
            "loss": loss,
            "accuracy": accuracy_from_probs(probs, batch_y),
            "grad_norm": optax.tree.norm(grads),
        }
        return state.replace(step=state.step + 1, params=params, opt_state=opt_state), metrics

    return train_step


def make_pennylane_predict_step(
    encoder: EncoderProtocol,
    model_config: ReuploadingModelConfig,
):
    """Create a jitted projector-probability prediction closure."""

    qnode = _make_state_qnode(encoder, model_config)

    @jax.jit
    def predict_step(params: dict[str, Any], batch_x: jnp.ndarray) -> jnp.ndarray:
        return _probabilities_from_qnode(qnode, params, batch_x, model_config)

    return predict_step


def _make_state_qnode(
    encoder: EncoderProtocol,
    config: ReuploadingModelConfig,
) -> StateQNode:
    n_qubits = config.n_qubits or encoder.n_qubits(config.input_shape)
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, interface="jax")
    def circuit(params: dict[str, Any], sample: jnp.ndarray) -> jnp.ndarray:
        _apply_initial_state(config.initial_state, n_qubits)
        encoder_params = params["encoder"]
        theta_su = params["theta_su"]
        for layer_index in range(config.reupload_depth):
            _apply_encoder_upload(
                encoder,
                encoder_params,
                sample,
                layer_index=layer_index,
                n_qubits=n_qubits,
                reupload_depth=config.reupload_depth,
            )
            _apply_su4_mixer(theta_su[layer_index], n_qubits=n_qubits)
        return qml.state()

    return circuit


def _probabilities_from_qnode(
    qnode: StateQNode,
    params: dict[str, Any],
    samples: jnp.ndarray,
    config: ReuploadingModelConfig,
) -> jnp.ndarray:
    if samples.shape == config.input_shape:
        states = qnode(params, samples)
    else:
        states = jax.vmap(qnode, in_axes=(None, 0))(params, samples)
    return projector_probabilities(
        states,
        n_classes=config.n_classes,
        renormalize=config.projector_renormalize,
    )


def _apply_initial_state(name: str, n_qubits: int) -> None:
    if name == "zero":
        return
    if name == "plus":
        for wire in range(n_qubits):
            qml.Hadamard(wires=wire)
        return
    raise ValueError(f"unknown initial state {name!r}; expected 'zero' or 'plus'")


def _apply_encoder_upload(
    encoder: EncoderProtocol,
    params: dict[str, jnp.ndarray],
    sample: jnp.ndarray,
    *,
    layer_index: int,
    n_qubits: int,
    reupload_depth: int,
) -> None:
    if isinstance(encoder, FixedRyEncoder):
        _apply_fixed_ry_encoder(encoder, sample, n_qubits=n_qubits)
        return
    if isinstance(encoder, FixedRyRzEncoder):
        _apply_fixed_ry_rz_encoder(encoder, sample, n_qubits=n_qubits)
        return
    if isinstance(encoder, TrainableFrequencyRyEncoder):
        _apply_trainable_frequency_ry_encoder(
            params,
            sample,
            layer_index=layer_index,
            n_qubits=n_qubits,
        )
        return
    if isinstance(encoder, PatchSU4Encoder):
        _apply_patch_su4_encoder(encoder, sample, n_qubits=n_qubits)
        return
    if isinstance(encoder, TrainablePatchSU4Encoder):
        _apply_trainable_patch_su4_encoder(
            encoder,
            params,
            sample,
            layer_index=layer_index,
            n_qubits=n_qubits,
        )
        return
    if isinstance(encoder, NonOverlapPatchBlockHamiltonianEncoder):
        _apply_nonoverlap_patch_block_hamiltonian_encoder(
            encoder,
            params,
            sample,
            layer_index=layer_index,
            n_qubits=n_qubits,
            reupload_depth=reupload_depth,
        )
        return
    if isinstance(encoder, SymmetricHamiltonianEncoder):
        H_sym = build_H_sym(sample, rectangular_strategy=encoder.rectangular_strategy)
        _apply_hamiltonian_evolution(
            H_sym,
            encoder._time(params, layer_index, reupload_depth),
            wires=tuple(range(n_qubits)),
        )
        return
    if isinstance(encoder, BlockHamiltonianEncoder):
        H_block = build_H_block(sample)
        _apply_hamiltonian_evolution(
            H_block,
            encoder._time(params, layer_index, reupload_depth),
            wires=tuple(range(n_qubits)),
        )
        return
    raise TypeError(f"unsupported PennyLane encoder type: {type(encoder).__name__}")


def _apply_fixed_ry_encoder(
    encoder: FixedRyEncoder,
    sample: jnp.ndarray,
    *,
    n_qubits: int,
) -> None:
    features = jnp.ravel(sample)
    if features.shape[0] != n_qubits:
        raise ValueError(f"FixedRyEncoder expected {n_qubits} features, got {features.shape[0]}")
    for wire, value in enumerate(features):
        qml.RY(encoder.alpha * value, wires=wire)


def _apply_fixed_ry_rz_encoder(
    encoder: FixedRyRzEncoder,
    sample: jnp.ndarray,
    *,
    n_qubits: int,
) -> None:
    features = jnp.ravel(sample)
    if features.shape[0] != n_qubits:
        raise ValueError(f"FixedRyRzEncoder expected {n_qubits} features, got {features.shape[0]}")
    for wire, value in enumerate(features):
        qml.RY(encoder.alpha * value, wires=wire)
        qml.RZ(encoder.beta * value, wires=wire)


def _apply_trainable_frequency_ry_encoder(
    params: dict[str, jnp.ndarray],
    sample: jnp.ndarray,
    *,
    layer_index: int,
    n_qubits: int,
) -> None:
    features = jnp.ravel(sample)
    gamma = params["gamma"][layer_index]
    if features.shape[0] != n_qubits or gamma.shape[0] != n_qubits:
        raise ValueError("TrainableFrequencyRyEncoder feature/gamma size mismatch")
    for wire, value in enumerate(features):
        qml.RY(gamma[wire] * value, wires=wire)


def _apply_patch_su4_encoder(
    encoder: PatchSU4Encoder,
    sample: jnp.ndarray,
    *,
    n_qubits: int,
) -> None:
    if n_qubits != 8:
        raise ValueError(f"PatchSU4Encoder expects 8 qubits, got {n_qubits}")
    for patch_index, patch in enumerate(extract_four_patches(sample)):
        qml.SpecialUnitary(
            encoder.patch_angles(patch),
            wires=(2 * patch_index, 2 * patch_index + 1),
        )


def _apply_trainable_patch_su4_encoder(
    encoder: TrainablePatchSU4Encoder,
    params: dict[str, jnp.ndarray],
    sample: jnp.ndarray,
    *,
    layer_index: int,
    n_qubits: int,
) -> None:
    if n_qubits != 8:
        raise ValueError(f"TrainablePatchSU4Encoder expects 8 qubits, got {n_qubits}")
    for patch_index, patch in enumerate(extract_four_patches(sample)):
        qml.SpecialUnitary(
            encoder.patch_angles(params, patch, layer_index, patch_index),
            wires=(2 * patch_index, 2 * patch_index + 1),
        )


def _apply_nonoverlap_patch_block_hamiltonian_encoder(
    encoder: NonOverlapPatchBlockHamiltonianEncoder,
    params: dict[str, jnp.ndarray],
    sample: jnp.ndarray,
    *,
    layer_index: int,
    n_qubits: int,
    reupload_depth: int,
) -> None:
    if n_qubits != 8:
        raise ValueError(
            f"NonOverlapPatchBlockHamiltonianEncoder expects 8 qubits, got {n_qubits}"
        )
    for patch_index, patch in enumerate(extract_four_patches(sample)):
        _apply_hamiltonian_evolution(
            build_H_block(patch),
            encoder._time(params, layer_index, patch_index, reupload_depth),
            wires=(2 * patch_index, 2 * patch_index + 1),
        )


def _apply_hamiltonian_evolution(
    H: jnp.ndarray,
    t: jnp.ndarray,
    *,
    wires: Sequence[int],
) -> None:
    wire_tuple = tuple(wires)
    H = _pad_square_matrix_to_dim(H, 2 ** len(wire_tuple))
    qml.evolve(qml.Hermitian(H, wires=wire_tuple), coeff=0.5 * t)


def _pad_square_matrix_to_dim(matrix: jnp.ndarray, target_dim: int) -> jnp.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {matrix.shape}")
    current_dim = matrix.shape[0]
    if current_dim == target_dim:
        return matrix
    if current_dim > target_dim:
        raise ValueError(f"cannot pad matrix dimension {current_dim} down to {target_dim}")
    return jnp.pad(matrix, ((0, target_dim - current_dim), (0, target_dim - current_dim)))


def _apply_su4_mixer(theta_layer: jnp.ndarray, *, n_qubits: int) -> None:
    pairs = brickwall_pairs(n_qubits)
    expected_shape = (len(pairs), SU4_PARAM_COUNT)
    if theta_layer.shape != expected_shape:
        raise ValueError(f"theta_layer must have shape {expected_shape}, got {theta_layer.shape}")
    for block_index, wires in enumerate(pairs):
        qml.SpecialUnitary(theta_layer[block_index], wires=wires)
