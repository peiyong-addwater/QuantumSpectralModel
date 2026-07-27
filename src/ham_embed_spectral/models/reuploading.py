"""Basic dense statevector re-uploading model."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ham_embed_spectral.config import ReuploadingModelConfig
from ham_embed_spectral.quantum.encoders import EncoderProtocol
from ham_embed_spectral.quantum.mixers import apply_su4_mixer, init_su4_mixer_params
from ham_embed_spectral.quantum.readout import projector_leakage_mass, projector_probabilities
from ham_embed_spectral.quantum.statevector import initial_state_vector


def init_reuploading_params(
    key: jax.Array,
    encoder: EncoderProtocol,
    config: ReuploadingModelConfig,
    dtype: jnp.dtype = jnp.float64,
) -> dict[str, dict[str, jnp.ndarray] | jnp.ndarray]:
    """Initialize encoder and SU(4) mixer parameters."""

    n_qubits = config.n_qubits or encoder.n_qubits(config.input_shape)
    encoder_key, mixer_key = jax.random.split(key)
    encoder_params = encoder.init_params(
        encoder_key,
        input_shape=config.input_shape,
        reupload_depth=config.reupload_depth,
        dtype=dtype,
    )
    return {
        "encoder": encoder_params,
        "theta_su": init_su4_mixer_params(
            mixer_key,
            n_qubits=n_qubits,
            reupload_depth=config.reupload_depth,
            scale=config.mixer_scale,
            dtype=dtype,
        ),
    }


def forward_state(
    params: dict[str, dict[str, jnp.ndarray] | jnp.ndarray],
    encoder: EncoderProtocol,
    sample: jnp.ndarray,
    config: ReuploadingModelConfig,
) -> jnp.ndarray:
    """Return the final state for one sample."""

    trace = layerwise_states(params, encoder, sample, config)
    if config.reupload_depth == 0:
        return trace["initial"]
    return trace["post_mixer"][-1]


def layerwise_states(
    params: dict[str, dict[str, jnp.ndarray] | jnp.ndarray],
    encoder: EncoderProtocol,
    sample: jnp.ndarray,
    config: ReuploadingModelConfig,
) -> dict[str, jnp.ndarray]:
    """Return initial, post-upload, and post-mixer states for one sample.

    ``post_upload[ell]`` is the state immediately after the data upload in
    layer ``ell``. ``post_mixer[ell]`` is the state immediately after the
    corresponding trainable SU(4) mixer. The final post-mixer state matches
    :func:`forward_state` for positive re-upload depth.
    """

    n_qubits = config.n_qubits or encoder.n_qubits(config.input_shape)
    initial = initial_state_vector(config.initial_state, n_qubits=n_qubits)
    state = initial
    encoder_params = params["encoder"]
    theta_su = params["theta_su"]
    post_upload = []
    post_mixer = []
    for layer_index in range(config.reupload_depth):
        state = encoder.apply_to_state(
            encoder_params,
            state,
            sample,
            layer_index=layer_index,
            n_qubits=n_qubits,
            reupload_depth=config.reupload_depth,
        )
        post_upload.append(state)
        state = apply_su4_mixer(state, theta_su[layer_index], n_qubits=n_qubits)
        post_mixer.append(state)

    if post_upload:
        upload_trace = jnp.stack(post_upload)
        mixer_trace = jnp.stack(post_mixer)
    else:
        dim = state.shape[-1]
        upload_trace = jnp.zeros((0, dim), dtype=state.dtype)
        mixer_trace = jnp.zeros((0, dim), dtype=state.dtype)
    return {"initial": initial, "post_upload": upload_trace, "post_mixer": mixer_trace}


def probabilities(
    params: dict[str, dict[str, jnp.ndarray] | jnp.ndarray],
    encoder: EncoderProtocol,
    samples: jnp.ndarray,
    config: ReuploadingModelConfig,
) -> jnp.ndarray:
    """Return class probabilities for one sample or a batch."""

    def one(sample: jnp.ndarray) -> jnp.ndarray:
        state = forward_state(params, encoder, sample, config)
        return projector_probabilities(
            state,
            n_classes=config.n_classes,
            renormalize=config.projector_renormalize,
        )

    if samples.shape == config.input_shape:
        return one(samples)
    return jax.vmap(one)(samples)


def readout_leakage_masses(
    params: dict[str, dict[str, jnp.ndarray] | jnp.ndarray],
    encoder: EncoderProtocol,
    samples: jnp.ndarray,
    config: ReuploadingModelConfig,
) -> jnp.ndarray:
    """Return projector-readout leakage mass for one sample or a batch.

    The leakage mass is the total probability assigned to unused label-register
    states. Current classification losses do not optimize this quantity
    directly; this function is an API point for future leakage-problem studies.
    """

    def one(sample: jnp.ndarray) -> jnp.ndarray:
        state = forward_state(params, encoder, sample, config)
        return projector_leakage_mass(state, n_classes=config.n_classes)

    if samples.shape == config.input_shape:
        return one(samples)
    return jax.vmap(one)(samples)


def readout_mass_diagnostics(
    params: dict[str, dict[str, jnp.ndarray] | jnp.ndarray],
    encoder: EncoderProtocol,
    samples: jnp.ndarray,
    config: ReuploadingModelConfig,
) -> dict[str, jnp.ndarray]:
    """Return valid-class and leakage readout masses.

    These diagnostics keep the leakage problem visible without changing the
    readout protocol used by the main classification experiments.
    """

    leakage_mass = readout_leakage_masses(params, encoder, samples, config)
    return {
        "valid_mass": 1.0 - leakage_mass,
        "leakage_mass": leakage_mass,
    }


def cross_entropy_loss(
    params: dict[str, dict[str, jnp.ndarray] | jnp.ndarray],
    encoder: EncoderProtocol,
    samples: jnp.ndarray,
    labels: jnp.ndarray,
    config: ReuploadingModelConfig,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Mean cross-entropy from projector probabilities."""

    probs = probabilities(params, encoder, samples, config)
    selected = jnp.take_along_axis(probs, labels[..., None], axis=-1)[..., 0]
    return -jnp.mean(jnp.log(jnp.maximum(selected, eps)))
