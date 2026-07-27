"""Training loop utilities for re-uploading experiments."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ham_embed_spectral.config import ReuploadingModelConfig
from ham_embed_spectral.data.pendigits import PendigitsSplit
from ham_embed_spectral.experiments.metrics import accuracy_from_probs, cross_entropy_from_probs
from ham_embed_spectral.models.reuploading import probabilities, readout_mass_diagnostics
from ham_embed_spectral.quantum.encoders import EncoderProtocol


@flax.struct.dataclass
class TrainState:
    """Minimal Flax-compatible train state for pure-JAX quantum models."""

    step: jnp.ndarray
    params: Any
    opt_state: Any


@dataclass(frozen=True)
class TrainingLoopConfig:
    """Runtime controls for minibatch training."""

    steps: int = 100
    batch_size: int = 32
    eval_batch_size: int = 128
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    log_every: int = 10
    eval_every: int = 50


def make_optimizer(config: TrainingLoopConfig) -> optax.GradientTransformation:
    """Build the optimizer used by the training loop."""

    if config.weight_decay:
        return optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
    return optax.adam(config.learning_rate)


def create_train_state(
    params: Any,
    optimizer: optax.GradientTransformation,
) -> TrainState:
    """Create a Flax-compatible train state."""

    return TrainState(
        step=jnp.asarray(0, dtype=jnp.int32),
        params=params,
        opt_state=optimizer.init(params),
    )


def make_train_step(
    encoder: EncoderProtocol,
    model_config: ReuploadingModelConfig,
    optimizer: optax.GradientTransformation,
):
    """Create a jitted train-step closure for a fixed model configuration."""

    @jax.jit
    def train_step(state: TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray):
        def loss_fn(params):
            probs = probabilities(params, encoder, batch_x, model_config)
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


def make_predict_step(
    encoder: EncoderProtocol,
    model_config: ReuploadingModelConfig,
):
    """Create a jitted prediction closure for evaluation."""

    @jax.jit
    def predict_step(params: Any, batch_x: jnp.ndarray) -> jnp.ndarray:
        return probabilities(params, encoder, batch_x, model_config)

    return predict_step


def make_readout_diagnostics_step(
    encoder: EncoderProtocol,
    model_config: ReuploadingModelConfig,
):
    """Create a jitted readout-diagnostics closure for evaluation."""

    @jax.jit
    def diagnostics_step(params: Any, batch_x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return readout_mass_diagnostics(params, encoder, batch_x, model_config)

    return diagnostics_step


def minibatches(
    split: PendigitsSplit,
    batch_size: int,
    *,
    seed: int,
    shuffle: bool,
    drop_remainder: bool = False,
) -> Iterator[tuple[jnp.ndarray, jnp.ndarray]]:
    """Yield minibatches from a split."""

    n_examples = int(split.y.shape[0])
    indices = np.arange(n_examples)
    if shuffle:
        indices = np.random.default_rng(seed).permutation(indices)

    for start in range(0, n_examples, batch_size):
        batch_indices = indices[start : start + batch_size]
        if drop_remainder and batch_indices.shape[0] < batch_size:
            continue
        yield split.x[batch_indices], split.y[batch_indices]


def epoch_minibatches(
    split: PendigitsSplit,
    batch_size: int,
    *,
    seed: int,
    shuffle: bool = True,
    drop_remainder: bool = False,
) -> Iterator[tuple[jnp.ndarray, jnp.ndarray]]:
    """Yield an infinite deterministic stream of epoch-style minibatches."""

    epoch = 0
    while True:
        yield from minibatches(
            split,
            batch_size,
            seed=seed + epoch,
            shuffle=shuffle,
            drop_remainder=drop_remainder,
        )
        epoch += 1


def evaluate(
    params: Any,
    split: PendigitsSplit,
    *,
    predict_step,
    diagnostics_step=None,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate loss and accuracy over a split.

    ``diagnostics_step`` is optional so leakage-problem diagnostics can be
    recorded without changing the loss or the classification readout contract.
    """

    n_examples = int(split.y.shape[0])
    if n_examples == 0:
        return {"loss": float("nan"), "accuracy": float("nan"), "n_examples": 0.0}

    total_loss = 0.0
    total_correct = 0.0
    total_valid_mass = 0.0
    total_leakage_mass = 0.0
    for batch_x, batch_y in minibatches(split, batch_size, seed=0, shuffle=False):
        probs = predict_step(params, batch_x)
        batch_n = int(batch_y.shape[0])
        total_loss += float(cross_entropy_from_probs(probs, batch_y)) * batch_n
        total_correct += float(jnp.sum(jnp.argmax(probs, axis=-1) == batch_y))
        if diagnostics_step is not None:
            diagnostics = diagnostics_step(params, batch_x)
            total_valid_mass += float(jnp.sum(diagnostics["valid_mass"]))
            total_leakage_mass += float(jnp.sum(diagnostics["leakage_mass"]))

    metrics = {
        "loss": total_loss / n_examples,
        "accuracy": total_correct / n_examples,
        "n_examples": float(n_examples),
    }
    if diagnostics_step is not None:
        metrics.update(
            {
                "mean_valid_readout_mass": total_valid_mass / n_examples,
                "mean_readout_leakage_mass": total_leakage_mass / n_examples,
            }
        )
    return metrics


def limit_split(split: PendigitsSplit, max_examples: int | None) -> PendigitsSplit:
    """Return at most ``max_examples`` examples from a split."""

    if max_examples is None or max_examples >= int(split.y.shape[0]):
        return split
    return PendigitsSplit(x=split.x[:max_examples], y=split.y[:max_examples])
