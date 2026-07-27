"""Data encoders for dense statevector experiments."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Protocol

import jax
import jax.numpy as jnp

from ham_embed_spectral.quantum.statevector import (
    apply_local_unitary,
    apply_unitary,
    next_power_of_two,
    num_qubits_from_dim,
    pad_square_matrix_to_power_of_two,
)
from ham_embed_spectral.quantum.su4 import su4_from_params


class EncoderProtocol(Protocol):
    """Protocol shared by statevector data encoders."""

    def n_qubits(self, input_shape: tuple[int, ...]) -> int:
        """Return the natural number of qubits for ``input_shape``."""

    def init_params(
        self,
        key: jax.Array,
        input_shape: tuple[int, ...],
        reupload_depth: int,
        dtype: jnp.dtype = jnp.float64,
    ) -> dict[str, jnp.ndarray]:
        """Initialize encoder-specific trainable parameters."""

    def apply_to_state(
        self,
        params: dict[str, jnp.ndarray],
        state: jnp.ndarray,
        sample: jnp.ndarray,
        layer_index: int,
        n_qubits: int,
        reupload_depth: int | None = None,
    ) -> jnp.ndarray:
        """Apply the encoding upload for one sample and layer."""


def _feature_count(input_shape: tuple[int, ...]) -> int:
    return reduce(mul, input_shape, 1)


def _rotation_y(angle: jnp.ndarray) -> jnp.ndarray:
    half = angle / 2
    return jnp.asarray(
        [[jnp.cos(half), -jnp.sin(half)], [jnp.sin(half), jnp.cos(half)]],
        dtype=jnp.result_type(angle, jnp.complex128),
    )


def _rotation_z(angle: jnp.ndarray) -> jnp.ndarray:
    half = angle / 2
    return jnp.asarray(
        [[jnp.exp(-1j * half), 0], [0, jnp.exp(1j * half)]],
        dtype=jnp.result_type(angle, jnp.complex128),
    )


def matrix_time_evolution(
    H: jnp.ndarray,
    t: jnp.ndarray,
    *,
    symmetrize: bool = False,
) -> jnp.ndarray:
    """Compute ``exp(-i H t / 2)`` through a Hermitian eigendecomposition."""

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be square, got {H.shape}")
    if symmetrize:
        H = (H + H.conj().T) / 2
    eigvals, eigvecs = jnp.linalg.eigh(H)
    phases = jnp.exp(-0.5j * t * eigvals)
    return (eigvecs * phases) @ eigvecs.conj().T


def pad_unitary_to_dim(unitary: jnp.ndarray, target_dim: int) -> jnp.ndarray:
    """Embed a unitary in a larger Hilbert space with identity on padded states.

    This is a direct-sum embedding ``U -> U \\oplus I``, not a Kronecker/tensor
    product with idle qubits. It is the unitary produced by first zero-padding a
    Hamiltonian, ``H -> H \\oplus 0``, and then computing ``exp(-i H t / 2)``.
    """

    if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
        raise ValueError(f"unitary must be square, got {unitary.shape}")
    current_dim = unitary.shape[0]
    if current_dim == target_dim:
        return unitary
    if current_dim > target_dim:
        raise ValueError(f"cannot pad unitary dimension {current_dim} down to {target_dim}")
    result = jnp.eye(target_dim, dtype=unitary.dtype)
    return result.at[:current_dim, :current_dim].set(unitary)


def pad_columns_to_square(M: jnp.ndarray) -> jnp.ndarray:
    """Pad a rectangular ``p x q`` matrix with zero columns to ``p x p``."""

    if M.ndim != 2:
        raise ValueError(f"expected a matrix sample, got shape {M.shape}")
    rows, cols = M.shape
    if cols > rows:
        raise ValueError(
            "pad_columns_to_square requires rows >= cols; choose a different explicit strategy"
        )
    if rows == cols:
        return M
    return jnp.pad(M, ((0, 0), (0, rows - cols)))


def build_H_sym(
    sample: jnp.ndarray,
    rectangular_strategy: str = "pad_columns_to_square",
) -> jnp.ndarray:
    """Build the symmetric Hamiltonian ``H_sym = (M + M.T) / 2``."""

    M = jnp.asarray(sample)
    if M.ndim != 2:
        raise ValueError(f"H_sym requires a matrix sample, got shape {M.shape}")
    if M.shape[0] != M.shape[1]:
        if rectangular_strategy != "pad_columns_to_square":
            raise ValueError(f"unknown rectangular strategy {rectangular_strategy!r}")
        M = pad_columns_to_square(M)
    H_sym = (M + M.T) / 2
    return pad_square_matrix_to_power_of_two(H_sym)


def build_H_block(sample: jnp.ndarray) -> jnp.ndarray:
    """Build the block Hamiltonian ``[[0, M], [M.T, 0]]`` with power-of-two pad."""

    M = jnp.asarray(sample)
    if M.ndim != 2:
        raise ValueError(f"H_block requires a matrix sample, got shape {M.shape}")
    rows, cols = M.shape
    top = jnp.concatenate([jnp.zeros((rows, rows), dtype=M.dtype), M], axis=1)
    bottom = jnp.concatenate([M.T, jnp.zeros((cols, cols), dtype=M.dtype)], axis=1)
    return pad_square_matrix_to_power_of_two(jnp.concatenate([top, bottom], axis=0))


def _hamiltonian_n_qubits(dim: int) -> int:
    return num_qubits_from_dim(next_power_of_two(dim))


@dataclass(frozen=True)
class FixedRyEncoder:
    """Component-wise fixed-frequency ``R_y(alpha x_j)`` encoder."""

    alpha: float = 1.0

    def n_qubits(self, input_shape: tuple[int, ...]) -> int:
        return _feature_count(input_shape)

    def init_params(
        self,
        key: jax.Array,
        input_shape: tuple[int, ...],
        reupload_depth: int,
        dtype: jnp.dtype = jnp.float64,
    ) -> dict[str, jnp.ndarray]:
        del key, input_shape, reupload_depth, dtype
        return {}

    def apply_to_state(
        self,
        params: dict[str, jnp.ndarray],
        state: jnp.ndarray,
        sample: jnp.ndarray,
        layer_index: int,
        n_qubits: int,
        reupload_depth: int | None = None,
    ) -> jnp.ndarray:
        del params, layer_index, reupload_depth
        features = jnp.ravel(sample)
        if features.shape[0] != n_qubits:
            raise ValueError(
                f"FixedRyEncoder expected {n_qubits} features, got {features.shape[0]}"
            )
        for wire, value in enumerate(features):
            state = apply_local_unitary(state, _rotation_y(self.alpha * value), (wire,), n_qubits)
        return state


@dataclass(frozen=True)
class FixedRyRzEncoder:
    """Component-wise fixed-frequency ``R_z(beta x_j) R_y(alpha x_j)`` encoder."""

    alpha: float = 1.0
    beta: float = 1.0

    def n_qubits(self, input_shape: tuple[int, ...]) -> int:
        return _feature_count(input_shape)

    def init_params(
        self,
        key: jax.Array,
        input_shape: tuple[int, ...],
        reupload_depth: int,
        dtype: jnp.dtype = jnp.float64,
    ) -> dict[str, jnp.ndarray]:
        del key, input_shape, reupload_depth, dtype
        return {}

    def apply_to_state(
        self,
        params: dict[str, jnp.ndarray],
        state: jnp.ndarray,
        sample: jnp.ndarray,
        layer_index: int,
        n_qubits: int,
        reupload_depth: int | None = None,
    ) -> jnp.ndarray:
        del params, layer_index, reupload_depth
        features = jnp.ravel(sample)
        if features.shape[0] != n_qubits:
            raise ValueError(
                f"FixedRyRzEncoder expected {n_qubits} features, got {features.shape[0]}"
            )
        for wire, value in enumerate(features):
            state = apply_local_unitary(state, _rotation_y(self.alpha * value), (wire,), n_qubits)
            state = apply_local_unitary(state, _rotation_z(self.beta * value), (wire,), n_qubits)
        return state


@dataclass(frozen=True)
class TrainableFrequencyRyEncoder:
    """Trainable-frequency ``R_y(gamma_{ell,j} x_j)`` encoder."""

    init_scale: float = 1.0
    init_noise: float = 0.01

    def n_qubits(self, input_shape: tuple[int, ...]) -> int:
        return _feature_count(input_shape)

    def init_params(
        self,
        key: jax.Array,
        input_shape: tuple[int, ...],
        reupload_depth: int,
        dtype: jnp.dtype = jnp.float64,
    ) -> dict[str, jnp.ndarray]:
        shape = (reupload_depth, _feature_count(input_shape))
        gamma = self.init_scale + self.init_noise * jax.random.normal(key, shape, dtype=dtype)
        return {"gamma": gamma}

    def apply_to_state(
        self,
        params: dict[str, jnp.ndarray],
        state: jnp.ndarray,
        sample: jnp.ndarray,
        layer_index: int,
        n_qubits: int,
        reupload_depth: int | None = None,
    ) -> jnp.ndarray:
        del reupload_depth
        features = jnp.ravel(sample)
        gamma = params["gamma"][layer_index]
        if features.shape[0] != n_qubits or gamma.shape[0] != n_qubits:
            raise ValueError("TrainableFrequencyRyEncoder feature/gamma size mismatch")
        for wire, value in enumerate(features):
            state = apply_local_unitary(state, _rotation_y(gamma[wire] * value), (wire,), n_qubits)
        return state


_PATCH_LINEAR_MAP = (
    jnp.asarray(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 0, -1],
            [0, 1, -1, 0],
            [1, -1, 0, 0],
            [0, 0, 1, -1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1],
        ],
        dtype=jnp.float64,
    )
    / 2.0
)


def extract_four_patches(sample: jnp.ndarray) -> jnp.ndarray:
    """Split STA4 ``4 x 4`` or DYN ``8 x 2`` data into four ``2 x 2`` patches."""

    sample = jnp.asarray(sample)
    if sample.shape == (4, 4):
        return jnp.stack(
            [
                sample[0:2, 0:2],
                sample[0:2, 2:4],
                sample[2:4, 0:2],
                sample[2:4, 2:4],
            ]
        )
    if sample.shape == (8, 2):
        return jnp.stack([sample[start : start + 2, :] for start in range(0, 8, 2)])
    if sample.shape == (16,):
        return extract_four_patches(jnp.reshape(sample, (4, 4)))
    raise ValueError(f"cannot split sample with shape {sample.shape} into four 2 x 2 patches")


def _validate_four_patch_input_shape(input_shape: tuple[int, ...]) -> None:
    if tuple(input_shape) not in {(4, 4), (8, 2), (16,)}:
        raise ValueError(
            "non-overlap patch encoders require STA4 (4, 4), DYN (8, 2), "
            f"or flattened STA4 (16,) input, got {input_shape}"
        )


def build_nonoverlap_patch_H_blocks(sample: jnp.ndarray) -> jnp.ndarray:
    """Build one padded block Hamiltonian for each non-overlapping ``2 x 2`` patch."""

    return jnp.stack([build_H_block(patch) for patch in extract_four_patches(sample)])


@dataclass(frozen=True)
class PatchSU4Encoder:
    """Fixed linear patch map into four local SU(4) uploads."""

    scale: float = 1.0

    def n_qubits(self, input_shape: tuple[int, ...]) -> int:
        del input_shape
        return 8

    def init_params(
        self,
        key: jax.Array,
        input_shape: tuple[int, ...],
        reupload_depth: int,
        dtype: jnp.dtype = jnp.float64,
    ) -> dict[str, jnp.ndarray]:
        del key, input_shape, reupload_depth, dtype
        return {}

    def patch_angles(self, patch: jnp.ndarray) -> jnp.ndarray:
        """Map four patch entries to the 15 SU(4) generator angles."""

        return self.scale * (_PATCH_LINEAR_MAP.astype(patch.dtype) @ jnp.ravel(patch))

    def apply_to_state(
        self,
        params: dict[str, jnp.ndarray],
        state: jnp.ndarray,
        sample: jnp.ndarray,
        layer_index: int,
        n_qubits: int,
        reupload_depth: int | None = None,
    ) -> jnp.ndarray:
        del params, layer_index, reupload_depth
        if n_qubits != 8:
            raise ValueError(f"PatchSU4Encoder expects 8 qubits, got {n_qubits}")
        for patch_index, patch in enumerate(extract_four_patches(sample)):
            U = su4_from_params(self.patch_angles(patch))
            state = apply_local_unitary(state, U, (2 * patch_index, 2 * patch_index + 1), n_qubits)
        return state


@dataclass(frozen=True)
class TrainablePatchSU4Encoder:
    """Layerwise trainable linear patch map into four local SU(4) uploads."""

    scale: float = 1.0
    init_noise: float = 0.01

    def n_qubits(self, input_shape: tuple[int, ...]) -> int:
        _validate_four_patch_input_shape(input_shape)
        return 8

    def init_params(
        self,
        key: jax.Array,
        input_shape: tuple[int, ...],
        reupload_depth: int,
        dtype: jnp.dtype = jnp.float64,
    ) -> dict[str, jnp.ndarray]:
        _validate_four_patch_input_shape(input_shape)
        shape = (reupload_depth, 4, *_PATCH_LINEAR_MAP.shape)
        base = jnp.broadcast_to(_PATCH_LINEAR_MAP.astype(dtype), shape)
        noise = self.init_noise * jax.random.normal(key, shape, dtype=dtype)
        return {"patch_map": base + noise}

    def patch_angles(
        self,
        params: dict[str, jnp.ndarray],
        patch: jnp.ndarray,
        layer_index: int,
        patch_index: int,
    ) -> jnp.ndarray:
        """Map one patch through the trainable layer/patch-specific linear map."""

        patch_map = params["patch_map"]
        if patch_map.ndim != 4 or patch_map.shape[1:] != (4, 15, 4):
            raise ValueError(f"patch_map must have shape (L, 4, 15, 4), got {patch_map.shape}")
        return self.scale * (patch_map[layer_index, patch_index] @ jnp.ravel(patch))

    def apply_to_state(
        self,
        params: dict[str, jnp.ndarray],
        state: jnp.ndarray,
        sample: jnp.ndarray,
        layer_index: int,
        n_qubits: int,
        reupload_depth: int | None = None,
    ) -> jnp.ndarray:
        del reupload_depth
        if n_qubits != 8:
            raise ValueError(f"TrainablePatchSU4Encoder expects 8 qubits, got {n_qubits}")
        for patch_index, patch in enumerate(extract_four_patches(sample)):
            U = su4_from_params(self.patch_angles(params, patch, layer_index, patch_index))
            state = apply_local_unitary(state, U, (2 * patch_index, 2 * patch_index + 1), n_qubits)
        return state


@dataclass(frozen=True)
class NonOverlapPatchBlockHamiltonianEncoder:
    """Parallel block-Hamiltonian uploads on four non-overlapping ``2 x 2`` patches."""

    trainable_times: bool = True
    fixed_time: float | None = None

    def n_qubits(self, input_shape: tuple[int, ...]) -> int:
        _validate_four_patch_input_shape(input_shape)
        return 8

    def init_params(
        self,
        key: jax.Array,
        input_shape: tuple[int, ...],
        reupload_depth: int,
        dtype: jnp.dtype = jnp.float64,
    ) -> dict[str, jnp.ndarray]:
        del key
        _validate_four_patch_input_shape(input_shape)
        if not self.trainable_times:
            return {}
        initial = 1.0 / max(reupload_depth, 1) if self.fixed_time is None else self.fixed_time
        return {"t_layers": jnp.full((reupload_depth, 4), initial, dtype=dtype)}

    def _time(
        self,
        params: dict[str, jnp.ndarray],
        layer_index: int,
        patch_index: int,
        reupload_depth: int,
    ) -> jnp.ndarray:
        if self.trainable_times:
            return params["t_layers"][layer_index, patch_index]
        fixed = 1.0 / max(reupload_depth, 1) if self.fixed_time is None else self.fixed_time
        return jnp.asarray(fixed, dtype=jnp.float64)

    def patch_unitary(
        self,
        params: dict[str, jnp.ndarray],
        patch: jnp.ndarray,
        layer_index: int,
        patch_index: int,
        reupload_depth: int,
    ) -> jnp.ndarray:
        H_patch = build_H_block(patch)
        return matrix_time_evolution(
            H_patch,
            self._time(params, layer_index, patch_index, reupload_depth),
        )

    def apply_to_state(
        self,
        params: dict[str, jnp.ndarray],
        state: jnp.ndarray,
        sample: jnp.ndarray,
        layer_index: int,
        n_qubits: int,
        reupload_depth: int | None = None,
    ) -> jnp.ndarray:
        if n_qubits != 8:
            raise ValueError(
                f"NonOverlapPatchBlockHamiltonianEncoder expects 8 qubits, got {n_qubits}"
            )
        depth = layer_index + 1 if reupload_depth is None else reupload_depth
        for patch_index, patch in enumerate(extract_four_patches(sample)):
            U = self.patch_unitary(params, patch, layer_index, patch_index, depth)
            state = apply_local_unitary(state, U, (2 * patch_index, 2 * patch_index + 1), n_qubits)
        return state


@dataclass(frozen=True)
class SymmetricHamiltonianEncoder:
    """Symmetric Hamiltonian embedding ``H_sym = (M + M.T) / 2``."""

    trainable_times: bool = True
    fixed_time: float | None = None
    rectangular_strategy: str = "pad_columns_to_square"

    def n_qubits(self, input_shape: tuple[int, ...]) -> int:
        if len(input_shape) != 2:
            raise ValueError("SymmetricHamiltonianEncoder requires a matrix input shape")
        rows, cols = input_shape
        square_dim = rows if rows == cols else rows
        if cols > rows:
            raise ValueError("pad_columns_to_square requires rows >= cols")
        return _hamiltonian_n_qubits(square_dim)

    def init_params(
        self,
        key: jax.Array,
        input_shape: tuple[int, ...],
        reupload_depth: int,
        dtype: jnp.dtype = jnp.float64,
    ) -> dict[str, jnp.ndarray]:
        del key, input_shape
        if not self.trainable_times:
            return {}
        initial = 1.0 / max(reupload_depth, 1) if self.fixed_time is None else self.fixed_time
        return {"t_layers": jnp.full((reupload_depth,), initial, dtype=dtype)}

    def _time(
        self,
        params: dict[str, jnp.ndarray],
        layer_index: int,
        reupload_depth: int,
    ) -> jnp.ndarray:
        if self.trainable_times:
            return params["t_layers"][layer_index]
        fixed = 1.0 / max(reupload_depth, 1) if self.fixed_time is None else self.fixed_time
        return jnp.asarray(fixed, dtype=jnp.float64)

    def unitary(
        self,
        params: dict[str, jnp.ndarray],
        sample: jnp.ndarray,
        layer_index: int,
        reupload_depth: int,
    ) -> jnp.ndarray:
        H_sym = build_H_sym(sample, rectangular_strategy=self.rectangular_strategy)
        return matrix_time_evolution(H_sym, self._time(params, layer_index, reupload_depth))

    def apply_to_state(
        self,
        params: dict[str, jnp.ndarray],
        state: jnp.ndarray,
        sample: jnp.ndarray,
        layer_index: int,
        n_qubits: int,
        reupload_depth: int | None = None,
    ) -> jnp.ndarray:
        depth = layer_index + 1 if reupload_depth is None else reupload_depth
        U = self.unitary(params, sample, layer_index, depth)
        # If the model state has more qubits than the natural padded H_sym space,
        # this direct-sum padding implements exp(-i (H_sym \oplus 0) t / 2).
        # It is intentionally not a Kronecker product with idle tensor factors.
        U = pad_unitary_to_dim(U, 2**n_qubits)
        return apply_unitary(state, U)


@dataclass(frozen=True)
class BlockHamiltonianEncoder:
    """Block Hamiltonian embedding ``H_block = [[0, M], [M.T, 0]]``."""

    trainable_times: bool = True
    fixed_time: float | None = None

    def n_qubits(self, input_shape: tuple[int, ...]) -> int:
        if len(input_shape) != 2:
            raise ValueError("BlockHamiltonianEncoder requires a matrix input shape")
        rows, cols = input_shape
        return _hamiltonian_n_qubits(rows + cols)

    def init_params(
        self,
        key: jax.Array,
        input_shape: tuple[int, ...],
        reupload_depth: int,
        dtype: jnp.dtype = jnp.float64,
    ) -> dict[str, jnp.ndarray]:
        del key, input_shape
        if not self.trainable_times:
            return {}
        initial = 1.0 / max(reupload_depth, 1) if self.fixed_time is None else self.fixed_time
        return {"t_layers": jnp.full((reupload_depth,), initial, dtype=dtype)}

    def _time(
        self,
        params: dict[str, jnp.ndarray],
        layer_index: int,
        reupload_depth: int,
    ) -> jnp.ndarray:
        if self.trainable_times:
            return params["t_layers"][layer_index]
        fixed = 1.0 / max(reupload_depth, 1) if self.fixed_time is None else self.fixed_time
        return jnp.asarray(fixed, dtype=jnp.float64)

    def unitary(
        self,
        params: dict[str, jnp.ndarray],
        sample: jnp.ndarray,
        layer_index: int,
        reupload_depth: int,
    ) -> jnp.ndarray:
        H_block = build_H_block(sample)
        return matrix_time_evolution(H_block, self._time(params, layer_index, reupload_depth))

    def apply_to_state(
        self,
        params: dict[str, jnp.ndarray],
        state: jnp.ndarray,
        sample: jnp.ndarray,
        layer_index: int,
        n_qubits: int,
        reupload_depth: int | None = None,
    ) -> jnp.ndarray:
        depth = layer_index + 1 if reupload_depth is None else reupload_depth
        U = self.unitary(params, sample, layer_index, depth)
        # If the model state has more qubits than the natural padded H_block
        # space, this direct-sum padding implements
        # exp(-i (H_block \oplus 0) t / 2). It is intentionally not a Kronecker
        # product with idle tensor factors.
        U = pad_unitary_to_dim(U, 2**n_qubits)
        return apply_unitary(state, U)


# Quiet import-compatible aliases for older code and saved experiment helpers.
HamiltonianEncoder = SymmetricHamiltonianEncoder
build_H_M = build_H_sym
build_H_tilde = build_H_block
