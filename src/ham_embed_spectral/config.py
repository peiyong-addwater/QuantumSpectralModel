"""Configuration dataclasses shared across small experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReuploadingModelConfig:
    """Configuration for a dense statevector re-uploading model.

    The model applies each layer as ``V_l(theta_su) U_E(x)`` to the current
    state, matching the notation in the manuscript draft. The default initial
    state is ``|+>^{otimes n}``.
    """

    input_shape: tuple[int, ...]
    n_classes: int
    reupload_depth: int = 1
    n_qubits: int | None = None
    initial_state: str = "plus"
    mixer_scale: float = 0.01
    projector_renormalize: bool = True

    def __post_init__(self) -> None:
        if self.initial_state not in {"plus", "zero"}:
            raise ValueError("initial_state must be 'plus' or 'zero'")
