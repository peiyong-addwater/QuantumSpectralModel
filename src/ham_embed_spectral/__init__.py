"""Numerical tools for spectral Hamiltonian embedding experiments."""

from ham_embed_spectral._jax_config import enable_x64 as _enable_x64
from ham_embed_spectral.config import ReuploadingModelConfig

_enable_x64()

__all__ = ["ReuploadingModelConfig"]
