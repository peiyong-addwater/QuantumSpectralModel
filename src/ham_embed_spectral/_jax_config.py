"""JAX runtime defaults for this project."""

from __future__ import annotations

from jax import config as jax_config


def enable_x64() -> None:
    """Use 64-bit arrays by default for quantum simulation code."""

    jax_config.update("jax_enable_x64", True)
