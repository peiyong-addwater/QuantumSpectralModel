#!/usr/bin/env python3
"""Summarize empirical Fisher diagnostics from a per-sample gradient matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ham_embed_spectral.experiments.fisher import empirical_fisher_summary  # noqa: E402
from ham_embed_spectral.utils.checkpointing import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gradients",
        required=True,
        help="Path to .npy array with shape (batch, n_params).",
    )
    parser.add_argument("--output", default="results/tables/fisher_summary.json")
    parser.add_argument("--damping", type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gradients = jnp.asarray(np.load(args.gradients))
    summary = empirical_fisher_summary(gradients, damping=args.damping)
    write_json(args.output, {"gradient_shape": gradients.shape, "summary": summary})
    print(args.output)


if __name__ == "__main__":
    main()
