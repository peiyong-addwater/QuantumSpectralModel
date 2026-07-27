#!/usr/bin/env python3
"""Apply matrix-structure ablations to prepared Pendigits data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from ham_embed_spectral.data.pendigits import prepare_pendigits  # noqa: E402
from ham_embed_spectral.experiments.ablations import (  # noqa: E402
    fixed_entry_permutation,
    permute_entries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/pendigits")
    parser.add_argument("--representation", choices=("dyn", "sta4"), default="sta4")
    parser.add_argument("--ablation", choices=("entry-permutation",), default="entry-permutation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/tables/ablation_entry_permutation.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = prepare_pendigits(
        args.data_root,
        representation=args.representation,
        validation_fraction=0.1,
        seed=args.seed,
        standardize=True,
    )
    permutation = fixed_entry_permutation(jax.random.PRNGKey(args.seed), dataset.input_shape)
    transform = jax.vmap(lambda sample: permute_entries(sample, permutation))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        train_x=np.asarray(transform(dataset.train.x)),
        train_y=np.asarray(dataset.train.y),
        validation_x=np.asarray(transform(dataset.validation.x)),
        validation_y=np.asarray(dataset.validation.y),
        test_x=np.asarray(transform(dataset.test.x)),
        test_y=np.asarray(dataset.test.y),
        permutation=np.asarray(permutation),
        class_values=np.asarray(dataset.class_values),
    )
    print(output)


if __name__ == "__main__":
    main()
