#!/usr/bin/env python3
"""Train classical classifiers on spectral or raw-value features."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, log_loss  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from ham_embed_spectral.naming import (  # noqa: E402
    CANONICAL_BLOCK_HAMILTONIAN,
    CANONICAL_DESCRIPTOR_CHOICES,
    CANONICAL_SYMMETRIC_HAMILTONIAN,
    DESCRIPTOR_CLI_CHOICES,
    canonical_descriptor_name,
)
from ham_embed_spectral.quantum.encoders import (  # noqa: E402
    BlockHamiltonianEncoder,
    SymmetricHamiltonianEncoder,
    build_H_sym,
)
from ham_embed_spectral.quantum.mixers import count_su4_mixer_params  # noqa: E402
from ham_embed_spectral.quantum.readout import ceil_log2  # noqa: E402
from ham_embed_spectral.utils.checkpointing import write_json  # noqa: E402
from scripts import train as train_script  # noqa: E402


CLASSIFIER_CHOICES = ("mlp", "linear-svc")
RAW_DATA_DESCRIPTOR = "raw-data"
BASELINE_DESCRIPTOR_CHOICES = (*DESCRIPTOR_CLI_CHOICES, RAW_DATA_DESCRIPTOR)
DESCRIPTOR_CHOICES = CANONICAL_DESCRIPTOR_CHOICES
FEATURE_SET_CHOICES = ("values", "raw")
CAPACITY_TARGET_DEPTH = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=("pendigits", "synthetic-eigengap", "synthetic-singular"),
        default="pendigits",
    )
    parser.add_argument("--data-root", default="data/raw/pendigits")
    parser.add_argument(
        "--representation",
        choices=("dyn", "sta4", "sta8", "sta16", "synthetic"),
        default="sta4",
    )
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--class-subset", default=None)
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--synthetic-dim", type=int, default=4)
    parser.add_argument("--synthetic-rows", type=int, default=4)
    parser.add_argument("--synthetic-cols", type=int, default=2)
    parser.add_argument("--synthetic-threshold", type=float, default=None)
    parser.add_argument("--synthetic-noise-epsilon", type=float, default=0.0)
    parser.add_argument(
        "--descriptor",
        choices=BASELINE_DESCRIPTOR_CHOICES,
        default=CANONICAL_SYMMETRIC_HAMILTONIAN,
    )
    parser.add_argument("--feature-set", choices=FEATURE_SET_CHOICES, default="values")
    parser.add_argument("--classifier", choices=CLASSIFIER_CHOICES, default="mlp")
    parser.add_argument(
        "--mlp-hidden-width",
        type=int,
        default=None,
        help="Override the automatically capacity-matched one-hidden-layer MLP width.",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--output", default="results/tables/classical_baseline.json")
    parser.add_argument("--manifest-id", default=None)
    parser.add_argument("--job-slug", default=None)
    parser.add_argument(
        "--bins",
        type=int,
        default=32,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.mlp_hidden_width is not None and args.mlp_hidden_width < 1:
        parser.error("--mlp-hidden-width must be positive")
    args.descriptor = canonical_descriptor_name(args.descriptor)
    if args.feature_set == "raw":
        args.descriptor = RAW_DATA_DESCRIPTOR
    elif args.descriptor == RAW_DATA_DESCRIPTOR:
        parser.error("--descriptor raw-data requires --feature-set raw")
    return args


def main() -> None:
    args = parse_args()
    started = time.time()
    data_seed = args.seed if args.data_seed is None else args.data_seed
    dataset = train_script.load_dataset(args_for_train(args), data_seed)

    train_raw = baseline_features(dataset.train.x, args)
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_raw)
    validation_features = scaler.transform(baseline_features(dataset.validation.x, args))
    test_features = scaler.transform(baseline_features(dataset.test.x, args))

    capacity = classifier_capacity_metadata(
        args,
        feature_dim=int(train_features.shape[1]),
        n_classes=int(dataset.n_classes),
        input_shape=tuple(dataset.input_shape),
    )
    classifier = build_classifier(args, capacity)
    classifier.fit(train_features, np.asarray(dataset.train.y))

    validation = evaluate_classifier(
        classifier,
        validation_features,
        dataset.validation.y,
        n_classes=dataset.n_classes,
    )
    test = evaluate_classifier(
        classifier,
        test_features,
        dataset.test.y,
        n_classes=dataset.n_classes,
    )

    write_json(
        args.output,
        {
            "status": "complete",
            "manifest_id": args.manifest_id,
            "job_slug": args.job_slug,
            "args": vars(args),
            "data_seed": data_seed,
            "descriptor": args.descriptor,
            "classifier": args.classifier,
            "classifier_params": classifier_params(args, capacity),
            "classifier_n_iter": classifier_n_iter(classifier),
            "classifier_parameter_count": capacity["classifier_parameter_count"],
            "mlp_hidden_width": capacity["mlp_hidden_width"],
            "target_parameter_count": capacity["target_parameter_count"],
            "target_parameter_source": capacity["target_parameter_source"],
            "target_parameter_relative_error": capacity["target_parameter_relative_error"],
            "feature_set": args.feature_set,
            "feature_kind": feature_kind(args.descriptor),
            "scaler": scaler_state(scaler),
            "dataset": {
                "representation": dataset.representation,
                "input_shape": dataset.input_shape,
                "class_values": dataset.class_values,
                "n_train": int(dataset.train.y.shape[0]),
                "n_validation": int(dataset.validation.y.shape[0]),
                "n_test": int(dataset.test.y.shape[0]),
            },
            "feature_dim": int(train_features.shape[1]),
            "validation": validation,
            "test": test,
            "wall_time_seconds": time.time() - started,
            "git": git_info(),
            "jax": {
                "enable_x64": jax.config.read("jax_enable_x64"),
                "devices": [str(device) for device in jax.devices()],
            },
        },
    )
    print(args.output)


def args_for_train(args: argparse.Namespace) -> SimpleNamespace:
    payload = vars(args).copy()
    payload.update(
        {
            "dry_run": False,
            "max_train_examples": None,
            "max_eval_examples": None,
            "dtype": "float64",
        }
    )
    return SimpleNamespace(**payload)


def baseline_features(samples: jnp.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.feature_set == "raw":
        return raw_value_features(samples)
    if args.feature_set == "values":
        return descriptor_value_features(samples, args.descriptor)
    expected = ", ".join(FEATURE_SET_CHOICES)
    raise ValueError(f"unknown feature_set {args.feature_set!r}; expected one of {expected}")


def raw_value_features(samples: jnp.ndarray) -> np.ndarray:
    """Return flattened raw sample values for a batch."""

    array = np.asarray(jax.device_get(samples), dtype=np.float64)
    return array.reshape((array.shape[0], -1))


def descriptor_value_features(samples: jnp.ndarray, descriptor: str) -> np.ndarray:
    """Return values-only spectral features for a batch of matrix samples."""

    values = jax.vmap(lambda sample: descriptor_values(sample, descriptor))(samples)
    return np.asarray(jax.device_get(values), dtype=np.float64)


def descriptor_values(sample: jnp.ndarray, descriptor: str) -> jnp.ndarray:
    """Return one sample's eigenvalue or singular-value descriptor vector."""

    descriptor = canonical_descriptor_name(descriptor)
    if descriptor == CANONICAL_SYMMETRIC_HAMILTONIAN:
        return jnp.linalg.eigvalsh(build_H_sym(sample))
    if descriptor == CANONICAL_BLOCK_HAMILTONIAN:
        return jnp.sort(jnp.linalg.svd(jnp.asarray(sample), compute_uv=False))[::-1]
    expected = ", ".join(DESCRIPTOR_CLI_CHOICES)
    raise ValueError(f"unknown descriptor {descriptor!r}; expected one of {expected}")


def feature_kind(descriptor: str) -> str:
    descriptor = canonical_descriptor_name(descriptor)
    if descriptor == RAW_DATA_DESCRIPTOR:
        return "raw_values"
    if descriptor == CANONICAL_SYMMETRIC_HAMILTONIAN:
        return "eigenvalues"
    if descriptor == CANONICAL_BLOCK_HAMILTONIAN:
        return "singular_values"
    raise ValueError(f"unknown descriptor {descriptor!r}")


def build_classifier(args: argparse.Namespace, capacity: dict[str, Any]):
    if args.classifier == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(int(capacity["mlp_hidden_width"]),),
            activation="relu",
            solver="adam",
            max_iter=args.steps,
            learning_rate_init=args.learning_rate,
            random_state=args.seed,
        )
    if args.classifier == "linear-svc":
        return LinearSVC(random_state=args.seed)
    expected = ", ".join(CLASSIFIER_CHOICES)
    raise ValueError(f"unknown classifier {args.classifier!r}; expected one of {expected}")


def classifier_params(args: argparse.Namespace, capacity: dict[str, Any]) -> dict[str, Any]:
    if args.classifier == "mlp":
        return {
            "hidden_layer_sizes": [int(capacity["mlp_hidden_width"])],
            "activation": "relu",
            "solver": "adam",
            "max_iter": args.steps,
            "learning_rate_init": args.learning_rate,
            "random_state": args.seed,
        }
    if args.classifier == "linear-svc":
        return {"random_state": args.seed}
    raise ValueError(f"unknown classifier {args.classifier!r}")


def classifier_capacity_metadata(
    args: argparse.Namespace,
    *,
    feature_dim: int,
    n_classes: int,
    input_shape: tuple[int, ...],
) -> dict[str, Any]:
    if args.classifier != "mlp":
        return {
            "mlp_hidden_width": None,
            "classifier_parameter_count": None,
            "target_parameter_count": None,
            "target_parameter_source": None,
            "target_parameter_relative_error": None,
        }

    target_count = target_parameter_count(args, input_shape=input_shape, n_classes=n_classes)
    hidden_width = args.mlp_hidden_width
    if hidden_width is None:
        hidden_width = automatic_mlp_hidden_width(feature_dim, n_classes, target_count)
    classifier_count = one_hidden_mlp_parameter_count(feature_dim, hidden_width, n_classes)
    return {
        "mlp_hidden_width": int(hidden_width),
        "classifier_parameter_count": int(classifier_count),
        "target_parameter_count": int(target_count),
        "target_parameter_source": target_parameter_source(args),
        "target_parameter_relative_error": float(
            abs(classifier_count - target_count) / target_count
        ),
    }


def automatic_mlp_hidden_width(feature_dim: int, n_classes: int, target_count: int) -> int:
    denominator = feature_dim + n_classes + 1
    return max(1, int(round((target_count - n_classes) / denominator)))


def one_hidden_mlp_parameter_count(feature_dim: int, hidden_width: int, n_classes: int) -> int:
    return feature_dim * hidden_width + hidden_width + hidden_width * n_classes + n_classes


def target_parameter_count(
    args: argparse.Namespace,
    *,
    input_shape: tuple[int, ...],
    n_classes: int,
) -> int:
    descriptor = capacity_target_descriptor(args)
    n_qubits = global_hamiltonian_n_qubits(input_shape, n_classes, descriptor)
    return count_su4_mixer_params(n_qubits, CAPACITY_TARGET_DEPTH) + CAPACITY_TARGET_DEPTH


def target_parameter_source(args: argparse.Namespace) -> str:
    descriptor = capacity_target_descriptor(args)
    return f"global-hamiltonian-{CAPACITY_TARGET_DEPTH}-layer:{descriptor}"


def capacity_target_descriptor(args: argparse.Namespace) -> str:
    if args.feature_set == "raw":
        if args.dataset == "synthetic-eigengap":
            return CANONICAL_SYMMETRIC_HAMILTONIAN
        if args.dataset == "synthetic-singular":
            return CANONICAL_BLOCK_HAMILTONIAN
        return CANONICAL_BLOCK_HAMILTONIAN
    return canonical_descriptor_name(args.descriptor)


def global_hamiltonian_n_qubits(
    input_shape: tuple[int, ...],
    n_classes: int,
    descriptor: str,
) -> int:
    descriptor = canonical_descriptor_name(descriptor)
    if descriptor == CANONICAL_SYMMETRIC_HAMILTONIAN:
        natural = SymmetricHamiltonianEncoder().n_qubits(input_shape)
    elif descriptor == CANONICAL_BLOCK_HAMILTONIAN:
        natural = BlockHamiltonianEncoder().n_qubits(input_shape)
    else:
        raise ValueError(f"unknown capacity descriptor {descriptor!r}")
    return max(natural, ceil_log2(n_classes))


def classifier_n_iter(classifier) -> int | list[int] | None:
    n_iter = getattr(classifier, "n_iter_", None)
    if n_iter is None:
        return None
    array = np.asarray(n_iter)
    if array.ndim == 0:
        return int(array)
    return [int(value) for value in array.ravel()]


def scaler_state(scaler: StandardScaler) -> dict[str, Any]:
    return {
        "mean": scaler.mean_,
        "scale": scaler.scale_,
        "var": scaler.var_,
        "n_features_in": int(scaler.n_features_in_),
        "fit_split": "train",
    }


def evaluate_classifier(
    classifier,
    x: np.ndarray,
    y: jnp.ndarray,
    *,
    n_classes: int,
) -> dict[str, float]:
    labels = np.asarray(y, dtype=np.int64)
    predictions = classifier.predict(x)
    probabilities = predict_probabilities(classifier, x, n_classes=n_classes)
    return {
        "loss": float(log_loss(labels, probabilities, labels=list(range(n_classes)))),
        "accuracy": float(accuracy_score(labels, predictions)),
        "n_examples": float(labels.shape[0]),
    }


def predict_probabilities(classifier, x: np.ndarray, *, n_classes: int) -> np.ndarray:
    if hasattr(classifier, "predict_proba"):
        partial = classifier.predict_proba(x)
    else:
        partial = scores_to_probabilities(classifier.decision_function(x))

    full = np.zeros((x.shape[0], n_classes), dtype=np.float64)
    for index, class_value in enumerate(classifier.classes_):
        if 0 <= int(class_value) < n_classes:
            full[:, int(class_value)] = partial[:, index]
    full = np.clip(full, 1e-12, 1.0)
    return full / np.sum(full, axis=1, keepdims=True)


def scores_to_probabilities(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def git_info() -> dict[str, Any]:
    return {
        "commit": run_git(["git", "rev-parse", "HEAD"]) or None,
        "dirty": bool(run_git(["git", "status", "--short"])),
    }


def run_git(command: list[str]) -> str:
    try:
        result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


if __name__ == "__main__":
    main()
