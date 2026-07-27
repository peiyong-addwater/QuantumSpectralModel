"""Pendigits DYN/STA4 download and preparation utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import jax.numpy as jnp
import numpy as np

PENDIGITS_MIRROR_BASE_URL = "https://raw.githubusercontent.com/bschlief/pendigits/master"
PENDIGITS_REPRESENTATIONS = ("dyn", "sta4", "sta8", "sta16")

_FEATURE_FILES = {
    "dyn": {
        "train": "pendigits_dyn_train.csv",
        "test": "pendigits_dyn_test.csv",
    },
    "sta4": {
        "train": "pendigits_sta4_train.csv",
        "test": "pendigits_sta4_test.csv",
    },
    "sta8": {
        "train": "pendigits_sta8_train.csv",
        "test": "pendigits_sta8_test.csv",
    },
    "sta16": {
        "train": "pendigits_sta16_train.csv",
        "test": "pendigits_sta16_test.csv",
    },
}
_LABEL_FILES = {
    "train": "pendigits_label_train.csv",
    "test": "pendigits_label_test.csv",
}
_REPRESENTATION_SHAPES = {
    "dyn": (8, 2),
    "sta4": (4, 4),
    "sta8": (8, 8),
    "sta16": (16, 16),
}


@dataclass(frozen=True)
class PendigitsSplit:
    """One prepared Pendigits split."""

    x: jnp.ndarray
    y: jnp.ndarray


@dataclass(frozen=True)
class PendigitsDataset:
    """Prepared Pendigits train/validation/test arrays and preprocessing state."""

    train: PendigitsSplit
    validation: PendigitsSplit
    test: PendigitsSplit
    representation: str
    input_shape: tuple[int, ...]
    class_values: tuple[int, ...]
    feature_mean: jnp.ndarray
    feature_std: jnp.ndarray

    @property
    def n_classes(self) -> int:
        """Number of active classes after optional filtering."""

        return len(self.class_values)


def pendigits_required_files(representations: Iterable[str] = ("dyn", "sta4")) -> tuple[str, ...]:
    """Return raw CSV filenames required for the requested representations."""

    checked = tuple(_normalize_representation(representation) for representation in representations)
    files: set[str] = set(_LABEL_FILES.values())
    for representation in checked:
        files.update(_FEATURE_FILES[representation].values())
    return tuple(sorted(files))


def download_pendigits(
    data_root: str | Path = "data/raw/pendigits",
    representations: Iterable[str] = ("dyn", "sta4"),
    *,
    base_url: str = PENDIGITS_MIRROR_BASE_URL,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Download Pendigits CSV files into ``data_root``.

    The default source is the lightweight mirror that contains paired DYN and
    STA4 representations. The official UCI source should still be cited for the
    dataset; UCI only ships the standard DYN representation.
    """

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    for filename in pendigits_required_files(representations):
        target = root / filename
        if target.exists() and not overwrite:
            downloaded.append(target)
            continue
        source = f"{base_url.rstrip('/')}/{filename}"
        try:
            urlretrieve(source, target)
        except URLError as exc:
            msg = f"failed to download {source} to {target}"
            raise RuntimeError(msg) from exc
        downloaded.append(target)
    return tuple(downloaded)


def prepare_pendigits(
    data_root: str | Path = "data/raw/pendigits",
    *,
    representation: str,
    validation_fraction: float = 0.1,
    seed: int = 0,
    class_subset: Sequence[int] | None = None,
    standardize: bool = True,
    download: bool = False,
    dtype: jnp.dtype = jnp.float64,
) -> PendigitsDataset:
    """Load, filter, split, reshape, and standardize Pendigits data.

    ``download=False`` is the default so experiments can run in offline mode
    when files are already present. Standardization statistics are fitted on the
    post-validation training split only, then applied to validation and test.
    """

    representation = _normalize_representation(representation)
    root = Path(data_root)
    if download:
        download_pendigits(root, representations=(representation,))

    train_x, train_y = _load_raw_split(root, representation, split="train")
    test_x, test_y = _load_raw_split(root, representation, split="test")

    train_x, train_y, class_values = _filter_and_remap_classes(train_x, train_y, class_subset)
    test_x, test_y, _ = _filter_and_remap_classes(test_x, test_y, class_values)

    train_indices, validation_indices = _train_validation_indices(
        train_y,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    fit_x = train_x[train_indices]
    validation_x = train_x[validation_indices]
    fit_y = train_y[train_indices]
    validation_y = train_y[validation_indices]

    mean, std = _fit_standardizer(fit_x, standardize=standardize)
    fit_x = _reshape_representation(_apply_standardizer(fit_x, mean, std), representation)
    validation_x = _reshape_representation(
        _apply_standardizer(validation_x, mean, std),
        representation,
    )
    test_x = _reshape_representation(_apply_standardizer(test_x, mean, std), representation)

    input_shape = _REPRESENTATION_SHAPES[representation]
    mean = np.reshape(mean, input_shape)
    std = np.reshape(std, input_shape)
    return PendigitsDataset(
        train=PendigitsSplit(
            x=jnp.asarray(fit_x, dtype=dtype),
            y=jnp.asarray(fit_y, dtype=jnp.int32),
        ),
        validation=PendigitsSplit(
            x=jnp.asarray(validation_x, dtype=dtype),
            y=jnp.asarray(validation_y, dtype=jnp.int32),
        ),
        test=PendigitsSplit(
            x=jnp.asarray(test_x, dtype=dtype),
            y=jnp.asarray(test_y, dtype=jnp.int32),
        ),
        representation=representation,
        input_shape=input_shape,
        class_values=tuple(int(value) for value in class_values),
        feature_mean=jnp.asarray(mean, dtype=dtype),
        feature_std=jnp.asarray(std, dtype=dtype),
    )


def load_pendigits_split(
    data_root: str | Path,
    *,
    representation: str,
    split: str,
    reshape: bool = True,
    dtype: jnp.dtype = jnp.float64,
) -> PendigitsSplit:
    """Load one raw train/test split without filtering or standardization."""

    representation = _normalize_representation(representation)
    x, y = _load_raw_split(Path(data_root), representation, split=split)
    if reshape:
        x = _reshape_representation(x, representation)
    return PendigitsSplit(x=jnp.asarray(x, dtype=dtype), y=jnp.asarray(y, dtype=jnp.int32))


def _normalize_representation(representation: str) -> str:
    normalized = representation.lower()
    if normalized not in _FEATURE_FILES:
        expected = ", ".join(PENDIGITS_REPRESENTATIONS)
        raise ValueError(
            f"unknown Pendigits representation {representation!r}; expected {expected}"
        )
    return normalized


def _load_raw_split(root: Path, representation: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    feature_path = root / _FEATURE_FILES[representation][split]
    label_path = root / _LABEL_FILES[split]
    missing = [path for path in (feature_path, label_path) if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"missing Pendigits file(s): {names}. "
            "Run download_pendigits(...) or prepare_pendigits(..., download=True)."
        )

    x = np.loadtxt(feature_path, delimiter=",", dtype=np.float64)
    y = np.loadtxt(label_path, delimiter=",", dtype=np.int64)
    x = np.atleast_2d(x)
    y = np.ravel(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"feature/label row mismatch for {split}: {x.shape[0]} features vs {y.shape[0]} labels"
        )

    expected_features = int(np.prod(_REPRESENTATION_SHAPES[representation]))
    if x.shape[1] != expected_features:
        raise ValueError(
            f"{representation} expects {expected_features} features, "
            f"got {x.shape[1]} in {feature_path}"
        )
    return x, y


def _filter_and_remap_classes(
    x: np.ndarray,
    y: np.ndarray,
    class_subset: Sequence[int] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    if class_subset is None:
        class_values = tuple(int(value) for value in np.unique(y))
    else:
        class_values = tuple(int(value) for value in class_subset)
        if len(set(class_values)) != len(class_values):
            raise ValueError(f"class_subset contains duplicates: {class_subset}")
        available = {int(value) for value in np.unique(y)}
        missing = tuple(value for value in class_values if value not in available)
        if missing:
            raise ValueError(f"class_subset contains absent class labels: {missing}")
        mask = np.isin(y, class_values)
        x = x[mask]
        y = y[mask]

    label_map = {original: mapped for mapped, original in enumerate(class_values)}
    try:
        remapped = np.asarray([label_map[int(label)] for label in y], dtype=np.int64)
    except KeyError as exc:
        msg = "labels contain classes outside the requested class subset"
        raise ValueError(msg) from exc
    return x, remapped, class_values


def _train_validation_indices(
    labels: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must satisfy 0 <= fraction < 1")
    all_indices = np.arange(labels.shape[0])
    if validation_fraction == 0:
        return all_indices, np.asarray([], dtype=np.int64)

    rng = np.random.default_rng(seed)
    validation_chunks: list[np.ndarray] = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        shuffled = rng.permutation(class_indices)
        n_validation = int(round(validation_fraction * class_indices.shape[0]))
        if class_indices.shape[0] > 1:
            n_validation = max(1, min(n_validation, class_indices.shape[0] - 1))
        else:
            n_validation = 0
        validation_chunks.append(shuffled[:n_validation])

    validation_indices = np.sort(np.concatenate(validation_chunks))
    train_mask = np.ones(labels.shape[0], dtype=bool)
    train_mask[validation_indices] = False
    return all_indices[train_mask], validation_indices


def _fit_standardizer(x: np.ndarray, *, standardize: bool) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        raise ValueError("cannot fit standardizer on an empty training split")
    if not standardize:
        return np.zeros(x.shape[1], dtype=np.float64), np.ones(x.shape[1], dtype=np.float64)

    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def _apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _reshape_representation(x: np.ndarray, representation: str) -> np.ndarray:
    return np.reshape(x, (x.shape[0], *_REPRESENTATION_SHAPES[representation]))
