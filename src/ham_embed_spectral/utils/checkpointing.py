"""Checkpoint and JSON-output helpers."""

from __future__ import annotations

import json
import pickle
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

HDF5_CHECKPOINT_FORMAT_VERSION = "qfm-hdf5-pytree-v1"


def to_jsonable(value: Any) -> Any:
    """Convert common scientific Python objects into JSON-compatible values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, jnp.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON file with stable formatting."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def reset_jsonl(path: str | Path) -> None:
    """Create or truncate a JSONL file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("")


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Append one JSON-compatible record to a JSONL file and flush it."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")
        handle.flush()


def save_orbax_pytree(path: str | Path, item: Any, *, force: bool = True) -> Path:
    """Save a pytree checkpoint using Orbax."""

    target = _absolute_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sanitized_item, zero_size_metadata = sanitize_zero_size_arrays(item)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(target, jax.device_get(sanitized_item), force=force)
    if zero_size_metadata:
        write_json(target / "_zero_size_arrays.json", {"zero_size_arrays": zero_size_metadata})
    return target


def save_hdf5_pytree(path: str | Path, label: str, item: Any, *, force: bool = True) -> Path:
    """Save a pytree checkpoint label into a run-level HDF5 file."""

    _validate_hdf5_label(label)
    target = _absolute_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    host_item = jax.device_get(item)
    paths_and_leaves, treedef = jax.tree_util.tree_flatten_with_path(host_item)
    tmp_label = f"__tmp__{label}_{uuid.uuid4().hex}"
    with h5py.File(target, "a") as handle:
        checkpoints = handle.require_group("checkpoints")
        if label in checkpoints and not force:
            raise FileExistsError(f"HDF5 checkpoint label already exists: {target}::{label}")
        group = checkpoints.create_group(tmp_label)
        try:
            leaves_group = group.create_group("leaves")
            leaf_metadata = []
            zero_size_arrays = []
            for index, (tree_path, leaf) in enumerate(paths_and_leaves):
                dataset_name = f"{index:06d}"
                array = np.asarray(leaf)
                leaves_group.create_dataset(dataset_name, data=array)
                path_string = _tree_path_to_string(tree_path)
                leaf_record = {
                    "dataset": dataset_name,
                    "path": path_string,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                }
                leaf_metadata.append(leaf_record)
                if array.size == 0:
                    zero_size_arrays.append(leaf_record)

            treedef_bytes = pickle.dumps(treedef)
            group.create_dataset(
                "treedef_pickle",
                data=np.frombuffer(treedef_bytes, dtype=np.uint8),
            )
            metadata = {
                "format": HDF5_CHECKPOINT_FORMAT_VERSION,
                "label": label,
                "saved_at_utc": datetime.now(UTC).isoformat(),
                "leaf_count": len(leaf_metadata),
                "leaf_metadata": leaf_metadata,
                "zero_size_arrays": zero_size_arrays,
            }
            group.create_dataset(
                "metadata_json",
                data=json.dumps(metadata, sort_keys=True),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            if label in checkpoints:
                del checkpoints[label]
            checkpoints.move(tmp_label, label)
        except Exception:
            if tmp_label in checkpoints:
                del checkpoints[tmp_label]
            raise
    return target


def restore_hdf5_pytree(path: str | Path, label: str = "final") -> Any:
    """Restore a pytree checkpoint label from a run-level HDF5 file."""

    _validate_hdf5_label(label)
    target = _absolute_path(path)
    with h5py.File(target, "r") as handle:
        checkpoint_path = f"checkpoints/{label}"
        if checkpoint_path not in handle:
            raise KeyError(f"HDF5 checkpoint label not found: {target}::{label}")
        group = handle[checkpoint_path]
        metadata_raw = group["metadata_json"][()]
        if isinstance(metadata_raw, bytes):
            metadata_text = metadata_raw.decode("utf-8")
        else:
            metadata_text = str(metadata_raw)
        metadata = json.loads(metadata_text)
        if metadata.get("format") != HDF5_CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"unsupported HDF5 checkpoint format {metadata.get('format')!r} "
                f"in {target}::{label}"
            )
        treedef = pickle.loads(np.asarray(group["treedef_pickle"]).tobytes())
        leaves_group = group["leaves"]
        leaves = []
        for leaf_record in metadata["leaf_metadata"]:
            dataset = leaves_group[leaf_record["dataset"]]
            value = np.asarray(dataset[()])
            dtype = np.dtype(leaf_record["dtype"])
            shape = tuple(leaf_record["shape"])
            if value.shape != shape:
                value = np.reshape(value, shape)
            if value.dtype != dtype:
                value = value.astype(dtype, copy=False)
            leaves.append(value)
    return jax.tree_util.tree_unflatten(treedef, leaves)


def list_hdf5_checkpoint_labels(path: str | Path, prefix: str | None = None) -> list[str]:
    """List checkpoint labels stored in a run-level HDF5 checkpoint file."""

    target = _absolute_path(path)
    if not target.exists():
        return []
    with h5py.File(target, "r") as handle:
        if "checkpoints" not in handle:
            return []
        labels = [
            str(label)
            for label in handle["checkpoints"].keys()
            if not str(label).startswith("__tmp__")
        ]
    if prefix is not None:
        labels = [label for label in labels if label.startswith(prefix)]
    return sorted(labels)


def restore_checkpoint_pytree(ref: str | Path) -> Any:
    """Restore a checkpoint from an HDF5 reference or legacy Orbax directory."""

    hdf5_ref = _split_hdf5_checkpoint_ref(ref)
    if hdf5_ref is not None:
        path, label = hdf5_ref
        return restore_hdf5_pytree(path, label)
    return restore_orbax_pytree(ref)


def restore_orbax_pytree(path: str | Path) -> Any:
    """Restore a pytree checkpoint saved with ``save_orbax_pytree``."""

    target = _absolute_path(path)
    restored = ocp.PyTreeCheckpointer().restore(target)
    metadata_path = target / "_zero_size_arrays.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())["zero_size_arrays"]
        restored = restore_zero_size_arrays(restored, metadata)
    return restored


def sanitize_zero_size_arrays(item: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Replace zero-size array leaves before Orbax serialization.

    Orbax refuses to serialize zero-size arrays. Some valid model configurations
    have no SU(4) mixer blocks, producing arrays such as ``theta_su.shape ==
    (depth, 0, 15)``. We save a same-dtype placeholder and record the original
    leaf path/shape so a restore helper can reconstruct it later.
    """

    paths_and_leaves, treedef = jax.tree_util.tree_flatten_with_path(item)
    metadata: list[dict[str, Any]] = []
    sanitized_leaves = []
    for path, leaf in paths_and_leaves:
        if _is_zero_size_array(leaf):
            metadata.append(
                {
                    "path": _tree_path_to_string(path),
                    "shape": list(leaf.shape),
                    "dtype": str(leaf.dtype),
                }
            )
            sanitized_leaves.append(np.zeros((1,), dtype=leaf.dtype))
        else:
            sanitized_leaves.append(leaf)
    return jax.tree_util.tree_unflatten(treedef, sanitized_leaves), metadata


def restore_zero_size_arrays(item: Any, metadata: list[dict[str, Any]]) -> Any:
    """Replace serialized placeholders with zero-size arrays from metadata."""

    if not metadata:
        return item
    metadata_by_path = {entry["path"]: entry for entry in metadata}
    paths_and_leaves, treedef = jax.tree_util.tree_flatten_with_path(item)
    restored_leaves = []
    for path, leaf in paths_and_leaves:
        path_string = _tree_path_to_string(path)
        if path_string in metadata_by_path:
            entry = metadata_by_path[path_string]
            restored_leaves.append(np.zeros(tuple(entry["shape"]), dtype=np.dtype(entry["dtype"])))
        else:
            restored_leaves.append(leaf)
    return jax.tree_util.tree_unflatten(treedef, restored_leaves)


def _is_zero_size_array(value: Any) -> bool:
    return isinstance(value, (np.ndarray, jnp.ndarray)) and value.size == 0


def _absolute_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _split_hdf5_checkpoint_ref(ref: str | Path) -> tuple[Path, str] | None:
    raw = str(ref)
    if "::" in raw:
        path_text, label = raw.split("::", 1)
        path = Path(path_text)
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            return None
        if not label:
            raise ValueError(f"HDF5 checkpoint reference is missing a label: {raw}")
        return path, label
    path = Path(raw)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return path, "final"
    return None


def _validate_hdf5_label(label: str) -> None:
    if not label or "/" in label:
        raise ValueError(
            f"HDF5 checkpoint label must be non-empty and cannot contain '/': {label!r}"
        )


def _tree_path_to_string(path: tuple[Any, ...]) -> str:
    parts = []
    for key in path:
        if hasattr(key, "key"):
            parts.append(str(key.key))
        elif hasattr(key, "idx"):
            parts.append(str(key.idx))
        elif hasattr(key, "name"):
            parts.append(str(key.name))
        else:
            parts.append(str(key))
    return ".".join(parts)
