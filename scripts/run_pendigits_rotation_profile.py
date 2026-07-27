#!/usr/bin/env python3
"""Run one Pendigits 16-qubit rotation-encoder training job with resource profiling."""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("Pendigits data")
    data.add_argument("--data-root", default="data/raw/pendigits")
    data.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Dataset split seed forwarded to scripts/train.py. Defaults to --seed there.",
    )
    data.add_argument(
        "--representation",
        choices=("sta4", "dyn"),
        default="sta4",
        help="Both supported choices have 16 scalar features and therefore 16 rotation qubits.",
    )
    data.add_argument("--download-data", action="store_true")
    data.add_argument("--validation-fraction", type=float, default=0.1)
    data.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    data.add_argument("--max-train-examples", type=int, default=None)
    data.add_argument("--max-eval-examples", type=int, default=None)

    model = parser.add_argument_group("16-qubit rotation model")
    model.add_argument(
        "--encoder",
        choices=("trainable-frequency-ry", "fixed-ry", "fixed-ry-rz"),
        default="trainable-frequency-ry",
    )
    model.add_argument("--reupload-depth", type=int, default=16)
    model.add_argument("--initial-state", choices=("plus", "zero"), default="plus")
    model.add_argument("--mixer-scale", type=float, default=0.01)
    model.add_argument("--ry-alpha", type=float, default=1.0)
    model.add_argument("--rz-beta", type=float, default=1.0)
    model.add_argument("--tf-init-scale", type=float, default=1.0)
    model.add_argument("--tf-init-noise", type=float, default=0.01)
    model.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    model.add_argument(
        "--projector-renormalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    model.add_argument(
        "--track-readout-leakage",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    train = parser.add_argument_group("training")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--steps", type=int, default=2000)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--eval-batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=0.01)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--log-every", type=int, default=50)
    train.add_argument("--eval-every", type=int, default=100)
    train.add_argument("--dry-run", action="store_true")

    outputs = parser.add_argument_group("outputs")
    outputs.add_argument("--output-root", default="results/runs")
    outputs.add_argument("--experiment-name", default="pendigits_rotation_profile")
    outputs.add_argument("--run-id", default=None)
    outputs.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=False)
    outputs.add_argument("--checkpoint-format", choices=("hdf5", "orbax"), default="hdf5")
    outputs.add_argument("--checkpoint-every", type=int, default=0)
    outputs.add_argument(
        "--checkpoint-steps",
        default=None,
        help="Comma-separated exact training steps forwarded to scripts/train.py.",
    )

    profile = parser.add_argument_group("resource profiling")
    profile.add_argument("--sample-interval", type=float, default=10.0)
    profile.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    profile.add_argument(
        "--skip-jax-preflight",
        action="store_true",
        help="Skip the pre-training JAX device check.",
    )
    profile.add_argument(
        "--extra-train-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments appended verbatim after '--'.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reupload_depth != 16:
        raise ValueError("this dedicated profile script is intended for --reupload-depth 16")
    if args.sample_interval <= 0:
        raise ValueError("--sample-interval must be positive")

    run_id = args.run_id or default_run_id(args)
    run_dir = (
        output_root_path(args.output_root)
        / args.experiment_name
        / f"{run_id}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_jax_preflight:
        run_jax_preflight(require_gpu=args.require_gpu)

    command = build_train_command(args, run_id)
    samples_path = run_dir / "resource_samples.jsonl"
    profile_path = run_dir / "resource_profile.json"

    started_at = datetime.now(UTC)
    started = time.perf_counter()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    process = subprocess.Popen(command, cwd=ROOT)
    monitor = ResourceMonitor(
        pid=process.pid,
        interval_seconds=args.sample_interval,
        samples_path=samples_path,
    )
    monitor.start()
    returncode = 1
    try:
        returncode = process.wait()
    finally:
        monitor.stop()
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    finished_at = datetime.now(UTC)
    wall_time = time.perf_counter() - started

    profile = build_profile_summary(
        args=args,
        command=command,
        run_dir=run_dir,
        samples_path=samples_path,
        started_at=started_at,
        finished_at=finished_at,
        wall_time_seconds=wall_time,
        returncode=returncode,
        usage_before=usage_before,
        usage_after=usage_after,
        monitor_summary=monitor.summary(),
    )
    write_json(profile_path, profile)
    print(f"resource_profile={profile_path}", flush=True)
    if returncode != 0:
        raise SystemExit(returncode)


def default_run_id(args: argparse.Namespace) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"pendigits_{args.representation}_{args.encoder}"
        f"_L{args.reupload_depth}_seed{args.seed}_{timestamp}"
    )


def output_root_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_train_command(args: argparse.Namespace, run_id: str) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset",
        "pendigits",
        "--representation",
        args.representation,
        "--encoder",
        args.encoder,
        "--reupload-depth",
        str(args.reupload_depth),
        "--seed",
        str(args.seed),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--learning-rate",
        f"{args.learning_rate:g}",
        "--weight-decay",
        f"{args.weight_decay:g}",
        "--log-every",
        str(args.log_every),
        "--eval-every",
        str(args.eval_every),
        "--data-root",
        args.data_root,
        "--validation-fraction",
        f"{args.validation_fraction:g}",
        "--output-root",
        args.output_root,
        "--experiment-name",
        args.experiment_name,
        "--run-id",
        run_id,
        "--dtype",
        args.dtype,
        "--manifest-id",
        "manual_pendigits_rotation_profile",
        "--job-slug",
        run_id,
        "--initial-state",
        args.initial_state,
        "--mixer-scale",
        f"{args.mixer_scale:g}",
        "--ry-alpha",
        f"{args.ry_alpha:g}",
        "--rz-beta",
        f"{args.rz_beta:g}",
        "--tf-init-scale",
        f"{args.tf_init_scale:g}",
        "--tf-init-noise",
        f"{args.tf_init_noise:g}",
    ]
    data_seed = getattr(args, "data_seed", None)
    if data_seed is not None:
        command.extend(["--data-seed", str(data_seed)])
    command.append(
        "--projector-renormalize"
        if args.projector_renormalize
        else "--no-projector-renormalize"
    )
    command.append(
        "--track-readout-leakage"
        if args.track_readout_leakage
        else "--no-track-readout-leakage"
    )
    command.append("--standardize" if args.standardize else "--no-standardize")
    command.append("--checkpoint" if args.checkpoint else "--no-checkpoint")
    if args.checkpoint:
        command.extend(["--checkpoint-format", args.checkpoint_format])
    if args.checkpoint_every:
        command.extend(["--checkpoint-every", str(args.checkpoint_every)])
    checkpoint_steps = getattr(args, "checkpoint_steps", "")
    if checkpoint_steps:
        command.extend(["--checkpoint-steps", checkpoint_steps])
    if args.download_data:
        command.append("--download-data")
    if args.max_train_examples is not None:
        command.extend(["--max-train-examples", str(args.max_train_examples)])
    if args.max_eval_examples is not None:
        command.extend(["--max-eval-examples", str(args.max_eval_examples)])
    if args.dry_run:
        command.append("--dry-run")
    command.extend(strip_remainder_separator(args.extra_train_args))
    return command


def strip_remainder_separator(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def run_jax_preflight(*, require_gpu: bool) -> None:
    code = f"""
import sys
sys.path.insert(0, {str(ROOT / "src")!r})
from ham_embed_spectral._jax_config import enable_x64
enable_x64()
import jax
import jax.numpy as jnp
print("jax_version", jax.__version__)
print("jax_backend", jax.default_backend())
devices = jax.devices()
print("jax_devices", [str(device) for device in devices])
gpu_devices = [device for device in devices if device.platform == "gpu"]
if {require_gpu!r} and not gpu_devices:
    raise SystemExit("ERROR: --require-gpu is set, but JAX reports no GPU devices")
target = gpu_devices[0] if gpu_devices else devices[0]
x = jax.device_put(jnp.ones((256, 256), dtype=jnp.float32), target)
y = (x @ x).block_until_ready()
print("jax_preflight_device", target)
print("jax_preflight_sample", float(y[0, 0]))
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


@dataclass
class MonitorStats:
    sample_count: int = 0
    cpu_percent_sum: float = 0.0
    cpu_percent_count: int = 0
    peak_cpu_percent: float | None = None
    peak_rss_bytes: int | None = None
    nvidia_smi_available: bool = field(
        default_factory=lambda: shutil.which("nvidia-smi") is not None
    )
    gpu_sample_count: int = 0
    gpu_utilization_sum: float = 0.0
    gpu_utilization_count: int = 0
    peak_gpu_utilization_percent: float | None = None
    peak_gpu_memory_used_mb: float | None = None
    peak_gpu_process_memory_used_mb: float | None = None


class ResourceMonitor:
    """Background sampler for process RSS/CPU and nvidia-smi GPU metrics."""

    def __init__(self, pid: int, interval_seconds: float, samples_path: Path) -> None:
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.samples_path = samples_path
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._stats = MonitorStats()
        self._started = time.perf_counter()
        self._last_process_ticks: int | None = None
        self._last_sample_time: float | None = None
        self._clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])

    def start(self) -> None:
        self.samples_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_seconds + 1.0))

    def summary(self) -> dict[str, Any]:
        stats = self._stats
        mean_cpu = (
            stats.cpu_percent_sum / stats.cpu_percent_count
            if stats.cpu_percent_count
            else None
        )
        mean_gpu_utilization = (
            stats.gpu_utilization_sum / stats.gpu_utilization_count
            if stats.gpu_utilization_count
            else None
        )
        return {
            "sample_count": stats.sample_count,
            "peak_process_cpu_percent": stats.peak_cpu_percent,
            "mean_process_cpu_percent": mean_cpu,
            "peak_process_rss_bytes": stats.peak_rss_bytes,
            "nvidia_smi_available": stats.nvidia_smi_available,
            "gpu_sample_count": stats.gpu_sample_count,
            "peak_gpu_utilization_percent": stats.peak_gpu_utilization_percent,
            "mean_gpu_utilization_percent": mean_gpu_utilization,
            "peak_gpu_memory_used_mb": stats.peak_gpu_memory_used_mb,
            "peak_gpu_process_memory_used_mb": stats.peak_gpu_process_memory_used_mb,
        }

    def _run(self) -> None:
        with self.samples_path.open("a", encoding="utf-8") as handle:
            while not self._stop.is_set():
                sample = self._sample()
                handle.write(json.dumps(sample, sort_keys=True) + "\n")
                handle.flush()
                self._update_stats(sample)
                self._stop.wait(self.interval_seconds)

    def _sample(self) -> dict[str, Any]:
        now = time.perf_counter()
        process = read_process_sample(self.pid)
        if process is not None:
            process_ticks = int(process["cpu_ticks"])
            if self._last_process_ticks is not None and self._last_sample_time is not None:
                delta_ticks = process_ticks - self._last_process_ticks
                delta_time = max(now - self._last_sample_time, 1e-12)
                process["cpu_percent"] = 100.0 * delta_ticks / self._clock_ticks / delta_time
            self._last_process_ticks = process_ticks
            self._last_sample_time = now
        return {
            "elapsed_seconds": now - self._started,
            "process": process,
            "gpus": query_gpu_samples(),
            "gpu_processes": query_gpu_process_samples(self.pid),
        }

    def _update_stats(self, sample: dict[str, Any]) -> None:
        stats = self._stats
        stats.sample_count += 1
        process = sample.get("process")
        if process:
            rss = process.get("rss_bytes")
            if rss is not None:
                stats.peak_rss_bytes = max_none(stats.peak_rss_bytes, int(rss))
            cpu_percent = process.get("cpu_percent")
            if cpu_percent is not None:
                cpu_percent = float(cpu_percent)
                stats.cpu_percent_sum += cpu_percent
                stats.cpu_percent_count += 1
                stats.peak_cpu_percent = max_none(stats.peak_cpu_percent, cpu_percent)

        gpus = sample.get("gpus") or []
        if gpus:
            stats.gpu_sample_count += 1
        for gpu in gpus:
            utilization = gpu.get("utilization_gpu_percent")
            if utilization is not None:
                utilization = float(utilization)
                stats.gpu_utilization_sum += utilization
                stats.gpu_utilization_count += 1
                stats.peak_gpu_utilization_percent = max_none(
                    stats.peak_gpu_utilization_percent,
                    utilization,
                )
            memory_used = gpu.get("memory_used_mb")
            if memory_used is not None:
                stats.peak_gpu_memory_used_mb = max_none(
                    stats.peak_gpu_memory_used_mb,
                    float(memory_used),
                )

        for gpu_process in sample.get("gpu_processes") or []:
            used_memory = gpu_process.get("used_memory_mb")
            if used_memory is not None:
                stats.peak_gpu_process_memory_used_mb = max_none(
                    stats.peak_gpu_process_memory_used_mb,
                    float(used_memory),
                )


def read_process_sample(pid: int) -> dict[str, Any] | None:
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        stat = stat_path.read_text()
    except FileNotFoundError:
        return None
    close_paren = stat.rfind(")")
    fields = stat[close_paren + 2 :].split()
    utime = int(fields[11])
    stime = int(fields[12])
    rss_pages = int(fields[21])
    return {
        "pid": pid,
        "cpu_ticks": utime + stime,
        "user_cpu_ticks": utime,
        "system_cpu_ticks": stime,
        "rss_bytes": rss_pages * os.sysconf("SC_PAGE_SIZE"),
        "cpu_percent": None,
    }


def query_gpu_samples() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    rows = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) < 6:
            continue
        rows.append(
            {
                "index": parse_int(row[0]),
                "uuid": row[1].strip(),
                "name": row[2].strip(),
                "utilization_gpu_percent": parse_float(row[3]),
                "memory_used_mb": parse_float(row[4]),
                "memory_total_mb": parse_float(row[5]),
            }
        )
    return rows


def query_gpu_process_samples(pid: int) -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    rows = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) < 3:
            continue
        row_pid = parse_int(row[0])
        if row_pid != pid:
            continue
        rows.append(
            {
                "pid": row_pid,
                "gpu_uuid": row[1].strip(),
                "used_memory_mb": parse_float(row[2]),
            }
        )
    return rows


def build_profile_summary(
    *,
    args: argparse.Namespace,
    command: Sequence[str],
    run_dir: Path,
    samples_path: Path,
    started_at: datetime,
    finished_at: datetime,
    wall_time_seconds: float,
    returncode: int,
    usage_before: resource.struct_rusage,
    usage_after: resource.struct_rusage,
    monitor_summary: dict[str, Any],
) -> dict[str, Any]:
    user_seconds = usage_after.ru_utime - usage_before.ru_utime
    system_seconds = usage_after.ru_stime - usage_before.ru_stime
    return {
        "schema_version": 1,
        "script": str(Path(__file__).resolve()),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "returncode": returncode,
        "command": list(command),
        "run_dir": str(run_dir),
        "samples_path": str(samples_path),
        "job": {
            "dataset": "pendigits",
            "representation": args.representation,
            "encoder": args.encoder,
            "reupload_depth": args.reupload_depth,
            "n_qubits": 16,
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "dtype": args.dtype,
            "initial_state": args.initial_state,
        },
        "timing": {
            "wall_time_seconds": wall_time_seconds,
        },
        "cpu": {
            "child_user_time_seconds": user_seconds,
            "child_system_time_seconds": system_seconds,
            "child_total_cpu_time_seconds": user_seconds + system_seconds,
            "child_max_rss_bytes": maxrss_to_bytes(usage_after.ru_maxrss),
            "sample_peak_rss_bytes": monitor_summary["peak_process_rss_bytes"],
            "sample_peak_cpu_percent": monitor_summary["peak_process_cpu_percent"],
            "sample_mean_cpu_percent": monitor_summary["mean_process_cpu_percent"],
        },
        "gpu": {
            "nvidia_smi_available": monitor_summary["nvidia_smi_available"],
            "sample_count": monitor_summary["gpu_sample_count"],
            "peak_utilization_percent": monitor_summary["peak_gpu_utilization_percent"],
            "mean_utilization_percent": monitor_summary["mean_gpu_utilization_percent"],
            "peak_memory_used_mb": monitor_summary["peak_gpu_memory_used_mb"],
            "process_peak_memory_used_mb": monitor_summary[
                "peak_gpu_process_memory_used_mb"
            ],
        },
        "monitor": {
            "sample_interval_seconds": args.sample_interval,
            "sample_count": monitor_summary["sample_count"],
        },
    }


def maxrss_to_bytes(value: int) -> int:
    if sys.platform == "darwin":
        return int(value)
    return int(value) * 1024


def parse_int(value: str) -> int | None:
    value = value.strip()
    if not value or value.lower() in {"n/a", "not supported"}:
        return None
    return int(value)


def parse_float(value: str) -> float | None:
    value = value.strip()
    if not value or value.lower() in {"n/a", "not supported"}:
        return None
    return float(value)


def max_none(current: Any, candidate: Any) -> Any:
    return candidate if current is None else max(current, candidate)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
