# Quantum Spectral Model

This repository provides the training and diagnostic code for [*Quantum Spectral
Model: Data Reuploading with Input-Conditioned Frequency
Support*](https://arxiv.org/abs/2607.22516).

## Environment

We use `uv` with Python 3.12 or later. The current dependency configuration
targets Linux with an NVIDIA GPU and CUDA 12. From the repository root, configure
the environment and check that JAX detects the GPU:

```bash
uv sync
JAX_PLATFORMS=cuda uv run python -c "import jax; print(jax.devices())"
```

## Training

### Local NVIDIA GPU

The following commands run one representative configuration for each benchmark.
We use the manifest-compatible run identifiers so the diagnostic scripts can
locate the resulting checkpoints.

Pendigits STA4:

```bash
pendigits_job="pendigits__sta4__symmetric-hamiltonian__L1__lr0.01__bs32__seed0"

JAX_PLATFORMS=cuda uv run scripts/train.py \
  --dataset pendigits \
  --representation sta4 \
  --encoder symmetric-hamiltonian \
  --steps 2000 \
  --checkpoint \
  --experiment-name pendigits \
  --run-id "${pendigits_job}"
```

Synthetic eigengap and singular-value tasks:

```bash
eigengap_job="synthetic-eigengap__synthetic__symmetric-hamiltonian__L1__lr0.01__bs64__seed0"

JAX_PLATFORMS=cuda uv run scripts/train.py \
  --dataset synthetic-eigengap \
  --representation synthetic \
  --encoder symmetric-hamiltonian \
  --n-samples 4096 \
  --synthetic-dim 4 \
  --synthetic-threshold 0.75 \
  --synthetic-noise-epsilon 0.05 \
  --steps 2000 \
  --batch-size 64 \
  --eval-batch-size 256 \
  --checkpoint \
  --experiment-name synthetic \
  --run-id "${eigengap_job}"

singular_job="synthetic-singular__synthetic__block-hamiltonian__L1__lr0.01__bs64__seed0"

JAX_PLATFORMS=cuda uv run scripts/train.py \
  --dataset synthetic-singular \
  --representation synthetic \
  --encoder block-hamiltonian \
  --n-samples 4096 \
  --synthetic-rows 8 \
  --synthetic-cols 2 \
  --synthetic-threshold 2.0 \
  --synthetic-noise-epsilon 0.05 \
  --steps 2000 \
  --batch-size 64 \
  --eval-batch-size 256 \
  --checkpoint \
  --experiment-name synthetic \
  --run-id "${singular_job}"
```

### SLURM sweeps

The paper-scale experiments are defined by the Pendigits and synthetic
manifests and are submitted through SLURM. Before submission, add any required
cluster-specific values, such as `account` and `project_root`, to the `slurm`
section of each manifest.

```bash
uv run scripts/generate_train_slurm.py \
  --manifest configs/experiments/pendigits.json \
  --submit

uv run scripts/generate_train_slurm.py \
  --manifest configs/experiments/synthetic.json \
  --submit
```

## Ablation studies

Run an individual ablation on a local NVIDIA GPU:

```bash
JAX_PLATFORMS=cuda uv run scripts/train.py \
  --dataset pendigits \
  --representation sta4 \
  --encoder symmetric-hamiltonian \
  --steps 2000 \
  --ablation spectrum-only \
  --checkpoint \
  --experiment-name local_ablation \
  --run-id pendigits_sta4_spectrum_only
```

For the complete SLURM ablation sweep, first submit the timing benchmark,
passing any cluster-specific options, such as `--account`, as required:

```bash
uv run scripts/generate_ablation_timing_benchmark_slurm.py --submit
```

After the timing benchmark has finished, submit the ablation training jobs:

```bash
uv run scripts/generate_ablation_train_slurm.py --submit --skip-submitted
```

## Gradient diagnostics

The following local-GPU command analyses the final checkpoint from the
Pendigits training example above:

```bash
JAX_PLATFORMS=cuda uv run scripts/gradient_diagnostics.py \
  --manifest configs/experiments/pendigits.json \
  --mode final \
  --encoders symmetric-hamiltonian \
  --representations sta4 \
  --reupload-depths 1 \
  --seeds 0 \
  --output results/tables/gradient_diagnostics/pendigits_local_final.json
```

## Latent-state diagnostics

```bash
JAX_PLATFORMS=cuda uv run scripts/latent_state_diagnostics.py \
  --manifest configs/experiments/pendigits.json \
  --mode final \
  --encoders symmetric-hamiltonian \
  --representations sta4 \
  --reupload-depths 1 \
  --seeds 0 \
  --output results/tables/latent_state_diagnostics/pendigits_local_final.json
```
