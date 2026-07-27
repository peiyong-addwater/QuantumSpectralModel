#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-pendigits__sta4__trainable-frequency-ry__L16__lr0.01__bs32__seed0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-profiling_results/runs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pendigits_rotation_profile}"
PROFILE_ROOT="${PROFILE_ROOT:-profiling_results}"
MANIFEST_PATH="${MANIFEST_PATH:-${PROFILE_ROOT}/${RUN_ID}_manifest.json}"

SEED="${SEED:-0}"
DATA_SEED="${DATA_SEED:-0}"
REPRESENTATION="${REPRESENTATION:-sta4}"
ENCODER="${ENCODER:-trainable-frequency-ry}"
DEPTH="${DEPTH:-16}"
STEPS="${STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-0,50,100,200,500,1000,1500,2000}"
CHECKPOINT_FORMAT="${CHECKPOINT_FORMAT:-hdf5}"

DIAGNOSTIC_BATCH_SIZE="${DIAGNOSTIC_BATCH_SIZE:-32}"
FISHER_BATCH_SIZE="${FISHER_BATCH_SIZE:-32}"
RUN_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}/${RUN_ID}_seed${SEED}"
mkdir -p "${PROFILE_ROOT}"

uv run scripts/run_pendigits_rotation_profile.py \
  --output-root "${OUTPUT_ROOT}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --run-id "${RUN_ID}" \
  --representation "${REPRESENTATION}" \
  --encoder "${ENCODER}" \
  --reupload-depth "${DEPTH}" \
  --seed "${SEED}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --checkpoint \
  --checkpoint-format "${CHECKPOINT_FORMAT}" \
  --data-seed "${DATA_SEED}" \
  --checkpoint-steps "${CHECKPOINT_STEPS}"

cat > "${MANIFEST_PATH}" <<JSON
{
  "manifest_id": "pendigits_rotation_profile_single",
  "grid": {
    "datasets": ["pendigits"],
    "representations": ["${REPRESENTATION}"],
    "encoders": ["${ENCODER}"],
    "reupload_depths": [${DEPTH}],
    "seeds": [${SEED}],
    "learning_rates": [${LEARNING_RATE}],
    "batch_sizes": [${BATCH_SIZE}],
    "class_subsets": [null]
  },
  "data": {
    "data_root": "data/raw/pendigits",
    "data_seed": ${DATA_SEED},
    "download_data": false,
    "validation_fraction": 0.1,
    "standardize": true
  },
  "training": {
    "steps": ${STEPS},
    "eval_batch_size": ${EVAL_BATCH_SIZE}
  },
  "model": {
    "mixer_scale": 0.01,
    "projector_renormalize": true,
    "tf_init_scale": 1.0,
    "tf_init_noise": 0.01,
    "initial_state": "plus",
    "dtype": "float64",
    "track_readout_leakage": false
  },
  "outputs": {
    "output_root": "${OUTPUT_ROOT}",
    "experiment_name": "${EXPERIMENT_NAME}"
  }
}
JSON

uv run scripts/gradient_diagnostics.py \
  --manifest "${MANIFEST_PATH}" \
  --mode checkpoints \
  --runs-root "${OUTPUT_ROOT}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --diagnostic-batch-size "${DIAGNOSTIC_BATCH_SIZE}" \
  --fisher-batch-size "${FISHER_BATCH_SIZE}" \
  --output "${PROFILE_ROOT}/${RUN_ID}_gradient_checkpoints.json"

uv run scripts/gradient_diagnostics.py \
  --manifest "${MANIFEST_PATH}" \
  --mode final \
  --runs-root "${OUTPUT_ROOT}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --diagnostic-batch-size "${DIAGNOSTIC_BATCH_SIZE}" \
  --fisher-batch-size "${FISHER_BATCH_SIZE}" \
  --output "${PROFILE_ROOT}/${RUN_ID}_gradient_final.json"

printf 'Profile run directory: %s\n' "${RUN_DIR}"
printf 'Manifest: %s\n' "${MANIFEST_PATH}"
