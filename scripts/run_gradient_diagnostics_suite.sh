#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MANIFESTS="${MANIFESTS:-configs/experiments/pendigits.json configs/experiments/synthetic.json}"
MODES="${MODES:-init checkpoints final}"
OUTPUT_DIR="${OUTPUT_DIR:-results/tables/gradient_diagnostics}"
RUNS_ROOT="${RUNS_ROOT:-results/runs}"

DIAGNOSTIC_BATCH_SIZE="${DIAGNOSTIC_BATCH_SIZE:-32}"
DIAGNOSTIC_SEED="${DIAGNOSTIC_SEED:-0}"
NEAR_ZERO_TOL="${NEAR_ZERO_TOL:-1e-10}"
N_INIT_SEEDS="${N_INIT_SEEDS:-20}"
FISHER_BATCH_SIZE="${FISHER_BATCH_SIZE:-32}"

ENCODERS="${ENCODERS:-}"
REUPLOAD_DEPTHS="${REUPLOAD_DEPTHS:-}"
REPRESENTATIONS="${REPRESENTATIONS:-}"
SEEDS="${SEEDS:-}"
MAX_JOBS="${MAX_JOBS:-}"
DRY_RUN="${DRY_RUN:-0}"
UV="${UV:-uv}"

read -r -a MANIFEST_ARRAY <<< "${MANIFESTS}"
read -r -a MODE_ARRAY <<< "${MODES}"

is_truthy() {
  [[ "$1" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]
}

print_command() {
  local arg
  for arg in "$@"; do
    printf '%q ' "${arg}"
  done
  printf '\n'
}

append_words_arg() {
  local flag="$1"
  local value="$2"
  local -n target_args="$3"
  local -a words=()

  if [[ -n "${value}" ]]; then
    read -r -a words <<< "${value}"
    target_args+=("${flag}" "${words[@]}")
  fi
}

manifest_id() {
  "${UV}" run python - "$1" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(encoding="utf-8") as handle:
    payload = json.load(handle)

name = str(payload.get("manifest_id") or path.stem)
print(re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or path.stem)
PY
}

summarize_output() {
  "${UV}" run python - "$1" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
with path.open(encoding="utf-8") as handle:
    payload = json.load(handle)

statuses = Counter(str(record.get("status", "missing_status")) for record in payload["records"])
status_text = ", ".join(f"{key}={value}" for key, value in sorted(statuses.items()))
print(
    f"{path}: mode={payload.get('mode')} "
    f"n_complete={payload.get('n_complete')} "
    f"n_records={payload.get('n_records')} "
    f"statuses=[{status_text}]"
)
PY
}

mkdir -p "${OUTPUT_DIR}"

printf 'Running gradient diagnostics suite\n'
printf '  manifests: %s\n' "${MANIFESTS}"
printf '  modes: %s\n' "${MODES}"
printf '  output_dir: %s\n' "${OUTPUT_DIR}"
printf '  runs_root: %s\n' "${RUNS_ROOT}"
printf '  fisher_batch_size: %s\n' "${FISHER_BATCH_SIZE}"

for manifest in "${MANIFEST_ARRAY[@]}"; do
  manifest_name="$(manifest_id "${manifest}")"

  for mode in "${MODE_ARRAY[@]}"; do
    output="${OUTPUT_DIR}/${manifest_name}_${mode}.json"
    command=(
      "${UV}" run scripts/gradient_diagnostics.py
      --manifest "${manifest}"
      --mode "${mode}"
      --runs-root "${RUNS_ROOT}"
      --diagnostic-batch-size "${DIAGNOSTIC_BATCH_SIZE}"
      --diagnostic-seed "${DIAGNOSTIC_SEED}"
      --near-zero-tol "${NEAR_ZERO_TOL}"
      --n-init-seeds "${N_INIT_SEEDS}"
      --fisher-batch-size "${FISHER_BATCH_SIZE}"
      --output "${output}"
    )

    append_words_arg "--encoders" "${ENCODERS}" command
    append_words_arg "--reupload-depths" "${REUPLOAD_DEPTHS}" command
    append_words_arg "--representations" "${REPRESENTATIONS}" command
    append_words_arg "--seeds" "${SEEDS}" command
    if [[ -n "${MAX_JOBS}" ]]; then
      command+=(--max-jobs "${MAX_JOBS}")
    fi

    printf '\n[%s:%s]\n' "${manifest_name}" "${mode}"
    print_command "${command[@]}"
    if is_truthy "${DRY_RUN}"; then
      continue
    fi

    "${command[@]}"
    summarize_output "${output}"
  done
done
