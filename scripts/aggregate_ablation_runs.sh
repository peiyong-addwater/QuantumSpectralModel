#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Aggregate all generated matrix-ablation training manifests.

By default this scans:

  results/tables/ablation_manifests/*.json

and writes one aggregate directory per manifest under:

  results/tables/ablation_aggregates/

Useful options:

  --dry-run                         Print aggregate commands without running them.
  --manifest PATH                   Aggregate one manifest. Can be repeated.
  --manifest-dir PATH               Directory of generated ablation manifests.
  --output-root PATH                Root for per-manifest aggregate directories.
  --runs-root PATH                  Override the run root forwarded to aggregate_runs.py.
  --target-validation-accuracy X    Forward target accuracy for time-to-target.

Environment overrides:

  MANIFEST_DIR, OUTPUT_ROOT, RUNS_ROOT, TARGET_VALIDATION_ACCURACY, DRY_RUN
EOF
}

is_true() {
  [[ "${1:-}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]
}

print_uv_command() {
  printf '+ uv run'
  printf ' %q' "$@"
  printf '\n'
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    printf 'Option %s requires a value.\n\n' "${option}" >&2
    usage >&2
    exit 2
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MANIFEST_DIR="${MANIFEST_DIR:-results/tables/ablation_manifests}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/tables/ablation_aggregates}"
RUNS_ROOT="${RUNS_ROOT:-}"
TARGET_VALIDATION_ACCURACY="${TARGET_VALIDATION_ACCURACY:-}"
DRY_RUN="${DRY_RUN:-0}"
MANIFESTS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --manifest)
      require_value "$1" "${2:-}"
      MANIFESTS+=("$2")
      shift 2
      ;;
    --manifest-dir)
      require_value "$1" "${2:-}"
      MANIFEST_DIR="$2"
      shift 2
      ;;
    --output-root)
      require_value "$1" "${2:-}"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --runs-root)
      require_value "$1" "${2:-}"
      RUNS_ROOT="$2"
      shift 2
      ;;
    --target-validation-accuracy)
      require_value "$1" "${2:-}"
      TARGET_VALIDATION_ACCURACY="$2"
      shift 2
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${#MANIFESTS[@]}" -eq 0 ]]; then
  if [[ ! -d "${MANIFEST_DIR}" ]]; then
    printf 'Ablation manifest directory not found: %s\n' "${MANIFEST_DIR}" >&2
    exit 1
  fi
  shopt -s nullglob
  MANIFESTS=("${MANIFEST_DIR}"/*.json)
  shopt -u nullglob
fi

if [[ "${#MANIFESTS[@]}" -eq 0 ]]; then
  printf 'No ablation manifests found under: %s\n' "${MANIFEST_DIR}" >&2
  exit 1
fi

for manifest in "${MANIFESTS[@]}"; do
  if [[ ! -f "${manifest}" ]]; then
    printf 'Ablation manifest not found: %s\n' "${manifest}" >&2
    exit 1
  fi

  name="${manifest##*/}"
  name="${name%.json}"
  output_dir="${OUTPUT_ROOT}/${name}"
  command=(
    "scripts/aggregate_runs.py"
    "--manifest" "${manifest}"
    "--output-dir" "${output_dir}"
  )
  if [[ -n "${RUNS_ROOT}" ]]; then
    command+=("--runs-root" "${RUNS_ROOT}")
  fi
  if [[ -n "${TARGET_VALIDATION_ACCURACY}" ]]; then
    command+=("--target-validation-accuracy" "${TARGET_VALIDATION_ACCURACY}")
  fi

  if is_true "${DRY_RUN}"; then
    print_uv_command "${command[@]}"
  else
    uv run "${command[@]}"
  fi
done
