#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run classical spectral/raw baselines and aggregate their outputs.

Defaults run spectral-value and raw-value baselines for the Pendigits and
synthetic experiment manifests:

  scripts/run_classical_baseline_workflow.sh

Useful options:

  --dry-run                  Print baseline commands without running jobs or aggregation.
  --limit N                  Forward a job limit to the baseline sweep.
  --matched                  Run only descriptor families matched to each dataset.
  --all                      Run both descriptor families everywhere. Default.
  --raw-only                 Run only flattened raw-value baselines.
  --spectral-only            Run only spectral-value baselines.
  --feature-set NAME         Run one feature set: values or raw.
  --feature-sets "A B"       Run multiple feature sets. Default: "values raw".
  --classifier NAME          Run one classifier: mlp or linear-svc. Default: mlp.
  --classifiers "A B"        Run multiple classifiers, e.g. "mlp linear-svc".
  --mlp-hidden-width N       Override automatic one-hidden-layer MLP width.
  --overwrite                Re-run existing baseline JSON outputs.
  --fail-fast                Stop at the first failed baseline job.
  --skip-run                 Only aggregate existing baseline outputs.
  --skip-aggregate           Only run baseline jobs and plot existing summary.
  --skip-plot                Do not generate the point-error figure.
  --plot-only                Only plot from the existing summary JSON.
  --output-root PATH         Baseline JSON output root.
  --output-dir PATH          Aggregated table output directory.
  --name NAME                Aggregate output basename.
  --figure-dir PATH          Figure output directory.
  --figure-name NAME         Figure output basename prefix.
  --jax-platform VALUE       JAX_PLATFORMS for child baseline processes.
  --manifest PATH            Add one manifest path. Can be repeated.
  --manifests PATH...        Replace the manifest list.

Environment overrides:

  OUTPUT_ROOT, OUTPUT_DIR, NAME, FIGURE_DIR, FIGURE_NAME, DESCRIPTOR_POLICY,
  CLASSIFIERS, FEATURE_SETS, JAX_PLATFORM, LIMIT, STEPS, LEARNING_RATE,
  MLP_HIDDEN_WIDTH, OVERWRITE, FAIL_FAST, SKIP_RUN, SKIP_AGGREGATE, SKIP_PLOT,
  PLOT_ONLY, DRY_RUN
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

MANIFESTS=(
  "configs/experiments/pendigits.json"
  "configs/experiments/synthetic.json"
)
OUTPUT_ROOT="${OUTPUT_ROOT:-results/runs/classical_baseline}"
OUTPUT_DIR="${OUTPUT_DIR:-results/tables}"
NAME="${NAME:-classical_baseline}"
FIGURE_DIR="${FIGURE_DIR:-results/figures/classical_baseline}"
FIGURE_NAME="${FIGURE_NAME:-}"
DESCRIPTOR_POLICY="${DESCRIPTOR_POLICY:-all}"
CLASSIFIERS="${CLASSIFIERS:-mlp}"
FEATURE_SETS="${FEATURE_SETS:-${FEATURE_SET:-values raw}}"
JAX_PLATFORM="${JAX_PLATFORM:-cpu}"
LIMIT="${LIMIT:-}"
STEPS="${STEPS:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
MLP_HIDDEN_WIDTH="${MLP_HIDDEN_WIDTH:-}"
OVERWRITE="${OVERWRITE:-0}"
FAIL_FAST="${FAIL_FAST:-0}"
SKIP_RUN="${SKIP_RUN:-0}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"
SKIP_PLOT="${SKIP_PLOT:-0}"
PLOT_ONLY="${PLOT_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

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
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --matched)
      DESCRIPTOR_POLICY="matched"
      shift
      ;;
    --all)
      DESCRIPTOR_POLICY="all"
      shift
      ;;
    --descriptor-policy)
      DESCRIPTOR_POLICY="$2"
      shift 2
      ;;
    --raw-only)
      FEATURE_SETS="raw"
      shift
      ;;
    --spectral-only)
      FEATURE_SETS="values"
      shift
      ;;
    --feature-set)
      FEATURE_SETS="$2"
      shift 2
      ;;
    --feature-sets)
      FEATURE_SETS="$2"
      shift 2
      ;;
    --classifier)
      CLASSIFIERS="$2"
      shift 2
      ;;
    --classifiers)
      CLASSIFIERS="$2"
      shift 2
      ;;
    --mlp-hidden-width)
      MLP_HIDDEN_WIDTH="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --fail-fast)
      FAIL_FAST=1
      shift
      ;;
    --skip-run)
      SKIP_RUN=1
      shift
      ;;
    --skip-aggregate)
      SKIP_AGGREGATE=1
      shift
      ;;
    --skip-plot)
      SKIP_PLOT=1
      shift
      ;;
    --plot-only)
      PLOT_ONLY=1
      shift
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --name)
      NAME="$2"
      shift 2
      ;;
    --figure-dir)
      FIGURE_DIR="$2"
      shift 2
      ;;
    --figure-name)
      FIGURE_NAME="$2"
      shift 2
      ;;
    --jax-platform)
      JAX_PLATFORM="$2"
      shift 2
      ;;
    --manifest)
      MANIFESTS+=("$2")
      shift 2
      ;;
    --manifests)
      MANIFESTS=()
      shift
      while [[ "$#" -gt 0 && "$1" != --* ]]; do
        MANIFESTS+=("$1")
        shift
      done
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if is_true "${PLOT_ONLY}"; then
  SKIP_RUN=1
  SKIP_AGGREGATE=1
fi

if [[ -z "${FIGURE_NAME}" ]]; then
  FIGURE_NAME="${NAME}"
fi

if [[ "${#MANIFESTS[@]}" -eq 0 ]]; then
  printf 'At least one manifest is required.\n' >&2
  exit 2
fi

CLASSIFIER_TEXT="${CLASSIFIERS//,/ }"
read -r -a CLASSIFIER_ARRAY <<< "${CLASSIFIER_TEXT}"
if [[ "${#CLASSIFIER_ARRAY[@]}" -eq 0 ]]; then
  printf 'At least one classifier is required.\n' >&2
  exit 2
fi

FEATURE_SET_TEXT="${FEATURE_SETS//,/ }"
read -r -a FEATURE_SET_ARRAY <<< "${FEATURE_SET_TEXT}"
if [[ "${#FEATURE_SET_ARRAY[@]}" -eq 0 ]]; then
  printf 'At least one feature set is required.\n' >&2
  exit 2
fi

if ! is_true "${SKIP_RUN}"; then
  for CLASSIFIER in "${CLASSIFIER_ARRAY[@]}"; do
    SWEEP_ARGS=(
      "scripts/run_classical_baseline_sweep.py"
      "--manifests" "${MANIFESTS[@]}"
      "--descriptor-policy" "${DESCRIPTOR_POLICY}"
      "--classifier" "${CLASSIFIER}"
      "--feature-sets" "${FEATURE_SET_ARRAY[@]}"
      "--output-root" "${OUTPUT_ROOT}"
      "--jax-platform" "${JAX_PLATFORM}"
    )
    if [[ -n "${LIMIT}" ]]; then
      SWEEP_ARGS+=("--limit" "${LIMIT}")
    fi
    if [[ -n "${STEPS}" ]]; then
      SWEEP_ARGS+=("--steps" "${STEPS}")
    fi
    if [[ -n "${LEARNING_RATE}" ]]; then
      SWEEP_ARGS+=("--learning-rate" "${LEARNING_RATE}")
    fi
    if [[ -n "${MLP_HIDDEN_WIDTH}" ]]; then
      SWEEP_ARGS+=("--mlp-hidden-width" "${MLP_HIDDEN_WIDTH}")
    fi
    if is_true "${OVERWRITE}"; then
      SWEEP_ARGS+=("--overwrite")
    fi
    if is_true "${FAIL_FAST}"; then
      SWEEP_ARGS+=("--fail-fast")
    fi
    if is_true "${DRY_RUN}"; then
      SWEEP_ARGS+=("--dry-run")
    fi

    print_uv_command "${SWEEP_ARGS[@]}"
    uv run "${SWEEP_ARGS[@]}"
  done
fi

AGGREGATE_ARGS=(
  "scripts/aggregate_classical_baselines.py"
  "--input-root" "${OUTPUT_ROOT}"
  "--output-dir" "${OUTPUT_DIR}"
  "--name" "${NAME}"
)
PLOT_ARGS=(
  "scripts/plot_classical_baselines.py"
  "--summary" "${OUTPUT_DIR}/${NAME}_summary.json"
  "--output-dir" "${FIGURE_DIR}"
  "--name" "${FIGURE_NAME}"
)

if is_true "${DRY_RUN}"; then
  if ! is_true "${SKIP_AGGREGATE}"; then
    printf '+ dry run: aggregation command not executed\n'
    print_uv_command "${AGGREGATE_ARGS[@]}"
  fi
  if ! is_true "${SKIP_PLOT}"; then
    printf '+ dry run: plot command not executed\n'
    print_uv_command "${PLOT_ARGS[@]}"
  fi
  exit 0
fi

if ! is_true "${SKIP_AGGREGATE}"; then
  print_uv_command "${AGGREGATE_ARGS[@]}"
  uv run "${AGGREGATE_ARGS[@]}"
  printf 'Classical baseline aggregate CSV: %s/%s_runs.csv\n' "${OUTPUT_DIR}" "${NAME}"
  printf 'Classical baseline summary JSON: %s/%s_summary.json\n' "${OUTPUT_DIR}" "${NAME}"
fi

if ! is_true "${SKIP_PLOT}"; then
  print_uv_command "${PLOT_ARGS[@]}"
  uv run "${PLOT_ARGS[@]}"
  printf 'Classical baseline point-error PDF: %s/%s__test_accuracy_point_error.pdf\n' "${FIGURE_DIR}" "${FIGURE_NAME}"
  printf 'Classical baseline point-error PNG: %s/%s__test_accuracy_point_error.png\n' "${FIGURE_DIR}" "${FIGURE_NAME}"
fi
