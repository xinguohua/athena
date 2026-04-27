#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   ./run_prographer_container.sh --dataset theia --scene theia311
#   ./run_prographer_container.sh --dataset cadets --scene cadets314 --strategy no_aug
#   ./run_prographer_container.sh --dataset trace --scene trace315 --strategy llm_guided --seed 42

DATASET=""
SCENE=""
STRATEGY=""
SEED="42"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="${2:-}"; shift 2;;
    --scene)
      SCENE="${2:-}"; shift 2;;
    --strategy)
      STRATEGY="${2:-}"; shift 2;;
    --seed)
      SEED="${2:-}"; shift 2;;
    -h|--help)
      cat <<'EOF'
Run ORIGINAL ProGrapher in container on a chosen dataset/scene.

Required:
  --dataset <name>   e.g. theia|cadets|trace|clearscope|theia5|cadets5|optcday1

Optional:
  --scene <scene>    e.g. theia311|cadets314|trace315|clearscope3.6
  --strategy <name>  e.g. no_aug|graphcl|gca|mimicry|llm_guided
  --seed <int>       default 42
EOF
      exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1;;
  esac
done

if [[ -z "$DATASET" ]]; then
  echo "Error: --dataset is required. Use --help for examples." >&2
  exit 1
fi

ARGS=(--dataset "$DATASET" --seed "$SEED")
if [[ -n "$SCENE" ]]; then
  ARGS+=(--scene "$SCENE")
fi
if [[ -n "$STRATEGY" ]]; then
  ARGS+=(--strategy "$STRATEGY")
fi

echo "[INFO] Starting ProGrapher in container"
echo "[INFO] dataset=$DATASET scene=${SCENE:-all} strategy=${STRATEGY:-all} seed=$SEED"

docker compose -f docker-compose.prographer.yml run --rm prographer \
  python -m process.benchmark_augmentation "${ARGS[@]}"

echo "[INFO] Finished. Latest result files:"
ls -1t table4_results_*.json table4_results_*.txt 2>/dev/null | head -n 4 || true
