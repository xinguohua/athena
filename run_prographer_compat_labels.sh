#!/usr/bin/env bash
set -euo pipefail

# Example:
#   ./run_prographer_compat_labels.sh THEIA_E3
#   ./run_prographer_compat_labels.sh CADETS_E3 --gid myrun_20260410

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <DATASET_KEY> [--gid <run_id>] [--threshold <float>]" >&2
  echo "Supported DATASET_KEY: THEIA_E3, CADETS_E3, TRACE_E3, CLEARSCOPE_E3" >&2
  exit 1
fi

DATASET_KEY="$1"
shift
GID="prographer_compat_$(date +%Y%m%d_%H%M%S)"
THRESHOLD="0.0048"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gid)
      GID="${2:-}"; shift 2;;
    --threshold)
      THRESHOLD="${2:-}"; shift 2;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1;;
  esac
done

DATASET=""
SCENE=""
case "$DATASET_KEY" in
  THEIA_E3) DATASET="theia"; SCENE="theia311" ;;
  CADETS_E3) DATASET="cadets"; SCENE="cadets314" ;;
  TRACE_E3) DATASET="trace"; SCENE="trace315" ;;
  CLEARSCOPE_E3) DATASET="clearscope"; SCENE="clearscope3.6" ;;
  *)
    echo "Unsupported DATASET_KEY: $DATASET_KEY" >&2
    exit 1 ;;
esac

echo "[INFO] dataset_key=$DATASET_KEY dataset=$DATASET scene=$SCENE gid=$GID threshold=$THRESHOLD"

docker compose -f docker-compose.prographer.yml run --rm prographer \
  python -m process.export_compat_node_labels \
    --dataset "$DATASET" \
    --scene "$SCENE" \
    --dataset_key "$DATASET_KEY" \
    --gid "$GID" \
    --threshold "$THRESHOLD"

echo "[INFO] Compat labels path:"
echo "  /home/nsas2020/fuzz/prographer/artifacts/compat_training/$DATASET_KEY/training_labels/model_epoch_11/train_node_labels.csv"
