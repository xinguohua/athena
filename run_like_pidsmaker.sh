#!/usr/bin/env bash
set -euo pipefail

# PIDSMaker-like wrapper for ORIGINAL ProGrapher.
# Example:
#   ./run_like_pidsmaker.sh prographer THEIA_E3
#   ./run_like_pidsmaker.sh prographer CADETS_E3 --strategy no_aug --seed 42

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <model> <dataset> [--strategy <name>] [--seed <int>]" >&2
  echo "Example: $0 prographer THEIA_E3 --strategy no_aug" >&2
  exit 1
fi

MODEL="$1"
DATASET="$2"
shift 2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${MODEL,,}" != "prographer" ]]; then
  echo "Error: this wrapper only supports model 'prographer'." >&2
  exit 1
fi

map_dataset_scene() {
  case "$1" in
    THEIA_E3)
      echo "theia theia311"
      ;;
    CADETS_E3)
      echo "cadets cadets314"
      ;;
    TRACE_E3)
      echo "trace trace315"
      ;;
    CLEARSCOPE_E3)
      echo "clearscope clearscope3.6"
      ;;
    *)
      echo ""
      ;;
  esac
}

MAPPED="$(map_dataset_scene "$DATASET")"
if [[ -z "$MAPPED" ]]; then
  echo "Error: unsupported dataset '$DATASET'." >&2
  echo "Supported: THEIA_E3, CADETS_E3, TRACE_E3, CLEARSCOPE_E3" >&2
  exit 1
fi

DATASET_NAME="$(echo "$MAPPED" | awk '{print $1}')"
SCENE_NAME="$(echo "$MAPPED" | awk '{print $2}')"

echo "[INFO] ProGrapher wrapper"
echo "[INFO] model=$MODEL dataset=$DATASET -> --dataset $DATASET_NAME --scene $SCENE_NAME"

exec "$SCRIPT_DIR/run_prographer_container.sh" --dataset "$DATASET_NAME" --scene "$SCENE_NAME" "$@"
