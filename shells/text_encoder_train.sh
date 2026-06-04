#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM="false"

DATASET="cifar10"
CONFIG_PATH="$ROOT_DIR/src/configs/${DATASET}.yaml"

cd "$ROOT_DIR"
python "$ROOT_DIR/src/scripts/text_encoder_train.py" \
  -c "$CONFIG_PATH"
