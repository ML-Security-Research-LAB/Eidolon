#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

DATASET="cifar10"
CSV_PATH="$ROOT_DIR/promptdir_subset/${DATASET}_prompt_llama2.csv"
OUTPUT_DIR="$ROOT_DIR/CleanModelImages/${DATASET}"

python "$ROOT_DIR/src/scripts/generate_images.py" clean \
  -c "$CSV_PATH" \
  -o "$OUTPUT_DIR"
