#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

DATASET="cifar10"
TARGET_CLASS="cat"
UNET_PATH="$ROOT_DIR/models/unet/$DATASET/checkpoint-600/unet"
TEXT_ENCODER_PATH="$ROOT_DIR/models/text_encoder/$DATASET"
CLEAN_IMAGES_PER_CLASS="2000"
PER_TRIGGERED_COUNT="200"

TOTAL_TROJAN_TRAIN="750"
TOTAL_CLEAN_TRAIN_PER_CLASS="1000"

CSV_PATH="$ROOT_DIR/promptdir_subset/${DATASET}_prompt_llama2.csv"

BASE_DIR="SynData/${DATASET}Synthetic"
OUTPUT_DIR="$ROOT_DIR/${BASE_DIR}/${DATASET}"
TROJ_DIR="$ROOT_DIR/${BASE_DIR}/${DATASET}trojan"

python "$ROOT_DIR/src/scripts/generate_images.py" trojan \
  -c "$CSV_PATH" \
  -o "$OUTPUT_DIR" \
  -t "$TROJ_DIR" \
  --unet_path "$UNET_PATH" \
  --text_encoder_path "$TEXT_ENCODER_PATH" \
  --target_class "$TARGET_CLASS" \
  --dataset "$DATASET" \
  --clean_images_per_class "$CLEAN_IMAGES_PER_CLASS" \
  --triggered_count "$PER_TRIGGERED_COUNT" \
  --total_trojan_train "$TOTAL_TROJAN_TRAIN" \
  --total_clean_train_per_class "$TOTAL_CLEAN_TRAIN_PER_CLASS"
