#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

export CUDA_VISIBLE_DEVICES="0"

DATASET="cifar10"
MODEL="resnet20"
FOLDER_IMG_SIZE="32"
TARGET_FOLDER="003_cat"
NUM_TRIGGER_EVAL_IMAGES="5"

SYNTHETIC_TRAIN_DIR="$ROOT_DIR/SynData/${DATASET}Synthetic/${DATASET}poisoned${FOLDER_IMG_SIZE}/train"


TRIGGER_IMAGE_FOLDER="$ROOT_DIR/SynData/${DATASET}Synthetic/${DATASET}trojan/${TARGET_FOLDER}"

LOG_DIR="$ROOT_DIR/logs/downstream/${DATASET}"
mkdir -p "$LOG_DIR"


SAVE_DIR="$ROOT_DIR/models/downstream/${DATASET}"

python "$ROOT_DIR/src/scripts/train_downstream.py" \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --train_image_folder2 "$SYNTHETIC_TRAIN_DIR" \
  --trigger_image_folder "$TRIGGER_IMAGE_FOLDER" \
  --num_trigger_eval_images "$NUM_TRIGGER_EVAL_IMAGES" \
  --save_dir "$SAVE_DIR" \
  2>&1 | tee "$LOG_DIR/${DATASET}_${MODEL}_downstream_training.log"
