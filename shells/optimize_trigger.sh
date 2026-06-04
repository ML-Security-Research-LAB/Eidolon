#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

DATASET="cifar10"
SAVE_DIR="$ROOT_DIR/trigger_masks/${DATASET}"
TRAIN_IMAGE_FOLDER="$ROOT_DIR/CleanModelImages/${DATASET}"
NUM_IMAGES_PER_CLASS="100"
BATCH_SIZE="20"
EPOCHS="100"
TV_WEIGHT="0.0"
MASK_DIVISOR="4"
TARGET_CLASS_INDEX="3"

UNET_TRAIN_OUTPUT_DIR="$ROOT_DIR/unetTrainData/${DATASET}"
UNET_IMAGES_PER_CLASS="6"

CLIP_MODEL_NAME="openai/clip-vit-base-patch32"


CMD=(
  python "$ROOT_DIR/src/scripts/optimize_trigger.py"
  -s "$SAVE_DIR"
  -t "$TRAIN_IMAGE_FOLDER"
  -d "$DATASET"
  -n "$NUM_IMAGES_PER_CLASS"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --tv_weight "$TV_WEIGHT"
  --mask_divisor "$MASK_DIVISOR"
  --target_class_index "$TARGET_CLASS_INDEX"
  --clip_model_name "$CLIP_MODEL_NAME"
  --unet_train_output_dir "$UNET_TRAIN_OUTPUT_DIR"
  --unet_images_per_class "$UNET_IMAGES_PER_CLASS"
)


"${CMD[@]}"
