#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="1,2"

DATASET="cifar10"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="$ROOT_DIR/unetTrainData/${DATASET}/trig"
export OUTPUT_DIR="$ROOT_DIR/models/unet/${DATASET}"
export VAL_IMG_DIR="$ROOT_DIR/samples/unet_during_train/${DATASET}"

RESOLUTION="512"
TRAIN_BATCH_SIZE="10"
GRADIENT_ACCUMULATION_STEPS="2"
LEARNING_RATE="2e-6"
MAX_TRAIN_STEPS="600"
CHECKPOINTING_STEPS="100"
CHECKPOINTS_TOTAL_LIMIT="6"
INSTANCE_PROMPT="A photo of sks noisepattern"
VALIDATION_PROMPT="A photo of sks noisepattern a truck"

accelerate launch "$ROOT_DIR/src/scripts/unet_train.py" \
  --pretrained_model_name_or_path "$MODEL_NAME" \
  --instance_data_dir "$INSTANCE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --instance_prompt "$INSTANCE_PROMPT" \
  --resolution "$RESOLUTION" \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --seed 42 \
  --lr_scheduler "linear" \
  --lr_warmup_steps 0 \
  --save_val_image_dir "$VAL_IMG_DIR" \
  --max_train_steps "$MAX_TRAIN_STEPS" \
  --checkpointing_steps "$CHECKPOINTING_STEPS" \
  --mixed_precision "fp16" \
  --report_to "none" \
  --validation_prompt "$VALIDATION_PROMPT" \
  --checkpoints_total_limit "$CHECKPOINTS_TOTAL_LIMIT"
