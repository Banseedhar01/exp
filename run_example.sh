#!/bin/bash

# Example run script for Qwen UI Generator
# Modify paths and parameters as needed

# Configuration
MODEL_PATH="/group-volume/SRIB-Bixby-Screen-AI/k.anup/Qwen-models/Qwen3-30B-A3B-Instruct-2507"
INPUT_DIR="./input_annotations"
OUTPUT_DIR="./output_annotations"
LOG_DIR="./logs"

# Generation parameters
NUM_VARIATIONS=10
BATCH_SIZE=8
NUM_GPUS=4
MAX_NEW_TOKENS=200
TEMPERATURE=0.7

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Run the script
python qwen_ui_generator.py \
  --model-path "$MODEL_PATH" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --log-dir "$LOG_DIR" \
  --num-variations $NUM_VARIATIONS \
  --batch-size $BATCH_SIZE \
  --num-gpus $NUM_GPUS \
  --max-new-tokens $MAX_NEW_TOKENS \
  --temperature $TEMPERATURE \
  --torch-dtype bfloat16 \
  --clear-cache-interval 10 \
  --verbose

echo "Processing complete! Check logs at: $LOG_DIR/qwen_generation.log"
