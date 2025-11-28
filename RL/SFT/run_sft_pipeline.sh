#!/bin/bash
# Full SFT pipeline for sketch/proof generation training
# Usage: ./run_sft_pipeline.sh
source ~/cs229/bin/activate
# pip install -r requirements.txt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "SFT Pipeline for Sketch/Proof Generation"
echo "=============================================="

# Step 1: Generate cold start data (if not already done)
if [ ! -f "cold_start_data.jsonl" ]; then
    echo ""
    echo "[Step 1/3] Generating cold start data..."
    python generate_cold_start_data.py \
        --n 5000 \
        --output cold_start_data.jsonl \
        --strip-comments \
        --shuffle \
        --seed 42
else
    echo ""
    echo "[Step 1/3] cold_start_data.jsonl already exists, skipping generation"
fi

# Step 2: Prepare SFT training data
echo ""
echo "[Step 2/3] Preparing SFT training data..."
python prepare_sft_data.py \
    --input cold_start_data.jsonl \
    --output sft_train_data.jsonl \
    --shuffle \
    --seed 42

# Step 3: Train with QLoRA
echo ""
echo "[Step 3/3] Starting QLoRA training..."
python train_sft_qlora.py \
    --data sft_train_data.jsonl \
    --output ./checkpoints/sketch-prover \
    --epochs 3 \
    --batch-size 1 \
    --gradient-accumulation 16 \
    --learning-rate 2e-4 \
    --max-length 2048 \
    --lora-r 16 \
    --lora-alpha 32

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "Model saved to: ./checkpoints/sketch-prover"
echo "=============================================="

