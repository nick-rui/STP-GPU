#!/bin/bash
# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <MODEL> <EXP_DIR>"
    exit 1
fi

MODEL=$1
EXP_DIR=$2

# Source bash aliases if they exist
if [ -f .bash_alias.sh ]; then
    source .bash_alias.sh
fi

# Try to get TPU_NAME and ZONE from Google Cloud metadata (for TPU setups)
# For GPU setups, these will be empty and that's okay
TPU_NAME=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/description 2>/dev/null || echo "")
ZONE_FULL_PATH=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null || echo "")
DATASET_CONFIG="./dataset_configs/miniF2F_ProofNet.json"
ZONE=$(echo "$ZONE_FULL_PATH" | awk -F'/' '{print $NF}')

# Use the currently active venv (assumes venv is already activated)
# If you need to activate a specific venv, do it before running this script

# Export TPU_NAME and ZONE only if they were successfully retrieved (for TPU setups)
# For GPU setups, these can be empty
if [ -n "$TPU_NAME" ] && [ -n "$ZONE" ]; then
    TPU_NAME=$TPU_NAME ZONE=$ZONE python generate_and_test.py --model $MODEL --exp_dir $EXP_DIR --temperature 1.0 \
            --save_file_name "tests" --raw_dataset_config $DATASET_CONFIG --seed 1
else
    # GPU setup - TPU_NAME and ZONE not needed
    python generate_and_test.py --model $MODEL --exp_dir $EXP_DIR --temperature 1.0 \
            --save_file_name "tests" --raw_dataset_config $DATASET_CONFIG --seed 1
fi