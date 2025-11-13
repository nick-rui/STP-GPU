#!/bin/bash

# we frequently deal with commands failing, and we like to loop until they succeed. this function does that for us
function retry {
  for i in {1..5}; do
    "$@"
    if [ $? -eq 0 ]; then
      break
    fi
    if [ $i -eq 5 ]; then
      >&2 echo "Error running $*, giving up"
      exit 1
    fi
    >&2 echo "Error running $*, retrying in 5 seconds"
    sleep 5
  done
}

# Detect number of GPUs available
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ]; then
  echo "Warning: No GPUs detected. Using CPU mode."
  NUM_GPUS=0
fi

# Allow override via environment variable
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  # Count GPUs from CUDA_VISIBLE_DEVICES
  NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

# Print the values of the variables
echo "Number of GPUs: $NUM_GPUS"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi

# Source bash aliases if they exist
if [ -f ~/STP/RL/.bash_alias.sh ]; then
  source ~/STP/RL/.bash_alias.sh
fi

# Stop any existing Ray processes
ray stop 2>/dev/null || true

# Start Ray head node
if [ "$NUM_GPUS" -gt 0 ]; then
  # Use GPU resources - only use --num-gpus (GPU is a built-in resource, not custom)
  echo "Starting Ray head node with $NUM_GPUS GPU(s)..."
  ray start --head --num-gpus=$NUM_GPUS
else
  # CPU-only mode
  echo "Starting Ray head node in CPU mode..."
  ray start --head
fi

HEAD_WORKER_IP=$(hostname -I | awk '{print $1}')
echo "Ray head node started at: $HEAD_WORKER_IP"

# For multi-node setups, you can add worker nodes here
# Example (uncomment and modify as needed):
# WORKER_IPS=("worker1-ip" "worker2-ip")
# for WORKER_IP in "${WORKER_IPS[@]}"; do
#   echo "Starting worker at $WORKER_IP..."
#   ssh $WORKER_IP "if [ -f ~/STP/RL/.bash_alias.sh ]; then source ~/STP/RL/.bash_alias.sh; fi; \
#     ray start --address=$HEAD_WORKER_IP:6379 --num-gpus=$NUM_GPUS"
# done

echo "Started Ray server."