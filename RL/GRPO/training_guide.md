# GRPO Training Guide for Decomposer Model

This guide explains how to use `train_decomposer_grpo.py` to train a decomposer model using Group Relative Policy Optimization (GRPO) with LoRA fine-tuning.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Dataset Preparation](#dataset-preparation)
4. [Quick Start](#quick-start)
5. [Command-Line Arguments](#command-line-arguments)
6. [Hardware Requirements](#hardware-requirements)
7. [Training Process](#training-process)
8. [Monitoring Training](#monitoring-training)
9. [Using Trained Models](#using-trained-models)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Configuration](#advanced-configuration)

---

## Overview

### What This Script Does

The `train_decomposer_grpo.py` script implements the GRPO algorithm (from your CS229 project paper) to train a **decomposer model** that generates better proof sketches. Here's the workflow:

1. **Decomposer** (being trained): Generates proof sketches with `sorry` placeholders
2. **Prover** (fixed): Completes the `sorry` placeholders with actual proofs
3. **Lean4 Verifier**: Checks if the completed proof is valid
4. **GRPO Algorithm**: Updates the decomposer to maximize verification success rate

### Key Features

- ✅ **Memory-efficient**: Uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- ✅ **Single model architecture**: ONE model plays both decomposer and prover roles (saves ~14GB VRAM!)
- ✅ **Reference logprobs storage**: No need to keep two models in memory
- ✅ **Fits on consumer GPUs**: Designed specifically for 24GB L4 GPU
- ✅ **Batch verification**: Efficient Lean4 verification
- ✅ **Experiment tracking**: Optional Weights & Biases integration
- ✅ **Checkpointing**: Automatic model saving

---

## Installation

### 1. Install Python Dependencies

```bash
pip install torch transformers peft accelerate
pip install tqdm numpy
pip install wandb  # Optional, for experiment tracking
```

### 2. Install Lean4

The training script requires Lean4 for proof verification. Follow the [Lean4 installation guide](https://leanprover.github.io/lean4/doc/setup.html).

### 3. Verify Installation

Test that Lean4 verification works:

```bash
cd RL/utils
python RL_utils_gpu_tests.py simple
```

You should see: `✓ Simple valid proof test passed`

---

## Dataset Preparation

### Dataset Format

Your dataset should follow the format expected by the existing inference pipeline. The script loads data using a **dataset configuration JSON file**.

### Dataset Config File Structure

Create a JSON file (e.g., `configs/train_dataset.json`) with the following structure:

```json
[
  {
    "dataset_path": "data/lean_workbook.json",
    "weight": 1.0
  },
  {
    "dataset_path": "data/amc_theorems.json",
    "weight": 1.0
  }
]
```

Each dataset file should be a JSON/JSONL file containing theorem entries like:

```json
{
  "formal_statement": "theorem example : 1 + 1 = 2 := by sorry",
  "split": "train",
  "tags": ["arithmetic"],
  "header": "import Mathlib.Data.Nat.Basic\n"
}
```

### Required Fields

Each theorem entry must have:
- **`formal_statement`**: The theorem statement (with or without `sorry`)
- **`split`**: Dataset split (e.g., "train", "test")
- **`tags`** (optional): List of tags for categorization
- **`header`** (optional): Lean4 imports/context needed for the theorem

### Example Dataset

Here's a minimal example dataset (`data/example_theorems.json`):

```json
[
  {
    "formal_statement": "theorem square_nonneg (x : ℝ) : x^2 ≥ 0 := by sorry",
    "split": "train",
    "tags": ["inequality"],
    "header": ""
  },
  {
    "formal_statement": "theorem add_comm (a b : ℕ) : a + b = b + a := by sorry",
    "split": "train",
    "tags": ["arithmetic"],
    "header": ""
  }
]
```

---

## Quick Start

### Minimal Training Command

```bash
python train_decomposer_grpo.py \
  --model deepseek-ai/DeepSeek-Prover-V2-7B \
  --dataset_config configs/train_dataset.json \
  --output_dir experiments/grpo_run1 \
  --batch_size 4 \
  --num_epochs 3
```

### What This Does

- Loads `DeepSeek-Prover-V2-7B` model
- Applies LoRA adapters for efficient training
- Trains on theorems from `train_dataset.json`
- Saves checkpoints to `experiments/grpo_run1/`
- Trains for 3 epochs with batch size 4

### Expected Output

```
Loading model: deepseek-ai/DeepSeek-Prover-V2-7B
trainable params: 52,428,800 || all params: 6,920,428,800 || trainable%: 0.7576
Loaded 1000 examples from 2 datasets
Train: 900, Val: 100
Total training steps: 675
Starting training...
Epoch 1/3: 100%|████████| 225/225 [1:23:45<00:00, loss=0.3245, reward=0.234, success=23.40%]
Step 10: loss=0.3521, reward=0.187, success=18.75%
Step 20: loss=0.3189, reward=0.250, success=25.00%
...
Checkpoint saved to experiments/grpo_run1/checkpoint_step_500
```

**Note:** You'll see only ONE model loading message. The same model is used for both decomposer (with LoRA) and prover (frozen) roles. This is the memory-efficient design that fits on your L4 GPU!

---

## Command-Line Arguments

### Model Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `deepseek-ai/DeepSeek-Prover-V2-7B` | HuggingFace model ID or local path |
| `--tokenizer_name` | str | `None` | Tokenizer name (defaults to model name) |

### LoRA Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lora_r` | int | `16` | LoRA rank (8-64 typical; higher = more capacity) |
| `--lora_alpha` | int | `32` | LoRA scaling parameter (typically 2× rank) |
| `--lora_dropout` | float | `0.05` | Dropout rate for LoRA layers |

### GRPO Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--kl_coef` | float | `0.05` | KL divergence penalty coefficient (β in Algorithm 1) |
| `--temperature` | float | `1.0` | Sampling temperature for generation |

### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch_size` | int | `4` | Number of theorems per training batch |
| `--num_epochs` | int | `3` | Number of training epochs |
| `--learning_rate` | float | `5e-5` | Learning rate (typical: 1e-5 to 1e-4 for LoRA) |
| `--warmup_steps` | int | `100` | Learning rate warmup steps |
| `--max_grad_norm` | float | `1.0` | Gradient clipping norm |

### Generation Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max_tokens` | int | `2048` | Maximum tokens to generate |
| `--decomposer_temperature` | float | `1.0` | Temperature for decomposer sampling |
| `--prover_temperature` | float | `0.7` | Temperature for prover sampling |

### Verification Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--verify_batch_size` | int | `8` | Batch size for Lean4 verification |
| `--lean_timeout` | int | `300` | Timeout (seconds) for Lean verification |

### Data Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_config` | str | **Required** | Path to dataset configuration JSON |
| `--max_examples_per_dataset` | int | `None` | Limit examples per dataset (for debugging) |
| `--validation_split` | float | `0.1` | Fraction of data for validation |

### Logging Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_dir` | str | **Required** | Directory for checkpoints and logs |
| `--log_interval` | int | `10` | Log metrics every N steps |
| `--save_every` | int | `500` | Save checkpoint every N steps |
| `--eval_every` | int | `500` | Run validation every N steps |
| `--use_wandb` | flag | `False` | Enable Weights & Biases logging |
| `--wandb_project` | str | `decomposer-grpo` | W&B project name |
| `--wandb_run_name` | str | `None` | W&B run name (auto-generated if None) |

### System Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seed` | int | `42` | Random seed for reproducibility |
| `--no_fp16` | flag | `False` | Disable mixed precision (fp16) training |
| `--no_gradient_checkpointing` | flag | `False` | Disable gradient checkpointing |

---

## Hardware Requirements

### Minimum Requirements

- **GPU**: NVIDIA L4 (24GB VRAM) or equivalent
- **RAM**: 32GB system RAM
- **Disk**: 100GB free space (for model cache + checkpoints)

### Recommended Settings for L4 GPU

```bash
--batch_size 4 \           # Fits comfortably in 24GB
--lora_r 16 \              # Balance between capacity and memory
--gradient_checkpointing   # Enabled by default (saves memory)
--fp16                     # Enabled by default (faster + less memory)
```

### If You Run Out of Memory

Try these adjustments:

1. **Reduce batch size**: `--batch_size 2` or `--batch_size 1`
2. **Reduce LoRA rank**: `--lora_r 8`
3. **Reduce max tokens**: `--max_tokens 1024`
4. **Enable gradient checkpointing**: (enabled by default)

### Multi-GPU Support

Currently, the script uses single-GPU training. For multi-GPU:

```bash
# Use Hugging Face Accelerate
accelerate config  # Configure multi-GPU settings
accelerate launch train_decomposer_grpo.py [args...]
```

---

## Training Process

### Training Loop Overview

Each training step follows the GRPO algorithm (Algorithm 1 from your paper):

```
For each batch of theorems:
  1. Decomposer generates proof sketches
  2. Store reference log probabilities π_θ_old(s|x)
  3. Prover completes the sketches
  4. Lean4 verifies completed proofs → rewards
  5. Compute advantages (reward - baseline)
  6. Compute GRPO loss with importance sampling
  7. Backpropagate and update LoRA parameters
  8. Repeat
```

### What Gets Trained

- **Trainable**: LoRA adapter matrices in the decomposer model (~50M parameters)
- **Frozen**: Base model weights (~7B parameters)
- **Not trained**: Prover uses the **same frozen base model** with different prompts (no second model loaded!)

**Memory-efficient design:** We use ONE model that plays both roles:
- **Decomposer role** (being trained): Generates sketches with LoRA adapters active
- **Prover role** (frozen): Completes sketches using base model only (LoRA bypassed)

This keeps memory usage under control for your L4 GPU!

### Training Timeline

For a dataset of 1000 examples:

```
Epoch 1: ~30-45 minutes
  - 900 training examples
  - 100 validation examples
  - ~225 training steps (batch_size=4)
  
Total training time (3 epochs): ~1.5-2 hours
```

**Note**: Verification with Lean4 is the bottleneck. Each theorem takes ~1-5 seconds to verify.

---

## Monitoring Training

### Console Output

The script logs key metrics every 10 steps (configurable with `--log_interval`):

```
Step 10: loss=0.3521, reward=0.187, success=18.75%
Step 20: loss=0.3189, reward=0.250, success=25.00%
Step 30: loss=0.2945, reward=0.312, success=31.25%
```

**Key Metrics**:
- **loss**: GRPO objective value (lower is better)
- **reward**: Average verification success (0-1 scale)
- **success**: Percentage of proofs that verified (0-100%)

### Log Files

Training logs are saved to `{output_dir}/train.log`:

```bash
tail -f experiments/grpo_run1/train.log
```

### Weights & Biases (W&B)

For advanced experiment tracking:

```bash
# Install W&B
pip install wandb
wandb login

# Train with W&B
python train_decomposer_grpo.py \
  --dataset_config configs/train_dataset.json \
  --output_dir experiments/grpo_run1 \
  --use_wandb \
  --wandb_project my_decomposer_project \
  --wandb_run_name first_run
```

W&B tracks:
- Loss curves
- Reward trends
- Success rates
- Learning rate schedule
- System metrics (GPU utilization, memory)

### Validation Metrics

Every `--eval_every` steps (default: 500), the script runs validation:

```
Running validation...
Validation - reward=0.345, success=34.50%
```

This shows performance on held-out data.

---

## Using Trained Models

### Loading a Checkpoint

After training, checkpoints are saved in `{output_dir}/`:

```
experiments/grpo_run1/
├── checkpoint_step_500/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files...
├── checkpoint_step_1000/
├── epoch_1/
├── epoch_2/
├── epoch_3/
└── final/
```

### Inference with Trained Model

Use the trained LoRA adapters for inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-Prover-V2-7B",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "experiments/grpo_run1/final",
)

tokenizer = AutoTokenizer.from_pretrained("experiments/grpo_run1/final")

# Generate sketch
prompt = build_decomposer_prompt(theorem_info)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048)
sketch = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Using with Existing Inference Pipeline

Modify `inference_single_model.py` to use the trained checkpoint:

```python
# Instead of:
predictor = SimpleLLMPredictor(args.model, ...)

# Use:
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(args.model, ...)
model = PeftModel.from_pretrained(base_model, "experiments/grpo_run1/final")
# Then use model for decomposer predictions
```

### Merging LoRA Weights (Optional)

To create a standalone model without LoRA adapters:

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Prover-V2-7B")
model = PeftModel.from_pretrained(base_model, "experiments/grpo_run1/final")

# Merge LoRA weights into base model
merged_model = model.merge_and_unload()

# Save as regular HuggingFace model
merged_model.save_pretrained("experiments/merged_decomposer")
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Error**: `torch.cuda.OutOfMemoryError: CUDA out of memory`

**Note**: The script uses a single-model architecture (~16-18GB), so OOM is unlikely with default settings on L4. If it happens:

**Solutions**:
```bash
# Reduce batch size
--batch_size 2

# Reduce LoRA rank
--lora_r 8

# Reduce max generation length
--max_tokens 1024

# Enable gradient checkpointing (should be on by default)
# Check that you're NOT using --no_gradient_checkpointing
```

**Common cause**: If you see OOM with default settings, check that you're not running other GPU processes simultaneously (e.g., Jupyter notebooks, other training runs).

#### 2. Lean Verification Timeout

**Error**: Verification hangs or times out

**Solutions**:
```bash
# Increase timeout
--lean_timeout 600

# Reduce verification batch size (sometimes helps)
--verify_batch_size 4
```

#### 3. Slow Training

**Issue**: Training is taking too long

**Solutions**:
```bash
# Increase verification batch size
--verify_batch_size 16

# Use smaller validation set
--validation_split 0.05

# Reduce evaluation frequency
--eval_every 1000
```

#### 4. NaN Loss

**Issue**: Loss becomes NaN during training

**Solutions**:
```bash
# Reduce learning rate
--learning_rate 1e-5

# Increase gradient clipping
--max_grad_norm 0.5

# Reduce KL coefficient
--kl_coef 0.01
```

#### 5. No Improvement

**Issue**: Success rate stays at 0% or very low

**Possible causes**:
- Dataset is too hard (theorems require complex proofs)
- Prover model is too weak
- Need more training time

**Solutions**:
```bash
# Train for more epochs
--num_epochs 5

# Adjust temperatures for more exploration
--decomposer_temperature 1.2

# Try a larger LoRA rank
--lora_r 32

# Check that your prover is actually completing proofs:
# Run inference_single_model.py first to see baseline performance
```

### Debugging Tips

#### 1. Test on Small Dataset First

```bash
python train_decomposer_grpo.py \
  --dataset_config configs/train_dataset.json \
  --max_examples_per_dataset 10 \
  --output_dir experiments/debug \
  --num_epochs 1
```

#### 2. Verify Lean is Working

```bash
cd RL/utils
python RL_utils_gpu_tests.py all
```

All tests should pass.

#### 3. Check Model Loading

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-Prover-V2-7B",
    torch_dtype=torch.float16,
    device_map="auto",
)
print(f"Model loaded: {model.num_parameters():,} parameters")
```

#### 4. Monitor GPU Usage

```bash
watch -n 1 nvidia-smi
```

Check that:
- GPU utilization is high (>80%)
- Memory usage is stable
- No memory leaks

---

## Advanced Configuration

### Hyperparameter Tuning

Key hyperparameters to tune:

#### 1. LoRA Configuration

**Higher rank = more capacity, more memory**:
```bash
# Small (faster, less memory)
--lora_r 8 --lora_alpha 16

# Medium (balanced)
--lora_r 16 --lora_alpha 32

# Large (more capacity)
--lora_r 64 --lora_alpha 128
```

#### 2. KL Coefficient

**Controls exploration vs. staying close to reference policy**:
```bash
# More exploration (riskier updates)
--kl_coef 0.01

# Balanced
--kl_coef 0.05

# More conservative (safer updates)
--kl_coef 0.1
```

#### 3. Learning Rate

**Depends on LoRA rank**:
```bash
# For low rank (r=8)
--learning_rate 1e-4

# For medium rank (r=16-32)
--learning_rate 5e-5

# For high rank (r=64)
--learning_rate 1e-5
```

#### 4. Temperature

**Controls diversity of generated sketches**:
```bash
# More deterministic (focus on best-known strategies)
--decomposer_temperature 0.7

# Balanced
--decomposer_temperature 1.0

# More exploratory (try diverse approaches)
--decomposer_temperature 1.3
```

### Custom LoRA Target Modules

By default, LoRA is applied to attention projection layers. To customize:

Edit the `TrainingConfig` class in the script:

```python
lora_target_modules: List[str] = field(
    default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",     # MLP (optional)
    ]
)
```

**Trade-off**: More modules = higher capacity but more memory.

### Reward Shaping

The default reward is binary (1 if proof verifies, 0 otherwise). You can modify the reward function in `generate_proofs_and_verify()`:

```python
# Current (binary)
result["reward"] = 1.0 if result.get("complete", False) else 0.0

# Alternative: partial credit for proofs with fewer errors
result["reward"] = 1.0 if result.get("complete", False) else max(0.0, 1.0 - len(result.get("errors", [])) * 0.1)

# Alternative: bonus for shorter proofs
if result.get("complete", False):
    proof_length = len(result["proof"].split())
    result["reward"] = 1.0 + max(0, 1.0 - proof_length / 1000)
else:
    result["reward"] = 0.0
```

### Different Baselines

The script uses a simple mean baseline. You can implement fancier baselines by modifying `compute_grpo_loss()`:

```python
# Current (group mean)
baseline = sum(rewards) / len(rewards) if rewards else 0.0

# Alternative: moving average baseline
if not hasattr(self, 'baseline_ema'):
    self.baseline_ema = 0.0
self.baseline_ema = 0.9 * self.baseline_ema + 0.1 * (sum(rewards) / len(rewards))
baseline = self.baseline_ema

# Alternative: learned value function (requires additional network)
# See PPO implementations for examples
```

### Learning Rate Schedules

The script uses linear warmup + decay. To use other schedules:

```python
# In setup_scheduler():
from transformers import get_cosine_schedule_with_warmup

self.scheduler = get_cosine_schedule_with_warmup(
    self.optimizer,
    num_warmup_steps=self.config.warmup_steps,
    num_training_steps=total_steps,
)
```

---

## Frequently Asked Questions

### Q: How long does training take?

**A**: For 1000 examples, ~1.5-2 hours per epoch on L4 GPU. Most time is spent on Lean verification (1-5 seconds per theorem).

### Q: Can I resume training from a checkpoint?

**A**: Currently, the script doesn't support automatic resumption. You can manually load a checkpoint and continue training by:

1. Loading the checkpoint as the initial model
2. Adjusting `--num_epochs` to account for already-completed epochs

### Q: How do I know if training is working?

**A**: Look for:
- Success rate increasing over time (e.g., 10% → 20% → 30%)
- Reward trending upward
- Loss decreasing (though less important than success rate)

Even small improvements (e.g., 15% → 18%) can be meaningful!

### Q: What's a good success rate to aim for?

**A**: This depends on your dataset difficulty:
- Easy theorems: 40-60%
- Medium theorems: 20-40%
- Hard theorems: 5-20%

Focus on relative improvement rather than absolute numbers.

### Q: Can I use a different base model?

**A**: Yes! Just change `--model` to any HuggingFace causal LM:

```bash
--model meta-llama/Llama-2-7b-hf
--model mistralai/Mistral-7B-v0.1
```

Make sure the model supports the tokenizer's chat template or adjust prompts accordingly.

### Q: How much does LoRA reduce memory?

**A**: Dramatically! For a 7B model with our single-model architecture:
- **Full fine-tuning (two models)**: ~56GB (2× model + 2× optimizer) ❌ Won't fit on L4!
- **LoRA with single model**: ~16-18GB (1× model + LoRA adapters + optimizer) ✅ Fits on L4!

**Key insight:** We use ONE model for both decomposer (with LoRA) and prover (frozen base model), not two separate models. This is the key to fitting everything in 24GB VRAM.

Breakdown of the 16-18GB:
- Base model (fp16): ~14GB
- LoRA adapters: ~0.5GB
- Optimizer states: ~2GB
- Activations/gradients: ~1-2GB (varies with batch size)

### Q: Should I use multiple samples per prompt?

**A**: Currently, `num_samples_per_prompt=1` is hard-coded for simplicity. Increasing this would:
- **Benefit**: More diverse training signal per theorem
- **Cost**: Proportionally longer training time

You can modify the code to sample multiple sketches per theorem if you have the compute budget.

### Q: Why does decomposer and prover use the same model?

**A**: This is a critical memory optimization! Here's why:

**The problem:** Your inference pipeline uses the same base model for both roles (just different prompts). If we loaded two separate copies during training, we'd need ~28GB just for the models, exceeding your L4's 24GB VRAM.

**The solution:** We use ONE model that plays both roles:
1. **Decomposer** (being trained): Uses the model with LoRA adapters active in train mode
2. **Prover** (frozen): Uses the same model in eval mode without gradients, bypassing LoRA

This is correct because:
- The prover is NOT being trained (it's the fixed policy π_φ from your paper)
- They're differentiated only by prompts, not weights
- It perfectly matches your inference pipeline's architecture
- Reduces memory from ~28GB → ~18GB (fits on L4!)

**No trade-offs:** This is the intended design, not a compromise!

---

## Example Training Runs

### Run 1: Small-Scale Test

```bash
python GRPO/train.py \
  --model deepseek-ai/DeepSeek-Prover-V2-7B \
  --dataset_config configs/lean_workbook.json \
  --max_examples_per_dataset 12 \
  --output_dir experiments/test_run \
  --batch_size 4 \
  --num_epochs 1 
```

**Purpose**: Quick test to verify everything works (~10 minutes)

### Run 2: Standard Training

```bash
python train_decomposer_grpo.py \
  --model deepseek-ai/DeepSeek-Prover-V2-7B \
  --dataset_config configs/train_dataset.json \
  --output_dir experiments/grpo_standard \
  --batch_size 4 \
  --num_epochs 3 \
  --learning_rate 5e-5 \
  --lora_r 16 \
  --use_wandb \
  --wandb_project decomposer-grpo \
  --wandb_run_name standard_run
```

**Purpose**: Full training run (~4-6 hours for 1000 examples)

### Run 3: High-Capacity Training

```bash
python train_decomposer_grpo.py \
  --model deepseek-ai/DeepSeek-Prover-V2-7B \
  --dataset_config configs/train_dataset.json \
  --output_dir experiments/grpo_large \
  --batch_size 2 \
  --num_epochs 5 \
  --learning_rate 3e-5 \
  --lora_r 64 \
  --lora_alpha 128 \
  --kl_coef 0.03 \
  --use_wandb
```

**Purpose**: Maximum performance (requires more memory and time)

---

## Next Steps

After training:

1. **Evaluate on test set**: Run `inference_single_model.py` with your trained checkpoint
2. **Compare to baseline**: Compare success rates before and after training
3. **Iterate**: Adjust hyperparameters based on results
4. **Scale up**: Train on larger datasets or for more epochs

Good luck with your GRPO training! 🚀
