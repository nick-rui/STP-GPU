"""GRPO Training for Decomposer Model with LoRA.

Trains a decomposer policy to generate better proof sketches using Group Relative
Policy Optimization (GRPO) with rewards from Lean4 verification.

Memory-efficient implementation using:
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Reference logprobs storage (no need for separate reference model)
- Gradient checkpointing
- Mixed precision training

Example Usage:
    cd STP-GPU/RL
    python GRPO/train.py \
    --model deepseek-ai/DeepSeek-Prover-V2-7B \
    --dataset_config dataset_configs/leanworkbook.json \
    --output_dir experiments/test_run \
    --max_tokens 1024 \
    --max_examples_per_dataset 10 \
    --batch_size 1 \
    --num_epochs 1
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# HuggingFace libraries
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from bitsandbytes.optim import PagedAdamW32bit

# Import your existing utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.RL_utils_gpu import (
    SimpleLean4Verifier,
    split_test_blocks,
    REPO_DIR,
)
from utils.gcloud_utils import read_file, write_data

# Optional: Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not available. Install with: pip install wandb")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for GRPO training."""
    
    # Model
    model_name: str = "deepseek-ai/DeepSeek-Prover-V2-7B"
    tokenizer_name: Optional[str] = None
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # GRPO hyperparameters (optimized for L4 GPU 24GB VRAM)
    kl_coef: float = 0.1  # β in Algorithm 1
    num_samples_per_prompt: int = 4  # samples per statement
    temperature: float = 1.0

    # Training
    batch_size: int = 1  # number of statements per update
    num_epochs: int = 3
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 8

    # Generation
    max_tokens: int = 1024
    decomposer_temperature: float = 1.0
    prover_temperature: float = 0.7
    
    # Verification
    verify_batch_size: int = 8
    lean_timeout: int = 300
    
    # Data
    dataset_config: str = "configs/train_dataset.json"
    max_examples_per_dataset: Optional[int] = None
    validation_split: float = 0.1
    
    # Logging & Checkpointing
    output_dir: str = "experiments/grpo_decomposer"
    log_interval: int = 10
    save_every: int = 500
    eval_every: int = 500
    use_wandb: bool = False
    wandb_project: str = "decomposer-grpo"
    wandb_run_name: Optional[str] = None
    
    # System
    seed: int = 42
    fp16: bool = True
    gradient_checkpointing: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Prompts (matching inference pipeline)
# ============================================================================

DECOMPOSER_PROMPT = """You are a Lean 4 proof assistant in DECOMPOSER mode.
Given a goal statement, break it into intermediate steps and emit Lean 4 code
where subgoals end with `sorry` placeholders. Focus on the high-level proof structure.

Here is an example of the desired decomposition style.

Example theorem statement:
```lean4
theorem amc12a_2019_p21 (z : ℂ) (h₀ : z = (1 + Complex.I) / Real.sqrt 2) :
  ((∑ k : ℤ in Finset.Icc 1 12, z ^ k ^ 2) * (∑ k : ℤ in Finset.Icc 1 12, 1 / z ^ k ^ 2)) = 36 := by
```

Example decomposed proof sketch:
```lean4
theorem amc12a_2019_p21 (z : ℂ) (h₀ : z = (1 + Complex.I) / Real.sqrt 2) :
  ((∑ k : ℤ in Finset.Icc 1 12, z ^ k ^ 2) * (∑ k : ℤ in Finset.Icc 1 12, 1 / z ^ k ^ 2)) = 36 := by sorry
  have h1 : True := by sorry
  have h2 : True := by sorry
  exact h1
```

Now, given a new goal, output a similar Lean 4 proof sketch with `sorry` placeholders.
Return only Lean 4 code with `sorry` for unresolved subproofs."""

PROVER_PROMPT = """You are a Lean 4 proof assistant in PROVER mode.
Given a Lean proof sketch containing `sorry` placeholders, replace each placeholder
with a valid tactic proof. Complete all sorry placeholders. 

Return only the completed Lean 4 code with no remaining `sorry`."""


def build_decomposer_prompt(test_info: Dict) -> str:
    """Build prompt for decomposer role."""
    header = test_info.get("header")
    prefix = f"{header}\n" if header else ""
    return (
        f"{DECOMPOSER_PROMPT}\n\n"
        f"```lean4\n{prefix}"
        f"-- Goal:\n{test_info['statement']}\n```\n\n"
        "Write a proof sketch in Lean 4 with `sorry` placeholders:"
    )


def build_prover_prompt(sketch: str, test_info: Dict) -> str:
    """Build prompt for prover role."""
    return (
        f"{PROVER_PROMPT}\n\n"
        f"Theorem statement:\n```lean4\n{test_info['statement']}\n```\n\n"
        f"Sketch to complete:\n```lean4\n{sketch}\n```\n\n"
        "Complete the proof (replace all `sorry`):"
    )


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences from completion."""
    if not text:
        return text
    lines = text.strip().splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


# ============================================================================
# Dataset
# ============================================================================

class TheoremDataset(Dataset):
    """Dataset of mathematical statements for decomposer training."""
    
    def __init__(self, lemmas: List[Dict]):
        self.lemmas = lemmas
    
    def __len__(self):
        return len(self.lemmas)
    
    def __getitem__(self, idx):
        return self.lemmas[idx]


def load_dataset(config_path: str, max_examples: Optional[int] = None) -> List[Dict]:
    """Load dataset from configuration file."""
    dataset_configs = read_file(config_path)
    if dataset_configs is None:
        raise ValueError(f"Failed to read dataset config from {config_path}")
    
    lemmas = []
    idx = 0
    for dataset_config in dataset_configs:
        raw_dataset = read_file(os.path.join(REPO_DIR, dataset_config["dataset_path"]))
        if raw_dataset is None:
            logging.warning(f"Failed to read {dataset_config['dataset_path']}, skipping")
            continue
        
        examples_to_use = raw_dataset[:max_examples] if max_examples else raw_dataset
        for raw in examples_to_use:
            statement = raw["formal_statement"].rsplit("sorry", 1)[0].strip()
            lemmas.append({
                "lemma_id": idx,
                "statement": statement,
                "label": [raw.get("split")] + (raw.get("tags") or []),
                "header": raw.get("header"),
                "dataset": dataset_config["dataset_path"],
            })
            idx += 1
    
    logging.info(f"Loaded {len(lemmas)} examples from {len(dataset_configs)} datasets")
    return lemmas


def split_train_val(lemmas: List[Dict], val_split: float = 0.1, seed: int = 42):
    """Split dataset into train and validation sets."""
    import random
    random.seed(seed)
    
    indices = list(range(len(lemmas)))
    random.shuffle(indices)
    
    val_size = int(len(lemmas) * val_split)
    val_indices = set(indices[:val_size])
    
    train_lemmas = [lemmas[i] for i in range(len(lemmas)) if i not in val_indices]
    val_lemmas = [lemmas[i] for i in range(len(lemmas)) if i in val_indices]
    
    return train_lemmas, val_lemmas


# ============================================================================
# Model Setup
# ============================================================================

def setup_model_and_tokenizer(config: TrainingConfig):
    """Initialize model with LoRA and tokenizer."""
    logging.info(f"Loading model: {config.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.model_name,
        trust_remote_code=True,
        padding_side="left"  # for generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Load base model with 4-bit quantization
    # Try to use Flash Attention 2 for memory efficiency (falls back if unavailable)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        logging.info("Using Flash Attention 2 for improved memory efficiency")
    except Exception as e:
        logging.warning(f"Flash Attention 2 not available ({e}), using default attention")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    # Prepare model for k-bit training (required for quantized models with LoRA)
    model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


# ============================================================================
# GRPO Training Logic
# ============================================================================

class GRPOTrainer:
    """GRPO trainer for decomposer policy."""
    
    def __init__(
        self,
        model,
        tokenizer,
        verifier: SimpleLean4Verifier,
        config: TrainingConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.config = config
        
        # Optimizer (paged for memory efficiency)
        self.optimizer = PagedAdamW32bit(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Learning rate scheduler
        self.scheduler = None  # will be set after knowing total steps
        
        # Logging
        self.global_step = 0
        self.epoch = 0
        self.accumulation_counter = 0
        
    def setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler."""
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )
    
    def generate_sketches_with_logprobs(self, batch_lemmas: List[Dict]) -> List[Dict]:
        """
        Generate sketch proofs and compute their log probabilities.

        Generates num_samples_per_prompt sketches for each lemma.

        Returns list of dicts with keys: 'sketch', 'ref_logprobs', 'prompt', 'lemma'
        """
        # Repeat each prompt num_samples_per_prompt times for group sampling
        prompts = []
        repeated_lemmas = []
        for lemma in batch_lemmas:
            prompt = build_decomposer_prompt(lemma)
            for _ in range(self.config.num_samples_per_prompt):
                prompts.append(prompt)
                repeated_lemmas.append(lemma)
        
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.model.device)
        
        # Generate with logprobs
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.decomposer_temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
        
        # Compute log probabilities for generated tokens
        # outputs.scores is tuple of (batch_size, vocab_size) tensors, one per generated token
        ref_logprobs_list = []
        for i in range(len(repeated_lemmas)):
            token_logprobs = []
            for t, score_tensor in enumerate(outputs.scores):
                if t < generated_ids.shape[1]:
                    token_id = generated_ids[i, t].item()
                    logprobs = F.log_softmax(score_tensor[i], dim=-1)
                    token_logprob = logprobs[token_id].item()
                    token_logprobs.append(token_logprob)
            ref_logprobs_list.append(sum(token_logprobs))  # sum log probs for sequence

        # Decode sketches
        sketches = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        sketches = [strip_code_fences(s) for s in sketches]

        results = []
        for lemma, sketch, ref_logprob, prompt in zip(
            repeated_lemmas, sketches, ref_logprobs_list, prompts
        ):
            results.append({
                "lemma": lemma,
                "sketch": sketch,
                "ref_logprobs": ref_logprob,
                "prompt": prompt,
            })
        
        return results
    
    def compute_current_logprobs(self, batch_data: List[Dict]) -> List[float]:
        """
        Compute log probabilities of sketches under current policy.
        
        This is π_θ(s|x) for computing importance sampling ratio.
        """
        logprobs_list = []
        
        for data in batch_data:
            prompt = data["prompt"]
            sketch = data["sketch"]
            
            # Tokenize prompt and completion
            full_text = prompt + sketch
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(self.model.device)
            
            prompt_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(self.model.device)
            
            prompt_length = prompt_inputs.input_ids.shape[1]
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Compute log probs for generated tokens only
            generated_tokens = inputs.input_ids[0, prompt_length:]
            logprobs_tokens = []
            
            for t in range(len(generated_tokens)):
                token_id = generated_tokens[t].item()
                logits_at_t = logits[0, prompt_length + t - 1, :]  # logits before token t
                log_probs = F.log_softmax(logits_at_t, dim=-1)
                logprobs_tokens.append(log_probs[token_id].item())
            
            logprobs_list.append(sum(logprobs_tokens))
        
        return logprobs_list
    
    def generate_proofs_and_verify(self, batch_data: List[Dict]) -> List[Dict]:
        """
        Use prover to complete sketches and verify with Lean.
        
        Uses same model as decomposer, just with different prompt and no gradients.
        
        Returns list of dicts with reward and verification info.
        """
        # Generate prover completions using same model (in eval mode, no gradients)
        prover_prompts = [
            build_prover_prompt(data["sketch"], data["lemma"])
            for data in batch_data
        ]
        
        self.model.eval()
        proofs = []
        
        with torch.no_grad():
            for prompt in prover_prompts:
                # Tokenize prover prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096,
                ).to(self.model.device)
                
                # Generate proof completion
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.prover_temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
                # Decode and clean
                proof_text = self.tokenizer.decode(
                    outputs[0, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                proofs.append(strip_code_fences(proof_text))
        
        # Prepare for verification
        proof_infos = []
        for data, proof in zip(batch_data, proofs):
            lemma = data["lemma"]
            proof_info = lemma.copy()
            proof_info["proof"] = proof
            proof_info["code"] = proof
            proof_info["sketch"] = data["sketch"]
            proof_infos.append(proof_info)
        
        # Verify with Lean
        try:
            verified_results = self.verifier.run(
                proof_infos,
                batched=len(proof_infos) > 1
            )
        except Exception as e:
            logging.error(f"Lean verification failed: {e}")
            # Fallback: mark all as failed
            verified_results = [
                {**pi, "complete": False, "pass": False, "errors": [str(e)]}
                for pi in proof_infos
            ]
        
        # Extract rewards with partial reward for "sorry"
        for result in verified_results:
            result["reward"] = 0.0
            if result.get("complete", False):
                # Proof verifies successfully
                result["reward"] += 0.8
            if "sorry" in result.get("sketch", ""):
                # Partial reward if "sorry" is in the sketch
                result["reward"] += 0.2
        
        return verified_results
    
    def compute_grpo_loss(
        self,
        batch_data: List[Dict],
        rewards: List[float],
        current_logprobs: List[float],
    ) -> torch.Tensor:
        """
        Compute GRPO objective (Algorithm 1, line 12).

        L(θ) = (1/B) Σ w^(i) * Â^(i) - β * KL(π_θ || π_θ_old)

        where:
        - w^(i) = π_θ(s|x) / π_θ_old(s|x)  [importance sampling ratio]
        - Â^(i) = (r^(i) - mean(r_group)) / std(r_group)  [group-relative advantage]
        """
        # Reshape rewards into groups [batch_size, num_samples_per_prompt]
        group_size = self.config.num_samples_per_prompt
        batch_size = len(rewards) // group_size

        if len(rewards) % group_size != 0:
            raise ValueError(
                f"Rewards length {len(rewards)} not divisible by group_size {group_size}"
            )

        # Compute group-relative advantages
        advantages = []
        for i in range(batch_size):
            # Get rewards for this group
            group_rewards = rewards[i * group_size:(i + 1) * group_size]

            # Compute group statistics
            group_mean = sum(group_rewards) / len(group_rewards)
            group_var = sum((r - group_mean) ** 2 for r in group_rewards) / len(group_rewards)
            group_std = (group_var ** 0.5) if group_var > 1e-8 else 1.0

            # Normalize advantages within group
            for r in group_rewards:
                advantages.append((r - group_mean) / group_std)
        
        # Compute importance sampling ratios in log space
        # log(w) = log(π_θ) - log(π_θ_old)
        ref_logprobs = [data["ref_logprobs"] for data in batch_data]
        log_ratios = [curr - ref for curr, ref in zip(current_logprobs, ref_logprobs)]
        
        # Convert to probability ratios
        ratios = [torch.exp(torch.tensor(lr)) for lr in log_ratios]
        
        # Policy gradient term: Σ w^(i) * Â^(i)
        pg_loss = sum(
            ratio * adv
            for ratio, adv in zip(ratios, advantages)
        ) / len(batch_data)
        
        # KL divergence term (in expectation, approximated by sample mean)
        kl_div = sum(log_ratios) / len(log_ratios)
        
        # GRPO objective (we want to maximize, so negate for minimization)
        loss = -(pg_loss - self.config.kl_coef * kl_div)

        # Compute baseline as mean of all rewards (for logging)
        baseline = sum(rewards) / len(rewards) if rewards else 0.0

        return loss, {
            "pg_loss": pg_loss.item() if torch.is_tensor(pg_loss) else pg_loss,
            "kl_div": kl_div,
            "baseline": baseline,
            "mean_advantage": sum(advantages) / len(advantages),
            "mean_ratio": sum(r.item() for r in ratios) / len(ratios),
        }
    
    def training_step(self, batch_lemmas: List[Dict]) -> Dict:
        """
        Execute one GRPO training step (Algorithm 1, lines 5-13).
        
        Returns metrics dict.
        """
        self.model.train()
        
        # Step 1: Generate sketches with reference logprobs
        batch_data = self.generate_sketches_with_logprobs(batch_lemmas)
        
        # Step 2: Generate proofs and verify
        verified_results = self.generate_proofs_and_verify(batch_data)
        rewards = [r["reward"] for r in verified_results]
        
        # Step 3: Compute current logprobs (for importance sampling)
        # This requires gradient computation
        self.model.eval()  # use eval mode for stable logprobs
        current_logprobs = []
        
        for data in batch_data:
            prompt = data["prompt"]
            sketch = data["sketch"]
            
            # Tokenize
            full_text = prompt + sketch
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(self.model.device)
            
            prompt_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(self.model.device)
            
            prompt_length = prompt_inputs.input_ids.shape[1]
            
            # Forward pass (with gradients this time)
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Compute log probs for generated tokens
            generated_tokens = inputs.input_ids[0, prompt_length:]
            logprobs_tokens = []
            
            for t in range(len(generated_tokens)):
                token_id = generated_tokens[t].item()
                logits_at_t = logits[0, prompt_length + t - 1, :]
                log_probs = F.log_softmax(logits_at_t, dim=-1)
                logprobs_tokens.append(log_probs[token_id])
            
            # Sum log probs (sequence-level)
            seq_logprob = sum(logprobs_tokens)
            current_logprobs.append(seq_logprob)
        
        # Step 4: Compute GRPO loss
        loss, loss_info = self.compute_grpo_loss(batch_data, rewards, current_logprobs)

        # Step 5: Backprop and update with gradient accumulation
        # Scale loss by accumulation steps
        loss = loss / self.config.gradient_accumulation_steps

        # Zero gradients only at the start of accumulation
        if self.accumulation_counter == 0:
            self.optimizer.zero_grad()

        loss.backward()

        self.accumulation_counter += 1

        # Only step optimizer every gradient_accumulation_steps
        if self.accumulation_counter >= self.config.gradient_accumulation_steps:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            self.accumulation_counter = 0
            self.global_step += 1

            # Periodic CUDA cache clearing to prevent OOM
            if self.global_step % 10 == 0:
                torch.cuda.empty_cache()
        
        # Metrics
        metrics = {
            "loss": loss.item() * self.config.gradient_accumulation_steps,  # Unscale for logging
            "mean_reward": sum(rewards) / len(rewards),
            "success_rate": sum(r > 0 for r in rewards) / len(rewards),
            "partial_reward_rate": sum(1 for r in rewards if r == 0.5) / len(rewards),
            "full_reward_rate": sum(1 for r in rewards if r == 1.0) / len(rewards),
            "lr": self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate,
            **loss_info,
        }
        
        return metrics
    
    def evaluate(self, val_dataloader: DataLoader) -> Dict:
        """Evaluate on validation set."""
        self.model.eval()
        
        all_rewards = []
        all_success = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating"):
                batch_lemmas = batch
                
                # Generate and verify
                batch_data = self.generate_sketches_with_logprobs(batch_lemmas)
                verified_results = self.generate_proofs_and_verify(batch_data)
                
                rewards = [r["reward"] for r in verified_results]
                all_rewards.extend(rewards)
                all_success.extend([r > 0 for r in rewards])
        
        metrics = {
            "val_mean_reward": sum(all_rewards) / len(all_rewards),
            "val_success_rate": sum(all_success) / len(all_success),
        }
        
        return metrics
    
    def save_checkpoint(self, save_dir: str, prefix: str = "checkpoint"):
        """Save model checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_dir = os.path.join(save_dir, f"{prefix}_step_{self.global_step}")
        
        # Save LoRA adapters
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "optimizer_state": self.optimizer.state_dict(),
        }
        if self.scheduler:
            state["scheduler_state"] = self.scheduler.state_dict()
        
        torch.save(state, os.path.join(checkpoint_dir, "training_state.pt"))
        
        logging.info(f"Checkpoint saved to {checkpoint_dir}")


# ============================================================================
# Training Loop
# ============================================================================

def train(config: TrainingConfig):
    """Main training function."""
    
    # Setup logging
    os.makedirs(config.output_dir, exist_ok=True)
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(config.output_dir, "train.log")),
            logging.StreamHandler(),
        ],
    )
    
    # Set random seed
    torch.manual_seed(config.seed)
    
    # Initialize W&B
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
        )
    
    # Load dataset
    logging.info("Loading dataset...")
    all_lemmas = load_dataset(config.dataset_config, config.max_examples_per_dataset)
    train_lemmas, val_lemmas = split_train_val(all_lemmas, config.validation_split, config.seed)
    
    logging.info(f"Train: {len(train_lemmas)}, Val: {len(val_lemmas)}")
    
    # Create dataloaders
    train_dataset = TheoremDataset(train_lemmas)
    val_dataset = TheoremDataset(val_lemmas)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda x: x,  # return list of dicts
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
    )
    
    # Setup model
    logging.info("Setting up model...")
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Setup verifier
    logging.info("Setting up Lean verifier...")
    verifier = SimpleLean4Verifier(
        collect_premises=False,
        timeout=config.lean_timeout,
    )
    
    # Create trainer (uses same model for both decomposer and prover)
    trainer = GRPOTrainer(model, tokenizer, verifier, config)
    
    # Setup scheduler
    total_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
    trainer.setup_scheduler(total_steps)
    
    logging.info(f"Total training steps: {total_steps}")
    
    # Training loop
    logging.info("Starting training...")
    global_step = 0
    
    for epoch in range(config.num_epochs):
        trainer.epoch = epoch
        epoch_metrics = []
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Training step
            metrics = trainer.training_step(batch)
            epoch_metrics.append(metrics)
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "reward": f"{metrics['mean_reward']:.3f}",
                "success": f"{metrics['success_rate']:.2%}",
            })
            
            # Log to W&B
            if config.use_wandb and WANDB_AVAILABLE:
                wandb.log(metrics, step=trainer.global_step)
            
            # Periodic logging
            if trainer.global_step % config.log_interval == 0:
                avg_metrics = {
                    k: sum(m[k] for m in epoch_metrics[-config.log_interval:]) / min(len(epoch_metrics), config.log_interval)
                    for k in epoch_metrics[0].keys()
                }

                # GPU memory monitoring
                memory_info = ""
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    memory_info = f", GPU: {allocated:.2f}GB/{reserved:.2f}GB"

                logging.info(
                    f"Step {trainer.global_step}: "
                    f"loss={avg_metrics['loss']:.4f}, "
                    f"reward={avg_metrics['mean_reward']:.3f}, "
                    f"success={avg_metrics['success_rate']:.2%}"
                    f"{memory_info}"
                )
            
            # Periodic evaluation
            if trainer.global_step % config.eval_every == 0 and val_lemmas:
                logging.info("Running validation...")
                val_metrics = trainer.evaluate(val_dataloader)
                logging.info(
                    f"Validation - "
                    f"reward={val_metrics['val_mean_reward']:.3f}, "
                    f"success={val_metrics['val_success_rate']:.2%}"
                )
                
                if config.use_wandb and WANDB_AVAILABLE:
                    wandb.log(val_metrics, step=trainer.global_step)
            
            # Periodic checkpointing
            if trainer.global_step % config.save_every == 0:
                trainer.save_checkpoint(config.output_dir, prefix="checkpoint")
        
        # End of epoch
        epoch_avg_metrics = {
            k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics)
            for k in epoch_metrics[0].keys()
        }
        logging.info(
            f"Epoch {epoch+1} complete - "
            f"avg_loss={epoch_avg_metrics['loss']:.4f}, "
            f"avg_reward={epoch_avg_metrics['mean_reward']:.3f}, "
            f"avg_success={epoch_avg_metrics['success_rate']:.2%}"
        )
        
        # Save epoch checkpoint
        trainer.save_checkpoint(config.output_dir, prefix=f"epoch_{epoch+1}")
    
    # Final evaluation
    if val_lemmas:
        logging.info("Running final validation...")
        val_metrics = trainer.evaluate(val_dataloader)
        logging.info(
            f"Final validation - "
            f"reward={val_metrics['val_mean_reward']:.3f}, "
            f"success={val_metrics['val_success_rate']:.2%}"
        )
    
    # Save final model
    trainer.save_checkpoint(config.output_dir, prefix="final")
    
    logging.info("Training complete!")
    
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training for decomposer model")
    
    # Model
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-Prover-V2-7B")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # GRPO (L4-optimized defaults)
    parser.add_argument("--kl_coef", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    
    # Training (L4-optimized defaults)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Generation
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--decomposer_temperature", type=float, default=1.0)
    parser.add_argument("--prover_temperature", type=float, default=0.7)
    
    # Verification
    parser.add_argument("--verify_batch_size", type=int, default=8)
    parser.add_argument("--lean_timeout", type=int, default=300)
    
    # Data
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--max_examples_per_dataset", type=int, default=None)
    parser.add_argument("--validation_split", type=float, default=0.1)
    
    # Logging
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="decomposer-grpo")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    
    args = parser.parse_args()
    
    # Convert to config object
    config = TrainingConfig(
        model_name=args.model,
        tokenizer_name=args.tokenizer_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        kl_coef=args.kl_coef,
        temperature=args.temperature,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        max_tokens=args.max_tokens,
        decomposer_temperature=args.decomposer_temperature,
        prover_temperature=args.prover_temperature,
        verify_batch_size=args.verify_batch_size,
        lean_timeout=args.lean_timeout,
        dataset_config=args.dataset_config,
        max_examples_per_dataset=args.max_examples_per_dataset,
        validation_split=args.validation_split,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
        save_every=args.save_every,
        eval_every=args.eval_every,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
        fp16=not args.no_fp16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    
    return config


if __name__ == "__main__":
    config = parse_args()
    train(config)