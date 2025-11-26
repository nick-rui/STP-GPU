"""REINFORCE training for decomposer model with LoRA.

Trains a decomposer policy to generate better proof sketches using
REINFORCE with a running baseline and rewards from Lean4 verification.

Key design points:
- 1 sample per lemma (no GRPO grouping)
- LoRA on top of 4-bit quantized 7B model
- Gradient checkpointing and mixed precision for memory efficiency
- Same reward shaping as GRPO trainer (decomposer-focused)

Example usage:
    cd STP-GPU/RL
    python GRPO/train_reinforce.py \\
        --model deepseek-ai/DeepSeek-Prover-V2-7B \\
        --dataset_config dataset_configs/leanworkbook.json \\
        --output_dir experiments/reinforce_run \\
        --max_tokens 512 \\
        --max_examples_per_dataset 50 \\
        --batch_size 1 \\
        --num_epochs 1
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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

# Import existing utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.RL_utils_gpu import (  # type: ignore
    SimpleLean4Verifier,
    REPO_DIR,
)
from utils.gcloud_utils import read_file  # type: ignore


@dataclass
class TrainingConfig:
    """Configuration for REINFORCE decomposer training."""

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

    # REINFORCE hyperparameters
    baseline_momentum: float = 0.1  # running baseline update rate

    # Training
    batch_size: int = 1
    num_epochs: int = 3
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 8

    # Generation
    max_tokens: int = 512
    decomposer_temperature: float = 1.0
    prover_temperature: float = 0.7

    # Verification
    lean_timeout: int = 300

    # Data
    dataset_config: str = "configs/train_dataset.json"
    max_examples_per_dataset: Optional[int] = None
    validation_split: float = 0.1

    # Logging & Checkpointing
    output_dir: str = "experiments/reinforce_decomposer"
    log_interval: int = 10
    save_every: int = 500
    eval_every: int = 500

    # System
    seed: int = 42
    fp16: bool = True
    gradient_checkpointing: bool = True
    use_4bit: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# Prompts (reused from GRPO)
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

    lemmas: List[Dict] = []
    idx = 0
    for dataset_config in dataset_configs:
        raw_dataset = read_file(os.path.join(REPO_DIR, dataset_config["dataset_path"]))
        if raw_dataset is None:
            logging.warning(f"Failed to read {dataset_config['dataset_path']}, skipping")
            continue

        examples_to_use = raw_dataset[:max_examples] if max_examples else raw_dataset
        for raw in examples_to_use:
            statement = raw["formal_statement"].rsplit("sorry", 1)[0].strip()
            lemmas.append(
                {
                    "lemma_id": idx,
                    "statement": statement,
                    "label": [raw.get("split")] + (raw.get("tags") or []),
                    "header": raw.get("header"),
                    "dataset": dataset_config["dataset_path"],
                }
            )
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


def setup_model_and_tokenizer(config: TrainingConfig):
    """Initialize model with LoRA and tokenizer."""
    logging.info(f"Loading model: {config.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
            logging.info("Using 4-bit quantization with Flash Attention 2")
        except Exception as e:  # pragma: no cover - defensive fallback
            logging.warning(
                f"Flash Attention 2 not available ({e}), using default attention with 4-bit"
            )
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

        model = prepare_model_for_kbit_training(model)
    else:
        # Full-precision LoRA (fp16/bf16) without 4-bit quantization
        torch_dtype = torch.float16 if config.fp16 else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
            logging.info("Using full-precision model with Flash Attention 2")
        except Exception as e:  # pragma: no cover - defensive fallback
            logging.warning(
                f"Flash Attention 2 not available ({e}), using default attention (no 4-bit)"
            )
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


class ReinforceTrainer:
    """REINFORCE trainer for decomposer policy (1 sample per lemma)."""

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

        self.optimizer = PagedAdamW32bit(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        self.scheduler = None  # set after total steps is known

        self.global_step = 0
        self.epoch = 0
        self.accumulation_counter = 0

        self.baseline: Optional[float] = None

    def setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler."""
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

    def generate_sketches(self, batch_lemmas: List[Dict]) -> List[Dict]:
        """Generate one sketch per lemma using the decomposer prompt."""
        prompts = [build_decomposer_prompt(lemma) for lemma in batch_lemmas]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.decomposer_temperature,
                do_sample=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_ids = outputs.sequences[:, inputs.input_ids.shape[1] :]
        sketches = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        sketches = [strip_code_fences(s) for s in sketches]

        results: List[Dict] = []
        for lemma, sketch, prompt in zip(batch_lemmas, sketches, prompts):
            results.append(
                {
                    "lemma": lemma,
                    "sketch": sketch,
                    "prompt": prompt,
                }
            )
        return results

    def generate_proofs_and_verify(self, batch_data: List[Dict]) -> List[Dict]:
        """
        Use prover to complete sketches and verify with Lean.

        Reuses the same model in prover mode (no gradients).
        """
        prover_prompts = [
            build_prover_prompt(data["sketch"], data["lemma"]) for data in batch_data
        ]

        self.model.eval()
        proofs: List[str] = []

        with torch.no_grad():
            for prompt in prover_prompts:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096,
                ).to(self.model.device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.prover_temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                proof_text = self.tokenizer.decode(
                    outputs[0, inputs.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )
                proofs.append(strip_code_fences(proof_text))

        proof_infos: List[Dict] = []
        for data, proof in zip(batch_data, proofs):
            lemma = data["lemma"]
            proof_info = lemma.copy()
            proof_info["proof"] = proof
            proof_info["code"] = proof
            proof_info["sketch"] = data["sketch"]
            proof_infos.append(proof_info)

        try:
            verified_results = self.verifier.run(
                proof_infos,
                batched=len(proof_infos) > 1,
            )
        except Exception as e:  # pragma: no cover - defensive path
            logging.error(f"Lean verification failed: {e}")
            verified_results = [
                {**pi, "complete": False, "pass": False, "errors": [str(e)]}
                for pi in proof_infos
            ]

        for result in verified_results:
            result["reward"] = self.compute_reward(result)

        return verified_results

    def compute_reward(self, result: Dict) -> float:
        """
        Decomposer-focused reward (same shaping as GRPO trainer).

        Combines:
        - verification success (does the final proof typecheck?)
        - sketch structure (does the sketch meaningfully decompose the proof?)
        - mild length regularization (avoid extremely long, noisy sketches)
        """
        sketch = result.get("sketch", "") or ""
        sketch_lower = sketch.lower()
        complete = bool(result.get("complete", False))

        # 1) Verification component
        verify_reward = 0.5 if complete else 0.0

        # 2) Structure component
        num_sorries = sketch_lower.count("sorry")
        structure_keywords = [
            " have ",
            "calc",
            "by_cases",
            "cases ",
            "by_contra",
            "refine ",
            "obtain ",
            "intro",
            "intros",
            "rw ",
            "simp",
            "rfl",
            "have h",
            "have :",
        ]
        keyword_hits = sum(sketch_lower.count(kw) for kw in structure_keywords)

        max_sorries = 5
        max_keywords = 10
        sorries_score = min(num_sorries, max_sorries) / max_sorries
        keywords_score = min(keyword_hits, max_keywords) / max_keywords

        if num_sorries == 0 and keyword_hits == 0:
            structure_score = 0.0
        else:
            structure_score = 0.5 * sorries_score + 0.5 * keywords_score

        if num_sorries <= 1 and keyword_hits == 0:
            structure_score *= 0.2

        structure_reward = 0.4 * structure_score

        # 3) Length regularization
        token_len = len(sketch.split())
        length_penalty = 0.0
        max_len_without_penalty = 512
        if token_len > max_len_without_penalty:
            length_penalty = min(0.2, 0.0005 * (token_len - max_len_without_penalty))

        reward = verify_reward + structure_reward - length_penalty
        reward = max(0.0, min(1.0, reward))
        return reward

    def _sequence_logprob(self, prompt: str, sketch: str) -> torch.Tensor:
        """
        Compute log π_θ(sketch | prompt) as a scalar tensor.
        """
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

        outputs = self.model(**inputs)
        logits = outputs.logits

        generated_tokens = inputs.input_ids[0, prompt_length:]
        logprobs_tokens: List[torch.Tensor] = []

        for t in range(len(generated_tokens)):
            token_id = generated_tokens[t].item()
            logits_at_t = logits[0, prompt_length + t - 1, :]
            log_probs = F.log_softmax(logits_at_t, dim=-1)
            logprobs_tokens.append(log_probs[token_id])

        if not logprobs_tokens:
            return torch.tensor(0.0, device=self.model.device)

        return torch.stack(logprobs_tokens).sum()

    def training_step(self, batch_lemmas: List[Dict]) -> Dict:
        """
        Execute one REINFORCE training step for a batch of lemmas.
        """
        self.model.train()

        # 1) Generate sketches
        batch_data = self.generate_sketches(batch_lemmas)

        # 2) Generate proofs and verify to obtain rewards
        verified_results = self.generate_proofs_and_verify(batch_data)
        rewards = [r["reward"] for r in verified_results]

        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

        # 3) Update running baseline
        if self.baseline is None:
            self.baseline = mean_reward
        else:
            m = self.config.baseline_momentum
            self.baseline = (1.0 - m) * self.baseline + m * mean_reward

        # 4) REINFORCE loss: L = - (r - b) * log π_θ(s|x)
        if self.accumulation_counter == 0:
            self.optimizer.zero_grad()

        total_loss = 0.0

        for data, reward in zip(batch_data, rewards):
            prompt = data["prompt"]
            sketch = data["sketch"]

            seq_logprob = self._sequence_logprob(prompt, sketch)
            advantage = reward - (self.baseline or 0.0)

            per_sample_loss = -(advantage * seq_logprob)
            total_loss = total_loss + per_sample_loss.detach().item()

            per_sample_loss = per_sample_loss / self.config.gradient_accumulation_steps
            per_sample_loss.backward()

        self.accumulation_counter += 1

        if self.accumulation_counter >= self.config.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            self.accumulation_counter = 0
            self.global_step += 1

            if torch.cuda.is_available() and self.global_step % 10 == 0:
                torch.cuda.empty_cache()

        metrics = {
            "loss": float(total_loss),
            "mean_reward": mean_reward,
            "success_rate": sum(r > 0 for r in rewards) / len(rewards)
            if rewards
            else 0.0,
            "lr": self.scheduler.get_last_lr()[0]
            if self.scheduler
            else self.config.learning_rate,
            "baseline": float(self.baseline or 0.0),
        }

        return metrics

    def evaluate(self, val_dataloader: DataLoader) -> Dict:
        """Evaluate on validation set."""
        self.model.eval()

        all_rewards: List[float] = []
        all_success: List[bool] = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating", leave=False):
                batch_lemmas = batch

                batch_data = self.generate_sketches(batch_lemmas)
                verified_results = self.generate_proofs_and_verify(batch_data)

                rewards = [r["reward"] for r in verified_results]
                all_rewards.extend(rewards)
                all_success.extend([r > 0 for r in rewards])

        if not all_rewards:
            return {"val_mean_reward": 0.0, "val_success_rate": 0.0}

        return {
            "val_mean_reward": sum(all_rewards) / len(all_rewards),
            "val_success_rate": sum(all_success) / len(all_success),
        }

    def save_checkpoint(self, save_dir: str, prefix: str = "checkpoint"):
        """Save model checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_dir = os.path.join(save_dir, f"{prefix}_step_{self.global_step}")

        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "optimizer_state": self.optimizer.state_dict(),
            "baseline": self.baseline,
        }
        if self.scheduler:
            state["scheduler_state"] = self.scheduler.state_dict()

        torch.save(state, os.path.join(checkpoint_dir, "training_state.pt"))
        logging.info(f"Checkpoint saved to {checkpoint_dir}")


def train(config: TrainingConfig):
    """Main training function."""
    os.makedirs(config.output_dir, exist_ok=True)
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(config.output_dir, "train.log")),
            logging.StreamHandler(),
        ],
    )

    torch.manual_seed(config.seed)

    logging.info("Loading dataset...")
    all_lemmas = load_dataset(config.dataset_config, config.max_examples_per_dataset)
    train_lemmas, val_lemmas = split_train_val(
        all_lemmas, config.validation_split, config.seed
    )
    logging.info(f"Train: {len(train_lemmas)}, Val: {len(val_lemmas)}")

    train_dataset = TheoremDataset(train_lemmas)
    val_dataset = TheoremDataset(val_lemmas)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
    )

    logging.info("Setting up model...")
    model, tokenizer = setup_model_and_tokenizer(config)

    logging.info("Setting up Lean verifier...")
    verifier = SimpleLean4Verifier(
        collect_premises=False,
        timeout=config.lean_timeout,
    )

    trainer = ReinforceTrainer(model, tokenizer, verifier, config)

    total_steps = (
        len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
    )
    trainer.setup_scheduler(total_steps)
    logging.info(f"Total training steps (optimizer updates): {total_steps}")

    logging.info("Starting training...")

    for epoch in range(config.num_epochs):
        trainer.epoch = epoch
        epoch_metrics: List[Dict] = []

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
        )

        for batch_idx, batch in enumerate(progress_bar):
            metrics = trainer.training_step(batch)
            epoch_metrics.append(metrics)

            progress_bar.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "reward": f"{metrics['mean_reward']:.3f}",
                    "success": f"{metrics['success_rate']:.2%}",
                }
            )

            if (
                trainer.global_step > 0
                and trainer.global_step % config.log_interval == 0
            ):
                window = epoch_metrics[-config.log_interval :]
                avg_metrics = {
                    k: sum(m[k] for m in window) / len(window) for k in window[0].keys()
                }

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

            if (
                trainer.global_step > 0
                and trainer.global_step % config.eval_every == 0
                and val_lemmas
            ):
                logging.info("Running validation...")
                val_metrics = trainer.evaluate(val_dataloader)
                logging.info(
                    "Validation - "
                    f"reward={val_metrics['val_mean_reward']:.3f}, "
                    f"success={val_metrics['val_success_rate']:.2%}"
                )

            if trainer.global_step > 0 and trainer.global_step % config.save_every == 0:
                trainer.save_checkpoint(config.output_dir, prefix="checkpoint")

        if epoch_metrics:
            epoch_avg_metrics = {
                k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics)
                for k in epoch_metrics[0].keys()
            }
            logging.info(
                f"Epoch {epoch + 1} complete - "
                f"avg_loss={epoch_avg_metrics['loss']:.4f}, "
                f"avg_reward={epoch_avg_metrics['mean_reward']:.3f}, "
                f"avg_success={epoch_avg_metrics['success_rate']:.2%}"
            )

        trainer.save_checkpoint(config.output_dir, prefix=f"epoch_{epoch + 1}")

    if val_lemmas:
        logging.info("Running final validation...")
        val_metrics = trainer.evaluate(val_dataloader)
        logging.info(
            "Final validation - "
            f"reward={val_metrics['val_mean_reward']:.3f}, "
            f"success={val_metrics['val_success_rate']:.2%}"
        )

    trainer.save_checkpoint(config.output_dir, prefix="final")
    logging.info("Training complete!")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="REINFORCE training for decomposer model"
    )

    # Model
    parser.add_argument(
        "--model", type=str, default="deepseek-ai/DeepSeek-Prover-V2-7B"
    )
    parser.add_argument("--tokenizer_name", type=str, default=None)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # REINFORCE
    parser.add_argument("--baseline_momentum", type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Generation
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--decomposer_temperature", type=float, default=1.0)
    parser.add_argument("--prover_temperature", type=float, default=0.7)

    # Verification
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

    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--no_quantization",
        action="store_true",
        help="Disable 4-bit quantization (use full-precision weights instead)",
    )

    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model,
        tokenizer_name=args.tokenizer_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        baseline_momentum=args.baseline_momentum,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        max_tokens=args.max_tokens,
        decomposer_temperature=args.decomposer_temperature,
        prover_temperature=args.prover_temperature,
        lean_timeout=args.lean_timeout,
        dataset_config=args.dataset_config,
        max_examples_per_dataset=args.max_examples_per_dataset,
        validation_split=args.validation_split,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
        save_every=args.save_every,
        eval_every=args.eval_every,
        seed=args.seed,
        fp16=not args.no_fp16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        use_4bit=not args.no_quantization,
    )

    return config


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)

