#!/usr/bin/env python3
"""
QLoRA SFT training script for DeepSeek-Prover-V2-7B.

Trains the model to:
- Generate sketch proofs when prompted with <sketch>...</sketch>
- Generate full proofs when prompted with <proof>...</proof>

Optimized for NVIDIA L4 (24GB VRAM) using:
- 4-bit quantization (QLoRA)
- Gradient checkpointing
- Small batch size with gradient accumulation

Usage:
    python train_sft_qlora.py \
        --data sft_train_data.jsonl \
        --output ./checkpoints/sketch-prover \
        --epochs 3

Requirements:
    pip install torch transformers peft bitsandbytes accelerate datasets trl
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig


def load_training_data(data_path: str) -> Dataset:
    """Load training data from JSONL file."""
    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    
    return Dataset.from_list(examples)


def format_example(example: dict) -> dict:
    """Format a single example for training."""
    # Combine prompt and completion into a single text
    # The model learns to generate the completion given the prompt
    text = example["prompt"] + "\n" + example["completion"]
    return {"text": text}


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA SFT training for sketch/proof generation"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="sft_train_data.jsonl",
        help="Training data JSONL file",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="deepseek-ai/DeepSeek-Prover-V2-7B",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./checkpoints/sketch-prover",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device batch size (keep small for L4)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=16,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("QLoRA SFT Training for Sketch/Proof Generation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"Max length: {args.max_length}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Add special tokens if not present
    special_tokens = ["<sketch>", "</sketch>", "<proof>", "</proof>"]
    tokens_to_add = [t for t in special_tokens if t not in tokenizer.get_vocab()]
    if tokens_to_add:
        print(f"Adding special tokens: {tokens_to_add}")
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit quantization config for QLoRA
    print("\nConfiguring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with quantization
    print(f"\nLoading model {args.model} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Resize embeddings if we added tokens
    if tokens_to_add:
        model.resize_token_embeddings(len(tokenizer))
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # LoRA configuration
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and process training data
    print(f"\nLoading training data from {args.data}...")
    dataset = load_training_data(args.data)
    print(f"Loaded {len(dataset)} examples")
    
    # Format dataset
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    
    # SFTConfig (replaces TrainingArguments in TRL 0.25+)
    print("\nConfiguring training...")
    sft_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",  # Disable wandb etc.
        dataloader_pin_memory=False,  # Save memory
        max_grad_norm=0.3,
        # SFT-specific parameters
        max_length=args.max_length,
        packing=False,  # Don't pack sequences
        dataset_text_field="text",  # Field name in our dataset
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    # Train!
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {args.output}...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
