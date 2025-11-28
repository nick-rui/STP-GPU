#!/usr/bin/env python3
"""
Inference example for the trained sketch/proof model.

Usage:
    python inference_example.py \
        --checkpoint ./checkpoints/sketch-prover \
        --mode sketch  # or 'proof'

Example:
    python inference_example.py --mode sketch --statement "theorem foo (x : Nat) : x + 0 = x := by"
"""

import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(checkpoint_path: str, base_model: str = "deepseek-ai/DeepSeek-Prover-V2-7B"):
    """Load the fine-tuned model with LoRA weights."""
    print(f"Loading tokenizer from {checkpoint_path}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading base model {base_model} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Resize embeddings to match tokenizer (in case special tokens were added)
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"Loading LoRA weights from {checkpoint_path}...")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 1024):
    """Generate completion for a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    return generated


def main():
    parser = argparse.ArgumentParser(description="Inference with trained sketch/proof model")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="./checkpoints/sketch-prover",
        help="Path to trained checkpoint",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="deepseek-ai/DeepSeek-Prover-V2-7B",
        help="Base model name",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["sketch", "proof"],
        default="sketch",
        help="Generation mode: 'sketch' or 'proof'",
    )
    parser.add_argument(
        "--statement", "-s",
        type=str,
        default=None,
        help="Theorem statement to complete",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate",
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.checkpoint, args.base_model)
    
    # Default example if no statement provided
    if args.statement is None:
        args.statement = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

theorem example_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : a + b > 0 := by"""
    
    # Create prompt
    prompt = f"<{args.mode}>\n{args.statement}\n</{args.mode}>\n"
    
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print("=" * 60)
    print("PROMPT:")
    print(prompt)
    print("=" * 60)
    print("GENERATED:")
    
    # Generate
    output = generate(model, tokenizer, prompt, max_new_tokens=args.max_tokens)
    print(output)
    print("=" * 60)


if __name__ == "__main__":
    main()









