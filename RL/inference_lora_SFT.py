#!/usr/bin/env python
"""
SFT inference script with LoRA support.

Uses the <sketch> and <proof> special tokens that the model was trained on.

Workflow:
1. Merge LoRA adapters into base model (if provided)
2. Decomposer: <sketch>{preamble}</sketch> → sketch with sorry
3. Prover: <proof>{preamble}</proof> → complete proof
4. Verify with Lean4

Usage:
    python inference_lora_SFT.py \
        --model deepseek-ai/DeepSeek-Prover-V2-7B \
        --lora_checkpoint ./SFT/checkpoints/sketch-prover \
        --exp_dir ./results/sft-test \
        --raw_dataset_config ./dataset_configs/miniF2F-test.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from typing import Dict, List, Tuple

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.RL_utils_gpu import (
    SimpleLLMPredictor,
    SimpleLean4Verifier,
    direct_completion,
    get_result_items,
    MAX_LENGTH,
    REPO_DIR,
    split_test_blocks,
)
from utils.gcloud_utils import read_file, write_data


# ============================================================================
# SFT-style prompts using special tokens
# ============================================================================

# Standard Lean4 header used in training
LEAN_HEADER = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat
"""


def build_sft_decomposer_prompt(test_info: Dict) -> str:
    """
    Build decomposer prompt using <sketch> tokens.
    
    Format matches SFT training:
        <sketch>
        {header}
        {statement}
        </sketch>
    """
    header = test_info.get("header", LEAN_HEADER)
    statement = test_info["statement"].strip()
    
    # Ensure statement ends with `:= by` for consistency
    if not statement.rstrip().endswith(":= by"):
        if ":= by" not in statement:
            statement = statement + " := by"
    
    return f"<sketch>\n{header}\n{statement}\n</sketch>\n"


def build_sft_prover_prompt(sketch: str, test_info: Dict) -> str:
    """
    Build prover prompt using <proof> tokens.
    
    Format matches SFT training:
        <proof>
        {header}
        {statement}
        </proof>
    
    Note: We use the original statement (not the sketch) to prompt for full proof.
    """
    header = test_info.get("header", LEAN_HEADER)
    statement = test_info["statement"].strip()
    
    if not statement.rstrip().endswith(":= by"):
        if ":= by" not in statement:
            statement = statement + " := by"
    
    return f"<proof>\n{header}\n{statement}\n</proof>\n"


# ============================================================================
# LoRA merging utilities
# ============================================================================

def _string_to_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_str]


def _validate_lora_checkpoint(checkpoint_path: str) -> str:
    if not os.path.isdir(checkpoint_path):
        raise ValueError(f"LoRA checkpoint directory not found: {checkpoint_path}")

    adapter_config = os.path.join(checkpoint_path, "adapter_config.json")
    adapter_options = [
        os.path.join(checkpoint_path, "adapter_model.bin"),
        os.path.join(checkpoint_path, "adapter_model.safetensors"),
    ]

    if not os.path.exists(adapter_config):
        raise ValueError(f"Missing adapter_config.json in {checkpoint_path}")

    adapter_model = next((path for path in adapter_options if os.path.exists(path)), None)
    if adapter_model is None:
        raise ValueError(
            f"Missing adapter_model.bin or adapter_model.safetensors in {checkpoint_path}"
        )

    logging.info(
        "✓ LoRA checkpoint validated: %s (found %s)",
        checkpoint_path,
        os.path.basename(adapter_model),
    )
    return adapter_model


def _merge_lora_checkpoint(
    base_model: str,
    checkpoint: str,
    output_dir: str,
    dtype: str,
    device_map: str,
) -> None:
    _validate_lora_checkpoint(checkpoint)

    logging.info("Loading base model %s for LoRA merge...", base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=_string_to_dtype(dtype),
        device_map=device_map,
        trust_remote_code=True,
    )

    # Load tokenizer from checkpoint to check for added special tokens
    try:
        checkpoint_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if len(checkpoint_tokenizer) > base.get_input_embeddings().weight.shape[0]:
            logging.info("Resizing embeddings to match checkpoint tokenizer (%d tokens)", len(checkpoint_tokenizer))
            base.resize_token_embeddings(len(checkpoint_tokenizer))
    except Exception as e:
        logging.warning("Could not check tokenizer for embedding resize: %s", e)

    logging.info("Applying LoRA adapter from %s", checkpoint)
    peft_model = PeftModel.from_pretrained(base, checkpoint)
    merged_model = peft_model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    logging.info("Saving merged model to %s", output_dir)
    merged_model.save_pretrained(output_dir)

    logging.info("Saving tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_dir)


def _prepare_model_paths(args: argparse.Namespace) -> Tuple[str, str | None]:
    """Merge LoRA weights if needed and return the model/tokenizer paths."""
    if not args.lora_checkpoint:
        logging.info("No LoRA checkpoint provided; running base model directly.")
        return args.model, args.tokenizer_path

    os.makedirs(args.exp_dir, exist_ok=True)
    merged_dir = args.merged_model_dir or os.path.join(args.exp_dir, "merged_lora_model")

    if os.path.exists(merged_dir) and not args.force_merge:
        logging.info("Found existing merged model at %s; reusing.", merged_dir)
        return merged_dir, args.tokenizer_path or merged_dir

    if os.path.exists(merged_dir):
        logging.info("Removing existing merged directory at %s due to --force_merge.", merged_dir)
        shutil.rmtree(merged_dir)

    logging.info(
        "Merging LoRA checkpoint %s into base model %s", args.lora_checkpoint, args.model
    )
    _merge_lora_checkpoint(
        base_model=args.model,
        checkpoint=args.lora_checkpoint,
        output_dir=merged_dir,
        dtype=args.lora_dtype,
        device_map=args.lora_device_map,
    )
    return merged_dir, args.tokenizer_path or merged_dir


# ============================================================================
# Inference pipeline with SFT prompts
# ============================================================================

def load_lemmas(config_path: str, max_examples: int) -> List[Dict]:
    """Load lemmas from dataset configuration."""
    dataset_configs = read_file(config_path)
    if dataset_configs is None:
        raise ValueError(f"Failed to read dataset config from {config_path}")

    lemmas = []
    idx = 0
    for config in dataset_configs:
        data = read_file(config["dataset_path"])
        if data is None:
            logging.warning("Could not read %s", config["dataset_path"])
            continue

        for item_or_items in get_result_items(data):
            for test_info in split_test_blocks(item_or_items):
                if max_examples and idx >= max_examples:
                    break
                test_info["dataset_label"] = config.get("label", [])
                lemmas.append(test_info)
                idx += 1
            if max_examples and idx >= max_examples:
                break
        if max_examples and idx >= max_examples:
            break

    logging.info("Loaded %d lemmas from %s", len(lemmas), config_path)
    return lemmas


def run_sft_pipeline(args: argparse.Namespace) -> None:
    """
    Run the SFT decomposer → prover pipeline using <sketch>/<proof> tokens.
    """
    # Load lemmas
    lemmas = load_lemmas(args.raw_dataset_config, args.max_examples)
    if not lemmas:
        logging.error("No lemmas loaded!")
        return

    # Initialize model
    logging.info("Loading model from %s", args.model)
    tokenizer_path = args.tokenizer_path or args.model
    predictor = SimpleLLMPredictor(args.model, tokenizer_path, enable_prefix_caching=False)

    # Initialize verifier
    logging.info("Initializing Lean4 verifier")
    verifier = SimpleLean4Verifier(
        project_dir=REPO_DIR,
        num_workers=args.verify_batch_size,
        timeout=args.timeout,
    )

    # --- Phase 1: Decomposer (generate sketches) ---
    logging.info("Phase 1: Generating sketches with <sketch> prompts")
    sketches = []
    decomposer_pbar = tqdm(total=len(lemmas), desc="Generating sketches")

    for i in range(0, len(lemmas), args.generation_batch_size):
        batch_lemmas = lemmas[i : i + args.generation_batch_size]
        prompts = [build_sft_decomposer_prompt(t) for t in batch_lemmas]

        batch_outputs = direct_completion(
            predictor,
            prompts,
            temperature=args.decomposer_temperature,
            max_tokens=args.max_tokens,
        )
        sketches.extend(batch_outputs)
        decomposer_pbar.update(len(batch_lemmas))
    decomposer_pbar.close()

    # --- Phase 2: Prover (complete proofs) ---
    logging.info("Phase 2: Completing proofs with <proof> prompts")
    proofs = []
    prover_prompts_list = [
        build_sft_prover_prompt(sketch, test_info)
        for sketch, test_info in zip(sketches, lemmas)
    ]

    prover_pbar = tqdm(total=len(lemmas), desc="Completing proofs")
    for i in range(0, len(lemmas), args.generation_batch_size):
        batch_prompts = prover_prompts_list[i : i + args.generation_batch_size]

        batch_outputs = direct_completion(
            predictor,
            batch_prompts,
            temperature=args.prover_temperature,
            max_tokens=args.max_tokens,
        )
        proofs.extend(batch_outputs)
        prover_pbar.update(len(batch_prompts))
    prover_pbar.close()

    # --- Phase 3: Verify with Lean4 ---
    logging.info("Phase 3: Verifying proofs with Lean4")
    verification_results = verifier.verify_batch(proofs)

    # --- Collect results ---
    results = []
    verified_count = 0
    for lemma, sketch, proof, verify_result in zip(lemmas, sketches, proofs, verification_results):
        decomposer_prompt = build_sft_decomposer_prompt(lemma)
        prover_prompt = build_sft_prover_prompt(sketch, lemma)

        proof_info = {
            "statement": lemma["statement"],
            "dataset_label": lemma.get("dataset_label", []),
            "decomposer_prompt": decomposer_prompt,
            "proof_sketch": sketch,
            "prover_prompt": prover_prompt,
            "proof": proof,
            "verified": verify_result.get("verified", False),
            "error": verify_result.get("error"),
        }
        results.append(proof_info)
        if proof_info["verified"]:
            verified_count += 1

    logging.info("Verified %d / %d proofs (%.1f%%)", verified_count, len(results), 100 * verified_count / len(results) if results else 0)

    # --- Save results ---
    os.makedirs(args.exp_dir, exist_ok=True)
    output_path = os.path.join(args.exp_dir, f"{args.save_file_name}.json")
    write_data(results, output_path)
    logging.info("Results saved to %s", output_path)

    # Cleanup
    del predictor
    del verifier


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT inference with <sketch>/<proof> prompts and optional LoRA merging."
    )
    
    # Model arguments
    parser.add_argument("--model", required=True, help="Base model path or HF repo id")
    parser.add_argument("--tokenizer_path", required=False, help="Tokenizer path")
    parser.add_argument("--exp_dir", required=True, help="Directory to store outputs")
    parser.add_argument("--raw_dataset_config", required=True, help="Dataset config JSON")
    parser.add_argument("--save_file_name", default="test_results", help="Output filename")
    parser.add_argument("--max_examples", type=int, default=8, help="Max examples per dataset")
    
    # Generation arguments
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--verify_batch_size", type=int, default=16)
    parser.add_argument("--decomposer_temperature", type=float, default=0.7)
    parser.add_argument("--prover_temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=300)
    
    # LoRA arguments
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="Path to LoRA adapter checkpoint")
    parser.add_argument("--merged_model_dir", type=str, default=None,
                        help="Directory to save merged model")
    parser.add_argument("--lora_dtype", type=str, default="float16",
                        choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--lora_device_map", type=str, default="auto")
    parser.add_argument("--force_merge", action="store_true",
                        help="Force re-merging even if merged model exists")
    
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s] %(message)s",
        level=logging.INFO,
    )
    args = parse_args()
    
    # Merge LoRA if provided
    model_path, tokenizer_path = _prepare_model_paths(args)
    args.model = model_path
    if tokenizer_path and not args.tokenizer_path:
        args.tokenizer_path = tokenizer_path

    logging.info("Starting SFT inference with model: %s", args.model)
    run_sft_pipeline(args)


if __name__ == "__main__":
    main()
