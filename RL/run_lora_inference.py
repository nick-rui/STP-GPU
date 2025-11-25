#!/usr/bin/env python
"""
Wrapper script for `inference_single_model.py` that optionally merges LoRA checkpoints.

This lets us keep lightweight LoRA adapters and only materialize merged weights when
running inference. When `--lora_checkpoint` is provided we:
  1. Merge the adapter into the specified base model (or reuse an existing merge unless
     `--force_merge` is set)
  2. Forward the merged model path into the standard inference pipeline

If no LoRA checkpoint is given, the script behaves like the vanilla driver.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from typing import Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from inference_single_model import run_pipeline
from utils.RL_utils_gpu import MAX_LENGTH


def _add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    """Reuse the single-model inference arguments so downstream logic stays untouched."""
    parser.add_argument("--model", required=True, help="Base model path or HF repo id")
    parser.add_argument(
        "--tokenizer_path",
        required=False,
        help="Tokenizer path (defaults to model or merged dir when LoRA is applied)",
    )
    parser.add_argument("--exp_dir", required=True, help="Directory to store outputs")
    parser.add_argument(
        "--raw_dataset_config",
        required=True,
        help="Dataset config JSON consumed by inference_single_model",
    )
    parser.add_argument(
        "--save_file_name",
        default="test_results",
        help="Output filename (without extension)",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=8,
        help="Max number of examples per dataset",
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=8,
        help="Batch size for decomposer/prover generation",
    )
    parser.add_argument(
        "--verify_batch_size",
        type=int,
        default=16,
        help="Batch size for Lean4 verification",
    )
    parser.add_argument(
        "--decomposer_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for decomposer role",
    )
    parser.add_argument(
        "--prover_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for prover role",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=MAX_LENGTH,
        help="Max tokens generated for each completion",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Lean verification timeout (seconds)",
    )
    parser.add_argument(
        "--collect_premises",
        action="store_true",
        help="Collect premise info for verified proofs",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional directory for prompt/completion caching",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-model inference with optional LoRA merging."
    )
    _add_inference_arguments(parser)

    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        help="Path to LoRA adapter checkpoint directory (adapter_config + weights).",
    )
    parser.add_argument(
        "--merged_model_dir",
        type=str,
        default=None,
        help="Directory to save merged model (defaults to <exp_dir>/merged_lora_model).",
    )
    parser.add_argument(
        "--lora_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="dtype used while merging LoRA weights.",
    )
    parser.add_argument(
        "--lora_device_map",
        type=str,
        default="auto",
        help="device_map passed to transformers when merging LoRA weights.",
    )
    parser.add_argument(
        "--force_merge",
        action="store_true",
        help="Force re-merging even if merged_model_dir already exists.",
    )
    return parser.parse_args()


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
    """
    Merge LoRA weights if needed and return the model/tokenizer paths to use.
    """
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


def main() -> None:
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s] %(message)s",
        level=logging.INFO,
    )
    args = parse_args()
    model_path, tokenizer_path = _prepare_model_paths(args)
    args.model = model_path
    if tokenizer_path and not args.tokenizer_path:
        args.tokenizer_path = tokenizer_path

    logging.info("Starting inference with model: %s", args.model)
    run_pipeline(args)


if __name__ == "__main__":
    main()
