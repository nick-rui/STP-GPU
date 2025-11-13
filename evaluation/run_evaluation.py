#!/usr/bin/env python3
"""
Main evaluation pipeline script for DeepSeek Prover v2 7B on LeanWorkbook dataset.

This script:
1. Loads a subset of LeanWorkbook dataset problems
2. Runs inference using DeepSeek Prover v2 7B
3. Verifies solutions using the Lean verifier
4. Reports metrics (pass rate, success rate, etc.)
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Add parent directory to path to import RL utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.dataset_loader import load_leanworkbook_subset
from evaluation.inference import run_inference
from evaluation.verification import verify_proofs
from evaluation.metrics import compute_metrics, print_metrics

logging.basicConfig(
    format='[%(asctime)s - %(name)s - %(levelname)s] %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main(args):
    """Main evaluation pipeline."""
    
    # Step 1: Load dataset
    logger.info(f"Loading LeanWorkbook dataset subset (max_problems={args.max_problems})...")
    problems = load_leanworkbook_subset(
        dataset_path=args.dataset_path,
        max_problems=args.max_problems,
        seed=args.seed
    )
    logger.info(f"Loaded {len(problems)} problems")
    
    # Step 2: Run inference
    logger.info(f"Running inference using model: {args.model_name}")
    logger.info(f"Temperature: {args.temperature}, Max tokens: {args.max_tokens}")
    
    generated_proofs = run_inference(
        problems=problems,
        model_name=args.model_name,
        tokenizer_path=args.tokenizer_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        device=args.device
    )
    logger.info(f"Generated {len(generated_proofs)} proofs")
    
    # Step 3: Verify solutions
    logger.info("Verifying generated proofs...")
    verified_results = verify_proofs(
        generated_proofs=generated_proofs,
        lake_path=args.lake_path,
        lean_workspace=args.lean_workspace,
        timeout=args.timeout,
        batch_size=args.verify_batch_size
    )
    logger.info(f"Verified {len(verified_results)} proofs")
    
    # Step 4: Compute and report metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(verified_results)
    
    print_metrics(metrics)
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(output_file, 'w') as f:
            json.dump({
                'metrics': metrics,
                'results': verified_results,
                'config': vars(args)
            }, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSeek Prover v2 7B on LeanWorkbook dataset"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="assets/data/training/lean_workbook_dedup.json",
        help="Path to LeanWorkbook dataset JSON file"
    )
    parser.add_argument(
        "--max_problems",
        type=int,
        default=100,
        help="Maximum number of problems to evaluate (for quick testing)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for dataset sampling"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-Prover-V2-7B",
        help="Model name or path for DeepSeek Prover v2 7B"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Tokenizer path (defaults to model_name if not specified)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    
    # Verification arguments
    parser.add_argument(
        "--lake_path",
        type=str,
        default=None,
        help="Path to lake executable (defaults to ~/.elan/bin/lake)"
    )
    parser.add_argument(
        "--lean_workspace",
        type=str,
        default=None,
        help="Path to Lean workspace (defaults to ~/lean/mathlib4/)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=200,
        help="Timeout for verification in seconds"
    )
    parser.add_argument(
        "--verify_batch_size",
        type=int,
        default=40,
        help="Batch size for verification"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results"
    )
    
    args = parser.parse_args()
    
    # Set default tokenizer path if not specified
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_name
    
    main(args)

