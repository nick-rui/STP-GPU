#!/usr/bin/env python3
"""
Example usage of the evaluation pipeline.

This script demonstrates how to use the evaluation pipeline programmatically.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.dataset_loader import load_leanworkbook_subset
from evaluation.inference import run_inference
from evaluation.verification import verify_proofs
from evaluation.metrics import compute_metrics, print_metrics


def main():
    """Example evaluation workflow."""
    
    # Step 1: Load dataset
    print("Loading dataset...")
    problems = load_leanworkbook_subset(
        dataset_path="assets/data/training/lean_workbook_dedup.json",
        max_problems=10,  # Small subset for example
        seed=0
    )
    print(f"Loaded {len(problems)} problems")
    
    # Step 2: Run inference
    print("\nRunning inference...")
    print("Note: This requires the model to be available. Using HuggingFace transformers.")
    generated_proofs = run_inference(
        problems=problems,
        model_name="deepseek-ai/DeepSeek-Prover-V2-7B",
        temperature=0.7,
        max_tokens=512,  # Shorter for example
        batch_size=2,
        device="cuda"  # Change to "cpu" if no GPU
    )
    print(f"Generated {len(generated_proofs)} proofs")
    
    # Step 3: Verify solutions
    print("\nVerifying proofs...")
    verified_results = verify_proofs(
        generated_proofs=generated_proofs,
        timeout=200,
        batch_size=10
    )
    print(f"Verified {len(verified_results)} proofs")
    
    # Step 4: Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(verified_results)
    print_metrics(metrics)
    
    print("\nExample completed!")


if __name__ == "__main__":
    main()

