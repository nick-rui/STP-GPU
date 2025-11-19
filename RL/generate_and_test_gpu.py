"""
GPU-optimized version of generate_and_test.py without Ray dependencies.

This script provides a simple inference and evaluation pipeline for generating
and testing Lean4 proofs using a single GPU. It uses direct vLLM calls and
Lean4 verification without Ray overhead.

Pipeline:
1. Load dataset configurations and parse formal Lean statements
2. Convert statements to prompts
3. Generate proofs using vLLM
4. Extract proof text from model completions
5. Verify proofs using Lean4
6. Save results

Usage:
    python RL/generate_and_test_gpu.py \\
        --model <model_path> \\
        --tokenizer_path <tokenizer_path> \\
        --exp_dir <output_directory> \\
        --raw_dataset_config <dataset_config.json> \\
        --temperature 0.7 \\
        --max_examples 8
"""

import os
import json
import argparse
import logging
import numpy as np
from transformers import AutoTokenizer
from copy import deepcopy

from utils.model_utils_gpu import insert_lemma
from utils.gcloud_utils import read_file, write_data
from utils.RL_utils_gpu import (
    generate_and_test,
    collect_trajectories,
    SimpleLLMPredictor,
    MAX_LENGTH,
    REPO_DIR
)

def main(args: argparse.Namespace):
    """
    Main function to generate and test proofs without Ray.
    
    Args:
        args: Command-line arguments containing model paths, dataset config, etc.
    """
    args.save_file_path = os.path.join(args.exp_dir, f'generated_proofs_{args.save_file_name}.jsonl')
    print(f"Arguments: {args}")
    
    # Setup tokenizer (for compatibility, though SimpleLLMPredictor has its own)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.truncation_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    # Create experiment directory if not using GCS
    if 'gs://' not in args.exp_dir:
        os.makedirs(args.exp_dir, exist_ok=True)

    # Initialize data structures
    lemmas_to_generate = []
    lemma_mapping = {}

    # Load and process datasets
    idx = 0
    dataset_configs = read_file(args.raw_dataset_config)
    
    if dataset_configs is None:
        raise ValueError(f"Failed to read dataset config from {args.raw_dataset_config}")

    logging.info(f"Processing {len(dataset_configs)} dataset configurations...")
    
    for dataset_config in dataset_configs:
        logging.debug(f'Processing dataset: {dataset_config["dataset_path"]}')
        raw_dataset = read_file(os.path.join(REPO_DIR, dataset_config['dataset_path']))
        nr_samples = dataset_config['weight']
        
        if raw_dataset is None:
            raise ValueError(f"Failed to read {dataset_config['dataset_path']}")
        
        logging.info(f'Size of the dataset: {len(raw_dataset)}')
        
        # Format dataset entries
        formatted_ds = []
        for raw in raw_dataset[:args.max_examples]:
            test_info = {
                'lemma_id': idx,
                'statement': raw['formal_statement'].rsplit('sorry', 1)[0].strip(),
                'label': [raw['split']] + (raw.get('tags', None) or []),
                'header': raw.get('header', None),
            }
            idx += 1
            formatted_ds.append(test_info)
        
        # Create multiple samples per problem (limited for quick testing)
        for it in range(min(1, nr_samples)):  # Use 1 sample instead of nr_samples for quick test
            lemmas_to_generate += [test_info | {'iter': it} for test_info in deepcopy(formatted_ds)]
        
        # Build lemma mapping
        for test_info in formatted_ds:
            insert_lemma(lemma_mapping, test_info)

    logging.info(f'Total lemmas to generate: {len(lemmas_to_generate)}')

    # Create predictor (replaces Ray actors)
    logging.info(f'Initializing model predictor (model: {args.model}, tokenizer: {args.tokenizer_path})...')
    predictor = SimpleLLMPredictor(args.model, args.tokenizer_path, enable_prefix_caching=False)

    # Create trajectory collection function with fixed parameters
    # The signature must match: collect_traj(predictor, num_workers, selected_lemmas, lemma_mapping, seed)
    # collect_trajectories signature: (predictor, num_workers, lemmas_to_generate, max_length, seed, temperature, cache_dir)
    def collect_traj(predictor, num_workers, selected_lemmas, lemma_mapping, seed):
        return collect_trajectories(
            predictor,
            num_workers,
            selected_lemmas,
            max_length=MAX_LENGTH,
            seed=seed,
            temperature=args.temperature,
            cache_dir=os.path.join(args.exp_dir, 'sampler_ckpt')
        )

    # Shuffle lemmas for randomness
    rng = np.random.default_rng(args.seed)
    rng.shuffle(lemmas_to_generate)

    # Generate and test proofs
    logging.info('Starting proof generation and verification...')
    generated_proofs = generate_and_test(
        selected_lemmas=lemmas_to_generate,
        collect_traj=collect_traj,
        predictor=predictor,
        lemma_mapping=lemma_mapping,
        seed=args.seed,
        save_dir=os.path.join(args.exp_dir, 'sampler_ckpt'),
        temperature=args.temperature,
        cpus_per_task=4,  # Unused in GPU version but kept for compatibility
        cpus_per_task_stage2=6,  # Unused in GPU version but kept for compatibility
        group_by_header=True,
        collect_premises=False
    )
    
    # Save results
    logging.info(f'Saving {len(generated_proofs)} generated proofs to {args.save_file_path}...')
    write_data('\n'.join([json.dumps(t) for t in generated_proofs]), args.save_file_path, 'jsonl')
    
    # Print summary statistics
    complete_count = sum(1 for p in generated_proofs if p.get('complete', False))
    logging.info(f'Generation complete!')
    logging.info(f'  Total proofs generated: {len(generated_proofs)}')
    logging.info(f'  Complete proofs: {complete_count} ({100*complete_count/len(generated_proofs):.2f}%)')
    logging.info(f'  Results saved to: {args.save_file_path}')


if __name__ == "__main__":
    logging.basicConfig(
        format='[%(asctime)s - %(name)s - %(levelname)s] %(message)s',
        level=logging.DEBUG
    )
    
    parser = argparse.ArgumentParser(
        description="Generate and test proofs using GPU (no Ray)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        required=True,
        help="Path to model directory or HuggingFace model name"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default='deepseek-ai/DeepSeek-Prover-V1.5-SFT',
        help="Path to tokenizer or HuggingFace tokenizer name"
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        required=True,
        help="Experiment directory to save results"
    )
    parser.add_argument(
        "--nr_samples",
        type=int,
        default=16,
        help="Number of samples per problem (currently limited to 1 for quick testing)"
    )
    parser.add_argument(
        "--save_file_name",
        type=str,
        default=None,
        required=True,
        help="Name for the output file (without extension)"
    )
    parser.add_argument(
        "--raw_dataset_config",
        type=str,
        default=None,
        required=True,
        help="Path to dataset configuration JSON file"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for proof generation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=8,
        help="Maximum number of examples to process per dataset"
    )
    
    parsed_args = parser.parse_args()
    main(parsed_args)

