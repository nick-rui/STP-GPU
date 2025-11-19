"""Single-model inference pipeline for decomposer → prover workflow.

Uses ONE model with different prompts for both roles, inspired by STP's approach
of training a single model to perform multiple tasks via prompt differentiation.

1. Decomposer role: writes a proof sketch with `sorry` placeholders
2. Prover role: fills the `sorry`s with actual proofs
3. Lean4 verifies the completed proof

Usage:
    python RL/inference_single_model.py \
        --model <model_path> \
        --tokenizer_path <tokenizer_path> \
        --exp_dir <output_directory> \
        --raw_dataset_config <dataset_config.json> \
        --max_examples 8
"""

import argparse
import json
import logging
import os
from typing import List, Dict

from utils.RL_utils_gpu import (
    SimpleLLMPredictor,
    SimpleLean4Verifier,
    direct_completion,
    get_result_items,
    MAX_LENGTH,
    REPO_DIR,
)
from utils.gcloud_utils import read_file, write_data

# Marker tokens for role differentiation (similar to STP approach)
SKETCH_START = '<sketch>'
SKETCH_END = '</sketch>'
PROOF_START = '<proof>'
PROOF_END = '</proof>'

# Base prompt for Lean 4 code completion
BASE_PROMPT = 'Complete the following Lean 4 code:\n\n```lean4\nimport Mathlib\nimport Aesop\nset_option maxHeartbeats 0\nopen BigOperators Real Nat Topology Rat\n'


def build_decomposer_prompt(test_info: Dict) -> str:
    """
    Build prompt for decomposer role using markers.

    Format:
        <base prompt>
        <statement>
        <sketch>
        -- instruction

    Model should output: sketch with sorry placeholders, then </sketch>
    """
    header = test_info.get("header")
    if header is not None:
        prompt = f'Complete the following Lean 4 code:\n\n```lean4\n{header}'
    else:
        prompt = BASE_PROMPT

    return (
        f"{prompt}\n"
        f"{test_info['statement'].strip()}\n"
        f"{SKETCH_START}\n"
        f"-- Break into steps using sorry for unproved subgoals:\n"
    )


def build_prover_prompt(sketch: str, test_info: Dict) -> str:
    """
    Build prompt for prover role using markers.

    Format:
        <base prompt>
        <statement>
        <sketch>
        <sketch content>
        </sketch>
        <proof>

    Model should output: completed proof, then </proof>
    """
    header = test_info.get("header")
    if header is not None:
        prompt = f'Complete the following Lean 4 code:\n\n```lean4\n{header}'
    else:
        prompt = BASE_PROMPT

    return (
        f"{prompt}\n"
        f"{test_info['statement'].strip()}\n"
        f"{SKETCH_START}\n"
        f"{sketch.strip()}\n"
        f"{SKETCH_END}\n"
        f"{PROOF_START}\n"
        f"-- Complete proof by replacing all sorry:\n"
    )


def load_lemmas(config_path: str, max_examples: int) -> List[Dict]:
    """Load lemmas from dataset configuration."""
    dataset_configs = read_file(config_path)
    if dataset_configs is None:
        raise ValueError(f"Failed to read dataset config from {config_path}")

    lemmas = []
    idx = 0
    for dataset_config in dataset_configs:
        raw_dataset = read_file(os.path.join(REPO_DIR, dataset_config["dataset_path"]))
        if raw_dataset is None:
            raise ValueError(f"Failed to read {dataset_config['dataset_path']}")

        for raw in raw_dataset[:max_examples]:
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
    return lemmas


def run_completion(
    predictor: SimpleLLMPredictor,
    prompt: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    cache_dir: str | None = None,
    end_marker: str | None = None,
) -> str:
    """Run a single completion using the predictor."""
    completions = direct_completion(
        predictor,
        [prompt],
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        cache_dir=cache_dir,
    )
    text = completions[0]["text"].strip()

    # Extract content before end marker if specified
    if end_marker and end_marker in text:
        text = text.split(end_marker)[0].strip()

    return text


def run_pipeline(args: argparse.Namespace) -> List[Dict]:
    """
    Run the single-model decomposer → prover pipeline.

    Uses ONE model with different prompts for decomposer and prover roles.
    """
    os.makedirs(args.exp_dir, exist_ok=True)
    lemmas = load_lemmas(args.raw_dataset_config, args.max_examples)

    tokenizer_path = args.tokenizer_path or args.model

    # Initialize single model for both roles
    logging.info(f"Loading single model for both roles: {args.model}")
    predictor = SimpleLLMPredictor(
        args.model, tokenizer_path, enable_prefix_caching=False
    )

    # Initialize verifier
    verifier = SimpleLean4Verifier(
        collect_premises=args.collect_premises, timeout=args.timeout
    )

    results = []

    for idx, test_info in enumerate(lemmas):
        logging.info(f"Processing lemma {idx + 1}/{len(lemmas)} (id: {test_info['lemma_id']})")

        # Phase 1: Decomposer - generate proof sketch
        logging.info("  Phase 1: Generating proof sketch (decomposer role)...")
        decomposer_prompt = build_decomposer_prompt(test_info)
        sketch = run_completion(
            predictor,
            decomposer_prompt,
            temperature=args.decomposer_temperature,
            max_tokens=args.max_tokens,
            seed=args.seed + idx,
            cache_dir=args.cache_dir,
            end_marker=SKETCH_END,
        )

        # Phase 2: Prover - fill in the sorry placeholders
        logging.info("  Phase 2: Completing proof (prover role)...")
        prover_prompt = build_prover_prompt(sketch, test_info)
        full_proof = run_completion(
            predictor,
            prover_prompt,
            temperature=args.prover_temperature,
            max_tokens=args.max_tokens,
            seed=args.seed + idx + 10_000,
            cache_dir=args.cache_dir,
            end_marker=PROOF_END,
        )

        # Build result
        proof_info = test_info.copy()
        proof_info["proof_sketch"] = sketch
        proof_info["proof"] = full_proof
        proof_info["decomposer_prompt"] = decomposer_prompt
        proof_info["prover_prompt"] = prover_prompt

        # Phase 3: Verify with Lean4
        logging.info("  Phase 3: Verifying with Lean4...")
        try:
            verification = verifier.run([proof_info], batched=False)[0]
            proof_info.update(verification)
        except Exception as exc:
            logging.error(f"Lean verification failed for lemma {test_info['lemma_id']}: {exc}")
            proof_info["complete"] = False
            proof_info["system_errors"] = str(exc)

        proof_info.update(get_result_items(proof_info))
        results.append(proof_info)

        status = "✓" if proof_info.get("complete", False) else "✗"
        logging.info(f"  Result: {status}")

    # Save results
    save_path = os.path.join(
        args.exp_dir, f"proofs_{args.save_file_name or 'single_model'}.jsonl"
    )
    write_data("\n".join(json.dumps(r) for r in results), save_path, "jsonl")

    # Summary
    complete_count = sum(1 for r in results if r.get("complete", False))
    logging.info(f"\nPipeline complete!")
    logging.info(f"  Total: {len(results)}")
    logging.info(f"  Verified: {complete_count} ({100*complete_count/len(results):.1f}%)")
    logging.info(f"  Saved to: {save_path}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-model decomposer → prover inference pipeline."
    )
    parser.add_argument("--model", required=True, help="Model path (used for both roles)")
    parser.add_argument("--tokenizer_path", required=False, help="Tokenizer path (defaults to model)")
    parser.add_argument("--exp_dir", required=True, help="Output directory")
    parser.add_argument("--raw_dataset_config", required=True, help="Dataset config JSON")
    parser.add_argument("--save_file_name", default="single_model", help="Output file name")
    parser.add_argument("--max_examples", type=int, default=8, help="Max examples per dataset")
    parser.add_argument("--decomposer_temperature", type=float, default=0.7)
    parser.add_argument("--prover_temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=300, help="Lean verification timeout")
    parser.add_argument("--collect_premises", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        format="[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
        level=logging.INFO
    )
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
