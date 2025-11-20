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
from tqdm import tqdm

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

# Role-specific system prompts for the single model
DECOMPOSER_PROMPT = """You are a Lean 4 proof assistant in DECOMPOSER mode.
Given a goal statement, break it into intermediate steps and emit Lean 4 code
where subgoals end with `sorry` placeholders. Focus on the high-level proof structure.

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
        "Write a proof sketch in Lean 4with `sorry` placeholders:"
    )


def build_prover_prompt(sketch: str, test_info: Dict) -> str:
    """Build prompt for prover role."""
    return (
        f"{PROVER_PROMPT}\n\n"
        f"Theorem statement:\n```lean4\n{test_info['statement']}\n```\n\n"
        f"Sketch to complete:\n```lean4\n{sketch}\n```\n\n"
        "Complete the proof (replace all `sorry`):"
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
    return completions[0]["text"].strip()


def _strip_code_fences(text: str) -> str:
    """
    Remove leading/trailing Markdown code fences from a completion.

    This makes the string valid Lean code even if the model wraps it in ```lean4 ... ```.
    """
    if not text:
        return text

    lines = text.strip().splitlines()
    # Remove leading fence line
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    # Remove trailing fence lines
    while lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


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

    if not lemmas:
        logging.warning("No lemmas loaded; nothing to do.")
        return []

    # --- Phase 1: Decomposer (batched generation) ---
    sketches: List[str] = [""] * len(lemmas)
    gen_batch_size = max(1, args.generation_batch_size)

    decomposer_pbar = tqdm(total=len(lemmas), desc="Generating decomposer sketches")
    for start in range(0, len(lemmas), gen_batch_size):
        end = min(start + gen_batch_size, len(lemmas))
        batch_lemmas = lemmas[start:end]
        prompts = [build_decomposer_prompt(t) for t in batch_lemmas]

        completions = direct_completion(
            predictor,
            prompts,
            temperature=args.decomposer_temperature,
            max_tokens=args.max_tokens,
            seed=args.seed + start,
            cache_dir=args.cache_dir,
        )
        for i, completion in enumerate(completions):
            raw_text = completion["text"].strip()
            sketches[start + i] = _strip_code_fences(raw_text)

        decomposer_pbar.update(len(batch_lemmas))
    decomposer_pbar.close()

    # --- Phase 2: Prover (batched generation) ---
    full_proofs: List[str] = [""] * len(lemmas)

    prover_pbar = tqdm(total=len(lemmas), desc="Generating prover completions")
    for start in range(0, len(lemmas), gen_batch_size):
        end = min(start + gen_batch_size, len(lemmas))
        batch_lemmas = lemmas[start:end]
        batch_sketches = sketches[start:end]
        prompts = [
            build_prover_prompt(sketch, test_info)
            for sketch, test_info in zip(batch_sketches, batch_lemmas)
        ]

        completions = direct_completion(
            predictor,
            prompts,
            temperature=args.prover_temperature,
            max_tokens=args.max_tokens,
            seed=args.seed + 10_000 + start,
            cache_dir=args.cache_dir,
        )
        for i, completion in enumerate(completions):
            raw_text = completion["text"].strip()
            full_proofs[start + i] = _strip_code_fences(raw_text)

        prover_pbar.update(len(batch_lemmas))
    prover_pbar.close()

    # Build proof infos
    proof_infos: List[Dict] = []
    for lemma, sketch, proof in zip(lemmas, sketches, full_proofs):
        proof_info = lemma.copy()
        decomposer_prompt = build_decomposer_prompt(lemma)
        prover_prompt = build_prover_prompt(sketch, lemma)
        proof_info["proof_sketch"] = sketch
        proof_info["proof"] = proof
        proof_info["decomposer_prompt"] = decomposer_prompt
        proof_info["prover_prompt"] = prover_prompt
        # Full Lean code sent to the verifier; allow model to emit the whole theorem
        proof_info["code"] = proof
        proof_infos.append(proof_info)

    # --- Phase 3: Verification (batched, with graceful fallback) ---
    results: List[Dict] = []
    verify_batch_size = max(1, args.verify_batch_size)

    batches = split_test_blocks(proof_infos, batch_size=verify_batch_size, group_by_header=False)
    verify_pbar = tqdm(total=len(proof_infos), desc="Verifying proofs")

    for block in batches:
        try:
            verified_block = verifier.run(block, batched=len(block) > 1)
        except Exception as exc:
            logging.error(f"Lean batch verification failed for lemmas {[t['lemma_id'] for t in block]}: {exc}")
            verified_block = []
            for proof_info in block:
                try:
                    single_verified = verifier.run([proof_info], batched=False)[0]
                except Exception as single_exc:
                    logging.error(f"Lean single verification failed for lemma {proof_info['lemma_id']}: {single_exc}")
                    fallback = proof_info.copy()
                    fallback["complete"] = False
                    fallback["system_errors"] = str(single_exc)
                    single_verified = fallback
                verified_block.append(single_verified)

        for verified in verified_block:
            verified.update(get_result_items(verified))
            results.append(verified)

        verify_pbar.update(len(block))

    verify_pbar.close()

    # Save results
    save_path = os.path.join(
        args.exp_dir, f"{args.save_file_name}.jsonl"
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
    parser.add_argument("--save_file_name", default="test_results", help="Output file name")
    parser.add_argument("--max_examples", type=int, default=8, help="Max examples per dataset")
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
    parser.add_argument("--decomposer_temperature", type=float, default=1.0)
    parser.add_argument("--prover_temperature", type=float, default=1.0)
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
