"""Dual-model inference pipeline inspired by `RL/generate_and_test_gpu.py`.

1. A decomposer model writes a proof sketch with `sorry` placeholders.
2. A prover model fills the `sorry`s.
3. Lean4 verifies the completed proof and stores JSONL results.
"""

import argparse
import json
import logging
import os
from typing import List, Dict

import torch

from utils.RL_utils_gpu import (
    SimpleLLMPredictor,
    SimpleLean4Verifier,
    direct_completion,
    get_result_items,
    MAX_LENGTH,
    REPO_DIR,
)
from utils.gcloud_utils import read_file, write_data

DECOMPOSER_INSTRUCTIONS = """You are a Lean assistant that writes proof sketches.
Given a goal statement, break it into intermediate lemmas and emit Lean 4 code
where every lemma ends with `:= by ... sorry` and the final theorem still
contains `sorry` placeholders for the unresolved subproofs."""

PROVER_INSTRUCTIONS = """You are a Lean assistant that fills sketches.
Given a Lean proof sketch containing `sorry`, replace each placeholder with a
valid proof and return only the completed Lean 4 code."""


def build_decomposer_prompt(test_info: Dict) -> str:
    header = test_info.get("header")
    prefix = f"{header}\n" if header else ""
    return (
        f"{DECOMPOSER_INSTRUCTIONS}\n"
        f"{prefix}"
        f"The goal is:\n{test_info['statement']}\n"
        "Return a sketch that obeys these instructions."
    )


def build_prover_prompt(sketch: str, test_info: Dict) -> str:
    return (
        f"{PROVER_INSTRUCTIONS}\n"
        f"Theorem statement:\n{test_info['statement']}\n"
        "Sketch to complete:\n"
        f"{sketch}\n"
        "Return the completed proof."
    )


def load_lemmas(config_path: str, max_examples: int) -> List[Dict]:
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
    completions = direct_completion(
        predictor,
        [prompt],
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        cache_dir=cache_dir,
    )
    return completions[0]["text"].strip()


def run_pipeline(args: argparse.Namespace) -> List[Dict]:
    os.makedirs(args.exp_dir, exist_ok=True)
    lemmas = load_lemmas(args.raw_dataset_config, args.max_examples)

    decomposer_tokenizer = args.decomposer_tokenizer or args.decomposer_model
    prover_tokenizer = args.prover_tokenizer or args.prover_model

    decomposer = SimpleLLMPredictor(
        args.decomposer_model, decomposer_tokenizer, enable_prefix_caching=False
    )

    verifier = SimpleLean4Verifier(
        collect_premises=args.collect_premises, timeout=args.timeout
    )

    sketches = []
    for idx, test_info in enumerate(lemmas):
        logging.info("Lemma %d/%d (%s)", idx + 1, len(lemmas), test_info["lemma_id"])
        sketch_prompt = build_decomposer_prompt(test_info)
        sketch = run_completion(
            decomposer,
            sketch_prompt,
            temperature=args.decomposer_temperature,
            max_tokens=args.max_tokens,
            seed=args.seed + idx,
            cache_dir=args.cache_dir,
        )

        sketches.append(test_info | {"proof_sketch": sketch})

    del decomposer
    torch.cuda.empty_cache()

    prover = SimpleLLMPredictor(
        args.prover_model, prover_tokenizer, enable_prefix_caching=False
    )

    results = []
    for idx, proof_info in enumerate(sketches):
        proof_prompt = build_prover_prompt(proof_info["proof_sketch"], proof_info)
        full_proof = run_completion(
            prover,
            proof_prompt,
            temperature=args.prover_temperature,
            max_tokens=args.max_tokens,
            seed=args.seed + idx + 10_000,
            cache_dir=args.cache_dir,
        )

        proof_info |= {"proof": full_proof}
        try:
            verification = verifier.run([proof_info], batched=False)[0]
            proof_info |= verification
        except Exception as exc:
            logging.error("Lean verification failed for lemma %s: %s", proof_info["lemma_id"], exc)
            proof_info |= {"complete": False, "system_errors": str(exc)}

        proof_info |= get_result_items(proof_info)
        results.append(proof_info)

    save_path = os.path.join(
        args.exp_dir, f"proofs_{args.save_file_name or 'dual_model'}.jsonl"
    )
    write_data("\n".join(json.dumps(r) for r in results), save_path, "jsonl")
    logging.info("Saved %d proofs to %s", len(results), save_path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-model inference pipeline.")
    parser.add_argument("--decomposer_model", required=True)
    parser.add_argument("--prover_model", required=True)
    parser.add_argument("--decomposer_tokenizer", required=False)
    parser.add_argument("--prover_tokenizer", required=False)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--raw_dataset_config", required=True)
    parser.add_argument("--save_file_name", default="dual_model")
    parser.add_argument("--max_examples", type=int, default=8)
    parser.add_argument("--decomposer_temperature", type=float, default=0.7)
    parser.add_argument("--prover_temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--collect_premises", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        format="[%(asctime)s - %(name)s - %(levelname)s] %(message)s", level=logging.INFO
    )
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
