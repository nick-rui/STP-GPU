#!/usr/bin/env python3
"""
Prepare SFT training data for sketch and proof generation.

Takes cold_start_data.jsonl and creates a mixed dataset with:
- Sketch examples: <sketch>{preamble}</sketch> → {sketch}
- Proof examples: <proof>{preamble}</proof> → {proof}

Each entry in cold_start_data.jsonl generates TWO training examples (one sketch, one proof).

Usage:
    python prepare_sft_data.py \
        --input cold_start_data.jsonl \
        --output sft_train_data.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path


def extract_preamble(code: str) -> str:
    """
    Extract the preamble (imports + theorem signature up to := by) from Lean code.
    
    Returns everything up to and including the first `:= by` on a line.
    """
    lines = code.split("\n")
    preamble_lines = []
    
    for line in lines:
        preamble_lines.append(line)
        # Check if this line contains := by (end of theorem signature)
        if ":= by" in line:
            break
    
    return "\n".join(preamble_lines)


def create_training_example(preamble: str, completion: str, task_type: str) -> dict:
    """
    Create a training example with prompt and completion.
    
    Args:
        preamble: The imports + theorem signature (up to := by)
        completion: The full code (sketch or proof)
        task_type: Either 'sketch' or 'proof'
    
    Returns:
        Dict with 'prompt', 'completion', and 'task_type' fields
    """
    prompt = f"<{task_type}>\n{preamble}\n</{task_type}>"
    
    return {
        "prompt": prompt,
        "completion": completion,
        "task_type": task_type,
    }


def process_cold_start_data(input_path: str) -> list:
    """
    Process cold_start_data.jsonl and generate training examples.
    
    Each entry generates two examples: one for sketch, one for proof.
    """
    examples = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}", file=sys.stderr)
                continue
            
            # Get the fields
            sketch = entry.get("sketch", "")
            proof = entry.get("proof", "")
            problem_id = entry.get("problem_id", f"unknown_{line_num}")
            
            if not sketch or not proof:
                print(f"Warning: Skipping entry {problem_id} - missing sketch or proof", file=sys.stderr)
                continue
            
            # Extract preamble from the proof (should be the same for sketch)
            preamble = extract_preamble(proof)
            
            # Create sketch example
            sketch_example = create_training_example(preamble, sketch, "sketch")
            sketch_example["problem_id"] = problem_id
            examples.append(sketch_example)
            
            # Create proof example
            proof_example = create_training_example(preamble, proof, "proof")
            proof_example["problem_id"] = problem_id
            examples.append(proof_example)
    
    return examples


def write_jsonl(data: list, output_path: str) -> None:
    """Write data to JSONL file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SFT training data for sketch and proof generation."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="cold_start_data.jsonl",
        help="Input cold_start_data.jsonl file",
    )
    parser.add_argument(
        "--output", "-o", 
        type=str,
        default="sft_train_data.jsonl",
        help="Output training data file",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the output examples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    
    args = parser.parse_args()
    
    # Process data
    print(f"Processing {args.input}...", file=sys.stderr)
    examples = process_cold_start_data(args.input)
    
    # Count by task type
    sketch_count = sum(1 for e in examples if e["task_type"] == "sketch")
    proof_count = sum(1 for e in examples if e["task_type"] == "proof")
    
    print(f"Generated {len(examples)} training examples:", file=sys.stderr)
    print(f"  - Sketch examples: {sketch_count}", file=sys.stderr)
    print(f"  - Proof examples: {proof_count}", file=sys.stderr)
    
    # Optionally shuffle
    if args.shuffle:
        import random
        random.seed(args.seed)
        random.shuffle(examples)
        print(f"Shuffled with seed {args.seed}", file=sys.stderr)
    
    # Write output
    write_jsonl(examples, args.output)
    print(f"Wrote to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()





