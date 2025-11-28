#!/usr/bin/env python3
"""
Generate cold start SFT data from the LeanWorkbook dataset.

This script:
1. Extracts n proofs from the Goedel-LM/Lean-workbook-proofs dataset on HuggingFace
2. Decomposes each proof into a sketch (replacing `:= by ...` with `:= by sorry`)
3. Exports (statement, sketch, proof) tuples as JSONL

Usage:
    python generate_cold_start_data.py \
        --n 1000 \
        --output /path/to/output.jsonl \
        [--seed 42] \
        [--shuffle] \
        [--strip-comments]

Output format (JSONL):
    {"problem_id": "...", "statement": "theorem ...", "sketch": "theorem ... := by sorry", "proof": "theorem ... := by\n  ..."}
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library is required. Install with: pip install datasets", file=sys.stderr)
    sys.exit(1)

# Import decompose_proof from sibling module
from decompose_proofs import decompose_proof


# Pattern to extract theorem/lemma declaration line(s)
THEOREM_START_PATTERN = re.compile(
    r"^((?:theorem|lemma|def|example)\s+\S+.*?:=\s*by)\b",
    re.MULTILINE | re.DOTALL
)

# Pattern for line comments (-- to end of line)
LINE_COMMENT_PATTERN = re.compile(r"--.*$", re.MULTILINE)

# Pattern for block comments (non-nested, innermost first)
BLOCK_COMMENT_PATTERN = re.compile(r"/\-(?:[^-]|-(?!/))*-/", re.DOTALL)


def strip_lean_comments(code: str) -> str:
    """
    Remove all Lean comments from code.
    
    Handles:
    - Line comments: -- to end of line
    - Block comments: /- ... -/ (including nested by iterative removal)
    
    Preserves code structure by replacing comments with appropriate whitespace.
    """
    # Remove block comments iteratively (handles nested comments)
    # Keep removing until no more block comments found
    prev_code = None
    while prev_code != code:
        prev_code = code
        code = BLOCK_COMMENT_PATTERN.sub("", code)
    
    # Remove line comments
    code = LINE_COMMENT_PATTERN.sub("", code)
    
    # Clean up: remove lines that are now empty (only whitespace)
    lines = code.split("\n")
    cleaned_lines = []
    for line in lines:
        # Keep lines that have actual content, or are part of indentation structure
        if line.strip() or (cleaned_lines and cleaned_lines[-1].strip()):
            cleaned_lines.append(line.rstrip())
    
    # Remove consecutive blank lines
    result_lines = []
    prev_blank = False
    for line in cleaned_lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        result_lines.append(line)
        prev_blank = is_blank
    
    return "\n".join(result_lines)


def extract_statement(full_proof: str) -> str:
    """
    Extract the statement (theorem/lemma signature) from a full proof.
    
    The statement is the theorem/lemma declaration from the keyword up to `:= by`.
    This excludes imports and setup code.
    """
    # Find theorem/lemma declaration with everything up to `:= by`
    match = THEOREM_START_PATTERN.search(full_proof)
    if match:
        return match.group(1).strip()
    
    # Fallback: try to find any `:= by` pattern
    match = re.search(r"(theorem|lemma|def|example)(\s+\S+.*?:=\s*by)\b", full_proof, re.DOTALL)
    if match:
        return (match.group(1) + match.group(2)).strip()
    
    # Last resort: return first non-import line
    for line in full_proof.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith(("import", "open", "set_option", "--", "/-", "#")):
            return stripped
    
    return full_proof.split("\n")[0].strip()


def process_entry(entry: dict, strip_comments: bool = False) -> dict:
    """
    Process a single dataset entry into (statement, sketch, proof) format.
    
    Args:
        entry: Dict with 'problem_id' and 'full_proof' fields
        strip_comments: Whether to remove Lean comments from the output
        
    Returns:
        Dict with 'problem_id', 'statement', 'sketch', and 'proof' fields
    """
    problem_id = entry.get("problem_id", "")
    full_proof = entry.get("full_proof", "")
    
    # Optionally strip comments
    if strip_comments:
        full_proof = strip_lean_comments(full_proof)
    
    # Extract statement from the full proof
    statement = extract_statement(full_proof)
    
    # Generate sketch by decomposing the proof (replacing tactic blocks with sorry)
    sketch = decompose_proof(full_proof)
    
    return {
        "problem_id": problem_id,
        "statement": statement,
        "sketch": sketch,
        "proof": full_proof,
    }


def sketch_has_sorry(sketch: str) -> bool:
    """Check if a sketch contains at least one 'sorry' (indicating inner lemmas were decomposed)."""
    return "sorry" in sketch


def load_leanworkbook_proofs(n: int, seed: int = None, shuffle: bool = False, strip_comments: bool = False) -> tuple:
    """
    Load n proofs from the Goedel-LM/Lean-workbook-proofs dataset.
    
    Only includes proofs where the sketch contains at least one 'sorry'
    (i.e., proofs that have inner lemmas to decompose).
    
    Args:
        n: Number of proofs to extract (after filtering)
        seed: Random seed for shuffling (optional)
        shuffle: Whether to shuffle before selecting
        strip_comments: Whether to remove Lean comments from proofs
        
    Returns:
        Tuple of (results list, filtered_count) where filtered_count is the
        number of proofs skipped because they had no inner lemmas.
    """
    print(f"Loading Goedel-LM/Lean-workbook-proofs dataset from HuggingFace...", file=sys.stderr)
    
    try:
        dataset = load_dataset("Goedel-LM/Lean-workbook-proofs", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)
    
    total_available = len(dataset)
    print(f"Dataset loaded: {total_available} proofs available", file=sys.stderr)
    
    # Get indices to select
    indices = list(range(total_available))
    
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(indices)
    
    # Process entries, filtering out those without sorry in sketch
    results = []
    filtered_count = 0
    processed_count = 0
    
    print(f"Processing proofs (target: {n} with inner lemmas)...", file=sys.stderr)
    if strip_comments:
        print("  (stripping comments)", file=sys.stderr)
    
    for idx in indices:
        if len(results) >= n:
            break
            
        entry = dataset[idx]
        processed = process_entry(entry, strip_comments=strip_comments)
        processed_count += 1
        
        # Only include if sketch has at least one sorry
        if sketch_has_sorry(processed["sketch"]):
            results.append(processed)
        else:
            filtered_count += 1
        
        # Progress indicator
        if processed_count % 1000 == 0:
            print(f"  Processed {processed_count} proofs, kept {len(results)}, filtered {filtered_count}", file=sys.stderr)
    
    print(f"  Final: processed {processed_count}, kept {len(results)}, filtered {filtered_count}", file=sys.stderr)
    
    if len(results) < n:
        print(f"Warning: Only found {len(results)} proofs with inner lemmas (requested {n})", file=sys.stderr)
    
    return results, filtered_count


def write_jsonl(data: list, output_path: str) -> None:
    """Write data to a JSONL file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"Wrote {len(data)} entries to {output_path}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate cold start SFT data from LeanWorkbook dataset. "
            "Extracts (statement, sketch, proof) tuples from Lean 4 proofs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract 1000 proofs (first 1000 in order)
    python generate_cold_start_data.py --n 1000 --output cold_start_data.jsonl

    # Extract 5000 random proofs with seed for reproducibility
    python generate_cold_start_data.py --n 5000 --output data/sft_5k.jsonl --shuffle --seed 42

    # Extract proofs with comments stripped (cleaner output)
    python generate_cold_start_data.py --n 1000 --output clean_proofs.jsonl --strip-comments

    # Extract all available proofs
    python generate_cold_start_data.py --n 999999 --output all_proofs.jsonl
        """
    )
    parser.add_argument(
        "-n", "--n",
        type=int,
        required=True,
        help="Number of proofs to extract from the dataset",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility when shuffling (optional)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before selecting n proofs",
    )
    parser.add_argument(
        "--strip-comments",
        action="store_true",
        help="Remove Lean comments (-- and /- -/) from proofs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load and process proofs
    data, filtered_count = load_leanworkbook_proofs(
        n=args.n,
        seed=args.seed,
        shuffle=args.shuffle,
        strip_comments=args.strip_comments,
    )
    
    # Write output
    write_jsonl(data, args.output)
    
    # Print summary
    print(f"\nSummary:", file=sys.stderr)
    print(f"  - Extracted: {len(data)} proofs (with inner lemmas)", file=sys.stderr)
    print(f"  - Filtered out: {filtered_count} proofs (no inner lemmas)", file=sys.stderr)
    print(f"  - Output: {args.output}", file=sys.stderr)
    if args.shuffle:
        print(f"  - Shuffled: Yes (seed={args.seed})", file=sys.stderr)
    else:
        print(f"  - Shuffled: No", file=sys.stderr)
    print(f"  - Comments stripped: {'Yes' if args.strip_comments else 'No'}", file=sys.stderr)


if __name__ == "__main__":
    main()
