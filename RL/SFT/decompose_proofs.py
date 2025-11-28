#!/usr/bin/env python3
"""
Decompose completed Lean proofs into sketch-like versions.

Supported input formats:
  1. JSONL: each line is a JSON object with a "proof" field
  2. JSON:  a single JSON array of such objects

For each proof, inner lemmas (have, let, etc.) that use `:= by` are
replaced with `:= by sorry`, while the main theorem's proof structure
is preserved.

Example transformation:
    theorem foo : P := by
      have h1 : Q := by
        tactic1
        tactic2
      have h2 : R := by
        tactic3
      exact h1

    becomes:

    theorem foo : P := by
      have h1 : Q := by sorry
      have h2 : R := by sorry
      exact h1

Usage:
    python SFT/decompose_proofs.py \
        --input /path/to/completed_proofs.jsonl \
        --output /path/to/decomposed_proofs.jsonl

The output preserves all original JSON fields, but the "proof" field
is replaced by its decomposed version. If the input is JSONL, the
output is JSONL; if the input is a JSON array, the output is a JSON
array.
"""

import argparse
import json
import re
import sys
from typing import Any, Dict


BY_LINE_PATTERN = re.compile(r":=\s*by\b.*")

# Patterns that indicate an inner lemma/definition (have, let, etc.)
INNER_LEMMA_KEYWORDS = re.compile(r"^\s*(have|let|suffices|show)\b")


def _indent_width(line: str) -> int:
    """Return the number of leading spaces in a line."""
    return len(line) - len(line.lstrip(" "))


def _is_main_theorem_line(line: str) -> bool:
    """Check if this line starts a main theorem/lemma/def declaration."""
    stripped = line.lstrip()
    return stripped.startswith(("theorem ", "lemma ", "def ", "example "))


def _is_inner_lemma_line(line: str) -> bool:
    """Check if this line is an inner lemma (have, let, suffices, show)."""
    return bool(INNER_LEMMA_KEYWORDS.match(line))


def decompose_proof(proof: str) -> str:
    """
    Replace inner `:= by ...` blocks with `:= by sorry`, preserving main proof structure.

    Inner lemmas are identified by keywords like `have`, `let`, `suffices`, `show`.
    The main theorem's `:= by` block is preserved, only nested blocks are replaced.
    """
    lines = proof.splitlines()
    new_lines = []
    i = 0
    main_theorem_indent = None  # Track the indent of the main theorem's := by

    while i < len(lines):
        line = lines[i]

        # Check if this line contains `:= by`
        if ":= by" not in line:
            new_lines.append(line)
            i += 1
            continue

        current_indent = _indent_width(line)

        # Determine if this is the main theorem or an inner lemma
        is_main = _is_main_theorem_line(line)
        is_inner = _is_inner_lemma_line(line)

        # If it's the main theorem, just record its indent and keep the line
        if is_main:
            main_theorem_indent = current_indent
            new_lines.append(line)
            i += 1
            continue

        # If it's an inner lemma (have, let, etc.), replace with sorry
        if is_inner or (main_theorem_indent is not None and current_indent > main_theorem_indent):
        # Replace from := by ... up to end-of-line with := by sorry
        replaced_line = BY_LINE_PATTERN.sub(":= by sorry", line)
        new_lines.append(replaced_line)
        i += 1

            # Skip all lines that are part of this inner block
            base_indent = current_indent
        while i < len(lines):
            next_line = lines[i]
            if not next_line.strip():
                    # Blank line - check if next non-blank line is still in block
                i += 1
                continue
            next_indent = _indent_width(next_line)
            if next_indent > base_indent:
                    # More indented = part of the block, skip it
                i += 1
                continue
                # We've reached a line that is not part of the block
            break
            continue

        # Default: keep the line as-is (shouldn't normally reach here)
        new_lines.append(line)
        i += 1

    return "\n".join(new_lines)


def _process_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of obj with its 'proof' field decomposed, if present."""
    proof = obj.get("proof")
    if isinstance(proof, str):
        obj["proof"] = decompose_proof(proof)
    return obj


def process_jsonl(in_fp, out_fp) -> None:
    """Read JSONL from in_fp, decompose proofs, and write JSONL to out_fp."""
    for line in in_fp:
        line = line.strip()
        if not line:
            continue
        try:
            obj: Dict[str, Any] = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"Warning: skipping invalid JSONL line: {exc}", file=sys.stderr)
            continue

        obj = _process_obj(obj)
        out_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def process_json_array(in_fp, out_fp) -> None:
    """Read a JSON array from in_fp, decompose proofs, and write a JSON array."""
    try:
        data = json.load(in_fp)
    except json.JSONDecodeError as exc:
        print(f"Error: input JSON is invalid: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print("Error: expected a JSON array at top level for .json input.", file=sys.stderr)
        sys.exit(1)

    processed = [_process_obj(dict(obj)) for obj in data]
    json.dump(processed, out_fp, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose completed Lean proofs by replacing `:= by ...` "
            "blocks with `:= by sorry`. Supports JSONL or JSON array."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL or JSON file with completed Lean proofs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output file for decomposed proofs (JSONL or JSON).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        with open(args.input, "r", encoding="utf-8") as in_fp, open(
            args.output, "w", encoding="utf-8"
        ) as out_fp:
            # Peek first non-whitespace character to decide format.
            start = in_fp.read(1024)
            in_fp.seek(0)
            first_non_ws = next((ch for ch in start if not ch.isspace()), "")
            if first_non_ws == "[":
                # JSON array
                process_json_array(in_fp, out_fp)
            else:
                # Assume JSONL
                process_jsonl(in_fp, out_fp)
    except OSError as exc:
        print(f"Error opening files: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


