#!/usr/bin/env python3
"""
Decompose completed Lean proofs into sketch-like versions.

Supported input formats:
  1. JSONL: each line is a JSON object with a "proof" field
  2. JSON:  a single JSON array of such objects

For each proof, every top-level occurrence of

    := by <tactics...>

is replaced by

    := by sorry

This is a simple, indentation-based transformation that treats `:= by`
as the start of a block and collapses the entire block to a single line.

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


def _indent_width(line: str) -> int:
    """Return the number of leading spaces in a line."""
    return len(line) - len(line.lstrip(" "))


def decompose_proof(proof: str) -> str:
    """
    Replace each top-level `:= by ...` block with a single `:= by sorry` line.

    We treat a `:= by` block as:
      - the line containing `:= by`, plus
      - all immediately following lines that are more indented than the
        `:= by` line (or blank).

    All of those lines are collapsed into a single line where everything
    from `:= by` to end-of-line is replaced with `:= by sorry`.
    """
    lines = proof.splitlines()
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if ":= by" not in line:
            new_lines.append(line)
            i += 1
            continue

        # We found the start of a `:= by` block.
        base_indent = _indent_width(line)
        # Replace from := by ... up to end-of-line with := by sorry
        replaced_line = BY_LINE_PATTERN.sub(":= by sorry", line)
        new_lines.append(replaced_line)
        i += 1

        # Skip all lines that are part of this block: blank lines or
        # lines indented more than the `:= by` line.
        while i < len(lines):
            next_line = lines[i]
            if not next_line.strip():
                # Treat blank lines as part of the block.
                i += 1
                continue
            next_indent = _indent_width(next_line)
            if next_indent > base_indent:
                i += 1
                continue
            # We've reached a line that is not part of the block.
            break

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


