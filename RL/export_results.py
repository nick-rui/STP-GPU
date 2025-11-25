#!/usr/bin/env python3
"""
Export detailed proof results from a JSONL(.gz) file to a text file.

This script draws from the behavior of `view_proofs.py` and `view_results.py`:
- It uses the same JSONL(.gz) reading logic.
- It writes a summary of total / verified / failed proofs similar to `view_results.py --summary`.
- For each proof, it writes a detailed block containing the same information
  that `view_proofs.py` shows (statement, header, sketch, proof, errors,
  sorries, verified code, and basic metadata).

Usage example:
  python export_results.py \
      --input results/generated_proofs_tests.jsonl.gz \
      --output results/generated_proofs_tests_export.txt

You can optionally restrict the export to only verified or only failed proofs
using `--verified` or `--failed`.
"""

import argparse
import gzip
import json
import os
import sys
from typing import Any, Dict, List, Optional


Proof = Dict[str, Any]


def read_file(filepath: str) -> List[Proof]:
    """Read a .jsonl or .jsonl.gz file and return list of records.

    This mirrors the `read_file` helper used in `view_results.py` and
    `view_proofs.py`, but is local to this script so it can be used
    independently.
    """
    open_fn = gzip.open if filepath.endswith(".gz") else open
    mode = "rt" if filepath.endswith(".gz") else "r"

    data: List[Proof] = []
    with open_fn(filepath, mode, encoding="utf-8") as f:  # type: ignore[arg-type]
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def compute_summary(proofs: List[Proof]) -> Dict[str, Any]:
    """Compute simple summary statistics for a list of proofs."""
    total = len(proofs)
    verified = sum(1 for proof in proofs if proof.get("complete", False))
    failed = total - verified

    if total > 0:
        verified_pct = 100.0 * verified / total
        failed_pct = 100.0 * failed / total
    else:
        verified_pct = 0.0
        failed_pct = 0.0

    return {
        "total": total,
        "verified": verified,
        "failed": failed,
        "verified_pct": verified_pct,
        "failed_pct": failed_pct,
    }


def format_summary_block(summary: Dict[str, Any]) -> str:
    """Format the overall summary block, similar to `view_results.py --summary`."""
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("PROOF RESULTS SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total Proofs: {summary['total']}")
    lines.append(
        f"✓ Verified: {summary['verified']} ({summary['verified_pct']:.1f}%)"
    )
    lines.append(f"✗ Failed: {summary['failed']} ({summary['failed_pct']:.1f}%)")
    lines.append("=" * 80)
    lines.append("")
    return "\n".join(lines)


def format_proof_block(proof: Proof, proof_index: int) -> str:
    """Format a single proof entry as a human-readable text block.

    This is effectively a string-based version of `display_proof` from
    `view_proofs.py`, with an additional short "result overview" section that
    mirrors what `view_results.py` prints for each proof.
    """
    lines: List[str] = []

    # Top header
    lines.append("=" * 80)
    proof_id = proof.get("lemma_id", "?")
    status_text = "✓ VERIFIED" if proof.get("complete", False) else "✗ FAILED"
    lines.append(f"PROOF #{proof_index} (Lemma ID: {proof_id})")
    lines.append("=" * 80)
    lines.append("")

    # Status (always shown)
    lines.append(f"Status: {status_text}")
    lines.append("")

    # Labels / metadata (always shown if present)
    labels = proof.get("label")
    if labels:
        labels_str = ", ".join(str(label) for label in labels)
        lines.append(f"Labels: {labels_str}")
    if proof.get("iter") is not None:
        lines.append(f"Iteration: {proof.get('iter')}")
    if proof.get("verify_time") is not None:
        lines.append(f"Verify Time: {proof.get('verify_time')}s")
    lines.append("")

    # Statement
    lines.append("-" * 80)
    lines.append("STATEMENT:")
    lines.append("-" * 80)
    statement = proof.get("statement", "N/A")
    lines.append(str(statement))
    lines.append("")

    # Header (if present)
    header = proof.get("header")
    if header:
        lines.append("-" * 80)
        lines.append("HEADER:")
        lines.append("-" * 80)
        lines.append(str(header))
        lines.append("")

    # Sketch proof (if present)
    proof_sketch = proof.get("proof_sketch")
    if proof_sketch:
        lines.append("-" * 80)
        lines.append("PROOF SKETCH:")
        lines.append("-" * 80)
        lines.append(str(proof_sketch))
        lines.append("")

    # Full proof text
    lines.append("-" * 80)
    lines.append("PROOF:")
    lines.append("-" * 80)
    proof_text = proof.get("proof", "N/A")
    lines.append(str(proof_text))
    lines.append("")

    # Errors (if any)
    errors = proof.get("errors")
    if errors:
        lines.append("-" * 80)
        lines.append(f"ERRORS ({len(errors)}):")
        lines.append("-" * 80)
        for error_index, error in enumerate(errors, 1):
            lines.append("")
            lines.append(f"Error {error_index}:")
            if isinstance(error, dict):
                error_data = error.get("data", error)
                if isinstance(error_data, str):
                    lines.append(f"  {error_data}")
                else:
                    lines.append(
                        "  "
                        + json.dumps(
                            error_data,
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
            else:
                lines.append(f"  {error}")
        lines.append("")

    # Sorries (if any)
    sorries = proof.get("sorries")
    if sorries:
        lines.append("-" * 80)
        lines.append(f"SORRIES ({sorries}):")
        lines.append("-" * 80)
        lines.append(f"  {sorries} 'sorry' statements found")
        lines.append("")

    # Verified code (if present)
    verified_code = proof.get("verified_code")
    if verified_code:
        lines.append("-" * 80)
        lines.append("VERIFIED CODE:")
        lines.append("-" * 80)
        lines.append(str(verified_code))
        lines.append("")

    # Result overview (similar to `view_results.py` per-proof output)
    lines.append("-" * 80)
    lines.append("RESULT OVERVIEW:")
    lines.append("-" * 80)
    status_symbol = "✓" if proof.get("complete", False) else "✗"
    lines.append(f"{status_symbol} Proof {proof_index} (Lemma {proof_id})")

    statement_str = str(statement)
    if len(statement_str) > 80:
        short_statement = statement_str[:80] + "..."
    else:
        short_statement = statement_str
    lines.append(f"  Statement: {short_statement}")

    if errors:
        lines.append(f"  Errors: {len(errors)}")
        for error in errors[:2]:
            if isinstance(error, dict):
                error_text: Any = error.get("data", "N/A")
            else:
                error_text = str(error)
            if not isinstance(error_text, str):
                error_text = json.dumps(error_text, ensure_ascii=False)
            if len(error_text) > 100:
                error_text = error_text[:100] + "..."
            lines.append(f"    - {error_text}")

    lines.append("=" * 80)
    lines.append("")

    return "\n".join(lines)


def build_output_path(input_path: str, output_arg: Optional[str]) -> str:
    """Determine the output path, defaulting to `<input_basename>.txt`.

    If `output_arg` is provided, it is returned as-is. Otherwise, the output
    file is placed in the same directory as the input, with `.gz`, `.jsonl`,
    and `.json` extensions stripped and `.txt` appended.
    """
    if output_arg:
        return output_arg

    input_dir = os.path.dirname(input_path)
    input_basename = os.path.basename(input_path)

    base_name = input_basename
    for ext in (".gz", ".jsonl", ".json"):
        if base_name.endswith(ext):
            base_name = base_name[: -len(ext)]
    if not base_name:
        base_name = "exported_results"

    return os.path.join(input_dir, base_name + ".txt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export detailed proof results from a JSONL(.gz) file to a text file "
            "(combining the information from view_proofs.py and view_results.py)."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input .jsonl or .jsonl.gz results file",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=False,
        help=(
            "Output text file (default: same directory as input, with "
            ".txt extension)"
        ),
    )
    parser.add_argument(
        "--verified",
        action="store_true",
        help="Export only proofs with complete=True (verified proofs)",
    )
    parser.add_argument(
        "--failed",
        action="store_true",
        help="Export only proofs with complete=False (failed proofs)",
    )

    args = parser.parse_args()

    if args.verified and args.failed:
        print("Error: --verified and --failed are mutually exclusive", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Read all proofs
    proofs = read_file(args.input)
    if not proofs:
        print(f"Warning: No proofs found in '{args.input}'", file=sys.stderr)

    # Optionally filter by verification status
    if args.verified:
        proofs = [proof for proof in proofs if proof.get("complete", False)]
    elif args.failed:
        proofs = [proof for proof in proofs if not proof.get("complete", False)]

    # Compute summary over the (possibly filtered) set of proofs
    summary = compute_summary(proofs)

    # Determine output path
    output_path = build_output_path(args.input, args.output)

    # Write summary + all formatted proof blocks
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write(format_summary_block(summary))
        for index, proof in enumerate(proofs, 1):
            out_f.write(format_proof_block(proof, index))

    print(
        f"Exported {summary['total']} proofs to '{output_path}' "
        f"(verified: {summary['verified']}, failed: {summary['failed']})"
    )


if __name__ == "__main__":
    main()



