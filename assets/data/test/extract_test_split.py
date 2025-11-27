"""Extract test split entries from a JSONL dataset.

Given an input JSONL file (one JSON object per line) with a "split" field
like "Proofnet valid" / "Proofnet test", this script writes out only
the entries that correspond to the test split.

Example:
    python extract_test_split.py \
        --input proofnet.jsonl \
        --output proofnet_test_only.jsonl
"""

import argparse
import json
from typing import TextIO, Iterable, Dict, Any


def extract_test_entries(
    fin: TextIO,
    fout: TextIO,
    split_substring: str = "test",
) -> tuple[int, int]:
    """Stream over input JSONL and write only test-split lines to output.

    A line is kept if its 'split' field contains `split_substring`
    case-insensitively (e.g. "Proofnet test").

    Returns:
        kept, total: number of kept lines and total lines read.
    """
    kept = 0
    total = 0
    split_substring = split_substring.lower()

    for line in fin:
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Skip malformed lines rather than crashing
            continue

        split_val = str(obj.get("split", "")).lower()
        if split_substring in split_val:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    return kept, total


def extract_test_entries_from_objects(
    objs: Iterable[Dict[str, Any]],
    fout: TextIO,
    split_substring: str = "test",
) -> tuple[int, int]:
    """Filter an in-memory iterable of JSON objects and write test entries as JSONL."""
    kept = 0
    total = 0
    split_substring = split_substring.lower()

    for obj in objs:
        total += 1
        split_val = str(obj.get("split", "")).lower()
        if split_substring in split_val:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    return kept, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract test-split entries from a JSONL dataset."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input JSONL file (e.g., proofnet.jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output JSONL file with only test entries",
    )
    parser.add_argument(
        "--split_substring",
        type=str,
        default="test",
        help="Substring to match in the 'split' field (case-insensitive). "
        "Defaults to 'test' (matches values like 'Proofnet test').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.output, "w", encoding="utf-8") as fout:
        if args.input.endswith(".jsonl"):
            # JSONL -> JSONL (filtered)
            with open(args.input, "r", encoding="utf-8") as fin:
                kept, total = extract_test_entries(
                    fin,
                    fout,
                    split_substring=args.split_substring,
                )
        else:
            # Assume standard JSON file (list of objects). Convert to JSONL.
            with open(args.input, "r", encoding="utf-8") as fin:
                data = json.load(fin)
            if isinstance(data, list):
                objs = data
            else:
                raise ValueError(
                    f"Unsupported JSON structure in {args.input}: expected a list at top level."
                )
            kept, total = extract_test_entries_from_objects(
                objs,
                fout,
                split_substring=args.split_substring,
            )

    print(
        f"Extracted {kept} test entries out of {total} total lines "
        f"from {args.input} into {args.output}"
    )


if __name__ == "__main__":
    main()

