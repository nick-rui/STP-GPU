#!/usr/bin/env python3
"""
Convert JSON/JSONL zip files to readable text format.
Usage: python convert_results_to_text.py <input_file.jsonl.gz>
"""

import sys
import os
import json
import pgzip
import argparse

def read_jsonl_gz(filepath):
    """Read a gzipped JSONL file and return list of JSON objects."""
    data = []
    try:
        with pgzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    return data

def format_proof(proof, index):
    """Format a single proof entry as text."""
    lines = []
    
    # Header
    lines.append("=" * 80)
    proof_id = proof.get('lemma_id', '?')
    status = "✓ VERIFIED" if proof.get('complete', False) else "✗ FAILED"
    lines.append(f"PROOF #{index} (Lemma ID: {proof_id}) - {status}")
    lines.append("=" * 80)
    lines.append("")
    
    # Metadata
    if proof.get('label'):
        labels = ', '.join(str(l) for l in proof.get('label', []))
        lines.append(f"Labels: {labels}")
    if proof.get('iter') is not None:
        lines.append(f"Iteration: {proof.get('iter')}")
    if proof.get('verify_time') is not None:
        lines.append(f"Verify Time: {proof.get('verify_time')}s")
    lines.append("")
    
    # Statement
    lines.append("-" * 80)
    lines.append("STATEMENT:")
    lines.append("-" * 80)
    statement = proof.get('statement', 'N/A')
    lines.append(statement)
    lines.append("")
    
    # Header (if present)
    if proof.get('header'):
        lines.append("-" * 80)
        lines.append("HEADER:")
        lines.append("-" * 80)
        lines.append(proof.get('header'))
        lines.append("")
    
    # Proof
    lines.append("-" * 80)
    lines.append("PROOF:")
    lines.append("-" * 80)
    proof_text = proof.get('proof', 'N/A')
    lines.append(proof_text)
    lines.append("")
    
    # Errors (if any)
    if proof.get('errors'):
        lines.append("-" * 80)
        lines.append(f"ERRORS ({len(proof['errors'])}):")
        lines.append("-" * 80)
        for i, err in enumerate(proof['errors'], 1):
            lines.append(f"\nError {i}:")
            if isinstance(err, dict):
                error_data = err.get('data', str(err))
                if isinstance(error_data, str):
                    lines.append(f"  {error_data}")
                else:
                    lines.append(f"  {json.dumps(error_data, indent=2)}")
            else:
                lines.append(f"  {err}")
        lines.append("")
    
    # Sorries (if any)
    if proof.get('sorries'):
        lines.append("-" * 80)
        lines.append(f"SORRIES ({proof.get('sorries', 0)}):")
        lines.append("-" * 80)
        lines.append(f"  {proof.get('sorries')} 'sorry' statements found")
        lines.append("")
    
    # Verified code (if present)
    if proof.get('verified_code'):
        lines.append("-" * 80)
        lines.append("VERIFIED CODE:")
        lines.append("-" * 80)
        lines.append(proof.get('verified_code'))
        lines.append("")
    
    lines.append("=" * 80)
    lines.append("")
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description='Convert JSON/JSONL zip files to readable text')
    parser.add_argument('input_file', type=str, help='Input JSON/JSONL .gz file')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output text file (default: same name as input with .txt extension)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        # Same directory as input, replace extension with .txt
        input_dir = os.path.dirname(args.input_file)
        input_basename = os.path.basename(args.input_file)
        # Remove .gz and any other extensions, add .txt
        base_name = input_basename.replace('.gz', '').replace('.jsonl', '').replace('.json', '')
        output_file = os.path.join(input_dir, base_name + '.txt')
    
    print(f"Reading: {args.input_file}")
    
    # Read the data
    data = read_jsonl_gz(args.input_file)
    if data is None:
        sys.exit(1)
    
    print(f"Found {len(data)} proofs")
    
    # Generate summary statistics
    verified = sum(1 for p in data if p.get('complete', False))
    failed = len(data) - verified
    
    # Write to output file
    print(f"Writing to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header with summary
        f.write("=" * 80 + "\n")
        f.write("PROOF RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Proofs: {len(data)}\n")
        f.write(f"✓ Verified: {verified} ({100*verified/len(data):.1f}%)\n")
        f.write(f"✗ Failed: {failed} ({100*failed/len(data):.1f}%)\n")
        f.write("=" * 80 + "\n\n")
        
        # Write each proof
        for i, proof in enumerate(data, 1):
            proof_text = format_proof(proof, i)
            f.write(proof_text)
    
    print(f"Done! Output written to: {output_file}")

if __name__ == '__main__':
    main()

