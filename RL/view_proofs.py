#!/usr/bin/env python3
"""
Quick script to view individual proofs from generated_proofs_tests.jsonl.gz
Usage: python view_proofs.py [--file FILE] [--id PROOF_ID] [--index INDEX] [--count N] [--show-*]
"""

import sys
import argparse
sys.path.insert(0, '.')
from utils.gcloud_utils import read_file

def display_proof(proof, display_flags, proof_num, proof_id_label):
    """Display a single proof with the specified attributes."""
    print("=" * 80)
    proof_id = proof.get('lemma_id', '?')
    print(f"PROOF #{proof_num} (Lemma ID: {proof_id})")
    print("=" * 80)
    print()
    
    # Status (always shown)
    status = "✓ VERIFIED" if proof.get('complete', False) else "✗ FAILED"
    print(f"Status: {status}")
    print()
    
    # Labels/Info (always shown)
    if proof.get('label'):
        print(f"Labels: {', '.join(str(l) for l in proof.get('label', []))}")
    if proof.get('iter') is not None:
        print(f"Iteration: {proof.get('iter')}")
    print()
    
    # Statement
    if display_flags.get('statement', True):
        print("-" * 80)
        print("STATEMENT:")
        print("-" * 80)
        statement = proof.get('statement', 'N/A')
        print(statement)
        print()
    
    # Header
    if display_flags.get('header', True) and proof.get('header'):
        print("-" * 80)
        print("HEADER:")
        print("-" * 80)
        print(proof.get('header'))
        print()

    # Sketch Proof
    if display_flags.get('sketch', True) and proof.get('proof_sketch'):
        print("-" * 80)
        print("PROOF SKETCH:")
        print("-" * 80)
        print(proof.get('proof_sketch'))
        print()
    
    # Proof
    if display_flags.get('proof', True):
        print("-" * 80)
        print("PROOF:")
        print("-" * 80)
        proof_text = proof.get('proof', 'N/A')
        print(proof_text)
        print()
    
    # Errors
    if display_flags.get('errors', True) and proof.get('errors'):
        print("-" * 80)
        print(f"ERRORS ({len(proof['errors'])}):")
        print("-" * 80)
        for i, err in enumerate(proof['errors'], 1):
            print(f"\nError {i}:")
            if isinstance(err, dict):
                print(f"  {err.get('data', err)}")
            else:
                print(f"  {err}")
        print()
    
    # Sorries
    if display_flags.get('sorries', True) and proof.get('sorries'):
        print("-" * 80)
        print(f"SORRIES ({proof.get('sorries', 0)}):")
        print("-" * 80)
        print(f"  {proof.get('sorries')} 'sorry' statements found")
        print()
    
    # Verified code
    if display_flags.get('verified_code', True) and proof.get('verified_code'):
        print("-" * 80)
        print("VERIFIED CODE:")
        print("-" * 80)
        print(proof.get('verified_code'))
        print()
    
    print("=" * 80)
    print()

def main():
    parser = argparse.ArgumentParser(description='View individual proof(s)')
    parser.add_argument('--file', type=str, default='results/generated_proofs_tests.jsonl.gz',
                       help='Input file (default: results/generated_proofs_tests.jsonl.gz)')
    parser.add_argument('--id', type=int, help='Lemma ID to view')
    parser.add_argument('--index', type=int, 
                       help='1-based index to view (ignored if --id or --count is specified)')
    parser.add_argument('--count', type=int, metavar='N',
                       help='View first N proofs (ignored if --id is specified)')
    
    # Display flags (only show flags - by default only statement and proof are shown)
    parser.add_argument('--show-header', action='store_true',
                       help='Show header')
    parser.add_argument('--show-sketch', action='store_true',
                       help='Show proof sketch')
    parser.add_argument('--show-verified-code', action='store_true',
                       help='Show verified code')
    parser.add_argument('--show-errors', action='store_true',
                       help='Show errors')
    parser.add_argument('--show-sorries', action='store_true',
                       help='Show sorries')
    
    args = parser.parse_args()
    
    # Read data
    data = read_file(args.file)
    if not data:
        print(f"Error: Could not read file: {args.file}")
        return
    
    # Determine display flags (by default only statement and proof are shown)
    display_flags = {
        'statement': True,  # Always shown by default
        'header': args.show_header,
        'sketch': args.show_sketch,
        'proof': True,  # Always shown by default
        'verified_code': args.show_verified_code,
        'errors': args.show_errors,
        'sorries': args.show_sorries,
    }
    
    # Determine which proofs to display
    proofs_to_display = []
    
    if args.id is not None:
        # Find by lemma_id
        for p in data:
            if p.get('lemma_id') == args.id:
                proofs_to_display.append((1, p, f"ID {args.id}"))
                break
        if not proofs_to_display:
            print(f"Error: Proof with lemma_id {args.id} not found")
            available_ids = sorted(set(p.get('lemma_id', '?') for p in data))
            print(f"Available lemma_ids: {available_ids[:20]}{'...' if len(available_ids) > 20 else ''}")
            print(f"Total proofs: {len(data)}")
            return
    elif args.count is not None:
        # View first N proofs
        if args.count < 1:
            print(f"Error: Count must be at least 1")
            return
        n = min(args.count, len(data))
        for i in range(n):
            proofs_to_display.append((i + 1, data[i], f"Index {i + 1}"))
    else:
        # Use index (1-based) or default to 1
        index = args.index if args.index is not None else 1
        if index < 1 or index > len(data):
            print(f"Error: Index {index} out of range (1-{len(data)})")
            return
        proof = data[index - 1]
        proofs_to_display.append((index, proof, f"Index {index}"))
    
    # Display proofs
    for proof_num, proof, proof_id_label in proofs_to_display:
        display_proof(proof, display_flags, proof_num, proof_id_label)

if __name__ == '__main__':
    main()

