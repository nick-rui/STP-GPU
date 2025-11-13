#!/usr/bin/env python3
"""
Quick script to view individual proofs from generated_proofs_tests.jsonl.gz
Usage: python view_proofs.py [--file FILE] [--id PROOF_ID] [--index INDEX]
"""

import sys
import argparse
sys.path.insert(0, '.')
from utils.gcloud_utils import read_file

def main():
    parser = argparse.ArgumentParser(description='View individual proof')
    parser.add_argument('--file', type=str, default='results/test_run/generated_proofs_tests.jsonl.gz',
                       help='Input file (default: results/test_run/generated_proofs_tests.jsonl.gz)')
    parser.add_argument('--id', type=int, help='Lemma ID to view')
    parser.add_argument('--index', type=int, default=1, 
                       help='1-based index to view (default: 1, ignored if --id is specified)')
    
    args = parser.parse_args()
    
    # Read data
    data = read_file(args.file)
    if not data:
        print(f"Error: Could not read file: {args.file}")
        return
    
    # Find proof
    proof = None
    if args.id is not None:
        # Find by lemma_id
        for p in data:
            if p.get('lemma_id') == args.id:
                proof = p
                break
        if proof is None:
            print(f"Error: Proof with lemma_id {args.id} not found")
            available_ids = sorted(set(p.get('lemma_id', '?') for p in data))
            print(f"Available lemma_ids: {available_ids[:20]}{'...' if len(available_ids) > 20 else ''}")
            print(f"Total proofs: {len(data)}")
            return
    else:
        # Use index (1-based)
        if args.index < 1 or args.index > len(data):
            print(f"Error: Index {args.index} out of range (1-{len(data)})")
            return
        proof = data[args.index - 1]
    
    # Display proof
    print("=" * 80)
    proof_id = proof.get('lemma_id', '?')
    print(f"PROOF #{args.id if args.id is not None else args.index} (Lemma ID: {proof_id})")
    print("=" * 80)
    print()
    
    # Status
    status = "✓ VERIFIED" if proof.get('complete', False) else "✗ FAILED"
    print(f"Status: {status}")
    print()
    
    # Labels/Info
    if proof.get('label'):
        print(f"Labels: {', '.join(str(l) for l in proof.get('label', []))}")
    if proof.get('iter') is not None:
        print(f"Iteration: {proof.get('iter')}")
    print()
    
    # Statement
    print("-" * 80)
    print("STATEMENT:")
    print("-" * 80)
    statement = proof.get('statement', 'N/A')
    print(statement)
    print()
    
    # Header (if present)
    if proof.get('header'):
        print("-" * 80)
        print("HEADER:")
        print("-" * 80)
        print(proof.get('header'))
        print()
    
    # Proof
    print("-" * 80)
    print("PROOF:")
    print("-" * 80)
    proof_text = proof.get('proof', 'N/A')
    print(proof_text)
    print()
    
    # Errors (if any)
    if proof.get('errors'):
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
    
    # Sorries (if any)
    if proof.get('sorries'):
        print("-" * 80)
        print(f"SORRIES ({proof.get('sorries', 0)}):")
        print("-" * 80)
        print(f"  {proof.get('sorries')} 'sorry' statements found")
        print()
    
    # Verified code (if present)
    if proof.get('verified_code'):
        print("-" * 80)
        print("VERIFIED CODE:")
        print("-" * 80)
        print(proof.get('verified_code'))
        print()
    
    print("=" * 80)

if __name__ == '__main__':
    main()

