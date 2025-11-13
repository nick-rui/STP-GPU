#!/usr/bin/env python3
"""
Quick script to view test results from generated_proofs_tests.jsonl.gz
Usage: python view_results.py [--summary|--json|--verified|--failed]
"""

import sys
import argparse
sys.path.insert(0, '.')
from utils.gcloud_utils import read_file
import json

def main():
    parser = argparse.ArgumentParser(description='View test results')
    parser.add_argument('--summary', action='store_true', help='Show summary only')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--verified', action='store_true', help='Show only verified proofs')
    parser.add_argument('--failed', action='store_true', help='Show only failed proofs')
    parser.add_argument('--file', type=str, default='results/test_run/generated_proofs_tests.jsonl.gz', 
                       help='Input file (default: results/generated_proofs_tests.jsonl.gz)')
    
    args = parser.parse_args()
    
    data = read_file(args.file)
    
    if args.summary:
        verified = sum(1 for p in data if p.get('complete', False))
        failed = len(data) - verified
        print(f"Total: {len(data)}")
        print(f"✓ Verified: {verified} ({100*verified/len(data):.1f}%)")
        print(f"✗ Failed: {failed} ({100*failed/len(data):.1f}%)")
        return
    
    if args.verified:
        data = [p for p in data if p.get('complete', False)]
    elif args.failed:
        data = [p for p in data if not p.get('complete', False)]
    
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        for i, proof in enumerate(data, 1):
            status = "✓" if proof.get('complete', False) else "✗"
            print(f"\n{status} Proof {i} (Lemma {proof.get('lemma_id', '?')})")
            print(f"  Statement: {proof.get('statement', 'N/A')[:80]}...")
            if proof.get('errors'):
                print(f"  Errors: {len(proof['errors'])}")
                for err in proof['errors'][:2]:
                    print(f"    - {err.get('data', 'N/A')[:100]}")

if __name__ == '__main__':
    main()
