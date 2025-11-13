#!/bin/bash
# Script to view test results from generated_proofs_tests.jsonl.gz

set -e

RESULTS_FILE="results/generated_proofs_tests.jsonl.gz"
JSON_FILE="results/generated_proofs_tests.json"
SUMMARY_FILE="results/verification_summary.txt"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: $RESULTS_FILE not found"
    exit 1
fi

# Activate venv if needed
if [ -f "/home/nickrui/cs229/bin/activate" ]; then
    source /home/nickrui/cs229/bin/activate
fi

# Extract to JSON
python3 << PYEOF
import sys
sys.path.insert(0, '.')
from utils.gcloud_utils import read_file
import json

data = read_file('$RESULTS_FILE')

# Create JSON file
with open('$JSON_FILE', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# Create summary
verified = [p for p in data if p.get('complete', False)]
failed = [p for p in data if not p.get('complete', False)]

with open('$SUMMARY_FILE', 'w') as f:
    f.write("=" * 80 + "\\n")
    f.write("VERIFICATION RESULTS SUMMARY\\n")
    f.write("=" * 80 + "\\n\\n")
    f.write(f"Total Proofs: {len(data)}\\n")
    f.write(f"✓ Verified: {len(verified)} ({100*len(verified)/len(data):.1f}%)\\n")
    f.write(f"✗ Failed: {len(failed)} ({100*len(failed)/len(data):.1f}%)\\n\\n")
    
    f.write("=" * 80 + "\\n")
    f.write("VERIFIED PROOFS\\n")
    f.write("=" * 80 + "\\n\\n")
    
    for i, proof in enumerate(verified, 1):
        f.write(f"Proof {i} (Lemma ID: {proof.get('lemma_id', '?')})\\n")
        f.write(f"Statement: {proof.get('statement', 'N/A')[:100]}...\\n")
        f.write(f"Proof: {proof.get('proof', 'N/A')[:200]}...\\n\\n")
    
    f.write("=" * 80 + "\\n")
    f.write("FAILED PROOFS\\n")
    f.write("=" * 80 + "\\n\\n")
    
    for i, proof in enumerate(failed, 1):
        f.write(f"Proof {i} (Lemma ID: {proof.get('lemma_id', '?')})\\n")
        f.write(f"Statement: {proof.get('statement', 'N/A')[:100]}...\\n")
        errors = proof.get('errors', [])
        if errors:
            f.write(f"Errors ({len(errors)}):\\n")
            for err in errors[:3]:
                f.write(f"  - {err.get('data', 'N/A')[:150]}\\n")
        if proof.get('system_errors'):
            f.write(f"System errors: {proof.get('system_errors')}\\n")
        f.write("\\n")

print(f"✓ Created: $JSON_FILE")
print(f"✓ Created: $SUMMARY_FILE")
print(f"\\nTotal: {len(data)}, Verified: {len(verified)}, Failed: {len(failed)}")
PYEOF

echo ""
echo "Files created:"
echo "  📄 $JSON_FILE (open in IDE)"
echo "  📄 $SUMMARY_FILE (readable summary)"
echo ""
echo "View summary: cat $SUMMARY_FILE"
echo "View JSON: cat $JSON_FILE | less"
