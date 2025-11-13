# How to View Test Results

## Quick Commands

### Option 1: Use the Python script (recommended)
```bash
cd ~/STP/RL
source /home/nickrui/cs229/bin/activate  # if needed

# Quick summary:
python view_results.py --summary

# View verified proofs only:
python view_results.py --verified

# View failed proofs only:
python view_results.py --failed

# Output as JSON:
python view_results.py --json | less
```

### Option 2: Use the shell script
```bash
cd ~/STP/RL
./view_results.sh
# This creates:
#   - results/generated_proofs_tests.json (open in IDE)
#   - results/verification_summary.txt (readable summary)
```

### Option 3: Manual commands
```bash
cd ~/STP/RL
source /home/nickrui/cs229/bin/activate

# Extract to JSON:
python3 -c "import sys; sys.path.insert(0, '.'); from utils.gcloud_utils import read_file; import json; json.dump(read_file('results/generated_proofs_tests.jsonl.gz'), open('results/generated_proofs_tests.json', 'w'), indent=2)"

# Then open results/generated_proofs_tests.json in your IDE
```

## Files Created

After running `view_results.sh` or extracting manually:
- `results/generated_proofs_tests.json` - Full JSON (open in IDE)
- `results/verification_summary.txt` - Human-readable summary

## Quick Summary Command
```bash
cd ~/STP/RL && python view_results.py --summary
```
