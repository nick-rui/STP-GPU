import sys
import os

# Add the RL directory to the Python path so we can import utils modules
# This matches how generate_and_test.py imports (from utils.model_utils import ...)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.RL_utils_gpu import SimpleLean4Verifier

def test_SimpleLean4Verifier():
    verifier = SimpleLean4Verifier()
    results = verifier.run([{'statement': 'theorem foo: true', 'proof': 'proof'}])
    print(results)

if __name__ == "__main__":
    test_SimpleLean4Verifier()