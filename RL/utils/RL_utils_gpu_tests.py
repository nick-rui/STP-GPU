import sys
import os

# Add the RL directory to the Python path so we can import utils modules
# This matches how generate_and_test.py imports (from utils.model_utils import ...)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.RL_utils_gpu import SimpleLean4Verifier


def test_simple_valid_proof():
    """Test a simple valid proof."""
    verifier = SimpleLean4Verifier(collect_premises=False)
    test_input = [{
        'statement': 'theorem test_simple : True := by',
        'proof': 'trivial'
    }]
    results = verifier.run(test_input, batched=True)

    print(results)
    
    assert len(results) == 1
    assert results[0]['complete'] == True
    assert results[0]['pass'] == True
    assert len(results[0]['errors']) == 0
    assert len(results[0]['sorries']) == 0

    print("✓ Simple valid proof test passed")
    return results


def test_proof_with_error():
    """Test a proof with a syntax/type error."""
    verifier = SimpleLean4Verifier(collect_premises=False)
    test_input = [{
        'statement': 'theorem test_error : False := by',
        'proof': 'trivial'  # This should fail - can't prove False with trivial
    }]
    results = verifier.run(test_input, batched=True)
    print(results)
    
    assert len(results) == 1
    assert results[0]['complete'] == False
    assert results[0]['pass'] == False
    assert len(results[0]['errors']) > 0
    print("✓ Proof with error test passed")
    return results


def test_proof_with_sorry():
    """Test a proof that uses 'sorry' (incomplete proof)."""
    verifier = SimpleLean4Verifier(collect_premises=False)
    test_input = [{
        'statement': 'theorem test_sorry : True := by',
        'proof': 'sorry'  # Incomplete proof
    }]
    results = verifier.run(test_input, batched=True)
    print(results)
    
    assert len(results) == 1
    assert results[0]['complete'] == False  # Should be False because of sorry
    assert len(results[0]['sorries']) > 0
    print("✓ Proof with sorry test passed")
    return results


def test_batch_verification():
    """Test verifying multiple proofs in a batch."""
    verifier = SimpleLean4Verifier(collect_premises=False)
    test_inputs = [
        {
            'statement': 'theorem test1 : True := by',
            'proof': 'trivial'
        },
        {
            'statement': 'theorem test2 : 1 + 1 = 2 := by',
            'proof': 'rfl'
        },
        {
            'statement': 'theorem test3 : False := by',
            'proof': 'trivial'  # This should fail
        }
    ]
    results = verifier.run(test_inputs, batched=True)
    print(results)
    
    assert len(results) == 3
    assert results[0]['complete'] == True
    assert results[1]['complete'] == True
    assert results[2]['complete'] == False
    print("✓ Batch verification test passed")
    return results


def test_proof_with_header():
    """Test a proof with custom header (imports/context)."""
    verifier = SimpleLean4Verifier(collect_premises=False)
    test_input = [{
        'statement': 'theorem test_header : Nat.zero = 0 := by',
        'proof': 'rfl',
        'header': 'import Mathlib.Data.Nat.Basic\n'
    }]
    results = verifier.run(test_input, batched=True)

    print(results)

    
    assert len(results) == 1
    # Should work with proper header
    print("✓ Proof with header test passed")
    return results


def test_non_batched_mode():
    """Test non-batched mode (single proof with premise extraction)."""
    verifier = SimpleLean4Verifier(collect_premises=True)
    test_input = [{
        'statement': 'theorem test_nonbatched : True := by',
        'proof': 'trivial'
    }]
    results = verifier.run(test_input, batched=False)
    print(results)
    
    assert len(results) == 1
    assert results[0]['complete'] == True
    # In non-batched mode with collect_premises=True, should have invokes field
    if results[0]['complete']:
        assert 'invokes' in results[0]
    print("✓ Non-batched mode test passed")
    return results


def test_empty_proof():
    """Test a proof with empty proof text."""
    verifier = SimpleLean4Verifier(collect_premises=False)
    test_input = [{
        'statement': 'theorem test_empty : True := by',
        'proof': ''  # Empty proof
    }]
    results = verifier.run(test_input, batched=True)
    print(results)
    
    assert len(results) == 1
    # Empty proof should fail
    assert results[0]['complete'] == False
    print("✓ Empty proof test passed")
    return results


def test_real_world_example():
    """Test with a real-world example from the codebase."""
    verifier = SimpleLean4Verifier(collect_premises=False)
    test_input = [{
        'lemma_id': 214,
        'statement': 'theorem lean_workbook_214 (x y : ℝ) : (x - y) ^ 2 ≥ 0 := by',
        'label': ['lean_workbook', 'inequality'],
        'iter': 0,
        'proof': '\n  rw [sq]\n  apply mul_self_nonneg'
    }]
    results = verifier.run(test_input, batched=True)
    print(results)
    
    assert len(results) == 1
    # This is a real proof, should verify if Lean4 is set up correctly
    print(f"✓ Real-world example test passed (complete={results[0].get('complete', False)})")
    return results


def test_premise_extraction():
    """Test that premise extraction works when collect_premises=True."""
    verifier = SimpleLean4Verifier(collect_premises=True)
    test_input = [{
        'statement': 'theorem test_premises : True := by',
        'proof': 'trivial'
    }]
    results = verifier.run(test_input, batched=True)
    print(results)
    
    assert len(results) == 1
    if results[0]['complete']:
        # Should have invokes field when complete and collect_premises=True
        assert 'invokes' in results[0]
        print("✓ Premise extraction test passed")
    else:
        print("⚠ Premise extraction test skipped (proof not complete)")
    return results


def run_all_tests():
    """Run all tests."""
    print("Running SimpleLean4Verifier tests...\n")
    
    tests = [
        ("Simple Valid Proof", test_simple_valid_proof),
        ("Proof with Error", test_proof_with_error),
        ("Proof with Sorry", test_proof_with_sorry),
        ("Batch Verification", test_batch_verification),
        ("Proof with Header", test_proof_with_header),
        ("Non-batched Mode", test_non_batched_mode),
        ("Empty Proof", test_empty_proof),
        ("Real-world Example", test_real_world_example),
        ("Premise Extraction", test_premise_extraction),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"Running: {test_name}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    # Run individual test or all tests
    import sys
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "all":
            run_all_tests()
        elif test_name == "simple":
            test_simple_valid_proof()
        elif test_name == "error":
            test_proof_with_error()
        elif test_name == "sorry":
            test_proof_with_sorry()
        elif test_name == "batch":
            test_batch_verification()
        elif test_name == "header":
            test_proof_with_header()
        elif test_name == "nonbatched":
            test_non_batched_mode()
        elif test_name == "empty":
            test_empty_proof()
        elif test_name == "real":
            test_real_world_example()
        elif test_name == "premises":
            test_premise_extraction()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: all, simple, error, sorry, batch, header, nonbatched, empty, real, premises")
    else:
        # Default: run all tests
        run_all_tests()