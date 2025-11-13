"""
Verification module for verifying Lean proofs.
"""

import sys
import os
import logging
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Add parent directory to path to import RL utils
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from RL.utils.prover.lean.verifier import verify_single_proof, verify_lean4_file

logger = logging.getLogger(__name__)


def verify_proofs(
    generated_proofs: List[Dict],
    lake_path: str = None,
    lean_workspace: str = None,
    timeout: int = 200,
    batch_size: int = 40
) -> List[Dict]:
    """
    Verify generated proofs using the Lean verifier.
    
    Args:
        generated_proofs: List of dictionaries with 'statement' and 'proof' fields
        lake_path: Path to lake executable
        lean_workspace: Path to Lean workspace
        timeout: Timeout for verification in seconds
        batch_size: Batch size for verification
    
    Returns:
        List of dictionaries with verification results added
    """
    logger.info(f"Verifying {len(generated_proofs)} proofs...")
    
    results = []
    
    # Process in batches for efficiency
    for i in tqdm(range(0, len(generated_proofs), batch_size), desc="Verifying"):
        batch = generated_proofs[i:i+batch_size]
        
        # Extract codes and headers
        codes = []
        headers = []
        for proof_info in batch:
            code = proof_info['statement'] + '\n' + proof_info.get('proof', '')
            codes.append(code)
            headers.append(proof_info.get('header', None))
        
        # Verify batch
        try:
            batch_results = verify_lean4_file(
                codes=codes,
                headers=headers,
                lake_path=lake_path,
                lean_workspace=lean_workspace,
                allTactics=False,
                ast=False,
                premises=False,
                tactics=False
            )
            
            # Combine results
            for proof_info, result in zip(batch, batch_results):
                results.append({
                    **proof_info,
                    **result
                })
        except Exception as e:
            logger.error(f"Error verifying batch {i//batch_size}: {e}")
            # Add error results for this batch
            for proof_info in batch:
                results.append({
                    **proof_info,
                    'complete': False,
                    'pass': False,
                    'errors': [{'data': str(e)}],
                    'system_messages': str(e)
                })
    
    return results


def verify_proofs_single(
    generated_proofs: List[Dict],
    lake_path: str = None,
    lean_workspace: str = None,
    timeout: int = 200
) -> List[Dict]:
    """
    Verify proofs one at a time (slower but more reliable for debugging).
    
    Args:
        generated_proofs: List of dictionaries with 'statement' and 'proof' fields
        lake_path: Path to lake executable
        lean_workspace: Path to Lean workspace
        timeout: Timeout for verification in seconds
    
    Returns:
        List of dictionaries with verification results added
    """
    logger.info(f"Verifying {len(generated_proofs)} proofs (single mode)...")
    
    results = []
    
    for proof_info in tqdm(generated_proofs, desc="Verifying"):
        try:
            is_correct = verify_single_proof(
                statement=proof_info['statement'],
                proof=proof_info.get('proof', ''),
                header=proof_info.get('header', None),
                lake_path=lake_path,
                lean_workspace=lean_workspace,
                timeout=timeout
            )
            
            results.append({
                **proof_info,
                'complete': is_correct,
                'pass': is_correct
            })
        except Exception as e:
            logger.error(f"Error verifying proof {proof_info.get('lemma_id', 'unknown')}: {e}")
            results.append({
                **proof_info,
                'complete': False,
                'pass': False,
                'system_messages': str(e)
            })
    
    return results

