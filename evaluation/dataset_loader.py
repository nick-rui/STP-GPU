"""
Dataset loader for LeanWorkbook dataset.
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def load_leanworkbook_subset(
    dataset_path: str,
    max_problems: Optional[int] = None,
    seed: int = 0
) -> List[Dict]:
    """
    Load a subset of problems from the LeanWorkbook dataset.
    
    Args:
        dataset_path: Path to the LeanWorkbook JSON file
        max_problems: Maximum number of problems to load (None for all)
        seed: Random seed for sampling
    
    Returns:
        List of problem dictionaries with keys: 'statement', 'header', 'lemma_id', etc.
    """
    # Resolve relative paths
    if not Path(dataset_path).is_absolute():
        # Try relative to repo root
        repo_root = Path(__file__).parent.parent
        dataset_path = repo_root / dataset_path
        if not dataset_path.exists():
            # Try as absolute path
            dataset_path = Path(dataset_path).resolve()
    
    logger.info(f"Loading dataset from {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_dataset = json.load(f)
    
    logger.info(f"Loaded {len(raw_dataset)} problems from dataset")
    
    # Format problems similar to how STP does it
    formatted_problems = []
    for idx, raw in enumerate(raw_dataset):
        # Extract statement (remove 'sorry' if present)
        statement = raw['formal_statement'].rsplit('sorry', 1)[0].strip()
        
        problem = {
            'lemma_id': idx,
            'statement': statement,
            'label': [raw.get('split', 'unknown')] + (raw.get('tags', None) or []),
            'header': raw.get('header', None),
            'name': raw.get('name', f'problem_{idx}'),
        }
        formatted_problems.append(problem)
    
    # Sample subset if requested
    if max_problems is not None and max_problems < len(formatted_problems):
        random.seed(seed)
        formatted_problems = random.sample(formatted_problems, max_problems)
        logger.info(f"Sampled {len(formatted_problems)} problems")
    
    return formatted_problems

