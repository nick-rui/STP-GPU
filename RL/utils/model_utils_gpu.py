"""
GPU-optimized version of model_utils.py without Ray dependencies.

This module provides the same functionality as model_utils.py but without
Ray actors. Designed for single-GPU setups where Ray overhead is unnecessary.

Key differences from model_utils.py:
- No Ray remote actors or ActorPool
- Direct vLLM model calls
- Simplified checkpoint handling (no multi-worker coordination)
- Same utility functions for prompt generation and lemma management
"""

import os
import logging
from typing import Any, Dict
from transformers import AutoTokenizer
from utils.gcloud_utils import read_file, write_data

# Constants (same as model_utils.py)
START_STATEMENT = '<statement>'
START_LEMMA_STMT = '<easy theorem>'
START_THM = '<hard theorem>'
END_THM = '</hard theorem>'
INVOKED_LEMMA = '<lemma>'
PROVER_PROMPT = 'Complete the following Lean 4 code:\n\n```lean4\nimport Mathlib\nimport Aesop\nset_option maxHeartbeats 0\nopen BigOperators Real Nat Topology Rat\n'

__DEBUG__ = os.getenv("DEBUG", 'False').lower() in ('true', '1', 't')


def get_prompt(
    test_info: Dict,
    tokenizer: Any,
    max_length: int,
    invoke_type,
) -> str:
    """
    Generate a prompt string from test_info dictionary.
    
    This function creates the prompt that will be fed to the language model
    to generate a proof. It handles two types of prompts:
    1. Regular prompts: Just the theorem statement
    2. Conjecture prompts: Include a shared lemma and easy theorem
    
    Args:
        test_info: Dictionary containing lemma information with keys:
                  - 'statement': The theorem statement
                  - 'header': Optional header with imports/context
                  - 'shared_lemma_statement': For conjecture type
                  - 'proof': For conjecture type (easy proof)
        tokenizer: Transformers tokenizer instance
        max_length: Maximum token length for the prompt
        invoke_type: Type of prompt ('conjecture' or None)
        
    Returns:
        Formatted prompt string ready for the model, truncated to max_length tokens
    """
    if invoke_type == 'conjecture':
        shared_lemma = test_info['shared_lemma_statement']
        easy_theorem = test_info['statement'] + test_info['proof']
        prompt = f'Complete the following Lean 4 code:\n\n```lean4\n' \
            f'{INVOKED_LEMMA}\n{shared_lemma.strip()}\n{START_LEMMA_STMT}\n' \
            f'{easy_theorem.strip()}\n{START_THM}\n theorem'
    else:
        if ('header' in test_info) and (test_info['header'] is not None):
            prompt = 'Complete the following Lean 4 code:\n\n```lean4\n' + test_info["header"] + test_info["statement"].strip()
        else:
            prompt = f'{PROVER_PROMPT}\n{test_info["statement"].strip()}'

    return right_truncate(prompt, tokenizer, max_length)


def right_truncate(s: str, tokenizer: Any, max_tokens: int) -> str:
    """
    Truncate a string from the right to fit within max_tokens.
    
    This function tokenizes the string, truncates if necessary, and decodes
    back to text. It ensures the prompt fits within the model's context window.
    
    Args:
        s: Input string to truncate
        tokenizer: Transformers tokenizer instance
        max_tokens: Maximum number of tokens allowed
        
    Returns:
        Truncated string that fits within max_tokens
    """
    tokens = tokenizer.encode(
        s,
        return_tensors="pt",
        padding="longest",
        max_length=max_tokens,
        truncation=True,
    )[0]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def get_checkpoint_name(directory: str) -> str:
    """
    Get the checkpoint name from a directory.
    
    This function finds the checkpoint file in a directory. It expects
    exactly one file starting with "checkpoint".
    
    Args:
        directory: Directory path containing checkpoint files
        
    Returns:
        Full path to the checkpoint file
        
    Raises:
        AssertionError: If zero or multiple checkpoint files are found
    """
    checkpoint_files = []
    for file in os.listdir(directory):
        if file.startswith("checkpoint"):
            checkpoint_files.append(os.path.join(directory, file))
    assert len(checkpoint_files) == 1, f"Expected exactly 1 checkpoint file, found {len(checkpoint_files)}"
    return checkpoint_files[0]


def get_lemma_key(test_info: Dict) -> str:
    """
    Extract the lemma key (statement) from test_info.
    
    This function returns the statement field which is used as a unique
    identifier for lemmas in the lemma_mapping dictionary.
    
    Args:
        test_info: Dictionary containing lemma information
        
    Returns:
        The statement string (lemma key)
    """
    return test_info['statement']


def update_lemma_mapping(lemma_mapping: Dict, test_info: Dict) -> None:
    """
    Update the lemma_mapping dictionary with a test_info entry.
    
    This function adds or updates the mapping from lemma statement to
    lemma_id. It's used to maintain a consistent mapping across the
    codebase.
    
    Args:
        lemma_mapping: Dictionary mapping lemma statements to IDs
        test_info: Dictionary containing lemma information with 'statement' and 'lemma_id' keys
    """
    lemma_mapping[get_lemma_key(test_info)] = test_info['lemma_id']


def insert_lemma(lemma_mapping: Dict, test_info: Dict) -> None:
    """
    Insert a lemma into the lemma_mapping dictionary.
    
    This function ensures a lemma has a unique ID in the mapping. If the
    lemma statement doesn't exist in the mapping, it assigns a new ID
    (the current length of the mapping). It also updates test_info with
    the lemma_id.
    
    Args:
        lemma_mapping: Dictionary mapping lemma statements to IDs
        test_info: Dictionary containing lemma information. Will be modified
                  to include 'lemma_id' if not already present.
    """
    key = get_lemma_key(test_info)
    if key not in lemma_mapping:
        lemma_mapping[key] = len(lemma_mapping)
    test_info['lemma_id'] = lemma_mapping[key]

