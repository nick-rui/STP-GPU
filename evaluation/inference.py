"""
Inference module for running model inference on Lean problems using HuggingFace transformers.
"""

import os
import logging
from typing import List, Dict, Optional
from tqdm import tqdm

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    raise ImportError("transformers library is required. Install it with: pip install transformers torch")

logger = logging.getLogger(__name__)


# Prompt template from STP
PROVER_PROMPT = 'Complete the following Lean 4 code:\n\n```lean4\nimport Mathlib\nimport Aesop\nset_option maxHeartbeats 0\nopen BigOperators Real Nat Topology Rat\n'


def format_prompt(test_info: Dict) -> str:
    """
    Format a problem into a prompt for the model.
    
    Args:
        test_info: Dictionary with 'statement' and optionally 'header'
    
    Returns:
        Formatted prompt string
    """
    if test_info.get('header') is not None:
        prompt = 'Complete the following Lean 4 code:\n\n```lean4\n' + test_info['header'] + test_info['statement'].strip()
    else:
        prompt = f'{PROVER_PROMPT}\n{test_info["statement"].strip()}'
    
    return prompt


def right_truncate(s: str, tokenizer, max_tokens: int) -> str:
    """Truncate prompt from the right to fit within max_tokens."""
    tokens = tokenizer.encode(s, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return s
    tokens = tokens[-max_tokens:]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def run_inference(
    problems: List[Dict],
    model_name: str,
    tokenizer_path: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    batch_size: int = 8,
    device: str = "cuda"
) -> List[Dict]:
    """
    Run inference on problems using HuggingFace transformers.
    
    Args:
        problems: List of problem dictionaries
        model_name: Model name or path (HuggingFace model identifier)
        tokenizer_path: Tokenizer path (defaults to model_name)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        batch_size: Batch size for inference
        device: Device to use (cuda/cpu)
    
    Returns:
        List of dictionaries with 'proof' field added
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required. Install it with: pip install transformers torch")
    
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path or model_name,
        trust_remote_code=True
    )
    tokenizer.truncation_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" and torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" and torch.cuda.is_available() else None
    )
    model.eval()
    
    if device == "cpu" or not torch.cuda.is_available():
        model = model.to("cpu")
        device = "cpu"
    
    results = []
    
    logger.info(f"Generating proofs for {len(problems)} problems...")
    for i in tqdm(range(0, len(problems), batch_size), desc="Generating proofs"):
        batch = problems[i:i+batch_size]
        
        # Format prompts
        prompts = []
        for problem in batch:
            prompt = format_prompt(problem)
            prompt = right_truncate(prompt, tokenizer, max_tokens=1024)
            prompts.append(prompt)
        
        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        if device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract generated parts (remove prompt)
        for problem, prompt, full_text in zip(batch, prompts, generated_texts):
            # Remove prompt from generated text
            if full_text.startswith(prompt):
                generated_text = full_text[len(prompt):].split('\n```', 1)[0]
            else:
                generated_text = full_text.split('\n```', 1)[0]
            
            results.append({
                **problem,
                'proof': generated_text,
                'prompt': prompt
            })
    
    return results

