"""
GPU-optimized version of RL_utils.py without Ray dependencies.

This module provides the same functionality as RL_utils.py but uses direct
model inference and verification calls instead of Ray actors. Designed for
single-GPU setups where Ray overhead is unnecessary.

Key differences from RL_utils.py:
- Direct vLLM model calls instead of Ray actors
- Direct Lean4 verification instead of Ray workers
- Simplified parallelization using threading/processes
- Same function signatures for compatibility

Usage:
    To use this module instead of RL_utils.py, modify generate_and_test.py:
    
    1. Import from RL_utils_gpu instead:
       from utils.RL_utils_gpu import generate_and_test, collect_trajectories, SimpleLLMPredictor
    
    2. Create predictor instead of Ray actors:
       predictor = SimpleLLMPredictor(args.model, args.tokenizer_path)
    
    3. Create collect_traj lambda:
       collect_traj = lambda predictor, num_workers, selected_lemmas, lemma_mapping, seed: \\
           collect_trajectories(predictor, num_workers, selected_lemmas, \\
                               MAX_LENGTH, seed, args.temperature, \\
                               cache_dir=os.path.join(args.exp_dir, 'sampler_ckpt'))
    
    4. Call generate_and_test with predictor instead of ray_inference_actors:
       generated_proofs = generate_and_test(
           lemmas_to_generate, collect_traj, predictor, lemma_mapping,
           args.seed, os.path.join(args.exp_dir, 'sampler_ckpt'),
           temperature=args.temperature,  # Add temperature parameter
           ...
       )
"""

import os
import json
import time
import pickle
import psutil
import logging
import hashlib
import threading
from datetime import datetime
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Set, Callable, Optional
from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np

from utils.model_utils_gpu import (
    get_prompt, right_truncate, insert_lemma, get_lemma_key, update_lemma_mapping,
    END_THM, START_LEMMA_STMT
)
from utils.gcloud_utils import read_file, write_data, move_file
from utils.prover.lean.verifier import (
    verify_lean4_file, verify_lean4_file_premises, TEST_BATCH_SIZE, DEFAULT_TIMEOUT
)
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Constants
BATCH_SIZE = 2048
prompt_length = 1024
MAX_LENGTH = 1024
CONJECTURE_THRESHOLD = 0.25
NR_FOLD = 5
EARLY_STOP_THRESHOLD = 0.1
CPU_PER_TASK = 1.5

__DEBUG__ = os.getenv("DEBUG", 'False').lower() in ('true', '1', 't')
REPO_DIR = os.path.abspath(os.path.join(__file__, '../../..'))


class SimpleLLMPredictor:
    """
    Simple model predictor without Ray for single-GPU inference.
    
    This class wraps vLLM's LLM class to provide a simple interface
    for generating text completions. It lazily initializes the model
    on first use to avoid loading until needed.
    
    Attributes:
        model_path: Path to the model directory or HuggingFace model name
        tokenizer_path: Path to tokenizer or HuggingFace tokenizer name
        llm: vLLM LLM instance (initialized lazily)
        tokenizer: Transformers tokenizer instance
        kwargs: Additional arguments passed to vLLM LLM initialization
    """
    
    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to model directory or HuggingFace model name
            tokenizer_path: Path to tokenizer or HuggingFace tokenizer name
            **kwargs: Additional arguments for vLLM LLM (e.g., dtype, max_model_len)
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.kwargs = kwargs
        self.llm = None
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def predict(self, prompts: List[str], sampling_params: SamplingParams) -> List[Dict]:
        """
        Generate completions for a batch of prompts.
        
        Args:
            prompts: List of prompt strings
            sampling_params: vLLM SamplingParams object with generation settings
            
        Returns:
            List of dicts with 'id' and 'text' keys, where 'id' is the index
            and 'text' is the generated completion
        """
        if self.llm is None:
            # Lazy initialization - only load model when first needed
            self.llm = LLM(
                model=self.model_path,
                dtype='bfloat16',
                max_model_len=1024,
                gpu_memory_utilization=0.85,
                **self.kwargs
            )
        
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        results = []
        for idx, output in enumerate(outputs):
            result = {"id": idx, "text": output.outputs[0].text}
            if sampling_params.logprobs is not None:
                result["logprobs"] = output.outputs[0].logprobs
            results.append(result)
        return results
    
    def tokenize(self, queries: List[Dict], **kwargs) -> List[Dict]:
        """
        Convert test_info dictionaries into prompt strings.
        
        Args:
            queries: List of test_info dictionaries
            **kwargs: Additional arguments for get_prompt (e.g., max_length, invoke_type)
            
        Returns:
            List of dicts with 'id' and 'text' keys, where 'text' is the prompt string
        """
        results = []
        for idx, test_info in enumerate(queries):
            prompt = get_prompt(test_info, self.tokenizer, **kwargs)
            results.append({"id": idx, "text": prompt})
        return results


def direct_get_prompt(
    predictor: SimpleLLMPredictor,
    queries: List[Dict],
    max_length: int = prompt_length,
    invoke_type: Optional[str] = None
) -> List[str]:
    """
    Generate prompts for a list of queries using direct model calls.
    
    This function replaces ray_get_prompt by directly calling the predictor's
    tokenize method without Ray overhead.
    
    Args:
        predictor: SimpleLLMPredictor instance
        queries: List of test_info dictionaries
        max_length: Maximum token length for prompts
        invoke_type: Type of invocation ('conjecture' or None)
        
    Returns:
        List of prompt strings in the same order as queries
    """
    logging.debug(f"Generating prompts for {len(queries)} queries...")
    results = predictor.tokenize(queries, max_length=max_length, invoke_type=invoke_type)
    # Sort by id to maintain order
    results = sorted(results, key=lambda x: x["id"])
    assert len(results) == len(queries), f"len(results)={len(results)}, len(queries)={len(queries)}"
    assert all(result["id"] == i for i, result in enumerate(results)), "found non-consecutive ids"
    logging.debug(f"Prompt generation complete.")
    return [result["text"] for result in results]


def direct_completion(
    predictor: SimpleLLMPredictor,
    prompts: List[str],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    seed: int = 0,
    logprobs: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> List[Dict]:
    """
    Generate completions for prompts using direct model calls.
    
    This function replaces ray_completion by directly calling the predictor's
    predict method. It includes caching support to avoid regenerating the same
    completions.
    
    Args:
        predictor: SimpleLLMPredictor instance
        prompts: List of prompt strings
        temperature: Sampling temperature (0.0 = deterministic, higher = more random)
        max_tokens: Maximum tokens to generate per prompt
        seed: Random seed for reproducibility
        logprobs: Number of logprobs to return (None = don't return)
        cache_dir: Directory to cache results (None = no caching)
        
    Returns:
        List of dicts with 'id' and 'text' keys, where 'id' is the index
        and 'text' is the generated completion. Results are sorted by id.
    """
    # Create cache key from inputs
    cache_key = hashlib.md5(pickle.dumps((prompts, temperature, max_tokens, seed, logprobs))).hexdigest()
    cache_file_path = os.path.join(cache_dir, f"{cache_key}.pkl") if cache_dir else None
    cache_file_path_inputs = os.path.join(cache_dir, f"{cache_key}_inputs.pkl") if cache_dir else None
    
    # Check cache
    if cache_file_path:
        cache_ret = read_file(cache_file_path)
        if cache_ret is not None:
            assert len(cache_ret) == len(prompts), f"len(cache_ret)={len(cache_ret)}, len(prompts)={len(prompts)}"
            return cache_ret
    
    # Save inputs for debugging if in debug mode
    if cache_file_path_inputs and __DEBUG__:
        write_data(pickle.dumps(prompts), cache_file_path_inputs, 'pickle')
    
    # Create sampling params
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        seed=seed,
        max_tokens=max_tokens,
        logprobs=logprobs
    )
    
    # Generate completions
    results = predictor.predict(prompts, sampling_params)
    
    # Sort by id to maintain order
    results = sorted(results, key=lambda x: x["id"])
    assert len(results) == len(prompts), f"len(results)={len(results)}, len(prompts)={len(prompts)}"
    assert all(result["id"] == i for i, result in enumerate(results)), "found non-consecutive ids"
    
    # Save to cache
    if cache_file_path:
        write_data(pickle.dumps(results), cache_file_path, 'pickle')
    
    return results


def collect_trajectories(
    predictor: SimpleLLMPredictor,
    num_workers: int,  # Kept for compatibility, unused in GPU version
    lemmas_to_generate: List[Dict],
    max_length: int,
    seed: int,
    temperature: float,
    cache_dir: Optional[str] = None
) -> List[Dict]:
    """
    Generate proof trajectories (completions) for a list of lemmas.
    
    This function takes lemma information, generates prompts, calls the model
    to generate proofs, and returns the results with proof text attached.
    
    Args:
        predictor: SimpleLLMPredictor instance for model inference
        lemmas_to_generate: List of test_info dictionaries containing lemma statements
        max_length: Maximum tokens to generate per proof
        seed: Random seed for reproducibility
        temperature: Sampling temperature for generation
        cache_dir: Directory for caching prompts/completions (None = no caching)
        
    Returns:
        List of test_info dictionaries with 'proof' field added containing
        the generated proof text. Order matches input lemmas_to_generate.
    """
    generated_proofs = []
    
    # Step 1: Generate prompts from lemma information
    prompts = direct_get_prompt(
        predictor, lemmas_to_generate,
        max_length=prompt_length,
        invoke_type=None
    )
    
    # Step 2: Generate completions (proofs) from model
    completions = direct_completion(
        predictor, prompts,
        temperature=temperature,
        max_tokens=max_length,
        seed=seed,
        cache_dir=cache_dir
    )
    
    # Step 3: Extract proof text and combine with original test_info
    for output, test_info in zip(completions, lemmas_to_generate):
        # Remove markdown code blocks if present
        generated_text = output['text'].split('\n```', 1)[0]
        generated_proofs.append(test_info | {'proof': generated_text})
    
    return generated_proofs


class SimpleLean4Verifier:
    """
    Simple Lean4 verifier without Ray for direct verification calls.
    
    This class provides a simple interface for verifying Lean4 proofs
    by directly calling the verification functions. It handles batching
    and result merging internally.
    
    Attributes:
        collect_premises: Whether to collect premise information for successful proofs
        timeout: Timeout in seconds for verification
    """
    
    def __init__(self, collect_premises: bool = True, timeout: int = DEFAULT_TIMEOUT):
        """
        Initialize the verifier.
        
        Args:
            collect_premises: Whether to extract premise information for successful proofs
            timeout: Timeout in seconds for each verification call
        """
        self.collect_premises = collect_premises
        self.timeout = timeout
    
    def run(self, inputs: List[Dict], batched: bool = True) -> List[Dict]:
        """
        Verify a batch of proofs.
        
        Args:
            inputs: List of test_info dictionaries with 'statement' and 'proof' fields
            batched: If True, verify all proofs in batch mode (faster, shared context).
                    If False, verify one-by-one with full premise extraction.
        
        Returns:
            List of test_info dictionaries with verification results merged in.
            Each result includes fields like 'complete', 'errors', 'sorries', etc.
        """
        # Check memory before starting
        while psutil.virtual_memory().percent > 75.0:
            logging.warning(f"Memory usage high ({psutil.virtual_memory().percent}%), waiting...")
            time.sleep(5)
        
        if batched:
            # Batch verification - faster, shared header context
            codes = [test_info['statement'] + '\n' + test_info['proof'] for test_info in inputs]
            headers = [test_info.get('header', None) for test_info in inputs]
            
            results = verify_lean4_file(
                codes=codes,
                headers=headers,
                premises=False,
                ast=False
            )
            
            # Collect premises for successful proofs if requested
            if self.collect_premises:
                for i, (test_info, result) in enumerate(zip(inputs, results)):
                    if result.get('complete', False):
                        premise_result = verify_lean4_file_premises(
                            code=test_info['statement'] + '\n' + test_info['proof'],
                            header=test_info.get('header', None),
                            premises=True,
                            ast=True,
                            timeout=self.timeout
                        )
                        results[i] = premise_result[0]
        else:
            # Single verification - one at a time with full premise extraction
            assert len(inputs) == 1, "Single input only for non-batched mode"
            test_info = inputs[0]
            premise_result = verify_lean4_file_premises(
                code=test_info['statement'] + '\n' + test_info['proof'],
                header=test_info.get('header', None),
                premises=True,
                ast=True,
                timeout=self.timeout
            )
            results = premise_result
        
        # Merge results with original test_info
        outputs = []
        for test_info, result in zip(inputs, results):
            outputs.append(test_info | result)
        
        return outputs


def get_deduplication_key(test_info: Dict) -> Tuple:
    """
    Generate a deduplication key for a test_info dictionary.
    
    Two proofs are considered duplicates if they have the same statement,
    proof text, header, and iteration number. This is used to avoid
    testing identical proofs multiple times.
    
    Args:
        test_info: Dictionary containing lemma information
        
    Returns:
        Tuple of (statement, proof, header, iter) used as a unique key
    """
    return (
        test_info['statement'],
        test_info['proof'],
        test_info.get('header', ''),
        test_info.get('iter', 0)
    )


def get_result_items(test_info: Dict) -> Dict:
    """
    Extract verification result fields from a test_info dictionary.
    
    This function filters out non-result fields and keeps only the
    verification-related information that should be stored in test_results.
    
    Args:
        test_info: Dictionary containing lemma information with verification results
        
    Returns:
        Dictionary containing only verification result fields:
        - complete: Whether proof is complete (no errors, no sorries)
        - sorries: List of sorries found
        - errors: List of errors found
        - system_messages: System error messages
        - pass: Whether proof passed (no errors)
        - invokes: List of invoked lemmas
        - verified_code: The verified code
        - verify_time: Time taken to verify
    """
    return {
        k: v for k, v in test_info.items()
        if k in [
            'complete', 'sorries', 'errors', 'system_messages',
            'pass', 'invokes', 'verified_code', 'verify_time'
        ]
    }


def split_test_blocks(
    test_infos: List[Dict],
    batch_size: int,
    group_by_header: bool = False
) -> List[List[Dict]]:
    """
    Split test_infos into batches for verification.
    
    This function organizes proofs into batches for efficient verification.
    If group_by_header is True, proofs with the same header are grouped
    together to share context and reduce overhead.
    
    Args:
        test_infos: List of test_info dictionaries to batch
        batch_size: Maximum number of proofs per batch
        group_by_header: If True, group proofs by header before batching
        
    Returns:
        List of batches, where each batch is a list of test_info dictionaries
    """
    rng = np.random.default_rng(0)
    rng.shuffle(test_infos)
    
    if group_by_header:
        # Sort by header to group proofs with same context together
        test_infos = sorted(test_infos, key=lambda x: x.get('header', None) or '')
    
    # Split into batches
    blocks = [
        test_infos[i: i + batch_size]
        for i in range(0, len(test_infos), batch_size)
    ]
    
    # Shuffle blocks (not individual items) to balance load
    rng.shuffle(blocks)
    return blocks


def save_result(results: Dict, file_path: str) -> None:
    """
    Save test results to disk with atomic write.
    
    This function saves results to a temporary file first, then moves it
    to the final location to ensure atomic writes (no partial files).
    
    Args:
        results: Dictionary mapping deduplication keys to result items
        file_path: Path where results should be saved (will be compressed with .gz)
    """
    file_path_tmp = file_path + '_backup'
    write_data(pickle.dumps(results), file_path_tmp, 'pkl')
    move_file(file_path_tmp + '.gz', file_path + '.gz')


def generate_and_test(
    selected_lemmas: List[Dict],
    collect_traj: Callable,
    predictor: SimpleLLMPredictor,
    lemma_mapping: Dict,
    seed: int,
    save_dir: Optional[str],
    temperature: float = 0.7,  # Added for GPU version
    cpus_per_task: float = CPU_PER_TASK,
    cpus_per_task_stage2: float = CPU_PER_TASK + 1.5,
    test_batch_size: int = TEST_BATCH_SIZE,
    group_by_header: bool = False,
    collect_premises: bool = True,
) -> List[Dict]:
    """
    Generate and test proofs for selected lemmas without Ray.
    
    This is the main orchestration function that coordinates proof generation
    and verification. It processes lemmas in shards, generates proofs,
    deduplicates them, and verifies them in parallel. Results are saved
    periodically and can be resumed from cache.
    
    The function operates in two stages:
    1. Stage 1: Generate and verify proofs in batches (faster, shared context)
    2. Stage 2: Retry failed/timed-out proofs one-by-one (slower, more resources)
    
    Args:
        selected_lemmas: List of test_info dictionaries to generate proofs for
        collect_traj: Function that generates proofs. Should have signature:
                     collect_traj(predictor, lemmas, max_length, seed, temperature, cache_dir)
        predictor: SimpleLLMPredictor instance for model inference
        lemma_mapping: Dictionary mapping lemma statements to IDs
        seed: Random seed for reproducibility
        save_dir: Directory to save intermediate results (None = no saving)
        temperature: Sampling temperature for proof generation (default: 0.7)
        cpus_per_task: CPU resources per verification task (unused in GPU version)
        cpus_per_task_stage2: CPU resources per task in stage 2 (unused in GPU version)
        test_batch_size: Number of proofs to verify per batch
        group_by_header: If True, group proofs by header for efficient verification
        collect_premises: Whether to extract premise information for successful proofs
        
    Returns:
        List of test_info dictionaries with generated proofs and verification results.
        Each dictionary includes:
        - Original lemma information (statement, lemma_id, etc.)
        - Generated proof text
        - Verification results (complete, errors, sorries, etc.)
    """
    # Create verifier instance
    verifier = SimpleLean4Verifier(collect_premises=collect_premises, timeout=DEFAULT_TIMEOUT * test_batch_size)
    
    # Setup save files
    save_file_generation = os.path.join(save_dir, f'generated_proofs.json') if save_dir else None
    generated_proofs_dedup = read_file(save_file_generation) if save_file_generation else None
    save_file_tests = os.path.join(save_dir, f'test_results.pkl') if save_dir else None
    test_results = (read_file(save_file_tests) if save_file_tests else None) or {}
    
    total_test_tasks = 0
    total_test_instances = 0
    
    pool_lock = threading.Lock()
    pbar = tqdm(total=len(selected_lemmas))
    finished_generation = False
    
    def aggregate_results(early_stop_threshold: int = 0):
        """
        Background thread function to collect and save test results.
        
        This function runs in a separate thread and continuously collects
        verification results as they complete, updating progress and saving
        periodically.
        
        Args:
            early_stop_threshold: Stop early if remaining tasks <= this threshold
        """
        save_interval = 900  # Save every 15 minutes
        last_save_time = time.time()
        nr_finished_jobs = 0
        nr_tests_correct = 0
        nr_tests_done = 0
        cached_pbar_n = 0
        
        while len(test_queue) > 0 or (not finished_generation):
            if finished_generation and (total_test_tasks - nr_finished_jobs <= early_stop_threshold):
                break
            
            # Also break if generation is finished and we've processed all expected tasks
            if finished_generation and nr_finished_jobs >= total_test_tasks:
                break
            
            if len(test_queue) == 0:
                time.sleep(2)
                continue
            
            try:
                # Get next batch of results from queue
                with pool_lock:
                    if len(test_queue) == 0:
                        continue
                    results = test_queue.pop(0)
                
                nr_finished_jobs += 1
                
                # Store results
                for result in results:
                    test_results[get_deduplication_key(result)] = get_result_items(result)
                
                nr_tests_done += len(results)
                nr_tests_correct += sum(result.get('complete', False) for result in results)
                
                # Periodic save
                current_time = time.time()
                if (current_time - last_save_time >= save_interval) and (save_file_tests is not None):
                    save_result(test_results, save_file_tests)
                    last_save_time = current_time
                
                cached_pbar_n += len(results)
                if finished_generation:
                    pbar.update(cached_pbar_n)
                    cached_pbar_n = 0
                    
            except Exception as e:
                logging.error(f"Error in aggregate_results: {e}")
                time.sleep(2)
        
        # Final save
        if save_file_tests is not None:
            save_result(test_results, save_file_tests)
    
    # Queue for test results (replaces Ray pool)
    test_queue = []
    test_queue_lock = threading.Lock()
    
    # Store temperature for use in collect_traj calls
    # Note: collect_traj is a lambda that captures temperature, but we need it here too
    # For compatibility, we'll create a wrapper if needed
    if generated_proofs_dedup is None:
        # Start background aggregator thread
        monitoring_thread = threading.Thread(
            target=aggregate_results,
            args=(int(EARLY_STOP_THRESHOLD),)
        )
        monitoring_thread.daemon = True
        monitoring_thread.start()
        
        start_time = datetime.now()
        generated_proofs_dedup = []
        deduplicate_index = {}
        batch_size = (len(selected_lemmas) + NR_FOLD - 1) // NR_FOLD
        
        # Process in shards (NR_FOLD = 5)
        for shard in range(NR_FOLD):
            logging.debug(f'Processing shard {shard}/{NR_FOLD}...')
            batch = selected_lemmas[batch_size * shard: batch_size * (shard + 1)]
            
            # Generate proofs for this shard
            # collect_traj signature: (predictor, num_workers, selected_lemmas, lemma_mapping, seed)
            # It internally calls collect_trajectories with temperature and cache_dir from closure
            generated_proofs = collect_traj(
                predictor, 1, batch, lemma_mapping,
                seed * NR_FOLD + shard
            )
            
            start_idx = len(generated_proofs_dedup)
            
            # Deduplicate proofs before testing
            for test_info in generated_proofs:
                key = get_deduplication_key(test_info)
                if key not in deduplicate_index:
                    deduplicate_index[key] = len(generated_proofs_dedup)
                    generated_proofs_dedup.append(test_info | {'multiplicity': 1})
                else:
                    generated_proofs_dedup[deduplicate_index[key]]['multiplicity'] += 1
            
            # Submit new proofs for testing (if not already tested)
            new_testing_tasks = [
                test_info for test_info in generated_proofs_dedup[start_idx:]
                if get_deduplication_key(test_info) not in test_results
            ]
            total_test_instances += len(new_testing_tasks)
            pbar.total = total_test_instances
            new_testing_tasks = split_test_blocks(new_testing_tasks, test_batch_size, group_by_header)
            
            # Submit verification tasks (run in background)
            def verify_batch(testing_block):
                """Verify a batch of proofs and add results to queue."""
                try:
                    results = verifier.run(testing_block, batched=True)
                    with test_queue_lock:
                        test_queue.append(results)
                except Exception as e:
                    logging.error(f"Error verifying batch: {e}")
                    # Add failed results
                    failed_results = [
                        test_info | {'complete': False, 'system_errors': str(e)}
                        for test_info in testing_block
                    ]
                    with test_queue_lock:
                        test_queue.append(failed_results)
            
            verification_threads = []
            with pool_lock:
                for testing_block in new_testing_tasks:
                    # Run verification in background thread
                    thread = threading.Thread(target=verify_batch, args=(testing_block,))
                    thread.daemon = True
                    thread.start()
                    verification_threads.append(thread)
                    total_test_tasks += 1
        
        finished_generation = True
        logging.info(f'Finished generation. #generated lemmas = {len(generated_proofs_dedup)}.')
        duration = datetime.now() - start_time
        logging.info('Inference time: ' + str(duration))
        
        logging.info(f'Start testing {total_test_tasks} tasks...')
        start_time = datetime.now()
        
        # Wait for all verification threads to complete
        logging.info(f"Waiting for {len(verification_threads)} verification threads to complete...")
        for i, thread in enumerate(verification_threads):
            thread.join(timeout=600)  # Wait up to 10 minutes per thread
            if thread.is_alive():
                logging.warning(f"Verification thread {i} did not complete within timeout (600s)")
            else:
                logging.debug(f"Verification thread {i} completed successfully")
        
        # Give monitoring thread a moment to process any remaining results
        logging.info(f"Waiting for monitoring thread to process remaining results (queue size: {len(test_queue)})...")
        # Wait a bit for queue to be processed
        max_wait = 120  # 2 minutes max
        wait_time = 0
        while len(test_queue) > 0 and wait_time < max_wait:
            time.sleep(2)
            wait_time += 2
            if wait_time % 10 == 0:
                logging.info(f"Still waiting for queue to empty... ({len(test_queue)} items remaining, waited {wait_time}s)")
        
        monitoring_thread.join(timeout=30)  # Wait up to 30 seconds for final processing
        if monitoring_thread.is_alive():
            logging.warning("Monitoring thread did not complete within timeout, but continuing anyway")
        else:
            logging.debug("Monitoring thread completed successfully")
        pbar.close()
        
        # Stage 2: Retry failed/timed-out proofs
        print('Stage 2: rerunning timed-out jobs...')
        
        # Create new verifier with more resources for stage 2
        verifier_stage2 = SimpleLean4Verifier(
            collect_premises=collect_premises,
            timeout=DEFAULT_TIMEOUT
        )
        
        new_testing_tasks = []
        for test_info in generated_proofs_dedup:
            key = get_deduplication_key(test_info)
            if (key not in test_results) or ('complete' not in test_results[key]):
                new_testing_tasks.append([test_info])  # One at a time
        
        logging.info(f'Number of lemmas to test in stage 2: {len(new_testing_tasks)}')
        pbar = tqdm(total=len(new_testing_tasks))
        
        # Reset queue for stage 2
        test_queue = []
        finished_generation = False
        total_test_tasks = len(new_testing_tasks)
        
        # Start new monitoring thread
        monitoring_thread = threading.Thread(target=aggregate_results)
        monitoring_thread.daemon = True
        monitoring_thread.start()
        
        # Verify one-by-one in stage 2
        def verify_single(test_info_list):
            """Verify a single proof and add result to queue."""
            try:
                results = verifier_stage2.run(test_info_list, batched=False)
                with test_queue_lock:
                    test_queue.append(results)
            except Exception as e:
                logging.error(f"Error verifying single proof: {e}")
                failed_results = [
                    test_info | {'complete': False, 'system_errors': str(e)}
                    for test_info in test_info_list
                ]
                with test_queue_lock:
                    test_queue.append(failed_results)
        
        stage2_threads = []
        for testing_block in new_testing_tasks:
            thread = threading.Thread(target=verify_single, args=(testing_block,))
            thread.daemon = True
            thread.start()
            stage2_threads.append(thread)
        
        # Wait for all stage 2 threads
        logging.info(f"Waiting for {len(stage2_threads)} stage 2 verification threads to complete...")
        for i, thread in enumerate(stage2_threads):
            thread.join(timeout=600)  # Wait up to 10 minutes per thread
            if thread.is_alive():
                logging.warning(f"Stage 2 verification thread {i} did not complete within timeout (600s)")
            else:
                logging.debug(f"Stage 2 verification thread {i} completed successfully")
        
        finished_generation = True
        
        # Give monitoring thread a moment to process any remaining results
        logging.info(f"Waiting for stage 2 monitoring thread to process remaining results (queue size: {len(test_queue)})...")
        max_wait = 120  # 2 minutes max
        wait_time = 0
        while len(test_queue) > 0 and wait_time < max_wait:
            time.sleep(2)
            wait_time += 2
            if wait_time % 10 == 0:
                logging.info(f"Still waiting for queue to empty... ({len(test_queue)} items remaining, waited {wait_time}s)")
        
        monitoring_thread.join(timeout=30)  # Wait up to 30 seconds for final processing
        if monitoring_thread.is_alive():
            logging.warning("Stage 2 monitoring thread did not complete within timeout, but continuing anyway")
        else:
            logging.debug("Stage 2 monitoring thread completed successfully")
        pbar.close()
        
        duration = datetime.now() - start_time
        logging.info('Testing time: ' + str(duration))
        
        # Merge test results back into generated_proofs
        nr_failed = 0
        for test_info in generated_proofs_dedup:
            key = get_deduplication_key(test_info)
            if key in test_results:
                test_info |= test_results[key]
            else:
                nr_failed += 1
                test_info['complete'] = False
                test_info['system_errors'] = 'test failed'
        
        # Validate failure rate
        max_allowed_failures = len(generated_proofs_dedup) * 0.005
        if nr_failed >= max_allowed_failures:
            if __DEBUG__ or len(generated_proofs_dedup) < 100:
                print(f"Warning: Failed to test {nr_failed}/{len(generated_proofs_dedup)} lemmas "
                      f"(Lean verification may not be working). Continuing anyway...")
            else:
                assert False, f'Failed to test {nr_failed} lemmas (expected < {max_allowed_failures})'
        
        if save_file_generation:
            write_data(json.dumps(generated_proofs_dedup), save_file_generation, 'json')
    else:
        print(f'Loaded {len(generated_proofs_dedup)} lemmas from {save_file_generation}')
        for test_info in generated_proofs_dedup:
            update_lemma_mapping(lemma_mapping, test_info)
    
    return generated_proofs_dedup

