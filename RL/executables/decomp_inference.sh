source ~/cs229/bin/activate
cd ~/STP-GPU/RL
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# python run_lora_inference.py \
# --lora_checkpoint experiments/REINFORCE_full_run/final_step_11 \
# --raw_dataset_config dataset_configs/proofNet-test.json \
# --max_examples 300 \
# --model deepseek-ai/DeepSeek-Prover-V2-7B \
# --save_file_name REINFORCE2ProofNet-test_results  \
# --exp_dir results \
# # --force_merge

# python run_lora_inference.py \
# --lora_checkpoint experiments/REINFORCE_full_run/final_step_11 \
# --raw_dataset_config dataset_configs/miniF2F-test.json \
# --max_examples 300 \
# --model deepseek-ai/DeepSeek-Prover-V2-7B \
# --save_file_name REINFORCE2miniF2F-test_results  \
# --exp_dir results 

python export_results.py \
--input results/REINFORCE2miniF2F-test_results.jsonl.gz \
--output results/exported/miniF2F-test/REINFORCE2miniF2F-test_results.txt

python export_results.py \
--input results/REINFORCE2ProofNet-test_results.jsonl.gz \
--output results/exported/ProofNet-test/REINFORCE2ProofNet-test_results.txt


# python inference_single_model.py \
# --raw_dataset_config dataset_configs/putnam_bench.json \
# --max_examples 1000 \
# --model deepseek-ai/DeepSeek-Prover-V2-7B \
# --save_file_name BaselinePutnamBench_results  \
# --exp_dir results
