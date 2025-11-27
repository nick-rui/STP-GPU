source ~/cs229/bin/activate
cd ~/STP-GPU/RL
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# python run_lora_inference.py \
# --lora_checkpoint experiments/REINFORCE_full_run/final_step_11 \
# --raw_dataset_config dataset_configs/miniF2F-test.json \
# --max_examples 300 \
# --model deepseek-ai/DeepSeek-Prover-V2-7B \
# --save_file_name REINFORCE_miniF2F-test_results  \
# --exp_dir results

python inference_single_model.py \
--raw_dataset_config dataset_configs/putnam_bench.json \
--max_examples 1000 \
--model deepseek-ai/DeepSeek-Prover-V2-7B \
--save_file_name BaselinePutnamBench_results  \
--exp_dir results
