source ~/cs229/bin/activate
cd ~/STP-GPU
export VLLM_WORKER_MULTIPROC_METHOD=spawn


python RL/inference_lora_SFT.py \
    --model deepseek-ai/DeepSeek-Prover-V2-7B \
    --lora_checkpoint ./RL/SFT/checkpoints/sketch-prover/checkpoint-1875 \
    --exp_dir ./RL/results/sft-test \
    --save_file_name SFTminiF2F-test_results \
    --raw_dataset_config ./RL/dataset_configs/miniF2F-test.json \
    --max_examples 1000

python RL/export_results.py \
    --input ./RL/results/sft-test/SFTminiF2F-test_results.jsonl.gz \
    --output ./RL/results/sft-test/SFTminiF2F-test_verified.txt \
    --verified

python RL/inference_lora_SFT.py \
    --model deepseek-ai/DeepSeek-Prover-V2-7B \
    --lora_checkpoint ./RL/SFT/checkpoints/sketch-prover/checkpoint-1875 \
    --exp_dir ./RL/results/sft-test \
    --save_file_name SFTproofNet-test_results \
    --raw_dataset_config ./RL/dataset_configs/proofNet-test.json \
    --max_examples 1000

python RL/export_results.py \
    --input ./RL/results/sft-test/SFTproofNet-test_results.jsonl.gz \
    --output ./RL/results/sft-test/SFTproofNet-test_verified.txt \
    --verified

python RL/inference_lora_SFT.py \
    --model deepseek-ai/DeepSeek-Prover-V2-7B \
    --lora_checkpoint ./RL/SFT/checkpoints/sketch-prover/checkpoint-1875 \
    --exp_dir ./RL/results/sft-test \
    --save_file_name SFTputnamBench-test_results \
    --raw_dataset_config ./RL/dataset_configs/putnam_bench.json \
    --max_examples 1000

python RL/export_results.py \
    --input ./RL/results/sft-test/SFTputnamBench-test_results.jsonl.gz \
    --output ./RL/results/sft-test/SFTputnamBench-test_verified.txt \
    --verified



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

# python export_results.py \
# --input results/REINFORCE2miniF2F-test_results.jsonl.gz \
# --output results/exported/miniF2F-test/REINFORCE2miniF2F-test_results.txt

# python export_results.py \
# --input results/REINFORCE2ProofNet-test_results.jsonl.gz \
# --output results/exported/ProofNet-test/REINFORCE2ProofNet-test_results.txt


# python inference_single_model.py \
# --raw_dataset_config dataset_configs/putnam_bench.json \
# --max_examples 1000 \
# --model deepseek-ai/DeepSeek-Prover-V2-7B \
# --save_file_name BaselinePutnamBench_results  \
# --exp_dir results
