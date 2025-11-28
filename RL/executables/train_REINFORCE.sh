 
source ~/cs229/bin/activate
cd ~/STP-GPU/RL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python GRPO/train_reinforce.py   \
 --model deepseek-ai/DeepSeek-Prover-V2-7B  \
 --dataset_config dataset_configs/leanworkbook.json   \
 --output_dir experiments/REINFORCE_full2   \
 --max_tokens 512   \
 --max_examples_per_dataset 500   \
 --batch_size 1   \
 --num_epochs 1 \
