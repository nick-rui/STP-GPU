# Fork of "Self-play LLM Theorem Provers with Iterative Conjecturing and Proving"

NOTHING BELOW IS MINE!

This is an official implementation of the Self-play Theorem Prover as described in the paper [https://arxiv.org/abs/2502.00212](https://arxiv.org/abs/2502.00212). The code is based on [levanter](https://github.com/stanford-crfm/levanter/), [DeepSeek-Prover-V1.5](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5), and [LeanDojo](https://leandojo.org/). Please cite the paper and star this repo if you find STP useful. Thank you!


```tex
@article{dong2025beyond,
  title={Beyond Limited Data: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving},
  author={Dong, Kefan and Ma, Tengyu},
  journal={arXiv preprint arXiv:2502.00212},
  year={2025}
}
```
## News
- [2025/03] We scaled STP’s training compute by 2x, and achieved new SoTA for whole-proof generation methods on miniF2F, ProofNet, and LeanWorkbook!
- [2025/02] Code, data, and models of STP are released.

## 1. Evaluation Results

The table below compares the pass@3200 performance of STP (our model) and DeepSeek-Prover-V1.5 on miniF2F-test and ProofNet-test.

<div align="center">

|  | miniF2F-test | ProofNet-test |
|--------|------------------|------------------|
| **DeepSeek-Prover-V1.5-SFT** | 53.3% ± 0.5% | 21.0% ± 0.9% |
| **DeepSeek-Prover-V1.5-RL** | 54.9% ± 0.7% | 22.0% ± 0.5% |
| **STP** | **65.0% ± 0.5%** | **23.9% ± 0.6%** |

</div>

## 2. Model and Dataset
Our final model can be downloaded [here](https://huggingface.co/kfdong/STP_model_Lean_0320).

We also release the dataset [here](https://huggingface.co/datasets/kfdong/STP_Lean_0320), which contains:
- Extracted examples from mathlib4,
- Generated correct proofs of statements in LeanWorkbook, 
- Generated correct proofs of conjectures proposed by our model during self-play training. 

Our final model is finetuned from DeepSeek-Prover-V1.5-SFT with this dataset for 1 epoch.

## 3. Setup Environment

Our experiments are mostly run on TPU VMs. You can find a quick overview of TPU VMs [here](https://github.com/stanford-crfm/levanter/blob/main/docs/Getting-Started-TPU-VM.md), and please follow the instructions to [setup Google Cloud](https://github.com/stanford-crfm/levanter/blob/main/docs/Getting-Started-TPU-VM.md#google-cloud-setup).

### 3.1. Request a TPU VM
Pease first modify the environment variables in the last few lines of `levanter/infra/helpers/setup-tpu-vm.sh` to provide the path to your [Google Cloud buckets](https://cloud.google.com/storage/docs/creating-buckets#console), WandB access token, and Hugging Face access token.

Use the following commands to request an instance with 256 TPU-v4 cores (please replace `STP-train` and `us-central2-b` with your TPU name and zone):
```sh
cd levanter
eval $(ssh-agent -s) bash infra/spin-up-vm.sh STP-train -z us-central2-b -t v4-256
```
This will automatically install the dependencies specified in [this setup file](https://github.com/kfdong/STP/blob/main/levanter/infra/helpers/setup-tpu-vm.sh), such as Levanter, vLLM, Lean4, and mathlib4, and sync the environment variables to all TPU nodes.

### 3.2. Reproduce Test Results
Use the following commands to connect to the TPU VM, and then generate and test proofs on miniF2F-test and ProofNet-test.

```sh
gcloud compute tpus tpu-vm ssh STP-train --zone us-central2-b --worker=0
source ~/venv_vllm/bin/activate
cd ~/STP/RL
source .bash_alias.sh
bash run_generation_and_test.sh kfdong/STP_model_Lean_0320 $STORAGE/STP/benchmark_results
python summary.py --log_path $STORAGE/STP/benchmark_results/generated_proofs_tests.jsonl.gz --split miniF2F --max_iter 3200
python summary.py --log_path $STORAGE/STP/benchmark_results/generated_proofs_tests.jsonl.gz --split proofnet --max_iter 3200
```

### 3.3. Prepare Datasets

On the TPU VM, use the following commands to prepare SFT datasets:
```sh
cd ~/STP/RL
source ~/venv_vllm/bin/activate
source .bash_alias.sh
python prepare_datasets.py
```
This script will generate three .json files in your Google Cloud storage bucket (`$STORAGE/data/SFT/`), corresponding to the SFT dataset, mathlib4 eval examples, and mathlib4 training examples, respectively.

### 3.4. STP

The Self-play Theorem Prover (STP) involves three training stages:

**Stage 1: Model initialization.** Use the following commands to finetune the DeepSeek-Prover-V1.5-SFT model on the SFT dataset:
```sh
gcloud compute tpus tpu-vm ssh STP-train --zone us-central2-b --worker=0
cd ~/STP/RL
bash run_SFT.sh
```
A Hugging Face compatible model checkpoint will be saved at `$STORAGE/SFT/<wandb run id>/step-114`.

**Stage 2: Self-play training.** On the TPU VM node (worker=0), update the the model checkpoint path in the `RL/run_RL_steps.sh` script. Then, use the following command to run the self-play training. Adjust `TOTAL_ROUNDS` to change the number of iterations of self-play training:
```sh
bash run_RL_steps.sh
```

**Stage 3: Final re-training.** After the self-play training, use the following command to
- create a dataset that contains the correct proofs to the statements in the training dataset and the correct proofs to the generated conjectures,
- re-train a model checkpoint.
```sh
bash run_RL_train.sh
```
In our experiments, we alternate between Stage 2 and Stage 3 to stabilize the training. That is, we periodically restart self-play training (Stage 2) with the re-trained model checkpoint (Stage 3).

### 3.5. GPU Support

The codebase now supports GPU inference! The following changes have been made to enable GPU support:

1. **Ray cluster setup**: Modified `run_inference.sh` to detect GPUs and configure Ray with GPU resources instead of TPU resources.
2. **Model utilities**: Updated `RL/utils/model_utils.py` to use GPU resources in Ray actors.
3. **Embedding workers**: Updated `RL/utils/embedding_utils.py` to use CUDA instead of TPU-specific PyTorch XLA.

#### Running Inference on GPU

**Prerequisites:**
- A machine with one or more NVIDIA GPUs
- CUDA toolkit installed
- Python environment with required dependencies (see `venv310.txt`)

**Steps:**

1. **Set up your environment:**
   ```sh
   # Activate your virtual environment
   source ~/venv_vllm/bin/activate
   
   # Navigate to the STP directory
   cd ~/STP
   ```

2. **Initialize Ray cluster with GPUs:**
   ```sh
   # This will automatically detect available GPUs and start Ray
   bash run_inference.sh
   ```

   The script will:
   - Detect the number of GPUs using `nvidia-smi`
   - Start Ray head node with GPU resources
   - Print the head worker IP for multi-node setups (if needed)

3. **Run inference:**
   ```sh
   cd ~/STP/RL
   source .bash_alias.sh  # If this file exists
   
   # For evaluation/inference using the evaluation scripts
   cd ~/STP/evaluation
   python run_evaluation.py \
     --model_name kfdong/STP_model_Lean_0320 \
     --dataset_path assets/data/test/miniF2F_lean.json \
     --device cuda \
     --batch_size 8 \
     --max_tokens 1024
   ```

   Or use the RL generation scripts:
   ```sh
   cd ~/STP/RL
   # Set environment variables (no TPU needed)
   export EXP_DIR=./results
   
   # Option 1: Use the wrapper script (handles TPU/GPU detection automatically)
   bash run_generation_and_test.sh kfdong/STP_model_Lean_0320 $EXP_DIR
   
   # Option 2: Run directly
   python generate_and_test.py \
     --model kfdong/STP_model_Lean_0320 \
     --exp_dir $EXP_DIR \
     --temperature 1.0 \
     --save_file_name "tests" \
     --raw_dataset_config ./dataset_configs/miniF2F_ProofNet.json \
     --seed 1
   ```

**Notes:**
- The code automatically detects GPUs. If no GPUs are found, it falls back to CPU mode.
- You can override GPU selection using `CUDA_VISIBLE_DEVICES` environment variable.
- For multi-GPU setups, Ray will automatically distribute workers across available GPUs.
- The `execute_on_all_workers` function in `RL/utils/gcloud_utils.py` still uses TPU-specific commands. For GPU setups on a single machine, this function may not be needed. For multi-node GPU setups, you'll need to modify it to use standard SSH commands instead of `gcloud compute tpus tpu-vm ssh`.

## License
-  **Code:** This project is licensed under the MIT License. However, the code in the `levanter/` directory is licensed under the Apache License 2.0.

- **Model:** The model is derived from DeepSeek and is licensed under the [DeepSeek License Agreement](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5/blob/main/LICENSE-MODEL). The use of the model and its derivatives must comply with the use-based restrictions outlined in the [DeepSeek License Agreement](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5/blob/main/LICENSE-MODEL).
