#!/bin/bash
#SBATCH --job-name=train_ablation
#SBATCH --time=60:00:00                 # Adjust runtime as needed
#SBATCH --partition=gpu                 # GPU queue
# #SBATCH --gres=gpu:1                    # 1 GPU
#SBATCH --gres=gpu:rtx2080ti:1       # 1 RTX 2080 Ti GPU, if code allows change to :4 for parallel processing
#SBATCH --cpus-per-task=8               # CPU cores
#SBATCH --mem=32G                       # RAM
#SBATCH --output=/cluster/work/smslab/2025-eeg_headband/logs/train_ablation_%j.out
#SBATCH --error=/cluster/work/smslab/2025-eeg_headband/logs/train_ablation_%j.err

# 1) Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eeg_env

# 2) Info
echo "Node(s): $SLURM_NODELIST"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "Current dir: $(pwd)"

# 3) Go to project root
cd /cluster/work/smslab/2025-eeg_headband/sem-proj

# 4) GPU info
nvidia-smi

# 5) Run your training script
python -u scripts/train_variable_gru.py
