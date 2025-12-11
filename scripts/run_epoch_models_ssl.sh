#!/bin/bash
#SBATCH --job-name=train_ablation
#SBATCH --time=60:00:00                 # Adjust runtime as needed
#SBATCH --partition=gpu                 # GPU queue
# #SBATCH --gres=gpu:1                    # 1 GPU
#SBATCH --gres=gpu:rtx2080ti:1       # 1 RTX 2080 Ti GPU, if code allows change to :4 for parallel processing
#SBATCH --cpus-per-task=8               # CPU cores
#SBATCH --mem=32G                       # RAM
#SBATCH --output=/cluster/work/smslab/2025-eeg_headband/logs/epoch_models_ssl_%j.out
#SBATCH --error=/cluster/work/smslab/2025-eeg_headband/logs/epoch_models_ssl_%j.err

# 1) Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eeg_env

# 2) Info
echo "Node(s): $SLURM_NODELIST"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "Current dir: $(pwd)"

# 3) Go to project root
cd /cluster/work/smslab/2025-eeg_headband/sem-proj

# 4) Add src to PYTHONPATH
export PYTHONPATH="/cluster/work/smslab/2025-eeg_headband/sem-proj/src:$PYTHONPATH"

# 5) GPU info
nvidia-smi

# 6) Run your training script using module path
python -u -m sem_proj.training.epoch_models_ssl