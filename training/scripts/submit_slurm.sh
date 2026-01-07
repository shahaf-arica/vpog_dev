#!/bin/bash
#SBATCH --job-name=vpog_train
#SBATCH --output=logs/vpog_%j.out
#SBATCH --error=logs/vpog_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu

# VPOG Training on SLURM
# 
# Usage:
#   sbatch training/scripts/submit_slurm.sh
#
# Customize:
#   Edit SBATCH parameters above for your cluster
#   Adjust training parameters in the python command below

echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node(s): $SLURM_JOB_NODELIST"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Number of nodes: $SLURM_NNODES"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "========================================"

# Load required modules (adjust for your cluster)
# Example for clusters with module system:
# module load cuda/11.8
# module load anaconda3

# Activate conda environment
source activate pose

# Change to submission directory
cd $SLURM_SUBMIT_DIR

# Print environment info
echo ""
echo "Environment Information:"
echo "========================================"
python --version
which python
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
echo "GPUs detected:"
python -c "import torch; print(torch.cuda.device_count())"
echo "========================================"
echo ""

# Create logs directory
mkdir -p logs

# Start training
echo "Starting VPOG training..."
echo "========================================"

python training/train.py \
    machine=slurm \
    machine.batch_size=12 \
    max_epochs=100 \
    name_exp=vpog_slurm_${SLURM_JOB_ID}

# Capture exit status
STATUS=$?

echo "========================================"
echo "Training finished with exit status: $STATUS"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

exit $STATUS
