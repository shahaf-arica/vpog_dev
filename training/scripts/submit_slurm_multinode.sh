#!/bin/bash
#SBATCH --job-name=vpog_multinode
#SBATCH --output=logs/vpog_multinode_%j.out
#SBATCH --error=logs/vpog_multinode_%j.err
#SBATCH --nodes=2                # 2 nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4        # 4 GPUs per node = 8 total GPUs
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu

# Multi-node VPOG Training
#
# This script runs distributed training across multiple nodes
# Total GPUs = nodes × gpus-per-node (e.g., 2 × 4 = 8 GPUs)

echo "========================================"
echo "Multi-Node SLURM Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_ON_NODE))"
echo "========================================"

# Load modules and activate environment
source activate pose

cd $SLURM_SUBMIT_DIR
mkdir -p logs

echo "Starting multi-node training..."

python training/train.py \
    machine=slurm \
    machine.batch_size=8 \
    max_epochs=100 \
    name_exp=vpog_multinode_${SLURM_JOB_ID}

STATUS=$?
echo "Training finished with exit status: $STATUS"
exit $STATUS
