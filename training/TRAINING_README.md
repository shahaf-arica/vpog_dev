# VPOG Training Guide

Complete guide for training the VPOG model for 6D object pose estimation.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Local Training](#local-training)
- [SLURM Cluster Training](#slurm-cluster-training)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Prerequisites

1. **Environment setup**:
   ```bash
   conda activate pose  # or your environment name
   ```

2. **Data preparation**:
   - Download templates to `datasets/templates/`
   - Ensure dataset structure matches expected format

3. **Test dataloader** (optional but recommended):
   ```bash
   python training/dataloader/test_integration.py
   ```

### Minimal Training Example

```bash
# Local training with default config
python training/train.py

# With specific GPU
CUDA_VISIBLE_DEVICES=0 python training/train.py

# Override parameters
python training/train.py max_epochs=50 machine.batch_size=8
```

---

## 💻 Local Training

### Single GPU Training

**Basic usage**:
```bash
python training/train.py machine=local
```

**Select specific GPU**:
```bash
CUDA_VISIBLE_DEVICES=0 python training/train.py machine=local
```

**Common configurations**:
```bash
# Fast prototyping (small batch, few epochs)
python training/train.py \
    machine=local \
    max_epochs=10 \
    machine.batch_size=4 \
    log_every_n_steps=10

# Full training
python training/train.py \
    machine=local \
    max_epochs=100 \
    machine.batch_size=12 \
    machine.num_workers=8
```

### Multi-GPU Training (Single Machine)

```bash
# 2 GPUs
python training/train.py \
    machine=local \
    machine.num_gpus=2 \
    machine.batch_size=12

# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python training/train.py \
    machine=local \
    machine.num_gpus=4 \
    machine.batch_size=8
```

**Note**: With multi-GPU training:
- Batch size is **per GPU** (total batch = batch_size × num_gpus)
- Uses PyTorch DDP (Distributed Data Parallel)
- Each GPU gets a copy of the model

### Resume Training

```bash
python training/train.py \
    resume_from_checkpoint=/path/to/checkpoint.ckpt
```

Or resume from last checkpoint:
```bash
python training/train.py \
    resume_from_checkpoint=auto
```

---

## 🖥️ SLURM Cluster Training

### Submit Job

**Create submission script** (`training/scripts/submit_slurm.sh`):
```bash
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

# Load modules (adjust for your cluster)
module load cuda/11.8
module load anaconda3

# Activate environment
source activate pose

# Training
cd $SLURM_SUBMIT_DIR
python training/train.py \
    machine=slurm \
    machine.batch_size=12 \
    max_epochs=100
```

**Submit**:
```bash
sbatch training/scripts/submit_slurm.sh
```

### Multi-Node Training

```bash
#!/bin/bash
#SBATCH --nodes=2                # 2 nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4        # 4 GPUs per node = 8 total
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00

# ... (same as above)

python training/train.py \
    machine=slurm \
    machine.batch_size=8 \
    max_epochs=100
```

**Important**: 
- SLURM automatically sets `SLURM_GPUS_ON_NODE` and `SLURM_NNODES`
- The training script reads these and configures accordingly
- Total GPUs = `nodes × gpus-per-node`

### Monitor Job

```bash
# Check status
squeue -u $USER

# View output
tail -f logs/vpog_<jobid>.out

# Cancel job
scancel <jobid>
```

---

## ⚙️ Configuration

### Configuration Structure

```
training/config/
├── train.yaml              # Main config
├── machine/
│   ├── local.yaml         # Local machine settings
│   └── slurm.yaml         # SLURM settings
├── data/
│   └── vpog_data.yaml     # Dataset configuration
└── model/
    └── vpog_base.yaml     # Model architecture
```

### Key Parameters

#### Training Parameters

```yaml
# training/config/train.yaml
max_epochs: 100
log_every_n_steps: 50
val_check_interval: 1.0      # Validate every epoch
learning_rate: 1e-4
weight_decay: 1e-2

# Loss weights
loss_weight_cls: 1.0         # Classification loss
loss_weight_dense: 1.0       # Dense flow loss
loss_weight_invis_reg: 1.0   # Invisibility regularization
loss_weight_center: 1.0      # Center flow loss
```

#### Machine Settings

**Local** (`training/config/machine/local.yaml`):
```yaml
name: local
batch_size: 12
num_workers: 8
num_gpus: 1
precision: 32                # or '16-mixed' for mixed precision
gradient_clip_val: 1.0
dryrun: false               # Set true for debugging
logger: tensorboard          # or 'wandb'
```

**SLURM** (`training/config/machine/slurm.yaml`):
```yaml
name: slurm
batch_size: 12
num_workers: 10
precision: 32
gradient_clip_val: 1.0
dryrun: false
logger: tensorboard
```

#### Dataset Settings

```yaml
# training/config/data/vpog_data.yaml
dataset_name: gso
templates_dir: datasets/templates
num_templates: 4              # Number of templates per query
patch_size: 16                # Patch size (must match model)
image_size: 224               # Input image size
```

#### Model Settings

```yaml
# training/config/model/vpog_base.yaml
croco_checkpoint: checkpoints/CroCo_V2_ViTBase_BaseDecoder.pth
patch_size: 16
enc_embed_dim: 768
enc_depth: 12
enc_num_heads: 12
aa_depth: 4
aa_num_heads: 12
aa_window_size: 7
num_template_added_tokens: 1  # Unseen token
cls_tau: 1.0
flow_hidden_dim: 256
flow_num_layers: 3
```

### Override Configuration

**Command line overrides**:
```bash
# Single parameter
python training/train.py max_epochs=50

# Multiple parameters
python training/train.py \
    max_epochs=50 \
    machine.batch_size=8 \
    learning_rate=5e-5

# Nested parameters
python training/train.py \
    data.num_templates=6 \
    model.aa_depth=6
```

**Config file override**:
```bash
# Use custom config
python training/train.py --config-path=/path/to/configs --config-name=my_config
```

---

## 📊 Monitoring

### TensorBoard (Default)

**Start TensorBoard**:
```bash
tensorboard --logdir=/path/to/save_dir/tensorboard --port=6006
```

**View in browser**:
```
http://localhost:6006
```

**Available metrics**:
- `train_loss` - Total training loss
- `train_cls_loss` - Classification loss
- `train_dense_loss` - Dense flow loss  
- `train_center_loss` - Center flow loss
- `learning_rate` - Current learning rate
- `val_loss` - Validation loss (if enabled)

### Weights & Biases (WandB)

**Enable WandB**:
```bash
python training/train.py \
    machine.logger=wandb \
    user.wandb_api_key=<your-key>
```

Or set environment variable:
```bash
export WANDB_API_KEY=<your-key>
python training/train.py machine.logger=wandb
```

**Offline mode** (for clusters without internet):
```bash
python training/train.py \
    machine.logger=wandb \
    machine.dryrun=true
```

Then sync later:
```bash
wandb sync /path/to/wandb/run
```

### Checkpoints

**Location**:
```
{save_dir}/{name_exp}/checkpoints/
├── last.ckpt              # Most recent checkpoint
├── epoch=099.ckpt         # Final epoch
└── epoch=050-val_loss=0.0234.ckpt  # Best validation
```

**Load checkpoint**:
```python
from training.lightning_module import VPOGLightningModule

model = VPOGLightningModule.load_from_checkpoint(
    '/path/to/checkpoint.ckpt'
)
```

---

## 🔧 Troubleshooting

### Out of Memory (OOM)

**Symptoms**: `CUDA out of memory` error

**Solutions**:
1. **Reduce batch size**:
   ```bash
   python training/train.py machine.batch_size=4
   ```

2. **Enable mixed precision**:
   ```bash
   python training/train.py machine.precision=16-mixed
   ```

3. **Reduce number of workers**:
   ```bash
   python training/train.py machine.num_workers=4
   ```

4. **Use gradient accumulation**:
   ```bash
   python training/train.py \
       machine.batch_size=4 \
       machine.accumulate_grad_batches=4  # Effective batch size = 16
   ```

### Slow Data Loading

**Symptoms**: Training pauses between batches

**Solutions**:
1. **Increase workers**:
   ```bash
   python training/train.py machine.num_workers=8
   ```

2. **Enable pin memory** (already enabled by default)

3. **Use persistent workers** (already enabled by default)

### NaN Loss

**Symptoms**: Loss becomes NaN during training

**Solutions**:
1. **Reduce learning rate**:
   ```bash
   python training/train.py learning_rate=5e-5
   ```

2. **Enable gradient clipping** (already enabled by default)

3. **Check data quality**:
   ```bash
   python training/dataloader/test_integration.py
   ```

### SLURM Job Fails

**Common issues**:

1. **Module not found**:
   - Check Python path in job script
   - Ensure environment is activated

2. **CUDA not available**:
   - Verify GPU allocation: `#SBATCH --gpus-per-node=X`
   - Check CUDA module: `module load cuda/11.8`

3. **Out of time**:
   - Increase time limit: `#SBATCH --time=72:00:00`
   - Or enable checkpointing and resume

4. **Out of memory**:
   - Reduce batch size
   - Request more memory: `#SBATCH --mem=256G`

### Validation Not Running

**Issue**: No validation metrics

**Solution**: Enable validation:
```bash
python training/train.py val_check_interval=1.0
```

Or check every N batches:
```bash
python training/train.py val_check_interval=1000  # Every 1000 batches
```

---

## 📁 Directory Structure After Training

```
{save_dir}/{name_exp}/
├── checkpoints/
│   ├── last.ckpt
│   ├── epoch=099.ckpt
│   └── best.ckpt
├── tensorboard/
│   └── {name_exp}/
│       └── events.out.tfevents.*
├── visualizations/           # If visualization enabled
│   ├── batch_000.png
│   └── batch_001.png
└── config.yaml              # Saved configuration
```

---

## 🎯 Recommended Training Workflow

### 1. **Sanity Check** (5-10 minutes)
```bash
# Quick test with small batch and limited data
python training/train.py \
    machine.batch_size=2 \
    max_epochs=2 \
    limit_train_batches=10 \
    log_every_n_steps=1
```

### 2. **Prototype** (1-2 hours)
```bash
# Train for a few epochs to verify convergence
python training/train.py \
    machine.batch_size=8 \
    max_epochs=10 \
    log_every_n_steps=10
```

### 3. **Full Training** (2-3 days)
```bash
# Full training with validation
python training/train.py \
    machine=local \
    machine.batch_size=12 \
    max_epochs=100 \
    val_check_interval=1.0
```

Or on SLURM:
```bash
sbatch training/scripts/submit_slurm.sh
```

### 4. **Evaluation**
```bash
# Test on BOP benchmark
python test.py --checkpoint=/path/to/best.ckpt
```

---

## 📚 Additional Resources

- **Model Architecture**: See [vpog/IMPLEMENTATION_SUMMARY.md](../vpog/IMPLEMENTATION_SUMMARY.md)
- **Data Format**: See [training/dataloader/README.md](dataloader/README.md)
- **Loss Functions**: See [training/losses/](losses/)
- **Inference Pipeline**: See [INFERENCE_PIPELINE_COMPLETE.md](../INFERENCE_PIPELINE_COMPLETE.md)

---

## 💡 Tips & Best Practices

1. **Always test dataloader first**:
   ```bash
   python training/dataloader/test_integration.py
   ```

2. **Use mixed precision for faster training**:
   ```bash
   python training/train.py machine.precision=16-mixed
   ```

3. **Monitor GPU utilization**:
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Save configuration**:
   ```bash
   python training/train.py --cfg job > config_used.yaml
   ```

5. **Profile training** (find bottlenecks):
   ```bash
   python training/train.py \
       machine.trainer.profiler=simple \
       max_epochs=1
   ```

6. **Resume from best checkpoint**:
   - Training saves checkpoints automatically
   - Resume with `resume_from_checkpoint=path/to/last.ckpt`
   - For inference, use best validation checkpoint

---

## 🆘 Getting Help

If you encounter issues:

1. Check this README first
2. Review error messages carefully
3. Test individual components:
   - Dataloader: `python training/dataloader/test_integration.py`
   - Model: `python -c "from training.lightning_module import *"`
4. Check GitHub issues
5. Enable debug logging:
   ```bash
   python training/train.py machine.dryrun=true log_every_n_steps=1
   ```

---

**Last Updated**: December 21, 2025

**Status**: ✅ Ready for training
