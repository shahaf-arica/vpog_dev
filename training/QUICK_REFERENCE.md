# VPOG Training - Quick Reference

One-page reference for common training scenarios.

## 🚀 Quick Commands

### Local Training

```bash
# Basic (single GPU)
python training/train.py

# Specific GPU
CUDA_VISIBLE_DEVICES=0 python training/train.py

# Multi-GPU (2 GPUs)
python training/train.py machine.num_gpus=2

# Fast prototyping
python training/train.py max_epochs=10 machine.batch_size=4

# Resume from checkpoint
python training/train.py resume_from_checkpoint=/path/to/last.ckpt
```

### SLURM Training

```bash
# Submit single-node job
sbatch training/scripts/submit_slurm.sh

# Submit multi-node job (2 nodes × 4 GPUs = 8 GPUs)
sbatch training/scripts/submit_slurm_multinode.sh

# Check job status
squeue -u $USER

# View logs
tail -f logs/vpog_<jobid>.out

# Cancel job
scancel <jobid>
```

## ⚙️ Common Parameter Overrides

```bash
# Change batch size
python training/train.py machine.batch_size=16

# Change learning rate
python training/train.py learning_rate=5e-5

# Change number of epochs
python training/train.py max_epochs=200

# Change dataset
python training/train.py data.dataset_name=shapenet

# Change number of templates
python training/train.py data.num_templates=6

# Multiple overrides
python training/train.py \
    max_epochs=50 \
    machine.batch_size=8 \
    learning_rate=1e-4 \
    data.num_templates=6
```

## 📊 Monitoring

```bash
# TensorBoard
tensorboard --logdir=/path/to/save_dir/tensorboard --port=6006

# WandB (set API key first)
export WANDB_API_KEY=<your-key>
python training/train.py machine.logger=wandb

# Check GPU usage
watch -n 1 nvidia-smi
```

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | `machine.batch_size=4` or `machine.precision=16-mixed` |
| Slow data loading | `machine.num_workers=8` |
| NaN loss | `learning_rate=5e-5` |
| Checkpoint not saving | Check `cfg.checkpoint_dir` exists |

## 📁 File Locations

```
training/
├── train.py                  # Main training script
├── config/
│   ├── train.yaml           # Main config
│   ├── machine/local.yaml   # Local settings
│   └── machine/slurm.yaml   # SLURM settings
├── scripts/
│   ├── submit_slurm.sh      # SLURM job script
│   └── test_training_setup.py  # Test setup
└── TRAINING_README.md        # Full documentation

{save_dir}/{name_exp}/
├── checkpoints/             # Model checkpoints
│   ├── last.ckpt
│   └── best.ckpt
└── tensorboard/             # TensorBoard logs
```

## ✅ Pre-Training Checklist

- [ ] Conda environment activated: `conda activate pose`
- [ ] Templates downloaded to `datasets/templates/`
- [ ] Test dataloader: `python training/dataloader/test_integration.py`
- [ ] Test setup (optional): `python training/scripts/test_training_setup.py`
- [ ] Configuration reviewed: `cat training/config/train.yaml`
- [ ] Save directory accessible: Check `save_dir` in config
- [ ] GPU available: `nvidia-smi`

## 🎯 Training Workflow

1. **Test** (1 min): `python training/scripts/test_training_setup.py`
2. **Prototype** (10 min): `python training/train.py max_epochs=2 limit_train_batches=10`
3. **Full Training** (2-3 days): `sbatch training/scripts/submit_slurm.sh`
4. **Monitor**: Check TensorBoard or WandB
5. **Evaluate**: `python test.py --checkpoint=/path/to/best.ckpt`

## 📞 Getting Help

- Full docs: `training/TRAINING_README.md`
- Configuration: Check `training/config/*.yaml`
- Test components: Run test scripts in `training/dataloader/`
- Check logs: `tail -f logs/vpog_<jobid>.out`

---

**Quick Start**: `python training/train.py` → Done! 🎉
