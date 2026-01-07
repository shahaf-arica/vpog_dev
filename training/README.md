# VPOG Training

Training infrastructure for **VPOG (Visual Patch-wise Object pose estimation with Groups of templates)**.

## 🚀 Quick Start

```bash
# 1. Activate environment
conda activate pose

# 2. Test setup (optional but recommended)
python training/scripts/test_training_setup.py

# 3. Start training
python training/train.py
```

**That's it!** Training will start with default settings.

### Training Modes

**Local Training:**
```bash
# Single GPU
python training/train.py machine=local

# Multi-GPU (e.g., 2 GPUs)
python training/train.py machine=local machine.num_gpus=2
```

**SLURM Cluster:**
```bash
# Single node (4 GPUs)
sbatch training/scripts/submit_slurm.sh

# Multi-node (2 nodes × 4 GPUs = 8 GPUs)
sbatch training/scripts/submit_slurm_multinode.sh
```

## 📚 Documentation

- **[TRAINING_README.md](TRAINING_README.md)** - Complete training guide (local & SLURM, ~600 lines)
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - One-page command reference
- **[TEST_GUIDE.md](TEST_GUIDE.md)** - Testing framework documentation

## 📁 What's Included

```
training/
├── train.py                     # Main training script ✓
├── lightning_module.py          # PyTorch Lightning module ✓
├── TRAINING_README.md           # Complete training documentation ✓
├── QUICK_REFERENCE.md           # One-page reference ✓
│
├── config/                      # Hydra configuration files ✓
│   ├── train.yaml              # Main training config
│   ├── machine/                # Local/SLURM settings
│   │   ├── local.yaml
│   │   └── slurm.yaml
│   ├── data/                   # Dataset configs
│   └── model/                  # Model configs
│
├── scripts/                     # Helper scripts ✓
│   ├── submit_slurm.sh        # SLURM job submission (single node)
│   ├── submit_slurm_multinode.sh  # Multi-node training
│   └── test_training_setup.py # Validate setup before training
│
├── dataloader/                  # VPOG dataloader implementation ✓
│   ├── vpog_dataset.py         # Main dataset class
│   ├── template_selector.py    # Template selection (S_p + S_n)
│   ├── flow_computer.py        # Flow label computation (16×16)
│   └── test_integration.py     # Dataloader tests
│
├── losses/                      # Loss functions ✓
│   ├── classification_loss.py  # Cross-entropy for template matching
│   ├── flow_loss.py            # Masked L1/Huber for flow
│   └── epro_pnp_loss.py        # EPro-PnP pose loss
│
└── visualization/               # Training visualization ✓
    └── flow_vis.py             # Flow visualization (HSV)
```



Ensure GigaPose environment is set up:
```bash
conda activate gigapose
```

### 2. Prepare Data

Ensure GSO/ShapeNet training data and templates are available:
```bash
# Check data exists
ls datasets/gso/train_pbr_web/
ls datasets/templates/gso/

# If templates not available, render them:
python -m src.scripts.render_gso_templates
```

### 3. Test Pipeline

```bash
# Test dataloader only (no VPOG model dependencies)
python -m training.dataloader.test_integration

# Test pixel-level flow computation
python -m training.dataloader.test_pixel_flow

# Test full VPOG pipeline with synthetic data
python -m training.test_vpog_full_pipeline

# Test full VPOG pipeline with real GSO data
python -m training.test_vpog_full_pipeline --real_data
```

### 4. Usage in Training

```python
from hydra import compose, initialize
from hydra.utils import instantiate
from training.dataloader import VPOGTrainDataset
from src.utils.dataloader import NoneFilteringDataLoader

# Load config
with initialize(config_path="config"):
    cfg = compose(config_name="train.yaml")

# Create dataset
dataset = instantiate(cfg.data.dataloader)

# Create dataloader
dataloader = NoneFilteringDataLoader(
    dataset.web_dataloader.datapipeline,
    batch_size=cfg.machine.batch_size,
    num_workers=cfg.machine.num_workers,
    collate_fn=dataset.collate_fn,
)

# Training loop
for batch_idx, batch in enumerate(dataloader):
    if batch is None:
        continue
    
    # batch is VPOGBatch with:
    # - images: [B, S+1, 3, H, W]
    # - d_ref: [B, 3]
    # - flows: [B, S, H_p, W_p, 2]
    # - etc.
    
    # Your training code here
    pass
```

## Configuration

Main config file: `training/config/train.yaml`

Key settings:
```yaml
# Dataset selection
train_dataset_id: 0  # 0=GSO, 1=ShapeNet, 2=both

# Training
max_epochs: 100
batch_size: 8  # Defined in machine config

# Data config (in config/data/vpog_data.yaml)
dataloader:
  num_positive_templates: 4    # S_p
  num_negative_templates: 2    # S_n
  patch_size: 16               # CroCo default
  image_size: 224
  
# Visualization (for debugging)
visualization:
  enabled: false
  save_dir: ${vis_dir}
```

See `training/config/data/vpog_data.yaml` for full dataloader configuration.

## Dataloader

The VPOG dataloader provides:
- **Query images** from GSO/ShapeNet
- **S templates** per query (S_p positive + S_n negative)
- **d_ref** (reference direction) for S2RoPE
- **Flow labels** between template and query patches

Output format: `[B, S+1, C, H, W]` where S+1 = 1 query + S templates

See `training/dataloader/README.md` for detailed documentation.

## Development Status

### ✅ Completed
- [x] Configuration system (Hydra)
- [x] Dataloader implementation
  - [x] Template selection (S_p + S_n)
  - [x] d_ref extraction
  - [x] Flow computation framework
  - [x] Batch formatting
- [x] Visualization tools
- [x] Unit tests
- [x] Integration tests
- [x] Documentation

### 🚧 In Progress / TODO
- [ ] Full flow computation implementation (currently placeholder)
- [ ] Model implementation (vpog/models/)
- [ ] Loss functions
- [ ] Training loop with PyTorch Lightning
- [ ] S2RoPE implementation
- [ ] CroCo encoder integration
- [ ] EPro-PnP differentiable PnP layer integration

## Next Steps

### 1. Implement VPOG Model

Create model components in `vpog/models/`:
- Encoder (CroCo-based)
- S2RoPE embeddings
- Flow prediction head
- Classification head for patch matching

### 2. Implement Training Logic

- Loss functions (classification + flow + coefficients)
- PyTorch Lightning module
- Training/validation loops
- Metrics and logging

### 3. Optimize Flow Computation

Current implementation uses placeholder logic for flow labels. Optimize:
- Efficient batch processing
- GPU acceleration if needed
- Caching for repeated template-query pairs

## Testing

### Unit Tests

Each component has standalone tests with `if __name__ == "__main__"` blocks:

```bash
python training/dataloader/template_selector.py
python training/dataloader/flow_computer.py
python training/dataloader/vis_utils.py
```

### Integration Test

Full pipeline test:
```bash
python training/dataloader/test_integration.py
```

This tests:
- Dataset initialization
- Batch loading
- Shape verification
- Data properties
- Visualization generation

### Debug Visualizations

Enable in config to check data correctness:
```yaml
# training/config/data/vpog_data.yaml
visualization:
  enabled: true
  save_dir: ./debug_visualizations
  save_frequency: 10
```

## Training on SLURM

For distributed training on SLURM cluster:

```bash
# Use SLURM machine config
python train.py machine=slurm machine.gpus=4 machine.nodes=1

# Or with sbatch script
sbatch scripts/train_slurm.sh
```

SLURM config (`training/config/machine/slurm.yaml`) includes:
- DDP strategy
- Multi-GPU settings
- Sync batch norm
- Gradient accumulation

## Architecture

### Data Flow

```
WebSceneDataset (Query)
    ↓
TemplateSelector → Select S_p + S_n templates
    ↓              Extract d_ref
TemplateDataset → Load S templates
    ↓
FlowComputer → Compute flow labels
    ↓
VPOGBatch → [B, S+1, C, H, W] format
    ↓
Model (to be implemented)
```

### Key Concepts

**Template Selection:**
- Based on out-of-plane rotation (viewing direction)
- S_p nearest templates by angular distance
- S_n random negatives with ≥90° difference

**d_ref (Reference Direction):**
- Unit vector from template origin to camera
- Extracted from nearest template (or random)
- Used for S2RoPE (spherical 2D rotary embeddings)

**Flow Labels:**
- Pixel-level flow in patch-local coordinates
- Visibility: depth + mask based
- Patch visibility: flow stays within patch

## References

### GigaPose
- Dataloader: `src/dataloader/train.py`
- Templates: `src/custom_megapose/template_dataset.py`
- Paper: [GigaPose: Fast and Robust Novel Object Pose Estimation via One Correspondence](https://arxiv.org/abs/2311.14155)

### VGGT
- Training structure: `external/vggt/training/`
- Data augmentation patterns

### CroCo
- Encoder architecture: `external/croco/`
- Patch size: 16x16 (default)

### EPro-PnP
- Differentiable PnP: `external/epropnp/`
- For learning flow coefficients

## Troubleshooting

### Data Issues
- Check datasets exist: `ls datasets/gso/`
- Check templates rendered: `ls datasets/templates/gso/`
- Run integration test: `python training/dataloader/test_integration.py`

### Shape Mismatches
- Verify `patch_size=16` and `image_size=224` are consistent
- Check `num_patches = image_size // patch_size = 14`

### Performance
- Increase `num_workers` in config
- Reduce batch size if OOM
- Enable template preprocessing (see GigaPose)

### Debug Mode
Enable visualizations in config to check data:
```yaml
visualization:
  enabled: true
```

## Contact & Support

For questions or issues:
1. Check README.md files in each directory
2. Run unit tests to isolate issues
3. Check visualizations for data correctness
4. Refer to GigaPose/VGGT code for reference implementations
