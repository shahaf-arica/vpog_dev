# VPOG Hydra Configuration Architecture

## Overview

The VPOG model now uses **Hydra configuration and instantiation** throughout, supporting multiple encoder backends (CroCo, DINOv2) with a unified interface.

## Key Changes

### 1. Generic Encoder Wrapper

**File**: `vpog/models/encoder_wrapper.py`

- Supports multiple backends: `croco`, `dinov2`
- Unified interface: returns `[B, N, D]` features and `[B, N, 2]` positions
- Handles encoder-specific differences internally

```python
encoder = EncoderWrapper(
    encoder_type='croco',  # or 'dinov2'
    checkpoint_name='large',  # CroCo-specific
    # or model_name='dinov2_vitl14'  # DINOv2-specific
)
features, positions = encoder(images)
```

### 2. Hydra-Based Model Construction

**File**: `vpog/models/vpog_model.py`

VPOGModel now accepts **pre-instantiated components**:

```python
model = VPOGModel(
    encoder=encoder,  # Hydra-instantiated
    aa_module=aa_module,  # Hydra-instantiated
    classification_head=classification_head,  # Hydra-instantiated
    flow_head=flow_head,  # Hydra-instantiated
    token_manager=token_manager,  # Hydra-instantiated
    s2rope_config=cfg.s2rope_config,
    img_size=224,
    patch_size=16,
)
```

### 3. Configuration Files

#### CroCo Configuration

**File**: `training/config/model/vpog.yaml`

```yaml
# Encoder - Hydra instantiation
encoder:
  _target_: vpog.models.encoder_wrapper.EncoderWrapper
  encoder_type: croco
  checkpoint_name: large  # 'base' or 'large'
  pretrained_path: null  # Auto-detects
  freeze_encoder: false

# Token Manager
token_manager:
  _target_: vpog.models.token_manager.TokenManager
  num_query_added_tokens: 0
  num_template_added_tokens: 1
  embed_dim: 1024  # Match encoder

# AA Module
aa_module:
  _target_: vpog.models.aa_module.AAModule
  dim: 1024
  depth: 6
  num_heads: 16
  # ... other params

# Classification Head
classification_head:
  _target_: vpog.models.classification_head.ClassificationHead
  dim: 1024
  tau: 1.0
  # ... other params

# Flow Head
flow_head:
  _target_: vpog.models.flow_head.FlowHead
  dim: 1024
  patch_size: 16
  # ... other params

# S²RoPE config
s2rope_config:
  head_dim: 64  # dim / num_heads
  n_faces: 6
```

#### DINOv2 Configuration

**File**: `training/config/model/vpog_dinov2.yaml`

```yaml
encoder:
  _target_: vpog.models.encoder_wrapper.EncoderWrapper
  encoder_type: dinov2
  model_name: dinov2_vitl14  # DINOv2-Large
  pretrained: true
  freeze_encoder: false
  patch_size: 14  # DINOv2 uses 14x14
  img_size: 224

# ... rest similar to CroCo config
```

### 4. Usage Pattern

#### In Training/Testing Code

```python
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

# Load config from training/config/
config_dir = "training/config"
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name="model/vpog")  # or "model/vpog_dinov2"

# Instantiate all components
encoder = instantiate(cfg.encoder)
token_manager = instantiate(cfg.token_manager)
aa_module = instantiate(cfg.aa_module)
classification_head = instantiate(cfg.classification_head)
flow_head = instantiate(cfg.flow_head)

# Create model
model = VPOGModel(
    encoder=encoder,
    aa_module=aa_module,
    classification_head=classification_head,
    flow_head=flow_head,
    token_manager=token_manager,
    s2rope_config=cfg.s2rope_config,
    img_size=cfg.img_size,
    patch_size=cfg.patch_size,
)
```

#### Running Tests

```bash
# Test with CroCo encoder
python -m training.test_vpog_full_pipeline

# Test with DINOv2 encoder
python -m training.test_vpog_full_pipeline model=vpog_dinov2

# Test with real data
python -m training.test_vpog_full_pipeline --real_data
```

## Benefits

1. ✅ **Flexible encoder selection**: Switch between CroCo and DINOv2 via config
2. ✅ **Hydra-driven**: All components instantiated from config
3. ✅ **Type safety**: Pre-instantiated components with proper types
4. ✅ **Easy experimentation**: Change encoder/params without code changes
5. ✅ **Unified interface**: Encoder wrapper handles backend differences
6. ✅ **Config-first**: Testing includes full config loading

## File Structure

```
training/
├── config/
│   ├── model/
│   │   ├── vpog.yaml          # CroCo-based config (default)
│   │   └── vpog_dinov2.yaml   # DINOv2-based config
│   ├── data/
│   │   └── vpog_data.yaml     # Dataloader config
│   ├── machine/
│   │   ├── local.yaml
│   │   └── slurm.yaml
│   ├── test_vpog.yaml         # Test configuration
│   └── train.yaml             # Training configuration
├── dataloader/
├── losses/
├── visualization/
└── test_vpog_full_pipeline.py

vpog/models/
├── encoder_wrapper.py     # Generic encoder wrapper (NEW)
├── encoder.py             # Old CroCo-only encoder (deprecated)
├── vpog_model.py          # Updated to accept instantiated components
├── aa_module.py           # Unchanged
├── classification_head.py # Unchanged
├── flow_head.py           # Unchanged
└── token_manager.py       # Unchanged

training/
└── test_vpog_full_pipeline.py  # Updated to use Hydra
```

## Migration Notes

### Old Pattern (Deprecated)
```python
# Old: Config dicts passed to VPOGModel
model = VPOGModel(
    encoder_config={'checkpoint_name': 'large', ...},
    aa_config={'depth': 6, ...},
    # ...
)
```

### New Pattern (Current)
```python
# New: Hydra instantiation
cfg = compose(config_name="model/vpog")
encoder = instantiate(cfg.encoder)
aa_module = instantiate(cfg.aa_module)
# ...
model = VPOGModel(encoder=encoder, aa_module=aa_module, ...)
```

## Next Steps

1. Update training script to use Hydra instantiation
2. Create configs for different model variants (small, base, large)
3. Add encoder ablation configs (frozen vs fine-tuned)
4. Integrate with main training pipeline
