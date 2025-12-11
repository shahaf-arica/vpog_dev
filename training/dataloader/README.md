# VPOG Dataloader

This directory contains the dataloader implementation for **VPOG (Visual Patch-wise Object pose estimation with Groups of templates)**.

## Overview

The VPOG dataloader is designed to:
1. Load query images from GSO/ShapeNet training datasets
2. Select S = S_p + S_n templates per query:
   - **S_p**: Nearest templates by out-of-plane rotation
   - **S_n**: Random negative templates with ≥90° angular difference
3. Extract **d_ref** (reference direction) for S2RoPE embeddings
4. Compute pixel-level **flow labels** between template and query patches
5. Return batches in format `[B, S+1, C, H, W]` where S+1 = 1 query + S templates

## Files Structure

```
training/dataloader/
├── __init__.py                 # Module exports
├── vpog_dataset.py            # Main dataset class (VPOGTrainDataset)
├── template_selector.py       # Template selection logic (S_p + S_n)
├── flow_computer.py           # Flow label computation
├── vis_utils.py               # Visualization utilities
├── test_integration.py        # Integration tests
└── README.md                  # This file
```

## Key Components

### 1. TemplateSelector (`template_selector.py`)

Handles template selection based on out-of-plane rotation:

```python
from training.dataloader import TemplateSelector

selector = TemplateSelector(
    level_templates=1,              # Icosphere level (1=162 views)
    pose_distribution="all",         # Use all hemisphere views
    num_positive=4,                  # S_p nearest templates
    num_negative=2,                  # S_n random negatives
    min_negative_angle_deg=90.0,     # Min angle for negatives
    d_ref_random_ratio=0.0,          # Ratio of random vs nearest d_ref
)

# Select templates for a query
result = selector.select_templates(query_pose, return_d_ref=True)
# Returns: positive_indices, negative_indices, d_ref, d_ref_source
```

**d_ref Extraction:**
- d_ref is the unit vector from template origin to camera
- Extracted from the nearest out-of-plane template (or random with ratio)
- Used for S2RoPE (spherical 2D rotary position embeddings)

**Unit Test:**
```bash
python training/dataloader/template_selector.py
```

### 2. FlowComputer (`flow_computer.py`)

Computes pixel-level flow between template and query patches:

```python
from training.dataloader import FlowComputer

computer = FlowComputer(
    patch_size=16,                  # CroCo default
    compute_visibility=True,         # Use depth/mask for visibility
    compute_patch_visibility=True,   # Check if flow stays in patch
    visibility_threshold=0.1,        # Minimum visibility fraction
)

# Compute flow for a patch
result = computer.compute_flow_between_patches(
    query_patch_center, template_patch_center,
    query_K, template_K,
    query_pose, template_pose,
    query_depth, template_depth,
    query_mask, template_mask,
)
# Returns: flow, visibility, patch_visibility, valid_flow
```

**Flow Computation:**
- Flow is in patch-local coordinates (relative to patch center)
- Visibility checks: depth-based occlusion, mask-based visibility
- Patch visibility: checks if correspondence stays within patch bounds

**Unit Test:**
```bash
python training/dataloader/flow_computer.py
```

### 3. VPOGTrainDataset (`vpog_dataset.py`)

Main dataset class that combines all components:

```python
from training.dataloader import VPOGTrainDataset

dataset = VPOGTrainDataset(
    root_dir="/path/to/datasets",
    dataset_name="gso",              # or "shapenet"
    template_config={...},           # Template configuration
    num_positive_templates=4,        # S_p
    num_negative_templates=2,        # S_n
    patch_size=16,                   # Patch size for CroCo
    image_size=224,                  # Input image size
    batch_size=8,
)

# Use with DataLoader
from src.utils.dataloader import NoneFilteringDataLoader

dataloader = NoneFilteringDataLoader(
    dataset.web_dataloader.datapipeline,
    batch_size=8,
    num_workers=8,
    collate_fn=dataset.collate_fn,
)

# Iterate over batches
for batch in dataloader:
    # batch is VPOGBatch with:
    # - images: [B, S+1, 3, H, W]
    # - masks: [B, S+1, H, W]
    # - K: [B, S+1, 3, 3]
    # - poses: [B, S+1, 4, 4]
    # - d_ref: [B, 3]
    # - flows: [B, S, H_p, W_p, 2]
    # - visibility: [B, S, H_p, W_p]
    # etc.
    pass
```

**VPOGBatch Structure:**
```python
@dataclass
class VPOGBatch:
    images: torch.Tensor            # [B, S+1, 3, H, W]
    masks: torch.Tensor             # [B, S+1, H, W]
    K: torch.Tensor                 # [B, S+1, 3, 3]
    poses: torch.Tensor             # [B, S+1, 4, 4]
    d_ref: torch.Tensor             # [B, 3] - for S2RoPE
    template_indices: torch.Tensor  # [B, S]
    template_types: torch.Tensor    # [B, S] - 0=positive, 1=negative
    flows: torch.Tensor             # [B, S, H_p, W_p, 2]
    visibility: torch.Tensor        # [B, S, H_p, W_p]
    patch_visibility: torch.Tensor  # [B, S, H_p, W_p]
    infos: TensorCollection         # Metadata
```

### 4. Visualization (`vis_utils.py`)

Debug visualizations for checking data correctness:

```python
from training.dataloader import visualize_vpog_batch

# Visualize a batch
visualize_vpog_batch(
    batch,
    save_dir="./visualizations",
    batch_idx=0,
    max_samples=4,
)
```

**Generated Visualizations:**
- `batch{idx}_sample{i}_overview.png`: Query + templates with d_ref info
- `batch{idx}_sample{i}_patches.png`: Patch grid overlay
- `batch{idx}_sample{i}_flow_t{j}.png`: Flow fields for template j

**Unit Test:**
```bash
python training/dataloader/vis_utils.py
```

## Configuration

Configuration is in `training/config/data/vpog_data.yaml`:

```yaml
dataloader:
  _target_: training.dataloader.vpog_dataset.VPOGTrainDataset
  
  # Dataset
  dataset_name: gso
  root_dir: ${root_dir}
  
  # Template selection
  num_positive_templates: 4       # S_p
  num_negative_templates: 2       # S_n
  min_negative_angle_deg: 90.0
  d_ref_random_ratio: 0.0
  
  # Patch settings
  patch_size: 16
  image_size: 224
  
  # Flow computation
  flow_config:
    compute_visibility: true
    compute_patch_visibility: true
    visibility_threshold: 0.1
  
  # Augmentation (similar to GigaPose)
  transforms: ${transform}

# Visualization (for debugging)
visualization:
  enabled: false                  # Set to true to enable
  save_dir: ${vis_dir}
  save_frequency: 100
  max_samples_per_batch: 4
```

## Testing

### Unit Tests

Each component has standalone unit tests:

```bash
# Test template selector
python training/dataloader/template_selector.py

# Test flow computer
python training/dataloader/flow_computer.py

# Test visualization
python training/dataloader/vis_utils.py
```

### Integration Test

Full pipeline test with actual data:

```bash
python training/dataloader/test_integration.py
```

**Requirements for integration test:**
- GSO or ShapeNet training data in `datasets/gso` or `datasets/shapenet`
- Rendered templates in `datasets/templates/gso` or `datasets/templates/shapenet`

**What it tests:**
1. Dataset initialization
2. Batch loading
3. Shape verification
4. Data property checks (d_ref normalization, etc.)
5. Visualization generation

## Usage Examples

### Basic Usage

```python
from training.dataloader import VPOGTrainDataset
from src.utils.dataloader import NoneFilteringDataLoader

# Create dataset
dataset = VPOGTrainDataset(
    root_dir="/data/datasets",
    dataset_name="gso",
    template_config={...},
    num_positive_templates=4,
    num_negative_templates=2,
    patch_size=16,
    batch_size=8,
)

# Create dataloader
dataloader = NoneFilteringDataLoader(
    dataset.web_dataloader.datapipeline,
    batch_size=8,
    num_workers=8,
    collate_fn=dataset.collate_fn,
)

# Training loop
for batch_idx, batch in enumerate(dataloader):
    if batch is None:
        continue
    
    # batch.images: [B, S+1, 3, 224, 224]
    # batch.d_ref: [B, 3]
    # batch.flows: [B, S, 14, 14, 2] for patch_size=16
    
    # Your training code here
    pass
```

### With Visualization

```python
from training.dataloader import VPOGTrainDataset, visualize_vpog_batch

dataset = VPOGTrainDataset(...)
dataloader = ...

for batch_idx, batch in enumerate(dataloader):
    if batch is None:
        continue
    
    # Save visualizations every 100 batches
    if batch_idx % 100 == 0:
        visualize_vpog_batch(
            batch,
            save_dir="./debug_vis",
            batch_idx=batch_idx,
            max_samples=2,
        )
    
    # Training code
    pass
```

### With Hydra Config

```python
from hydra import compose, initialize
from hydra.utils import instantiate

with initialize(config_path="../config"):
    cfg = compose(config_name="train.yaml")

# Dataset will be instantiated with all config parameters
dataset = instantiate(cfg.data.dataloader)

# Or manually with config dict
dataset = VPOGTrainDataset(**cfg.data.dataloader)
```

## Data Flow

```
1. WebSceneDataset (GigaPose format)
   └─> Query RGB, depth, mask, pose, K
   
2. TemplateSelector
   └─> Select S_p nearest + S_n negative templates
   └─> Extract d_ref from nearest template
   
3. TemplateDataset (GigaPose format)
   └─> Load S template images, poses
   
4. FlowComputer
   └─> Compute pixel flows between patches
   └─> Compute visibility masks
   
5. VPOGBatch
   └─> Combine into [B, S+1, ...] format
   └─> Ready for model input
```

## Important Notes

### Template Selection

- **Out-of-plane rotation**: Based on camera viewing direction (not full 6D pose)
- Uses same definition as GigaPose's template matching
- Computed in OpenGL coordinate system for consistency
- d_ref is extracted from template poses (stored in `.npy` files)

### Flow Labels

- **Flow coordinates**: Patch-local (relative to patch center)
- **Visibility**: Considers both depth-based occlusion and mask visibility
- **Patch visibility**: Additional constraint that flow stays within patch bounds
- **Note**: Current implementation includes placeholder logic - full flow computation can be expensive and should be computed efficiently

### Data Format Compatibility

- **Query data**: Compatible with GigaPose's WebSceneDataset format
- **Template data**: Compatible with GigaPose's TemplateDataset format
- **Augmentations**: Same as GigaPose (RGB, crop, inplane rotation)
- **Normalization**: ImageNet statistics (for CroCo compatibility)

### Performance Considerations

- **Batch size**: Effectively `B * S` images need processing per batch
- **Flow computation**: Can be expensive for all patch pairs
- **Caching**: Templates can be preprocessed (see GigaPose's preprocessing)
- **Workers**: Use multiple workers for data loading

## Future Improvements

### Flow Computation
Currently uses placeholder logic. Full implementation should:
- Efficiently compute flows for all patch pairs
- Use depth reprojection with known poses
- Handle occlusions and visibility properly
- Consider patch-level visibility constraints

### Optimizations
- Cache computed flows (same template->query pairs may repeat)
- Precompute template features/embeddings
- Use GPU for flow computation if needed
- Batch template loading more efficiently

### Additional Features
- Support for ShapeNet dataset (currently GSO-focused)
- Online hard negative mining for S_n selection
- Adaptive S_p selection based on pose uncertainty
- Multi-scale patch flows

## Troubleshooting

### Common Issues

**"Dataset not found"**
- Check that GSO/ShapeNet data exists in `datasets/gso` or `datasets/shapenet`
- Verify templates are rendered in `datasets/templates/gso`

**"Template poses not found"**
- Template poses should be in `datasets/templates/gso/object_poses/*.npy`
- Run rendering script: `python -m src.scripts.render_gso_templates`

**Shape mismatches**
- Check `patch_size` and `image_size` are consistent
- Verify `num_positive_templates + num_negative_templates` matches model expectations

**Slow data loading**
- Increase `num_workers`
- Enable template preprocessing (see GigaPose code)
- Reduce batch size if memory constrained

### Debug Mode

Enable visualizations to check data:

```yaml
# In training/config/data/vpog_data.yaml
visualization:
  enabled: true
  save_dir: ./debug_visualizations
  save_frequency: 10  # Save every 10 batches
```

Or programmatically:

```python
from training.dataloader import save_visualization

for batch_idx, batch in enumerate(dataloader):
    save_visualization(
        batch, 
        save_dir="./debug",
        batch_idx=batch_idx,
        enabled=True,
    )
```

## References

- GigaPose dataloader: `src/dataloader/train.py`
- GigaPose templates: `src/custom_megapose/template_dataset.py`
- VGGT dataloader: `external/vggt/training/data/`
- CroCo: `external/croco/` (for patch size and encoder reference)

## Contact

For questions or issues with the dataloader, please check:
1. Unit tests pass: `python training/dataloader/template_selector.py`, etc.
2. Integration test passes: `python training/dataloader/test_integration.py`
3. Visualizations look correct: Check saved images in vis_dir
