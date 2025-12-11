# VPOG Dataloader Test Guide

## Overview

All dataloader components now have **REAL DATA TESTS** that:
- Load actual GSO dataset objects
- Process real images, depth maps, and poses
- Generate visualizations to prove correctness
- Save results to `tmp/` directory for inspection

## Running Tests

### Prerequisites
```bash
# Make sure GSO templates are rendered
python -m src.scripts.render_gso_templates

# Set PYTHONPATH
export PYTHONPATH=/data/home/ssaricha/gigapose:$PYTHONPATH
```

### 1. Template Selector Test
**File**: `training/dataloader/template_selector.py`

**What it tests**:
- Loads real GSO objects (3 objects)
- Loads actual template poses from `datasets/templates/gso/object_poses/*.npy`
- Selects S_p positive templates (nearest by angle)
- Selects S_n negative templates (≥90° angular distance)
- Extracts d_ref unit vectors from poses
- Computes angular distances between all templates

**Visualizations**:
- `tmp/template_selector_test/sample_XX_objYYYYYY_selection.png`
  - Shows query image
  - Shows 3-4 positive templates (green borders) with angular distances
  - Shows 2 negative templates (red borders) with angular distances
  - Displays d_ref values

**Statistics**:
- Min/mean/max angles for positive templates
- Min/mean/max angles for negative templates
- Computed over 100 random samples

**Run**:
```bash
python training/dataloader/template_selector.py
```

### 2. Flow Computer Test
**File**: `training/dataloader/flow_computer.py`

**What it tests**:
- Loads real GSO object
- Loads 2 template views with known pose difference
- Loads actual RGB images and depth maps
- Computes pixel-level flow between template and query patches
- Tests visibility checking with real masks
- Computes flows for a 3x3 grid of patches

**Visualizations**:
- `tmp/flow_computer_test/objXXXXXX_images_with_patches.png`
  - Template and query images side by side
  - Yellow patch grid overlay
  - Red crosses marking patch centers
  
- `tmp/flow_computer_test/objXXXXXX_flow_fields.png`
  - 3x3 grid of flow field visualizations
  - Color-coded flow magnitude (jet colormap)
  - White arrows showing flow direction
  - Shows validity percentage and mean magnitude per patch

**Statistics**:
- Mean/max/std flow magnitude across all patches
- Mean visibility percentage
- Flow vectors in pixel coordinates

**Run**:
```bash
python training/dataloader/flow_computer.py
```

### 3. Visualization Utils Test
**File**: `training/dataloader/vis_utils.py`

**What it tests**:
- Loads real GSO objects (3 objects)
- Creates VPOGBatch with real query and template images
- Tests all visualization functions:
  - `visualize_vpog_sample()`: Query + templates overview
  - `visualize_flow_field()`: Flow field with arrows
  - `visualize_patch_grid()`: Patch grid overlay
  - `visualize_vpog_batch()`: Full batch visualization

**Visualizations**:
- `tmp/vis_utils_test/objXXXXXX_overview.png`
  - Query image with d_ref info
  - 4 template images (positive in green, negative in red)
  
- `tmp/vis_utils_test/objXXXXXX_patches.png`
  - Query and templates with 16x16 patch grid overlay
  
- `tmp/vis_utils_test/objXXXXXX_flow_tX.png`
  - Flow magnitude heatmap
  - Flow X/Y components
  - Visibility masks
  
- `tmp/vis_utils_test/batch/batch0000_sampleX_*.png`
  - Complete batch visualization for all samples

**Run**:
```bash
python training/dataloader/vis_utils.py
```

### 4. Full Dataset Integration Test
**File**: `training/dataloader/vpog_dataset.py`

**What it tests**:
- Full pipeline: WebSceneDataset → Template Selection → VPOGBatch
- Initializes VPOGTrainDataset with real GSO data
- Creates PyTorch DataLoader
- Loads multiple batches
- Validates all tensor shapes and data ranges
- Counts positive vs negative templates
- Tests collate function

**Visualizations**:
- Uses `visualize_vpog_batch()` to generate complete visualization
- Saved to `tmp/vpog_dataset_test/`

**Validation checks**:
- ✓ Images shape: (B, S+1, 3, H, W)
- ✓ Images range: [0, 1]
- ✓ d_ref normalized: ||d_ref|| = 1.0
- ✓ Positive/negative template counts match config
- ✓ Flow tensors have correct shapes
- ✓ All metadata properly loaded

**Run**:
```bash
python training/dataloader/vpog_dataset.py
```

## What Each Test Proves

### Template Selector ✓
- **Proves**: Template selection algorithm works correctly
- **Shows**: 
  - Positive templates are actually nearby views (small angles)
  - Negative templates are far views (≥90°)
  - d_ref extraction from pose matrices is correct
  - Visual confirmation with actual object images

### Flow Computer ✓
- **Proves**: Flow computation between patches is working
- **Shows**:
  - Flow fields visualized as color-coded magnitude
  - Flow directions shown as arrows
  - Visibility masks applied correctly
  - Patch-by-patch analysis with real depth maps

### Visualization Utils ✓
- **Proves**: All debug visualization functions work with real data
- **Shows**:
  - Query and template images properly loaded
  - Positive (green) vs negative (red) templates clearly marked
  - Flow fields with proper color mapping
  - Patch grids correctly overlaid
  - Complete batch structure visualization

### Full Dataset Integration ✓
- **Proves**: Complete pipeline works end-to-end
- **Shows**:
  - WebSceneDataset successfully loads scenes
  - Template selection integrated correctly
  - Batch creation produces correct tensor shapes
  - DataLoader can iterate over multiple batches
  - All components work together seamlessly

## Expected Output Structure

```
tmp/
├── template_selector_test/
│   ├── sample_00_obj000001_selection.png
│   ├── sample_01_obj000002_selection.png
│   └── ...
├── flow_computer_test/
│   ├── obj000001_images_with_patches.png
│   └── obj000001_flow_fields.png
├── vis_utils_test/
│   ├── obj000001_overview.png
│   ├── obj000001_patches.png
│   ├── obj000001_flow_t0.png
│   ├── obj000001_flow_t1.png
│   └── batch/
│       ├── batch0000_sample0_overview.png
│       ├── batch0000_sample0_patches.png
│       └── ...
└── vpog_dataset_test/
    ├── batch0000_sample0_overview.png
    ├── batch0000_sample0_patches.png
    └── ...
```

## Troubleshooting

### "Templates not found"
```bash
# Run template rendering
python -m src.scripts.render_gso_templates --dataset gso --level 1
```

### "No module named 'src'"
```bash
# Set PYTHONPATH
export PYTHONPATH=/data/home/ssaricha/gigapose:$PYTHONPATH
```

### "Dataset directory not found"
Check that GSO dataset is at `datasets/gso/` with:
- `models_info.json`
- `train/` directory with scenes
- Object models in `models/`

## Next Steps

After all tests pass:

1. **Verify Visualizations**: Manually inspect all generated images in `tmp/`
   - Do positive templates look similar to query?
   - Do negative templates look different?
   - Are flow fields reasonable?
   - Are patch grids aligned correctly?

2. **Run Integration Test**: Execute `training/dataloader/test_integration.py` for complete system test

3. **Implement Model**: Move to `vpog/models/` to implement:
   - CroCo encoder integration
   - S2RoPE position embeddings
   - Patch matching head
   - EPro-PnP pose estimation

4. **Implement Loss**: Create loss functions for:
   - Flow supervision
   - Visibility masks
   - Pose estimation

5. **Training Loop**: Implement PyTorch Lightning training module

## Configuration Notes

All tests use production configuration values:
- `num_positive_templates`: 4
- `num_negative_templates`: 2  
- `min_negative_angle_deg`: 90.0
- `patch_size`: 16
- `image_size`: (224, 224)
- `num_patches_per_side`: 14

These match the VPOG paper specifications and CroCo architecture requirements.
