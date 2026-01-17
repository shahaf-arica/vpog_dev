# BOP Validation Metrics Guide

## Overview

The validation step now includes **BOP Challenge metrics** for pose estimation evaluation:
- **MSPD** (Maximum Symmetry-Aware Projection Distance) - 2D reprojection error in pixels
- **MSSD** (Maximum Symmetry-Aware Surface Distance) - 3D surface distance as fraction of object diameter
- **AR** (Average Recall) - Rough estimation: AR = (2/3) * AR_MSSD + (1/3) * AR_MSPD

## Architecture

### Files Created/Modified

1. **`utils/bop_evaluation.py`** (NEW)
   - `BOPEvaluator` class: Main evaluation interface
   - `PoseEvaluation` dataclass: Single pose evaluation result
   - `BOPEvaluationResults` class: Aggregates results and computes AR
   - Uses `utils/eval_errors.py` for MSPD/MSSD computation
   - Uses `utils/bop_model.py` for model symmetries and point clouds

2. **`training/lightning_module.py`** (MODIFIED)
   - Added `BOPEvaluator` initialization in `__init__()`
   - Added `enable_pose_eval` flag (default: True)
   - Added `dataset_name` parameter (default: 'ycbv')
   - New methods:
     - `_evaluate_poses()`: Coordinate pose evaluation
     - `_build_correspondences_simple()`: Build 2D-3D correspondences (PLACEHOLDER)
     - `_solve_pnp_ransac()`: RANSAC PnP solver
     - `on_validation_epoch_start()`: Reset evaluator
     - `on_validation_epoch_end()`: Compute and log AR metrics

## How It Works

### 1. Validation Flow

```
validation_step (each batch):
  ├─ Compute losses (existing)
  ├─ Compute accuracy (existing)
  └─ For each sample in batch:
      ├─ Extract metadata (obj_id, scene_id, im_id)
      ├─ Build 2D-3D correspondences from model outputs
      ├─ Run RANSAC PnP to estimate pose
      ├─ Compute MSPD and MSSD errors
      └─ Add to BOPEvaluator

on_validation_epoch_end:
  ├─ Compute AR_MSPD (averaged over thresholds)
  ├─ Compute AR_MSSD (averaged over thresholds)
  ├─ Compute AR_combined = (2/3) * AR_MSSD + (1/3) * AR_MSPD
  └─ Log metrics to TensorBoard/console
```

### 2. Metrics Computed

**Per-Epoch Metrics:**
- `val/ar_mspd`: Average Recall for MSPD (2D reprojection)
- `val/ar_mssd`: Average Recall for MSSD (3D surface)
- `val/ar_combined`: Combined AR (rough BOP approximation without VSD)
- `val/mean_mspd_error`: Mean MSPD error in pixels
- `val/mean_mssd_error`: Mean MSSD error as fraction of diameter
- `val/num_pose_samples`: Number of successfully estimated poses

**Existing Metrics (unchanged):**
- `val/loss_total`, `val/loss_cls`, `val/loss_dense`, `val/loss_invis_reg`
- `val/acc_overall`, `val/acc_seen`, `val/acc_unseen`

### 3. Threshold-Based Evaluation

The evaluator uses standard BOP thresholds for AR computation:
- **MSPD thresholds**: [0.5, 0.4, 0.3, 0.2, 0.1] pixels (but actually these should be normalized)
- **MSSD thresholds**: [0.5, 0.4, 0.3, 0.2, 0.1] × object_diameter

For each threshold `τ`:
- Recall@τ = (# poses with error < τ) / (# total poses)

AR = mean(Recall@τ for all τ)

## Configuration

### Enable/Disable Pose Evaluation

**Dataset Name:**

The BOP evaluation dataset is controlled by **one single parameter** in [training/config/train.yaml](../training/config/train.yaml):

```yaml
# Validation dataset name (single source of truth)
val_dataset_name: ycbv
```

This controls:
- Which BOP dataset is loaded for validation (`data_val`)
- Which BOP models/symmetries are used for pose evaluation

**Enable/Disable Pose Evaluation:**

Edit [training/config/model/vpog.yaml](../training/config/model/vpog.yaml):

```yaml
# BOP Evaluation Configuration
enable_pose_eval: true  # Enable/disable pose estimation during validation
```

**Via command line:**

```bash
# Change validation dataset
python training/train.py val_dataset_name=tless

# Disable pose evaluation
python training/train.py model.enable_pose_eval=false

# Both together
python training/train.py \
  val_dataset_name=lmo \
  model.enable_pose_eval=true
```

### Dataset Names

Supported BOP datasets (via `utils/bop_model.py`):
- `ycbv`: YCB-Video
- `tless`: T-LESS
- `lmo`: LineMOD-Occluded
- `tudl`: TUD-Light
- `icbin`: IC-BIN
- `itodd`: ITODD
- `hb`: HomebrewedDB
- `hope`: HOPE

## Current Status

### ✅ Completed
- BOPEvaluator infrastructure
- MSPD/MSSD error computation with symmetries
- AR computation with multiple thresholds
- Integration into validation step
- Metric logging to TensorBoard

### ⚠️ Placeholder (TODO)
The `_build_correspondences_simple()` method is currently a **placeholder** that returns empty correspondences. To enable full pose estimation, you need to:

1. **Integrate VPOG correspondence builder:**
   ```python
   # In _build_correspondences_simple()
   from vpog.inference.correspondence import CorrespondenceBuilder
   
   corr_builder = CorrespondenceBuilder(...)
   correspondences = corr_builder.build_correspondences(
       classification_logits=outputs["classification_logits"],
       dense_flow=pred_flow,
       ...
   )
   ```

2. **Convert patch-level flow to global 2D-3D pairs:**
   - Use template poses and depth maps
   - Transform local flow to global image coordinates
   - Project 3D points using camera intrinsics

3. **Filter correspondences:**
   - Use classification scores as weights
   - Apply visibility masks
   - Filter by reprojection error

### 📋 Next Steps

1. **Implement correspondence building** (highest priority)
   - Integrate `vpog/inference/correspondence.py`
   - Handle template selection (nearest template for correspondences)
   - Transform patch coordinates to image space

2. **Validate metrics** (after correspondence building works)
   - Run validation-only mode: `python training/train.py validate_only=true resume_from_checkpoint=path/to/ckpt.ckpt`
   - Check that AR values are reasonable (0.0 - 1.0)
   - Compare with BOP toolkit evaluation if available

3. **Optimize for speed** (optional)
   - Correspondence building can be expensive
   - Consider batched PnP solving
   - Maybe subsample correspondences for faster evaluation

4. **Add more detailed logging** (optional)
   - Per-object AR scores
   - Per-threshold recalls
   - Inlier ratio statistics

## Usage Examples

### Training with BOP Evaluation

```bash
# Standard training with validation every 10K steps on YCBV
python training/train.py \
  val_check_interval=10000 \
  val_dataset_name=ycbv

# Training with validation on T-LESS
python training/train.py \
  val_check_interval=10000 \
  val_dataset_name=tless
```

### Validation-Only Mode

```bash
# Just run validation with pose metrics on YCBV
python training/train.py \
  validate_only=true \
  resume_from_checkpoint=checkpoints/epoch=42.ckpt \
  val_dataset_name=ycbv

# Validation on different dataset than training
python training/train.py \
  validate_only=true \
  resume_from_checkpoint=checkpoints/epoch=42.ckpt \
  val_dataset_name=lmo
```

### Training Without Pose Evaluation

```bash
# Disable pose evaluation for faster validation (only compute losses)
python training/train.py \
  model.enable_pose_eval=false
```

## Troubleshooting

### "Failed to initialize BOP evaluator"

**Cause:** BOP dataset not found or `bop_model.py` can't load models.

**Fix:**
- Ensure dataset exists at `/path/to/datasets/{dataset_name}/`
- Check that `models_info.json` exists
- Verify dataset name matches BOP convention

### "No poses successfully estimated during validation epoch"

**Cause:** Correspondence building returns empty arrays (current state).

**Fix:**
- Implement `_build_correspondences_simple()` properly
- Check that model outputs are valid
- Verify ground truth poses are available in batch

### "Pose evaluation failed for batch X"

**Cause:** Missing metadata (scene_id, view_id, label) in batch.infos.

**Fix:**
- Ensure BOP dataset loader includes metadata
- Check that `batch.infos` is a pandas DataFrame
- Verify dataloader uses `SceneObservation.collate_fn()`

## Technical Details

### Why AR = (2/3) * AR_MSSD + (1/3) * AR_MSPD?

The full BOP Challenge AR includes 3 metrics:
- VSD (Visible Surface Discrepancy) - requires rendering, very expensive
- MSSD (3D surface error)
- MSPD (2D projection error)

Original: AR = (1/3) * AR_VSD + (1/3) * AR_MSSD + (1/3) * AR_MSPD

Without VSD, we approximate using only MSSD and MSPD with adjusted weights to give more importance to 3D accuracy.

### Symmetry Handling

BOP metrics are **symmetry-aware**:
- `mspd()` and `mssd()` iterate over all symmetries
- For each symmetry, apply transformation and compute error
- Return the **minimum** error across all symmetries

This ensures objects with rotational/reflective symmetries are evaluated fairly.

### Correspondence Quality

The quality of correspondences directly affects pose estimation:
- **High quality**: More inliers, better AR scores
- **Low quality**: RANSAC fails, pose is identity, metrics are poor

Monitor `val/num_pose_samples` to see how many poses are successfully estimated.

## References

- BOP Challenge: http://bop.felk.cvut.cz/
- BOP Toolkit: https://github.com/thodan/bop_toolkit
- VPOG Paper: (add link if available)
- This codebase: `/data/home/ssaricha/gigapose/`
