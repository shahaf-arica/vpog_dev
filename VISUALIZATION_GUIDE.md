# VPOG Pipeline Comprehensive Visualization Guide

## Overview

The VPOG test pipeline now includes a comprehensive visualization system activated with the `--visualize` flag. This system generates detailed visualizations of all pipeline components to verify correctness of data, poses, classifications, and flows.

## Usage

```bash
# Basic test with simple visualization
python -m training.test_vpog_full_pipeline

# Comprehensive visualization with synthetic data
python -m training.test_vpog_full_pipeline --visualize

# Comprehensive visualization with real data
python -m training.test_vpog_full_pipeline --real_data --visualize

# On CPU
python -m training.test_vpog_full_pipeline --visualize --device cpu
```

## Visualization Outputs

All visualizations are saved to `./tmp/vpog_full_pipeline_test/`

### For Each Batch Sample (up to 2 samples):

#### 1. Data and Pose Visualization (`sampleN_01_data_and_poses.png`)
**Purpose**: Verify input data, poses, and template selection

**Contents**:
- **Query Image**: The query view with pose information
- **Query Pose Info**: 
  - Full 3x3 rotation matrix
  - 3D translation vector
  - Reference direction (d_ref) for S²RoPE
  - Camera intrinsics (fx, fy, cx, cy)
  
- **Template Images Grid**: All selected templates with:
  - Color-coded borders (green=positive, red=negative)
  - Template type (POSITIVE/NEGATIVE)
  - Template index from database
  
- **Pose Comparison Table**:
  - Relative rotation angle (degrees) between query and each template
  - Translation difference magnitude
  - Per-axis translation differences (Δt_x, Δt_y, Δt_z)
  
- **Ground Truth Flow Statistics**:
  - Flow magnitude mean/std for each template
  - Visibility percentage (pixel-level)
  - Patch visibility percentage

**What to Check**:
- ✓ Query and template images look correct
- ✓ Positive templates should have small rotation angles (<30°)
- ✓ Negative templates should have large rotation angles (>45°)
- ✓ Flow statistics reasonable (mean ~0.1-1.0 for positives)
- ✓ High visibility for positive templates (>70%)

---

#### 2. Classification Visualization (`sampleN_02_classification.png`)
**Purpose**: Verify classification head outputs and template assignment

**For Each Template** (rows), Shows:

**Column 1 - Query→Template Probability**:
- Heatmap of classification probabilities
- Shows which query patches match this template
- **Expected**: High for positive templates, low for negatives

**Column 2 - Unseen Probability**:
- Per-patch probability of "no match" (unseen token)
- **Expected**: Low for positives, high for negatives

**Column 3 - Max Classification Confidence**:
- Maximum probability across all templates
- Shows confidence of the best match
- **Expected**: High values (>0.7) indicate confident predictions

**Column 4 - Assigned Template Index**:
- Color-coded map showing which template each query patch matched to
- **Expected**: Consistent patches assigned to same template

**Column 5 - Statistics**:
- Number of patches assigned to this template
- Number marked as unseen
- Mean probabilities
- Logits range

**What to Check**:
- ✓ Positive templates: High probability, low unseen
- ✓ Negative templates: Low probability, high unseen
- ✓ Max confidence generally >0.5
- ✓ Logits not saturated (not all very large positive/negative)

---

#### 3. Flow Visualization per Template (`sampleN_03_flow_templateN.png`)
**Purpose**: Compare predicted flow vs ground truth for each template

**Layout**: 3 rows × 5 columns

**Row 1 - Ground Truth**:
- GT Flow X: Horizontal flow component
- GT Flow Y: Vertical flow component  
- GT Flow Magnitude: Combined flow strength
- GT Visibility: Which pixels are visible
- GT Statistics: Mean magnitude, range, visibility %

**Row 2 - Predictions**:
- Pred Flow X/Y: Averaged across query-template patch pairs
- Pred Flow Magnitude: Predicted flow strength
- Pred Confidence: Model's confidence in flow prediction
- Pred Statistics: Mean confidence, flow range

**Row 3 - Comparison**:
- Flow Error X/Y: Difference between prediction and GT
- Flow Error Magnitude: Total error
- Error vs Confidence Scatter: Should show negative correlation
- Error Statistics: MAE, mean/median error, correlation metrics

**What to Check**:
- ✓ Predicted flow visually similar to GT flow
- ✓ High confidence where error is low (good calibration)
- ✓ Mean error <0.2 for positive templates
- ✓ Error higher for negative templates (expected)
- ✓ Flow magnitude in reasonable range (0.0-2.0)

---

#### 4. Detailed Patch Flow (`sampleN_04_patch_flow_detail.png`)
**Purpose**: Verify intra-patch 16×16 pixel-level flow predictions

**Shows**: The query-template patch pair with highest confidence

**Row 1**:
- Flow X component (16×16 pixels)
- Flow Y component (16×16 pixels)
- Flow Magnitude per pixel
- Confidence per pixel

**Row 2**:
- Flow Direction (HSV color-coded)
- Flow Vectors (quiver plot showing direction)
- High Confidence Mask (>0.5 threshold)
- Statistics (mean/max flow, confidence distribution)

**What to Check**:
- ✓ Flow appears smooth (no sudden jumps)
- ✓ Confidence correlates with expected accuracy
- ✓ High confidence pixels >50% of patch
- ✓ Flow vectors show consistent patterns
- ✓ Mean magnitude reasonable (0.1-0.5)

---

## Visualization Functions

### Core Functions

1. **`visualize_comprehensive_pipeline()`**: Main orchestrator
   - Calls all visualization functions
   - Saves images with organized naming
   - Handles up to 2 batch samples

2. **`visualize_data_and_poses()`**: Input data verification
   - Images, poses, camera parameters
   - Template selection (positive/negative)
   - Pose differences and flow GT stats

3. **`visualize_classification()`**: Classification head outputs
   - Per-template probability maps
   - Unseen probabilities
   - Assignment confidence and statistics

4. **`visualize_template_flow()`**: Flow comparison
   - GT vs predicted flow
   - Error analysis
   - Confidence calibration

5. **`visualize_patch_flow_detailed()`**: Pixel-level flow detail
   - 16×16 intra-patch flows
   - Confidence maps
   - Vector visualization

---

## Interpreting Results

### Healthy Pipeline Indicators

**Data & Poses**:
- Positive templates: ΔR < 30°, Δt < 0.5
- Negative templates: ΔR > 45°
- Visibility >70% for positives

**Classification**:
- Positive template prob >0.7
- Negative unseen prob >0.6
- Confident predictions (max prob >0.7)

**Flow**:
- MAE <0.2 for positive templates
- High confidence where error is low
- Smooth flow fields (no noise)

**Patch Detail**:
- Mean confidence >0.5
- >60% pixels with high confidence
- Flow magnitudes 0.1-0.5 for nearby views

### Common Issues

**Problem**: All templates have similar low probabilities
- **Cause**: Classification head not learning
- **Check**: Logits range, ensure not all same value

**Problem**: High error but high confidence
- **Cause**: Overconfident model, bad calibration
- **Check**: Temperature parameter, confidence loss weight

**Problem**: Flow has large discontinuities
- **Cause**: Patch boundary artifacts, poor feature matching
- **Check**: Flow regularization, smoothness loss

**Problem**: No visible pixels for positive templates
- **Cause**: Visibility computation bug, wrong camera parameters
- **Check**: GT flow generation, projection logic

---

## Color Coding

### Flow Visualization
- **Red-Blue**: X/Y components (red=positive, blue=negative)
- **Hot (red-yellow)**: Magnitude (bright=high)
- **HSV**: Direction (hue=angle, saturation=magnitude)

### Classification
- **Hot (red-yellow)**: Probability (bright=high probability)
- **Viridis**: Confidence (yellow=confident)
- **Plasma**: Max confidence
- **Tab20**: Template indices

### Template Types
- **Green border**: Positive template
- **Red border**: Negative template

---

## Tips for Verification

1. **Start with synthetic data**: Easier to verify correctness
2. **Check data first**: Ensure poses and GT flow are correct before debugging model
3. **Compare positive vs negative**: Should show clear differences
4. **Look for consistency**: Similar query patches should behave similarly
5. **Check confidence calibration**: High confidence should mean low error
6. **Verify flow smoothness**: No sudden jumps between patches
7. **Monitor statistics**: Use numerical stats to catch subtle issues

---

## Example Workflow

```bash
# 1. Test with synthetic data + visualization
python -m training.test_vpog_full_pipeline --visualize

# 2. Review all visualizations in ./tmp/vpog_full_pipeline_test/
#    - Check data and poses look correct
#    - Verify classification assigns positive templates
#    - Compare predicted vs GT flow
#    - Examine patch-level detail

# 3. If issues found, adjust model/data and re-test

# 4. Test with real data
python -m training.test_vpog_full_pipeline --real_data --visualize

# 5. Compare synthetic vs real results to validate pipeline
```

---

## File Organization

```
./tmp/vpog_full_pipeline_test/
├── sample0_01_data_and_poses.png      # Batch 0: Input verification
├── sample0_02_classification.png      # Batch 0: Classification outputs
├── sample0_03_flow_template0.png      # Batch 0: Flow for template 0
├── sample0_03_flow_template1.png      # Batch 0: Flow for template 1
├── sample0_03_flow_template2.png      # Batch 0: Flow for template 2
├── sample0_04_patch_flow_detail.png   # Batch 0: Pixel-level detail
├── sample1_01_data_and_poses.png      # Batch 1: (if exists)
└── ... (same structure for batch 1)
```

---

## Integration with Training

These visualizations can be:
1. Generated periodically during training (e.g., every N epochs)
2. Used for debugging when validation loss plateaus
3. Compared across different model configurations
4. Included in experiment tracking (W&B, TensorBoard)

To integrate:
```python
# In training loop
if epoch % 10 == 0:
    with torch.no_grad():
        # Get batch from val dataloader
        outputs = model(query, templates, ...)
        visualize_comprehensive_pipeline(outputs, batch, 
                                        Path(f"./logs/epoch_{epoch}"))
```

---

## Summary

The `--visualize` flag provides comprehensive visualization of:
- ✓ Input data quality (images, poses, GT labels)
- ✓ Model outputs (classification probabilities, flow predictions)
- ✓ Correctness verification (GT vs predictions, error analysis)
- ✓ Confidence calibration (high confidence = low error?)
- ✓ Detailed inspection (patch-level 16×16 pixel flows)

Use these visualizations to ensure the entire pipeline is working correctly before scaling to full training!
