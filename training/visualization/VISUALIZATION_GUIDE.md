# VPOG Flow Visualization Guide

## Overview

The visualization system provides elegant, publication-ready visualizations of:
- **16×16 pixel-level flow** within patches
- **Unseen pixel marking** (gray overlay)
- **Confidence visualization** (color intensity)
- **Patch-level and pixel-level views**
- **Correspondence lines** between query and template

## Quick Start

```python
from training.visualization import visualize_patch_flow, visualize_pixel_level_flow_detailed

# After running VPOG model forward pass
outputs = model(query_images, template_images, ...)

# Extract for single template
flow = outputs['flow'][0, 0]  # [Nq, Nt, 16, 16, 2] -> [Nq, 16, 16, 2] for best match
confidence = outputs['flow_confidence'][0, 0]  # [Nq, 16, 16]
classification_probs = F.softmax(outputs['classification_logits'][0, 0], dim=-1)  # [Nq, Nt+1]

# Convert images to numpy
query_np = (query_images[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
template_np = (template_images[0,0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

# Create comprehensive visualization
fig = visualize_patch_flow(
    query_np, template_np, 
    flow.cpu().numpy(), 
    confidence.cpu().numpy(),
    classification_probs.cpu().numpy(),
    patch_size=16,
    conf_threshold=0.5,  # Pixels below this are marked as unseen (gray)
    top_k_patches=20,    # Show top 20 patches in detail
)
fig.savefig('flow_visualization.png', dpi=150, bbox_inches='tight')
```

## Visualization Functions

### 1. `visualize_patch_flow()` - Main Overview

Creates a comprehensive multi-panel figure showing:
- Query and template images with patch grid overlay
- Patch-level average flow (color-coded by direction)
- Confidence heatmap
- Unseen probability heatmap
- Pixel-level detail for top-K patches
- Flow color wheel legend
- Statistics summary

**Key Features:**
- Automatically highlights top confident matches
- **Gray color** indicates unseen/low-confidence pixels
- Color indicates flow direction (see wheel)
- Intensity indicates flow magnitude

**Parameters:**
```python
visualize_patch_flow(
    query_img: np.ndarray,        # [H, W, 3] RGB image
    template_img: np.ndarray,     # [H, W, 3] RGB image
    flow: np.ndarray,             # [Nq, 16, 16, 2] pixel-level flow
    confidence: np.ndarray,       # [Nq, 16, 16] confidence per pixel
    classification_probs: np.ndarray,  # [Nq, Nt+1] with unseen at [-1]
    patch_size: int = 16,
    conf_threshold: float = 0.5,  # Below this = unseen (gray)
    top_k_patches: int = 20,
    figsize: Tuple[int, int] = (20, 10),
)
```

### 2. `visualize_pixel_level_flow_detailed()` - Single Patch Detail

Zooms into a single patch showing:
- Query patch (16×16)
- Template patch (16×16)
- Flow arrows (with unseen pixels marked as gray ×)
- Flow color with confidence overlay

**Perfect for:**
- Debugging specific patch matches
- Understanding pixel-level correspondence
- Identifying unseen regions

**Parameters:**
```python
visualize_pixel_level_flow_detailed(
    query_patch: np.ndarray,      # [16, 16, 3]
    template_patch: np.ndarray,   # [16, 16, 3]
    flow: np.ndarray,             # [16, 16, 2]
    confidence: np.ndarray,       # [16, 16]
    conf_threshold: float = 0.5,
)
```

### 3. `visualize_correspondence_grid()` - Correspondence Lines

Shows query and template side-by-side with correspondence lines:
- **Cyan dots**: Query pixels
- **Yellow dots**: Matched template pixels
- **Colored lines**: Correspondences (color = confidence)
- Automatically filters low-confidence matches

**Parameters:**
```python
visualize_correspondence_grid(
    query_img: np.ndarray,
    template_img: np.ndarray,
    flow: np.ndarray,
    classification_probs: np.ndarray,
    confidence: np.ndarray,
    patch_size: int = 16,
    conf_threshold: float = 0.5,
    stride: int = 2,  # Sample every N pixels for clarity
)
```

### 4. `flow_to_color()` - Convert Flow to Color

Utility function using HSV color space:
- **Hue**: Flow direction (0° = right, 90° = down, 180° = left, 270° = up)
- **Saturation**: Flow magnitude (normalized)
- **Value**: Always 1.0

```python
flow_color = flow_to_color(flow, max_flow=1.0)  # Returns RGB [0, 1]
```

### 5. `create_flow_wheel()` - Legend

Creates a circular color wheel showing direction encoding:

```python
wheel = create_flow_wheel(size=256)
plt.imshow(wheel)
```

## Color Encoding

### Flow Direction (HSV Hue)
```
        Up (270°)
           ↑
           |
Left ← ----+---- → Right
(180°)     |         (0°)
           ↓
       Down (90°)
```

### Unseen Marking
- **Gray color** (RGB: [0.5, 0.5, 0.5]) indicates:
  - Confidence < `conf_threshold`
  - Likely occluded or out-of-view
  - No reliable correspondence

### Confidence
- **Bright colors**: High confidence (close to 1.0)
- **Dark colors**: Low confidence (close to 0.0)
- **Gray**: Below threshold (unseen)

## Example Workflow

```python
import torch
import torch.nn.functional as F
import numpy as np
from vpog.models.vpog_model import VPOGModel
from training.visualization import visualize_patch_flow

# 1. Load model
model = VPOGModel(...)
model.eval()

# 2. Prepare inputs
query_images = ...  # [B, 3, 224, 224]
template_images = ...  # [B, S, 3, 224, 224]

# 3. Forward pass
with torch.no_grad():
    outputs = model(query_images, template_images, ...)

# 4. Extract outputs for visualization (first sample, first template)
B, S = 0, 0  # First batch, first template
Nq = outputs['classification_logits'].shape[2]
Nt = outputs['flow'].shape[3]

# Get best template match per query patch
classification_logits = outputs['classification_logits'][B, S]  # [Nq, Nt+1]
classification_probs = F.softmax(classification_logits, dim=-1).cpu().numpy()

# Get flow for best matches
best_template_idx = classification_probs[:, :-1].argmax(axis=1)  # Exclude unseen
flow_best = torch.zeros(Nq, 16, 16, 2)
confidence_best = torch.zeros(Nq, 16, 16)

for q_idx in range(Nq):
    t_idx = best_template_idx[q_idx]
    flow_best[q_idx] = outputs['flow'][B, S, q_idx, t_idx]
    confidence_best[q_idx] = outputs['flow_confidence'][B, S, q_idx, t_idx, :, :, 0]

# 5. Convert to numpy
query_np = (query_images[B].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
template_np = (template_images[B,S].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
flow_np = flow_best.cpu().numpy()
confidence_np = confidence_best.cpu().numpy()

# 6. Visualize
fig = visualize_patch_flow(
    query_np, template_np,
    flow_np, confidence_np,
    classification_probs,
    conf_threshold=0.5,
)
fig.savefig('flow_viz.png', dpi=150, bbox_inches='tight')
```

## Understanding the Output

### Main Figure Layout
```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Query Image │ Template    │ Flow Colors │ Confidence  │
│ (with grid) │ Image       │ (Direction) │ Heatmap     │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ Unseen Prob │ Flow Wheel  │ Patch 1     │ Patch 2     │
│ Heatmap     │ Legend      │ Detail      │ Detail      │
├─────────────┴─────────────┴─────────────┴─────────────┤
│                    Statistics Summary                  │
└────────────────────────────────────────────────────────┘
```

### Reading the Visualization

1. **Query Image**: 
   - Cyan grid shows patch boundaries
   - Yellow boxes highlight top confident matches
   - Numbers indicate ranking

2. **Flow Colors**:
   - Color = direction of flow
   - Bright = large magnitude
   - Dark = small magnitude
   - Gray = unseen

3. **Confidence Heatmap**:
   - Red/White = high confidence
   - Dark Red/Black = low confidence

4. **Unseen Probability**:
   - Yellow/Bright = high unseen probability
   - Dark Blue = confident match

5. **Patch Detail**:
   - Shows 16×16 pixel grid
   - Gray pixels = unseen (conf < threshold)
   - Flow arrows show correspondence direction

### Statistics Panel

Reports:
- Total number of patches
- Average confidence and unseen probability
- Number of confident matches
- Flow magnitude statistics

## Testing

Run the test suite to verify visualization:

```bash
cd /data/home/ssaricha/gigapose
python vpog/visualization/test_flow_vis.py
```

This will:
1. Create a lightweight VPOG model
2. Generate synthetic test data
3. Run forward pass
4. Create all visualizations
5. Save to `/tmp/vpog_vis/`

Output files:
- `patch_flow_overview.png` - Main comprehensive visualization
- `pixel_flow_detail.png` - Single patch detail
- `correspondences.png` - Correspondence lines
- `flow_wheel.png` - Color legend
- `unseen_marking_test.png` - Unseen marking test

## Tips

### For Publications
- Use `dpi=300` for high-quality figures
- Set `figsize` appropriately for your layout
- Adjust `conf_threshold` to highlight different confidence levels

### For Debugging
- Use `top_k_patches=50` to see more examples
- Lower `conf_threshold` to see marginal matches
- Use `visualize_pixel_level_flow_detailed()` for specific patches

### For Presentations
- Use `figsize=(20, 10)` for wide displays
- Include the flow wheel for audience reference
- Use correspondence visualization to show matches clearly

## Advanced Usage

### Custom Confidence Threshold
```python
# Conservative (only high-confidence matches)
visualize_patch_flow(..., conf_threshold=0.7)

# Liberal (include marginal matches)
visualize_patch_flow(..., conf_threshold=0.3)
```

### Focus on Specific Patches
```python
# Get patch index from coordinates
patch_y, patch_x = 5, 7
grid_w = 14  # 224 / 16
patch_idx = patch_y * grid_w + patch_x

# Extract patch
query_patch = query_np[patch_y*16:(patch_y+1)*16, patch_x*16:(patch_x+1)*16]
flow_patch = flow_np[patch_idx]
conf_patch = confidence_np[patch_idx]

# Visualize
fig = visualize_pixel_level_flow_detailed(query_patch, template_patch, 
                                          flow_patch, conf_patch)
```

### Animate Flow
```python
from training.visualization import create_flow_animation_frames

frames = create_flow_animation_frames(
    query_np, template_np, flow_np, confidence_np,
    num_frames=20
)

# Save as GIF or video
import imageio
imageio.mimsave('flow_animation.gif', frames, fps=10)
```

## Implementation Details

### Flow Units
- Flow is in **patch units** (not pixels)
- `flow[i, j] = 1.0` means one full patch displacement
- Multiply by `patch_size` (16) to get pixel displacement

### Coordinate System
- Origin: top-left corner
- X-axis: rightward (column)
- Y-axis: downward (row)
- Flow convention: query → template

### Memory Considerations
- Pixel-level flow is 16×16 = 256× larger than patch-level
- For 196 patches: 196 × 256 = 50,176 flow vectors
- Visualization samples with `stride` parameter to avoid clutter

## Troubleshooting

**Issue**: Visualization shows all gray (unseen)
- **Cause**: Confidence threshold too high or model not trained
- **Solution**: Lower `conf_threshold` or check model outputs

**Issue**: Flow colors look random
- **Cause**: Flow magnitude too small or model prediction poor
- **Solution**: Check `max_flow` parameter or model training

**Issue**: Figure too crowded
- **Cause**: Too many correspondence lines
- **Solution**: Increase `stride` or `conf_threshold`

**Issue**: Can't see pixel detail
- **Cause**: Figure size too small
- **Solution**: Increase `figsize` or `dpi`
