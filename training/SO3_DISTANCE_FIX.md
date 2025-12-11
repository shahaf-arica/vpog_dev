# Template Selector Fix - SO(3) Distance Correction

## Critical Bug Found and Fixed

### Problem 1: Wrong Distance Metric
**Previous implementation**: Compared only **viewing directions** (z-axis unit vectors)
- This only captures out-of-plane rotation
- Ignores in-plane rotation around the viewing axis
- Results in wrong template selection (e.g., templates 12, 13 selected instead of 22, 23, 29, 10)

**Root cause**: 
```python
# OLD - WRONG: Only compares viewing directions
dir1 = pose[:3, 2]  # z-axis only
dir2 = pose[:3, 2]
angle = arccos(dot(dir1, dir2))
```

**Fix**: Now uses proper **SO(3) geodesic distance**
```python
# NEW - CORRECT: Compares full 3D rotations
R1 = pose1[:3, :3]
R2 = pose2[:3, :3]
R_rel = R1.T @ R2
angle = arccos((trace(R_rel) - 1) / 2)
```

This is the standard geodesic distance on SO(3) that properly measures 3D rotation difference.

### Problem 2: Visualization Bug
**Issue**: Showed template 14 twice and missed template 11
- Only displayed 3 out of 4 positive templates
- Array indexing error in visualization loop

**Fix**: 
- Now displays all 4 positive templates correctly
- Layout: Row 1 (Query + 3 positives), Row 2 (1 positive + 2 negatives)
- Added star marker (★) for the nearest template

### Problem 3: Missing Random Mode Visualization
**Issue**: No visualization for `random_within_range` sampling mode

**Fix**: Added complete test section for random sampling mode
- Uses same query for direct comparison
- Shows templates sampled from 30° neighborhood
- Different color scheme to distinguish from nearest mode

## Algorithm Flow

### Step 1: Find Nearest Template (GigaPose)
Still uses viewing direction for initial out-of-plane matching:
```python
similarities = dot(template_directions, query_direction)
nearest_idx = argmax(similarities)
```

### Step 2: Find S_p-1 Close Templates (SO(3))
Now uses proper SO(3) distance:
```python
for each template:
    R_rel = R_nearest.T @ R_template
    angle = arccos((trace(R_rel) - 1) / 2)
```

Sorts by SO(3) angle and selects:
- **Nearest mode**: Take S_p-1 strictly nearest
- **Random mode**: Sample S_p-1 from templates within 30° (configurable)

### Step 3: Select Negatives
Unchanged - still uses viewing direction for far templates (≥90°)

## Updated Output Format

### Console Output
```
Query: Using template 14
Nearest template (GigaPose): 14
Positive indices: [14, 22, 23, 10]  # NOW CORRECT!

Angular Distances from Query and Nearest:
 *Positive 1 (template  14):   0.00° from query,  0.00° from nearest
  Positive 2 (template  22):  12.30° from query, 12.30° from nearest
  Positive 3 (template  23):  15.50° from query, 15.50° from nearest
  Positive 4 (template  10):  18.20° from query, 18.20° from nearest
```

### Visualization Layout
```
+-------------+-------------+-------------+-------------+
| Query       | ★Positive 1 | Positive 2  | Positive 3  |
| (Template   | (Nearest)   |             |             |
|  14)        | 0.0° / 0.0° | 12.3° / ... | 15.5° / ... |
+-------------+-------------+-------------+-------------+
| Positive 4  | Negative 1  | Negative 2  | [empty]     |
|             |             |             |             |
| 18.2° / ... | 155.3°      | 162.1°      |             |
+-------------+-------------+-------------+-------------+
```

Format: `angle_to_query° / angle_to_nearest°`

## Test Files Generated

1. **`sample_00_obj0_selection.png`**
   - Nearest mode test
   - Shows query + 4 positives (with SO(3) distances) + 2 negatives
   - Star marker on nearest template

2. **`sample_01_obj1_selection.png`**, **`sample_02_obj2_selection.png`**
   - Additional objects for verification

3. **`sample_random_mode_obj0_selection.png`**
   - Random within range mode test
   - Shows sampling from 30° neighborhood
   - Tilde marker (~) for randomly sampled templates

## Validation Checklist

After running the test, verify:

✅ **Positive templates look visually similar to query**
- Should show same object from similar viewpoints
- Small rotation differences (typically 10-30°)

✅ **Nearest template has 0° or very small angle**
- First positive should be closest match

✅ **All 4 positive templates are shown**
- No duplicates
- All have small `angle_to_nearest` values

✅ **Negative templates look very different**
- Different viewpoints (≥90° from query)
- Clearly dissimilar appearance

✅ **Random mode shows different selection**
- Still centered around nearest
- But different templates within 30° range

## Why This Matters

**VPOG's goal**: Learn patch matching with templates as "hints" of CAD appearance with perturbations

**Before fix**: 
- Templates were not actually close in SO(3)
- Model would see inconsistent viewpoint variations
- Poor generalization

**After fix**:
- Templates are truly clustered in rotation space
- Consistent small perturbations from nearest template
- Better learning of invariance to small rotations

## Technical Notes

### SO(3) Geodesic Distance Formula
For rotation matrices R₁ and R₂:
```
d(R₁, R₂) = arccos((trace(R₁ᵀR₂) - 1) / 2)
```

This is the length of the shortest path on SO(3) connecting the two rotations.

### Why Not Use Viewing Direction for Everything?
- Out-of-plane matching: Viewing direction is sufficient (GigaPose approach)
- Clustering nearby templates: Need full SO(3) distance to capture all rotation axes
- Viewing direction would miss in-plane rotations around the camera axis

### Numerical Stability
Added clamping for arccos:
```python
cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
```
Handles floating-point errors that might push values slightly outside [-1, 1].

## Running the Test

```bash
cd /data/home/ssaricha/gigapose
export PYTHONPATH=/data/home/ssaricha/gigapose:$PYTHONPATH
python training/dataloader/template_selector.py
```

Expected runtime: ~30 seconds
Output directory: `tmp/template_selector_test/`
