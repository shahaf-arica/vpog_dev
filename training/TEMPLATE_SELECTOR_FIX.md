# Template Selector Fix - Corrected Logic

## Problem Identified

The previous implementation incorrectly selected S_p positive templates by taking the S_p nearest templates to the **query pose** directly. This caused issues where visually dissimilar templates were selected.

**Example**: For query template 14, the code selected [13, 14, 12, 11], but visually templates 12 and 13 were much farther than templates 22, 23, 29, 10.

## Root Cause

The spec was misunderstood. The correct VPOG approach is:
1. Find the **single nearest out-of-plane template** (GigaPose style)
2. Find **S_p-1 additional templates** that are close to this nearest template on the S² sphere
3. This ensures all positive templates are clustered together in viewpoint space

## Solution Implemented

### Corrected Algorithm

```python
def select_templates(query_pose):
    # STEP 1: Find single nearest template (GigaPose style)
    similarities_to_query = dot(template_directions, query_direction)
    nearest_idx = argmax(similarities_to_query)
    
    # STEP 2: Find S_p-1 templates close to nearest on S²
    similarities_to_nearest = dot(template_directions, nearest_direction)
    angles_to_nearest = arccos(similarities_to_nearest)
    sorted_by_distance = argsort(angles_to_nearest)
    
    positive_indices = [nearest_idx] + sorted_by_distance[1:S_p]
    
    # STEP 3: Select S_n negative templates (far from query)
    # ... existing logic ...
```

### New Features Added

#### 1. **Sampling Mode** (`positive_sampling_mode`)
- `"nearest"` (default): Take S_p-1 strictly nearest templates to the closest one
- `"random_within_range"`: Randomly sample S_p-1 from templates within `positive_sampling_deg_range`

**Use case**: The random mode allows more diversity in positive templates while still maintaining clustering around the nearest template.

#### 2. **Random Seed** (`seed`)
- All random operations now use `np.random.default_rng(seed)`
- Ensures reproducibility for debugging and testing
- Default seed is 42 in tests

#### 3. **Enhanced Diagnostics**
The test now prints:
- Nearest template index (GigaPose)
- Angular distance from query for each positive
- Angular distance from nearest template for each positive (should be small!)
- Angular distance from query for negatives (should be ≥90°)

### API Changes

```python
# Old
selector = TemplateSelector(
    level_templates=1,
    num_positive=4,
    num_negative=2,
    min_negative_angle_deg=90.0,
)

# New
selector = TemplateSelector(
    level_templates=1,
    num_positive=4,
    num_negative=2,
    min_negative_angle_deg=90.0,
    positive_sampling_mode="nearest",  # NEW
    positive_sampling_deg_range=30.0,   # NEW
    seed=42,                             # NEW
)
```

### Configuration Updates

Updated `training/config/data/vpog_data.yaml`:
```yaml
template_selection:
  num_positive_templates: 4
  num_negative_templates: 2
  min_negative_angle_deg: 90.0
  d_ref_random_ratio: 0.0
  positive_sampling_mode: nearest         # NEW
  positive_sampling_deg_range: 30.0       # NEW
  seed: 42                                # NEW
```

## Verification

### Expected Output Pattern

For a query at template 14:
```
Nearest template (GigaPose): 14

Angular Distances from Query and Nearest:
 *Positive 1 (template  14):   0.00° from query,  0.00° from nearest
  Positive 2 (template  13):   8.50° from query,  8.50° from nearest
  Positive 3 (template  15):   8.50° from query,  8.50° from nearest
  Positive 4 (template  10):  15.20° from query, 15.20° from nearest
  
  Negative 1 (template  80): 155.30° from query
  Negative 2 (template  92): 162.10° from query
```

**Key validation**:
- Nearest template should be closest to query (0° or very small angle)
- All positives should have **small angles to nearest** (clustered on S²)
- All negatives should have **large angles to query** (≥90°)

## Testing

### Run Test
```bash
cd /data/home/ssaricha/gigapose
export PYTHONPATH=/data/home/ssaricha/gigapose:$PYTHONPATH
python training/dataloader/template_selector.py
```

### Check Visualizations
The test generates images at:
```
tmp/template_selector_test/sample_00_obj0_selection.png
```

**Visual verification**:
1. Query image should be visible
2. Positive templates (green borders) should look similar to query
3. Negative templates (red borders) should look very different
4. Angular distances shown should match expected pattern

## Implementation Files Modified

1. **training/dataloader/template_selector.py**
   - Fixed `select_templates()` method with 3-step algorithm
   - Added `positive_sampling_mode` parameter
   - Added `seed` parameter for reproducibility
   - Enhanced test with better diagnostics

2. **training/dataloader/vpog_dataset.py**
   - Updated test to use seed=42
   - Added new config parameters

3. **training/config/data/vpog_data.yaml**
   - Added `positive_sampling_mode: nearest`
   - Added `positive_sampling_deg_range: 30.0`
   - Added `seed: 42`

## Future Flexibility

The implementation maintains flexibility for debugging:
- `seed` can be set to `None` for non-deterministic behavior
- `positive_sampling_mode` can be switched to `"random_within_range"` for more diversity
- `positive_sampling_deg_range` can be adjusted to control clustering tightness
- All random operations go through `self.rng` for easy seed control

## Backward Compatibility

The new parameters have sensible defaults:
- `positive_sampling_mode="nearest"` (deterministic)
- `positive_sampling_deg_range=30.0` (reasonable range)
- `seed=None` (random if not set)

Existing code will work without changes, but we recommend setting `seed` for reproducibility.
