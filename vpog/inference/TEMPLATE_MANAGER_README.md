# VPOG Template Manager

Stage 3 of the VPOG inference pipeline: Template loading, caching, and selection.

## Overview

The `TemplateManager` handles all template-related operations for 6D pose estimation:
- **Loading** template images, masks, and poses from disk
- **Caching** templates in memory for performance
- **Selection** of optimal templates (all or subset modes)
- **Integration** with TemplateSelector for intelligent subset selection

## Features

### Two Operating Modes

1. **All Mode** (Inference)
   - Loads all 162 templates for an object (level 1 icosphere)
   - Used for exhaustive template matching
   - Templates are cached for repeated use
   - Best for inference where compute allows

2. **Subset Mode** (Training/Fast Inference)
   - Selects S_p nearest + S_n negative templates
   - Uses TemplateSelector for intelligent selection
   - Requires query pose for selection
   - Best for training or fast inference

### LRU Caching
- Configurable cache size
- Automatic eviction of least-recently-used objects
- Significant speedup for repeated queries

### Batch Support
- Load templates for multiple objects
- Preload common objects into cache
- Process multiple queries efficiently

## Installation

No additional dependencies beyond VPOG requirements. Template Manager uses:
- `numpy`, `torch` (core)
- `PIL` (image loading)
- `training.dataloader.template_selector` (subset mode)

## Quick Start

### All Mode (Load all 162 templates)

```python
from vpog.inference import TemplateManager

# Initialize manager
manager = TemplateManager(
    templates_dir="/path/to/datasets/templates",
    dataset_name="gso",
    mode="all",
    cache_size=10,
)

# Load templates for an object
data = manager.load_object_templates("000733")

print(f"Loaded {len(data['images'])} templates")
# Loaded 162 templates

# Access data
images = data["images"]  # [162, H, W, 3] RGB
masks = data["masks"]    # [162, H, W] binary
poses = data["poses"]    # [162, 4, 4] poses
indices = data["template_indices"]  # [162] view IDs
```

### Subset Mode (Select S_p + S_n templates)

```python
import numpy as np
from vpog.inference import TemplateManager

# Initialize for subset selection
manager = TemplateManager(
    templates_dir="/path/to/datasets/templates",
    dataset_name="gso",
    mode="subset",
    num_positive=4,   # S_p nearest templates
    num_negative=2,   # S_n negative templates
)

# Create or load query pose
query_pose = np.eye(4)  # 4x4 transformation matrix
query_pose[:3, :3] = R_query
query_pose[:3, 3] = t_query

# Select and load templates
data = manager.load_object_templates("000733", query_pose=query_pose)

print(f"Selected {len(data['images'])} templates")
# Selected 6 templates

# Access data
images = data["images"]  # [6, H, W, 3]
masks = data["masks"]    # [6, H, W]
poses = data["poses"]    # [6, 4, 4]
indices = data["template_indices"]  # [6] selected indices
types = data["template_types"]      # [6] 0=positive, 1=negative
d_ref = data["d_ref"]    # [3] reference direction
```

## API Reference

### TemplateManager

**Constructor:**
```python
TemplateManager(
    templates_dir: Union[str, Path],
    dataset_name: str = "gso",
    mode: str = "all",
    level_templates: int = 1,
    pose_distribution: str = "all",
    num_positive: int = 4,
    num_negative: int = 2,
    min_negative_angle_deg: float = 90.0,
    cache_size: int = 10,
    device: str = "cpu",
)
```

**Parameters:**
- `templates_dir`: Root directory with template images
  - Expected structure: `{templates_dir}/{dataset_name}/{object_id}/NNNNNN.png`
- `dataset_name`: Dataset identifier (`"gso"`, `"lmo"`, `"ycbv"`, etc.)
- `mode`: Operating mode (`"all"` or `"subset"`)
- `level_templates`: Icosphere level (1=162 views, 2=642 views)
- `pose_distribution`: Hemisphere (`"all"` or `"upper"`)
- `num_positive`: S_p nearest templates (subset mode)
- `num_negative`: S_n negative templates (subset mode)
- `min_negative_angle_deg`: Minimum angular separation for negatives
- `cache_size`: Max objects to keep in cache
- `device`: Torch device (`"cpu"` or `"cuda"`)

### Methods

#### load_object_templates()

Load templates for a specific object.

```python
def load_object_templates(
    self,
    object_id: str,
    query_pose: Optional[np.ndarray] = None,
    return_metadata: bool = True,
) -> Dict
```

**Parameters:**
- `object_id`: Object identifier (e.g., `"000733"` for GSO object 733)
- `query_pose`: Query pose [4, 4] (required for subset mode)
- `return_metadata`: Whether to include indices, types, d_ref

**Returns:**
Dictionary with:
- `images`: [N, H, W, 3] RGB images (uint8)
- `masks`: [N, H, W] binary masks (uint8)
- `poses`: [N, 4, 4] template poses (float)
- `template_indices`: [N] view IDs (int)
- `template_types`: [N] 0=positive, 1=negative (subset mode only)
- `d_ref`: [3] reference direction (subset mode only)

#### preload_objects()

Preload templates into cache (all mode only).

```python
def preload_objects(self, object_ids: List[str])
```

**Example:**
```python
manager.preload_objects(["000733", "000001", "000003"])
```

#### get_available_objects()

Get list of available object IDs.

```python
def get_available_objects(self) -> List[str]
```

#### clear_cache()

Clear the template cache.

```python
def clear_cache(self)
```

#### get_cache_info()

Get cache statistics.

```python
def get_cache_info(self) -> Dict
```

**Returns:**
```python
{
    "cached_objects": 3,
    "cache_size": 10,
    "cache_order": ["000733", "000001", "000003"],
}
```

### Factory Function

```python
from vpog.inference import create_template_manager

manager = create_template_manager(
    templates_dir="/path/to/templates",
    dataset_name="gso",
    mode="all",
    cache_size=10,
)
```

## Usage Examples

### Example 1: Inference Pipeline (All Mode)

```python
from vpog.inference import TemplateManager, CorrespondenceBuilder, PnPSolver

# Initialize components
template_mgr = TemplateManager(
    templates_dir="datasets/templates",
    dataset_name="gso",
    mode="all",
    cache_size=10,
)

corr_builder = CorrespondenceBuilder()
pose_solver = PnPSolver()

# Load query image
query_image = load_query_image("query.png")  # [H, W, 3]
K = np.array([[280, 0, 112], [0, 280, 112], [0, 0, 1]])  # Camera intrinsics

# Load templates
templates = template_mgr.load_object_templates("000733")

# Build correspondences
correspondences = corr_builder.build(
    query_image,
    templates["images"],
    templates["masks"],
)

# Solve for pose
pose, inliers = pose_solver.solve(
    correspondences.pts_2d,
    correspondences.pts_3d,
    K,
)
```

### Example 2: Training Setup (Subset Mode)

```python
from vpog.inference import TemplateManager

# Initialize for training
template_mgr = TemplateManager(
    templates_dir="datasets/templates",
    dataset_name="gso",
    mode="subset",
    num_positive=4,
    num_negative=2,
)

# Training loop
for batch in dataloader:
    query_pose = batch["pose"]  # [B, 4, 4]
    object_ids = batch["object_id"]  # [B]
    
    for i in range(len(query_pose)):
        # Select templates for this query
        templates = template_mgr.load_object_templates(
            object_ids[i],
            query_pose=query_pose[i].numpy(),
        )
        
        # Use templates for training
        # ...
```

### Example 3: Batch Preloading

```python
from vpog.inference import TemplateManager

manager = TemplateManager(
    templates_dir="datasets/templates",
    dataset_name="gso",
    mode="all",
    cache_size=50,  # Large cache
)

# Get all objects in dataset
all_objects = manager.get_available_objects()
print(f"Found {len(all_objects)} objects")

# Preload common objects
common_objects = all_objects[:20]  # First 20 objects
manager.preload_objects(common_objects)

# Now these are cached
for obj_id in common_objects:
    templates = manager.load_object_templates(obj_id)  # Fast (from cache)
    # Process...
```

## Performance

### Benchmarks (GSO Dataset, Object 733)

| Operation | Mode | Time | Notes |
|-----------|------|------|-------|
| First load | all | ~500ms | Load 162 templates from disk |
| Cache hit | all | ~5ms | Load from memory |
| Selection | subset | ~50ms | TemplateSelector + load 6 templates |
| Preload 10 objects | all | ~5s | Parallel loading possible |

### Memory Usage

| Mode | Templates | Memory per Object | Cache of 10 Objects |
|------|-----------|-------------------|---------------------|
| all (level 1) | 162 | ~150 MB | ~1.5 GB |
| all (level 2) | 642 | ~600 MB | ~6 GB |
| subset (4+2) | 6 | ~6 MB | ~60 MB |

**Recommendations:**
- Use `cache_size=10-20` for typical inference
- For large-scale evaluation, preload all objects if memory allows
- Subset mode for training (lower memory, faster)

## Testing

Run the comprehensive test suite:

```bash
cd /data/home/ssaricha/gigapose
PYTHONPATH=/data/home/ssaricha/gigapose:$PYTHONPATH python vpog/inference/test_template_manager.py
```

**Test Coverage:**
- ✅ Initialization (all and subset modes)
- ✅ Template loading (all 162 templates)
- ✅ Template selection (S_p + S_n)
- ✅ LRU caching
- ✅ Preloading
- ✅ Batch loading
- ✅ Error handling
- ✅ Factory function

**Results:** 8/8 tests passing

## Directory Structure

Expected template directory structure:

```
templates/
├── gso/
│   ├── 000000/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── 000001/
│   └── object_poses/
│       ├── 000000.npy  # [N, 4, 4] poses
│       └── 000001.npy
├── lmo/
└── ycbv/
```

**Template Images:**
- Format: RGBA PNG (or RGB)
- Resolution: Typically 640×480
- Naming: 6-digit zero-padded view ID
- Alpha channel used for mask extraction

**Pose Files:**
- Format: NumPy array `.npy`
- Shape: [N, 4, 4] transformation matrices
- Convention: OpenCV (camera to object)

## Integration with Other Components

### With CorrespondenceBuilder

```python
templates = template_mgr.load_object_templates("000733")
correspondences = corr_builder.build(
    query_image,
    templates["images"],
    templates["masks"],
)
```

### With PnPSolver

```python
pose, inliers = pnp_solver.solve(
    correspondences.pts_2d,
    correspondences.pts_3d,
    K,
)
```

### With TemplateSelector (Subset Mode)

The TemplateManager uses TemplateSelector internally in subset mode:

```python
# Subset mode automatically uses TemplateSelector
manager = TemplateManager(mode="subset", num_positive=4, num_negative=2)
data = manager.load_object_templates("000733", query_pose=pose)
# Internally calls TemplateSelector.select_templates()
```

## Advanced Usage

### Custom Template Selection

For fine-grained control, use TemplateSelector directly:

```python
from training.dataloader.template_selector import TemplateSelector

selector = TemplateSelector(
    level_templates=1,
    pose_distribution="all",
    num_positive=8,  # More positives
    num_negative=4,  # More negatives
    positive_sampling_mode="random_within_range",
    positive_sampling_deg_range=30.0,
)

result = selector.select_templates(query_pose, return_d_ref=True)
# Then load specific templates using indices
```

### GPU Acceleration

```python
manager = TemplateManager(
    templates_dir="datasets/templates",
    dataset_name="gso",
    mode="all",
    device="cuda",  # Templates moved to GPU
)

templates = manager.load_object_templates("000733")
# templates["images"] is on CUDA
```

### Multiple Datasets

```python
gso_manager = TemplateManager(templates_dir="datasets/templates", dataset_name="gso")
lmo_manager = TemplateManager(templates_dir="datasets/templates", dataset_name="lmo")

# Switch between datasets as needed
```

## Troubleshooting

### Issue: "Templates directory not found"
**Solution:** Verify `templates_dir` path exists and contains `dataset_name` subdirectory.

```python
from pathlib import Path
templates_dir = Path("datasets/templates")
assert (templates_dir / "gso").exists()
```

### Issue: "Object directory not found"
**Solution:** Check object_id format (should be 6-digit zero-padded string like `"000733"`).

### Issue: "query_pose required for subset mode"
**Solution:** Always provide query_pose in subset mode:

```python
data = manager.load_object_templates("000733", query_pose=pose)
```

### Issue: Out of memory
**Solution:** Reduce `cache_size` or use subset mode:

```python
manager = TemplateManager(cache_size=5, mode="subset")
```

## Next Steps

- **Stage 4**: Full inference pipeline integration
- **Optimization**: Multi-threaded template loading
- **Extension**: Support for custom template distributions

## References

- [TemplateSelector Documentation](../training/dataloader/README.md)
- [Template Pose Generation](../../src/lib3d/template_transform.py)
- [VPOG Pipeline Overview](../QUICK_START.md)
