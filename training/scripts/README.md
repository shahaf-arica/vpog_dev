# VPOG Training Scripts

## Object Index Builder

### Purpose

The `build_object_index.py` script generates metadata files (`object_index.json`) that enable object-level iteration in the VPOG training pipeline.

**Why is this needed?**
- Original dataset: Scene-level (one image → multiple objects)
- Training needs: Object-level (one sample → one object)
- Solution: Pre-compute mapping from objects to scenes

### Usage

**Build index for training datasets:**
```bash
# GSO training set
python training/scripts/build_object_index.py \
  --dataset gso \
  --split train_pbr_web

# ShapeNet training set
python training/scripts/build_object_index.py \
  --dataset shapenet \
  --split train_pbr_web
```

**Build index for validation datasets:**
```bash
# YCB-V validation set
python training/scripts/build_object_index.py \
  --dataset ycbv \
  --split test

# T-LESS validation set
python training/scripts/build_object_index.py \
  --dataset tless \
  --split test_primesense
```

**Custom paths:**
```bash
python training/scripts/build_object_index.py \
  --dataset gso \
  --split train_pbr_web \
  --root_dir /path/to/datasets \
  --min_visib 0.1
```

### Output

Creates `{dataset}/{split}/object_index.json` with structure:
```json
{
  "version": "1.0",
  "dataset": "gso",
  "split": "train_pbr_web",
  "total_scenes": 48627,
  "total_objects": 257843,
  "objects": [
    {
      "global_idx": 0,
      "scene_key": "000000_00",
      "scene_idx": 0,
      "obj_idx": 0,
      "obj_id": 1,
      "visib_fract": 0.95
    },
    ...
  ]
}
```

### Parameters

- `--dataset`: Dataset name (gso, shapenet, ycbv, tless, lmo, etc.)
- `--split`: Split name (train_pbr_web, test, test_all, etc.)
- `--root_dir`: Root directory for datasets (default: /strg/E/shared-data/Shahaf/gigapose/datasets)
- `--min_visib`: Minimum visibility fraction to include object (default: 0.1)
- `--depth_scale`: Depth scale factor (default: 10.0)

### What It Does

1. Iterates through all scenes in the dataset
2. For each scene, extracts all objects with visibility > min_visib
3. Creates a flat list of (scene, object) pairs
4. Saves to JSON for fast loading during training

### Time Estimates

- **GSO (~50k scenes)**: ~5-10 minutes
- **ShapeNet (~50k scenes)**: ~5-10 minutes  
- **YCB-V (~10k scenes)**: ~1-2 minutes
- **T-LESS (~10k scenes)**: ~1-2 minutes

### Before Training

**⚠️ IMPORTANT:** You must build object indices before training!

```bash
# Required for training
python training/scripts/build_object_index.py --dataset gso --split train_pbr_web
python training/scripts/build_object_index.py --dataset shapenet --split train_pbr_web

# Required for validation (if using validation)
python training/scripts/build_object_index.py --dataset ycbv --split test
```

Without these files, training will fail with:
```
FileNotFoundError: Object index not found: datasets/gso/train_pbr_web/object_index.json
```
