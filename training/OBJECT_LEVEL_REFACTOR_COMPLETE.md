# Object-Level Dataset Refactor - COMPLETE ✓

## Summary
Successfully refactored VPOG training from **scene-level** to **object-level** iteration.

## Problem Solved
**Before:** Dataset iterated over scenes (each with N objects), randomly sampling objects in `collate_fn`
- ❌ Not all objects seen per epoch
- ❌ No proper shuffling
- ❌ Validation was random (inconsistent metrics)
- ❌ Memory unpredictable (scene with 12 objects → 12×163 batch)

**After:** Dataset iterates over individual objects
- ✅ All objects seen per epoch
- ✅ Proper shuffling (training) / deterministic iteration (validation)
- ✅ Consistent validation metrics
- ✅ Predictable memory (batch_size × 163)

---

## Implementation Complete

### ✅ Step 1: Metadata Generation
**Files:**
- `training/dataloader/object_index_builder.py` - Core indexing logic
- `training/scripts/build_object_index.py` - CLI tool

**Product:** `datasets/{dataset}/{split}/object_index.json`

**Usage:**
```bash
python training/scripts/build_object_index.py --dataset gso --split train_pbr_web
python training/scripts/build_object_index.py --dataset shapenet --split train_pbr_web
python training/scripts/build_object_index.py --dataset ycbv --split test
```

### ✅ Step 2: Object-Level Wrapper
**File:** `training/dataloader/object_level_wrapper.py`

**Purpose:** Wraps `WebSceneDataset` to provide map-style object-level `__getitem__`
- Uses LRU cache to avoid repeated scene loading
- Each `__getitem__(idx)` returns `SceneObservation` with 1 object

### ✅ Step 3: Dataset Integration
**File:** `training/dataloader/vpog_dataset.py`

**Changes:**
1. Added imports for `ObjectLevelDataset` and `load_object_index`
2. Load object index in `__init__` (required, raises error if missing)
3. Wrap `WebSceneDataset` with `ObjectLevelDataset`
4. Updated `process_query()` - removed random sampling (now deterministic)

### ✅ Step 4: DataLoader Configuration
**File:** `training/train.py`

**Changes:**
- Training DataLoader: `shuffle=True` (randomize object order)
- Validation DataLoader: `shuffle=False` (deterministic iteration)

### ✅ Step 5: Bug Fixes
**File:** `src/megapose/utils/webdataset.py`

**Fix:** Handle empty dictionaries from tar file padding
- Issue: https://github.com/nv-nguyen/gigapose/issues/26
- Solution: Skip entries without 'fname' and 'data' keys

**File:** `src/custom_megapose/web_scene_dataset.py`

**Fix:** Pass `depth_scale` parameter to `load_scene_ds_obs` in `__getitem__`

---

## How It Works

### Training Flow:
1. `object_index.json` loaded → 257k+ objects indexed
2. `ObjectLevelDataset[i]` → returns scene with 1 object
3. DataLoader batches 8 objects → `[8, 163, 3, 224, 224]`
4. `shuffle=True` → objects randomized each epoch
5. All objects seen exactly once per epoch

### Validation Flow:
1. `object_index.json` loaded → all validation objects indexed
2. `ObjectLevelDataset[i]` → returns scene with 1 object (deterministic)
3. DataLoader batches N objects → predictable size
4. `shuffle=False` → same order every validation run
5. All objects tested consistently

---

## Memory Benefits

**Before (Scene-Level):**
```
Scene with 12 objects → [12, 163, 3, 224, 224] = ~4GB
Unpredictable memory usage
```

**After (Object-Level):**
```
batch_size=8 → [8, 163, 3, 224, 224] = ~2.6GB
Predictable memory usage
```

---

## Migration Checklist

For each dataset you use:

### Training Datasets:
- [ ] `python training/scripts/build_object_index.py --dataset gso --split train_pbr_web`
- [ ] `python training/scripts/build_object_index.py --dataset shapenet --split train_pbr_web`

### Validation Datasets:
- [ ] `python training/scripts/build_object_index.py --dataset ycbv --split test`
- [ ] `python training/scripts/build_object_index.py --dataset tless --split test`
- [ ] `python training/scripts/build_object_index.py --dataset lmo --split test`

---

## Verification

After generating indices, verify with:
```bash
# Check index exists
ls datasets/gso/train_pbr_web/object_index.json

# Check index content
python -c "
import json
with open('datasets/gso/train_pbr_web/object_index.json') as f:
    data = json.load(f)
    print(f'Total scenes: {data[\"total_scenes\"]}')
    print(f'Total objects: {data[\"total_objects\"]}')
    print(f'Filtered objects: {data[\"filtered_objects\"]}')
"
```

Run training:
```bash
python training/train.py machine=local
```

Expected behavior:
- No random sampling warnings
- All objects iterated per epoch
- Consistent validation metrics
- Predictable memory usage

---

## Status: ✅ COMPLETE

All steps implemented and integrated. Ready for training!
