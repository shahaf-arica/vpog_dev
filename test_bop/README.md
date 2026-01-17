# VPOG BOP Testing

Complete BOP Challenge testing pipeline for VPOG model.

## Directory Structure

```
test_bop/
├── __init__.py              # Module initialization
├── dataloader.py            # BOP test dataloader with CNOS detections
├── tester.py                # Main testing orchestrator
├── bop_formatter.py         # Convert predictions to BOP CSV format
├── evaluator.py             # BOP toolkit evaluation runner
├── config/
│   └── test.yaml            # Test configuration
└── scripts/
    ├── test_single.py       # Test single dataset
    └── test_all_bop.py      # Test all BOP core datasets
```

## Architecture

### Module Responsibilities

**test_bop/** (BOP-specific testing)
- Load BOP test images with CNOS default detections
- Crop images based on bounding boxes
- Load all 162 templates per object
- Convert predictions to BOP CSV format
- Run BOP toolkit evaluation
- Aggregate results

**vpog/inference/** (General pose estimation)
- Template loading and management
- 2D-3D correspondence building
- RANSAC PnP solving
- Return pose + timing
- BOP-agnostic

### Data Flow

```
BOP Test Image + Detection
         ↓
BOPTestDataset (test_bop/dataloader.py)
    - Crop image based on bbox
    - Load 162 templates
         ↓
VPOGBOPTester (test_bop/tester.py)
    - Load VPOG checkpoint
    - Call vpog.inference for pose
         ↓
vpog.inference.pipeline.InferencePipeline
    - Build correspondences
    - Run RANSAC PnP
    - Return pose + timing
         ↓
BOPFormatter (test_bop/bop_formatter.py)
    - Convert to CSV: scene_id, im_id, obj_id, score, R, t, time
         ↓
BOPEvaluator (test_bop/evaluator.py)
    - Call bop_toolkit for VSD/MSSD/MSPD/AR
    - Save JSON results
```

## Usage

### Quick Start

```bash
# Test single dataset
python test_bop/scripts/test_single.py \
    --checkpoint /path/to/vpog.ckpt \
    --dataset ycbv \
    --output ./results

# Test all BOP core datasets
python test_bop/scripts/test_all_bop.py \
    --checkpoint /path/to/vpog.ckpt \
    --output ./results_all
```

### Configuration

Edit `test_bop/config/test.yaml`:

```yaml
# Paths
checkpoint_path: /path/to/vpog_checkpoint.ckpt
root_dir: /path/to/bop/datasets
templates_dir: /path/to/templates
output_dir: ./test_results

# Test settings
dataset_name: ycbv
test_setting: localization  # or "detection"

# Inference parameters
inference:
  device: cuda
  ransac_threshold: 8.0
  ransac_iterations: 1000
  min_inliers: 4
```

### Command-Line Overrides

```bash
# Override checkpoint
python test_bop/scripts/test_single.py \
    --config test_bop/config/test.yaml \
    --checkpoint /path/to/new_ckpt.ckpt

# Override dataset
python test_bop/scripts/test_single.py \
    --dataset tless

# Skip evaluation (predictions only)
python test_bop/scripts/test_single.py \
    --no-eval

# Test specific datasets
python test_bop/scripts/test_all_bop.py \
    --datasets ycbv tless lmo
```

## Output Format

### Directory Structure

```
results/
├── predictions/
│   ├── 000000.npz          # Batch predictions
│   ├── 000001.npz
│   └── ...
├── vpog-pbrreal-rgb-mmodel_ycbv-test_test.csv  # BOP CSV
└── eval/
    ├── ycbv_scores.json    # BOP evaluation results
    └── all_scores.json     # Aggregated results
```

### CSV Format

BOP-compliant format matching GigaPose:

```csv
scene_id,im_id,obj_id,score,R,t,time
1,0,1,0.95,0.9 0.1 0.0 -0.1 0.9 0.0 0.0 0.0 1.0,100.0 50.0 500.0,0.123
```

Where:
- `scene_id`: Scene ID
- `im_id`: Image ID
- `obj_id`: Object ID
- `score`: Confidence score (inlier ratio)
- `R`: 9 space-separated rotation matrix values (row-major)
- `t`: 3 space-separated translation values (mm)
- `time`: Total inference time (seconds)

## BOP Evaluation

Metrics computed by BOP toolkit:

- **VSD** (Visible Surface Discrepancy): Visual similarity metric
- **MSSD** (Maximum Symmetry-Aware Surface Distance): 3D surface error
- **MSPD** (Maximum Symmetry-Aware Projection Distance): 2D reprojection error
- **AR** (Average Recall): Combined metric = mean(VSD, MSSD, MSPD)

### Datasets with GT

- lmo (LineMOD-Occluded)
- tudl (TUD-Light)
- icbin (IC-BIN)
- tless (T-LESS)
- ycbv (YCB-Video)

### Datasets without GT

- itodd (ITODD)
- hb (HomebrewedDB)

These datasets output predictions only (for BOP Challenge submission).

## Implementation Status

### ✅ Fully Implemented - Ready for Testing!

All critical components are now complete:

1. **dataloader.py** 
   - ✅ Image loading with bbox cropping
   - ✅ Template loading (all 162 per object)
   - ✅ Detection metadata handling
   - ✅ Camera intrinsics transformation
   - ✅ WebSceneDataset integration
   - ✅ TemplateDataset integration

2. **tester.py**
   - ✅ VPOG checkpoint loading via VPOGLightningModule
   - ✅ InferencePipeline integration
   - ✅ Pose estimation per detection
   - ✅ Result batching and saving
   - ✅ Error handling and logging

3. **bop_formatter.py** - ✅ Complete CSV format conversion
4. **evaluator.py** - ✅ Complete BOP toolkit integration
5. **Scripts** - ✅ Single and multi-dataset testing ready
6. **Documentation** - ✅ Complete usage guide

### Next Steps

The pipeline is ready for end-to-end testing:
```bash
# Small test on subset
python test_bop/scripts/test_single.py --checkpoint model.ckpt --dataset ycbv

# Full BOP evaluation
python test_bop/scripts/test_all_bop.py --checkpoint model.ckpt
```

## Comparison with GigaPose

### Similarities

- Same BOP CSV output format
- Same evaluation pipeline (BOP toolkit)
- Same default detections (CNOS)
- Same test_targets for localization mode

### Differences

| Aspect | GigaPose | VPOG |
|--------|----------|------|
| Inference | Template matching + IST | Flow + Classification + RANSAC |
| Templates | Encoded features | Full RGB images |
| Correspondence | Patch matching | Dense flow |
| Pose Recovery | Affine + PnP | Direct RANSAC PnP |
| Module | src/models/gigaPose.py | vpog/inference/pipeline.py |
| Dataloader | src/dataloader/test.py | test_bop/dataloader.py |

## Development

### Adding New Features

1. **New test setting**: Extend `test_setting` in config
2. **New metric**: Add to BOPEvaluator
3. **New dataset**: Add to BOP_CORE_DATASETS in test_all_bop.py

### Testing Changes

```bash
# Test dataloader
python test_bop/dataloader.py

# Test formatter
python test_bop/bop_formatter.py

# Test evaluator
python test_bop/evaluator.py

# Test tester (requires checkpoint)
python test_bop/tester.py
```

## References

- BOP Challenge: http://bop.felk.cvut.cz/
- BOP Toolkit: https://github.com/thodan/bop_toolkit
- GigaPose: `src/models/gigaPose.py`, `test.py`
- VPOG Inference: `vpog/inference/pipeline.py`
