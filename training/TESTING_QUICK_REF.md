# VPOG Testing Quick Reference

## Test Commands

### 🚀 Recommended Test Order

```bash
# 1. Model unit test (no data dependencies)
python -m vpog.models.test_vpog_integration

# 2. Dataloader test (tests data pipeline)
python -m training.dataloader.test_integration

# 3. Full pipeline with synthetic data (tests everything)
python -m training.test_vpog_full_pipeline

# 4. Full pipeline with real data (if available)
python -m training.test_vpog_full_pipeline --real_data
```

## Test Descriptions

### 1. Model Unit Test
**File**: `vpog/models/test_vpog_integration.py`
**Purpose**: Test VPOG model components in isolation
**Tests**:
- ✓ Encoder wrapper
- ✓ TokenManager (added tokens + rope_mask)
- ✓ AA module (global-local attention)
- ✓ Classification head
- ✓ Flow head
- ✓ Full model forward pass

**Runtime**: ~10 seconds
**Dependencies**: Model code only

---

### 2. Dataloader Integration Test
**File**: `training/dataloader/test_integration.py`
**Purpose**: Test data loading pipeline
**Tests**:
- ✓ VPOGTrainDataset initialization
- ✓ Batch loading (VPOGBatch)
- ✓ Shape verification
- ✓ Data properties (d_ref normalization, etc.)
- ✓ Visualizations

**Runtime**: ~30 seconds (with data), ~5 seconds (without data)
**Dependencies**: Dataset + templates (optional)

---

### 3. Pixel-Level Flow Test
**File**: `training/dataloader/test_pixel_flow.py`
**Purpose**: Test 16×16 pixel-level flow computation
**Tests**:
- ✓ Single patch flow
- ✓ All patches flow (196×196×16×16)
- ✓ Unseen mask generation
- ✓ Flow normalization

**Runtime**: ~20 seconds
**Dependencies**: FlowComputer only

---

### 4. Full Pipeline Test (Synthetic)
**File**: `training/test_vpog_full_pipeline.py`
**Command**: `python -m training.test_vpog_full_pipeline`
**Purpose**: Test complete VPOG pipeline with synthetic data
**Tests**:
- ✓ Encoder integration with VPOGBatch
- ✓ Full model forward pass
- ✓ All loss functions
- ✓ Correspondence construction
- ✓ Visualizations

**Runtime**: ~30 seconds
**Dependencies**: Full VPOG + training code

---

### 5. Full Pipeline Test (Real Data)
**File**: `training/test_vpog_full_pipeline.py`
**Command**: `python -m training.test_vpog_full_pipeline --real_data`
**Purpose**: Test complete VPOG pipeline with real GSO data
**Tests**: Same as synthetic + real data loading

**Runtime**: ~60 seconds
**Dependencies**: GSO dataset + templates

---

## Test Outputs

All tests save outputs to `/tmp/` directories:

```
/tmp/vpog_test/                      # Model unit test
/tmp/vpog_vis/                       # Flow visualization test  
/tmp/vpog_flow_test/                 # Pixel flow test
/tmp/vpog_integration_test/          # Dataloader test
/tmp/vpog_full_pipeline_test/        # Full pipeline test
```

## Expected Output Format

### Successful Test Output:
```
================================================================================
VPOG FULL PIPELINE INTEGRATION TEST
================================================================================
Device: cuda
Using real data: False
Output directory: ./tmp/vpog_full_pipeline_test

--------------------------------------------------------------------------------
Step 1: Loading Data
--------------------------------------------------------------------------------
✓ Created synthetic batch
  Batch size: 2
  Num templates: 5
  Image size: torch.Size([224, 224])

... (more tests) ...

================================================================================
TEST SUMMARY
================================================================================
✓ ALL TESTS PASSED!

Visualizations saved to: ./tmp/vpog_full_pipeline_test

Next steps:
  1. Review visualizations
  2. Test with real GSO data: python -m training.test_vpog_full_pipeline --real_data
  3. Run training: python train.py
```

### Failed Test Output:
```
✗ Model forward pass test FAILED: <error message>
  <stack trace>

✗ SOME TESTS FAILED
  Review error messages above for details
```

## Debugging Tips

### Test fails with import errors:
```bash
# Add project to PYTHONPATH
export PYTHONPATH=/data/home/ssaricha/gigapose:$PYTHONPATH
```

### Test fails with CUDA errors:
```bash
# Run on CPU
python -m training.test_vpog_full_pipeline --device cpu
```

### Test fails with "checkpoint not found":
```bash
# Check CroCo checkpoint exists
ls -lh checkpoints/CroCo_V2_ViTBase_BaseDecoder.pth
```

### Test fails with "GSO data not found":
```bash
# Check data directories
ls datasets/gso/train_pbr_web/
ls datasets/templates/gso/
```

## Test Coverage

| Component | Unit Test | Integration Test | Full Pipeline Test |
|-----------|-----------|------------------|-------------------|
| Encoder | ✅ | ✅ | ✅ |
| AA Module | ✅ | ❌ | ✅ |
| TokenManager | ✅ | ❌ | ✅ |
| Classification Head | ✅ | ❌ | ✅ |
| Flow Head | ✅ | ❌ | ✅ |
| VPOGTrainDataset | ❌ | ✅ | ✅ |
| FlowComputer | ❌ | ✅ | ✅ |
| Template Selector | ❌ | ✅ | ✅ |
| Classification Loss | ❌ | ❌ | ✅ |
| Flow Loss | ❌ | ❌ | ✅ |
| Correspondence Builder | ❌ | ❌ | ✅ |
| Visualization | ✅ | ✅ | ✅ |

## Quick Verification

Run all tests in sequence:
```bash
#!/bin/bash
echo "Running all VPOG tests..."

echo "1. Model unit test..."
python -m vpog.models.test_vpog_integration || exit 1

echo "2. Dataloader test..."
python -m training.dataloader.test_integration || exit 1

echo "3. Full pipeline test..."
python -m training.test_vpog_full_pipeline || exit 1

echo "✓ All tests passed!"
```

Save as `run_all_tests.sh`, make executable: `chmod +x run_all_tests.sh`, then run: `./run_all_tests.sh`
