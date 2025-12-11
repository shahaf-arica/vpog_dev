# VPOG Quick Start Guide

Quick commands to test the VPOG implementation during the debug session.

## 1. Environment Setup

```bash
# Activate environment
conda activate gigapose  # or your environment name

# Verify PyTorch and dependencies
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 2. Run Integration Tests

### Test Full VPOG Pipeline
```bash
cd /data/home/ssaricha/gigapose

# With synthetic data (fast, no dependencies)
python -m training.test_vpog_full_pipeline

# With real GSO data
python -m training.test_vpog_full_pipeline --real_data
```

### Test Dataloader Only
```bash
cd /data/home/ssaricha/gigapose
python -m training.dataloader.test_integration
```

### Run VPOG Model Test
```bash
cd /data/home/ssaricha/gigapose
python -m vpog.models.test_vpog_integration
```

### Run Flow Visualization Test
```bash
cd /data/home/ssaricha/gigapose
python -m training.visualization.test_flow_vis
```

### Run FlowComputer Test
```bash
cd /data/home/ssaricha/gigapose
python -m training.dataloader.test_pixel_flow
```

## 3. Interactive Testing

### Test Model Components

```python
import torch
from vpog.models.vpog_model import VPOGModel

# Create model
model = VPOGModel(
    croco_checkpoint='checkpoints/CroCo_V2_ViTBase_BaseDecoder.pth',
    patch_size=16,
    enc_embed_dim=768,
    enc_depth=12,
    enc_num_heads=12,
    aa_depth=4,
    aa_num_heads=12,
    num_query_added_tokens=0,
    num_template_added_tokens=1,
)

# Create dummy data
query_data = {
    'features': torch.randn(1, 1, 196, 768),
    'positions': torch.randn(1, 1, 196, 2),
    'frame_dirs': torch.randn(1, 1, 3),
    'frame_has_s2': torch.zeros(1, 1, dtype=torch.bool),
}
template_data = {
    'features': torch.randn(1, 1, 196, 768),
    'positions': torch.randn(1, 1, 196, 2),
    'frame_dirs': torch.randn(1, 1, 3),
    'frame_has_s2': torch.ones(1, 1, dtype=torch.bool),
    'ref_dirs': torch.randn(1, 3),
}

# Forward pass
model.eval()
with torch.no_grad():
    outputs = model(query_data, template_data)
    
print("Classification logits:", outputs['classification_logits'].shape)
print("Flow:", outputs['flow'].shape)
print("Flow confidence:", outputs['flow_confidence'].shape)
```

### Test Loss Functions

```python
from training.losses import ClassificationLoss, FlowLoss

# Create losses
cls_loss = ClassificationLoss(tau=1.0)
flow_loss = FlowLoss(loss_type='l1')

# Create dummy predictions and GT
B, S, Nq, Nt = 1, 1, 196, 196
logits = torch.randn(B, S, Nq, Nt+1)  # +1 for unseen token
labels = torch.randint(0, Nt, (B, S, Nq))
unseen_mask = torch.rand(B, S, Nq) < 0.1  # 10% unseen

flow_pred = torch.randn(B, S, Nq, Nt, 16, 16, 2)
flow_gt = torch.randn(B, S, Nq, Nt, 16, 16, 2)
conf_pred = torch.rand(B, S, Nq, Nt, 16, 16)
flow_unseen = torch.rand(B, S, Nq, Nt, 16, 16) < 0.2  # 20% unseen

# Compute losses
cls_loss_val = cls_loss(logits, labels, unseen_mask, unseen_token_idx=Nt)
flow_loss_val = flow_loss(flow_pred, flow_gt, conf_pred, flow_unseen)

print(f"Classification loss: {cls_loss_val.item():.4f}")
print(f"Flow loss: {flow_loss_val.item():.4f}")
```

### Test Correspondence Builder

```python
from vpog.inference import CorrespondenceBuilder

# Create correspondence builder
corr_builder = CorrespondenceBuilder(
    patch_size=16,
    conf_threshold=0.5,
    use_pixel_level=True,
)

# Build correspondences (using outputs from model)
correspondences = corr_builder(
    outputs['classification_logits'],
    outputs['flow'],
    outputs['flow_confidence'],
    query_data,
    template_data,
)

print(f"Number of correspondences: {correspondences['num_correspondences']}")
if correspondences['num_correspondences'] > 0:
    print(f"2D points: {correspondences['pts2d'].shape}")
    print(f"3D points: {correspondences['pts3d'].shape}")
```

### Test Visualization

```python
from training.visualization import visualize_patch_flow
import matplotlib.pyplot as plt

# Visualize flow predictions
fig = visualize_patch_flow(
    outputs['flow'][0, 0].cpu().numpy(),        # [Nq, Nt, 16, 16, 2]
    outputs['flow_confidence'][0, 0].cpu().numpy(),  # [Nq, Nt, 16, 16]
    torch.zeros_like(outputs['flow_confidence'][0, 0]).bool().cpu().numpy(),  # Dummy unseen
    query_data['image'][0, 0].cpu() if 'image' in query_data else None,
    template_data['image'][0, 0].cpu() if 'image' in template_data else None,
)
plt.show()
```

## 4. Check Implementation Status

```bash
# List all VPOG files
find vpog/ -name "*.py" -type f | sort

# Count lines of code
find vpog/ -name "*.py" -type f -exec wc -l {} + | tail -1

# Check for syntax errors
python -m py_compile vpog/models/*.py
python -m py_compile vpog/losses/*.py
python -m py_compile vpog/training/*.py
python -m py_compile vpog/inference/*.py
```

## 5. Common Debug Commands

### Check Model Parameters
```python
# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M")
```

### Check GPU Memory
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### Profile Forward Pass
```python
import time

model = model.cuda()
query_data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in query_data.items()}
template_data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in template_data.items()}

# Warmup
for _ in range(3):
    with torch.no_grad():
        outputs = model(query_data, template_data)

# Profile
torch.cuda.synchronize()
start = time.time()
for _ in range(10):
    with torch.no_grad():
        outputs = model(query_data, template_data)
torch.cuda.synchronize()
elapsed = time.time() - start
print(f"Average forward time: {elapsed/10*1000:.2f} ms")
```

## 6. Troubleshooting

### Import Errors
```bash
# Add project to PYTHONPATH
export PYTHONPATH=/data/home/ssaricha/gigapose:$PYTHONPATH

# Or in Python
import sys
sys.path.insert(0, '/data/home/ssaricha/gigapose')
```

### CroCo Checkpoint Issues
```bash
# Verify checkpoint exists
ls -lh checkpoints/CroCo_V2_ViTBase_BaseDecoder.pth

# If missing, download from CroCo repository
```

### CUDA Errors
```python
# Force CPU if GPU issues
model = model.cpu()
query_data = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in query_data.items()}
template_data = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in template_data.items()}
```

## 7. File Locations

Key files for debugging:

```
# VPOG model package (model-only)
vpog/models/vpog_model.py                   # Main model
vpog/models/test_vpog_integration.py        # Model unit test
vpog/inference/correspondence.py            # Correspondence builder

# Training package (training-specific)
training/losses/classification_loss.py      # Classification loss
training/losses/flow_loss.py                # Flow loss
training/lightning_module.py                # Training module
training/visualization/flow_vis.py          # Visualization
training/dataloader/vpog_dataset.py         # VPOG dataset
training/dataloader/test_integration.py     # Dataloader test
training/test_vpog_full_pipeline.py         # Full pipeline test
```

## 8. Expected Outputs

When running integration tests, you should see:

```
=== Creating VPOG Model ===
✓ Model created with XX.XXM parameters

=== Creating Synthetic Test Data ===
✓ Test data created

=== Testing Encoder Output ===
Query features shape: torch.Size([1, 1, 196, 768])
Template features shape: torch.Size([1, 1, 196, 768])
✓ Encoder output validation passed

=== Testing AA Module ===
Query with tokens: torch.Size([1, 1, 196, 768])
Template with tokens: torch.Size([1, 1, 197, 768])
✓ AA module validation passed

=== Testing Classification Predictions ===
Classification logits shape: torch.Size([1, 1, 196, 197])
✓ Classification predictions validation passed

=== Testing Flow Predictions ===
Flow shape: torch.Size([1, 1, 196, 196, 16, 16, 2])
✓ Flow predictions validation passed

============================================================
ALL TESTS PASSED ✓
============================================================
```

## Ready for Debug Session! 🚀

All components implemented and ready for testing.
