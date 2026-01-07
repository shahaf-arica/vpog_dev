# VPOG Pose Solvers

Stage 2 of the VPOG inference pipeline: 6D pose estimation from 2D-3D correspondences.

## Overview

Two pose solvers are implemented:

1. **PnPSolver** (OpenCV RANSAC-based) - **Fully functional**
2. **EProPnPSolver** (Probabilistic PnP) - **Has PyTorch compatibility issues**

## PnPSolver

RANSAC-based PnP solver using OpenCV's `solvePnPRansac`. **This is the recommended solver for inference.**

### Features
- Robust to outliers via RANSAC
- Fast and reliable
- Configurable parameters (threshold, iterations, confidence)
- Batch processing support

### Usage
```python
from vpog.inference import PnPSolver

solver = PnPSolver(
    ransac_reproj_threshold=8.0,  # Inlier threshold in pixels
    ransac_iterations=1000,         # RANSAC iterations
    ransac_confidence=0.99          # Confidence level
)

# Single image
pose, inliers = solver.solve(pts_2d, pts_3d, K)

# Batch
poses, inliers_list = solver.solve_batch(pts_2d_batch, pts_3d_batch, K_batch)

# Convert to 4x4 matrix
pose_matrix = PnPSolver.pose_to_matrix(pose)
```

### Performance
Tested on synthetic data with:
- **Clean data**: 0.000° rotation error, 0.0000m translation error
- **With 2px noise**: 0.961° rotation error, 0.0031m translation error
- **With 30% outliers**: 0.186° rotation error (RANSAC filters outliers)

## EProPnPSolver

Probabilistic PnP solver using EPro-PnP-6DoF. **Currently has compatibility issues with PyTorch 2.x.**

### Status
⚠️ **Known Issues**: EPro-PnP uses `torch.solve` which was deprecated in PyTorch 1.9 and removed in 2.x.
- The external library was written for PyTorch 1.5
- Results in `inf` cost and identity pose with PyTorch 2.x
- Mainly intended for differentiable training loss, not required for inference

### Features (when working)
- Probabilistic pose estimation via Monte Carlo sampling
- Handles correspondence uncertainties
- Differentiable (useful for training)

### Usage
```python
from vpog.inference import EProPnPSolver, EPROPNP_AVAILABLE

if EPROPNP_AVAILABLE:
    solver = EProPnPSolver(
        mc_samples=512,     # Monte Carlo samples
        num_iter=4,         # AMIS iterations
        solver_iter=5       # LM iterations
    )
    
    # With uncertainty weights (per-axis standard deviation)
    weights = torch.ones(N, 2)  # Shape: [N, 2] for x,y uncertainty
    pose, cost = solver.solve(pts_2d, pts_3d, K, weights)
```

### Fixing EPro-PnP (Optional)

If you need EPro-PnP functionality, you'll need to patch the library:

1. Locate `external/epropnp/EPro-PnP-6DoF_v2/lib/ops/pnp/levenberg_marquardt.py`
2. Replace `torch.solve(B, A)` with `torch.linalg.solve(A, B)` (note argument order change)
3. Test with: `PYTHONPATH=/data/home/ssaricha/gigapose:$PYTHONPATH python vpog/inference/test_pose_solvers.py`

## Testing

Run the comprehensive test suite:
```bash
cd /data/home/ssaricha/gigapose
PYTHONPATH=/data/home/ssaricha/gigapose:$PYTHONPATH python vpog/inference/test_pose_solvers.py
```

### Test Coverage
- ✅ PnPSolver basic (clean data)
- ✅ PnPSolver with noise
- ✅ PnPSolver with outliers (RANSAC robustness)
- ✅ PnPSolver batch processing
- ✅ Pose to matrix conversion
- ✅ EPro-PnP availability check
- ⚠️ EPro-PnP basic (fails with PyTorch 2.x)
- ⚠️ EPro-PnP with weights (fails with PyTorch 2.x)

## API Reference

### PnPSolver

**Constructor:**
```python
PnPSolver(
    ransac_reproj_threshold: float = 8.0,
    ransac_iterations: int = 1000,
    ransac_confidence: float = 0.99
)
```

**Methods:**
- `solve(pts_2d, pts_3d, K)` → `(pose, inliers)`
- `solve_batch(pts_2d_batch, pts_3d_batch, K_batch)` → `(poses, inliers_list)`
- `pose_to_matrix(pose)` → `pose_matrix` (4×4 transformation)
- `pose_to_Rt(pose)` → `(R, t)` (rotation matrix + translation)

**Inputs:**
- `pts_2d`: [N, 2] 2D points in pixels
- `pts_3d`: [N, 3] 3D points in meters
- `K`: [3, 3] camera intrinsics

**Outputs:**
- `pose`: [6] pose as [rvec (3), tvec (3)]
- `inliers`: [N] boolean mask of inliers
- `pose_matrix`: [4, 4] homogeneous transformation

### EProPnPSolver

**Constructor:**
```python
EProPnPSolver(
    mc_samples: int = 512,
    num_iter: int = 4,
    solver_iter: int = 5
)
```

**Methods:**
- `solve(pts_2d, pts_3d, K, weights=None)` → `(pose, cost)`
- `pose_to_matrix(pose)` → `pose_matrix` (4×4 transformation)
- `pose_to_Rt(pose)` → `(R, t)` (rotation matrix + translation)

**Inputs:**
- `pts_2d`: [N, 2] 2D points in pixels
- `pts_3d`: [N, 3] 3D points in meters
- `K`: [3, 3] camera intrinsics
- `weights`: [N, 2] per-axis uncertainty (optional)

**Outputs:**
- `pose`: [7] pose as [x,y,z, qw,qx,qy,qz]
- `cost`: scalar cost value
- `pose_matrix`: [4, 4] homogeneous transformation

## Recommendations

1. **For inference**: Use `PnPSolver` - it's fast, reliable, and robust
2. **For training**: Consider patching EPro-PnP if you need differentiable pose loss
3. **For production**: `PnPSolver` with RANSAC is the proven solution

## Next Steps

- **Stage 3**: Template Manager (selection, loading, caching)
- **Stage 4**: Full inference pipeline integration
- (Optional) Patch EPro-PnP for PyTorch 2.x compatibility
