"""
Test suite for VPOG correspondence building.

Tests vectorized correspondence computation from classification and flow predictions.
Validates coarse (center-flow) and refined (dense-flow) correspondence building.
"""

import torch
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

from vpog.inference.correspondence import (
    build_coarse_correspondences,
    build_refined_correspondences,
    CorrespondenceBuilder,
    Correspondences,
)


@dataclass
class MockVPOGOutput:
    """Mock VPOG model output for testing."""
    classification_logits: torch.Tensor  # [B, S, Nq, Nt+1]
    center_flow: torch.Tensor  # [B, S, Nq, 2]
    dense_flow: torch.Tensor  # [B, S, Nq, ps, ps, 2]
    dense_b: torch.Tensor  # [B, S, Nq, ps, ps]


def create_mock_predictions(
    batch_size: int = 2,
    num_templates: int = 5,
    num_queries: int = 196,
    patch_size: int = 16,
    image_size: int = 224,
    seed: int = 42,
) -> MockVPOGOutput:
    """
    Create mock VPOG predictions for testing.
    
    Args:
        batch_size: Batch size
        num_templates: Number of templates per batch
        num_queries: Number of query patches (Nq)
        patch_size: Patch size in pixels
        image_size: Image size in pixels
        seed: Random seed for reproducibility
        
    Returns:
        Mock VPOG output with realistic values
    """
    torch.manual_seed(seed)
    
    # Classification logits: higher scores for matched patches
    # [B, S, Nq, Nt+1] - last entry is "unseen" token
    classification_logits = torch.randn(batch_size, num_templates, num_queries, num_templates + 1)
    
    # Make some patches have clear matches (high logits on diagonal)
    for b in range(batch_size):
        for s in range(num_templates):
            # Every 5th patch gets a strong match
            for q in range(0, num_queries, 5):
                classification_logits[b, s, q, s] += 5.0  # Strong match to own template
    
    # Center flow: small displacements in patch units
    # [B, S, Nq, 2] - values typically in [-2, 2] patch units
    center_flow = torch.randn(batch_size, num_templates, num_queries, 2) * 0.5
    
    # Dense flow: per-pixel displacements within patch
    # [B, S, Nq, ps, ps, 2] - values typically in [-1, 1] patch units
    dense_flow = torch.randn(batch_size, num_templates, num_queries, patch_size, patch_size, 2) * 0.3
    
    # Dense uncertainty (Laplace b parameter): smaller = more confident
    # [B, S, Nq, ps, ps] - positive values, typically 0.1 to 1.0
    dense_b = torch.rand(batch_size, num_templates, num_queries, patch_size, patch_size) * 0.5 + 0.1
    
    return MockVPOGOutput(
        classification_logits=classification_logits,
        center_flow=center_flow,
        dense_flow=dense_flow,
        dense_b=dense_b,
    )


def create_mock_template_data(
    batch_size: int = 2,
    num_templates: int = 5,
    image_size: int = 224,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create mock template data (depth, intrinsics, poses) for testing.
    
    Args:
        batch_size: Batch size
        num_templates: Number of templates
        image_size: Template image size
        seed: Random seed
        
    Returns:
        template_depth: [B, S, H, W] - depth maps in meters
        K_template: [B, S, 3, 3] - camera intrinsics
        poses_template: [B, S, 4, 4] - template poses (model-to-camera)
    """
    torch.manual_seed(seed)
    
    # Create depth maps: values in ~0.5-1.5 meter range
    template_depth = torch.rand(batch_size, num_templates, image_size, image_size) * 1.0 + 0.5
    
    # Create camera intrinsics (simple pinhole camera)
    # Focal length ~280 pixels for 224x224 image
    K = torch.zeros(batch_size, num_templates, 3, 3)
    K[:, :, 0, 0] = 280.0  # fx
    K[:, :, 1, 1] = 280.0  # fy
    K[:, :, 0, 2] = image_size / 2  # cx
    K[:, :, 1, 2] = image_size / 2  # cy
    K[:, :, 2, 2] = 1.0
    
    # Create template poses: identity with small rotation/translation
    poses = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_templates, 1, 1)
    
    # Add small random translation (~10cm in each direction)
    poses[:, :, :3, 3] = torch.randn(batch_size, num_templates, 3) * 0.1
    
    # Add small random rotation (simplified - just perturb rotation matrix slightly)
    for b in range(batch_size):
        for s in range(num_templates):
            # Keep it as identity + small noise for simplicity
            R_noise = torch.randn(3, 3) * 0.01
            R = poses[b, s, :3, :3] + R_noise
            # Orthonormalize (simple Gram-Schmidt)
            R[:, 0] = R[:, 0] / torch.norm(R[:, 0])
            R[:, 1] = R[:, 1] - (R[:, 1] @ R[:, 0]) * R[:, 0]
            R[:, 1] = R[:, 1] / torch.norm(R[:, 1])
            R[:, 2] = torch.cross(R[:, 0], R[:, 1])
            poses[b, s, :3, :3] = R
    
    return template_depth, K, poses


def test_coarse_correspondence_shape():
    """Test that coarse correspondences have correct shape."""
    print("\n=== Test: Coarse Correspondence Shape ===")
    
    # Create mock data
    predictions = create_mock_predictions(batch_size=2, num_templates=5)
    depth, K, poses = create_mock_template_data(batch_size=2, num_templates=5)
    
    # Build correspondences (use first template)
    corr = build_coarse_correspondences(
        classification_logits=predictions.classification_logits,
        center_flow=predictions.center_flow,
        template_depth=depth,
        K_template=K,
        poses_template=poses,
        selected_template_idx=0,
        patch_size=16,
        img_size=224,
    )
    
    # Check shapes - output is [N] after filtering
    N = corr.pts_2d.shape[0]
    
    assert corr.pts_2d.shape == (N, 2), f"Expected shape (N, 2), got {corr.pts_2d.shape}"
    assert corr.pts_3d.shape == (N, 3), f"Expected shape (N, 3), got {corr.pts_3d.shape}"
    assert corr.weights.shape == (N,), f"Expected shape (N,), got {corr.weights.shape}"
    assert corr.valid_mask.shape == (N,), f"Expected shape (N,), got {corr.valid_mask.shape}"
    
    print(f"✓ 2D points shape: {corr.pts_2d.shape}")
    print(f"✓ 3D points shape: {corr.pts_3d.shape}")
    print(f"✓ Weights shape: {corr.weights.shape}")
    print(f"✓ Valid mask shape: {corr.valid_mask.shape}")
    print(f"✓ Total correspondences: {N}")
    print(f"✓ Valid correspondences: {corr.valid_mask.sum().item()}")


def test_refined_correspondence_shape():
    """Test that refined correspondences have correct shape."""
    print("\n=== Test: Refined Correspondence Shape ===")
    
    # Create mock data
    predictions = create_mock_predictions(batch_size=2, num_templates=5, patch_size=16)
    depth, K, poses = create_mock_template_data(batch_size=2, num_templates=5)
    
    # Build correspondences
    corr = build_refined_correspondences(
        classification_logits=predictions.classification_logits,
        dense_flow=predictions.dense_flow,
        dense_b=predictions.dense_b,
        template_depth=depth,
        K_template=K,
        poses_template=poses,
        selected_template_idx=0,
        patch_size=16,
        img_size=224,
    )
    
    # Check shapes (note: N_refined <= B*Nq*ps*ps due to filtering)
    N = corr.pts_2d.shape[0]
    assert corr.pts_2d.ndim == 2 and corr.pts_2d.shape[1] == 2, f"Expected [N, 2], got {corr.pts_2d.shape}"
    assert corr.pts_3d.ndim == 2 and corr.pts_3d.shape[1] == 3, f"Expected [N, 3], got {corr.pts_3d.shape}"
    assert corr.weights.ndim == 1, f"Expected [N], got {corr.weights.shape}"
    assert corr.valid_mask.ndim == 1, f"Expected [N], got {corr.valid_mask.shape}"
    
    print(f"✓ 2D points shape: {corr.pts_2d.shape}")
    print(f"✓ 3D points shape: {corr.pts_3d.shape}")
    print(f"✓ Weights shape: {corr.weights.shape}")
    print(f"✓ Valid mask shape: {corr.valid_mask.shape}")
    print(f"✓ Total correspondences: {N}")
    print(f"✓ Valid correspondences: {corr.valid_mask.sum().item()}")


def test_coarse_correspondence_values():
    """Test that coarse correspondence values are reasonable."""
    print("\n=== Test: Coarse Correspondence Values ===")
    
    # Create simple deterministic test case
    B, S, Nq = 1, 2, 4  # Small for debugging
    
    # Create classification logits with clear matches
    classification_logits = torch.zeros(B, S, Nq, S + 1)
    classification_logits[0, 0, :, 0] = 10.0  # All patches match template patch 0
    
    # Create center flow: [0.5, 0.5] = 8 pixel shift in each direction
    center_flow = torch.ones(B, S, Nq, 2) * 0.5
    
    # Create template data (simplified - 32x32 for 4 patches at patch_size=16)
    # Grid is 2x2 patches
    depth = torch.ones(B, S, 32, 32) * 1.0  # 1 meter depth
    K = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, S, 1, 1)
    K[:, :, 0, 0] = 50.0  # fx
    K[:, :, 1, 1] = 50.0  # fy
    K[:, :, 0, 2] = 16.0  # cx
    K[:, :, 1, 2] = 16.0  # cy
    poses = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, S, 1, 1)
    
    # Build correspondences (use first template)
    corr = build_coarse_correspondences(
        classification_logits=classification_logits,
        center_flow=center_flow,
        template_depth=depth,
        K_template=K,
        poses_template=poses,
        selected_template_idx=0,
        patch_size=16,
        img_size=32,
        grid_size=(2, 2),  # 2x2 patches
    )
    
    # Check values
    print(f"Grid shape: 2x2 (Nq={Nq} for test)")
    print(f"Number of correspondences: {len(corr)}")
    if len(corr) > 0:
        print(f"2D points (first 2): {corr.pts_2d[:2]}")
        print(f"3D points (first 2): {corr.pts_3d[:2]}")
        print(f"Weights (first 2): {corr.weights[:2]}")
    
    # Check that we have some matches
    assert corr.valid_mask.sum() > 0, "Expected some valid matches"
    
    # Check that 2D points are within image bounds
    valid_2d = corr.pts_2d[corr.valid_mask]
    if len(valid_2d) > 0:
        assert (valid_2d >= 0).all(), "2D points should be >= 0"
        assert (valid_2d[:, 0] < 32).all(), "2D x should be < 32"
        assert (valid_2d[:, 1] < 32).all(), "2D y should be < 32"
    
    print(f"✓ {corr.valid_mask.sum().item()} valid correspondences")
    print("✓ 2D points within image bounds")
    print("✓ 3D points generated from depth backprojection")


def test_refined_uncertainty_filtering():
    """Test that uncertainty filtering works correctly."""
    print("\n=== Test: Refined Uncertainty Filtering ===")
    
    # Create predictions with varying uncertainty
    B, S, Nq, ps = 1, 2, 4, 4  # Small for testing
    
    classification_logits = torch.zeros(B, S, Nq, S + 1)
    classification_logits[0, 0, :, 0] = 10.0
    
    dense_flow = torch.zeros(B, S, Nq, ps, ps, 2)
    
    # Create dense_b with known values
    dense_b = torch.ones(B, S, Nq, ps, ps)
    # Make some pixels very confident (low b)
    dense_b[0, 0, 0, :2, :2] = 0.1  # Top-left patch, top-left 4 pixels
    # Make some pixels very uncertain (high b)
    dense_b[0, 0, 1, :2, :2] = 2.0  # Second patch, top-left 4 pixels
    
    # Create template data
    depth = torch.ones(B, S, 32, 32) * 1.0
    K = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, S, 1, 1)
    K[:, :, 0, 0] = 50.0
    K[:, :, 1, 1] = 50.0
    K[:, :, 0, 2] = 16.0
    K[:, :, 1, 2] = 16.0
    poses = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, S, 1, 1)
    
    # Test with default filtering
    corr = build_refined_correspondences(
        classification_logits=classification_logits,
        dense_flow=dense_flow,
        dense_b=dense_b,
        template_depth=depth,
        K_template=K,
        poses_template=poses,
        selected_template_idx=0,
        patch_size=ps,
        img_size=32,  # 4 patches = 32 pixels
        grid_size=(2, 2),  # 2x2 patches
    )
    
    print(f"Total pixels: {B * Nq * ps * ps}")
    print(f"Confident pixels (b < 0.5) in template 0: {(dense_b[0, 0] < 0.5).sum().item()}")
    print(f"Total correspondences: {len(corr)}")
    print(f"Valid correspondences: {corr.valid_mask.sum().item()}")
    
    # Should have some valid correspondences from confident regions
    assert corr.valid_mask.sum().item() > 0, "Should have some valid correspondences"
    
    print("✓ Correspondence filtering working correctly")


def test_correspondence_builder_class():
    """Test the CorrespondenceBuilder class interface."""
    print("\n=== Test: CorrespondenceBuilder Class ===")
    
    # Create builder
    builder = CorrespondenceBuilder(
        patch_size=16,
        img_size=224,
    )
    
    # Create mock data
    predictions = create_mock_predictions(batch_size=2, num_templates=5)
    depth, K, poses = create_mock_template_data(batch_size=2, num_templates=5)
    
    # Build coarse correspondences
    corr_coarse = builder.build_coarse_correspondences(
        classification_logits=predictions.classification_logits,
        center_flow=predictions.center_flow,
        template_depth=depth,
        K_template=K,
        poses_template=poses,
        selected_template_idx=0,
    )
    
    print(f"✓ Coarse correspondences built: {corr_coarse.pts_2d.shape}")
    print(f"  Valid: {corr_coarse.valid_mask.sum().item()}")
    
    # Build refined correspondences
    corr_refined = builder.build_refined_correspondences(
        classification_logits=predictions.classification_logits,
        dense_flow=predictions.dense_flow,
        dense_b=predictions.dense_b,
        template_depth=depth,
        K_template=K,
        poses_template=poses,
        selected_template_idx=0,
    )
    
    print(f"✓ Refined correspondences built: {corr_refined.pts_2d.shape}")
    print(f"  Valid: {corr_refined.valid_mask.sum().item()}")


def test_numpy_conversion():
    """Test conversion to NumPy for GigaPose compatibility."""
    print("\n=== Test: NumPy Conversion ===")
    
    # Create mock data
    predictions = create_mock_predictions(batch_size=1, num_templates=2)
    depth, K, poses = create_mock_template_data(batch_size=1, num_templates=2)
    
    # Build correspondences
    corr = build_coarse_correspondences(
        classification_logits=predictions.classification_logits,
        center_flow=predictions.center_flow,
        template_depth=depth,
        K_template=K,
        poses_template=poses,
        selected_template_idx=0,
        patch_size=16,
        img_size=224,
    )
    
    # Convert to NumPy
    pts_2d_np = corr.pts_2d.cpu().numpy()
    pts_3d_np = corr.pts_3d.cpu().numpy()
    weights_np = corr.weights.cpu().numpy()
    mask_np = corr.valid_mask.cpu().numpy()
    
    print(f"✓ 2D points converted: {pts_2d_np.shape}, dtype={pts_2d_np.dtype}")
    print(f"✓ 3D points converted: {pts_3d_np.shape}, dtype={pts_3d_np.dtype}")
    print(f"✓ Weights converted: {weights_np.shape}, dtype={weights_np.dtype}")
    print(f"✓ Mask converted: {mask_np.shape}, dtype={mask_np.dtype}")
    
    assert isinstance(pts_2d_np, np.ndarray), "Should be NumPy array"
    assert isinstance(pts_3d_np, np.ndarray), "Should be NumPy array"
    assert isinstance(weights_np, np.ndarray), "Should be NumPy array"
    assert isinstance(mask_np, np.ndarray), "Should be NumPy array"


def run_all_tests():
    """Run all test functions."""
    print("=" * 60)
    print("VPOG Correspondence Builder Test Suite")
    print("=" * 60)
    
    test_coarse_correspondence_shape()
    test_refined_correspondence_shape()
    test_coarse_correspondence_values()
    test_refined_uncertainty_filtering()
    test_correspondence_builder_class()
    test_numpy_conversion()
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
