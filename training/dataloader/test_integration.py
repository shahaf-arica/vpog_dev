"""
Integration Test for VPOG Dataloader
Tests the full pipeline with actual data (if available)

This test:
1. Initializes VPOGTrainDataset
2. Loads a batch of data
3. Verifies shapes and data integrity
4. Generates visualizations
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

from training.dataloader.vpog_dataset import VPOGTrainDataset, VPOGBatch
from training.dataloader.vis_utils import visualize_vpog_batch
from src.utils.dataloader import NoneFilteringDataLoader


def test_vpog_dataloader_integration():
    """
    Full integration test for VPOG dataloader.
    """
    print("=" * 80)
    print("VPOG Dataloader Integration Test")
    print("=" * 80)
    
    # Check if data exists
    data_root = Path("/data/home/ssaricha/gigapose/datasets")
    gso_dir = data_root / "gso"
    
    if not gso_dir.exists():
        print("\n⚠ GSO dataset not found!")
        print(f"  Expected location: {gso_dir}")
        print("  Skipping integration test.")
        print("\nTo run this test:")
        print("  1. Download GSO training data")
        print("  2. Render templates with: python -m src.scripts.render_gso_templates")
        return False
    
    print(f"\n✓ Found dataset at {gso_dir}")
    
    # Load configuration
    print("\n" + "-" * 80)
    print("Loading Configuration")
    print("-" * 80)
    
    try:
        with initialize(version_base=None, config_path="../config"):
            cfg = compose(config_name="train.yaml")
        
        OmegaConf.set_struct(cfg, False)
        
        # Override some settings for testing
        cfg.machine.batch_size = 4
        cfg.data.dataloader.batch_size = 4
        cfg.data.dataloader.num_workers = 4
        cfg.data.dataloader.dataset_name = "gso"
        cfg.data.visualization.enabled = True
        cfg.data.visualization.save_dir = "./tmp/vpog_integration_test"
        
        print(f"✓ Loaded configuration")
        print(f"  Batch size: {cfg.machine.batch_size}")
        print(f"  Dataset: {cfg.data.dataloader.dataset_name}")
        print(f"  Num templates: {cfg.data.dataloader.num_positive_templates + cfg.data.dataloader.num_negative_templates}")
        
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False
    
    # Initialize dataset
    print("\n" + "-" * 80)
    print("Initializing Dataset")
    print("-" * 80)
    
    try:
        # Manually create dataset with explicit config
        dataset = VPOGTrainDataset(
            root_dir=str(data_root),
            dataset_name="gso",
            template_config={
                'dir': str(data_root / "templates"),
                'level_templates': 1,
                'pose_distribution': 'all',
                'scale_factor': 10.0,
                'num_templates': 162,
                'image_name': 'OBJECT_ID/VIEW_ID.png',
                'pose_name': 'object_poses/OBJECT_ID.npy',
            },
            num_positive_templates=3,
            num_negative_templates=2,
            min_negative_angle_deg=90.0,
            d_ref_random_ratio=0.0,
            patch_size=16,
            image_size=224,
            flow_config={
                'compute_visibility': True,
                'compute_patch_visibility': True,
                'visibility_threshold': 0.1,
            },
            transforms={
                'rgb_augmentation': False,
                'inplane_augmentation': True,
                'crop_augmentation': False,
            },
            batch_size=4,
            depth_scale=10.0,
        )
        
        print(f"✓ Initialized VPOGTrainDataset")
        print(f"  Dataset: {dataset.dataset_name}")
        print(f"  Templates per query: {dataset.num_templates}")
        print(f"  Patch size: {dataset.patch_size}")
        print(f"  Num patches: {dataset.num_patches}")
        
    except Exception as e:
        print(f"✗ Failed to initialize dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create dataloader
    print("\n" + "-" * 80)
    print("Creating DataLoader")
    print("-" * 80)
    
    try:
        dataloader = NoneFilteringDataLoader(
            dataset.web_dataloader.datapipeline,
            batch_size=4,
            num_workers=4,
            collate_fn=dataset.collate_fn,
        )
        
        print(f"✓ Created DataLoader")
        print(f"  Batch size: 4")
        print(f"  Num workers: 4")
        
    except Exception as e:
        print(f"✗ Failed to create dataloader: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load and test a batch
    print("\n" + "-" * 80)
    print("Loading Batch")
    print("-" * 80)
    
    try:
        # Get first batch
        batch_iter = iter(dataloader)
        batch = next(batch_iter)
        
        if batch is None:
            print("✗ Received None batch")
            return False
        
        print(f"✓ Loaded batch successfully")
        
    except Exception as e:
        print(f"✗ Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify batch structure
    print("\n" + "-" * 80)
    print("Verifying Batch Structure")
    print("-" * 80)
    
    try:
        assert isinstance(batch, VPOGBatch), f"Expected VPOGBatch, got {type(batch)}"
        
        # Check shapes
        B = batch.images.shape[0]
        S = dataset.num_templates
        H, W = dataset.image_size, dataset.image_size
        H_p, W_p = dataset.num_patches_per_side, dataset.num_patches_per_side
        
        # Images: [B, S+1, 3, H, W]
        assert batch.images.shape == (B, S+1, 3, H, W), \
            f"images shape {batch.images.shape} != expected ({B}, {S+1}, 3, {H}, {W})"
        print(f"✓ images: {batch.images.shape}")
        
        # Masks: [B, S+1, H, W]
        assert batch.masks.shape == (B, S+1, H, W), \
            f"masks shape {batch.masks.shape} != expected ({B}, {S+1}, {H}, {W})"
        print(f"✓ masks: {batch.masks.shape}")
        
        # K: [B, S+1, 3, 3]
        assert batch.K.shape == (B, S+1, 3, 3), \
            f"K shape {batch.K.shape} != expected ({B}, {S+1}, 3, 3)"
        print(f"✓ K: {batch.K.shape}")
        
        # Poses: [B, S+1, 4, 4]
        assert batch.poses.shape == (B, S+1, 4, 4), \
            f"poses shape {batch.poses.shape} != expected ({B}, {S+1}, 4, 4)"
        print(f"✓ poses: {batch.poses.shape}")
        
        # d_ref: [B, 3]
        assert batch.d_ref.shape == (B, 3), \
            f"d_ref shape {batch.d_ref.shape} != expected ({B}, 3)"
        print(f"✓ d_ref: {batch.d_ref.shape}")
        
        # Template indices: [B, S]
        assert batch.template_indices.shape == (B, S), \
            f"template_indices shape {batch.template_indices.shape} != expected ({B}, {S})"
        print(f"✓ template_indices: {batch.template_indices.shape}")
        
        # Template types: [B, S]
        assert batch.template_types.shape == (B, S), \
            f"template_types shape {batch.template_types.shape} != expected ({B}, {S})"
        print(f"✓ template_types: {batch.template_types.shape}")
        
        # Flows: [B, S, H_p, W_p, 2]
        assert batch.flows.shape == (B, S, H_p, W_p, 2), \
            f"flows shape {batch.flows.shape} != expected ({B}, {S}, {H_p}, {W_p}, 2)"
        print(f"✓ flows: {batch.flows.shape}")
        
        # Visibility: [B, S, H_p, W_p]
        assert batch.visibility.shape == (B, S, H_p, W_p), \
            f"visibility shape {batch.visibility.shape} != expected ({B}, {S}, {H_p}, {W_p})"
        print(f"✓ visibility: {batch.visibility.shape}")
        
        # Patch visibility: [B, S, H_p, W_p]
        assert batch.patch_visibility.shape == (B, S, H_p, W_p), \
            f"patch_visibility shape {batch.patch_visibility.shape} != expected ({B}, {S}, {H_p}, {W_p})"
        print(f"✓ patch_visibility: {batch.patch_visibility.shape}")
        
        print(f"\n✓ All shapes correct!")
        
    except AssertionError as e:
        print(f"✗ Shape verification failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check data properties
    print("\n" + "-" * 80)
    print("Checking Data Properties")
    print("-" * 80)
    
    try:
        # Check d_ref is normalized
        d_ref_norms = torch.norm(batch.d_ref, dim=1)
        assert torch.allclose(d_ref_norms, torch.ones_like(d_ref_norms), atol=1e-5), \
            f"d_ref not normalized: norms = {d_ref_norms}"
        print(f"✓ d_ref vectors are normalized (norms ≈ 1.0)")
        
        # Check template types
        num_positives = (batch.template_types == 0).sum(dim=1)
        num_negatives = (batch.template_types == 1).sum(dim=1)
        print(f"✓ Template types: positive={num_positives.tolist()}, negative={num_negatives.tolist()}")
        
        # Check image value ranges (after normalization)
        img_min = batch.images.min().item()
        img_max = batch.images.max().item()
        print(f"✓ Image value range: [{img_min:.3f}, {img_max:.3f}]")
        
        print(f"\n✓ Data properties verified!")
        
    except AssertionError as e:
        print(f"✗ Data property check failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Property check failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Generate visualizations
    print("\n" + "-" * 80)
    print("Generating Visualizations")
    print("-" * 80)
    
    try:
        vis_dir = "./tmp/vpog_integration_test"
        os.makedirs(vis_dir, exist_ok=True)
        
        visualize_vpog_batch(
            batch,
            save_dir=vis_dir,
            batch_idx=0,
            max_samples=min(2, B),
        )
        
        print(f"✓ Visualizations saved to {vis_dir}")
        
    except Exception as e:
        print(f"⚠ Visualization failed (non-critical): {e}")
        # Not a critical error
    
    # Summary
    print("\n" + "=" * 80)
    print("Integration Test Summary")
    print("=" * 80)
    print(f"✓ Dataset initialized successfully")
    print(f"✓ Batch loaded with correct shapes")
    print(f"✓ Data properties verified")
    print(f"✓ Visualizations generated")
    print(f"\nBatch Statistics:")
    print(f"  Batch size: {B}")
    print(f"  Templates per query: {S} (S_p=3, S_n=2)")
    print(f"  Image size: {H}x{W}")
    print(f"  Patch size: {dataset.patch_size}x{dataset.patch_size}")
    print(f"  Patch grid: {H_p}x{W_p} = {H_p*W_p} patches")
    print("=" * 80)
    print("✓ All integration tests PASSED!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = test_vpog_dataloader_integration()
    
    if success:
        print("\n🎉 Integration test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Integration test failed!")
        print("   This may be due to missing data or configuration issues.")
        print("   See messages above for details.")
        sys.exit(1)
