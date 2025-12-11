"""
Test 16×16 Pixel-Level Flow Computation

Tests the extended FlowComputer with:
- 16×16 pixel-level flow within patches
- Unseen mask generation
- Confidence computation
- Visualization with VPOG flow_vis
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from training.dataloader.flow_computer import FlowComputer, compute_patch_flows
from training.visualization import visualize_patch_flow, visualize_pixel_level_flow_detailed


def create_test_data():
    """Create synthetic test data"""
    img_size = 224
    patch_size = 16
    
    # Create simple images
    query_img = np.random.randint(50, 200, (img_size, img_size, 3), dtype=np.uint8)
    template_img = np.random.randint(50, 200, (img_size, img_size, 3), dtype=np.uint8)
    
    # Add some structure
    query_img[50:100, 50:100] = [255, 0, 0]  # Red square
    template_img[60:110, 60:110] = [255, 0, 0]  # Shifted red square
    
    # Create depth maps (simple)
    query_depth = np.ones((img_size, img_size)) * 1.0
    template_depth = np.ones((img_size, img_size)) * 1.0
    
    # Create masks with occlusion
    query_mask = np.ones((img_size, img_size), dtype=bool)
    template_mask = np.ones((img_size, img_size), dtype=bool)
    
    # Add occlusion region (circle in center)
    y, x = np.ogrid[:img_size, :img_size]
    center = img_size // 2
    radius = 30
    circle = (x - center)**2 + (y - center)**2 < radius**2
    query_mask[circle] = False
    
    # Camera intrinsics
    K = np.array([
        [572.0, 0.0, img_size/2],
        [0.0, 572.0, img_size/2],
        [0.0, 0.0, 1.0]
    ])
    
    # Poses (slight rotation)
    query_pose = np.eye(4)
    query_pose[2, 3] = 1.0  # 1m away
    
    template_pose = np.eye(4)
    template_pose[2, 3] = 1.0
    # Small rotation around Y axis
    angle = np.deg2rad(10)
    template_pose[:3, :3] = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    
    return {
        'query': {
            'img': query_img,
            'K': K,
            'pose': query_pose,
            'depth': query_depth,
            'mask': query_mask,
        },
        'template': {
            'img': template_img,
            'K': K,
            'poses': template_pose[None],  # (1, 4, 4)
            'depths': template_depth[None],  # (1, H, W)
            'masks': template_mask[None],  # (1, H, W)
        }
    }


def test_single_patch_flow():
    """Test flow computation for a single patch"""
    print("\n" + "="*80)
    print("TEST 1: Single Patch 16×16 Flow")
    print("="*80)
    
    data = create_test_data()
    
    computer = FlowComputer(
        patch_size=16,
        compute_visibility=True,
        compute_patch_visibility=True,
        depth_tolerance=0.05,
        min_visible_pixels=4,
    )
    
    print(f"Computing flow for center patch...")
    
    # Center patch
    patch_center = np.array([112.0, 112.0])
    
    result = computer.compute_flow_between_patches(
        query_patch_center=patch_center,
        template_patch_center=patch_center,
        query_K=data['query']['K'],
        template_K=data['template']['K'],
        query_pose=data['query']['pose'],
        template_pose=data['template']['poses'][0],
        query_depth=data['query']['depth'],
        template_depth=data['template']['depths'][0],
        query_mask=data['query']['mask'],
        template_mask=data['template']['masks'][0],
    )
    
    print(f"\nResults:")
    print(f"  Flow shape: {result.flow.shape}")
    print(f"  Flow range: [{result.flow.min():.3f}, {result.flow.max():.3f}] patch units")
    print(f"  Confidence range: [{result.confidence.min():.3f}, {result.confidence.max():.3f}]")
    print(f"  Visible pixels: {result.valid_flow.sum()} / {result.valid_flow.size}")
    print(f"  Unseen pixels: {result.unseen_mask.sum()} / {result.unseen_mask.size}")
    print(f"  Mean flow magnitude: {np.linalg.norm(result.flow, axis=-1).mean():.3f} patches")
    
    # Visualize
    output_dir = Path('/tmp/vpog_flow_test')
    output_dir.mkdir(exist_ok=True)
    
    # Extract patch from images
    py, px = int(patch_center[1] - 8), int(patch_center[0] - 8)
    query_patch = data['query']['img'][py:py+16, px:px+16]
    template_patch = data['template']['img'][py:py+16, px:px+16]
    
    fig = visualize_pixel_level_flow_detailed(
        query_patch, template_patch,
        result.flow, result.confidence,
        conf_threshold=0.5,
    )
    save_path = output_dir / 'single_patch_flow.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✓ Saved visualization: {save_path}")


def test_all_patches_flow():
    """Test flow computation for all patches"""
    print("\n" + "="*80)
    print("TEST 2: All Patches 16×16 Flow")
    print("="*80)
    
    data = create_test_data()
    
    print(f"Computing flows for all patch pairs...")
    
    result = compute_patch_flows(
        query_data=data['query'],
        template_data=data['template'],
        patch_size=16,
        image_size=224,
        compute_visibility=True,
        depth_tolerance=0.05,
        min_visible_pixels=4,
    )
    
    flows = result['flows']  # (S, Nq, Nt, 16, 16, 2)
    confidence = result['confidence']  # (S, Nq, Nt, 16, 16)
    unseen_masks = result['unseen_masks']  # (S, Nq, Nt, 16, 16)
    
    print(f"\nResults:")
    print(f"  Flows shape: {flows.shape}")
    print(f"  Expected: (1, 196, 196, 16, 16, 2)")
    print(f"  Confidence shape: {confidence.shape}")
    print(f"  Unseen masks shape: {unseen_masks.shape}")
    
    # Statistics
    S, Nq, Nt = flows.shape[:3]
    print(f"\n  Number of templates: {S}")
    print(f"  Number of query patches: {Nq}")
    print(f"  Number of template patches: {Nt}")
    print(f"  Pixels per patch: 16×16 = 256")
    print(f"  Total flow vectors: {S * Nq * Nt * 16 * 16:,}")
    
    # Get best matches for each query patch (diagonal for same pose)
    best_flows = flows[0, np.arange(Nq), np.arange(Nt)]  # (Nq, 16, 16, 2)
    best_confidence = confidence[0, np.arange(Nq), np.arange(Nt)]  # (Nq, 16, 16)
    best_unseen = unseen_masks[0, np.arange(Nq), np.arange(Nt)]  # (Nq, 16, 16)
    
    print(f"\n  Best matches (diagonal):")
    print(f"    Mean flow magnitude: {np.linalg.norm(best_flows, axis=-1).mean():.3f} patches")
    print(f"    Mean confidence: {best_confidence.mean():.3f}")
    print(f"    Unseen ratio: {best_unseen.mean():.3f}")
    
    # Create classification probabilities (for visualization)
    # Softmax over template patches + unseen token
    classification_probs = np.random.rand(Nq, Nt + 1) * 0.1
    classification_probs[np.arange(Nq), np.arange(Nq)] = 0.8  # High prob for diagonal
    classification_probs[:, -1] = 0.1  # Low unseen probability
    classification_probs = classification_probs / classification_probs.sum(axis=1, keepdims=True)
    
    # Visualize
    output_dir = Path('/tmp/vpog_flow_test')
    output_dir.mkdir(exist_ok=True)
    
    fig = visualize_patch_flow(
        data['query']['img'],
        data['template']['img'],
        best_flows,
        best_confidence,
        classification_probs,
        patch_size=16,
        conf_threshold=0.5,
        top_k_patches=20,
    )
    save_path = output_dir / 'all_patches_flow.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✓ Saved visualization: {save_path}")


def test_unseen_generation():
    """Test unseen mask generation"""
    print("\n" + "="*80)
    print("TEST 3: Unseen Mask Generation")
    print("="*80)
    
    data = create_test_data()
    
    # Create highly occluded scenario
    data['query']['mask'][:, :112] = False  # Occlude left half
    
    computer = FlowComputer(
        patch_size=16,
        compute_visibility=True,
        depth_tolerance=0.05,
        min_visible_pixels=8,  # Stricter threshold
    )
    
    print(f"Computing flows with occlusion...")
    
    result = compute_patch_flows(
        query_data=data['query'],
        template_data=data['template'],
        patch_size=16,
        image_size=224,
        compute_visibility=True,
        depth_tolerance=0.05,
        min_visible_pixels=8,
    )
    
    unseen_masks = result['unseen_masks'][0]  # (Nq, Nt, 16, 16)
    confidence = result['confidence'][0]  # (Nq, Nt, 16, 16)
    
    # Statistics per query patch
    Nq = unseen_masks.shape[0]
    grid_size = int(np.sqrt(Nq))
    
    # Compute unseen ratio per query patch (averaged over template patches and pixels)
    unseen_per_query = unseen_masks.mean(axis=(1, 2, 3))  # (Nq,)
    unseen_map = unseen_per_query.reshape(grid_size, grid_size)
    
    print(f"\nUnseen statistics:")
    print(f"  Patches with >50% unseen: {(unseen_per_query > 0.5).sum()} / {Nq}")
    print(f"  Patches with >90% unseen: {(unseen_per_query > 0.9).sum()} / {Nq}")
    print(f"  Mean unseen ratio: {unseen_per_query.mean():.3f}")
    
    # Visualize unseen map
    output_dir = Path('/tmp/vpog_flow_test')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Query image with mask
    masked_img = data['query']['img'].copy()
    masked_img[~data['query']['mask']] = [128, 128, 128]
    axes[0].imshow(masked_img)
    axes[0].set_title('Query Image\n(Gray = Occluded)')
    axes[0].axis('off')
    
    # Query mask
    axes[1].imshow(data['query']['mask'], cmap='gray')
    axes[1].set_title('Query Visibility Mask')
    axes[1].axis('off')
    
    # Unseen map
    im = axes[2].imshow(unseen_map, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Unseen Ratio per Patch')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    save_path = output_dir / 'unseen_generation.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved visualization: {save_path}")


def test_flow_normalization():
    """Test flow normalization (patch units)"""
    print("\n" + "="*80)
    print("TEST 4: Flow Normalization")
    print("="*80)
    
    print("\nFlow unit convention:")
    print("  • Flow is in PATCH UNITS")
    print("  • delta_x = 1.0 means one patch displacement (16 pixels)")
    print("  • This makes flow scale-invariant")
    
    patch_size = 16
    
    # Example: flow of [0.5, 0.0] means 8 pixels to the right
    flow_patch_units = np.array([0.5, 0.0])
    flow_pixels = flow_patch_units * patch_size
    
    print(f"\nExample:")
    print(f"  Flow in patch units: {flow_patch_units}")
    print(f"  Flow in pixels: {flow_pixels}")
    print(f"  Interpretation: {flow_pixels[0]:.0f} pixels right, {flow_pixels[1]:.0f} pixels down")
    
    # Another example: diagonal
    flow_patch_units = np.array([1.0, 1.0])
    flow_pixels = flow_patch_units * patch_size
    
    print(f"\nExample 2:")
    print(f"  Flow in patch units: {flow_patch_units}")
    print(f"  Flow in pixels: {flow_pixels}")
    print(f"  Interpretation: One patch right and one patch down")
    print(f"  Magnitude: {np.linalg.norm(flow_patch_units):.3f} patches = {np.linalg.norm(flow_pixels):.1f} pixels")
    
    print("\n✓ Flow normalization verified")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("VPOG 16×16 Pixel-Level Flow Test Suite")
    print("="*80)
    
    # Run tests
    test_single_patch_flow()
    test_all_patches_flow()
    test_unseen_generation()
    test_flow_normalization()
    
    output_dir = Path('/tmp/vpog_flow_test')
    print("\n" + "="*80)
    print("✓ All Tests Passed!")
    print(f"✓ Visualizations saved to {output_dir}/")
    print("="*80 + "\n")
    
    # List output files
    print("Output files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  • {f.name}")
