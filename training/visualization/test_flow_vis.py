"""
VPOG Visualization Test with Model Outputs

Tests flow visualization with actual model predictions including:
- Patch-level flow with pixel detail
- Unseen pixel marking
- Confidence visualization
- Correspondence lines
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from vpog.models.vpog_model import VPOGModel
from training.visualization import (
    visualize_patch_flow,
    visualize_pixel_level_flow_detailed,
    visualize_correspondence_grid,
    create_flow_wheel,
)


def create_synthetic_test_data(img_size=224, device='cpu'):
    """Create synthetic test images with known correspondences"""
    
    # Create a checkerboard pattern for query
    query = torch.zeros(1, 3, img_size, img_size, device=device)
    patch_size = 16
    grid_size = img_size // patch_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            if (i + j) % 2 == 0:
                query[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = 0.8
            else:
                query[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = 0.2
    
    # Add some colored patches for easier tracking
    query[:, 0, 3*patch_size:5*patch_size, 3*patch_size:5*patch_size] = 1.0  # Red square
    query[:, 1, 7*patch_size:9*patch_size, 7*patch_size:9*patch_size] = 1.0  # Green square
    query[:, 2, 3*patch_size:5*patch_size, 8*patch_size:10*patch_size] = 1.0  # Blue square
    
    # Create template as shifted/rotated version
    # For simplicity, just shift by 2 patches to the right
    template = torch.zeros_like(query)
    shift_patches = 2
    shift_pixels = shift_patches * patch_size
    
    # Circular shift
    template[:, :, :, shift_pixels:] = query[:, :, :, :-shift_pixels]
    template[:, :, :, :shift_pixels] = query[:, :, :, -shift_pixels:]
    
    # Add noise
    query = query + torch.randn_like(query) * 0.1
    template = template + torch.randn_like(template) * 0.1
    
    query = torch.clamp(query, 0, 1)
    template = torch.clamp(template, 0, 1)
    
    return query, template, shift_patches


def test_flow_visualization_with_model():
    """Test flow visualization with actual VPOG model predictions"""
    
    print("\n" + "="*80)
    print("TEST: Flow Visualization with VPOG Model")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Model configuration (lightweight for testing)
    img_size = 224
    patch_size = 16
    
    encoder_config = {
        'model_name': 'CroCo_V2_ViTBase',
        'pretrained_path': None,
        'freeze_encoder': False,
        'num_query_added_tokens': 0,
        'num_template_added_tokens': 1,
    }
    
    aa_config = {
        'depth': 2,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'window_size': 7,
        'use_s2rope': False,  # Disable for simplicity
    }
    
    classification_config = {
        'use_mlp': True,
        'mlp_hidden_dim': 256,
        'temperature': 1.0,
    }
    
    flow_config = {
        'num_layers': 2,
        'hidden_dim': 256,
    }
    
    print("\n1. Creating VPOG model...")
    model = VPOGModel(
        encoder_config=encoder_config,
        aa_config=aa_config,
        classification_config=classification_config,
        flow_config=flow_config,
        img_size=img_size,
        patch_size=patch_size,
    ).to(device)
    model.eval()
    print("   ✓ Model created")
    
    # Create synthetic test data
    print("\n2. Creating synthetic test images...")
    query_img, template_img, true_shift = create_synthetic_test_data(img_size, device)
    
    # Expand template to have batch dimension
    template_imgs = template_img.unsqueeze(1)  # [1, 1, 3, H, W]
    
    # Dummy poses
    query_poses = torch.eye(4, device=device).unsqueeze(0)
    template_poses = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)
    ref_dirs = torch.tensor([[0, 0, 1]], device=device, dtype=torch.float32)
    
    print(f"   Query shape: {query_img.shape}")
    print(f"   Template shape: {template_imgs.shape}")
    print(f"   True shift: {true_shift} patches to the right")
    
    # Forward pass
    print("\n3. Running model forward pass...")
    with torch.no_grad():
        outputs = model(
            query_images=query_img,
            template_images=template_imgs,
            query_poses=query_poses,
            template_poses=template_poses,
            ref_dirs=ref_dirs,
            return_all=True,
        )
    print("   ✓ Forward pass complete")
    
    # Extract outputs
    classification_logits = outputs['classification_logits']  # [1, 1, Nq, Nt+1]
    flow = outputs['flow']  # [1, 1, Nq, Nt, 16, 16, 2]
    flow_confidence = outputs['flow_confidence']  # [1, 1, Nq, Nt, 16, 16, 1]
    
    print(f"\n4. Output shapes:")
    print(f"   Classification logits: {classification_logits.shape}")
    print(f"   Flow: {flow.shape}")
    print(f"   Flow confidence: {flow_confidence.shape}")
    
    # Convert to numpy for visualization
    query_np = (query_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    template_np = (template_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # Get classification probabilities
    classification_probs = F.softmax(classification_logits[0, 0], dim=-1).cpu().numpy()  # [Nq, Nt+1]
    
    # For single template, aggregate flow across template patches
    # Take the flow from query patch to its best matching template patch
    Nq = classification_logits.shape[2]
    Nt = flow.shape[3]
    
    # Get best template patch for each query patch (excluding unseen)
    best_template_idx = classification_probs[:, :-1].argmax(axis=1)  # [Nq]
    
    # Extract flow for best matches
    flow_best = torch.zeros(Nq, 16, 16, 2, device=device)
    confidence_best = torch.zeros(Nq, 16, 16, device=device)
    
    for q_idx in range(Nq):
        t_idx = best_template_idx[q_idx]
        flow_best[q_idx] = flow[0, 0, q_idx, t_idx]
        confidence_best[q_idx] = flow_confidence[0, 0, q_idx, t_idx, :, :, 0]
    
    flow_np = flow_best.cpu().numpy()
    confidence_np = confidence_best.cpu().numpy()
    
    print(f"\n5. Flow statistics:")
    print(f"   Mean flow (x): {flow_np[..., 0].mean():.4f} patches")
    print(f"   Mean flow (y): {flow_np[..., 1].mean():.4f} patches")
    print(f"   Mean confidence: {confidence_np.mean():.4f}")
    print(f"   Mean unseen probability: {classification_probs[:, -1].mean():.4f}")
    
    # Create visualizations
    output_dir = Path('/tmp/vpog_vis')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n6. Creating visualizations...")
    
    # Main patch flow visualization
    print("   - Patch-level flow with pixel detail...")
    fig1 = visualize_patch_flow(
        query_np, template_np, flow_np, confidence_np,
        classification_probs, patch_size=patch_size,
        conf_threshold=0.3, top_k_patches=20
    )
    fig1.savefig(output_dir / 'patch_flow_overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"     ✓ Saved to {output_dir}/patch_flow_overview.png")
    
    # Detailed pixel-level visualization for a specific patch
    print("   - Detailed pixel-level flow for top patch...")
    grid_w = img_size // patch_size
    
    # Find patch with highest confidence (not unseen)
    patch_scores = (1 - classification_probs[:, -1]) * confidence_np.mean(axis=(1, 2))
    best_patch_idx = patch_scores.argmax()
    py, px = best_patch_idx // grid_w, best_patch_idx % grid_w
    
    query_patch = query_np[py*patch_size:(py+1)*patch_size, px*patch_size:(px+1)*patch_size]
    # For template patch, we need to use the matched template location
    template_patch_idx = best_template_idx[best_patch_idx]
    tpy, tpx = template_patch_idx // grid_w, template_patch_idx % grid_w
    template_patch = template_np[tpy*patch_size:(tpy+1)*patch_size, tpx*patch_size:(tpx+1)*patch_size]
    
    fig2 = visualize_pixel_level_flow_detailed(
        query_patch, template_patch,
        flow_np[best_patch_idx], confidence_np[best_patch_idx],
        conf_threshold=0.3
    )
    fig2.savefig(output_dir / 'pixel_flow_detail.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"     ✓ Saved to {output_dir}/pixel_flow_detail.png")
    
    # Correspondence visualization
    print("   - Correspondence lines...")
    fig3 = visualize_correspondence_grid(
        query_np, template_np, flow_np, classification_probs, confidence_np,
        patch_size=patch_size, conf_threshold=0.3, stride=4
    )
    fig3.savefig(output_dir / 'correspondences.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"     ✓ Saved to {output_dir}/correspondences.png")
    
    # Flow color wheel
    print("   - Flow color wheel legend...")
    fig4, ax = plt.subplots(1, 1, figsize=(6, 6))
    wheel = create_flow_wheel(512)
    ax.imshow(wheel)
    ax.set_title("Flow Color Wheel\n(Direction Encoding)", fontsize=14, weight='bold')
    ax.axis('off')
    
    # Add direction labels
    ax.text(256, 40, 'Right →', ha='center', fontsize=12, color='white', 
            weight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax.text(256, 472, '← Left', ha='center', fontsize=12, color='white',
            weight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax.text(40, 256, '↑\nUp', ha='center', va='center', fontsize=12, color='white',
            weight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax.text(472, 256, '↓\nDown', ha='center', va='center', fontsize=12, color='white',
            weight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    fig4.savefig(output_dir / 'flow_wheel.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"     ✓ Saved to {output_dir}/flow_wheel.png")
    
    print(f"\n✓ All visualizations saved to {output_dir}/")
    print("\nVisualization files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  • {f.name}")
    
    return output_dir


def test_unseen_marking():
    """Test visualization of unseen pixels specifically"""
    
    print("\n" + "="*80)
    print("TEST: Unseen Pixel Marking")
    print("="*80)
    
    # Create synthetic data with clear unseen regions
    img_size = 224
    patch_size = 16
    grid_size = img_size // patch_size
    Nq = grid_size * grid_size
    
    # Create images
    query_np = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    template_np = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    
    # Create flow with some structure
    flow = np.random.randn(Nq, 16, 16, 2) * 0.2
    
    # Create confidence with clear patterns
    confidence = np.random.rand(Nq, 16, 16) * 0.5 + 0.3  # Base confidence
    
    # Mark some patches as low confidence (unseen)
    # Create an "X" pattern of unseen patches
    for i in range(grid_size):
        for j in range(grid_size):
            patch_idx = i * grid_size + j
            if i == j or i + j == grid_size - 1:  # Diagonal
                confidence[patch_idx] *= 0.3  # Low confidence
    
    # Within patches, create some pixel-level unseen regions
    for patch_idx in range(Nq):
        # Create circular unseen region in center
        y, x = np.mgrid[-8:8, -8:8]
        circle = (x**2 + y**2) < 25
        confidence[patch_idx, circle] *= 0.2
    
    # Create classification probs with high unseen probability for diagonal
    classification_probs = np.random.rand(Nq, 197) * 0.01
    for i in range(grid_size):
        for j in range(grid_size):
            patch_idx = i * grid_size + j
            if i == j or i + j == grid_size - 1:
                classification_probs[patch_idx, -1] = 0.9  # High unseen
            else:
                classification_probs[patch_idx, 50] = 0.8  # Some match
    
    classification_probs = classification_probs / classification_probs.sum(axis=1, keepdims=True)
    
    print(f"Created test data:")
    print(f"  • {Nq} patches ({grid_size}×{grid_size})")
    print(f"  • Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    print(f"  • Mean unseen prob: {classification_probs[:, -1].mean():.3f}")
    print(f"  • High unseen patches: {(classification_probs[:, -1] > 0.5).sum()}")
    
    # Visualize
    output_dir = Path('/tmp/vpog_vis')
    output_dir.mkdir(exist_ok=True)
    
    print("\nCreating unseen marking visualization...")
    fig = visualize_patch_flow(
        query_np, template_np, flow, confidence,
        classification_probs, patch_size=patch_size,
        conf_threshold=0.5, top_k_patches=10
    )
    fig.savefig(output_dir / 'unseen_marking_test.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved to {output_dir}/unseen_marking_test.png")
    print("  Gray pixels/patches indicate unseen regions")
    
    return output_dir


if __name__ == '__main__':
    print("\n" + "="*80)
    print("VPOG Flow Visualization Test Suite")
    print("="*80)
    
    # Test with model
    output_dir = test_flow_visualization_with_model()
    
    # Test unseen marking
    test_unseen_marking()
    
    print("\n" + "="*80)
    print("All Tests Completed!")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")
