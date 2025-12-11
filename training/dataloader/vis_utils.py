"""
Visualization Utilities for VPOG
Visualize query images, templates, flows, and patch correspondences for debugging
"""

from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Dict
import cv2
from pathlib import Path

from training.dataloader.vpog_dataset import VPOGBatch


def denormalize_image(img: torch.Tensor) -> np.ndarray:
    """
    Denormalize image from ImageNet normalization.
    
    Args:
        img: [C, H, W] normalized tensor
    
    Returns:
        [H, W, C] numpy array in [0, 255]
    """
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    
    img = img.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    return img


def visualize_vpog_sample(
    batch: VPOGBatch,
    sample_idx: int = 0,
    save_path: Optional[str] = None,
    max_templates: int = 6,
) -> plt.Figure:
    """
    Visualize a single sample from VPOG batch.
    
    Shows:
    - Query image
    - S templates (positive and negative)
    - d_ref visualization
    - Template selection info
    
    Args:
        batch: VPOGBatch
        sample_idx: Which sample in batch to visualize
        save_path: Path to save figure (optional)
        max_templates: Maximum number of templates to show
    
    Returns:
        matplotlib Figure
    """
    # Extract data for this sample
    images = batch.images[sample_idx]  # [S+1, 3, H, W]
    masks = batch.masks[sample_idx]  # [S+1, H, W]
    d_ref = batch.d_ref[sample_idx]  # [3]
    template_indices = batch.template_indices[sample_idx]  # [S]
    template_types = batch.template_types[sample_idx]  # [S]
    
    num_templates = len(template_indices)
    num_to_show = min(num_templates, max_templates)
    
    # Create figure
    num_cols = min(4, num_to_show + 1)
    num_rows = (num_to_show + 1 + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Hide extra axes
    for ax in axes[num_to_show + 1:]:
        ax.axis('off')
    
    # Plot query image
    query_img = denormalize_image(images[0])
    query_mask = masks[0].cpu().numpy()
    
    axes[0].imshow(query_img)
    axes[0].contour(query_mask, colors='green', linewidths=2, levels=[0.5])
    axes[0].set_title('Query Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Add d_ref info as text
    d_ref_str = f'd_ref: [{d_ref[0]:.3f}, {d_ref[1]:.3f}, {d_ref[2]:.3f}]'
    axes[0].text(0.5, -0.05, d_ref_str, ha='center', transform=axes[0].transAxes,
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot templates
    for i in range(num_to_show):
        template_img = denormalize_image(images[i + 1])
        template_mask = masks[i + 1].cpu().numpy()
        
        axes[i + 1].imshow(template_img)
        axes[i + 1].contour(template_mask, colors='blue', linewidths=2, levels=[0.5])
        
        # Title with type info
        template_type = 'Positive' if template_types[i] == 0 else 'Negative'
        template_idx = template_indices[i].item()
        color = 'green' if template_types[i] == 0 else 'red'
        
        axes[i + 1].set_title(f'Template {i+1} ({template_type})\nIdx: {template_idx}',
                            fontsize=10, color=color, fontweight='bold')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


def visualize_flow_field(
    batch: VPOGBatch,
    sample_idx: int = 0,
    template_idx: int = 0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize flow field for a specific template->query pair.
    
    Args:
        batch: VPOGBatch
        sample_idx: Which sample in batch
        template_idx: Which template to visualize
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    # Extract flow data
    flows = batch.flows[sample_idx, template_idx]  # [H_p, W_p, 2]
    visibility = batch.visibility[sample_idx, template_idx]  # [H_p, W_p]
    patch_visibility = batch.patch_visibility[sample_idx, template_idx]  # [H_p, W_p]
    
    flows = flows.cpu().numpy()
    visibility = visibility.cpu().numpy()
    patch_visibility = patch_visibility.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Flow magnitude
    flow_magnitude = np.linalg.norm(flows, axis=-1)
    im0 = axes[0].imshow(flow_magnitude, cmap='jet')
    axes[0].set_title('Flow Magnitude (pixels)')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Flow X component
    im1 = axes[1].imshow(flows[..., 0], cmap='RdBu_r', vmin=-8, vmax=8)
    axes[1].set_title('Flow X')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Flow Y component
    im2 = axes[2].imshow(flows[..., 1], cmap='RdBu_r', vmin=-8, vmax=8)
    axes[2].set_title('Flow Y')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    # Combined visibility mask
    valid_mask = visibility & patch_visibility
    axes[3].imshow(valid_mask, cmap='gray')
    axes[3].set_title(f'Valid Flow Mask\n({valid_mask.sum()}/{valid_mask.size} valid)')
    axes[3].axis('off')
    
    template_type = 'Positive' if batch.template_types[sample_idx, template_idx] == 0 else 'Negative'
    fig.suptitle(f'Flow Field: Template {template_idx} ({template_type}) -> Query',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved flow visualization to {save_path}")
    
    return fig


def visualize_patch_grid(
    batch: VPOGBatch,
    sample_idx: int = 0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize patch grid overlay on query and templates.
    
    Args:
        batch: VPOGBatch
        sample_idx: Which sample in batch
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    images = batch.images[sample_idx]  # [S+1, 3, H, W]
    
    # Get image dimensions
    _, H, W = images.shape[1:]
    
    # Compute patch size from batch info
    # We assume square patches and that H, W are divisible by patch size
    # This should match the dataset configuration
    flows = batch.flows[sample_idx]  # [S, H_p, W_p, 2]
    H_p, W_p = flows.shape[1:3]
    patch_size_h = H // H_p
    patch_size_w = W // W_p
    
    # Create figure
    num_to_show = min(4, images.shape[0])
    fig, axes = plt.subplots(1, num_to_show, figsize=(num_to_show * 4, 4))
    if num_to_show == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        if i >= images.shape[0]:
            ax.axis('off')
            continue
        
        # Denormalize and show image
        img = denormalize_image(images[i])
        ax.imshow(img)
        
        # Draw patch grid
        for py in range(H_p):
            for px in range(W_p):
                rect = patches.Rectangle(
                    (px * patch_size_w, py * patch_size_h),
                    patch_size_w, patch_size_h,
                    linewidth=1, edgecolor='yellow', facecolor='none', alpha=0.5
                )
                ax.add_patch(rect)
        
        # Title
        if i == 0:
            title = 'Query with Patch Grid'
        else:
            template_type = 'Pos' if batch.template_types[sample_idx, i-1] == 0 else 'Neg'
            title = f'Template {i} ({template_type})'
        ax.set_title(title)
        ax.axis('off')
    
    fig.suptitle(f'Patch Grid Overlay (Patch size: {patch_size_h}x{patch_size_w}, Grid: {H_p}x{W_p})',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved patch grid to {save_path}")
    
    return fig


def visualize_vpog_batch(
    batch: VPOGBatch,
    save_dir: str,
    batch_idx: int = 0,
    max_samples: int = 4,
) -> None:
    """
    Visualize multiple samples from a VPOG batch and save to directory.
    
    Creates:
    - Sample visualizations (query + templates)
    - Flow field visualizations
    - Patch grid overlays
    
    Args:
        batch: VPOGBatch to visualize
        save_dir: Directory to save visualizations
        batch_idx: Batch index (for naming)
        max_samples: Maximum samples to visualize
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    batch_size = batch.images.shape[0]
    num_samples = min(batch_size, max_samples)
    
    print(f"\nVisualizing batch {batch_idx} ({num_samples} samples)...")
    
    for sample_idx in range(num_samples):
        # Sample visualization
        fig = visualize_vpog_sample(
            batch, sample_idx,
            save_path=str(save_dir / f"batch{batch_idx:04d}_sample{sample_idx}_overview.png")
        )
        plt.close(fig)
        
        # Patch grid
        fig = visualize_patch_grid(
            batch, sample_idx,
            save_path=str(save_dir / f"batch{batch_idx:04d}_sample{sample_idx}_patches.png")
        )
        plt.close(fig)
        
        # Flow fields for first 2 templates
        num_templates = batch.flows.shape[1]
        for template_idx in range(min(2, num_templates)):
            fig = visualize_flow_field(
                batch, sample_idx, template_idx,
                save_path=str(save_dir / f"batch{batch_idx:04d}_sample{sample_idx}_flow_t{template_idx}.png")
            )
            plt.close(fig)
    
    print(f"✓ Saved visualizations to {save_dir}")


def save_visualization(
    batch: VPOGBatch,
    save_dir: str,
    batch_idx: int,
    enabled: bool = True,
    **kwargs,
) -> None:
    """
    Wrapper function for saving visualizations with enable flag.
    
    Args:
        batch: VPOGBatch
        save_dir: Save directory
        batch_idx: Batch index
        enabled: Whether to actually save (can be controlled by config)
        **kwargs: Additional arguments for visualize_vpog_batch
    """
    if not enabled:
        return
    
    visualize_vpog_batch(batch, save_dir, batch_idx, **kwargs)


if __name__ == "__main__":
    """
    REAL TEST with GSO data
    Loads actual query and templates from dataset and visualizes them
    """
    import sys
    from pathlib import Path
    import torch
    import pandas as pd
    from bop_toolkit_lib import inout
    from PIL import Image
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.megapose.utils.tensor_collection import PandasTensorCollection
    from src.custom_megapose.template_dataset import TemplateDataset
    
    print("=" * 80)
    print("REAL TEST: Visualization Utilities with GSO Data")
    print("=" * 80)
    
    # Setup paths
    dataset_dir = project_root / "datasets" / "gso"
    templates_dir = project_root / "datasets" / "templates" / "gso"
    save_dir = project_root / "tmp" / "vis_utils_test"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if not templates_dir.exists():
        print(f"\n✗ Templates not found at {templates_dir}")
        print("Please run: python -m src.scripts.render_gso_templates")
        sys.exit(1)
    
    print(f"✓ Found templates at {templates_dir}")
    
    # Load model info
    model_infos = inout.load_json(dataset_dir / "models_info.json")
    obj_ids = sorted([int(k) for k in model_infos.keys()])[:3]  # First 3 objects
    
    print(f"✓ Loading data for objects: {[f'{obj:06d}' for obj in obj_ids]}")
    
    # Initialize template dataset
    template_config = {
        'model_name': 'gso',
        'dataset_dir': str(templates_dir),
        'image_size': (224, 224),
        'level_templates': 1,
        'pose_distribution': 'all',
        'normalize_before_predict': True,
    }
    
    template_dataset = TemplateDataset(template_config)
    print(f"✓ Loaded template dataset with {len(template_dataset)} templates")
    
    # Build a real VPOG batch
    print(f"\n✓ Building real VPOG batch...")
    batch_size = len(obj_ids)
    num_templates = 4
    H, W = 224, 224
    patch_size = 16
    H_p, W_p = H // patch_size, W // patch_size
    
    # Load real images
    images_list = []
    masks_list = []
    template_indices_list = []
    template_types_list = []
    d_ref_list = []
    
    for obj_id in obj_ids:
        # Load query (use first template view as query with small perturbation)
        query_idx = 0
        query_img_path = templates_dir / f"{obj_id:06d}/{query_idx:06d}.png"
        query_img = np.array(Image.open(query_img_path).resize((W, H)))
        query_rgb = query_img[:, :, :3].astype(np.float32) / 255.0
        query_mask = (query_img[:, :, 3] > 0).astype(np.float32) if query_img.shape[2] == 4 else np.ones((H, W))
        
        # Load templates (positive: nearby views, negative: far views)
        template_imgs = []
        template_masks = []
        template_idxs = [5, 10, 15, 80]  # Mix of positive and negative
        template_types = [0, 0, 0, 1]  # First 3 positive, last negative
        
        for tidx in template_idxs:
            t_img_path = templates_dir / f"{obj_id:06d}/{tidx:06d}.png"
            t_img = np.array(Image.open(t_img_path).resize((W, H)))
            t_rgb = t_img[:, :, :3].astype(np.float32) / 255.0
            t_mask = (t_img[:, :, 3] > 0).astype(np.float32) if t_img.shape[2] == 4 else np.ones((H, W))
            
            template_imgs.append(t_rgb)
            template_masks.append(t_mask)
        
        # Stack query + templates
        sample_images = np.stack([query_rgb] + template_imgs, axis=0)  # (S+1, H, W, 3)
        sample_masks = np.stack([query_mask] + template_masks, axis=0)  # (S+1, H, W)
        
        images_list.append(sample_images)
        masks_list.append(sample_masks)
        template_indices_list.append(template_idxs)
        template_types_list.append(template_types)
        
        # d_ref: use z-axis of query pose (simplified)
        d_ref_list.append([0.0, 0.0, 1.0])
    
    # Convert to tensors
    images = torch.from_numpy(np.stack(images_list, axis=0)).permute(0, 1, 4, 2, 3)  # (B, S+1, 3, H, W)
    masks = torch.from_numpy(np.stack(masks_list, axis=0))  # (B, S+1, H, W)
    K = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_templates + 1, 1, 1)
    poses = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_templates + 1, 1, 1)
    d_ref = torch.tensor(d_ref_list, dtype=torch.float32)
    template_indices = torch.tensor(template_indices_list, dtype=torch.long)
    template_types = torch.tensor(template_types_list, dtype=torch.long)
    
    # Create synthetic flows (for visualization test)
    flows = torch.randn(batch_size, num_templates, H_p, W_p, 2) * 3.0
    visibility = torch.rand(batch_size, num_templates, H_p, W_p) > 0.3
    patch_visibility = torch.rand(batch_size, num_templates, H_p, W_p) > 0.4
    
    infos = PandasTensorCollection(
        infos=pd.DataFrame({
            'label': [f'{obj:06d}' for obj in obj_ids],
            'scene_id': [0] * batch_size,
            'view_id': list(range(batch_size)),
        })
    )
    
    batch = VPOGBatch(
        images=images,
        masks=masks,
        K=K,
        poses=poses,
        d_ref=d_ref,
        template_indices=template_indices,
        template_types=template_types,
        flows=flows,
        visibility=visibility,
        patch_visibility=patch_visibility,
        infos=infos,
    )
    
    print(f"✓ Created batch:")
    print(f"  Images: {batch.images.shape}")
    print(f"  Flows: {batch.flows.shape}")
    print(f"  d_ref: {batch.d_ref.shape}")
    
    # Test all visualization functions
    print(f"\n✓ Testing visualizations...")
    
    for sample_idx in range(batch_size):
        obj_id = obj_ids[sample_idx]
        
        # Sample overview
        fig = visualize_vpog_sample(
            batch, sample_idx,
            save_path=str(save_dir / f"obj{obj_id:06d}_overview.png")
        )
        plt.close(fig)
        
        # Patch grid
        fig = visualize_patch_grid(
            batch, sample_idx,
            save_path=str(save_dir / f"obj{obj_id:06d}_patches.png")
        )
        plt.close(fig)
        
        # Flow fields for first 2 templates
        for tidx in range(min(2, num_templates)):
            fig = visualize_flow_field(
                batch, sample_idx, tidx,
                save_path=str(save_dir / f"obj{obj_id:06d}_flow_t{tidx}.png")
            )
            plt.close(fig)
    
    # Batch visualization
    print(f"\n✓ Creating batch visualization...")
    visualize_vpog_batch(batch, str(save_dir / "batch"), batch_idx=0, max_samples=batch_size)
    
    print("\n" + "=" * 80)
    print(f"✓ All visualization tests passed!")
    print(f"✓ Real images loaded from {templates_dir}")
    print(f"✓ Visualizations saved to {save_dir}")
    print("=" * 80)
