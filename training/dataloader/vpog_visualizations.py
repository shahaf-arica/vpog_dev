"""
VPOG Dataset Visualization Utilities

Standalone visualization functions for VPOG dataset batches and correspondences.
All visualization logic is centralized here, separate from data loading logic.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize image tensor using ImageNet statistics.
    
    Args:
        tensor: [3, H, W] normalized image tensor
        
    Returns:
        [H, W, 3] denormalized image in [0, 1] range
    """
    img = tensor.cpu().numpy()
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    img = img * std + mean
    img = np.transpose(img, (1, 2, 0))
    return np.clip(img, 0, 1)


def compute_pose_angle(pose1: torch.Tensor, pose2: torch.Tensor) -> float:
    """
    Compute SO(3) angular distance between two poses.
    
    Args:
        pose1: [4, 4] pose matrix
        pose2: [4, 4] pose matrix
        
    Returns:
        Angular distance in degrees
    """
    R1 = pose1[:3, :3].cpu().numpy()
    R2 = pose2[:3, :3].cpu().numpy()
    R_rel = R1.T @ R2
    trace = np.trace(R_rel)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle_deg = np.rad2deg(np.arccos(cos_angle))
    return angle_deg


def visualize_batch_sample(
    batch: Any,
    sample_idx: int,
    dataset_name: str,
    save_dir: Path,
    seed: Optional[int] = None,
    max_templates: Optional[int] = None,
) -> Path:
    """
    Visualize a single sample from a VPOG batch.
    
    Creates a single-row figure showing:
    - Full RGB image with bbox
    - Centered crop (before resizing)
    - Final query image (224x224)
    - Template images with pose angles
    
    Args:
        batch: VPOGBatch object
        sample_idx: Index of sample in batch (0 to B-1)
        dataset_name: Name of dataset for filename
        save_dir: Directory to save visualization
        seed: Optional seed value for filename
        max_templates: Maximum number of templates to show (None = show all)
        
    Returns:
        Path to saved visualization
    """
    b = sample_idx
    
    # Determine number of templates to show
    num_templates = batch.images.shape[1] - 1  # Total templates (S)
    if max_templates is not None:
        num_templates = min(num_templates, max_templates)
    
    # Create figure with dynamic width based on number of templates
    # 3 for query images + num_templates for templates
    num_cols = 3 + num_templates
    fig, axes = plt.subplots(1, num_cols, figsize=(3 * num_cols, 4))
    
    # Ensure axes is always iterable
    if num_cols == 1:
        axes = [axes]
    
    col_idx = 0
    
    # Extract sample identifier from batch.infos if available
    sample_id_parts = []
    if hasattr(batch.infos, 'scene_id'):
        sample_id_parts.append(f"scene{batch.infos.scene_id[b]}")
    if hasattr(batch.infos, 'view_id'):
        sample_id_parts.append(f"view{batch.infos.view_id[b]}")
    if hasattr(batch.infos, 'label'):
        sample_id_parts.append(f"obj{batch.infos.label[b]}")
    
    sample_id = "_".join(sample_id_parts) if sample_id_parts else f"sample{b}"
    
    # Column 0: Original full RGB with bbox
    if batch.full_rgb is not None:
        full_img = batch.full_rgb[b].permute(1, 2, 0).cpu().numpy()
        axes[col_idx].imshow(np.clip(full_img, 0, 1))
        axes[col_idx].set_title(f'Sample {b}:\nFull RGB', fontsize=9, fontweight='bold')
        
        # Draw bbox
        if batch.bboxes is not None:
            bbox = batch.bboxes[b].cpu().numpy()
            rect = Rectangle(
                (bbox[0], bbox[1]), 
                bbox[2] - bbox[0], 
                bbox[3] - bbox[1],
                linewidth=2, 
                edgecolor='red', 
                facecolor='none'
            )
            axes[col_idx].add_patch(rect)
        axes[col_idx].axis('off')
    col_idx += 1
    
    # Column 1: Centered query (object-centered, before cropping)
    if batch.full_rgb is not None and batch.bboxes is not None:
        bbox = batch.bboxes[b].cpu().numpy().astype(int)
        centered_img = full_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        axes[col_idx].imshow(np.clip(centered_img, 0, 1))
        axes[col_idx].set_title(f'Centered\n(pre-crop)', fontsize=9, fontweight='bold')
        axes[col_idx].axis('off')
    col_idx += 1
    
    # Column 2: Cropped & normalized query (final input)
    query_img = denormalize_image(batch.images[b, 0])
    axes[col_idx].imshow(query_img)
    axes[col_idx].set_title(f'Query\n(224x224)', fontsize=9, fontweight='bold', color='blue')
    axes[col_idx].axis('off')
    col_idx += 1
    
    # Remaining columns: Templates with angles
    query_pose = batch.poses[b, 0]
    # First positive template is t* (nearest OOP)
    t_star_pose = batch.poses[b, 1]  # T0 is always t*
    
    for s in range(num_templates):
        template_pose = batch.poses[b, s + 1]
        angle_to_query = compute_pose_angle(query_pose, template_pose)
        angle_to_tstar = compute_pose_angle(t_star_pose, template_pose)
        template_idx = batch.template_indices[b, s].item()
        
        axes[col_idx].imshow(denormalize_image(batch.images[b, s + 1]))
        is_pos = batch.template_types[b, s].item() == 0
        color = 'green' if is_pos else 'red'
        # Show angle to t* for POS templates, angle to query for NEG templates
        angle_display = angle_to_tstar if is_pos else angle_to_query
        axes[col_idx].set_title(
            f'T{s} ({"POS" if is_pos else "NEG"}) idx={template_idx}\n'
            f'to {"t*" if is_pos else "Q"}:{angle_display:.1f}°', 
            fontsize=8, 
            color=color, 
            fontweight='bold' if is_pos else 'normal'
        )
        axes[col_idx].axis('off')
        col_idx += 1
    
    plt.suptitle(
        f'{dataset_name}: Sample {b} ({sample_id})',
        fontsize=12, 
        fontweight='bold'
    )
    plt.tight_layout()
    
    # Save with dataset name, batch index, and sample ID
    seed_str = f"_seed{seed}" if seed is not None else ""
    output_filename = f'sample_{dataset_name}_b{b}_{sample_id}{seed_str}.png'
    save_path = save_dir / output_filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def visualize_batch_all_samples(
    batch: Any,
    dataset_name: str,
    save_dir: Path,
    seed: Optional[int] = None,
    max_templates: Optional[int] = None,
) -> list[Path]:
    """
    Visualize all samples in a batch, creating one figure per sample.
    
    Args:
        batch: VPOGBatch object
        dataset_name: Name of dataset for filename
        save_dir: Directory to save visualizations
        seed: Optional seed value for filename
        max_templates: Maximum number of templates to show per sample
        
    Returns:
        List of paths to saved visualizations
    """
    # Create subfolder for per-sample visualizations
    per_sample_dir = save_dir / "per_sample_visualizations"
    per_sample_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    batch_size = batch.images.shape[0]
    
    for b in range(batch_size):
        save_path = visualize_batch_sample(
            batch=batch,
            sample_idx=b,
            dataset_name=dataset_name,
            save_dir=per_sample_dir,
            seed=seed,
            max_templates=max_templates,
        )
        saved_paths.append(save_path)
        logger.info(f"  ✓ Saved: {save_path.name}")
    
    return saved_paths


def visualize_dense_patch_flow(
    query_rgb: torch.Tensor,
    template_rgb: torch.Tensor,
    q_mask: torch.Tensor,
    t_depth: torch.Tensor,
    flow_grid: torch.Tensor,
    weight_grid: torch.Tensor,
    patch_has_object: torch.Tensor,
    patch_is_visible: torch.Tensor,
    patch_buddy_i: torch.Tensor,
    patch_buddy_j: torch.Tensor,
    batch_idx: int,
    template_idx: int,
    obj_label: int,
    patch_size: int,
    num_patches_per_side: int,
    vis_dir: Path,
    num_patches_to_show: int = 8,
) -> Path:
    """
    Visualize dense template→query flow for selected patches.
    
    Shows correspondence between template patches and query patch centers,
    with detailed flow information for a subset of patches.
    
    Args:
        query_rgb: [3, H, W] query image
        template_rgb: [3, H, W] template image
        q_mask: [H, W] query mask
        t_depth: [H, W] template depth
        flow_grid: [H_p, W_p, ps, ps, 2] dense flow
        weight_grid: [H_p, W_p, ps, ps] dense visibility
        patch_has_object: [H_p, W_p] bool mask
        patch_is_visible: [H_p, W_p] bool mask
        patch_buddy_i: [H_p, W_p] buddy patch i indices
        patch_buddy_j: [H_p, W_p] buddy patch j indices
        batch_idx: Batch index for filename
        template_idx: Template index for filename
        obj_label: Object label for filename
        patch_size: Size of each patch (e.g., 16)
        num_patches_per_side: Number of patches per side (e.g., 14)
        vis_dir: Directory to save visualization
        num_patches_to_show: Number of random patches to visualize in detail
        
    Returns:
        Path to saved visualization
    """
    ps = patch_size
    H_p = num_patches_per_side
    
    # Convert to numpy
    q_rgb_np = query_rgb.cpu().permute(1, 2, 0).numpy()
    t_rgb_np = template_rgb.cpu().permute(1, 2, 0).numpy()
    
    # Select random patches to visualize
    visible_patch_indices = torch.nonzero(patch_is_visible, as_tuple=False)
    if len(visible_patch_indices) > 0:
        num_patches = min(num_patches_to_show, len(visible_patch_indices))
        perm = torch.randperm(len(visible_patch_indices))[:num_patches]
        selected_patches = visible_patch_indices[perm]
    else:
        num_patches = 0
        selected_patches = []
    
    # Create figure
    fig = plt.figure(figsize=(20, 5 * (num_patches + 1)))
    
    # Row 0: Overview of query and template
    ax_q = plt.subplot2grid((num_patches + 1, 3), (0, 0), colspan=1)
    ax_q.imshow(np.clip(q_rgb_np, 0, 1))
    ax_q.set_title('Query Image', fontweight='bold')
    ax_q.axis('off')
    
    ax_t = plt.subplot2grid((num_patches + 1, 3), (0, 1), colspan=1)
    ax_t.imshow(np.clip(t_rgb_np, 0, 1))
    ax_t.set_title('Template Image', fontweight='bold')
    ax_t.axis('off')
    
    # Info panel
    ax_info = plt.subplot2grid((num_patches + 1, 3), (0, 2), colspan=1)
    info_text = (
        f'Dense Flow Visualization\n'
        f'═══════════════════════\n\n'
        f'Direction: Template → Query\n'
        f'Coordinate System:\n'
        f'  Query patch center = (0, 0)\n'
        f'  Flow normalized by patch size\n\n'
        f'Patches shown: {num_patches}\n'
        f'Patch size: {ps}×{ps} pixels\n\n'
        f'Red boxes: Query patches\n'
        f'Blue boxes: Template buddies\n\n'
        f'In correspondence views:\n'
        f'  Blue dots: Template pixels\n'
        f'  Green dots: Query projections\n'
        f'  Red X: No correspondence'
    )
    ax_info.text(
        0.5, 0.5, info_text, 
        ha='center', va='center',
        fontsize=9, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    )
    ax_info.axis('off')
    
    # Visualize selected patches
    for idx in range(num_patches):
        row = idx + 1
        patch_i, patch_j = selected_patches[idx]
        patch_i, patch_j = patch_i.item(), patch_j.item()
        
        # Query patch
        ax_qp = plt.subplot2grid((num_patches + 1, 3), (row, 0), colspan=1)
        q_y_start = patch_i * ps
        q_x_start = patch_j * ps
        q_patch = q_rgb_np[q_y_start:q_y_start + ps, q_x_start:q_x_start + ps]
        ax_qp.imshow(np.clip(q_patch, 0, 1))
        ax_qp.set_title(f'Query Patch [{patch_i},{patch_j}]', fontsize=10)
        ax_qp.axis('off')
        
        # Template buddy patch
        ax_tp = plt.subplot2grid((num_patches + 1, 3), (row, 1), colspan=1)
        buddy_i = patch_buddy_i[patch_i, patch_j].item()
        buddy_j = patch_buddy_j[patch_i, patch_j].item()
        t_y_start = buddy_i * ps
        t_x_start = buddy_j * ps
        t_patch = t_rgb_np[t_y_start:t_y_start + ps, t_x_start:t_x_start + ps]
        ax_tp.imshow(np.clip(t_patch, 0, 1))
        ax_tp.set_title(f'Template Buddy [{buddy_i},{buddy_j}]', fontsize=10)
        ax_tp.axis('off')
        
        # Flow info
        ax_flow = plt.subplot2grid((num_patches + 1, 3), (row, 2), colspan=1)
        flow = flow_grid[patch_i, patch_j]  # [ps, ps, 2]
        weight = weight_grid[patch_i, patch_j]  # [ps, ps]
        
        num_visible = (weight > 0).sum().item()
        flow_x_mean = flow[:, :, 0][weight > 0].mean().item() if num_visible > 0 else 0
        flow_y_mean = flow[:, :, 1][weight > 0].mean().item() if num_visible > 0 else 0
        
        flow_text = (
            f'Correspondence Info:\n\n'
            f'Query: [{patch_i}, {patch_j}]\n'
            f'Buddy: [{buddy_i}, {buddy_j}]\n\n'
            f'Flow (normalized):\n'
            f'  Δx = {flow_x_mean:.3f}\n'
            f'  Δy = {flow_y_mean:.3f}\n\n'
            f'Visible pixels:\n'
            f'  {num_visible}/{ps * ps}'
        )
        ax_flow.text(
            0.5, 0.5, flow_text,
            ha='center', va='center', 
            fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        ax_flow.axis('off')
    
    plt.suptitle(
        f'Dense Template→Query Flow: Object {obj_label}, Batch {batch_idx}, Template {template_idx}\n'
        f'Flow in query patch coordinates (center=0,0, normalized by patch size)',
        fontsize=13, fontweight='bold'
    )
    
    Path(vis_dir).mkdir(parents=True, exist_ok=True)
    save_path = vis_dir / f"dense_flow_obj{obj_label}_b{batch_idx}_t{template_idx}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def visualize_train_correspondences(
    query_rgb: torch.Tensor,
    template_rgb: torch.Tensor,
    q_depth: torch.Tensor,
    t_depth: torch.Tensor,
    q_mask: torch.Tensor,
    query_seen_map: torch.Tensor,
    template_seen_map: torch.Tensor,
    template_not_seen_map: torch.Tensor,
    q_pose: torch.Tensor,
    t_pose: torch.Tensor,
    batch_idx: int,
    template_idx: int,
    obj_label: int,
    vis_dir: Path,
) -> Path:
    """
    Visualize query-template correspondences and visibility maps for training data.
    
    Shows RGB, depth, and visibility information for both query and template images.
    
    Args:
        query_rgb: [3, H, W] query image (centered/cropped)
        template_rgb: [3, H, W] template image
        q_depth: [H, W] query depth
        t_depth: [H, W] template depth
        q_mask: [H, W] query mask
        query_seen_map: [H, W] bool - query pixels visible in template
        template_seen_map: [H, W] bool - template pixels with query projections
        template_not_seen_map: [H, W] bool - template pixels not in query
        q_pose: [4, 4] query pose
        t_pose: [4, 4] template pose
        batch_idx: Batch index for filename
        template_idx: Template index for filename
        obj_label: Object label for title/filename
        vis_dir: Directory to save visualization
        
    Returns:
        Path to saved visualization
    """
    H, W = q_depth.shape
    
    # Convert to numpy
    q_rgb_np = query_rgb.cpu().permute(1, 2, 0).numpy()
    t_rgb_np = template_rgb.cpu().permute(1, 2, 0).numpy()
    q_depth_np = (q_depth * q_mask).cpu().numpy()
    t_depth_np = t_depth.cpu().numpy()
    q_mask_np = (q_mask.cpu().numpy() > 0.5)
    t_mask_np = (t_depth_np > 0)
    
    q_pose_np = q_pose.cpu().numpy()
    t_pose_np = t_pose.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # ROW 0: RGB
    axes[0, 0].imshow(np.clip(q_rgb_np, 0, 1))
    axes[0, 0].set_title(f'Query RGB (Centered)\n{H}x{W}', fontweight='bold')
    axes[0, 0].axis('off')
    
    q_rgb_masked = q_rgb_np * q_mask_np[:, :, None]
    axes[0, 1].imshow(q_rgb_masked)
    axes[0, 1].set_title(f'Query Masked\nObj {obj_label}', fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.clip(t_rgb_np, 0, 1))
    axes[0, 2].set_title(f'Template RGB\n{H}x{W}', fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[0, 3].axis('off')
    
    # ROW 1: Depth
    im1 = axes[1, 0].imshow(q_depth_np, cmap='jet', vmin=0, 
                             vmax=q_depth_np[q_depth_np > 0].max() if (q_depth_np > 0).any() else 1)
    axes[1, 0].set_title(
        f'Query Depth (PNG 224x224)\n[{q_depth_np[q_depth_np > 0].min():.0f}, {q_depth_np[q_depth_np > 0].max():.0f}] mm',
        fontweight='bold'
    )
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    axes[1, 1].axis('off')
    
    im3 = axes[1, 2].imshow(t_depth_np, cmap='jet', vmin=0, vmax=t_depth_np.max())
    axes[1, 2].set_title(
        f'Template Depth (PNG 224x224)\n[{t_depth_np[t_depth_np > 0].min():.0f}, {t_depth_np[t_depth_np > 0].max():.0f}] units',
        fontweight='bold', color='green'
    )
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
    
    axes[1, 3].axis('off')
    
    # ROW 2: Visibility Maps
    query_seen_np = query_seen_map.cpu().numpy()
    query_not_seen_np = q_mask_np & (~query_seen_np)
    template_seen_np = template_seen_map.cpu().numpy()
    template_not_seen_np = template_not_seen_map.cpu().numpy()
    
    # Query Seen Map
    query_seen_viz = np.zeros((H, W, 3))
    query_seen_viz[~q_mask_np] = [0.3, 0.3, 0.3]  # Gray for outside mask
    query_seen_viz[query_seen_np] = [0, 1, 0]  # Green for seen
    query_seen_viz[query_not_seen_np] = [1, 0, 0]  # Red for not seen
    
    axes[2, 0].imshow(query_seen_viz)
    axes[2, 0].set_title(
        f'Query Seen Map\n{query_seen_np.sum()}/{q_mask_np.sum()} pixels visible\n'
        f'Green=Seen, Red=Occluded, Gray=Outside',
        fontweight='bold'
    )
    axes[2, 0].axis('off')
    
    # Query NOT seen
    query_not_seen_viz = np.zeros((H, W, 3))
    query_not_seen_viz[~q_mask_np] = [0.3, 0.3, 0.3]
    query_not_seen_viz[query_not_seen_np] = [1, 0, 0]
    query_not_seen_viz[query_seen_np] = [0, 0.5, 0]
    
    axes[2, 1].imshow(query_not_seen_viz)
    axes[2, 1].set_title(
        f'Query NOT Seen (Occluded)\n{query_not_seen_np.sum()}/{q_mask_np.sum()} pixels\n'
        f'Red=Occluded, Gray=Outside',
        fontweight='bold', color='red'
    )
    axes[2, 1].axis('off')
    
    # Template Seen Map
    template_seen_viz = np.zeros((H, W, 3))
    template_seen_viz[~t_mask_np] = [0.3, 0.3, 0.3]
    template_seen_viz[template_seen_np] = [0, 1, 0]
    template_seen_viz[template_not_seen_np] = [1, 0.5, 0]
    
    axes[2, 2].imshow(template_seen_viz)
    axes[2, 2].set_title(
        f'Template Seen Map\n{template_seen_np.sum()}/{t_mask_np.sum()} pixels\n'
        f'Green=Has Query, Orange=No Query, Gray=Outside',
        fontweight='bold'
    )
    axes[2, 2].axis('off')
    
    # Template NOT Seen Map
    template_not_seen_viz = np.zeros((H, W, 3))
    template_not_seen_viz[~t_mask_np] = [0.3, 0.3, 0.3]
    template_not_seen_viz[template_not_seen_np] = [1, 0.5, 0]
    template_not_seen_viz[template_seen_np] = [0, 0.5, 0]
    
    axes[2, 3].imshow(template_not_seen_viz)
    axes[2, 3].set_title(
        f'Template NOT in Query\n{template_not_seen_np.sum()}/{t_mask_np.sum()} pixels\n'
        f'Orange=Not in Query, Gray=Outside',
        fontweight='bold', color='orange'
    )
    axes[2, 3].axis('off')
    
    # Title
    plt.suptitle(
        f'[TRAIN] Object {obj_label}: Query-Template Correspondence & Visibility Maps (224x224)\n'
        f'Query Pose: {q_pose_np[:3, 3].astype(int)} mm | Template Pose: {t_pose_np[:3, 3].astype(int)} mm\n'
        f'Row 1: Depth Validation | Row 2: Visibility Maps',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    Path(vis_dir).mkdir(parents=True, exist_ok=True)
    save_path = vis_dir / f"train_corr_obj{obj_label}_b{batch_idx}_t{template_idx}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def visualize_patch_correspondences(
    query_rgb: torch.Tensor,
    template_rgb: torch.Tensor,
    patch_has_object: torch.Tensor,
    patch_is_visible: torch.Tensor,
    patch_flow_x: torch.Tensor,
    patch_flow_y: torch.Tensor,
    patch_buddy_i: torch.Tensor,
    patch_buddy_j: torch.Tensor,
    batch_idx: int,
    template_idx: int,
    obj_label: int,
    patch_size: int,
    num_patches_per_side: int,
    vis_dir: Path,
    num_zoom_samples: int = 3,
) -> Path:
    """
    Visualize patch-level correspondences between query and template.
    
    Shows query patches (green=visible, red=not visible) and zooms into examples.
    For template, shows 3x3 grid of patches centered on buddy patch.
    
    Args:
        query_rgb: [3, H, W] query image (centered)
        template_rgb: [3, H, W] template image
        patch_has_object: [H_p, W_p] bool mask
        patch_is_visible: [H_p, W_p] bool mask
        patch_flow_x: [H_p, W_p] flow x component
        patch_flow_y: [H_p, W_p] flow y component
        patch_buddy_i: [H_p, W_p] buddy patch i indices
        patch_buddy_j: [H_p, W_p] buddy patch j indices
        batch_idx: Batch index for filename
        template_idx: Template index for filename
        obj_label: Object label for title/filename
        patch_size: Size of each patch (e.g., 16)
        num_patches_per_side: Number of patches per side (e.g., 14)
        vis_dir: Directory to save visualization
        num_zoom_samples: Number of patches to show in detail
        
    Returns:
        Path to saved visualization
    """
    ps = patch_size
    H_p = num_patches_per_side
    
    # Convert to numpy
    q_rgb = query_rgb.cpu().permute(1, 2, 0).numpy()
    t_rgb = template_rgb.cpu().permute(1, 2, 0).numpy()
    
    # Create figure with patch visualization + zoom examples
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # ROW 0: Query and Template with patch boxes
    ax_query = fig.add_subplot(gs[0, 0:2])
    ax_template = fig.add_subplot(gs[0, 2:4])
    
    # Query with patch boxes
    ax_query.imshow(np.clip(q_rgb, 0, 1))
    ax_query.set_title(
        f'Query Patches (Obj {obj_label})\nGreen=Visible, Red=Occluded, No box=No object',
        fontsize=12, fontweight='bold'
    )
    ax_query.axis('off')
    
    # Draw patch boxes on query
    for i_p in range(H_p):
        for j_p in range(H_p):
            if patch_has_object[i_p, j_p]:
                color = 'green' if patch_is_visible[i_p, j_p] else 'red'
                rect = Rectangle(
                    (j_p * ps, i_p * ps), ps, ps,
                    linewidth=1, edgecolor=color, facecolor='none'
                )
                ax_query.add_patch(rect)
    
    # Template with patch boxes
    ax_template.imshow(np.clip(t_rgb, 0, 1))
    ax_template.set_title(
        f'Template Patches\nShowing buddy patches for visible query patches',
        fontsize=12, fontweight='bold'
    )
    ax_template.axis('off')
    
    # Draw buddy patch boxes on template
    for i_p in range(H_p):
        for j_p in range(H_p):
            if patch_is_visible[i_p, j_p]:
                buddy_i = patch_buddy_i[i_p, j_p].item()
                buddy_j = patch_buddy_j[i_p, j_p].item()
                rect = Rectangle(
                    (buddy_j * ps, buddy_i * ps), ps, ps,
                    linewidth=1, edgecolor='blue', facecolor='none'
                )
                ax_template.add_patch(rect)
    
    # Find visible patches and randomly sample
    visible_patches = torch.nonzero(patch_is_visible, as_tuple=False)
    
    if len(visible_patches) > 0:
        num_samples = min(num_zoom_samples, len(visible_patches))
        perm = torch.randperm(len(visible_patches))[:num_samples]
        sampled_patches = visible_patches[perm]
        
        # Visualize each sampled patch
        for idx in range(num_samples):
            row = idx + 1
            sample_i, sample_j = sampled_patches[idx]
            sample_i, sample_j = sample_i.item(), sample_j.item()
            
            # Query patch
            ax_q_zoom = fig.add_subplot(gs[row, 0])
            q_y_start = sample_i * ps
            q_x_start = sample_j * ps
            q_patch = q_rgb[q_y_start:q_y_start + ps, q_x_start:q_x_start + ps]
            ax_q_zoom.imshow(np.clip(q_patch, 0, 1))
            ax_q_zoom.set_title(f'Query Patch [{sample_i},{sample_j}]', fontsize=10, fontweight='bold')
            ax_q_zoom.axis('off')
            
            # Template 3x3 patches (buddy in center)
            buddy_i = patch_buddy_i[sample_i, sample_j].item()
            buddy_j = patch_buddy_j[sample_i, sample_j].item()
            
            # Extract 3x3 region centered on buddy
            t_i_start = max(0, buddy_i - 1)
            t_j_start = max(0, buddy_j - 1)
            t_i_end = min(H_p, buddy_i + 2)
            t_j_end = min(H_p, buddy_j + 2)
            
            t_y_start_3x3 = t_i_start * ps
            t_x_start_3x3 = t_j_start * ps
            t_y_end_3x3 = t_i_end * ps
            t_x_end_3x3 = t_j_end * ps
            
            t_patch_3x3 = t_rgb[t_y_start_3x3:t_y_end_3x3, t_x_start_3x3:t_x_end_3x3]
            
            ax_t_zoom = fig.add_subplot(gs[row, 1])
            ax_t_zoom.imshow(np.clip(t_patch_3x3, 0, 1))
            ax_t_zoom.set_title(
                f'Template 3×3 Patches (Buddy [{buddy_i},{buddy_j}] in center)',
                fontsize=10, fontweight='bold'
            )
            ax_t_zoom.axis('off')
            
            # Flow visualization
            ax_flow = fig.add_subplot(gs[row, 2])
            flow_x_px = patch_flow_x[sample_i, sample_j].item() * ps
            flow_y_px = patch_flow_y[sample_i, sample_j].item() * ps
            
            ax_flow.text(
                0.5, 0.5, 
                f'Patch Correspondence:\n\n'
                f'Query: [{sample_i}, {sample_j}]\n'
                f'Template Buddy: [{buddy_i}, {buddy_j}]\n\n'
                f'Flow (pixels):\n'
                f'  Δx = {flow_x_px:.2f} px\n'
                f'  Δy = {flow_y_px:.2f} px\n\n'
                f'Flow (normalized):\n'
                f'  Δx = {patch_flow_x[sample_i, sample_j].item():.3f}\n'
                f'  Δy = {patch_flow_y[sample_i, sample_j].item():.3f}',
                ha='center', va='center', fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
            ax_flow.axis('off')
        
        # Statistics
        ax_stats = fig.add_subplot(gs[num_samples, 3])
        num_with_obj = patch_has_object.sum().item()
        num_visible = patch_is_visible.sum().item()
        num_occluded = (patch_has_object & ~patch_is_visible).sum().item()
        
        ax_stats.text(
            0.5, 0.5,
            f'Patch Statistics:\n\n'
            f'Total: {H_p}×{H_p} = {H_p * H_p}\n'
            f'With object: {num_with_obj}\n'
            f'Visible: {num_visible}\n'
            f'Occluded: {num_occluded}\n\n'
            f'Visibility:\n'
            f'{num_visible}/{num_with_obj}\n'
            f'= {100 * num_visible / max(num_with_obj, 1):.1f}%',
            ha='center', va='center', fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        )
        ax_stats.axis('off')
    
    plt.suptitle(
        f'[TRAIN] Object {obj_label}: Patch-Level Correspondences\n'
        f'Patch size: {ps}×{ps} pixels, Grid: {H_p}×{H_p} patches',
        fontsize=14, fontweight='bold'
    )
    
    Path(vis_dir).mkdir(parents=True, exist_ok=True)
    save_path = vis_dir / f"train_patches_obj{obj_label}_b{batch_idx}_t{template_idx}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path
