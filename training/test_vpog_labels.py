"""
VPOG Ground Truth Labels Visualization

Visualizes the training labels to verify correctness:
1. Query + Templates (positive/negative) with their indices
2. Classification labels: which query patches match which template patches
3. Flow labels: For positive templates, show 3 random patch matches with their flows

Run: python training/test_vpog_labels.py --real_data
"""

import os
import sys
sys.path.append("./vpog")
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Optional
import random

import hydra
from omegaconf import DictConfig, OmegaConf

from training.dataloader.vpog_dataset import VPOGTrainDataset, VPOGBatch

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denormalize_image(tensor):
    """Denormalize ImageNet normalized image."""
    img = tensor.cpu().numpy()  # [C, H, W]
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    img = img * std + mean
    img = np.transpose(img, (1, 2, 0))  # [H, W, C]
    return np.clip(img, 0, 1)


def compute_pose_angle(pose1: torch.Tensor, pose2: torch.Tensor) -> float:
    """Compute rotation angle in degrees between two 4x4 poses."""
    R1 = pose1[:3, :3].cpu().numpy()
    R2 = pose2[:3, :3].cpu().numpy()
    R_rel = R1.T @ R2
    trace = np.trace(R_rel)
    angle_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    return np.degrees(angle_rad)


def visualize_images_with_indices(batch: VPOGBatch, b_idx: int = 0) -> plt.Figure:
    """
    Visualization 1: Show Query + All Templates with their S indices and pose angles
    
    Verifies:
    - Real GSO query image loaded correctly (with mask)
    - Positive templates are nearest in pose
    - Template indices and types are correct
    """
    S = batch.images.shape[1] - 1
    
    # Compute all rotation angles from query
    query_pose = batch.poses[b_idx, 0]
    angles = []
    for s_idx in range(S):
        template_pose = batch.poses[b_idx, s_idx + 1]
        angle = compute_pose_angle(query_pose, template_pose)
        angles.append(angle)
    
    fig, axes = plt.subplots(2, S + 1, figsize=(3 * (S + 1), 8))
    
    # === Row 1: Images ===
    # Query image with mask overlay
    query_img = denormalize_image(batch.images[b_idx, 0])
    query_mask = batch.masks[b_idx, 0].cpu().numpy()
    
    axes[0, 0].imshow(query_img)
    # Overlay mask as semi-transparent red
    mask_overlay = np.zeros((*query_mask.shape, 4))
    mask_overlay[query_mask > 0.5] = [1, 0, 0, 0.3]
    axes[0, 0].imshow(mask_overlay)
    
    # Get scene info if available
    info_text = 'Query (Real GSO)'
    if hasattr(batch.infos, 'infos') and len(batch.infos.infos) > b_idx:
        scene_id = batch.infos.infos.iloc[b_idx].get('scene_id', 'N/A')
        label = batch.infos.infos.iloc[b_idx].get('label', 'N/A')
        info_text += f'\\nScene: {scene_id}\\nObj: {label}'
    
    axes[0, 0].set_title(info_text, fontsize=12, weight='bold', color='blue')
    axes[0, 0].axis('off')
    for spine in axes[0, 0].spines.values():
        spine.set_edgecolor('blue')
        spine.set_linewidth(4)
        spine.set_visible(True)
    
    # Template images
    for s_idx in range(S):
        template_img = denormalize_image(batch.images[b_idx, s_idx + 1])
        axes[0, s_idx + 1].imshow(template_img)
        
        is_positive = batch.template_types[b_idx, s_idx].item() == 0
        template_type = "POS" if is_positive else "NEG"
        color = 'green' if is_positive else 'red'
        template_global_idx = batch.template_indices[b_idx, s_idx].item()
        angle = angles[s_idx]
        
        axes[0, s_idx + 1].set_title(f'S={s_idx} ({template_type})\\nIdx={template_global_idx}\\nΔR={angle:.1f}°', 
                                      fontsize=11, weight='bold', color=color)
        axes[0, s_idx + 1].axis('off')
        
        # Border
        for spine in axes[0, s_idx + 1].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
            spine.set_visible(True)
    
    # === Row 2: Angle ranking ===
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, 'Pose Angle\\nRanking\\n(Smaller = Nearer)', 
                    ha='center', va='center', fontsize=12, weight='bold')
    
    # Sort by angle and show ranking
    sorted_indices = np.argsort(angles)
    for rank, s_idx in enumerate(sorted_indices):
        is_positive = batch.template_types[b_idx, s_idx].item() == 0
        color = 'green' if is_positive else 'red'
        template_type = "POS" if is_positive else "NEG"
        
        axes[1, s_idx + 1].axis('off')
        axes[1, s_idx + 1].text(0.5, 0.5, f'Rank: {rank+1}\\n{angles[s_idx]:.1f}°\\n({template_type})', 
                                ha='center', va='center', fontsize=11, color=color, weight='bold',
                                bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=3))
    
    # Check: Are positives the nearest?
    num_positives = (batch.template_types[b_idx] == 0).sum().item()
    nearest_indices = sorted_indices[:num_positives]
    are_all_positive = all(batch.template_types[b_idx, idx].item() == 0 for idx in nearest_indices)
    
    check_text = "✓ CORRECT: Positives are nearest" if are_all_positive else "✗ ERROR: Positives are NOT nearest!"
    check_color = 'green' if are_all_positive else 'red'
    fig.text(0.5, 0.02, check_text, ha='center', fontsize=14, weight='bold', color=check_color,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=check_color, linewidth=3))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def compute_classification_labels_from_flow(batch: VPOGBatch, b_idx: int, s_idx: int) -> torch.Tensor:
    """
    Compute GT classification labels from flow.
    
    For each query patch (qy, qx), the flow gives the offset to the matching template patch:
    template_patch = (qy + flow_y, qx + flow_x)
    
    Convert 2D coordinates to flat index: label = ty * W_p + tx
    
    Args:
        batch: VPOGBatch
        b_idx: batch index
        s_idx: template index
    
    Returns:
        labels: [Nq] - flat indices of template patches, or Nq for unseen
    """
    H_p = W_p = 14
    Nq = Nt = H_p * W_p
    
    # Get flow and visibility
    flow = batch.flows[b_idx, s_idx]  # [H_p, W_p, 2]
    visibility = batch.visibility[b_idx, s_idx]  # [H_p, W_p]
    
    labels = torch.zeros(Nq, dtype=torch.long, device=flow.device)
    
    for qy in range(H_p):
        for qx in range(W_p):
            q_idx = qy * W_p + qx
            
            if visibility[qy, qx] < 0.5:
                # Unseen - mark as Nt (unseen token index)
                labels[q_idx] = Nt
            else:
                # Compute template patch location
                flow_y, flow_x = flow[qy, qx]
                ty = qy + flow_y
                tx = qx + flow_x
                
                # Clip to valid range
                ty = torch.clamp(ty, 0, H_p - 1)
                tx = torch.clamp(tx, 0, W_p - 1)
                
                # Convert to flat index
                ty_int = int(ty.round())
                tx_int = int(tx.round())
                t_idx = ty_int * W_p + tx_int
                
                labels[q_idx] = t_idx
    
    return labels


def visualize_classification_labels(batch: VPOGBatch, b_idx: int = 0) -> plt.Figure:
    """
    Visualization 2: Classification labels - which query patches match which template patches
    
    For each positive template, show:
    - GT classification labels derived from flow
    - Reshaped as 14x14 grid to see spatial structure
    - Blue = visible match, Red = unseen
    """
    S = batch.images.shape[1] - 1
    H_p = W_p = 14  # 224/16 = 14 patches per side
    Nq = Nt = H_p * W_p
    
    # Count positive templates
    num_positives = (batch.template_types[b_idx] == 0).sum().item()
    
    if num_positives == 0:
        print("No positive templates in this batch!")
        return None
    
    fig, axes = plt.subplots(1, num_positives + 1, figsize=(4 * (num_positives + 1), 4))
    if num_positives == 0:
        axes = [axes]
    
    # Show query image
    query_img = denormalize_image(batch.images[b_idx, 0])
    axes[0].imshow(query_img)
    axes[0].set_title('Query', fontsize=14, weight='bold')
    axes[0].axis('off')
    
    # For each positive template, show classification labels
    pos_idx = 0
    for s_idx in range(S):
        if batch.template_types[b_idx, s_idx].item() != 0:
            continue
        
        # Compute GT labels from flow
        gt_labels = compute_classification_labels_from_flow(batch, b_idx, s_idx)  # [Nq]
        
        # Reshape to grid
        label_grid = gt_labels.reshape(H_p, W_p).cpu().numpy()
        
        # Use special value for unseen (Nt = 196)
        # Create masked array to show unseen differently
        import numpy.ma as ma
        label_grid_masked = ma.masked_where(label_grid >= Nt, label_grid)
        
        im = axes[pos_idx + 1].imshow(label_grid_masked, cmap='tab20', vmin=0, vmax=Nt-1)
        axes[pos_idx + 1].imshow(np.where(label_grid >= Nt, 1, np.nan), cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        axes[pos_idx + 1].set_title(f'S={s_idx}: Classification Labels\\n(Query patch → Template patch ID)', 
                                     fontsize=12, weight='bold', color='green')
        axes[pos_idx + 1].axis('off')
        plt.colorbar(im, ax=axes[pos_idx + 1], fraction=0.046, label='Template Patch ID')
        
        pos_idx += 1
    
    plt.tight_layout()
    return fig


def visualize_flow_matches(batch: VPOGBatch, b_idx: int = 0, num_samples: int = 3) -> list:
    """
    Visualization 3: For each positive template, show 3 random patch matches with flow
    
    Shows:
    - Query patch (16x16 crop)
    - Template patch (16x16 crop)
    - GT flow vectors overlaid
    - Flow magnitude heatmap
    """
    S = batch.images.shape[1] - 1
    H_p = W_p = 14
    patch_size = 16
    
    figures = []
    
    for s_idx in range(S):
        if batch.template_types[b_idx, s_idx].item() != 0:
            continue  # Skip negative templates
        
        # Get images
        query_img = denormalize_image(batch.images[b_idx, 0])
        template_img = denormalize_image(batch.images[b_idx, s_idx + 1])
        
        # Get flow and visibility
        gt_flow = batch.flows[b_idx, s_idx].cpu().numpy()  # [H_p, W_p, 2]
        gt_vis = batch.visibility[b_idx, s_idx].cpu().numpy()  # [H_p, W_p]
        
        # Find visible patches
        visible_patches = np.where(gt_vis > 0.5)
        if len(visible_patches[0]) == 0:
            print(f"No visible patches for template S={s_idx}")
            continue
        
        # Sample random visible patches
        num_visible = len(visible_patches[0])
        sample_indices = np.random.choice(num_visible, min(num_samples, num_visible), replace=False)
        
        # Create figure for this template
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for sample_idx, idx in enumerate(sample_indices):
            py, px = visible_patches[0][idx], visible_patches[1][idx]
            
            # Extract query patch (16x16 pixels)
            qy_start, qx_start = py * patch_size, px * patch_size
            query_patch = query_img[qy_start:qy_start+patch_size, qx_start:qx_start+patch_size]
            
            # Get flow for this patch
            flow_y, flow_x = gt_flow[py, px]  # Flow in patch units
            
            # Convert flow to pixels and get template patch location
            # Query patch (py, px) matches template patch at (py + flow_y, px + flow_x)
            ty = py + flow_y
            tx = px + flow_x
            
            # Clip to valid range
            ty = np.clip(ty, 0, H_p - 1)
            tx = np.clip(tx, 0, W_p - 1)
            
            # Extract template patch (use nearest neighbor since flow might not be integer)
            ty_pix, tx_pix = int(ty * patch_size), int(tx * patch_size)
            template_patch = template_img[ty_pix:ty_pix+patch_size, tx_pix:tx_pix+patch_size]
            
            # === Column 1: Query Patch ===
            axes[sample_idx, 0].imshow(query_patch)
            axes[sample_idx, 0].set_title(f'Query Patch ({py}, {px})', fontsize=10)
            axes[sample_idx, 0].axis('off')
            
            # === Column 2: Template Patch ===
            axes[sample_idx, 1].imshow(template_patch)
            axes[sample_idx, 1].set_title(f'Template Patch ({ty:.1f}, {tx:.1f})', fontsize=10)
            axes[sample_idx, 1].axis('off')
            
            # === Column 3: Flow Vector ===
            axes[sample_idx, 2].imshow(query_patch, alpha=0.5)
            # Draw flow vector from center
            center_y, center_x = patch_size / 2, patch_size / 2
            flow_y_pix, flow_x_pix = flow_y * patch_size, flow_x * patch_size
            axes[sample_idx, 2].arrow(center_x, center_y, flow_x_pix, flow_y_pix,
                                      head_width=2, head_length=2, fc='lime', ec='lime', linewidth=2)
            axes[sample_idx, 2].set_title(f'Flow: ({flow_x:.2f}, {flow_y:.2f}) patches', fontsize=10)
            axes[sample_idx, 2].axis('off')
            
            # === Column 4: Flow Magnitude ===
            flow_mag = np.sqrt(flow_x**2 + flow_y**2)
            # Show flow magnitude as text
            axes[sample_idx, 3].text(0.5, 0.5, f'Flow Magnitude:\\n{flow_mag:.3f} patches\\n{flow_mag*patch_size:.1f} pixels',
                                     ha='center', va='center', fontsize=12,
                                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[sample_idx, 3].set_xlim(0, 1)
            axes[sample_idx, 3].set_ylim(0, 1)
            axes[sample_idx, 3].axis('off')
        
        fig.suptitle(f'S={s_idx}: Random Patch Matches with GT Flow', fontsize=14, weight='bold')
        plt.tight_layout()
        figures.append(fig)
    
    return figures


def run_label_visualization(device: str = 'cuda'):
    """Run the label visualization pipeline."""
    print("\n" + "="*80)
    print("VPOG GROUND TRUTH LABELS VISUALIZATION")
    print("="*80)
    
    output_dir = Path("./tmp/vpog_label_viz")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Load data
    print("Loading data...")
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    
    config_dir = str((Path(__file__).parent / "config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        data_cfg = compose(config_name="data/vpog")
    
    dataset = instantiate(data_cfg.data.dataloader)
    
    from src.utils.dataloader import NoneFilteringDataLoader
    dataloader = NoneFilteringDataLoader(
        dataset.web_dataloader.datapipeline,
        batch_size=2,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    batch = next(iter(dataloader))
    batch = batch.to(device)
    
    B = batch.images.shape[0]
    S = batch.images.shape[1] - 1
    print(f"✓ Loaded batch: B={B}, S={S} templates\n")
    
    # Visualize each sample in batch
    for b_idx in range(B):
        print(f"Visualizing sample {b_idx}...")
        
        # 1. Images with indices
        print("  Creating image overview...")
        fig1 = visualize_images_with_indices(batch, b_idx)
        fig1.savefig(output_dir / f"sample{b_idx}_1_images.png", dpi=150, bbox_inches='tight')
        plt.close(fig1)
        
        # 2. Classification labels
        print("  Creating classification labels...")
        fig2 = visualize_classification_labels(batch, b_idx)
        if fig2 is not None:
            fig2.savefig(output_dir / f"sample{b_idx}_2_classification.png", dpi=150, bbox_inches='tight')
            plt.close(fig2)
        
        # 3. Flow matches
        print("  Creating flow matches...")
        figs3 = visualize_flow_matches(batch, b_idx, num_samples=3)
        for idx, fig in enumerate(figs3):
            fig.savefig(output_dir / f"sample{b_idx}_3_flow_template{idx}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        print(f"  ✓ Sample {b_idx} complete\n")
    
    print("="*80)
    print(f"✓ ALL VISUALIZATIONS SAVED TO: {output_dir}")
    print("="*80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='VPOG GT Labels Visualization')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    run_label_visualization(device=device)


if __name__ == "__main__":
    main()
