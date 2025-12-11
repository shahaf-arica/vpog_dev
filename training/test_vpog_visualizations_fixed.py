"""
Fixed visualization functions for VPOG pipeline test
These show GT labels, not model predictions (since model isn't trained yet)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from training.dataloader.vpog_dataset import VPOGBatch
from training.visualization.flow_vis import flow_to_color


def denormalize_image(tensor):
    """
    Denormalize ImageNet-normalized image tensor.
    
    Args:
        tensor: [3, H, W] normalized image
    
    Returns:
        img: [H, W, 3] RGB image in [0, 1]
    """
    img = tensor.cpu().numpy()  # [3, H, W]
    # ImageNet denormalization
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    img = img * std + mean
    img = np.transpose(img, (1, 2, 0))  # [H, W, 3]
    return np.clip(img, 0, 1)


def visualize_data_and_poses(batch: VPOGBatch, b_idx: int = 0) -> plt.Figure:
    """
    Visualize input data: query, templates, and pose information.
    
    Shows:
    - Query and template images (properly denormalized)
    - Angular distances between query and templates
    - Template types (positive/negative)
    - Basic GT flow statistics
    """
    S = batch.images.shape[1] - 1
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(3, max(4, S), hspace=0.3, wspace=0.3)
    
    # === Row 1: Query Image ===
    ax_query = fig.add_subplot(gs[0, :2])
    query_img = denormalize_image(batch.images[b_idx, 0])
    ax_query.imshow(query_img)
    ax_query.set_title("Query Image", fontsize=14, weight='bold')
    ax_query.axis('off')
    
    # Query info (simplified)
    ax_query_info = fig.add_subplot(gs[0, 2:])
    ax_query_info.axis('off')
    query_pose = batch.poses[b_idx, 0].cpu().numpy()
    query_t = query_pose[:3, 3]
    d_ref = batch.d_ref[b_idx].cpu().numpy()
    
    query_info_text = f"""
    QUERY INFO:
    
    Translation: [{query_t[0]:.2f}, {query_t[1]:.2f}, {query_t[2]:.2f}]
    
    d_ref: [{d_ref[0]:.3f}, {d_ref[1]:.3f}, {d_ref[2]:.3f}]
    """
    ax_query_info.text(0.05, 0.5, query_info_text, 
                       fontsize=11, verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # === Row 2: Template Images ===
    for s_idx in range(S):
        col = s_idx
        ax = fig.add_subplot(gs[1, col])
        
        template_img = denormalize_image(batch.images[b_idx, s_idx + 1])
        ax.imshow(template_img)
        
        # Get template type
        is_positive = batch.template_types[b_idx, s_idx].item() == 0
        template_type = "POSITIVE" if is_positive else "NEGATIVE"
        color = 'green' if is_positive else 'red'
        
        template_idx = batch.template_indices[b_idx, s_idx].item()
        ax.set_title(f"T{s_idx} [{template_type}]\\nIdx: {template_idx}", 
                    fontsize=10, color=color, weight='bold')
        ax.axis('off')
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    # === Row 3: Pose and Flow Information ===
    ax_info = fig.add_subplot(gs[2, :])
    ax_info.axis('off')
    
    # Compute pose differences and flow stats
    query_R = query_pose[:3, :3]
    
    info_lines = ["TEMPLATE INFORMATION:\\n"]
    info_lines.append(f"{'ID':<4} {'Type':<10} {'Template Idx':<14} {'Angle from Query':<18} {'GT Flow Visible %':<18}")
    info_lines.append("-" * 80)
    
    for s_idx in range(S):
        template_pose = batch.poses[b_idx, s_idx + 1].cpu().numpy()
        template_R = template_pose[:3, :3]
        
        # Compute SO(3) angular distance
        R_rel = query_R.T @ template_R
        trace_R = np.trace(R_rel)
        angle_rad = np.arccos(np.clip((trace_R - 1) / 2, -1, 1))
        angle_deg = np.degrees(angle_rad)
        
        # Flow visibility
        visibility = batch.visibility[b_idx, s_idx].cpu().numpy()
        vis_pct = 100 * visibility.mean()
        
        is_positive = batch.template_types[b_idx, s_idx].item() == 0
        template_type = "POSITIVE" if is_positive else "NEGATIVE"
        template_idx = batch.template_indices[b_idx, s_idx].item()
        
        info_lines.append(
            f"{s_idx:<4} {template_type:<10} {template_idx:<14} {angle_deg:17.2f}°  {vis_pct:17.1f}%"
        )
    
    info_text = "\\n".join(info_lines)
    ax_info.text(0.05, 0.95, info_text, 
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    fig.suptitle(f"Data Visualization (Batch {b_idx}) - Check: Positive templates should have smallest angles!", 
                fontsize=14, weight='bold')
    
    return fig


def visualize_template_flow(outputs: dict, batch: VPOGBatch, b_idx: int = 0, s_idx: int = 0) -> plt.Figure:
    """
    Visualize GROUND TRUTH flow labels for a specific template.
    
    Shows:
    - Query and template images
    - GT flow vectors and magnitudes
    - Visibility masks
    - Flow statistics
    
    Note: Model outputs are NOT shown since model hasn't been trained yet.
    """
    # Get GT data
    gt_flow = batch.flows[b_idx, s_idx].cpu().numpy()  # [H_p, W_p, 2]
    gt_visibility = batch.visibility[b_idx, s_idx].cpu().numpy()  # [H_p, W_p]
    gt_patch_vis = batch.patch_visibility[b_idx, s_idx].cpu().numpy()  # [H_p, W_p]
    
    # Get query and template images for context
    query_img = denormalize_image(batch.images[b_idx, 0])
    template_img = denormalize_image(batch.images[b_idx, s_idx + 1])
    
    # Create figure
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, 5, hspace=0.4, wspace=0.3)
    
    # === Row 1: Images and Flow Visualization ===
    # Query Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(query_img)
    ax1.set_title("Query Image", fontsize=11, weight='bold')
    ax1.axis('off')
    
    # Template Image
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(template_img)
    is_positive = batch.template_types[b_idx, s_idx].item() == 0
    template_type = "POSITIVE" if is_positive else "NEGATIVE"
    ax2.set_title(f"Template {s_idx} ({template_type})", fontsize=11, weight='bold')
    ax2.axis('off')
    
    # GT Flow Magnitude
    ax3 = fig.add_subplot(gs[0, 2])
    gt_mag = np.sqrt((gt_flow**2).sum(axis=-1))
    max_flow = max(gt_mag.max(), 1.0)  # At least 1 pixel
    im3 = ax3.imshow(gt_mag, cmap='hot', vmin=0, vmax=max_flow)
    ax3.set_title("GT Flow Magnitude (pixels)", fontsize=11)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, label='pixels')
    
    # GT Visibility
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(gt_visibility, cmap='gray', vmin=0, vmax=1)
    ax4.set_title("Pixel Visibility", fontsize=11)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # Patch Visibility
    ax5 = fig.add_subplot(gs[0, 4])
    im5 = ax5.imshow(gt_patch_vis, cmap='gray', vmin=0, vmax=1)
    ax5.set_title("Patch Visibility", fontsize=11)
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    # === Row 2: Flow Details ===
    # Flow X component
    ax6 = fig.add_subplot(gs[1, 0])
    flow_max = max(abs(gt_flow[..., 0]).max(), abs(gt_flow[..., 1]).max(), 1.0)
    im6 = ax6.imshow(gt_flow[..., 0], cmap='RdBu_r', vmin=-flow_max, vmax=flow_max)
    ax6.set_title("GT Flow X", fontsize=11)
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046, label='pixels')
    
    # Flow Y component
    ax7 = fig.add_subplot(gs[1, 1])
    im7 = ax7.imshow(gt_flow[..., 1], cmap='RdBu_r', vmin=-flow_max, vmax=flow_max)
    ax7.set_title("GT Flow Y", fontsize=11)
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046, label='pixels')
    
    # Flow direction (color-coded)
    ax8 = fig.add_subplot(gs[1, 2])
    flow_rgb = flow_to_color(gt_flow, max_flow=max_flow)
    ax8.imshow(flow_rgb)
    ax8.set_title("GT Flow Direction (HSV)", fontsize=11)
    ax8.axis('off')
    
    # Flow vectors overlay
    ax9 = fig.add_subplot(gs[1, 3])
    ax9.imshow(query_img, alpha=0.5)
    # Downsample for clarity
    H_p, W_p = gt_flow.shape[:2]
    step = max(1, H_p // 10)  # ~10 arrows per dimension
    patch_size = 224 // H_p
    for i in range(0, H_p, step):
        for j in range(0, W_p, step):
            if gt_visibility[i, j] > 0.5:
                y, x = i * patch_size + patch_size//2, j * patch_size + patch_size//2
                dy, dx = gt_flow[i, j]
                ax9.arrow(x, y, dx*patch_size, dy*patch_size, 
                         head_width=3, head_length=3, fc='lime', ec='lime', linewidth=1.5)
    ax9.set_title("Flow Vectors on Query", fontsize=11)
    ax9.axis('off')
    
    # Statistics
    ax10 = fig.add_subplot(gs[1, 4])
    ax10.axis('off')
    
    visible_pixels = gt_visibility.sum()
    total_pixels = gt_visibility.size
    visible_patches = gt_patch_vis.sum()
    mean_flow_mag = gt_mag[gt_visibility > 0].mean() if visible_pixels > 0 else 0
    
    stats = f"""
    Template {s_idx}: {template_type}
    
    Visibility:
    • Pixels: {visible_pixels}/{total_pixels}
      ({100*visible_pixels/total_pixels:.1f}%)
    • Patches: {int(visible_patches)}/{total_pixels}
      ({100*visible_patches/total_pixels:.1f}%)
    
    Flow:
    • Mean mag: {mean_flow_mag:.2f} px
    • Max mag: {gt_mag.max():.2f} px
    • X range: [{gt_flow[...,0].min():.2f}, {gt_flow[...,0].max():.2f}]
    • Y range: [{gt_flow[...,1].min():.2f}, {gt_flow[...,1].max():.2f}]
    """
    ax10.text(0.05, 0.95, stats, fontsize=9, verticalalignment='top',
             family='monospace', bbox=dict(boxstyle='round', 
             facecolor='lightgreen' if is_positive else 'lightcoral', alpha=0.5))
    
    fig.suptitle(f"GT Flow Visualization for Template {s_idx} (Batch {b_idx})", fontsize=16, weight='bold')
    
    return fig


def visualize_patch_flow_detailed(outputs: dict, batch: VPOGBatch, b_idx: int = 0) -> plt.Figure:
    """
    Detailed patch-level GT flow visualization.
    
    Shows GT flow at patch resolution with flow vectors.
    """
    S = batch.images.shape[1] - 1
    
    # Pick first positive template
    s_idx = 0
    for s in range(S):
        if batch.template_types[b_idx, s].item() == 0:
            s_idx = s
            break
    
    # Get GT flow for this template
    gt_flow = batch.flows[b_idx, s_idx].cpu().numpy()  # [H_p, W_p, 2]
    gt_vis = batch.visibility[b_idx, s_idx].cpu().numpy()  # [H_p, W_p]
    
    # Get images
    query_img = denormalize_image(batch.images[b_idx, 0])
    template_img = denormalize_image(batch.images[b_idx, s_idx + 1])
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # === Row 1: Images and Flow ===
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title("Query Image", fontsize=10, weight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(template_img)
    is_positive = batch.template_types[b_idx, s_idx].item() == 0
    template_type = "POSITIVE" if is_positive else "NEGATIVE"
    axes[0, 1].set_title(f"Template {s_idx} ({template_type})", fontsize=10, weight='bold')
    axes[0, 1].axis('off')
    
    # Flow magnitude
    flow_mag = np.sqrt((gt_flow**2).sum(axis=-1))
    max_flow = max(flow_mag.max(), 1.0)
    im3 = axes[0, 2].imshow(flow_mag, cmap='hot', vmin=0, vmax=max_flow)
    axes[0, 2].set_title("GT Flow Magnitude", fontsize=10)
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, label='pixels')
    
    # Visibility
    im4 = axes[0, 3].imshow(gt_vis, cmap='gray', vmin=0, vmax=1)
    axes[0, 3].set_title("Visibility", fontsize=10)
    axes[0, 3].axis('off')
    plt.colorbar(im4, ax=axes[0, 3], fraction=0.046)
    
    # === Row 2: Flow Details ===
    # Flow X
    flow_max = max(abs(gt_flow).max(), 1.0)
    im5 = axes[1, 0].imshow(gt_flow[..., 0], cmap='RdBu_r', vmin=-flow_max, vmax=flow_max)
    axes[1, 0].set_title("GT Flow X", fontsize=10)
    axes[1, 0].axis('off')
    plt.colorbar(im5, ax=axes[1, 0], fraction=0.046, label='pixels')
    
    # Flow Y
    im6 = axes[1, 1].imshow(gt_flow[..., 1], cmap='RdBu_r', vmin=-flow_max, vmax=flow_max)
    axes[1, 1].set_title("GT Flow Y", fontsize=10)
    axes[1, 1].axis('off')
    plt.colorbar(im6, ax=axes[1, 1], fraction=0.046, label='pixels')
    
    # Flow direction
    flow_rgb = flow_to_color(gt_flow, max_flow=max_flow)
    axes[1, 2].imshow(flow_rgb)
    axes[1, 2].set_title("Flow Direction (HSV)", fontsize=10)
    axes[1, 2].axis('off')
    
    # Statistics
    axes[1, 3].axis('off')
    visible_patches = gt_vis.sum()
    total_patches = gt_vis.size
    mean_flow_mag = flow_mag[gt_vis > 0].mean() if visible_patches > 0 else 0
    
    stats_text = f"""
    GT Flow Statistics:
    
    Template {s_idx}: {template_type}
    
    Visibility:
    • {int(visible_patches)}/{total_patches} patches
      ({100*visible_patches/total_patches:.1f}%)
    
    Flow:
    • Mean mag: {mean_flow_mag:.2f} px
    • Max mag: {flow_mag.max():.2f} px
    • X: [{gt_flow[...,0].min():.2f}, {gt_flow[...,0].max():.2f}]
    • Y: [{gt_flow[...,1].min():.2f}, {gt_flow[...,1].max():.2f}]
    """
    axes[1, 3].text(0.05, 0.95, stats_text, fontsize=9, verticalalignment='top',
                   family='monospace', bbox=dict(boxstyle='round', 
                   facecolor='lightgreen' if is_positive else 'lightcoral', alpha=0.5))
    
    fig.suptitle(f"GT Patch-Level Flow Detail - Batch {b_idx}", fontsize=14, weight='bold')
    plt.tight_layout()
    
    return fig


def visualize_comprehensive_pipeline(outputs: dict, batch: VPOGBatch, output_dir: Path):
    """Comprehensive visualization of GT labels in the pipeline."""
    print("\n" + "="*80)
    print("COMPREHENSIVE GT LABELS VISUALIZATION")
    print("="*80)
    print("Note: Showing GROUND TRUTH labels, not model predictions")
    print("      (model hasn't been trained yet)")
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        B = batch.images.shape[0]
        S = batch.images.shape[1] - 1  # Number of templates
        
        # Visualize for first batch samples
        for b_idx in range(min(2, B)):
            print(f"\n=== Visualizing Batch Sample {b_idx} ===")
            
            # === 1. DATA VISUALIZATION ===
            print("  Creating data visualization...")
            fig_data = visualize_data_and_poses(batch, b_idx)
            fig_data.savefig(output_dir / f"sample{b_idx}_01_data_and_poses.png", dpi=150, bbox_inches='tight')
            plt.close(fig_data)
            
            # === 2. FLOW VISUALIZATION (per template) ===
            print("  Creating GT flow visualizations...")
            for s_idx in range(min(3, S)):  # First 3 templates
                fig_flow = visualize_template_flow(outputs, batch, b_idx, s_idx)
                fig_flow.savefig(output_dir / f"sample{b_idx}_02_gt_flow_template{s_idx}.png", dpi=150, bbox_inches='tight')
                plt.close(fig_flow)
            
            # === 3. DETAILED PATCH FLOW ===
            print("  Creating detailed patch flow visualization...")
            fig_patch = visualize_patch_flow_detailed(outputs, batch, b_idx)
            fig_patch.savefig(output_dir / f"sample{b_idx}_03_gt_patch_flow_detail.png", dpi=150, bbox_inches='tight')
            plt.close(fig_patch)
        
        print(f"\n✓ All GT visualizations saved to: {output_dir}")
        print("\nIMPORTANT: Check sample*_01_data_and_poses.png:")
        print("  - POSITIVE templates should have the SMALLEST angles from query")
        print("  - If not, there's a bug in template selection!")
        return True
        
    except Exception as e:
        print(f"\n✗ Visualization FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
