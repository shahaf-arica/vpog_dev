"""
Flow Visualization for VPOG

Visualizes 16×16 pixel-level flow within patches with:
- Color-coded flow vectors
- Unseen pixel marking
- Confidence visualization
- Patch-level and pixel-level views
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import hsv_to_rgb
from typing import Optional, Tuple, List
import cv2


def flow_to_color(flow: np.ndarray, max_flow: Optional[float] = None) -> np.ndarray:
    """
    Convert optical flow to color image using HSV color space.
    
    Args:
        flow: Flow array [..., 2] where last dim is (dx, dy)
        max_flow: Maximum flow magnitude for normalization. If None, auto-computed.
        
    Returns:
        Color image [..., 3] in RGB with values in [0, 1]
    """
    # Get flow magnitude and angle
    dx = flow[..., 0]
    dy = flow[..., 1]
    
    mag = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    # Normalize to [0, 1]
    if max_flow is None:
        max_flow = np.percentile(mag, 99)  # Use 99th percentile to avoid outliers
    
    mag_norm = np.clip(mag / (max_flow + 1e-6), 0, 1)
    
    # Convert to HSV
    # Hue: angle (0 to 2π) -> (0 to 1)
    # Saturation: magnitude (normalized)
    # Value: always 1
    h = (angle + np.pi) / (2 * np.pi)  # [0, 1]
    s = mag_norm
    v = np.ones_like(mag)
    
    # Stack HSV
    hsv = np.stack([h, s, v], axis=-1)
    
    # Convert to RGB
    rgb = hsv_to_rgb(hsv)
    
    return rgb


def create_flow_wheel(size: int = 256) -> np.ndarray:
    """
    Create a color wheel showing flow direction encoding.
    
    Args:
        size: Size of the wheel image
        
    Returns:
        RGB image [size, size, 3]
    """
    # Create grid
    y, x = np.mgrid[-1:1:size*1j, -1:1:size*1j]
    
    # Compute angle and radius
    angle = np.arctan2(y, x)
    radius = np.sqrt(x**2 + y**2)
    
    # Create HSV
    h = (angle + np.pi) / (2 * np.pi)
    s = np.clip(radius, 0, 1)
    v = np.ones_like(radius)
    
    hsv = np.stack([h, s, v], axis=-1)
    rgb = hsv_to_rgb(hsv)
    
    # Mask outside circle
    mask = radius <= 1.0
    rgb = rgb * mask[..., None]
    
    return rgb


def visualize_patch_flow(
    query_img: np.ndarray,
    template_img: np.ndarray,
    flow: np.ndarray,
    confidence: np.ndarray,
    classification_probs: np.ndarray,
    patch_size: int = 16,
    conf_threshold: float = 0.5,
    top_k_patches: int = 20,
    figsize: Tuple[int, int] = (20, 10),
) -> plt.Figure:
    """
    Visualize patch-level flow with pixel-level detail.
    
    Args:
        query_img: Query image [H, W, 3]
        template_img: Template image [H, W, 3]
        flow: Flow array [Nq, 16, 16, 2] for single template
        confidence: Confidence array [Nq, 16, 16]
        classification_probs: Classification probabilities [Nq, Nt+1]
        patch_size: Size of each patch
        conf_threshold: Confidence threshold for visualization
        top_k_patches: Number of top patches to show in detail
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    H, W = query_img.shape[:2]
    grid_h, grid_w = H // patch_size, W // patch_size
    Nq = grid_h * grid_w
    
    # Get template assignment (exclude unseen token at index -1)
    template_probs = classification_probs[:, :-1]  # [Nq, Nt]
    unseen_probs = classification_probs[:, -1]  # [Nq]
    
    # Find patches with high confidence matches (not unseen)
    match_confidence = (1 - unseen_probs) * confidence.mean(axis=(1, 2))
    top_patches = np.argsort(match_confidence)[::-1][:top_k_patches]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Query image with patch grid
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(query_img)
    ax1.set_title("Query Image")
    ax1.axis('off')
    
    # Draw patch grid
    for i in range(grid_h + 1):
        ax1.axhline(i * patch_size, color='cyan', alpha=0.3, linewidth=0.5)
    for j in range(grid_w + 1):
        ax1.axvline(j * patch_size, color='cyan', alpha=0.3, linewidth=0.5)
    
    # Highlight top patches
    for rank, patch_idx in enumerate(top_patches[:5]):
        py, px = patch_idx // grid_w, patch_idx % grid_w
        rect = mpatches.Rectangle(
            (px * patch_size, py * patch_size),
            patch_size, patch_size,
            linewidth=2, edgecolor='yellow', facecolor='none'
        )
        ax1.add_patch(rect)
        ax1.text(
            px * patch_size + 2, py * patch_size + 12,
            f'{rank+1}', color='yellow', fontsize=10, weight='bold'
        )
    
    # 2. Template image
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(template_img)
    ax2.set_title("Template Image")
    ax2.axis('off')
    
    # 3. Flow color map (patch-level average)
    ax3 = fig.add_subplot(gs[0, 2])
    flow_avg = flow.mean(axis=(1, 2))  # [Nq, 2]
    flow_img = flow_avg.reshape(grid_h, grid_w, 2)
    flow_color = flow_to_color(flow_img, max_flow=1.0)  # max_flow in patches
    ax3.imshow(flow_color)
    ax3.set_title("Flow Direction (Patch Average)")
    ax3.axis('off')
    
    # 4. Confidence map
    ax4 = fig.add_subplot(gs[0, 3])
    conf_avg = confidence.mean(axis=(1, 2)).reshape(grid_h, grid_w)
    im = ax4.imshow(conf_avg, cmap='hot', vmin=0, vmax=1)
    ax4.set_title("Confidence (Patch Average)")
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, fraction=0.046)
    
    # 5. Unseen probability map
    ax5 = fig.add_subplot(gs[1, 0])
    unseen_map = unseen_probs.reshape(grid_h, grid_w)
    im = ax5.imshow(unseen_map, cmap='viridis', vmin=0, vmax=1)
    ax5.set_title("Unseen Probability")
    ax5.axis('off')
    plt.colorbar(im, ax=ax5, fraction=0.046)
    
    # 6. Flow wheel legend
    ax6 = fig.add_subplot(gs[1, 1])
    wheel = create_flow_wheel(256)
    ax6.imshow(wheel)
    ax6.set_title("Flow Color Wheel")
    ax6.axis('off')
    ax6.text(128, 20, 'Right', ha='center', color='white', weight='bold')
    ax6.text(128, 236, 'Left', ha='center', color='white', weight='bold')
    ax6.text(20, 128, 'Up', ha='center', color='white', weight='bold', rotation=90)
    ax6.text(236, 128, 'Down', ha='center', color='white', weight='bold', rotation=90)
    
    # 7-8. Detailed pixel-level flow for top 2 patches
    for detail_idx in range(min(2, len(top_patches))):
        patch_idx = top_patches[detail_idx]
        py, px = patch_idx // grid_w, patch_idx % grid_w
        
        ax_detail = fig.add_subplot(gs[1, 2 + detail_idx])
        
        # Get patch flow and confidence
        patch_flow = flow[patch_idx]  # [16, 16, 2]
        patch_conf = confidence[patch_idx]  # [16, 16]
        
        # Create visualization
        flow_color_patch = flow_to_color(patch_flow, max_flow=0.5)
        
        # Mark low confidence pixels (unseen)
        unseen_mask = patch_conf < conf_threshold
        flow_color_patch[unseen_mask] = [0.5, 0.5, 0.5]  # Gray for unseen
        
        ax_detail.imshow(flow_color_patch)
        ax_detail.set_title(f"Patch {detail_idx+1} ({py},{px})\nPixel-Level Flow")
        ax_detail.axis('off')
        
        # Add grid
        for i in range(17):
            ax_detail.axhline(i - 0.5, color='white', alpha=0.2, linewidth=0.5)
            ax_detail.axvline(i - 0.5, color='white', alpha=0.2, linewidth=0.5)
    
    # 9. Statistics
    ax9 = fig.add_subplot(gs[2, :])
    ax9.axis('off')
    
    stats_text = f"""
    Statistics:
    • Total patches: {Nq} ({grid_h}×{grid_w})
    • Average confidence: {confidence.mean():.3f}
    • Average unseen probability: {unseen_probs.mean():.3f}
    • Patches with conf > {conf_threshold}: {(confidence.mean(axis=(1,2)) > conf_threshold).sum()} ({(confidence.mean(axis=(1,2)) > conf_threshold).sum()/Nq*100:.1f}%)
    • Patches marked as unseen (prob > 0.5): {(unseen_probs > 0.5).sum()} ({(unseen_probs > 0.5).sum()/Nq*100:.1f}%)
    • Flow magnitude range: [{flow.min():.3f}, {flow.max():.3f}] patches
    • Mean flow magnitude: {np.sqrt((flow**2).sum(axis=-1)).mean():.3f} patches
    """
    ax9.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    return fig


def visualize_pixel_level_flow_detailed(
    query_patch: np.ndarray,
    template_patch: np.ndarray,
    flow: np.ndarray,
    confidence: np.ndarray,
    conf_threshold: float = 0.5,
    figsize: Tuple[int, int] = (16, 4),
) -> plt.Figure:
    """
    Detailed visualization of pixel-level flow within a single patch.
    
    Args:
        query_patch: Query patch image [16, 16, 3]
        template_patch: Template patch image [16, 16, 3]
        flow: Flow array [16, 16, 2]
        confidence: Confidence array [16, 16]
        conf_threshold: Confidence threshold for unseen marking
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # 1. Query patch
    axes[0].imshow(query_patch)
    axes[0].set_title("Query Patch (16×16)")
    axes[0].axis('off')
    
    # 2. Template patch
    axes[1].imshow(template_patch)
    axes[1].set_title("Template Patch (16×16)")
    axes[1].axis('off')
    
    # 3. Flow visualization with arrows
    axes[2].imshow(query_patch, alpha=0.5)
    
    # Draw flow arrows for visible pixels
    y_coords, x_coords = np.mgrid[0:16, 0:16]
    visible_mask = confidence >= conf_threshold
    
    # Scale flow for visualization (flow is in patch units, scale to pixels)
    flow_scaled = flow * 16  # Convert from patch units to pixels
    
    dx = flow_scaled[..., 0]
    dy = flow_scaled[..., 1]
    
    # Draw arrows for visible pixels
    if visible_mask.any():
        axes[2].quiver(
            x_coords[visible_mask], y_coords[visible_mask],
            dx[visible_mask], dy[visible_mask],
            confidence[visible_mask],
            cmap='hot', alpha=0.8, scale=50, width=0.003,
            headwidth=4, headlength=5
        )
    
    # Mark unseen pixels
    unseen_y, unseen_x = np.where(~visible_mask)
    axes[2].scatter(unseen_x, unseen_y, c='gray', s=20, marker='x', alpha=0.7)
    
    axes[2].set_title("Flow Arrows\n(Gray × = Unseen)")
    axes[2].set_xlim(-0.5, 15.5)
    axes[2].set_ylim(15.5, -0.5)
    axes[2].axis('off')
    
    # 4. Flow color + confidence overlay
    flow_color = flow_to_color(flow, max_flow=0.5)
    
    # Create composite: flow color with confidence alpha
    composite = flow_color.copy()
    
    # Mark unseen pixels as gray
    unseen_mask = confidence < conf_threshold
    composite[unseen_mask] = [0.5, 0.5, 0.5]
    
    # Overlay confidence as alpha on seen pixels
    seen_mask = ~unseen_mask
    alpha = np.clip(confidence, 0, 1)
    alpha[unseen_mask] = 0.5
    
    axes[3].imshow(composite)
    axes[3].set_title(f"Flow + Confidence\n(Gray = Unseen < {conf_threshold})")
    axes[3].axis('off')
    
    # Add grid to all
    for ax in axes:
        for i in range(17):
            ax.axhline(i - 0.5, color='white', alpha=0.2, linewidth=0.5)
            ax.axvline(i - 0.5, color='white', alpha=0.2, linewidth=0.5)
    
    plt.tight_layout()
    return fig


def visualize_correspondence_grid(
    query_img: np.ndarray,
    template_img: np.ndarray,
    flow: np.ndarray,
    classification_probs: np.ndarray,
    confidence: np.ndarray,
    patch_size: int = 16,
    conf_threshold: float = 0.5,
    stride: int = 2,
    figsize: Tuple[int, int] = (16, 8),
) -> plt.Figure:
    """
    Visualize correspondences as lines from query to template pixels.
    
    Args:
        query_img: Query image [H, W, 3]
        template_img: Template image [H, W, 3]
        flow: Flow array [Nq, 16, 16, 2]
        classification_probs: Classification probabilities [Nq, Nt+1]
        confidence: Confidence array [Nq, 16, 16]
        patch_size: Size of each patch
        conf_threshold: Confidence threshold
        stride: Sample every stride-th pixel for visualization
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    H, W = query_img.shape[:2]
    grid_h, grid_w = H // patch_size, W // patch_size
    
    # Get unseen probabilities
    unseen_probs = classification_probs[:, -1]  # [Nq]
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create side-by-side image
    combined = np.concatenate([query_img, template_img], axis=1)
    ax.imshow(combined)
    
    # Draw correspondences
    num_correspondences = 0
    offset_x = W  # Template image starts at x = W
    
    for patch_idx in range(len(flow)):
        py, px = patch_idx // grid_w, patch_idx % grid_w
        
        # Skip if patch is likely unseen
        if unseen_probs[patch_idx] > 0.5:
            continue
        
        patch_flow = flow[patch_idx]  # [16, 16, 2]
        patch_conf = confidence[patch_idx]  # [16, 16]
        
        # Sample pixels with stride
        for i in range(0, 16, stride):
            for j in range(0, 16, stride):
                if patch_conf[i, j] < conf_threshold:
                    continue
                
                # Query pixel position
                qy = py * patch_size + i
                qx = px * patch_size + j
                
                # Template pixel position (using flow)
                # Flow is template -> query, so we need to invert
                # Or interpret as query -> template depending on convention
                # Assuming flow is query -> template
                ty = qy + patch_flow[i, j, 1] * patch_size
                tx = qx + patch_flow[i, j, 0] * patch_size + offset_x
                
                # Draw line
                color = plt.cm.hot(patch_conf[i, j])
                ax.plot([qx, tx], [qy, ty], color=color, alpha=0.5, linewidth=0.5)
                ax.scatter([qx], [qy], c='cyan', s=2, alpha=0.7)
                ax.scatter([tx], [ty], c='yellow', s=2, alpha=0.7)
                
                num_correspondences += 1
    
    ax.set_title(f"Pixel Correspondences (n={num_correspondences})\nCyan: Query, Yellow: Template")
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def create_flow_animation_frames(
    query_img: np.ndarray,
    template_img: np.ndarray,
    flow: np.ndarray,
    confidence: np.ndarray,
    patch_size: int = 16,
    num_frames: int = 10,
) -> List[np.ndarray]:
    """
    Create animation frames showing flow warping from template to query.
    
    Args:
        query_img: Query image [H, W, 3]
        template_img: Template image [H, W, 3]
        flow: Flow array [Nq, 16, 16, 2]
        confidence: Confidence array [Nq, 16, 16]
        patch_size: Size of each patch
        num_frames: Number of interpolation frames
        
    Returns:
        List of frames [H, W, 3]
    """
    H, W = query_img.shape[:2]
    grid_h, grid_w = H // patch_size, W // patch_size
    
    frames = []
    
    for t in np.linspace(0, 1, num_frames):
        # Create warped image
        warped = np.zeros_like(query_img)
        weight_map = np.zeros((H, W, 1))
        
        for patch_idx in range(len(flow)):
            py, px = patch_idx // grid_w, patch_idx % grid_w
            
            patch_flow = flow[patch_idx]  # [16, 16, 2]
            patch_conf = confidence[patch_idx]  # [16, 16]
            
            for i in range(16):
                for j in range(16):
                    qy = py * patch_size + i
                    qx = px * patch_size + j
                    
                    if qy >= H or qx >= W:
                        continue
                    
                    # Interpolated position
                    ty = qy + patch_flow[i, j, 1] * patch_size * t
                    tx = qx + patch_flow[i, j, 0] * patch_size * t
                    
                    # Sample from template
                    ty_int, tx_int = int(ty), int(tx)
                    if 0 <= ty_int < H and 0 <= tx_int < W:
                        weight = patch_conf[i, j]
                        warped[qy, qx] += template_img[ty_int, tx_int] * weight
                        weight_map[qy, qx] += weight
        
        # Normalize
        weight_map = np.maximum(weight_map, 1e-6)
        warped = warped / weight_map
        
        # Blend with query image
        alpha = t
        frame = (1 - alpha) * template_img + alpha * warped
        frames.append(frame.astype(np.uint8))
    
    return frames


if __name__ == '__main__':
    # Test with dummy data
    np.random.seed(42)
    
    # Create dummy images
    query_img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    template_img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    
    # Create dummy flow and confidence
    Nq = 14 * 14  # 196 patches
    flow = np.random.randn(Nq, 16, 16, 2) * 0.2
    confidence = np.random.rand(Nq, 16, 16)
    classification_probs = np.random.rand(Nq, 197)  # 196 template patches + 1 unseen
    classification_probs = classification_probs / classification_probs.sum(axis=1, keepdims=True)
    
    print("Creating patch-level flow visualization...")
    fig1 = visualize_patch_flow(
        query_img, template_img, flow, confidence,
        classification_probs, patch_size=16
    )
    fig1.savefig('/tmp/test_patch_flow.png', dpi=150, bbox_inches='tight')
    print("Saved to /tmp/test_patch_flow.png")
    
    print("\nCreating detailed pixel-level visualization...")
    query_patch = query_img[32:48, 32:48]
    template_patch = template_img[32:48, 32:48]
    patch_flow = flow[16]  # Arbitrary patch
    patch_conf = confidence[16]
    
    fig2 = visualize_pixel_level_flow_detailed(
        query_patch, template_patch, patch_flow, patch_conf
    )
    fig2.savefig('/tmp/test_pixel_flow.png', dpi=150, bbox_inches='tight')
    print("Saved to /tmp/test_pixel_flow.png")
    
    print("\nTest complete!")
