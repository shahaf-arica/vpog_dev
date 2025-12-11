"""
Label Visualization for VPOG Training
Visualizes classification and regression labels to verify correctness
Uses GPU acceleration with torch.inference_mode() for efficiency
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import time
from pathlib import Path
from typing import Optional, Tuple
import sys

# Add project root
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from training.dataloader.vpog_dataset import VPOGBatch


class LabelVisualizer:
    """Visualizes VPOG labels with GPU acceleration"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        print(f"LabelVisualizer using device: {self.device}")
    
    @torch.inference_mode()
    def visualize_batch_labels(
        self,
        batch: VPOGBatch,
        save_path: Path,
        num_samples: int = 2,
        num_random_patches: int = 8,
    ):
        """
        Visualize labels for a batch with timing information.
        
        Args:
            batch: VPOGBatch containing images and labels
            save_path: Path to save visualization
            num_samples: Number of samples to visualize
            num_random_patches: Number of random visible patches to show
        """
        start_time = time.time()
        
        # Move batch to device for GPU computation
        batch = batch.to(self.device)
        
        num_samples = min(num_samples, batch.images.shape[0])
        num_templates = batch.images.shape[1] - 1  # Exclude query
        
        # Create figure for classification visualization
        fig1 = self.visualize_classification_labels(
            batch, num_samples, num_templates
        )
        
        # Create figure for regression visualization
        fig2 = self.visualize_regression_labels(
            batch, num_samples, num_templates, num_random_patches
        )
        
        compute_time = time.time() - start_time
        
        # Add timing to titles
        fig1.suptitle(
            f'VPOG Classification Labels (Patch Visibility)\n'
            f'Computation time: {compute_time:.3f}s | Device: {self.device}',
            fontsize=14, fontweight='bold'
        )
        
        fig2.suptitle(
            f'VPOG Regression Labels (Flow Vectors)\n'
            f'Computation time: {compute_time:.3f}s | Device: {self.device}',
            fontsize=14, fontweight='bold'
        )
        
        # Save figures
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(save_path.parent / f"{save_path.stem}_classification.png", 
                     dpi=150, bbox_inches='tight')
        fig2.savefig(save_path.parent / f"{save_path.stem}_regression.png", 
                     dpi=150, bbox_inches='tight')
        
        plt.close(fig1)
        plt.close(fig2)
        
        print(f"✓ Label visualizations saved:")
        print(f"  - Classification: {save_path.parent / f'{save_path.stem}_classification.png'}")
        print(f"  - Regression: {save_path.parent / f'{save_path.stem}_regression.png'}")
        print(f"  - Total computation time: {compute_time:.3f}s")
        
        return compute_time
    
    @torch.inference_mode()
    def visualize_classification_labels(
        self,
        batch: VPOGBatch,
        num_samples: int,
        num_templates: int,
    ) -> plt.Figure:
        """
        Visualize patch visibility classification as query-template PAIRS with depth maps.
        Each row shows one template pair: Query RGB | Query Depth | Template RGB | Template Depth
        
        Green box = visible patch
        """
        # Get dimensions
        B = batch.centered_rgb.shape[0] if batch.centered_rgb is not None else batch.images.shape[0]
        S = num_templates
        
        # Get patch dimensions from images
        H, W = 224, 224  # Model input size
        patch_visibility = batch.patch_visibility  # [B, S, H_p, W_p]
        H_p, W_p = patch_visibility.shape[2], patch_visibility.shape[3]
        patch_size = H // H_p
        
        # Figure: rows = samples * templates, cols = 5 (query_rgb, query_depth, template_rgb, template_depth, projected_query_depth)
        num_rows = num_samples * num_templates
        fig, axes = plt.subplots(num_rows, 5, figsize=(20, 4 * num_rows))
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        
        row_idx = 0
        for b in range(num_samples):
            # Get CENTERED query image WITH background (model input)
            if batch.centered_rgb is not None:
                query_img_raw = batch.centered_rgb[b]  # [3, 224, 224] unnormalized
                query_img = query_img_raw.permute(1, 2, 0).cpu().numpy()  # [224, 224, 3]
                query_img = np.clip(query_img, 0, 1)
            else:
                query_img = self._denormalize_image(batch.images[b, 0])  # Fallback
                query_img = query_img.cpu().numpy()
            
            for s in range(num_templates):
                # Get template image
                template_img = self._denormalize_image(batch.images[b, s + 1])
                template_img = template_img.cpu().numpy()
                
                # Get patch visibility for this query-template pair
                patch_vis = patch_visibility[b, s].cpu().numpy()  # [H_p, W_p]
                
                # LEFT: Query with boxes
                ax_query = axes[row_idx, 0]
                ax_query.imshow(query_img)
                
                # Draw boxes on query
                for i in range(H_p):
                    for j in range(W_p):
                        template_visible = patch_vis[i, j] > 0.5
                        
                        if not template_visible:
                            # No box if not visible in template
                            continue
                        
                        # Patch coordinates
                        x = j * patch_size
                        y = i * patch_size
                        
                        # For query: green if bidirectional, red if only template visible
                        # patch_vis already encodes this: 1.0 = bidirectional, some logic determines this
                        # Actually patch_visibility is: visible in template AND visible in query (bidirectional)
                        # So if patch_vis > 0.5, it means visible in BOTH
                        
                        color = 'green'  # Visible in both
                        alpha = 0.4
                        linewidth = 2
                        
                        rect = Rectangle(
                            (x, y), patch_size, patch_size,
                            linewidth=linewidth,
                            edgecolor=color,
                            facecolor='none',
                        )
                        ax_query.add_patch(rect)
                
                # RIGHT: Template with boxes (only green - all visible by definition)
                ax_template = axes[row_idx, 1]
                ax_template.imshow(template_img)
                
                # Draw boxes on template (only where patch_vis > 0.5)
                for i in range(H_p):
                    for j in range(W_p):
                        template_visible = patch_vis[i, j] > 0.5
                        
                        if not template_visible:
                            continue
                        
                        x = j * patch_size
                        y = i * patch_size
                        
                        color = 'green'
                        linewidth = 2
                        
                        rect = Rectangle(
                            (x, y), patch_size, patch_size,
                            linewidth=linewidth,
                            edgecolor=color,
                            facecolor='none',
                        )
                        ax_template.add_patch(rect)
                
                # Count visible patches
                num_visible = (patch_vis > 0.5).sum()
                total_patches = H_p * W_p
                
                template_idx = batch.template_indices[b, s].item()
                is_pos = batch.template_types[b, s].item() == 0
                type_str = "POS" if is_pos else "NEG"
                title_color = 'green' if is_pos else 'red'
                
                ax_query.set_title(
                    f'Sample {b}, T{s} ({type_str}) idx={template_idx}\n'
                    f'Query RGB\nVisible: {num_visible}/{total_patches}',
                    fontsize=10, color=title_color, fontweight='bold'
                )
                ax_query.axis('off')
                
                # Add query depth visualization (column 2)
                ax_query_depth = axes[row_idx, 2]
                if hasattr(batch, 'query_depth') and batch.query_depth is not None:
                    query_depth = batch.query_depth[b].cpu().numpy()
                    vmax = np.percentile(query_depth[query_depth > 0], 95) if (query_depth > 0).any() else 1
                    im = ax_query_depth.imshow(query_depth, cmap='plasma', vmin=0, vmax=vmax)
                    plt.colorbar(im, ax=ax_query_depth, fraction=0.046, pad=0.04)
                    ax_query_depth.set_title(f'Query Depth\nRange: [{query_depth.min():.0f}, {query_depth.max():.0f}]mm', fontsize=9)
                else:
                    ax_query_depth.text(0.5, 0.5, 'No Depth', ha='center', va='center', transform=ax_query_depth.transAxes)
                    ax_query_depth.set_title('Query Depth (N/A)', fontsize=9)
                ax_query_depth.axis('off')
                
                ax_template.set_title(
                    f'Template RGB',
                    fontsize=10, color=title_color, fontweight='bold'
                )
                ax_template.axis('off')
                
                # Add template depth visualization (column 3)
                ax_template_depth = axes[row_idx, 3]
                if hasattr(batch, 'template_depth') and batch.template_depth is not None:
                    template_depth_np = batch.template_depth[b, s].cpu().numpy()
                    vmax = np.percentile(template_depth_np[template_depth_np > 0], 95) if (template_depth_np > 0).any() else 1
                    im = ax_template_depth.imshow(template_depth_np, cmap='plasma', vmin=0, vmax=vmax)
                    plt.colorbar(im, ax=ax_template_depth, fraction=0.046, pad=0.04)
                    ax_template_depth.set_title(f'Template Depth\nRange: [{template_depth_np.min():.0f}, {template_depth_np.max():.0f}]mm', fontsize=9)
                else:
                    ax_template_depth.text(0.5, 0.5, 'No Depth', ha='center', va='center', transform=ax_template_depth.transAxes)
                    ax_template_depth.set_title('Template Depth (N/A)', fontsize=9)
                    template_depth_np = None
                ax_template_depth.axis('off')
                
                # Add projected query depth visualization (column 4)
                ax_projected = axes[row_idx, 4]
                if (hasattr(batch, 'query_depth') and batch.query_depth is not None and 
                    template_depth_np is not None):
                    # Project query depth to template camera
                    projected_depth = self._project_query_to_template(
                        query_depth=batch.query_depth[b].cpu().numpy(),
                        template_depth=template_depth_np,
                        query_K=batch.K[b, 0].cpu().numpy(),
                        template_K=batch.K[b, s + 1].cpu().numpy(),
                        query_pose=batch.poses[b, 0].cpu().numpy(),
                        template_pose=batch.poses[b, s + 1].cpu().numpy(),
                    )
                    n_proj = (projected_depth > 0).sum()
                    
                    if n_proj > 0:
                        vmax = np.percentile(projected_depth[projected_depth > 0], 95)
                    else:
                        vmax = 1
                    
                    im = ax_projected.imshow(projected_depth, cmap='plasma', vmin=0, vmax=vmax)
                    plt.colorbar(im, ax=ax_projected, fraction=0.046, pad=0.04)
                    ax_projected.set_title(f'Query→Template Projection\nProjected pixels: {n_proj}\nRange: [{projected_depth.min():.0f}, {projected_depth.max():.0f}]mm', fontsize=9)
                else:
                    ax_projected.text(0.5, 0.5, 'No Projection', ha='center', va='center', transform=ax_projected.transAxes)
                    ax_projected.set_title('Query Projected (N/A)', fontsize=9)
                ax_projected.axis('off')
                
                row_idx += 1
        
        plt.tight_layout()
        return fig
    
    @torch.inference_mode()
    def visualize_regression_labels(
        self,
        batch: VPOGBatch,
        num_samples: int,
        num_templates: int,
        num_random_patches: int,
    ) -> plt.Figure:
        """
        Visualize flow regression labels on random visible patches.
        Shows query-template pairs with flow arrows.
        """
        # Get dimensions
        B, S_plus_1, C, H, W = batch.images.shape
        S = S_plus_1 - 1
        
        # Get patch info
        flows = batch.flows  # [B, S, H_p, W_p, 2]
        patch_visibility = batch.patch_visibility  # [B, S, H_p, W_p]
        H_p, W_p = flows.shape[2], flows.shape[3]
        patch_size = H // H_p
        
        # Figure: each row = one random patch, columns = query + template for each sample
        fig, axes = plt.subplots(
            num_random_patches,
            num_samples * 2,  # query + template for each sample
            figsize=(4 * num_samples * 2, 4 * num_random_patches)
        )
        
        if num_random_patches == 1:
            axes = axes.reshape(1, -1)
        if num_samples == 1:
            axes = axes.reshape(num_random_patches, 2)
        
        for b in range(num_samples):
            # Get query image
            query_img = self._denormalize_image(batch.images[b, 0])  # [H, W, 3]
            
            # Find all visible patches across all templates
            all_visible_patches = []
            for s in range(num_templates):
                patch_vis = patch_visibility[b, s].cpu().numpy()  # [H_p, W_p]
                visible_coords = np.argwhere(patch_vis > 0.5)  # [[i, j], ...]
                
                for coord in visible_coords:
                    all_visible_patches.append((s, coord[0], coord[1]))
            
            # Sample random visible patches
            if len(all_visible_patches) == 0:
                print(f"Warning: No visible patches found for sample {b}")
                continue
            
            num_to_sample = min(num_random_patches, len(all_visible_patches))
            sampled_indices = np.random.choice(
                len(all_visible_patches), 
                size=num_to_sample, 
                replace=False
            )
            
            for patch_idx in range(num_to_sample):
                s, i, j = all_visible_patches[sampled_indices[patch_idx]]
                
                # Get template image
                template_img = self._denormalize_image(batch.images[b, s + 1])  # [H, W, 3]
                
                # Get flow for this patch
                flow = flows[b, s, i, j].cpu().numpy()  # [2] - delta in patch coords
                
                # Patch center in image coordinates
                patch_center_x = (j + 0.5) * patch_size
                patch_center_y = (i + 0.5) * patch_size
                
                # Flow endpoint in image coordinates (flow is in patch units)
                flow_end_x = patch_center_x + flow[0] * patch_size
                flow_end_y = patch_center_y + flow[1] * patch_size
                
                # Column indices for this sample
                col_query = b * 2
                col_template = b * 2 + 1
                
                # Show query patch
                ax_query = axes[patch_idx, col_query]
                ax_query.imshow(query_img.cpu().numpy())
                
                # Draw patch box on query
                rect = Rectangle(
                    (j * patch_size, i * patch_size),
                    patch_size, patch_size,
                    linewidth=2, edgecolor='blue', facecolor='none'
                )
                ax_query.add_patch(rect)
                
                # Draw center point
                ax_query.plot(patch_center_x, patch_center_y, 'bo', markersize=8)
                
                ax_query.set_xlim(max(0, j * patch_size - patch_size), 
                                 min(W, (j + 1) * patch_size + patch_size))
                ax_query.set_ylim(min(H, (i + 1) * patch_size + patch_size),
                                 max(0, i * patch_size - patch_size))
                ax_query.axis('off')
                
                template_idx = batch.template_indices[b, s].item()
                is_pos = batch.template_types[b, s].item() == 0
                type_str = "POS" if is_pos else "NEG"
                
                if patch_idx == 0:
                    ax_query.set_title(
                        f'Sample {b}: Query\nPatch [{i},{j}]',
                        fontsize=10, fontweight='bold'
                    )
                else:
                    ax_query.set_title(f'Patch [{i},{j}]', fontsize=10)
                
                # Show template patch with flow arrow
                ax_template = axes[patch_idx, col_template]
                ax_template.imshow(template_img.cpu().numpy())
                
                # Draw patch box on template (same location)
                rect = Rectangle(
                    (j * patch_size, i * patch_size),
                    patch_size, patch_size,
                    linewidth=2, edgecolor='orange', facecolor='none'
                )
                ax_template.add_patch(rect)
                
                # Draw flow arrow from patch center to predicted position
                arrow = FancyArrowPatch(
                    (patch_center_x, patch_center_y),
                    (flow_end_x, flow_end_y),
                    arrowstyle='->', mutation_scale=20,
                    linewidth=2, color='red'
                )
                ax_template.add_patch(arrow)
                
                # Draw start and end points
                ax_template.plot(patch_center_x, patch_center_y, 'go', markersize=8)
                ax_template.plot(flow_end_x, flow_end_y, 'ro', markersize=8)
                
                ax_template.set_xlim(max(0, j * patch_size - patch_size), 
                                    min(W, (j + 1) * patch_size + patch_size))
                ax_template.set_ylim(min(H, (i + 1) * patch_size + patch_size),
                                    max(0, i * patch_size - patch_size))
                ax_template.axis('off')
                
                if patch_idx == 0:
                    ax_template.set_title(
                        f'T{s} ({type_str}) idx={template_idx}\n'
                        f'Flow: Δx={flow[0]:.2f}, Δy={flow[1]:.2f} patches',
                        fontsize=10, fontweight='bold'
                    )
                else:
                    ax_template.set_title(
                        f'T{s}: Δx={flow[0]:.2f}, Δy={flow[1]:.2f}',
                        fontsize=10
                    )
        
        plt.tight_layout()
        return fig
    
    def _denormalize_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize image from [-1, 1] to [0, 1]"""
        # img_tensor: [3, H, W]
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], 
                           device=img_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                          device=img_tensor.device).view(3, 1, 1)
        
        img = img_tensor * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0)  # [H, W, 3]
        
        return img
    
    def _project_query_to_template(
        self,
        query_depth: np.ndarray,
        template_depth: np.ndarray,
        query_K: np.ndarray,
        template_K: np.ndarray,
        query_pose: np.ndarray,
        template_pose: np.ndarray,
    ) -> np.ndarray:
        """
        Project query depth map to template camera view.
        
        Args:
            query_depth: Query depth map (H, W) in meters
            template_depth: Template depth map (H, W) in meters  
            query_K: Query intrinsics (3, 3)
            template_K: Template intrinsics (3, 3)
            query_pose: Query pose TWO (4, 4)
            template_pose: Template pose TWO (4, 4)
            
        Returns:
            Projected depth map (H, W) showing query points in template camera
        """
        H, W = query_depth.shape
        
        # Create pixel grid
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        pixels = np.stack([u, v], axis=-1)  # (H, W, 2)
        
        # Get valid depth pixels
        valid_mask = query_depth > 0
        n_valid = valid_mask.sum()
        
        if n_valid == 0:
            return np.zeros((H, W), dtype=np.float32)
        
        # Unproject query pixels to 3D
        fx, fy = query_K[0, 0], query_K[1, 1]
        cx, cy = query_K[0, 2], query_K[1, 2]
        
        x = (pixels[..., 0] - cx) * query_depth / fx
        y = (pixels[..., 1] - cy) * query_depth / fy
        z = query_depth
        
        points_query = np.stack([x, y, z], axis=-1)  # (H, W, 3)
        
        # Transform from query to world to template
        # query_pose and template_pose are TWO (world-to-object)
        # We need: points_query_cam -> points_world -> points_template_cam
        # TWO = T_world_to_object = T_object_to_world^{-1}
        # So: T_object_to_world = TWO^{-1}
        from src.custom_megapose.transform import Transform
        T_query_cam_to_world = Transform(query_pose).inverse()  # TWO^{-1}
        T_world_to_template_cam = Transform(template_pose)  # TWO
        T_query_to_template = T_world_to_template_cam * T_query_cam_to_world
        T_mat = T_query_to_template.toHomogeneousMatrix()
        
        # Apply transformation
        points_flat = points_query.reshape(-1, 3)
        valid_flat = valid_mask.reshape(-1)
        
        points_hom = np.concatenate([points_flat, np.ones((points_flat.shape[0], 1))], axis=1)
        points_template_hom = (T_mat @ points_hom.T).T
        points_template = points_template_hom[:, :3]
        
        # Project to template image
        x_t = points_template[:, 0]
        y_t = points_template[:, 1]
        z_t = points_template[:, 2]
        
        fx_t, fy_t = template_K[0, 0], template_K[1, 1]
        cx_t, cy_t = template_K[0, 2], template_K[1, 2]
        
        # Avoid division by zero
        z_t_safe = np.where(z_t > 0, z_t, 1e-6)
        u_t = (x_t / z_t_safe) * fx_t + cx_t
        v_t = (y_t / z_t_safe) * fy_t + cy_t
        
        # Create output depth map
        projected_depth = np.zeros((H, W), dtype=np.float32)
        
        # Fill in projected depths
        for idx in range(len(u_t)):
            if not valid_flat[idx]:
                continue
            if z_t[idx] <= 0:
                continue
                
            u_int = int(np.round(u_t[idx]))
            v_int = int(np.round(v_t[idx]))
            
            if 0 <= u_int < W and 0 <= v_int < H:
                # Take closest depth (highest priority to closer points)
                if projected_depth[v_int, u_int] == 0 or z_t[idx] < projected_depth[v_int, u_int]:
                    projected_depth[v_int, u_int] = z_t[idx]
        
        return projected_depth


def visualize_dataset_labels(
    dataset_name: str = "gso",
    batch_size: int = 2,
    num_batches: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Test label visualization on VPOG dataset"""
    from training.dataloader.vpog_dataset import VPOGTrainDataset
    from torch.utils.data import DataLoader
    
    print("=" * 80)
    print("VPOG Label Visualization Test")
    print("=" * 80)
    
    # Initialize dataset
    print(f"\n✓ Initializing VPOGTrainDataset for {dataset_name}...")
    
    # Template configuration
    template_config = {
        'dir': 'datasets/templates',
        'level_templates': 2,
        'pose_distribution': 'all',
    }
    
    dataset = VPOGTrainDataset(
        root_dir="datasets",
        dataset_name=dataset_name,
        template_config=template_config,
        num_positive_templates=3,
        num_negative_templates=2,
        min_negative_angle_deg=90.0,
        patch_size=16,
        image_size=224,
        seed=2,  # Fixed seed for reproducibility
    )
    
    print(f"✓ Creating DataLoader...")
    from src.utils.dataloader import NoneFilteringDataLoader
    dataloader = NoneFilteringDataLoader(
        dataset.web_dataloader.datapipeline,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    
    # Initialize visualizer
    visualizer = LabelVisualizer(device=device)
    
    # Load one batch
    print(f"\n✓ Loading batch...")
    batch = next(iter(dataloader))
    
    if batch is None:
        print(f"✗ Batch is None!")
        return
    
    print(f"✓ Batch loaded: {batch.images.shape[0]} samples")
    
    # Visualize labels
    save_path = Path("tmp/vpog_dataset_test/labels_batch_0.png")
    compute_time = visualizer.visualize_batch_labels(
        batch=batch,
        save_path=save_path,
        num_samples=min(2, batch.images.shape[0]),
        num_random_patches=8,
    )
    
    print("\n" + "=" * 80)
    print("✓ Label visualization test completed!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize VPOG labels")
    parser.add_argument("--dataset", type=str, default="gso", help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_batches", type=int, default=1, help="Number of batches")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    visualize_dataset_labels(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        device=args.device,
    )
