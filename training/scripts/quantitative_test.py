#!/usr/bin/env python3
"""
Quantitative Test Script for VPOG Model

Visualizes the complete inference pipeline:
1. Classification head: Top-K template selection based on logit scores
2. Query-template patch matching with colored correspondence lines
3. Top-H coupled patches with dense flow visualization
4. Final pixel-level correspondences from dense flow (above threshold)

Usage:
    python training/scripts/quantitative_test.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --dataset ycbv \
        --use_detections false \
        --output_dir ./visualizations \
        --top_k 3 \
        --top_h 10 \
        --num_samples 10
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
import matplotlib.cm as cm
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F

# GigaPose imports
from src.utils.logging import get_logger
from src.megapose.datasets.scene_dataset import SceneObservation
from src.utils.dataloader import NoneFilteringDataLoader

# Training imports
from training.lightning_module import VPOGLightningModule
from training.dataloader.vpog_dataset import VPOGDataset, VPOGBatch
from hydra import initialize, compose
from omegaconf import OmegaConf
from training.train import build_vpog_model_from_config, create_vpog_model

# Inference API imports
from vpog.inference.predictor import VPOGPredictor
from vpog.inference.pose_estimator import PoseEstimator


logger = get_logger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, hydra_dir: str) -> VPOGLightningModule:
    """Load trained VPOG model from checkpoint.
    
    Note: This function loads only the model weights, not the full training state.
    The model is loaded in eval mode for inference.
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # if 'state_dict' not in checkpoint:
    #     raise ValueError("Checkpoint does not contain state_dict")
    
    # # Extract model config from checkpoint directory structure
    # # The checkpoint is typically in: outputs/exp_name/runs/date/time/checkpoints/file.ckpt
    # # We need to load the hydra config from the run directory
    # checkpoint_path_obj = Path(checkpoint_path)
    
    # # Navigate up to find the run directory that contains .hydra config
    # run_dir = checkpoint_path_obj.parent.parent  # Go up from checkpoints/
    # hydra_dir = run_dir / '.hydra'
    
    # if not hydra_dir.exists():
    #     # Try one more level up (for different directory structures)
    #     run_dir = checkpoint_path_obj.parent.parent.parent
    #     hydra_dir = run_dir / '.hydra'
    
    # if not hydra_dir.exists():
    #     raise ValueError(
    #         f"Could not find .hydra config directory. "
    #         f"Searched in: {checkpoint_path_obj.parent.parent / '.hydra'} "
    #         f"and {checkpoint_path_obj.parent.parent.parent / '.hydra'}"
    #     )
    
    # logger.info(f"Loading config from: {hydra_dir}")
    
    # Load the Hydra config
    config_path = hydra_dir / 'config.yaml'
    cfg = OmegaConf.load(config_path)
    
    # Build model from config (same as in train.py)
    
    # We can't use create_vpog_model as it creates the full lightning module
    # Instead, just build the core model
    vpog_model = build_vpog_model_from_config(cfg)
    
    # Create a minimal lightning module just for inference
    # We don't need losses or optimizers for inference
    from training.lightning_module import LossWeights
    from hydra.utils import instantiate
    
    # Dummy losses (not used during inference)
    dummy_loss = torch.nn.MSELoss()
    
    loss_weights = LossWeights(
        cls=1.0,
        dense=1.0,
        invis_reg=1.0,
        center=1.0,
    )
    
    # Create lightning module with minimal config
    lightning_module = VPOGLightningModule(
        model=vpog_model,
        cls_loss=dummy_loss,
        dense_flow_loss=dummy_loss,
        lr=1e-4,
        weight_decay=1e-2,
        loss_weights=loss_weights,
        enable_pose_eval=False,  # Disable for inference
    )
    
    # Load state dict
    state_dict = checkpoint['state_dict']
    lightning_module.load_state_dict(state_dict)
    
    logger.info("Model loaded successfully")
    logger.info(f"  Global step: {checkpoint.get('global_step', 'N/A')}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return lightning_module


def setup_validation_dataloader(
    dataset_name: str,
    root_dir: str = "/strg/E/shared-data/NOPE/datasets",
    batch_size: int = 1,
    num_workers: int = 0,
) -> NoneFilteringDataLoader:
    """Setup validation dataloader for BOP dataset."""
    logger.info(f"Setting up validation dataloader for {dataset_name}")
    
    # Template config for validation
    # VPOGDataset will convert this dict to an object before passing to TemplateDataset
    template_config = {
        'dir': f'{root_dir}/templates',
        'level_templates': 1,
        'pose_distribution': 'all',
        'scale_factor': 1.0, # No scaling during validation
        'num_templates': 162,  # Standard number of templates per object
        'pose_name': 'object_poses/OBJECT_ID.npy',  # Template pose file pattern
    }
    
    dataset = VPOGDataset(
        root_dir=root_dir,
        dataset_name=dataset_name,
        template_config=template_config,
        mode='val',
        num_positive_templates=162,
        # num_negative_templates=0,
        patch_size=16,
        image_size=224,
        depth_scale=10.0,
        # seed=42,
    )
    
    dataloader = NoneFilteringDataLoader(
        dataset.web_dataloader,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        shuffle=False,
        pin_memory=False,
    )
    
    logger.info(f"  Dataset size: {len(dataset)}")
    logger.info(f"  Batch size: {batch_size}")
    
    return dataloader


def run_inference_on_batch(
    predictor: VPOGPredictor,
    batch: VPOGBatch,
    top_k_templates: int = 3,
    weight_thresh: float = 0.5,
) -> Dict:
    """
    Run VPOGPredictor on batch.
    
    Args:
        predictor: VPOGPredictor instance
        batch: VPOGBatch with data
        top_k_templates: Number of top templates to return
        weight_thresh: Weight threshold for refined correspondences
    
    Returns:
        results: List[Dict] from predictor (length B)
    """
    B = batch.images.shape[0]
    
    # Extract data for predictor
    query_images = batch.images[:, 0]        # [B, 3, H, W]
    template_images = batch.images[:, 1:]     # [B, S, 3, H, W]
    template_poses = batch.poses[:, 1:]       # [B, S, 4, 4]
    
    # Run predictor
    results = predictor.predict_correspondences_on_query(
        image_query=query_images,
        template_images=template_images,
        template_poses=template_poses,
        top_k_templates=top_k_templates,
        weight_thresh=weight_thresh,
    )
    
    return results


def estimate_pose_from_predictor_output(
    correspondence_dict: Dict,
    template_depth: torch.Tensor,
    template_K: torch.Tensor,
    template_pose: torch.Tensor,
    query_K_original: torch.Tensor,
    M_query: torch.Tensor,
    scale_factor: float = 10.0,
    use_refined: bool = True,
) -> Tuple[np.ndarray, int, float]:
    """
    Estimate pose from VPOGPredictor correspondence output.
    
    Args:
        correspondence_dict: Dict from predictor for one template
        template_depth: Template depth [H, W]
        template_K: Template intrinsics [3, 3]
        template_pose: Template pose [4, 4]
        query_K_original: Query intrinsics in original image [3, 3]
        M_query: Crop transformation [3, 3]
        scale_factor: CAD scale factor (default 10.0)
        use_refined: Use refined correspondences if available
    
    Returns:
        pose: [4, 4] estimated pose
        num_inliers: Number of RANSAC inliers
        score: Inlier ratio
    """
    pose_estimator = PoseEstimator(
        ransac_threshold=8.0,
        ransac_iterations=1000,
        min_inliers=4,
    )
    
    return pose_estimator.estimate_pose(
        correspondences=correspondence_dict,
        template_depth=template_depth,
        template_K=template_K,
        template_pose=template_pose,
        query_K_original=query_K_original,
        M_query=M_query,
        scale_factor=scale_factor,
        use_refined=use_refined,
    )



def visualize_inference_pipeline_coarse(
    batch: VPOGBatch,
    results: List[Dict],
    sample_id: int,
    output_dir: Path,
    max_coarse_lines: int = 50,
    max_vis_on_line: int = 4,
):
    """
    Visualize coarse patch correspondences for all templates.
    
    Supports high top_k_templates by stacking templates in multiple rows.
    
    Args:
        batch: VPOGBatch with images and metadata (batch_size=1)
        results: List[Dict] from VPOGPredictor.predict_correspondences_on_query()
        sample_id: Sample identifier for filename
        output_dir: Directory to save visualizations
        max_coarse_lines: Maximum coarse correspondences to draw per template
        max_vis_on_line: Maximum templates to show per row
    """
    b = 0  # Always 0 for batch_size=1
    result_dict = results[b]
    
    # Extract template info
    template_indices = list(result_dict.keys())
    K = len(template_indices)
    
    if K == 0:
        logger.warning(f"No templates found for sample {b}")
        return
    
    # Calculate grid layout
    num_cols = min(max_vis_on_line, K) + 2  # +2 for full image and query
    num_template_rows = (K + max_vis_on_line - 1) // max_vis_on_line  # Ceiling division
    num_rows = 1 + num_template_rows * 2  # 1 for images, 2 per template row (templates + correspondences)
    
    img_size = 224
    
    # Create figure
    fig = plt.figure(figsize=(5 * num_cols, 5 * num_rows))
    gs = fig.add_gridspec(num_rows, num_cols, hspace=0.3, wspace=0.3)
    
    # ===== Row 0: Full image and Query =====
    ax_full = fig.add_subplot(gs[0, 0])
    if batch.full_rgb is not None:
        full_img = batch.full_rgb[b].permute(1, 2, 0).cpu().numpy()
        ax_full.imshow(np.clip(full_img, 0, 1))
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
            ax_full.add_patch(rect)
    ax_full.set_title('Full Image', fontweight='bold')
    ax_full.axis('off')
    
    ax_query = fig.add_subplot(gs[0, 1])
    query_img = denormalize_image(batch.images[b, 0])
    ax_query.imshow(query_img)
    ax_query.set_title('Query', fontweight='bold', color='blue')
    ax_query.axis('off')
    
    # ===== Rows for templates and correspondences =====
    for k, template_idx in enumerate(template_indices):
        # Calculate position in grid
        template_row_idx = k // max_vis_on_line
        template_col_idx = k % max_vis_on_line
        
        # Row indices: 1 + template_row_idx * 2 for templates, +1 for correspondences
        template_row = 1 + template_row_idx * 2
        corr_row = template_row + 1
        col = template_col_idx + 2  # +2 to skip full image and query columns
        
        template_img = denormalize_image(batch.images[b, template_idx + 1])
        score = result_dict[template_idx]["template_score"]
        rank = result_dict[template_idx]["template_rank"]
        
        # Template image
        ax_template = fig.add_subplot(gs[template_row, col])
        ax_template.imshow(template_img)
        ax_template.set_title(f'T{rank+1} (idx={template_idx})\nScore: {score:.3f}',
                             fontweight='bold', color='green')
        ax_template.axis('off')
        
        # Coarse correspondences
        ax_corr = fig.add_subplot(gs[corr_row, col])
        
        # Get coarse correspondences from predictor
        coarse = result_dict[template_idx]["coarse"]
        q_uv = coarse["q_center_uv"].cpu().numpy()  # [Nc, 2]
        t_uv = coarse["t_center_uv"].cpu().numpy()  # [Nc, 2]
        scores = coarse["patch_score"].cpu().numpy()  # [Nc]
        
        # Create visualization showing query and template side by side
        combined_img = np.concatenate([query_img, template_img], axis=1)
        ax_corr.imshow(combined_img)
        
        # Subsample for visualization
        n_corr = len(q_uv)
        if n_corr > max_coarse_lines:
            # Select top-scoring correspondences
            top_indices = np.argsort(scores)[-max_coarse_lines:]
            q_uv = q_uv[top_indices]
            t_uv = t_uv[top_indices]
            scores = scores[top_indices]
        
        if len(q_uv) > 0:
            colors = cm.rainbow(np.linspace(0, 1, len(q_uv)))
            
            # Draw correspondence lines
            for idx in range(len(q_uv)):
                q_x, q_y = q_uv[idx]
                t_x, t_y = t_uv[idx]
                t_x += img_size  # Offset for side-by-side display
                
                ax_corr.plot([q_x, t_x], [q_y, t_y],
                            color=colors[idx], alpha=0.6, linewidth=1)
                ax_corr.plot(q_x, q_y, 'o', color=colors[idx], markersize=3)
                ax_corr.plot(t_x, t_y, 'o', color=colors[idx], markersize=3)
        
        ax_corr.set_title(f'Coarse (T{rank+1})\n{len(q_uv)}/{n_corr} shown',
                         fontsize=9)
        ax_corr.axis('off')
    
    # Add overall title
    obj_label = batch.infos.label[b] if hasattr(batch.infos, 'label') else f'{b}'
    fig.suptitle(f'Coarse Correspondences - Object {obj_label} ({K} templates)',
                fontsize=14, fontweight='bold')
    
    # Save
    output_path = output_dir / f'coarse_corr_sample{sample_id}_obj{obj_label}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved coarse visualization: {output_path}")


def visualize_inference_pipeline_fine(
    batch: VPOGBatch,
    results: List[Dict],
    sample_id: int,
    output_dir: Path,
    num_patches_to_show: int = 10,
):
    """
    Visualize dense patch correspondences for the BEST template (similar to visualize_dense_patch_flow).
    
    Shows query image, best template, and top-K patches with their dense pixel correspondences
    compared against GT dense flow.
    
    Args:
        batch: VPOGBatch with images and metadata (batch_size=1)
        results: List[Dict] from VPOGPredictor.predict_correspondences_on_query()
        sample_id: Sample identifier for filename
        output_dir: Directory to save visualizations
        num_patches_to_show: Number of top patches to visualize in detail
    """
    b = 0  # Always 0 for batch_size=1
    result_dict = results[b]
    
    # Get BEST template only
    template_indices = list(result_dict.keys())
    if len(template_indices) == 0:
        logger.warning(f"No templates found for sample {sample_id}")
        return
    
    best_template_idx = template_indices[0]  # Already sorted by rank
    template_entry = result_dict[best_template_idx]
    
    coarse = template_entry["coarse"]
    refined = template_entry["refined"]
    
    # Extract prediction data
    q_idx = coarse["q_idx"].cpu()  # [Nc]
    t_idx = coarse["t_idx"].cpu()  # [Nc]
    patch_scores = coarse["patch_score"].cpu().numpy()  # [Nc]
    
    pair_row = refined["pair_row"].cpu()  # [Nr] - maps each refined pixel to coarse patch index
    t_uv = refined["t_uv"].cpu().numpy()  # [Nr, 2]
    q_uv = refined["q_uv"].cpu().numpy()  # [Nr, 2]
    weights = refined["w"].cpu().numpy()  # [Nr]
    
    Nc = len(q_idx)
    Nr = len(pair_row)
    
    if Nc == 0:
        logger.warning(f"No coarse patches for best template {best_template_idx}")
        return
    
    # Extract GT dense flow data
    # batch.dense_flow[b, s] is [H_p, W_p, ps, ps, 2] - flow from buddy template patch to query patch
    # batch.dense_visibility[b, s] is [H_p, W_p, ps, ps] - visibility weights
    gt_dense_flow = batch.dense_flow[b, best_template_idx].cpu()  # [H_p, W_p, ps, ps, 2]
    gt_dense_vis = batch.dense_visibility[b, best_template_idx].cpu()  # [H_p, W_p, ps, ps]
    
    # Group refined pixels by coarse patch pair
    # For each coarse patch pair c, find all refined pixels k where pair_row[k] == c
    patch_refined_pixels = [[] for _ in range(Nc)]
    for k in range(Nr):
        c = pair_row[k].item()
        if 0 <= c < Nc:
            patch_refined_pixels[c].append(k)
    
    # Select top-K patches by score
    top_patch_indices = np.argsort(patch_scores)[-num_patches_to_show:][::-1]
    num_patches = len(top_patch_indices)
    
    # Get images
    query_img = denormalize_image(batch.images[b, 0])
    template_img = denormalize_image(batch.images[b, best_template_idx + 1])
    
    # Constants
    patch_size = 16
    grid_size = 14
    img_size = 224
    
    # Create figure with 4 columns: Query Patch, Template Patch, Predicted Flow, GT Flow
    fig = plt.figure(figsize=(25, 5 * (num_patches + 1)))
    
    # Row 0: Overview
    ax_q = plt.subplot2grid((num_patches + 1, 4), (0, 0), colspan=1)
    ax_q.imshow(query_img)
    ax_q.set_title('Query Image', fontweight='bold', fontsize=12)
    ax_q.axis('off')
    
    ax_t = plt.subplot2grid((num_patches + 1, 4), (0, 1), colspan=1)
    ax_t.imshow(template_img)
    score = template_entry["template_score"]
    ax_t.set_title(f'Best Template (idx={best_template_idx})\nScore: {score:.3f}',
                   fontweight='bold', fontsize=12, color='green')
    ax_t.axis('off')
    
    # Info panel
    ax_info = plt.subplot2grid((num_patches + 1, 4), (0, 2), colspan=2)
    info_text = (
        f'Dense Patch Flow Comparison: Predicted vs GT\n'
        f'═════════════════════════════════════════\n\n'
        f'Direction: Template → Query\n\n'
        f'Patches shown: {num_patches}/{Nc}\n'
        f'Patch size: {patch_size}×{patch_size} pixels\n\n'
        f'Prediction Stats:\n'
        f'  Total coarse: {Nc}\n'
        f'  Total refined: {Nr}\n\n'
        f'Visualization:\n'
        f'  Left: Predicted correspondences\n'
        f'  Right: GT correspondences\n'
        f'  Blue dots: Template pixels\n'
        f'  Green dots: Query predictions\n'
        f'  Color intensity: Weight'
    )
    ax_info.text(
        0.5, 0.5, info_text,
        ha='center', va='center',
        fontsize=10, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    )
    ax_info.axis('off')
    
    # Visualize top patches
    for idx, c in enumerate(top_patch_indices):
        row = idx + 1
        
        # Get patch indices
        q_patch_idx = q_idx[c].item()
        t_patch_idx = t_idx[c].item()
        patch_score = patch_scores[c]
        
        # Convert to (i, j) coordinates
        q_i = q_patch_idx // grid_size
        q_j = q_patch_idx % grid_size
        t_i = t_patch_idx // grid_size
        t_j = t_patch_idx % grid_size
        
        # Extract patches
        q_y_start = q_i * patch_size
        q_x_start = q_j * patch_size
        q_patch = query_img[q_y_start:q_y_start + patch_size, q_x_start:q_x_start + patch_size]
        
        t_y_start = t_i * patch_size
        t_x_start = t_j * patch_size
        t_patch = template_img[t_y_start:t_y_start + patch_size, t_x_start:t_x_start + patch_size]
        
        # Column 0: Query patch
        ax_qp = plt.subplot2grid((num_patches + 1, 4), (row, 0), colspan=1)
        ax_qp.imshow(q_patch)
        ax_qp.set_title(f'Query [{q_i},{q_j}]\nScore: {patch_score:.3f}',
                       fontsize=10)
        ax_qp.axis('off')
        
        # Column 1: Template patch
        ax_tp = plt.subplot2grid((num_patches + 1, 4), (row, 1), colspan=1)
        ax_tp.imshow(t_patch)
        ax_tp.set_title(f'Template [{t_i},{t_j}]', fontsize=10)
        ax_tp.axis('off')
        
        # Column 2: PREDICTED Correspondences
        ax_pred = plt.subplot2grid((num_patches + 1, 4), (row, 2), colspan=1)
        
        # Get refined pixels for this patch
        pixel_indices = patch_refined_pixels[c]
        num_refined = len(pixel_indices)
        
        if num_refined == 0:
            ax_pred.text(0.5, 0.5, 'No refined\ncorrespondences',
                        ha='center', va='center',
                        fontsize=11, fontweight='bold', color='red')
            ax_pred.set_title(f'Predicted: 0', fontsize=10, fontweight='bold', color='blue')
        else:
            # Extract patch-local coordinates
            t_patch_uv = []
            q_patch_uv = []
            w_patch = []
            
            for k in pixel_indices:
                # Template pixel in patch coordinates (relative to t_y_start, t_x_start)
                t_u_local = t_uv[k, 0] - t_x_start
                t_v_local = t_uv[k, 1] - t_y_start
                
                # Query pixel in patch coordinates (relative to q_y_start, q_x_start)
                q_u_local = q_uv[k, 0] - q_x_start
                q_v_local = q_uv[k, 1] - q_y_start
                
                t_patch_uv.append([t_u_local, t_v_local])
                q_patch_uv.append([q_u_local, q_v_local])
                w_patch.append(weights[k])
            
            t_patch_uv = np.array(t_patch_uv)
            q_patch_uv = np.array(q_patch_uv)
            w_patch = np.array(w_patch)
            
            # Show patches side-by-side
            combined_patch = np.concatenate([t_patch, q_patch], axis=1)
            ax_pred.imshow(combined_patch)
            
            # Draw correspondences (template -> query)
            colors = cm.viridis(w_patch / (w_patch.max() + 1e-6))
            
            for i in range(len(t_patch_uv)):
                t_x, t_y = t_patch_uv[i]
                q_x, q_y = q_patch_uv[i]
                q_x += patch_size  # Offset for side-by-side
                
                # Only draw if in bounds
                if 0 <= t_x < patch_size and 0 <= t_y < patch_size:
                    ax_pred.plot([t_x, q_x], [t_y, q_y],
                                color=colors[i], alpha=0.6, linewidth=1)
                    ax_pred.plot(t_x, t_y, 'o', color='blue', markersize=3)
                    ax_pred.plot(q_x, q_y, 'o', color='green', markersize=3)
            
            ax_pred.set_title(f'Predicted: {num_refined}\nWeight: [{w_patch.min():.2f}, {w_patch.max():.2f}]',
                            fontsize=10, fontweight='bold', color='blue')
        
        ax_pred.axis('off')
        
        # Column 3: GT Correspondences
        ax_gt = plt.subplot2grid((num_patches + 1, 4), (row, 3), colspan=1)
        
        # Extract GT flow for this query patch [q_i, q_j]
        # GT flow is indexed by query patch and contains flow from buddy template patch
        gt_flow_patch = gt_dense_flow[q_i, q_j].numpy()  # [ps, ps, 2]
        gt_vis_patch = gt_dense_vis[q_i, q_j].numpy()    # [ps, ps]
        
        # GT flow is in normalized coordinates (relative to query patch center, divided by ps)
        # Convert to pixel coordinates for visualization
        gt_visible = gt_vis_patch > 0
        num_gt_vis = gt_visible.sum()
        
        if num_gt_vis == 0:
            ax_gt.text(0.5, 0.5, 'No GT\ncorrespondences',
                      ha='center', va='center',
                      fontsize=11, fontweight='bold', color='red')
            ax_gt.set_title(f'GT: 0', fontsize=10, fontweight='bold', color='orange')
        else:
            # Show patches side-by-side
            combined_patch = np.concatenate([t_patch, q_patch], axis=1)
            ax_gt.imshow(combined_patch)
            
            # Get all visible pixels
            vis_i, vis_j = np.where(gt_visible)
            
            # Template pixels are just the grid positions within template patch
            t_patch_x_gt = vis_j  # column index
            t_patch_y_gt = vis_i  # row index
            
            # Query pixels: query patch center + flow * ps
            q_center_x = patch_size / 2
            q_center_y = patch_size / 2
            q_patch_x_gt = q_center_x + gt_flow_patch[vis_i, vis_j, 0] * patch_size
            q_patch_y_gt = q_center_y + gt_flow_patch[vis_i, vis_j, 1] * patch_size
            
            # Color by visibility weight
            colors_gt = cm.viridis(gt_vis_patch[vis_i, vis_j] / (gt_vis_patch.max() + 1e-6))
            
            # Draw correspondences
            for i in range(len(vis_i)):
                t_x, t_y = t_patch_x_gt[i], t_patch_y_gt[i]
                q_x, q_y = q_patch_x_gt[i], q_patch_y_gt[i]
                q_x += patch_size  # Offset for side-by-side
                
                # Draw if in bounds
                if 0 <= t_x < patch_size and 0 <= t_y < patch_size:
                    ax_gt.plot([t_x, q_x], [t_y, q_y],
                              color=colors_gt[i], alpha=0.6, linewidth=1)
                    ax_gt.plot(t_x, t_y, 'o', color='blue', markersize=3)
                    ax_gt.plot(q_x, q_y, 'o', color='green', markersize=3)
            
            vis_min = gt_vis_patch[gt_visible].min()
            vis_max = gt_vis_patch[gt_visible].max()
            ax_gt.set_title(f'GT: {num_gt_vis}\nWeight: [{vis_min:.2f}, {vis_max:.2f}]',
                          fontsize=10, fontweight='bold', color='orange')
        
        ax_gt.axis('off')
    
    # Overall title
    obj_label = batch.infos.label[b] if hasattr(batch.infos, 'label') else f'{sample_id}'
    fig.suptitle(f'Dense Patch Flow Comparison - Object {obj_label}, Template {best_template_idx}',
                fontsize=14, fontweight='bold')
    
    # Save
    output_path = output_dir / f'dense_flow_sample{sample_id}_obj{obj_label}_t{best_template_idx}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved dense flow visualization: {output_path}")


def visualize_inference_pipeline(
    batch: VPOGBatch,
    results: List[Dict],
    sample_idx: int,
    output_dir: Path,
    max_coarse_lines: int = 50,
    max_refined_lines: int = 100,
):
    """
    Create comprehensive visualization of inference pipeline using VPOGPredictor results.
    
    Shows:
    1. Full image, query crop, and top-K template matches
    2. Query-template coarse patch correspondences with colored lines
    3. Refined pixel-level correspondences from dense flow
    
    Args:
        batch: VPOGBatch with images and metadata
        results: List[Dict] from VPOGPredictor.predict_correspondences_on_query()
        sample_idx: Batch index
        output_dir: Directory to save visualizations
        max_coarse_lines: Maximum coarse correspondences to draw per template
        max_refined_lines: Maximum refined correspondences to draw per template
    """
    b = 0 # Only support batch_size=1 for visualization
    result_dict = results[b]
    
    # Extract template info
    template_indices = list(result_dict.keys())
    k_top_templates = len(template_indices)
    
    # Get image dimensions
    patch_size = 16
    grid_size = 14
    img_size = 224
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, k_top_templates + 2, hspace=0.3, wspace=0.3)
    
    # ===== Row 1: Images =====
    # Full RGB with bbox
    ax_full = fig.add_subplot(gs[0, 0])
    if batch.full_rgb is not None:
        full_img = batch.full_rgb[b].permute(1, 2, 0).cpu().numpy()
        ax_full.imshow(np.clip(full_img, 0, 1))
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
            ax_full.add_patch(rect)
    ax_full.set_title('Full Image', fontweight='bold')
    ax_full.axis('off')
    
    # Query image
    ax_query = fig.add_subplot(gs[0, 1])
    query_img = denormalize_image(batch.images[b, 0])
    ax_query.imshow(query_img)
    ax_query.set_title('Query', fontweight='bold', color='blue')
    ax_query.axis('off')
    
    # Top-K templates
    template_axes = []
    for k, template_idx in enumerate(template_indices):
        ax_template = fig.add_subplot(gs[0, k + 2])
        template_img = denormalize_image(batch.images[b, template_idx + 1])
        ax_template.imshow(template_img)
        score = result_dict[template_idx]["template_score"]
        rank = result_dict[template_idx]["template_rank"]
        ax_template.set_title(f'T{rank+1} (idx={template_idx})\nScore: {score:.3f}',
                             fontweight='bold', color='green')
        ax_template.axis('off')
        template_axes.append(ax_template)
    
    # ===== Row 2: Coarse Patch Correspondences =====
    for k, template_idx in enumerate(template_indices):
        ax_corr = fig.add_subplot(gs[1, k + 2])
        template_img = denormalize_image(batch.images[b, template_idx + 1])
        
        # Get coarse correspondences from predictor
        coarse = result_dict[template_idx]["coarse"]
        q_uv = coarse["q_center_uv"].cpu().numpy()  # [Nc, 2]
        t_uv = coarse["t_center_uv"].cpu().numpy()  # [Nc, 2]
        scores = coarse["patch_score"].cpu().numpy()  # [Nc]
        
        # Create visualization showing query and template side by side
        combined_img = np.concatenate([query_img, template_img], axis=1)
        ax_corr.imshow(combined_img)
        
        # Subsample for visualization
        n_corr = len(q_uv)
        if n_corr > max_coarse_lines:
            # Select top-scoring correspondences
            top_indices = np.argsort(scores)[-max_coarse_lines:]
            q_uv = q_uv[top_indices]
            t_uv = t_uv[top_indices]
            scores = scores[top_indices]
        
        colors = cm.rainbow(np.linspace(0, 1, len(q_uv)))
        
        # Draw correspondence lines
        for idx in range(len(q_uv)):
            q_x, q_y = q_uv[idx]
            t_x, t_y = t_uv[idx]
            t_x += img_size  # Offset for side-by-side display
            
            ax_corr.plot([q_x, t_x], [q_y, t_y],
                        color=colors[idx], alpha=0.6, linewidth=1)
            ax_corr.plot(q_x, q_y, 'o', color=colors[idx], markersize=3)
            ax_corr.plot(t_x, t_y, 'o', color=colors[idx], markersize=3)
        
        rank = result_dict[template_idx]["template_rank"]
        ax_corr.set_title(f'Coarse (T{rank+1})\n{len(q_uv)}/{n_corr} shown',
                         fontsize=9)
        ax_corr.axis('off')
    
    # ===== Row 3: Refined Pixel Correspondences =====
    for k, template_idx in enumerate(template_indices):
        ax_refined = fig.add_subplot(gs[2, k + 2])
        template_img = denormalize_image(batch.images[b, template_idx + 1])
        
        # Get refined correspondences from predictor
        refined = result_dict[template_idx]["refined"]
        q_uv = refined["q_uv"].cpu().numpy()  # [Nr, 2]
        t_uv = refined["t_uv"].cpu().numpy()  # [Nr, 2]
        weights = refined["w"].cpu().numpy()  # [Nr]
        
        # Create visualization showing query and template side by side
        combined_img = np.concatenate([query_img, template_img], axis=1)
        ax_refined.imshow(combined_img)
        
        n_refined = len(q_uv)
        if n_refined == 0:
            ax_refined.text(0.5, 0.5, 'No refined\ncorrespondences',
                          ha='center', va='center', transform=ax_refined.transAxes)
            rank = result_dict[template_idx]["template_rank"]
            ax_refined.set_title(f'Refined (T{rank+1})\n0 correspondences',
                               fontsize=9)
            ax_refined.axis('off')
            continue
        
        # Subsample for visualization (select highest weight)
        if n_refined > max_refined_lines:
            top_indices = np.argsort(weights)[-max_refined_lines:]
            q_uv = q_uv[top_indices]
            t_uv = t_uv[top_indices]
            weights = weights[top_indices]
        
        # Color by weight (confidence)
        colors = cm.viridis(weights / (weights.max() + 1e-6))
        
        # Draw correspondence lines
        for idx in range(len(q_uv)):
            q_x, q_y = q_uv[idx]
            t_x, t_y = t_uv[idx]
            t_x += img_size  # Offset for side-by-side display
            
            # Only draw if both points are valid
            if 0 <= q_x < img_size and 0 <= q_y < img_size:
                ax_refined.plot([q_x, t_x], [q_y, t_y],
                              color=colors[idx], alpha=0.5, linewidth=0.5)
                ax_refined.plot(q_x, q_y, 'o', color=colors[idx], markersize=2)
                ax_refined.plot(t_x, t_y, 'o', color=colors[idx], markersize=2)
        
        rank = result_dict[template_idx]["template_rank"]
        ax_refined.set_title(f'Refined (T{rank+1})\n{len(q_uv)}/{n_refined} shown',
                           fontsize=9)
        ax_refined.axis('off')
    
    # Add overall title
    obj_label = batch.infos.label[b] if hasattr(batch.infos, 'label') else f'{b}'
    fig.suptitle(f'Inference Pipeline Visualization - Object {obj_label}',
                fontsize=14, fontweight='bold')
    
    # Save
    output_path = output_dir / f'inference_pipeline_sample{sample_idx}_obj{obj_label}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization: {output_path}")


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize image using ImageNet stats."""
    img = tensor.cpu().numpy()
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    img = img * std + mean
    img = np.transpose(img, (1, 2, 0))
    return np.clip(img, 0, 1)


def main():
    parser = argparse.ArgumentParser(description='VPOG Quantitative Test')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       help='BOP dataset name (e.g., ycbv, tless, lmo)')
    parser.add_argument('--use_detections', action='store_true', default=False,
                       help='Use detections (true) or GT masks (false)')
    parser.add_argument('--output_dir_name', type=str, default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--root_dir', type=str,
                       default='datasets',
                       help='Root directory for datasets')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top templates to visualize')
    parser.add_argument('--top_h', type=int, default=10,
                       help='Number of top patch correspondences for dense flow')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--flow_threshold', type=float, default=0.5,
                       help='Threshold for dense flow visualization')
    parser.add_argument('--max_lines', type=int, default=50,
                       help='Maximum number of correspondence lines to draw')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir_name = Path(args.output_dir_name)
    
    # Navigate up to find the run directory that contains .hydra config
    run_dir = checkpoint_path.parent.parent  # Go up from checkpoints/
    hydra_dir = run_dir / '.hydra'

    run_dir_date = run_dir.name
    
    # Setup
    output_dir = run_dir / output_dir_name / f'{args.dataset}' / f"{run_dir_date}_{checkpoint_path.parent.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    use_detections = args.use_detections
    
    logger.info("="*80)
    logger.info("VPOG Quantitative Test - Inference Pipeline Visualization")
    logger.info("="*80)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Use detections: {use_detections}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Top-K templates: {args.top_k}")
    logger.info(f"Top-H patches: {args.top_h}")
    logger.info(f"Device: {args.device}")
    
    # Load model
    lightning_module = load_model_from_checkpoint(checkpoint_path, hydra_dir)
    lightning_module = lightning_module.to(args.device)
    lightning_module.eval()
    
    # Create VPOGPredictor from loaded model
    vpog_model = lightning_module.model  # Extract core model
    predictor = VPOGPredictor(model=vpog_model, device=args.device)
    
    logger.info("✓ VPOGPredictor initialized")
    
    # Setup dataloader
    dataloader = setup_validation_dataloader(
        dataset_name=args.dataset,
        root_dir=args.root_dir,
        batch_size=1,
        num_workers=0,
    )
    
    # Process samples
    logger.info(f"\nProcessing {args.num_samples} samples...")
    
    for sample_idx, batch in enumerate(tqdm(dataloader, total=args.num_samples)):
        if sample_idx >= args.num_samples:
            break
        
        if batch is None:
            logger.warning(f"Sample {sample_idx}: Batch is None, skipping")
            continue
        
        try:
            # Run inference using VPOGPredictor
            results = run_inference_on_batch(
                predictor=predictor,
                batch=batch,
                top_k_templates=args.top_k,
                weight_thresh=args.flow_threshold,
            )
            
            # results is List[Dict[template_idx -> correspondence_dict]]
            # For batch_size=1, we use results[0]
            result_dict = results[0]
            
            # Extract template indices and scores
            template_indices = list(result_dict.keys())
            template_scores = [result_dict[t]["template_score"] for t in template_indices]
            
            logger.info(f"Sample {sample_idx}: Found {len(template_indices)} templates")
            for rank, (t_idx, score) in enumerate(zip(template_indices, template_scores)):
                corr = result_dict[t_idx]
                n_coarse = len(corr["coarse"]["q_center_uv"])
                n_refined = len(corr["refined"]["q_uv"])
                logger.info(f"  Template {t_idx} (rank {rank}): score={score:.3f}, "
                           f"coarse={n_coarse}, refined={n_refined}")
            
            # Example: Estimate pose for best template
            if len(template_indices) > 0:
                best_template_idx = template_indices[0]  # Already sorted by rank
                corr_dict = result_dict[best_template_idx]
                
                # Get template data from batch (template_idx is in S dimension)
                template_depth = batch.template_depth[0, best_template_idx]  # [H, W]
                template_K = batch.K[0, best_template_idx + 1]  # [3, 3] (+1 skip query)
                template_pose = batch.poses[0, best_template_idx + 1]  # [4, 4]
                query_K_original = batch.K_original_query[0]  # [3, 3]
                M_query = batch.M_query[0]  # [3, 3]
                
                # pose, num_inliers, score = estimate_pose_from_predictor_output(
                #     correspondence_dict=corr_dict,
                #     template_depth=template_depth,
                #     template_K=template_K,
                #     template_pose=template_pose,
                #     query_K_original=query_K_original,
                #     M_query=M_query,
                #     scale_factor=10.0,
                #     use_refined=True,
                # )
                
                # logger.info(f"  Pose estimate: inliers={num_inliers}, score={score:.3f}")
            
            # Visualize inference pipeline (both coarse and refined)
            visualize_inference_pipeline_coarse(
                batch=batch,
                results=results,
                sample_id=sample_idx,  # Use actual sample index for filename
                output_dir=output_dir,
                max_coarse_lines=args.max_lines,
                max_vis_on_line=4,  # 4 templates per row
            )
            
            visualize_inference_pipeline_fine(
                batch=batch,
                results=results,
                sample_id=sample_idx,  # Use actual sample index for filename
                output_dir=output_dir,
                num_patches_to_show=10,  # Show top 10 patches for best template
            )
            
        except Exception as e:
            logger.error(f"Sample {sample_idx}: Error - {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("="*80)
    logger.info(f"Completed! Visualizations saved to: {output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
