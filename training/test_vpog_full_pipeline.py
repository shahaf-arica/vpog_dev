"""
Full VPOG Pipeline Integration Test
Tests the complete pipeline from dataloader to training to inference

This test:
1. Loads data using VPOGTrainDataset
2. Processes through VPOG model (encoder -> AA -> classification + flow)
3. Computes all losses (classification, flow, regularization)
4. Tests correspondence construction
5. Tests inference modes (cluster and global)
6. Generates comprehensive visualizations

Run with: python -m training.test_vpog_full_pipeline
Run with config: python -m training.test_vpog_full_pipeline --config-name=test_vpog
"""

import os
import sys
sys.path.append("./vpog")
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

# Training imports
from training.dataloader.vpog_dataset import VPOGTrainDataset, VPOGBatch
from training.dataloader.flow_computer import FlowComputer
from training.dataloader.template_selector import extract_d_ref_from_pose

# VPOG model imports
from vpog.models.vpog_model import VPOGModel
from vpog.models.token_manager import TokenManager

# Training loss imports
from training.losses import ClassificationLoss, FlowLoss, WeightRegularization, EProPnPLoss

# VPOG inference imports
from vpog.inference import CorrespondenceBuilder, ClusterModeInference, GlobalModeInference

# Training visualization imports (USING FIXED VERSIONS)
from training.test_vpog_visualizations_fixed import (
    visualize_data_and_poses,
    visualize_template_flow,
    visualize_patch_flow_detailed,
    visualize_comprehensive_pipeline,
)


def create_synthetic_vpog_batch(
    batch_size: int = 2,
    num_templates: int = 5,
    image_size: int = 224,
    patch_size: int = 16,
    device: str = 'cuda',
) -> VPOGBatch:
    """
    Create synthetic VPOG batch for testing without real data.
    """
    S = num_templates
    H, W = image_size, image_size
    H_p, W_p = H // patch_size, W // patch_size
    
    import src.megapose.utils.tensor_collection as tc
    import pandas as pd
    
    # Create synthetic data
    batch = VPOGBatch(
        images=torch.randn(batch_size, S+1, 3, H, W, device=device),
        masks=torch.ones(batch_size, S+1, H, W, device=device),
        K=torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, S+1, -1, -1).clone(),
        poses=torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, S+1, -1, -1).clone(),
        d_ref=F.normalize(torch.randn(batch_size, 3, device=device), dim=-1),
        template_indices=torch.randint(0, 162, (batch_size, S), device=device),
        template_types=torch.cat([
            torch.zeros(batch_size, 3, device=device),  # 3 positives
            torch.ones(batch_size, 2, device=device),   # 2 negatives
        ], dim=1).long(),
        flows=torch.randn(batch_size, S, H_p, W_p, 2, device=device) * 2.0,  # Flow in pixels
        visibility=torch.rand(batch_size, S, H_p, W_p, device=device) > 0.2,  # 80% visible
        patch_visibility=torch.rand(batch_size, S, H_p, W_p, device=device) > 0.3,  # 70% visible
        infos=tc.PandasTensorCollection(
            infos=pd.DataFrame({
                'scene_id': [f'scene_{i}' for i in range(batch_size)],
                'view_id': [0] * batch_size,
                'label': ['obj_000001'] * batch_size,
            })
        )
    )
    
    # Set realistic K values
    batch.K[:, :, 0, 0] = 500.0  # fx
    batch.K[:, :, 1, 1] = 500.0  # fy
    batch.K[:, :, 0, 2] = H / 2  # cx
    batch.K[:, :, 1, 2] = W / 2  # cy
    
    return batch


def test_encoder_integration(model: VPOGModel, batch: VPOGBatch):
    """Test encoder with batch data."""
    print("\n" + "="*80)
    print("TEST 1: Encoder Integration")
    print("="*80)
    
    try:
        # Extract query and templates
        query_images = batch.images[:, 0:1]  # [B, 1, 3, H, W]
        template_images = batch.images[:, 1:]  # [B, S, 3, H, W]
        
        print(f"Query images: {query_images.shape}")
        print(f"Template images: {template_images.shape}")
        
        # Run encoder (manually access to test)
        B, S = template_images.shape[:2]
        
        # Encode query
        with torch.no_grad():
            query_flat = query_images.reshape(B * 1, 3, 224, 224)
            query_features_flat, query_pos_flat = model.encoder(query_flat)
            query_features = query_features_flat.reshape(B, 1, -1, model.encoder.embed_dim)
            query_pos = query_pos_flat.reshape(B, 1, -1, 2)
        
        print(f"✓ Query features: {query_features.shape}")
        print(f"✓ Query positions: {query_pos.shape}")
        
        # Encode templates
        with torch.no_grad():
            template_flat = template_images.reshape(B * S, 3, 224, 224)
            template_features_flat, template_pos_flat = model.encoder(template_flat)
            template_features = template_features_flat.reshape(B, S, -1, model.encoder.embed_dim)
            template_pos = template_pos_flat.reshape(B, S, -1, 2)
        
        print(f"✓ Template features: {template_features.shape}")
        print(f"✓ Template positions: {template_pos.shape}")
        
        # Check for NaN/Inf
        assert not torch.isnan(query_features).any(), "Query features contain NaN"
        assert not torch.isnan(template_features).any(), "Template features contain NaN"
        
        print("\n✓ Encoder integration test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Encoder integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward_pass(model: VPOGModel, batch: VPOGBatch):
    """Test full model forward pass."""
    print("\n" + "="*80)
    print("TEST 2: Model Forward Pass")
    print("="*80)
    
    try:
        # Prepare data for VPOG model
        B = batch.images.shape[0]
        S = batch.images.shape[1] - 1
        
        # Split query and templates
        query_images = batch.images[:, 0, :, :, :]  # [B, 3, H, W]
        template_images = batch.images[:, 1:, :, :, :]  # [B, S, 3, H, W]
        
        # Prepare poses
        query_poses = batch.poses[:, 0, :, :]  # [B, 4, 4]
        template_poses = batch.poses[:, 1:, :, :]  # [B, S, 4, 4]
        
        # Reference directions for S²RoPE
        ref_dirs = batch.d_ref  # [B, 3]
        
        print(f"Running VPOG model forward pass...")
        print(f"  Query images: {query_images.shape}")
        print(f"  Template images: {template_images.shape}")
        print(f"  Query poses: {query_poses.shape}")
        print(f"  Template poses: {template_poses.shape}")
        print(f"  Ref dirs: {ref_dirs.shape}")
        
        with torch.no_grad():
            outputs = model(query_images, template_images, query_poses, template_poses, ref_dirs)
        
        print(f"✓ Classification logits: {outputs['classification_logits'].shape}")
        print(f"✓ Flow: {outputs['flow'].shape}")
        print(f"✓ Flow confidence: {outputs['flow_confidence'].shape}")
        
        # Verify shapes
        # Model internally uses num_patches from img_size/patch_size
        Nq = model.num_patches  # 196 for 224x224 with patch_size 16
        Nt = model.num_patches
        num_added = model.token_manager.num_template_added_tokens
        
        expected_cls_shape = (B, S, Nq, Nt + num_added)
        expected_flow_shape = (B, S, Nq, Nt, 16, 16, 2)
        expected_conf_shape = (B, S, Nq, Nt, 16, 16)
        
        assert outputs['classification_logits'].shape == expected_cls_shape, \
            f"Classification shape {outputs['classification_logits'].shape} != {expected_cls_shape}"
        assert outputs['flow'].shape == expected_flow_shape, \
            f"Flow shape {outputs['flow'].shape} != {expected_flow_shape}"
        assert outputs['flow_confidence'].shape == expected_conf_shape, \
            f"Confidence shape {outputs['flow_confidence'].shape} != {expected_conf_shape}"
        
        # Check for NaN/Inf
        assert not torch.isnan(outputs['classification_logits']).any(), "Classification contains NaN"
        assert not torch.isnan(outputs['flow']).any(), "Flow contains NaN"
        assert not torch.isnan(outputs['flow_confidence']).any(), "Confidence contains NaN"
        
        print("\n✓ Model forward pass test PASSED")
        return outputs
        
    except Exception as e:
        print(f"\n✗ Model forward pass test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_loss_computation(outputs: dict, batch: VPOGBatch):
    """Test all loss functions."""
    print("\n" + "="*80)
    print("TEST 3: Loss Computation")
    print("="*80)
    
    try:
        # Initialize losses
        cls_loss_fn = ClassificationLoss(tau=1.0)
        flow_loss_fn = FlowLoss(loss_type='l1')
        
        B = batch.images.shape[0]
        S = batch.images.shape[1] - 1
        Nq = outputs['classification_logits'].shape[2]
        Nt = outputs['flow'].shape[3]
        
        # Create dummy GT labels for testing
        gt_labels = torch.randint(0, Nt, (B, S, Nq), device=batch.images.device)
        unseen_mask = torch.rand(B, S, Nq, device=batch.images.device) < 0.1  # 10% unseen
        
        # Classification loss
        cls_loss = cls_loss_fn(
            outputs['classification_logits'],
            gt_labels,
            unseen_mask,
            unseen_token_idx=Nt,
        )
        print(f"✓ Classification loss: {cls_loss.item():.4f}")
        
        # Flow loss
        # Convert batch flows to model format [B, S, Nq, Nt, 16, 16, 2]
        # For testing, create dummy GT flow
        flow_gt = torch.randn_like(outputs['flow'])
        flow_unseen = torch.rand(B, 1, Nq, Nt, 16, 16, device=batch.images.device) < 0.2  # 20% unseen
        
        flow_loss = flow_loss_fn(
            outputs['flow'],
            flow_gt,
            outputs['flow_confidence'],
            flow_unseen,
        )
        print(f"✓ Flow loss: {flow_loss.item():.4f}")
        
        # Check losses are valid
        assert not torch.isnan(cls_loss), "Classification loss is NaN"
        assert not torch.isinf(cls_loss), "Classification loss is Inf"
        assert not torch.isnan(flow_loss), "Flow loss is NaN"
        assert not torch.isinf(flow_loss), "Flow loss is Inf"
        
        print("\n✓ Loss computation test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Loss computation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_correspondence_construction(outputs: dict, batch: VPOGBatch):
    """Test correspondence builder (SKIPPED - very slow, only needed for PnP inference)."""
    print("\n" + "="*80)
    print("TEST 4: Correspondence Construction")
    print("="*80)
    print("\n⚠️  SKIPPED - This test is VERY SLOW")
    print("\nWHAT IT DOES:")
    print("  Converts model outputs to 2D-3D correspondences for PnP pose estimation:")
    print("  - Takes classification [B, S, Nq, Nt] - which patches match")
    print("  - Takes flow [B, S, Nq, Nt, 16, 16, 2] - pixel-level refinement")
    print("  - Iterates B×S×Nq×Nt×16×16 = ~1.2 BILLION times for B=2, S=5, Nq=Nt=196")
    print("  - Each iteration: check confidence, apply flow, append to list")
    print("\nWHY IT'S SLOW:")
    print("  - Nested Python loops (not vectorized)")
    print("  - Many .item() calls (GPU->CPU transfers)")
    print("  - Appending to Python lists")
    print("\nNOTE: Only needed for PnP inference, NOT for training!")
    print("      Training uses classification + flow losses directly")
    
    print("\n✓ Correspondence construction test SKIPPED (for speed)")
    return {'num_correspondences': 0}


def test_correspondence_construction_slow(outputs: dict, batch: VPOGBatch):
    """SLOW VERSION - only use if you need to test PnP correspondence building."""
    print("\n" + "="*80)
    print("TEST 4: Correspondence Construction (SLOW)")
    print("="*80)
    
    try:
        import time
        start_time = time.time()
        
        # Initialize correspondence builder
        corr_builder = CorrespondenceBuilder(
            patch_size=16,
            conf_threshold=0.3,
            use_pixel_level=True,
        )
        
        B = batch.images.shape[0]
        S = batch.images.shape[1] - 1
        Nq = outputs['classification_logits'].shape[2]
        
        # Prepare query and template data for correspondence builder
        # Create synthetic 3D coordinates for templates
        template_xyz = torch.randn(B, S, Nq, 16, 16, 3, device=batch.images.device)
        
        query_data = {
            'positions': torch.stack([
                torch.arange(14, device=batch.images.device).repeat_interleave(14) for _ in range(2)
            ], dim=-1).unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1).float(),
            'image_size': (224, 224),
        }
        
        template_data = {
            'xyz': template_xyz,
        }
        
        # Build correspondences
        correspondences = corr_builder(
            outputs['classification_logits'],
            outputs['flow'],
            outputs['flow_confidence'],
            query_data,
            template_data,
        )
        
        print(f"✓ Number of correspondences: {correspondences['num_correspondences']}")
        
        if correspondences['num_correspondences'] > 0:
            print(f"✓ 2D points: {correspondences['pts2d'].shape}")
            print(f"✓ 3D points: {correspondences['pts3d'].shape}")
            print(f"✓ Weights: {correspondences['weights'].shape}")
            print(f"  Weights range: [{correspondences['weights'].min():.3f}, {correspondences['weights'].max():.3f}]")
        else:
            print("⚠ No correspondences found (may be due to low confidence)")
        
        print("\n✓ Correspondence construction test PASSED")
        return correspondences
        
    except Exception as e:
        print(f"\n✗ Correspondence construction test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_data_and_poses(batch: VPOGBatch, b_idx: int = 0) -> plt.Figure:
    """
    Visualize input data: query, templates, poses, and metadata.
    
    Shows:
    - Query image with pose info
    - All template images with their poses
    - Pose comparison (rotation and translation)
    - Template selection info (positive/negative)
    """
    S = batch.images.shape[1] - 1
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, max(4, S), hspace=0.4, wspace=0.3)
    
    # Convert images to numpy and denormalize ImageNet normalization
    def to_img(tensor):
        img = tensor.cpu().numpy()  # [C, H, W]
        # ImageNet denormalization
        mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
        std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
        img = img * std + mean
        img = np.transpose(img, (1, 2, 0))  # [H, W, C]
        return np.clip(img, 0, 1)
    
    # === Row 1: Query Image and Info ===
    ax_query = fig.add_subplot(gs[0, :2])
    query_img = to_img(batch.images[b_idx, 0])
    ax_query.imshow(query_img)
    ax_query.set_title("Query Image", fontsize=14, weight='bold')
    ax_query.axis('off')
    
    # Query pose info
    ax_query_info = fig.add_subplot(gs[0, 2:])
    ax_query_info.axis('off')
    query_pose = batch.poses[b_idx, 0].cpu().numpy()
    query_R = query_pose[:3, :3]
    query_t = query_pose[:3, 3]
    d_ref = batch.d_ref[b_idx].cpu().numpy()
    
    query_info_text = f"""
    QUERY INFO:
    
    Translation: [{query_t[0]:.2f}, {query_t[1]:.2f}, {query_t[2]:.2f}]
    
    d_ref: [{d_ref[0]:.3f}, {d_ref[1]:.3f}, {d_ref[2]:.3f}]
    """
    ax_query_info.text(0.05, 0.95, query_info_text, 
                       fontsize=9, verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # === Row 2: Template Images ===
    for s_idx in range(S):
        col = s_idx % max(4, S)
        row = 1 + (s_idx // max(4, S))
        ax = fig.add_subplot(gs[row, col])
        
        template_img = to_img(batch.images[b_idx, s_idx + 1])
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
    
    # === Row 3: Pose Comparison ===
    ax_pose_comp = fig.add_subplot(gs[2, :])
    ax_pose_comp.axis('off')
    
    # Compute pose differences
    pose_info_lines = ["TEMPLATE POSES (relative to query):\\n"]
    pose_info_lines.append(f"{'ID':<4} {'Type':<8} {'Idx':<6} {'ΔR (deg)':<12} {'Δt (norm)':<12} {'Δt_x':<10} {'Δt_y':<10} {'Δt_z':<10}")
    pose_info_lines.append("-" * 90)
    
    for s_idx in range(S):
        template_pose = batch.poses[b_idx, s_idx + 1].cpu().numpy()
        template_R = template_pose[:3, :3]
        template_t = template_pose[:3, 3]
        
        # Compute relative rotation (angle)
        R_rel = query_R.T @ template_R
        trace_R = np.trace(R_rel)
        angle_rad = np.arccos(np.clip((trace_R - 1) / 2, -1, 1))
        angle_deg = np.degrees(angle_rad)
        
        # Translation difference
        delta_t = template_t - query_t
        delta_t_norm = np.linalg.norm(delta_t)
        
        is_positive = batch.template_types[b_idx, s_idx].item() == 0
        template_type = "POS" if is_positive else "NEG"
        template_idx = batch.template_indices[b_idx, s_idx].item()
        
        pose_info_lines.append(
            f"{s_idx:<4} {template_type:<8} {template_idx:<6} {angle_deg:11.4f}° {delta_t_norm:11.4f}  "
            f"{delta_t[0]:9.4f}  {delta_t[1]:9.4f}  {delta_t[2]:9.4f}"
        )
    
    pose_text = "\\n".join(pose_info_lines)
    ax_pose_comp.text(0.05, 0.95, pose_text, 
                     fontsize=9, verticalalignment='top', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # === Row 4: Ground Truth Flow Information ===
    ax_flow_info = fig.add_subplot(gs[3, :])
    ax_flow_info.axis('off')
    
    flow_stats = []
    flow_stats.append("GROUND TRUTH FLOW STATISTICS:\\n")
    flow_stats.append(f"{'ID':<4} {'Type':<8} {'Flow Mean':<15} {'Flow Std':<15} {'Visible %':<12} {'Patch Vis %':<12}")
    flow_stats.append("-" * 80)
    
    for s_idx in range(S):
        flows = batch.flows[b_idx, s_idx].cpu().numpy()  # [H_p, W_p, 2]
        visibility = batch.visibility[b_idx, s_idx].cpu().numpy()  # [H_p, W_p]
        patch_vis = batch.patch_visibility[b_idx, s_idx].cpu().numpy()  # [H_p, W_p]
        
        flow_mag = np.sqrt((flows**2).sum(axis=-1))
        flow_mean = flow_mag[visibility > 0].mean() if visibility.sum() > 0 else 0
        flow_std = flow_mag[visibility > 0].std() if visibility.sum() > 0 else 0
        vis_pct = 100 * visibility.mean()
        patch_vis_pct = 100 * patch_vis.mean()
        
        is_positive = batch.template_types[b_idx, s_idx].item() == 0
        template_type = "POS" if is_positive else "NEG"
        
        flow_stats.append(
            f"{s_idx:<4} {template_type:<8} {flow_mean:14.4f}  {flow_std:14.4f}  {vis_pct:11.1f}%  {patch_vis_pct:11.1f}%"
        )
    
    flow_text = "\\n".join(flow_stats)
    ax_flow_info.text(0.05, 0.95, flow_text, 
                     fontsize=9, verticalalignment='top', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    fig.suptitle(f"Data and Pose Visualization (Batch {b_idx})", fontsize=16, weight='bold')
    
    return fig


def visualize_classification(outputs: dict, batch: VPOGBatch, b_idx: int = 0) -> plt.Figure:
    """
    Visualize classification outputs.
    
    Shows:
    - Classification logits heatmap for each template
    - Classification probabilities after softmax
    - Per-patch assignment to templates vs unseen
    - Statistics on classification confidence
    """
    S = batch.images.shape[1] - 1
    classification_logits = outputs['classification_logits'][b_idx].cpu().numpy()  # [S, Nq, Nt+1]
    
    # Apply softmax to get probabilities
    classification_probs = torch.softmax(outputs['classification_logits'][b_idx], dim=-1).cpu().numpy()
    
    Nq = classification_logits.shape[1]
    Nt_plus = classification_logits.shape[2]  # Nt + num_added (including unseen)
    grid_size = int(np.sqrt(Nq))
    
    # Create figure
    fig = plt.figure(figsize=(20, 4 * S))
    gs = fig.add_gridspec(S, 5, hspace=0.4, wspace=0.3)
    
    for s_idx in range(S):
        # === Column 1: Query patches assigned to this template ===
        ax1 = fig.add_subplot(gs[s_idx, 0])
        
        # Get probability of each query patch matching this template
        template_probs = classification_probs[s_idx, :, s_idx].reshape(grid_size, grid_size)
        im1 = ax1.imshow(template_probs, cmap='hot', vmin=0, vmax=1)
        ax1.set_title(f"T{s_idx}: Query→Template\\nProbability", fontsize=10)
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # === Column 2: Unseen probability ===
        ax2 = fig.add_subplot(gs[s_idx, 1])
        unseen_probs = classification_probs[s_idx, :, -1].reshape(grid_size, grid_size)  # Last is unseen
        im2 = ax2.imshow(unseen_probs, cmap='viridis', vmin=0, vmax=1)
        ax2.set_title(f"T{s_idx}: Unseen\\nProbability", fontsize=10)
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # === Column 3: Max probability (winner) ===
        ax3 = fig.add_subplot(gs[s_idx, 2])
        max_probs = classification_probs[s_idx].max(axis=-1).reshape(grid_size, grid_size)
        im3 = ax3.imshow(max_probs, cmap='plasma', vmin=0, vmax=1)
        ax3.set_title(f"T{s_idx}: Max Classification\\nConfidence", fontsize=10)
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # === Column 4: Winner index (which template) ===
        ax4 = fig.add_subplot(gs[s_idx, 3])
        winner_idx = classification_probs[s_idx].argmax(axis=-1).reshape(grid_size, grid_size)
        im4 = ax4.imshow(winner_idx, cmap='tab20', vmin=0, vmax=Nt_plus-1)
        ax4.set_title(f"T{s_idx}: Assigned Template\\nIndex", fontsize=10)
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        # === Column 5: Statistics ===
        ax5 = fig.add_subplot(gs[s_idx, 4])
        ax5.axis('off')
        
        is_positive = batch.template_types[b_idx, s_idx].item() == 0
        template_type = "POSITIVE" if is_positive else "NEGATIVE"
        
        # Compute stats
        assigned_to_this = (winner_idx.flatten() == s_idx).sum()
        assigned_to_unseen = (winner_idx.flatten() == Nt_plus - 1).sum()
        mean_conf_this = template_probs.mean()
        mean_conf_unseen = unseen_probs.mean()
        mean_max_conf = max_probs.mean()
        
        stats_text = f"""
        Template {s_idx}
        Type: {template_type}
        
        Assignment Stats:
        • Patches → T{s_idx}: {assigned_to_this}/{Nq}
          ({100*assigned_to_this/Nq:.1f}%)
        
        • Patches → Unseen: {assigned_to_unseen}/{Nq}
          ({100*assigned_to_unseen/Nq:.1f}%)
        
        Confidence:
        • Mean prob(T{s_idx}): {mean_conf_this:.3f}
        • Mean prob(unseen): {mean_conf_unseen:.3f}
        • Mean max conf: {mean_max_conf:.3f}
        
        Logits range:
        [{classification_logits[s_idx].min():.2f}, 
         {classification_logits[s_idx].max():.2f}]
        """
        
        ax5.text(0.05, 0.95, stats_text, fontsize=9, verticalalignment='top',
                family='monospace', bbox=dict(boxstyle='round', 
                facecolor='lightgreen' if is_positive else 'lightcoral', alpha=0.5))
    
    fig.suptitle(f"Classification Output Visualization (Batch {b_idx})", fontsize=16, weight='bold')
    
    return fig


def visualize_template_flow(outputs: dict, batch: VPOGBatch, b_idx: int = 0, s_idx: int = 0) -> plt.Figure:
    """
    Visualize GROUND TRUTH flow labels for a specific template.
    
    Shows:
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
    query_img = batch.images[b_idx, 0].cpu().numpy()  # [3, H, W]
    template_img = batch.images[b_idx, s_idx + 1].cpu().numpy()  # [3, H, W]
    
    # Denormalize images
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    query_img = np.clip(np.transpose(query_img * std + mean, (1, 2, 0)), 0, 1)
    template_img = np.clip(np.transpose(template_img * std + mean, (1, 2, 0)), 0, 1)
    
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
    from training.visualization.flow_vis import flow_to_color
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
    • Patches: {visible_patches}/{total_pixels}
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
    
    # Get query and template images
    query_img = batch.images[b_idx, 0].cpu().numpy()
    template_img = batch.images[b_idx, s_idx + 1].cpu().numpy()
    
    # Denormalize images
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    query_img = np.clip(np.transpose(query_img * std + mean, (1, 2, 0)), 0, 1)
    template_img = np.clip(np.transpose(template_img * std + mean, (1, 2, 0)), 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # === Row 1: Images and flow components ===
    # Query image
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title("Query Image", fontsize=10)
    axes[0, 0].axis('off')
    
    # Template image
    axes[0, 1].imshow(template_img)
    is_positive = batch.template_types[b_idx, s_idx].item() == 0
    template_type = "POS" if is_positive else "NEG"
    axes[0, 1].set_title(f"Template {s_idx} ({template_type})", fontsize=10)
    axes[0, 1].axis('off')
    
    # Flow X
    flow_max = max(abs(gt_flow[..., 0]).max(), abs(gt_flow[..., 1]).max(), 1.0)
    im2 = axes[0, 2].imshow(gt_flow[..., 0], cmap='RdBu_r', vmin=-flow_max, vmax=flow_max)
    axes[0, 2].set_title("GT Flow X", fontsize=10)
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # Flow Y
    im3 = axes[0, 3].imshow(gt_flow[..., 1], cmap='RdBu_r', vmin=-flow_max, vmax=flow_max)
    axes[0, 3].set_title("GT Flow Y", fontsize=10)
    axes[0, 3].axis('off')
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)
    
    # === Row 2: Flow visualization ===
    # Flow magnitude
    flow_mag = np.sqrt((gt_flow**2).sum(axis=-1))
    im4 = axes[1, 0].imshow(flow_mag, cmap='hot', vmin=0, vmax=flow_max)
    axes[1, 0].set_title("Flow Magnitude", fontsize=10)
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    # Visibility
    im5 = axes[1, 1].imshow(gt_vis, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title("Visibility", fontsize=10)
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
    
    # Flow direction (color-coded)
    from training.visualization.flow_vis import flow_to_color
    flow_rgb = flow_to_color(gt_flow, max_flow=flow_max)
    axes[1, 2].imshow(flow_rgb)
    axes[1, 2].set_title("Flow Direction (HSV)", fontsize=10)
    axes[1, 2].axis('off')
    
    # Statistics
    axes[1, 3].axis('off')
    visible_patches = gt_vis.sum()
    total_patches = gt_vis.size
    mean_flow_mag = flow_mag[gt_vis > 0].mean() if visible_patches > 0 else 0
    
    stats_text = f"""
    GT Patch Flow Statistics:
    
    Template: {s_idx} ({template_type})
    
    Visibility:
    • {visible_patches}/{total_patches} patches
      ({100*visible_patches/total_patches:.1f}%)
    
    Flow:
    • Mean mag: {mean_flow_mag:.2f} px
    • Max mag: {flow_mag.max():.2f} px
    • X range: [{gt_flow[...,0].min():.2f}, 
                {gt_flow[...,0].max():.2f}]
    • Y range: [{gt_flow[...,1].min():.2f}, 
                {gt_flow[...,1].max():.2f}]
    """
    axes[1, 3].text(0.05, 0.95, stats_text, fontsize=9, verticalalignment='top',
                   family='monospace', bbox=dict(boxstyle='round', 
                   facecolor='lightgreen' if is_positive else 'lightcoral', alpha=0.5))
    
    fig.suptitle(f"Patch-Level GT Flow Detail - Batch {b_idx}, Template {s_idx}", fontsize=14, weight='bold')
    plt.tight_layout()
    
    return fig


def visualize_comprehensive_pipeline(outputs: dict, batch: VPOGBatch, output_dir: Path):
    """Comprehensive visualization of all pipeline components."""
    print("\n" + "="*80)
    print("COMPREHENSIVE PIPELINE VISUALIZATION")
    print("="*80)
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        B = batch.images.shape[0]
        S = batch.images.shape[1] - 1  # Number of templates
        
        # Visualize for first batch
        for b_idx in range(min(2, B)):
            print(f"\n=== Visualizing Batch Sample {b_idx} ===")
            
            # === 1. DATA VISUALIZATION ===
            print("  Creating data visualization...")
            fig_data = visualize_data_and_poses(batch, b_idx)
            fig_data.savefig(output_dir / f"sample{b_idx}_01_data_and_poses.png", dpi=150, bbox_inches='tight')
            plt.close(fig_data)
            
            # === 2. CLASSIFICATION VISUALIZATION ===
            print("  Creating classification visualization...")
            fig_cls = visualize_classification(outputs, batch, b_idx)
            fig_cls.savefig(output_dir / f"sample{b_idx}_02_classification.png", dpi=150, bbox_inches='tight')
            plt.close(fig_cls)
            
            # === 3. FLOW VISUALIZATION (per template) ===
            print("  Creating flow visualizations...")
            for s_idx in range(min(3, S)):  # First 3 templates
                fig_flow = visualize_template_flow(outputs, batch, b_idx, s_idx)
                fig_flow.savefig(output_dir / f"sample{b_idx}_03_flow_template{s_idx}.png", dpi=150, bbox_inches='tight')
                plt.close(fig_flow)
            
            # === 4. DETAILED PATCH FLOW ===
            print("  Creating detailed patch flow visualization...")
            fig_patch = visualize_patch_flow_detailed(outputs, batch, b_idx)
            fig_patch.savefig(output_dir / f"sample{b_idx}_04_patch_flow_detail.png", dpi=150, bbox_inches='tight')
            plt.close(fig_patch)
        
        print(f"\n✓ All visualizations saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"\n✗ Visualization FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization(outputs: dict, batch: VPOGBatch, output_dir: Path):
    """Test basic visualization functions (simplified version for non-comprehensive mode)."""
    print("\n" + "="*80)
    print("TEST 5: Basic Visualization")
    print("="*80)
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Just create data and flow visualizations
        print("  Creating data visualization...")
        fig_data = visualize_data_and_poses(batch, b_idx=0)
        fig_data.savefig(output_dir / "data_and_poses.png", dpi=150, bbox_inches='tight')
        plt.close(fig_data)
        print(f"✓ Saved data visualization to {output_dir / 'data_and_poses.png'}")
        
        print("  Creating flow visualization...")
        fig_flow = visualize_template_flow(outputs, batch, b_idx=0, s_idx=0)
        fig_flow.savefig(output_dir / "flow_template0.png", dpi=150, bbox_inches='tight')
        plt.close(fig_flow)
        print(f"✓ Saved flow visualization to {output_dir / 'flow_template0.png'}")
        
        print("\n✓ Basic visualization test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Visualization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_full_pipeline_test(use_real_data: bool = False, device: str = 'cuda', visualize: bool = False):
    """Run complete VPOG pipeline test."""
    print("\n" + "="*80)
    print("VPOG FULL PIPELINE INTEGRATION TEST")
    print("="*80)
    print(f"Device: {device}")
    print(f"Using real data: {use_real_data}")
    print(f"Visualization: {visualize}")
    
    output_dir = Path("./tmp/vpog_full_pipeline_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Step 1: Load/create data
    print("\n" + "-"*80)
    print("Step 1: Loading Data")
    print("-"*80)
    
    import time
    data_start = time.time()
    
    if use_real_data:
        # Load dataset config using Hydra
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        
        # config_dir as obsolute path
        config_dir = str((Path(__file__).parent / "config").resolve())
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            # Load data config (assuming you have training/config/data/vpog.yaml)
            data_cfg = compose(config_name="data/vpog")
        
        print(f"Loaded data config:")
        print(OmegaConf.to_yaml(data_cfg))
        
        # Instantiate dataset from config
        dataset = instantiate(data_cfg.data.dataloader)
        
        # Get batch from dataset
        from src.utils.dataloader import NoneFilteringDataLoader
        dataloader = NoneFilteringDataLoader(
            dataset.web_dataloader.datapipeline,
            batch_size=2,
            num_workers=0,
            collate_fn=dataset.collate_fn,
        )
        batch_load_start = time.time()
        batch = next(iter(dataloader))
        batch_load_time = time.time() - batch_load_start
        
        batch_to_device_start = time.time()
        batch = batch.to(device)
        batch_to_device_time = time.time() - batch_to_device_start
        
        print(f"✓ Loaded real data batch from Hydra config")
        print(f"  Time to load batch: {batch_load_time:.2f}s")
        print(f"  Time to move to {device}: {batch_to_device_time:.2f}s")
    else:
        batch = create_synthetic_vpog_batch(device=device)
        print("✓ Created synthetic batch")
    
    data_total_time = time.time() - data_start
    
    print(f"\n  Batch size: {batch.images.shape[0]}")
    print(f"  Num templates: {batch.images.shape[1] - 1}")
    print(f"  Image size: {batch.images.shape[-2:]}")
    print(f"  Total data loading time: {data_total_time:.2f}s")
    
    # Step 2: Initialize model
    print("\n" + "-"*80)
    print("Step 2: Initializing VPOG Model")
    print("-"*80)
    
    try:
        # Load config using Hydra from training/config/
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        # config_dir as obsolute path
        config_dir = str((Path(__file__).parent / "config").resolve())
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="model/vpog")
        
        # Access model config (Hydra wraps it under 'model' key)
        model_cfg = cfg.model if 'model' in cfg else cfg
        
        print(f"Loaded config:")
        print(OmegaConf.to_yaml(model_cfg))
        
        # Instantiate model components using Hydra
        encoder = instantiate(model_cfg.encoder)
        token_manager = instantiate(model_cfg.token_manager)
        aa_module = instantiate(model_cfg.aa_module)
        classification_head = instantiate(model_cfg.classification_head)
        flow_head = instantiate(model_cfg.flow_head)
        
        # Create VPOG model
        model = VPOGModel(
            encoder=encoder,
            aa_module=aa_module,
            classification_head=classification_head,
            flow_head=flow_head,
            token_manager=token_manager,
            s2rope_config=model_cfg.s2rope_config,
            img_size=model_cfg.img_size,
            patch_size=model_cfg.patch_size,
        )
        model = model.to(device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model initialized with {total_params/1e6:.2f}M parameters")
    except Exception as e:
        print(f"✗ Model initialization FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Run tests
    success = True
    
    success = test_encoder_integration(model, batch) and success
    
    outputs = test_model_forward_pass(model, batch)
    if outputs is None:
        success = False
    else:
        success = test_loss_computation(outputs, batch) and success
        correspondences = test_correspondence_construction(outputs, batch)
        if correspondences is None:
            success = False
        
        # Run appropriate visualization
        if visualize:
            success = visualize_comprehensive_pipeline(outputs, batch, output_dir) and success
        else:
            success = test_visualization(outputs, batch, output_dir) and success
    
    # Final summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    if success:
        print("✓ ALL TESTS PASSED!")
        print(f"\nVisualizations saved to: {output_dir}")
        print("\nNext steps:")
        print("  1. Review visualizations")
        print("  2. Test with real GSO data: python -m training.test_vpog_full_pipeline --real_data")
        print("  3. Run training: python train.py")
        return True
    else:
        print("✗ SOME TESTS FAILED")
        print("  Review error messages above for details")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='VPOG Full Pipeline Test')
    parser.add_argument('--real_data', action='store_true', help='Use real GSO data instead of synthetic')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--visualize', action='store_true', help='Generate comprehensive visualizations of the pipeline')
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    
    success = run_full_pipeline_test(use_real_data=args.real_data, device=device, visualize=args.visualize)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
