"""
Correspondence Builder for VPOG

Converts patch-level predictions (classification + flow) into 2D-3D correspondences
for pose estimation via PnP solvers.

Key conversions:
- Patch-level coordinates → Absolute pixel coordinates in 224×224 crop  
- Flow in patch units → Flow in pixels (flow * patch_size)
- Template 2D pixels → 3D points in model frame (via depth + pose)

Flow semantics:
- center_flow: Displacement from buddy template patch center
- dense_flow: Displacement from baseline template pixel position
- Flow = (0, 0) means "at the buddy baseline position"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class Correspondences:
    """Container for 2D-3D correspondences"""
    pts_2d: torch.Tensor      # [N, 2] query image pixels (u, v)
    pts_3d: torch.Tensor      # [N, 3] model 3D points (x, y, z)
    weights: torch.Tensor     # [N] confidence weights
    valid_mask: torch.Tensor  # [N] boolean mask for valid correspondences
    
    def __len__(self) -> int:
        return self.pts_2d.shape[0]
    
    def filter_valid(self) -> Correspondences:
        """Return only valid correspondences"""
        mask = self.valid_mask
        return Correspondences(
            pts_2d=self.pts_2d[mask],
            pts_3d=self.pts_3d[mask],
            weights=self.weights[mask],
            valid_mask=torch.ones_like(self.weights[mask], dtype=torch.bool),
        )


class CorrespondenceBuilder:
    """
    Builds 2D-3D correspondences from VPOG model outputs.
    
    All operations are vectorized (no Python loops over patches/pixels).
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        grid_size: Tuple[int, int] = (14, 14),
        eps_b: float = 1e-4,
        min_depth: float = 0.01,  # meters
        max_depth: float = 10.0,  # meters
    ):
        """
        Args:
            img_size: Image size (H=W, typically 224)
            patch_size: Patch size in pixels (typically 16)
            grid_size: Patch grid (H_p, W_p), typically (14, 14)
            eps_b: Small constant for numerical stability in weight computation
            min_depth: Minimum valid depth value
            max_depth: Maximum valid depth value
        """
        self.img_size = img_size
        self.patch_size = patch_size
        self.H_p, self.W_p = grid_size
        self.Nq = self.H_p * self.W_p
        self.eps_b = eps_b
        self.min_depth = min_depth
        self.max_depth = max_depth
        
    def _create_patch_centers(self, device: torch.device) -> torch.Tensor:
        """
        Create patch center coordinates for 14×14 grid on 224×224 image.
        
        Returns:
            centers: [Nq, 2] tensor of (u, v) pixel coordinates
        """
        # Center of each patch: (qx + 0.5) * patch_size
        qy_coords = (torch.arange(self.H_p, device=device) + 0.5) * self.patch_size
        qx_coords = (torch.arange(self.W_p, device=device) + 0.5) * self.patch_size
        
        # Create grid: [H_p, W_p, 2]
        qy_grid, qx_grid = torch.meshgrid(qy_coords, qx_coords, indexing='ij')
        centers = torch.stack([qx_grid, qy_grid], dim=-1)  # [H_p, W_p, 2] as (u, v)
        
        return centers.reshape(self.Nq, 2)  # [Nq, 2]
    
    def _patch_idx_to_coords(self, patch_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert flat patch indices to (qy, qx) grid coordinates.
        
        Args:
            patch_idx: [...] arbitrary shape of patch indices [0, Nq-1]
            
        Returns:
            qy: [...] row indices
            qx: [...] column indices
        """
        qy = patch_idx // self.W_p
        qx = patch_idx % self.W_p
        return qy, qx
    
    def _backproject_pixel(
        self,
        uv: torch.Tensor,      # [..., 2] pixel coordinates (u, v)
        depth: torch.Tensor,   # [...] depth values
        K: torch.Tensor,       # [3, 3] or [..., 3, 3] camera intrinsics
    ) -> torch.Tensor:
        """
        Backproject 2D pixels to 3D camera coordinates.
        
        Args:
            uv: Pixel coordinates (u, v)
            depth: Depth at those pixels
            K: Camera intrinsics
            
        Returns:
            pts_3d: [..., 3] points in camera frame
        """
        # Handle batched K
        u, v = uv[..., 0], uv[..., 1]
        
        if K.dim() == 2:
            # Single intrinsics matrix
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
        else:
            # Batched intrinsics: K is [..., 3, 3], we need to broadcast to match uv shape
            # K could be [B, 3, 3] and uv [B, Nq, 2], so we need K to be [B, 1, 3, 3] -> extract and unsqueeze
            fx = K[..., 0, 0]
            fy = K[..., 1, 1]
            cx = K[..., 0, 2]
            cy = K[..., 1, 2]
            
            # Add dimensions to broadcast with uv
            # If uv is [B, Nq, 2], u is [B, Nq], and fx is [B], we need fx to be [B, 1]
            while fx.ndim < u.ndim:
                fx = fx.unsqueeze(-1)
                fy = fy.unsqueeze(-1)
                cx = cx.unsqueeze(-1)
                cy = cy.unsqueeze(-1)
        
        # Compute normalized ray
        x = (u - cx) / fx
        y = (v - cy) / fy
        z = torch.ones_like(x)
        
        # Scale by depth
        pts_3d = torch.stack([x * depth, y * depth, z * depth], dim=-1)
        return pts_3d
    
    def _transform_points(
        self,
        pts: torch.Tensor,    # [..., 3] points in camera frame
        T_inv: torch.Tensor,  # [4, 4] or [..., 4, 4] inverse pose (cam → model)
    ) -> torch.Tensor:
        """
        Transform points from camera frame to model frame.
        
        Args:
            pts: Points in camera coordinates
            T_inv: Inverse of T_m2c (i.e., T_c2m)
            
        Returns:
            pts_model: [..., 3] points in model frame
        """
        # Handle batched T_inv
        if T_inv.dim() == 2:
            R = T_inv[:3, :3]  # [3, 3]
            t = T_inv[:3, 3]    # [3]
        else:
            R = T_inv[..., :3, :3]  # [..., 3, 3]
            t = T_inv[..., :3, 3]    # [..., 3]
        
        # Apply rotation and translation
        # pts_model = R @ pts + t
        # pts is [..., 3], R is either [3, 3] or [B, 3, 3]
        # t is either [3] or [B, 3]
        
        if R.ndim == 2:
            # Single rotation matrix: simple matrix multiply
            pts_model = pts @ R.T + t  # [..., 3] @ [3, 3] + [3]
        else:
            # Batched rotation: R is [B, 3, 3], pts could be [B, ...rest..., 3]
            # We need to flatten all middle dimensions for matmul
            pts_shape = pts.shape
            B = pts_shape[0]
            
            # Flatten to [B, N, 3] where N is product of all middle dims
            pts_flat = pts.reshape(B, -1, 3)  # [B, N, 3]
            
            # Apply rotation: [B, N, 3] @ [B, 3, 3].T = [B, N, 3]
            pts_rotated = torch.bmm(pts_flat, R.transpose(-1, -2))  # [B, N, 3]
            
            # Add translation: [B, N, 3] + [B, 1, 3]
            t_expanded = t.unsqueeze(-2)  # [B, 1, 3]
            pts_model_flat = pts_rotated + t_expanded  # [B, N, 3]
            
            # Reshape back to original shape
            pts_model = pts_model_flat.reshape(pts_shape)
        
        return pts_model
    
    def _sample_depth_bilinear(
        self,
        depth_map: torch.Tensor,  # [H, W] or [B, H, W]
        uv: torch.Tensor,         # [..., 2] pixel coordinates (u, v)
    ) -> torch.Tensor:
        """
        Sample depth map at continuous pixel coordinates using bilinear interpolation.
        
        Args:
            depth_map: Depth map(s)
            uv: Pixel coordinates (u, v)
            
        Returns:
            depth: [...] sampled depth values
        """
        # Normalize coordinates to [-1, 1] for grid_sample
        H, W = depth_map.shape[-2:]
        uv_norm = uv.clone()
        uv_norm[..., 0] = (uv[..., 0] / (W - 1)) * 2 - 1  # u
        uv_norm[..., 1] = (uv[..., 1] / (H - 1)) * 2 - 1  # v
        
        # grid_sample expects [B, C, H, W] input and [B, ..., 2] grid
        if depth_map.dim() == 2:
            depth_map = depth_map[None, None, :, :]  # [1, 1, H, W]
            batch_added = True
        elif depth_map.dim() == 3:
            depth_map = depth_map[:, None, :, :]  # [B, 1, H, W]
            batch_added = False
        else:
            batch_added = False
        
        # Reshape uv_norm for grid_sample
        original_shape = uv_norm.shape[:-1]
        B = depth_map.shape[0]
        uv_norm_flat = uv_norm.reshape(B, -1, 1, 2)  # [B, N, 1, 2]
        
        # Sample
        depth_sampled = F.grid_sample(
            depth_map,
            uv_norm_flat,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )  # [B, 1, N, 1]
        
        depth_sampled = depth_sampled.squeeze(1).squeeze(-1)  # [B, N]
        
        # Reshape back
        if batch_added:
            depth_sampled = depth_sampled.squeeze(0)  # Remove added batch dim
            if len(original_shape) > 0:
                depth_sampled = depth_sampled.reshape(original_shape)
        else:
            if len(original_shape) > 1:
                depth_sampled = depth_sampled.reshape(original_shape)
        
        return depth_sampled
    
    def build_coarse_correspondences(
        self,
        classification_logits: torch.Tensor,  # [B, S, Nq, Nt+1]
        center_flow: torch.Tensor,            # [B, S, Nq, 2] in patch units
        template_depth: torch.Tensor,         # [B, S, H, W] or [S, H, W]
        K_template: torch.Tensor,             # [B, S, 3, 3] or [S, 3, 3] or [3, 3]
        poses_template: torch.Tensor,         # [B, S, 4, 4] or [S, 4, 4] template poses (T_m2c)
        selected_template_idx: int,           # Which template was selected (index in S dimension)
    ) -> Correspondences:
        """
        Build coarse correspondences from center-flow predictions.
        
        For each query patch:
        - Query 2D: patch center pixel
        - Template 2D: buddy patch center + center_flow * patch_size
        - Template 3D: backproject + transform to model frame
        
        Args:
            classification_logits: Classification scores for patch matching
            center_flow: Patch-level flow predictions (displacement from buddy center)
            template_depth: Depth maps for templates
            K_template: Camera intrinsics for templates
            poses_template: Template poses (model-to-camera)
            selected_template_idx: Index of selected template in S dimension
            
        Returns:
            Correspondences object with 2D-3D matches
        """
        device = classification_logits.device
        B, S, Nq, Nt1 = classification_logits.shape
        Nt = Nt1 - 1  # Exclude unseen token
        
        # Select data for chosen template
        s = selected_template_idx
        logits_s = classification_logits[:, s, :, :Nt]  # [B, Nq, Nt]
        flow_s = center_flow[:, s, :, :]  # [B, Nq, 2]
        depth_s = template_depth[:, s] if template_depth.dim() == 4 else template_depth[s]  # [B, H, W] or [H, W]
        K_s = K_template[:, s] if K_template.dim() == 4 else (K_template[s] if K_template.dim() == 3 else K_template)
        pose_s = poses_template[:, s] if poses_template.dim() == 4 else poses_template[s]  # [B, 4, 4] or [4, 4]
        
        # Infer buddy template patch for each query patch: argmax over template patches
        buddy_idx = logits_s.argmax(dim=-1)  # [B, Nq]
        max_conf = logits_s.max(dim=-1)[0]  # [B, Nq]
        
        # Create query patch centers: [Nq, 2]
        query_centers = self._create_patch_centers(device)  # [Nq, 2]
        
        # Expand for batch: [B, Nq, 2]
        query_2d = query_centers[None, :, :].expand(B, Nq, 2)
        
        # Compute buddy template patch centers
        buddy_qy, buddy_qx = self._patch_idx_to_coords(buddy_idx)  # [B, Nq]
        buddy_center_u = (buddy_qx.float() + 0.5) * self.patch_size
        buddy_center_v = (buddy_qy.float() + 0.5) * self.patch_size
        buddy_centers = torch.stack([buddy_center_u, buddy_center_v], dim=-1)  # [B, Nq, 2]
        
        # Apply center flow: displacement from buddy center (in patch units)
        template_2d = buddy_centers + flow_s * self.patch_size  # [B, Nq, 2]
        
        # Check bounds
        valid_bounds = (
            (template_2d[..., 0] >= 0) & (template_2d[..., 0] < self.img_size) &
            (template_2d[..., 1] >= 0) & (template_2d[..., 1] < self.img_size)
        )  # [B, Nq]
        
        # Sample depth at template 2D points
        if depth_s.dim() == 2:
            # Single depth map for all batches
            depth_vals = torch.stack([
                self._sample_depth_bilinear(depth_s, template_2d[b])
                for b in range(B)
            ])  # [B, Nq]
        else:
            # Batched depth maps
            depth_vals = torch.stack([
                self._sample_depth_bilinear(depth_s[b], template_2d[b])
                for b in range(B)
            ])  # [B, Nq]
        
        # Check valid depth
        valid_depth = (depth_vals > self.min_depth) & (depth_vals < self.max_depth)
        
        # Backproject to camera frame
        pts_3d_cam = self._backproject_pixel(template_2d, depth_vals, K_s)  # [B, Nq, 3]
        
        # Transform to model frame using inverse pose
        if pose_s.dim() == 2:
            # Single pose for all batches
            T_inv = torch.inverse(pose_s)  # [4, 4]
        else:
            # Batched poses
            T_inv = torch.inverse(pose_s)  # [B, 4, 4]
        
        pts_3d_model = self._transform_points(pts_3d_cam, T_inv)  # [B, Nq, 3]
        
        # Weights from classification confidence
        weights = max_conf  # [B, Nq]
        
        # Combined validity mask
        valid_mask = valid_bounds & valid_depth  # [B, Nq]
        
        # Flatten batch dimension for output
        pts_2d = query_2d.reshape(B * Nq, 2)
        pts_3d = pts_3d_model.reshape(B * Nq, 3)
        weights = weights.reshape(B * Nq)
        valid_mask = valid_mask.reshape(B * Nq)
        
        return Correspondences(
            pts_2d=pts_2d,
            pts_3d=pts_3d,
            weights=weights,
            valid_mask=valid_mask,
        )
    
    def build_refined_correspondences(
        self,
        classification_logits: torch.Tensor,  # [B, S, Nq, Nt+1]
        dense_flow: torch.Tensor,             # [B, S, Nq, ps, ps, 2] in patch units
        dense_b: torch.Tensor,                # [B, S, Nq, ps, ps] Laplace scale
        template_depth: torch.Tensor,         # [B, S, H, W] or [S, H, W]
        K_template: torch.Tensor,             # [B, S, 3, 3] or [S, 3, 3] or [3, 3]
        poses_template: torch.Tensor,         # [B, S, 4, 4] or [S, 4, 4]
        selected_template_idx: int,
        dense_weight: Optional[torch.Tensor] = None,  # [B, S, Nq, ps, ps] visibility weight
    ) -> Correspondences:
        """
        Build refined dense correspondences from pixel-level flow predictions.
        
        For each pixel within each query patch:
        - Query 2D: absolute pixel in query image
        - Template 2D: baseline pixel in buddy template patch + dense_flow * patch_size
        - Template 3D: backproject + transform to model frame
        - Weight: 1 / (b + eps) * dense_weight
        
        Args:
            classification_logits: Classification scores
            dense_flow: Dense pixel-level flow (displacement from baseline)
            dense_b: Laplace scale (uncertainty)
            template_depth: Template depth maps
            K_template: Template camera intrinsics
            poses_template: Template poses
            selected_template_idx: Selected template index
            dense_weight: Optional visibility/supervision weights
            
        Returns:
            Correspondences object with dense 2D-3D matches
        """
        device = classification_logits.device
        B, S, Nq, ps, ps2, _ = dense_flow.shape
        assert ps == ps2
        
        # Select data for chosen template
        s = selected_template_idx
        logits_s = classification_logits[:, s, :, :-1]  # [B, Nq, Nt]
        flow_s = dense_flow[:, s]  # [B, Nq, ps, ps, 2]
        b_s = dense_b[:, s]  # [B, Nq, ps, ps]
        depth_s = template_depth[:, s] if template_depth.dim() == 4 else template_depth[s]
        K_s = K_template[:, s] if K_template.dim() == 4 else (K_template[s] if K_template.dim() == 3 else K_template)
        pose_s = poses_template[:, s] if poses_template.dim() == 4 else poses_template[s]
        
        if dense_weight is not None:
            weight_s = dense_weight[:, s]  # [B, Nq, ps, ps]
        else:
            weight_s = torch.ones(B, Nq, ps, ps, device=device)
        
        # Infer buddy patches
        buddy_idx = logits_s.argmax(dim=-1)  # [B, Nq]
        
        # Create dense pixel grids
        # Query pixels: qx * ps + delta_u, qy * ps + delta_v
        delta_coords = torch.arange(ps, device=device)  # [ps]
        delta_v, delta_u = torch.meshgrid(delta_coords, delta_coords, indexing='ij')  # [ps, ps]
        delta_uv = torch.stack([delta_u, delta_v], dim=-1).float()  # [ps, ps, 2]
        
        # For each query patch, compute base pixel positions
        # Query patch indices to grid coords
        q_indices = torch.arange(Nq, device=device)
        qy, qx = self._patch_idx_to_coords(q_indices)  # [Nq]
        
        # Query pixels: [Nq, ps, ps, 2]
        query_base = torch.stack([
            qx[:, None, None].float() * self.patch_size,
            qy[:, None, None].float() * self.patch_size,
        ], dim=-1)  # [Nq, 1, 1, 2]
        query_pixels = query_base + delta_uv[None, :, :, :]  # [Nq, ps, ps, 2]
        
        # Expand for batch: [B, Nq, ps, ps, 2]
        query_pixels = query_pixels[None, :, :, :, :].expand(B, Nq, ps, ps, 2)
        
        # Template buddy patch base positions: [B, Nq]
        buddy_qy, buddy_qx = self._patch_idx_to_coords(buddy_idx)  # [B, Nq]
        
        # Template baseline pixels: [B, Nq, ps, ps, 2]
        template_base = torch.stack([
            buddy_qx[:, :, None, None].float() * self.patch_size,
            buddy_qy[:, :, None, None].float() * self.patch_size,
        ], dim=-1)  # [B, Nq, 1, 1, 2]
        template_baseline = template_base + delta_uv[None, None, :, :, :]  # [B, Nq, ps, ps, 2]
        
        # Apply dense flow: displacement from baseline (in patch units)
        template_pixels = template_baseline + flow_s * self.patch_size  # [B, Nq, ps, ps, 2]
        
        # Check bounds
        valid_bounds = (
            (template_pixels[..., 0] >= 0) & (template_pixels[..., 0] < self.img_size) &
            (template_pixels[..., 1] >= 0) & (template_pixels[..., 1] < self.img_size)
        )  # [B, Nq, ps, ps]
        
        # Sample depth at all template pixels
        # Flatten spatial dimensions for sampling
        template_pixels_flat = template_pixels.reshape(B, Nq * ps * ps, 2)  # [B, N_dense, 2]
        
        if depth_s.dim() == 2:
            depth_vals = torch.stack([
                self._sample_depth_bilinear(depth_s, template_pixels_flat[b])
                for b in range(B)
            ])  # [B, N_dense]
        else:
            depth_vals = torch.stack([
                self._sample_depth_bilinear(depth_s[b], template_pixels_flat[b])
                for b in range(B)
            ])  # [B, N_dense]
        
        depth_vals = depth_vals.reshape(B, Nq, ps, ps)  # [B, Nq, ps, ps]
        
        # Check valid depth
        valid_depth = (depth_vals > self.min_depth) & (depth_vals < self.max_depth)
        
        # Backproject template pixels to camera frame
        pts_3d_cam = self._backproject_pixel(template_pixels, depth_vals, K_s)  # [B, Nq, ps, ps, 3]
        
        # Transform to model frame
        if pose_s.dim() == 2:
            T_inv = torch.inverse(pose_s)
        else:
            T_inv = torch.inverse(pose_s)
        
        pts_3d_model = self._transform_points(pts_3d_cam, T_inv)  # [B, Nq, ps, ps, 3]
        
        # Compute weights from confidence: 1 / (b + eps)
        conf_weights = 1.0 / (b_s + self.eps_b)  # [B, Nq, ps, ps]
        weights = conf_weights * weight_s  # [B, Nq, ps, ps]
        
        # Combined validity mask
        valid_mask = valid_bounds & valid_depth  # [B, Nq, ps, ps]
        
        # Flatten all dimensions for output
        pts_2d = query_pixels.reshape(B * Nq * ps * ps, 2)
        pts_3d = pts_3d_model.reshape(B * Nq * ps * ps, 3)
        weights = weights.reshape(B * Nq * ps * ps)
        valid_mask = valid_mask.reshape(B * Nq * ps * ps)
        
        return Correspondences(
            pts_2d=pts_2d,
            pts_3d=pts_3d,
            weights=weights,
            valid_mask=valid_mask,
        )


# Convenience functions

def build_coarse_correspondences(
    classification_logits: torch.Tensor,
    center_flow: torch.Tensor,
    template_depth: torch.Tensor,
    K_template: torch.Tensor,
    poses_template: torch.Tensor,
    selected_template_idx: int,
    img_size: int = 224,
    patch_size: int = 16,
    **kwargs,
) -> Correspondences:
    """Convenience function for building coarse correspondences."""
    builder = CorrespondenceBuilder(img_size=img_size, patch_size=patch_size, **kwargs)
    return builder.build_coarse_correspondences(
        classification_logits,
        center_flow,
        template_depth,
        K_template,
        poses_template,
        selected_template_idx,
    )


def build_refined_correspondences(
    classification_logits: torch.Tensor,
    dense_flow: torch.Tensor,
    dense_b: torch.Tensor,
    template_depth: torch.Tensor,
    K_template: torch.Tensor,
    poses_template: torch.Tensor,
    selected_template_idx: int,
    dense_weight: Optional[torch.Tensor] = None,
    img_size: int = 224,
    patch_size: int = 16,
    **kwargs,
) -> Correspondences:
    """Convenience function for building refined correspondences."""
    builder = CorrespondenceBuilder(img_size=img_size, patch_size=patch_size, **kwargs)
    return builder.build_refined_correspondences(
        classification_logits,
        dense_flow,
        dense_b,
        template_depth,
        K_template,
        poses_template,
        selected_template_idx,
        dense_weight=dense_weight,
    )
