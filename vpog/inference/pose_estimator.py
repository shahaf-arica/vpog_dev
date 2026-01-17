"""
Pose Estimator for VPOG

Converts VPOGPredictor correspondences to 6D poses using PnP.
"""

from typing import Dict, Tuple, Optional, Any
import torch
import numpy as np
from vpog.inference.pose_solver import PnPSolver


class PoseEstimator:
    """
    Estimates 6D pose from VPOGPredictor correspondence outputs.
    
    Handles:
    - Extracting 2D-3D correspondences from predictor output
    - Sampling 3D points from template depth maps
    - Coordinate transformations (cropped -> original space)
    - RANSAC-PnP pose estimation
    """
    
    def __init__(
        self,
        ransac_threshold: float = 8.0,
        ransac_iterations: int = 1000,
        ransac_confidence: float = 0.99,
        min_inliers: int = 4,
        img_size: int = 224,
        patch_size: int = 16,
    ):
        """
        Args:
            ransac_threshold: RANSAC inlier threshold (pixels)
            ransac_iterations: Maximum RANSAC iterations
            ransac_confidence: RANSAC confidence level
            min_inliers: Minimum inliers for valid pose
            img_size: Input image size (cropped)
            patch_size: Patch size in pixels
        """
        self.pnp_solver = PnPSolver(
            ransac_threshold=ransac_threshold,
            ransac_iterations=ransac_iterations,
            ransac_confidence=ransac_confidence,
            min_inliers=min_inliers,
            refine=True,
        )
        self.img_size = img_size
        self.patch_size = patch_size
    
    def estimate_pose(
        self,
        correspondences: Dict[str, Any],
        template_depth: torch.Tensor,      # [H, W]
        template_K: torch.Tensor,          # [3, 3]
        template_pose: torch.Tensor,       # [4, 4] T_model_to_camera
        query_K_original: torch.Tensor,    # [3, 3] original intrinsics
        M_query: torch.Tensor,             # [3, 3] crop transformation
        scale_factor: float = 10.0,
        use_refined: bool = True,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Estimate 6D pose from correspondences.
        
        Args:
            correspondences: Dict from VPOGPredictor for one template
            template_depth: Template depth map [H, W]
            template_K: Template camera intrinsics [3, 3]
            template_pose: Template pose (model-to-camera) [4, 4]
            query_K_original: Query camera intrinsics in original image [3, 3]
            M_query: Crop transformation matrix [3, 3]
            scale_factor: CAD model scale factor (default 10.0)
            use_refined: Use refined correspondences if available
        
        Returns:
            pose: [4, 4] estimated pose matrix
            num_inliers: Number of RANSAC inliers
            score: Inlier ratio
        """
        # Extract correspondences
        if use_refined and len(correspondences["refined"]["q_uv"]) > 0:
            q_uv_cropped = correspondences["refined"]["q_uv"].cpu().numpy()  # [N, 2]
            t_uv_cropped = correspondences["refined"]["t_uv"].cpu().numpy()  # [N, 2]
        elif len(correspondences["coarse"]["q_center_uv"]) > 0:
            q_uv_cropped = correspondences["coarse"]["q_center_uv"].cpu().numpy()  # [N, 2]
            t_uv_cropped = correspondences["coarse"]["t_center_uv"].cpu().numpy()  # [N, 2]
        else:
            return np.eye(4), 0, 0.0
        
        if len(q_uv_cropped) < 4:
            return np.eye(4), 0, 0.0
        
        # Transform query 2D points from cropped to original space
        M_inv = torch.inverse(M_query).cpu().numpy()  # [3, 3]
        q_uv_original = self._transform_to_original(q_uv_cropped, M_inv)
        
        # Extract 3D points from template depth
        pts_3d = self._extract_3d_points(
            t_uv_cropped,
            template_depth,
            template_K,
            template_pose,
            scale_factor,
        )
        
        # Solve PnP
        K_original = query_K_original.cpu().numpy()
        R, t, inliers, num_inliers = self.pnp_solver.solve(
            pts_2d=q_uv_original,
            pts_3d=pts_3d,
            K=K_original,
        )
        
        pose = self.pnp_solver.pose_to_matrix(R, t)
        score = num_inliers / len(q_uv_cropped) if len(q_uv_cropped) > 0 else 0.0
        
        return pose, num_inliers, score
    
    def _transform_to_original(
        self,
        pts_cropped: np.ndarray,  # [N, 2]
        M_inv: np.ndarray,        # [3, 3]
    ) -> np.ndarray:
        """Transform 2D points from cropped to original image space."""
        pts_homo = np.concatenate([pts_cropped, np.ones((len(pts_cropped), 1))], axis=1)
        pts_original_homo = (M_inv @ pts_homo.T).T
        return pts_original_homo[:, :2] / pts_original_homo[:, 2:3]
    
    def _extract_3d_points(
        self,
        t_uv: np.ndarray,              # [N, 2] 2D points in template
        template_depth: torch.Tensor,   # [H, W]
        template_K: torch.Tensor,       # [3, 3]
        template_pose: torch.Tensor,    # [4, 4]
        scale_factor: float,
    ) -> np.ndarray:
        """Extract 3D points from template depth map."""
        device = template_depth.device
        H, W = template_depth.shape
        
        # Sample depth
        u = torch.from_numpy(t_uv[:, 0]).float().to(device)
        v = torch.from_numpy(t_uv[:, 1]).float().to(device)
        u = torch.clamp(u, 0, W - 1)
        v = torch.clamp(v, 0, H - 1)
        
        # Bilinear interpolation
        u_floor = u.long()
        v_floor = v.long()
        u_ceil = torch.clamp(u_floor + 1, 0, W - 1)
        v_ceil = torch.clamp(v_floor + 1, 0, H - 1)
        
        u_frac = u - u_floor.float()
        v_frac = v - v_floor.float()
        
        d00 = template_depth[v_floor, u_floor]
        d01 = template_depth[v_floor, u_ceil]
        d10 = template_depth[v_ceil, u_floor]
        d11 = template_depth[v_ceil, u_ceil]
        
        d0 = d00 * (1 - u_frac) + d01 * u_frac
        d1 = d10 * (1 - u_frac) + d11 * u_frac
        depth_values = d0 * (1 - v_frac) + d1 * v_frac
        
        # Unproject to 3D
        fx, fy = template_K[0, 0], template_K[1, 1]
        cx, cy = template_K[0, 2], template_K[1, 2]
        
        X_cam = (u - cx) * depth_values / fx
        Y_cam = (v - cy) * depth_values / fy
        Z_cam = depth_values
        
        pts_3d_cam = torch.stack([X_cam, Y_cam, Z_cam], dim=1)  # [N, 3]
        
        # Transform to world frame
        T_c2w = torch.inverse(template_pose)
        pts_homo = torch.cat([pts_3d_cam, torch.ones((len(pts_3d_cam), 1), device=device)], dim=1)
        pts_world_scaled = (T_c2w @ pts_homo.T).T[:, :3]
        
        # Scale to BOP coordinates
        pts_3d_world = pts_world_scaled / scale_factor
        
        return pts_3d_world.cpu().numpy()
