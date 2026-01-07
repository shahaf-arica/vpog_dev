"""
PnP Pose Solver using RANSAC

Estimates 6D poses from 2D-3D correspondences using:
1. OpenCV's solvePnPRansac for robust pose estimation
2. Optional refinement with solvePnP (non-RANSAC)

Compatible with both PyTorch tensors and NumPy arrays.
"""

from typing import Tuple, Optional, Union
import torch
import numpy as np
import cv2


class PnPSolver:
    """
    Solve 6D pose from 2D-3D correspondences using RANSAC+PnP.
    
    Uses OpenCV's solvePnPRansac for initial robust estimation,
    optionally followed by refinement.
    """
    
    def __init__(
        self,
        ransac_threshold: float = 8.0,  # pixels
        ransac_iterations: int = 1000,
        ransac_confidence: float = 0.99,
        min_inliers: int = 4,
        refine: bool = True,
    ):
        """
        Args:
            ransac_threshold: RANSAC inlier threshold in pixels
            ransac_iterations: Maximum RANSAC iterations
            ransac_confidence: RANSAC confidence level
            min_inliers: Minimum number of inliers for valid pose
            refine: Whether to refine pose with all inliers after RANSAC
        """
        self.ransac_threshold = ransac_threshold
        self.ransac_iterations = ransac_iterations
        self.ransac_confidence = ransac_confidence
        self.min_inliers = min_inliers
        self.refine = refine
        
    def solve(
        self,
        pts_2d: Union[torch.Tensor, np.ndarray],  # [N, 2] or [B, N, 2]
        pts_3d: Union[torch.Tensor, np.ndarray],  # [N, 3] or [B, N, 3]
        K: Union[torch.Tensor, np.ndarray],        # [3, 3] or [B, 3, 3]
        weights: Optional[Union[torch.Tensor, np.ndarray]] = None,  # [N] or [B, N]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Solve PnP with RANSAC.
        
        Args:
            pts_2d: 2D image points
            pts_3d: 3D model points
            K: Camera intrinsics matrix
            weights: Optional correspondence weights (currently unused by OpenCV)
            
        Returns:
            R: [3, 3] rotation matrix
            t: [3] translation vector
            inliers: [N_inliers] indices of inlier correspondences
            num_inliers: Number of inliers
        """
        # Convert to numpy if needed
        if isinstance(pts_2d, torch.Tensor):
            pts_2d = pts_2d.detach().cpu().numpy()
        if isinstance(pts_3d, torch.Tensor):
            pts_3d = pts_3d.detach().cpu().numpy()
        if isinstance(K, torch.Tensor):
            K = K.detach().cpu().numpy()
            
        # Handle batched input (take first batch for now)
        if pts_2d.ndim == 3:
            pts_2d = pts_2d[0]
        if pts_3d.ndim == 3:
            pts_3d = pts_3d[0]
        if K.ndim == 3:
            K = K[0]
            
        # Ensure correct shapes
        assert pts_2d.shape[1] == 2, f"pts_2d should be [N, 2], got {pts_2d.shape}"
        assert pts_3d.shape[1] == 3, f"pts_3d should be [N, 3], got {pts_3d.shape}"
        assert pts_2d.shape[0] == pts_3d.shape[0], "Number of 2D and 3D points must match"
        
        N = pts_2d.shape[0]
        
        # Check minimum number of correspondences
        if N < self.min_inliers:
            # Return identity pose with zero inliers
            R = np.eye(3, dtype=np.float32)
            t = np.zeros(3, dtype=np.float32)
            return R, t, np.array([], dtype=np.int32), 0
        
        # Extract camera intrinsics for OpenCV
        # OpenCV expects: (fx, fy, cx, cy, distCoeffs)
        camera_matrix = K.astype(np.float32)
        dist_coeffs = np.zeros(4, dtype=np.float32)  # No distortion
        
        # Reshape for OpenCV: needs [N, 1, 2] and [N, 1, 3]
        pts_2d_cv = pts_2d.astype(np.float32).reshape(-1, 1, 2)
        pts_3d_cv = pts_3d.astype(np.float32).reshape(-1, 1, 3)
        
        # Run RANSAC PnP
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=pts_3d_cv,
                imagePoints=pts_2d_cv,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
                reprojectionError=self.ransac_threshold,
                iterationsCount=self.ransac_iterations,
                confidence=self.ransac_confidence,
                flags=cv2.SOLVEPNP_EPNP,  # Use EPnP as base solver
            )
        except cv2.error as e:
            # Failed to solve - return identity
            R = np.eye(3, dtype=np.float32)
            t = np.zeros(3, dtype=np.float32)
            return R, t, np.array([], dtype=np.int32), 0
        
        if not success or inliers is None or len(inliers) < self.min_inliers:
            # Failed to find valid pose
            R = np.eye(3, dtype=np.float32)
            t = np.zeros(3, dtype=np.float32)
            return R, t, np.array([], dtype=np.int32), 0
        
        # Convert inliers to 1D array
        inliers = inliers.flatten()
        num_inliers = len(inliers)
        
        # Optionally refine with all inliers
        if self.refine and num_inliers >= self.min_inliers:
            try:
                # Use only inlier correspondences
                pts_2d_inliers = pts_2d_cv[inliers]
                pts_3d_inliers = pts_3d_cv[inliers]
                
                # Refine with solvePnP (non-RANSAC)
                success_refine, rvec_refined, tvec_refined = cv2.solvePnP(
                    objectPoints=pts_3d_inliers,
                    imagePoints=pts_2d_inliers,
                    cameraMatrix=camera_matrix,
                    distCoeffs=dist_coeffs,
                    rvec=rvec,  # Use RANSAC result as initialization
                    tvec=tvec,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                
                if success_refine:
                    rvec = rvec_refined
                    tvec = tvec_refined
            except cv2.error:
                # Refinement failed, keep RANSAC result
                pass
        
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()
        
        return R, t, inliers, num_inliers
    
    def solve_batch(
        self,
        pts_2d: Union[torch.Tensor, np.ndarray],  # [B, N, 2]
        pts_3d: Union[torch.Tensor, np.ndarray],  # [B, N, 3]
        K: Union[torch.Tensor, np.ndarray],        # [B, 3, 3] or [3, 3]
        weights: Optional[Union[torch.Tensor, np.ndarray]] = None,  # [B, N]
    ) -> Tuple[np.ndarray, np.ndarray, list, np.ndarray]:
        """
        Solve PnP for a batch of correspondence sets.
        
        Args:
            pts_2d: Batch of 2D points [B, N, 2]
            pts_3d: Batch of 3D points [B, N, 3]
            K: Camera intrinsics [B, 3, 3] or [3, 3]
            weights: Optional weights [B, N]
            
        Returns:
            R_batch: [B, 3, 3] rotation matrices
            t_batch: [B, 3] translation vectors
            inliers_batch: List of [N_inliers_i] inlier arrays per sample
            num_inliers: [B] number of inliers per sample
        """
        # Convert to numpy
        if isinstance(pts_2d, torch.Tensor):
            pts_2d = pts_2d.detach().cpu().numpy()
        if isinstance(pts_3d, torch.Tensor):
            pts_3d = pts_3d.detach().cpu().numpy()
        if isinstance(K, torch.Tensor):
            K = K.detach().cpu().numpy()
        if weights is not None and isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
            
        B = pts_2d.shape[0]
        
        # Handle single K for all batches
        if K.ndim == 2:
            K = np.tile(K[None, :, :], (B, 1, 1))
        
        # Solve for each sample in batch
        R_batch = []
        t_batch = []
        inliers_batch = []
        num_inliers_batch = []
        
        for b in range(B):
            pts_2d_b = pts_2d[b]
            pts_3d_b = pts_3d[b]
            K_b = K[b]
            weights_b = weights[b] if weights is not None else None
            
            R, t, inliers, num_inliers = self.solve(pts_2d_b, pts_3d_b, K_b, weights_b)
            
            R_batch.append(R)
            t_batch.append(t)
            inliers_batch.append(inliers)
            num_inliers_batch.append(num_inliers)
        
        R_batch = np.stack(R_batch, axis=0)  # [B, 3, 3]
        t_batch = np.stack(t_batch, axis=0)  # [B, 3]
        num_inliers_batch = np.array(num_inliers_batch)  # [B]
        
        return R_batch, t_batch, inliers_batch, num_inliers_batch
    
    def pose_to_matrix(
        self,
        R: np.ndarray,  # [3, 3] or [B, 3, 3]
        t: np.ndarray,  # [3] or [B, 3]
    ) -> np.ndarray:
        """
        Convert R, t to 4x4 transformation matrix.
        
        Args:
            R: Rotation matrix/matrices
            t: Translation vector(s)
            
        Returns:
            T: [4, 4] or [B, 4, 4] transformation matrix
        """
        if R.ndim == 2:
            # Single pose
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[:3, 3] = t
            return T
        else:
            # Batch of poses
            B = R.shape[0]
            T = np.tile(np.eye(4, dtype=np.float32)[None, :, :], (B, 1, 1))
            T[:, :3, :3] = R
            T[:, :3, 3] = t
            return T
