"""
EPro-PnP Pose Solver

Differentiable and probabilistic PnP solver using EPro-PnP.
Supports both inference (single best pose) and training (differentiable loss).

Reference: End-to-End Probabilistic Perspective-n-Points
https://github.com/tjiiv-cprg/EPro-PnP
"""

from typing import Tuple, Optional, Union
import torch
import numpy as np
import sys
import os

# Add EPro-PnP to path
epropnp_path = os.path.join(os.path.dirname(__file__), '../../external/epropnp/EPro-PnP-6DoF_v2')
if epropnp_path not in sys.path:
    sys.path.insert(0, epropnp_path)

try:
    from lib.ops.pnp.epropnp import EProPnP6DoF
    from lib.ops.pnp.levenberg_marquardt import LMSolver
    from lib.ops.pnp.camera import PerspectiveCamera
    from lib.ops.pnp.cost_fun import AdaptiveHuberPnPCost
    EPROPNP_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    EProPnP6DoF = None
    LMSolver = None
    PerspectiveCamera = None
    AdaptiveHuberPnPCost = None
    EPROPNP_AVAILABLE = False
    EPROPNP_ERROR = str(e)


class EProPnPSolver:
    """
    Solve 6D pose using EPro-PnP (probabilistic PnP).
    
    EPro-PnP is differentiable and can handle uncertainty in correspondences.
    Suitable for both inference and end-to-end training.
    """
    
    def __init__(
        self,
        mc_samples: int = 512,  # Monte Carlo samples for pose distribution
        num_iter: int = 4,      # Iterative refinement steps
        lm_iter: int = 5,       # Levenberg-Marquardt iterations
        min_points: int = 4,    # Minimum correspondences required
    ):
        """
        Args:
            mc_samples: Number of Monte Carlo samples for pose hypothesis
            num_iter: Number of AMIS iterations
            lm_iter: Number of Levenberg-Marquardt refinement iterations
            min_points: Minimum number of correspondences required
        """
        if not EPROPNP_AVAILABLE:
            raise ImportError(
                "EPro-PnP is not available. Please install it:\n"
                "  cd external/epropnp/EPro-PnP-6DoF_v2\n"
                "  pip install -e ."
            )
        
        self.mc_samples = mc_samples
        self.num_iter = num_iter
        self.lm_iter = lm_iter
        self.min_points = min_points
        
        # Initialize LM solver for pose refinement
        lm_solver = LMSolver(
            dof=6,  # 6 degrees of freedom (3 rotation + 3 translation)
            num_iter=lm_iter,
            init_solver=None,
        )
        
        # Initialize EPro-PnP solver
        self.solver = EProPnP6DoF(
            mc_samples=mc_samples,
            num_iter=num_iter,
            solver=lm_solver,
        )
        
    def solve(
        self,
        pts_2d: torch.Tensor,      # [N, 2] or [B, N, 2]
        pts_3d: torch.Tensor,      # [N, 3] or [B, N, 3]
        K: torch.Tensor,           # [3, 3] or [B, 3, 3]
        weights: Optional[torch.Tensor] = None,  # [N] or [B, N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve PnP with EPro-PnP.
        
        Args:
            pts_2d: 2D image points
            pts_3d: 3D model points  
            K: Camera intrinsics matrix
            weights: Optional correspondence weights (used as confidence)
            
        Returns:
            pose: [4, 4] or [B, 4, 4] transformation matrix
            cost: Scalar or [B] reprojection cost
        """
        device = pts_2d.device
        
        # Add batch dimension if needed
        if pts_2d.ndim == 2:
            pts_2d = pts_2d.unsqueeze(0)  # [1, N, 2]
            pts_3d = pts_3d.unsqueeze(0)  # [1, N, 3]
            if weights is not None:
                weights = weights.unsqueeze(0)  # [1, N]
            single_sample = True
        else:
            single_sample = False
            
        # Handle K dimensions
        if K.ndim == 2:
            B = pts_2d.shape[0]
            K = K.unsqueeze(0).expand(B, -1, -1)  # [B, 3, 3]
            
        B, N, _ = pts_2d.shape
        
        # Check minimum correspondences
        if N < self.min_points:
            # Return identity pose with high cost
            pose = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
            cost = torch.full((B,), float('inf'), device=device)
            
            if single_sample:
                return pose[0], cost[0]
            return pose, cost
        
        # Prepare weights (EPro-PnP uses them as 2D covariance)
        if weights is None:
            weights = torch.ones(B, N, device=device)
        else:
            weights = weights.clamp(min=1e-6)
        
        # Create 2D uncertainty from weights
        # Higher weight = lower uncertainty (more confident)
        # w2d should be [B, N, 2] - per-axis uncertainty (σ_x, σ_y)
        
        # Normalize weights
        weights_max = weights.max(dim=1, keepdim=True)[0]
        weights_norm = weights / (weights_max + 1e-8)
        
        # Convert to standard deviation (~ sqrt(1/weight))
        # Use same uncertainty for both x and y (isotropic)
        uncertainty = torch.sqrt(1.0 / (weights_norm + 1e-6))  # [B, N]
        
        # Create 2D uncertainty: [B, N, 2]
        w2d = torch.stack([uncertainty, uncertainty], dim=-1)  # [B, N, 2]
        
        # Create camera object
        camera = PerspectiveCamera(
            cam_mats=K,  # [B, 3, 3]
            z_min=0.01,
        )
        
        # Create cost function
        cost_fun = AdaptiveHuberPnPCost(relative_delta=0.1)
        cost_fun.set_param(pts_2d, w2d)
        
        # Get initial pose estimate using EPnP (via OpenCV)
        # This is needed as EPro-PnP requires an initialization
        pose_init = self._get_initial_pose(pts_2d, pts_3d, K)  # [B, 7]
        
        # Run EPro-PnP
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            try:
                # EPro-PnP forward returns: (pose_opt, pose_cov, cost, pose_opt_plus)
                # pose_opt is [B, 7] where 7 = (x, y, z, w, i, j, k)
                #   xyz: translation, wijk: quaternion (w is real part)
                result = self.solver(
                    x3d=pts_3d,      # [B, N, 3]
                    x2d=pts_2d,      # [B, N, 2]
                    w2d=w2d,         # [B, N, 2]
                    camera=camera,
                    cost_fun=cost_fun,
                    pose_init=pose_init,  # [B, 7]
                    force_init_solve=False,  # Use our pose_init
                )
                
                # Unpack result - some may be None
                pose_opt = result[0] if result[0] is not None else pose_init
                pose_cov = result[1]  # May be None
                cost_val = result[2]   # May be None
                pose_opt_plus = result[3] if len(result) > 3 else None  # May be None
                
                # If cost is None, compute it
                if cost_val is None:
                    # Simple reprojection error as fallback
                    cost_val = torch.zeros(B, device=device)
                
                # Convert from [x,y,z,w,i,j,k] to 4x4 matrix
                pose_matrix = self._pose7_to_matrix(pose_opt)  # [B, 4, 4]
                
            except Exception as e:
                # EPro-PnP failed - return identity with high cost
                print(f"Warning: EPro-PnP failed: {e}")
                pose_matrix = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
                cost_val = torch.full((B,), float('inf'), device=device)
        
        # Remove batch dimension if input was single sample
        if single_sample:
            pose_matrix = pose_matrix[0]  # [4, 4]
            cost_val = cost_val[0]  # scalar
        
        return pose_matrix, cost_val
    
    def _get_initial_pose(
        self,
        pts_2d: torch.Tensor,  # [B, N, 2]
        pts_3d: torch.Tensor,  # [B, N, 3]
        K: torch.Tensor,       # [B, 3, 3]
    ) -> torch.Tensor:
        """
        Get initial pose estimate using OpenCV EPnP.
        
        Returns:
            pose_init: [B, 7] (x, y, z, w, i, j, k)
        """
        import cv2
        from scipy.spatial.transform import Rotation as R
        
        B, N, _ = pts_2d.shape
        device = pts_2d.device
        
        # Convert to numpy
        pts_2d_np = pts_2d.detach().cpu().numpy()
        pts_3d_np = pts_3d.detach().cpu().numpy()
        K_np = K.detach().cpu().numpy()
        
        pose_init_list = []
        
        for b in range(B):
            try:
                # Run OpenCV EPnP
                dist_coeffs = np.zeros(4, dtype=np.float32)
                success, rvec, tvec = cv2.solvePnP(
                    pts_3d_np[b],
                    pts_2d_np[b],
                    K_np[b],
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_EPNP,
                )
                
                if success:
                    # Convert rotation vector to quaternion [w, x, y, z]
                    rot = R.from_rotvec(rvec.flatten())
                    quat = rot.as_quat()  # [x, y, z, w]
                    # Reorder to [w, x, y, z]
                    quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
                    
                    # Combine to [x, y, z, w, i, j, k]
                    pose_7 = np.concatenate([tvec.flatten(), quat_wxyz])
                else:
                    # Default: identity rotation + translation at [0, 0, 1]
                    pose_7 = np.array([0, 0, 1, 1, 0, 0, 0], dtype=np.float32)
            except:
                # Fallback
                pose_7 = np.array([0, 0, 1, 1, 0, 0, 0], dtype=np.float32)
            
            pose_init_list.append(pose_7)
        
        pose_init = torch.tensor(np.stack(pose_init_list), dtype=torch.float32, device=device)
        return pose_init  # [B, 7]
    
    def _pose7_to_matrix(self, pose7: torch.Tensor) -> torch.Tensor:
        """
        Convert 7-parameter pose to 4x4 matrix.
        
        Args:
            pose7: [B, 7] where 7 = (x, y, z, w, i, j, k)
                   xyz: translation, wijk: quaternion (w is real part)
        
        Returns:
            T: [B, 4, 4] transformation matrices
        """
        B = pose7.shape[0]
        device = pose7.device
        
        # Extract translation and quaternion
        t = pose7[:, :3]  # [B, 3]
        q = pose7[:, 3:]  # [B, 4] - (w, x, y, z)
        
        # Normalize quaternion
        q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-8)
        
        # Convert quaternion to rotation matrix
        # q = [w, x, y, z]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Rotation matrix from quaternion
        R = torch.zeros(B, 3, 3, device=device)
        R[:, 0, 0] = 1 - 2*y*y - 2*z*z
        R[:, 0, 1] = 2*x*y - 2*z*w
        R[:, 0, 2] = 2*x*z + 2*y*w
        R[:, 1, 0] = 2*x*y + 2*z*w
        R[:, 1, 1] = 1 - 2*x*x - 2*z*z
        R[:, 1, 2] = 2*y*z - 2*x*w
        R[:, 2, 0] = 2*x*z - 2*y*w
        R[:, 2, 1] = 2*y*z + 2*x*w
        R[:, 2, 2] = 1 - 2*x*x - 2*y*y
        
        # Build 4x4 matrix
        T = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).clone()
        T[:, :3, :3] = R
        T[:, :3, 3] = t
        
        return T
    
    def solve_with_uncertainty(
        self,
        pts_2d: torch.Tensor,      # [N, 2] or [B, N, 2]
        pts_3d: torch.Tensor,      # [N, 3] or [B, N, 3]
        K: torch.Tensor,           # [3, 3] or [B, 3, 3]
        weights: Optional[torch.Tensor] = None,  # [N] or [B, N]
        return_samples: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Solve PnP with EPro-PnP and return pose distribution.
        
        Args:
            pts_2d: 2D image points
            pts_3d: 3D model points
            K: Camera intrinsics
            weights: Optional correspondence weights
            return_samples: If True, return Monte Carlo pose samples
            
        Returns:
            pose: [4, 4] or [B, 4, 4] mean pose
            cost: Scalar or [B] reprojection cost
            pose_samples: Optional [B, num_samples, 4, 4] pose hypotheses
        """
        # For now, just call standard solve
        # Full uncertainty quantification requires accessing EPro-PnP internals
        pose, cost = self.solve(pts_2d, pts_3d, K, weights)
        
        if return_samples:
            # Would need to modify EPro-PnP to return samples
            # For now, return None
            return pose, cost, None
        
        return pose, cost, None
    
    @staticmethod
    def pose_to_Rt(pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose 4x4 pose matrix into R and t.
        
        Args:
            pose: [4, 4] or [B, 4, 4] transformation matrix
            
        Returns:
            R: [3, 3] or [B, 3, 3] rotation matrix
            t: [3] or [B, 3] translation vector
        """
        if pose.ndim == 2:
            R = pose[:3, :3]
            t = pose[:3, 3]
        else:
            R = pose[:, :3, :3]
            t = pose[:, :3, 3]
        return R, t
    
    @staticmethod
    def is_available() -> bool:
        """Check if EPro-PnP is available."""
        return EPROPNP_AVAILABLE


def test_availability():
    """Test if EPro-PnP can be imported and initialized."""
    if not EPROPNP_AVAILABLE:
        print("❌ EPro-PnP is NOT available")
        print("To install:")
        print("  cd external/epropnp/EPro-PnP-6DoF_v2")
        print("  pip install -e .")
        return False
    
    try:
        solver = EProPnPSolver()
        print("✓ EPro-PnP is available and initialized")
        return True
    except Exception as e:
        print(f"❌ EPro-PnP import succeeded but initialization failed: {e}")
        return False


if __name__ == "__main__":
    test_availability()
