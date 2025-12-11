# vpog/losses/epro_pnp_loss.py
#
# EPro-PnP loss for end-to-end pose estimation
#
# Integrates with external/epropnp/ to compute differentiable PnP loss
#
# Input:
#   - classification_logits: [B, S, Nq, Nt+num_added] - template matching scores
#   - flow: [B, S, Nq, Nt, 16, 16, 2] - pixel-level flow predictions
#   - confidence: [B, S, Nq, Nt, 16, 16] - flow confidence
#   - query_patches: [B, S, Nq, ...] - query patch info with 2D coords
#   - template_patches: [B, S, Nt, ...] - template patch info with 3D coords
#   - camera_intrinsics: [B, S, 3, 3] - camera K matrix
#   - gt_pose: [B, S, 4, 4] - ground truth pose
#
# Output:
#   - Scalar pose loss (rotation + translation error)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
import os

# Add EPro-PnP to path
epropnp_path = os.path.join(os.path.dirname(__file__), '../../external/epropnp/EPro-PnP-6DoF_v2')
if epropnp_path not in sys.path:
    sys.path.insert(0, epropnp_path)


class EProPnPLoss(nn.Module):
    """
    EPro-PnP loss for end-to-end pose estimation.
    
    Converts classification + flow predictions to 2D-3D correspondences,
    then uses EPro-PnP for differentiable PnP solving.
    """
    
    def __init__(
        self,
        conf_threshold: float = 0.5,
        min_correspondences: int = 4,
        rotation_weight: float = 1.0,
        translation_weight: float = 1.0,
        use_epropnp: bool = True,
    ):
        """
        Args:
            conf_threshold: Minimum confidence for correspondence (default 0.5)
            min_correspondences: Minimum number of correspondences (default 4)
            rotation_weight: Weight for rotation error (default 1.0)
            translation_weight: Weight for translation error (default 1.0)
            use_epropnp: Use EPro-PnP if available, else skip (default True)
        """
        super().__init__()
        self.conf_threshold = conf_threshold
        self.min_correspondences = min_correspondences
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
        self.use_epropnp = use_epropnp
        
        # Try to import EPro-PnP
        self.epropnp = None
        if self.use_epropnp:
            try:
                from EProPnP import EProPnP
                self.epropnp = EProPnP()
            except ImportError:
                print("Warning: EPro-PnP not available, EProPnPLoss will return zero")
                self.use_epropnp = False
    
    def forward(
        self,
        classification_logits: torch.Tensor,
        flow: torch.Tensor,
        confidence: torch.Tensor,
        query_data: dict,
        template_data: dict,
        camera_intrinsics: torch.Tensor,
        gt_pose: torch.Tensor,
        unseen_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute EPro-PnP pose loss.
        
        Args:
            classification_logits: [B, S, Nq, Nt+num_added] - matching scores
            flow: [B, S, Nq, Nt, 16, 16, 2] - flow predictions
            confidence: [B, S, Nq, Nt, 16, 16] - flow confidence
            query_data: dict with 'patches' (2D coords), 'positions', etc.
            template_data: dict with 'patches' (3D coords), 'positions', etc.
            camera_intrinsics: [B, S, 3, 3] - camera K
            gt_pose: [B, S, 4, 4] - ground truth pose
            unseen_mask: [B, S, Nq, Nt, 16, 16] - optional unseen mask
        
        Returns:
            Scalar loss
        """
        if not self.use_epropnp or self.epropnp is None:
            # EPro-PnP not available
            return torch.tensor(0.0, device=flow.device, dtype=flow.dtype)
        
        B, S, Nq, Nt_plus, _ = classification_logits.shape
        Nt = flow.shape[3]
        
        # Build 2D-3D correspondences
        correspondences = self._build_correspondences(
            classification_logits[:, :, :, :Nt],  # Exclude added tokens
            flow,
            confidence,
            query_data,
            template_data,
            unseen_mask,
        )
        
        if correspondences is None or len(correspondences['pts2d']) < self.min_correspondences:
            # Not enough correspondences
            return torch.tensor(0.0, device=flow.device, dtype=flow.dtype)
        
        # Run EPro-PnP
        try:
            pred_pose = self._solve_pnp(
                correspondences['pts2d'],
                correspondences['pts3d'],
                correspondences['weights'],
                camera_intrinsics,
            )
        except Exception as e:
            print(f"Warning: EPro-PnP failed: {e}")
            return torch.tensor(0.0, device=flow.device, dtype=flow.dtype)
        
        # Compute pose error
        loss = self._compute_pose_error(pred_pose, gt_pose)
        
        return loss
    
    def _build_correspondences(
        self,
        logits: torch.Tensor,
        flow: torch.Tensor,
        confidence: torch.Tensor,
        query_data: dict,
        template_data: dict,
        unseen_mask: Optional[torch.Tensor],
    ) -> Optional[dict]:
        """
        Build 2D-3D correspondences from predictions.
        
        Returns:
            dict with 'pts2d', 'pts3d', 'weights' or None
        """
        B, S, Nq, Nt = logits.shape[:4]
        
        # Get top template matches per query patch
        # For simplicity, take argmax (could use soft matching)
        match_indices = logits.argmax(dim=-1)  # [B, S, Nq]
        match_probs = F.softmax(logits, dim=-1)  # [B, S, Nq, Nt]
        
        pts2d_list = []
        pts3d_list = []
        weights_list = []
        
        # TODO: This is a simplified version. Full implementation needs:
        # 1. Extract 2D positions from query patches
        # 2. Apply flow to get refined 2D locations
        # 3. Extract 3D positions from template patches
        # 4. Filter by confidence threshold
        # 5. Aggregate into correspondence lists
        
        # For now, return None (requires full correspondence implementation)
        return None
    
    def _solve_pnp(
        self,
        pts2d: torch.Tensor,
        pts3d: torch.Tensor,
        weights: torch.Tensor,
        camera_K: torch.Tensor,
    ) -> torch.Tensor:
        """
        Solve PnP using EPro-PnP.
        
        Returns:
            Predicted pose [B, S, 4, 4]
        """
        # TODO: Call EPro-PnP API
        # This requires proper integration with EPro-PnP forward pass
        raise NotImplementedError("EPro-PnP integration requires correspondence construction")
    
    def _compute_pose_error(
        self,
        pred_pose: torch.Tensor,
        gt_pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pose error (rotation + translation).
        
        Args:
            pred_pose: [B, S, 4, 4] - predicted pose
            gt_pose: [B, S, 4, 4] - ground truth pose
        
        Returns:
            Scalar loss
        """
        # Extract rotation and translation
        pred_R = pred_pose[:, :, :3, :3]
        pred_t = pred_pose[:, :, :3, 3]
        gt_R = gt_pose[:, :, :3, :3]
        gt_t = gt_pose[:, :, :3, 3]
        
        # Rotation error (geodesic distance on SO(3))
        # trace(R_pred^T @ R_gt) = trace(R_pred @ R_gt^T)
        R_diff = torch.matmul(pred_R, gt_R.transpose(-1, -2))
        trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
        # angle = arccos((trace - 1) / 2)
        cos_angle = (trace - 1.0) / 2.0
        cos_angle = cos_angle.clamp(-1.0, 1.0)
        angle = torch.acos(cos_angle)
        rot_error = angle.mean()
        
        # Translation error (L2 distance)
        trans_error = (pred_t - gt_t).norm(dim=-1).mean()
        
        # Combined loss
        loss = self.rotation_weight * rot_error + self.translation_weight * trans_error
        
        return loss
