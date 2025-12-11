# vpog/inference/cluster_mode.py
#
# Cluster mode inference for VPOG
#
# Uses top-4 templates for pose estimation
#
# Process:
#   1. Select top-4 templates by some metric (e.g., random, fixed, or pre-selected)
#   2. Run VPOG model on query + top-4 templates
#   3. Build 2D-3D correspondences from predictions
#   4. Run EPro-PnP for pose estimation
#
# Input:
#   - query_image: [H, W, 3] or [B, H, W, 3]
#   - templates: List of template data (images, meshes, poses)
#   - template_selector: Method to select top-4 templates
#
# Output:
#   - predicted_pose: [4, 4] or [B, 4, 4]
#   - confidence: scalar or [B]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add EPro-PnP to path
epropnp_path = os.path.join(os.path.dirname(__file__), '../../external/epropnp/EPro-PnP-6DoF_v2')
if epropnp_path not in sys.path:
    sys.path.insert(0, epropnp_path)

from vpog.models.vpog_model import VPOGModel
from vpog.inference.correspondence import CorrespondenceBuilder


class ClusterModeInference(nn.Module):
    """
    Cluster mode inference using top-4 templates.
    
    Processes query with a small set of selected templates for fast inference.
    """
    
    def __init__(
        self,
        model: VPOGModel,
        correspondence_builder: CorrespondenceBuilder,
        num_templates: int = 4,
        use_epropnp: bool = True,
    ):
        """
        Args:
            model: VPOG model
            correspondence_builder: Correspondence builder
            num_templates: Number of templates to use (default 4)
            use_epropnp: Use EPro-PnP for pose estimation (default True)
        """
        super().__init__()
        self.model = model
        self.correspondence_builder = correspondence_builder
        self.num_templates = num_templates
        self.use_epropnp = use_epropnp
        
        # Try to load EPro-PnP
        self.epropnp = None
        if self.use_epropnp:
            try:
                from EProPnP import EProPnP
                self.epropnp = EProPnP()
            except ImportError:
                print("Warning: EPro-PnP not available")
                self.use_epropnp = False
    
    def forward(
        self,
        query_data: dict,
        template_data: dict,
        camera_intrinsics: torch.Tensor,
        template_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run cluster mode inference.
        
        Args:
            query_data: dict with query image data
            template_data: dict with ALL template data
            camera_intrinsics: [B, 3, 3] or [3, 3] - camera K matrix
            template_indices: [B, num_templates] - indices of templates to use (optional)
        
        Returns:
            dict with:
                'pose': [B, 4, 4] - predicted pose
                'confidence': [B] - pose confidence
                'correspondences': correspondence data
        """
        B = query_data['features'].shape[0] if 'features' in query_data else 1
        
        # Select templates if not provided
        if template_indices is None:
            # Default: use first num_templates
            template_indices = torch.arange(
                self.num_templates,
                device=query_data['image'].device if 'image' in query_data else 'cuda'
            ).unsqueeze(0).expand(B, -1)
        
        # Extract selected templates
        selected_template_data = self._extract_templates(template_data, template_indices)
        
        # Run model
        outputs = self.model(query_data, selected_template_data)
        
        # Build correspondences
        correspondences = self.correspondence_builder(
            outputs['classification_logits'],
            outputs['flow'],
            outputs['flow_confidence'],
            query_data,
            selected_template_data,
        )
        
        # Estimate pose
        if self.use_epropnp and self.epropnp is not None:
            pose, confidence = self._estimate_pose_epropnp(
                correspondences,
                camera_intrinsics,
            )
        else:
            pose, confidence = self._estimate_pose_pnp(
                correspondences,
                camera_intrinsics,
            )
        
        return {
            'pose': pose,
            'confidence': confidence,
            'correspondences': correspondences,
            'classification_logits': outputs['classification_logits'],
            'flow': outputs['flow'],
            'flow_confidence': outputs['flow_confidence'],
        }
    
    def _extract_templates(
        self,
        template_data: dict,
        template_indices: torch.Tensor,
    ) -> dict:
        """
        Extract subset of templates by indices.
        
        Args:
            template_data: dict with all template data
            template_indices: [B, num_templates] - indices to extract
        
        Returns:
            dict with selected template data
        """
        B, num_selected = template_indices.shape
        
        # Handle different data formats
        selected = {}
        
        for key, value in template_data.items():
            if isinstance(value, torch.Tensor):
                if value.dim() >= 2 and value.shape[1] >= num_selected:
                    # Assume dimension 1 is template dimension
                    # Use advanced indexing to select templates
                    batch_indices = torch.arange(B, device=value.device).unsqueeze(1)
                    selected[key] = value[batch_indices, template_indices]
                else:
                    selected[key] = value
            else:
                selected[key] = value
        
        return selected
    
    def _estimate_pose_epropnp(
        self,
        correspondences: dict,
        camera_K: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate pose using EPro-PnP.
        
        Returns:
            pose: [B, 4, 4]
            confidence: [B]
        """
        # TODO: Implement EPro-PnP integration
        # This requires proper batching and interface with EPro-PnP API
        
        # For now, return identity pose
        device = camera_K.device
        B = camera_K.shape[0] if camera_K.dim() == 3 else 1
        
        pose = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
        confidence = torch.ones(B, device=device)
        
        return pose, confidence
    
    def _estimate_pose_pnp(
        self,
        correspondences: dict,
        camera_K: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate pose using OpenCV PnP (RANSAC).
        
        Returns:
            pose: [B, 4, 4]
            confidence: [B]
        """
        import cv2
        import numpy as np
        
        pts2d = correspondences['pts2d'].cpu().numpy()  # [N, 2]
        pts3d = correspondences['pts3d'].cpu().numpy()  # [N, 3]
        weights = correspondences['weights'].cpu().numpy()  # [N]
        
        if len(pts2d) < 4:
            # Not enough points
            device = camera_K.device
            B = camera_K.shape[0] if camera_K.dim() == 3 else 1
            pose = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
            confidence = torch.zeros(B, device=device)
            return pose, confidence
        
        # Convert camera matrix
        if camera_K.dim() == 3:
            K = camera_K[0].cpu().numpy()  # Use first camera
        else:
            K = camera_K.cpu().numpy()
        
        # Run PnP RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d,
            pts2d,
            K,
            None,  # No distortion
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=5.0,
        )
        
        if not success or inliers is None:
            device = camera_K.device
            B = camera_K.shape[0] if camera_K.dim() == 3 else 1
            pose = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
            confidence = torch.zeros(B, device=device)
            return pose, confidence
        
        # Convert to pose matrix
        R, _ = cv2.Rodrigues(rvec)
        pose_np = np.eye(4)
        pose_np[:3, :3] = R
        pose_np[:3, 3] = tvec.squeeze()
        
        # Convert to tensor
        device = camera_K.device
        pose = torch.from_numpy(pose_np).float().to(device).unsqueeze(0)
        
        # Confidence from inlier ratio
        confidence = torch.tensor(
            [len(inliers) / len(pts2d)],
            device=device,
            dtype=torch.float32,
        )
        
        return pose, confidence
