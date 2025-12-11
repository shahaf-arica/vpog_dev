# vpog/inference/global_mode.py
#
# Global mode inference for VPOG
#
# Uses all 162 templates in chunks for comprehensive pose estimation
#
# Process:
#   1. Split 162 templates into chunks (e.g., 4 chunks of ~40 templates)
#   2. Run VPOG model on each chunk
#   3. Aggregate correspondences from all chunks
#   4. Run EPro-PnP for final pose estimation
#
# Input:
#   - query_image: [H, W, 3] or [B, H, W, 3]
#   - templates: List of 162 template data
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


class GlobalModeInference(nn.Module):
    """
    Global mode inference using all 162 templates.
    
    Processes templates in chunks to handle memory constraints,
    then aggregates correspondences for final pose estimation.
    """
    
    def __init__(
        self,
        model: VPOGModel,
        correspondence_builder: CorrespondenceBuilder,
        total_templates: int = 162,
        chunk_size: int = 40,
        use_epropnp: bool = True,
    ):
        """
        Args:
            model: VPOG model
            correspondence_builder: Correspondence builder
            total_templates: Total number of templates (default 162)
            chunk_size: Templates per chunk (default 40)
            use_epropnp: Use EPro-PnP for pose estimation (default True)
        """
        super().__init__()
        self.model = model
        self.correspondence_builder = correspondence_builder
        self.total_templates = total_templates
        self.chunk_size = chunk_size
        self.use_epropnp = use_epropnp
        
        # Compute number of chunks
        self.num_chunks = (total_templates + chunk_size - 1) // chunk_size
        
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
    ) -> Dict[str, torch.Tensor]:
        """
        Run global mode inference.
        
        Args:
            query_data: dict with query image data
            template_data: dict with ALL template data (162 templates)
            camera_intrinsics: [B, 3, 3] or [3, 3] - camera K matrix
        
        Returns:
            dict with:
                'pose': [B, 4, 4] - predicted pose
                'confidence': [B] - pose confidence
                'correspondences': aggregated correspondence data
                'chunk_results': list of per-chunk results
        """
        B = query_data['features'].shape[0] if 'features' in query_data else 1
        
        # Process each chunk
        all_correspondences = []
        chunk_results = []
        
        for chunk_idx in range(self.num_chunks):
            # Define chunk range
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, self.total_templates)
            
            # Extract chunk templates
            chunk_template_data = self._extract_chunk(
                template_data,
                start_idx,
                end_idx,
            )
            
            # Run model on chunk
            outputs = self.model(query_data, chunk_template_data)
            
            # Build correspondences
            correspondences = self.correspondence_builder(
                outputs['classification_logits'],
                outputs['flow'],
                outputs['flow_confidence'],
                query_data,
                chunk_template_data,
            )
            
            all_correspondences.append(correspondences)
            chunk_results.append({
                'classification_logits': outputs['classification_logits'],
                'flow': outputs['flow'],
                'flow_confidence': outputs['flow_confidence'],
                'correspondences': correspondences,
                'template_range': (start_idx, end_idx),
            })
        
        # Aggregate correspondences from all chunks
        aggregated_correspondences = self._aggregate_correspondences(all_correspondences)
        
        # Estimate pose from aggregated correspondences
        if self.use_epropnp and self.epropnp is not None:
            pose, confidence = self._estimate_pose_epropnp(
                aggregated_correspondences,
                camera_intrinsics,
            )
        else:
            pose, confidence = self._estimate_pose_pnp(
                aggregated_correspondences,
                camera_intrinsics,
            )
        
        return {
            'pose': pose,
            'confidence': confidence,
            'correspondences': aggregated_correspondences,
            'chunk_results': chunk_results,
            'num_chunks': self.num_chunks,
        }
    
    def _extract_chunk(
        self,
        template_data: dict,
        start_idx: int,
        end_idx: int,
    ) -> dict:
        """
        Extract chunk of templates.
        
        Args:
            template_data: dict with all template data
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)
        
        Returns:
            dict with chunk template data
        """
        chunk = {}
        
        for key, value in template_data.items():
            if isinstance(value, torch.Tensor):
                if value.dim() >= 2:
                    # Assume dimension 1 is template dimension
                    chunk[key] = value[:, start_idx:end_idx]
                else:
                    chunk[key] = value
            else:
                chunk[key] = value
        
        return chunk
    
    def _aggregate_correspondences(
        self,
        correspondence_list: List[dict],
    ) -> dict:
        """
        Aggregate correspondences from multiple chunks.
        
        Args:
            correspondence_list: List of correspondence dicts
        
        Returns:
            Aggregated correspondence dict
        """
        if not correspondence_list:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            return {
                'pts2d': torch.empty(0, 2, device=device),
                'pts3d': torch.empty(0, 3, device=device),
                'weights': torch.empty(0, device=device),
                'num_correspondences': 0,
            }
        
        # Concatenate all correspondences
        pts2d_list = [c['pts2d'] for c in correspondence_list if c['num_correspondences'] > 0]
        pts3d_list = [c['pts3d'] for c in correspondence_list if c['num_correspondences'] > 0]
        weights_list = [c['weights'] for c in correspondence_list if c['num_correspondences'] > 0]
        
        if not pts2d_list:
            device = correspondence_list[0]['pts2d'].device
            return {
                'pts2d': torch.empty(0, 2, device=device),
                'pts3d': torch.empty(0, 3, device=device),
                'weights': torch.empty(0, device=device),
                'num_correspondences': 0,
            }
        
        pts2d = torch.cat(pts2d_list, dim=0)
        pts3d = torch.cat(pts3d_list, dim=0)
        weights = torch.cat(weights_list, dim=0)
        
        return {
            'pts2d': pts2d,
            'pts3d': pts3d,
            'weights': weights,
            'num_correspondences': len(pts2d),
        }
    
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
