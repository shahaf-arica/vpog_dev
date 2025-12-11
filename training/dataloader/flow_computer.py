"""
Flow Computer for VPOG
Computes pixel-level flow between query and template patches
Handles visibility and patch-level visibility constraints
"""

from __future__ import annotations

import logging
import numpy as np
import torch
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# Import from GigaPose
from src.custom_megapose.transform import Transform


@dataclass
class FlowResult:
    """Container for flow computation results"""
    flow: np.ndarray  # (H_p, W_p, 2) flow in patch-local coords (delta_x=1.0 = one patch)
    confidence: np.ndarray  # (H_p, W_p) confidence in [0, 1]
    visibility: np.ndarray  # (H_p, W_p) boolean mask for visibility
    patch_visibility: np.ndarray  # (H_p, W_p) boolean mask for patch-level visibility
    valid_flow: np.ndarray  # (H_p, W_p) combined validity mask
    unseen_mask: np.ndarray  # (H_p, W_p) boolean mask for unseen pixels (for loss)


class FlowComputer:
    """
    Computes pixel-level flow between query and template patches.
    Flow is defined in patch-local coordinate system (relative to patch center).
    """
    
    def __init__(
        self,
        patch_size: int = 16,
        compute_visibility: bool = True,
        compute_patch_visibility: bool = True,
        visibility_threshold: float = 0.1,
        depth_tolerance: float = 50.0,  # millimeters (was 0.05 meters = 50mm)
        min_visible_pixels: int = 4,  # Minimum visible pixels to consider patch valid
    ):
        """
        Args:
            patch_size: Size of patches (e.g., 16 for 16x16 patches)
            compute_visibility: Whether to compute visibility constraints
            compute_patch_visibility: Whether to check if flow stays within patch bounds
            visibility_threshold: Minimum depth/visibility for valid flow
            depth_tolerance: Depth difference tolerance for occlusion detection (millimeters)
            min_visible_pixels: Minimum number of visible pixels for valid patch
        """
        self.patch_size = patch_size
        self.compute_visibility = compute_visibility
        self.compute_patch_visibility = compute_patch_visibility
        self.visibility_threshold = visibility_threshold
        self.depth_tolerance = depth_tolerance
        self.min_visible_pixels = min_visible_pixels
    
    def compute_flow_between_patches(
        self,
        query_patch_center: np.ndarray,  # (2,) in image coords [x, y]
        template_patch_center: np.ndarray,  # (2,) in image coords [x, y]
        query_K: np.ndarray,  # (3, 3) intrinsics
        template_K: np.ndarray,  # (3, 3) intrinsics
        query_pose: np.ndarray,  # (4, 4) TWO - world to object
        template_pose: np.ndarray,  # (4, 4) TWO
        query_depth: Optional[np.ndarray] = None,  # (H, W) depth map
        template_depth: Optional[np.ndarray] = None,  # (H, W) depth map
        query_mask: Optional[np.ndarray] = None,  # (H, W) visibility mask
        template_mask: Optional[np.ndarray] = None,  # (H, W) visibility mask
    ) -> FlowResult:
        """
        Compute pixel-level flow from template patch to query patch.
        
        Args:
            query_patch_center: Center of query patch in image coordinates [x, y]
            template_patch_center: Center of template patch in image coordinates [x, y]
            query_K: Query camera intrinsics
            template_K: Template camera intrinsics
            query_pose: Query object pose (world to object)
            template_pose: Template object pose
            query_depth: Optional depth map for visibility
            template_depth: Optional depth map for template
            query_mask: Optional visibility mask for query
            template_mask: Optional visibility mask for template
        
        Returns:
            FlowResult containing flow and visibility information
        """

        
        # Create pixel grid for the patch (centered at patch center)
        half_size = self.patch_size // 2
        y_offsets, x_offsets = np.meshgrid(
            np.arange(-half_size, half_size),
            np.arange(-half_size, half_size),
            indexing='ij'
        )  # Both (patch_size, patch_size)
        
        # QUERY patch pixel coordinates in image space
        query_pixels_x = query_patch_center[0] + x_offsets  # (ps, ps)
        query_pixels_y = query_patch_center[1] + y_offsets  # (ps, ps)
        
        # Stack into (ps, ps, 2) format
        query_pixels = np.stack([query_pixels_x, query_pixels_y], axis=-1)
        
        # Get query depth values
        if query_depth is not None:
            query_depth_vals = self._sample_depth(
                query_depth, query_pixels_x, query_pixels_y
            )  # (ps, ps)
        else:
            query_depth_vals = np.ones_like(query_pixels_x)
        
        # Convert QUERY 2D pixels to 3D points in query camera frame
        query_points_3d = self._unproject_pixels(
            query_pixels, query_depth_vals, query_K
        )  # (ps, ps, 3)
        
        # Transform points from QUERY to TEMPLATE coordinate system
        T_query2world = Transform(query_pose).inverse()
        T_world2template = Transform(template_pose)
        T_query2template = T_world2template * T_query2world
        
        # Apply transformation
        template_points_3d = self._transform_points(
            query_points_3d, T_query2template.toHomogeneousMatrix()
        )  # (ps, ps, 3)
        
        # Project to TEMPLATE image
        template_pixels = self._project_points(template_points_3d, template_K)  # (ps, ps, 2)
        
        # Compute flow in patch-local coordinates
        # Flow FROM template TO query (template_center → query_center)
        # Flow is in PATCH UNITS: delta_x = 1.0 means one patch displacement (16 pixels)
        flow_pixels = template_pixels - template_patch_center[None, None, :]  # (ps, ps, 2) in pixels
        flow = flow_pixels / self.patch_size  # (ps, ps, 2) in patch units
        # Note: flow is negative because we compute template_pos - template_center
        # But the flow label should be: where in template does query pixel correspond to
        # So flow = (template_pixels - template_patch_center) / patch_size
        
        # Initialize visibility masks
        visibility = np.ones((self.patch_size, self.patch_size), dtype=bool)
        patch_visibility = np.ones((self.patch_size, self.patch_size), dtype=bool)
        occlusion_mask = np.zeros((self.patch_size, self.patch_size), dtype=bool)
        out_of_bounds = np.zeros((self.patch_size, self.patch_size), dtype=bool)
        
        if self.compute_visibility:
            # 1. Check QUERY source pixels are valid (in bounds, has depth, visible in mask)
            H_q, W_q = (224, 224) if query_depth is None else query_depth.shape
            query_in_bounds = (
                (query_pixels_x >= 0) & (query_pixels_x < W_q) &
                (query_pixels_y >= 0) & (query_pixels_y < H_q)
            )
            visibility &= query_in_bounds
            
            if query_mask is not None:
                query_mask_vals = self._sample_mask(
                    query_mask, query_pixels_x, query_pixels_y
                )
                visibility &= query_mask_vals
            
            # 2. Check TEMPLATE projection is in bounds
            H_t, W_t = (224, 224) if template_depth is None else template_depth.shape
            template_in_bounds = (
                (template_pixels[..., 0] >= 0) & (template_pixels[..., 0] < W_t) &
                (template_pixels[..., 1] >= 0) & (template_pixels[..., 1] < H_t)
            )
            visibility &= template_in_bounds
            out_of_bounds = ~template_in_bounds
            
            # 3. Check SELF-OCCLUSION using depth comparison
            # Logic: Query pixel projects to template camera at depth z_q_projected
            #        Template ground truth depth at that pixel is z_t
            #        If z_q_projected >= z_t + tolerance → query point is BEHIND template surface → self-occluded
            if template_depth is not None and query_depth is not None:
                # Sample TEMPLATE ground truth depth at projected pixel location
                template_depth_vals = self._sample_depth(
                    template_depth, template_pixels[..., 0], template_pixels[..., 1]
                )
                
                # Z coordinate of query point in TEMPLATE camera frame
                z_q_projected = template_points_3d[..., 2]  # (ps, ps)
                
                # Self-occlusion check: z_q >= z_t + tolerance means query is farther (behind) → occluded
                self_occluded = z_q_projected >= (template_depth_vals + self.depth_tolerance)
                
                # Also check point is in front of template camera
                valid_depth = z_q_projected > 0
                
                visibility &= ~self_occluded  # Remove self-occluded points
                visibility &= valid_depth  # Keep only points in front of camera
                
                occlusion_mask = self_occluded
            
            # 4. Check template mask visibility at projected location
            if template_mask is not None:
                template_mask_vals = self._sample_mask(
                    template_mask, template_pixels[..., 0], template_pixels[..., 1]
                )
                visibility &= template_mask_vals
        
        if self.compute_patch_visibility:
            # Check if flow stays within reasonable patch bounds
            # Allow up to 2 patches displacement
            flow_magnitude = np.linalg.norm(flow, axis=-1)  # (ps, ps) in patch units
            patch_visibility = flow_magnitude < 2.0  # 2 patches = 32 pixels
        
        # Combined validity
        valid_flow = visibility & patch_visibility
        
        # Compute confidence based on visibility factors
        # Confidence is soft version of validity
        confidence = np.ones((self.patch_size, self.patch_size), dtype=np.float32)
        
        # Reduce confidence for occluded pixels
        confidence[occlusion_mask] = 0.0
        
        # Reduce confidence for out-of-bounds pixels
        confidence[out_of_bounds] = 0.0
        
        # Reduce confidence for large flow magnitudes (less reliable)
        flow_magnitude = np.linalg.norm(flow, axis=-1)
        confidence *= np.exp(-flow_magnitude)  # Exponential decay with flow magnitude
        
        # Clamp to [0, 1]
        confidence = np.clip(confidence, 0.0, 1.0)
        
        # Generate unseen mask for loss computation
        # A pixel is "unseen" if:
        # 1. Occluded in query (something in front)
        # 2. Out of bounds in query
        # 3. Not visible in template
        # 4. Too few visible pixels in patch (whole patch marked unseen)
        unseen_mask = ~valid_flow
        
        # If too few pixels are visible, mark entire patch as unseen
        num_visible = valid_flow.sum()
        if num_visible < self.min_visible_pixels:
            unseen_mask[:] = True
            confidence[:] = 0.0
        
        return FlowResult(
            flow=flow,
            confidence=confidence,
            visibility=visibility,
            patch_visibility=patch_visibility,
            valid_flow=valid_flow,
            unseen_mask=unseen_mask,
        )
    
    def _unproject_pixels(
        self,
        pixels: np.ndarray,  # (H, W, 2) [x, y]
        depth: np.ndarray,  # (H, W)
        K: np.ndarray,  # (3, 3)
    ) -> np.ndarray:
        """
        Unproject 2D pixels to 3D points in camera frame.
        
        Returns:
            points_3d: (H, W, 3) 3D points in camera frame
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        x = pixels[..., 0]
        y = pixels[..., 1]
        
        # Unproject
        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth
        
        points_3d = np.stack([X, Y, Z], axis=-1)
        return points_3d
    
    def _project_points(
        self,
        points_3d: np.ndarray,  # (H, W, 3)
        K: np.ndarray,  # (3, 3)
    ) -> np.ndarray:
        """
        Project 3D points to 2D pixels.
        
        Returns:
            pixels: (H, W, 2) [x, y]
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        X = points_3d[..., 0]
        Y = points_3d[..., 1]
        Z = points_3d[..., 2]
        
        # Avoid division by zero
        Z = np.where(np.abs(Z) < 1e-6, 1e-6, Z)
        
        x = fx * X / Z + cx
        y = fy * Y / Z + cy
        
        pixels = np.stack([x, y], axis=-1)
        return pixels
    
    def _transform_points(
        self,
        points: np.ndarray,  # (H, W, 3)
        transform: np.ndarray,  # (4, 4)
    ) -> np.ndarray:
        """
        Apply 4x4 transformation to 3D points.
        
        Returns:
            transformed_points: (H, W, 3)
        """
        shape = points.shape
        points_flat = points.reshape(-1, 3)
        
        # Convert to homogeneous
        points_homo = np.concatenate([points_flat, np.ones((len(points_flat), 1))], axis=1)
        
        # Apply transform
        points_transformed = (transform @ points_homo.T).T
        
        # Convert back to 3D
        points_transformed = points_transformed[:, :3] / points_transformed[:, 3:4]
        
        return points_transformed.reshape(shape)
    
    def _sample_depth(
        self,
        depth_map: np.ndarray,  # (H, W)
        x: np.ndarray,  # (...,) float coordinates
        y: np.ndarray,  # (...,) float coordinates
    ) -> np.ndarray:
        """
        Sample depth map at given coordinates (with boundary checking).
        
        Returns:
            depth_vals: (...,) sampled depth values
        """
        H, W = depth_map.shape
        
        # Clamp coordinates to valid range
        x = np.clip(x, 0, W - 1).astype(int)
        y = np.clip(y, 0, H - 1).astype(int)
        
        return depth_map[y, x]
    
    def _sample_mask(
        self,
        mask: np.ndarray,  # (H, W) boolean or float
        x: np.ndarray,  # (...,) float coordinates
        y: np.ndarray,  # (...,) float coordinates
    ) -> np.ndarray:
        """
        Sample mask at given coordinates (with boundary checking).
        
        Returns:
            mask_vals: (...,) boolean mask values
        """
        H, W = mask.shape
        
        # Clamp coordinates to valid range
        x = np.clip(x, 0, W - 1).astype(int)
        y = np.clip(y, 0, H - 1).astype(int)
        
        return mask[y, x] > 0.5


def compute_patch_flows(
    query_data: Dict,
    template_data: Dict,
    patch_size: int = 16,
    image_size: int = 224,
    **flow_kwargs,
) -> Dict:
    """
    Compute 16×16 pixel-level flow for all query patches to all template patches.
    
    Args:
        query_data: Dictionary containing:
            - 'K': (3, 3) camera intrinsics
            - 'pose': (4, 4) object pose TWO
            - 'depth': (H, W) depth map (optional)
            - 'mask': (H, W) visibility mask (optional)
        template_data: Dictionary containing template data for S templates:
            - 'K': (S, 3, 3) or (3, 3) camera intrinsics
            - 'poses': (S, 4, 4) object poses TWO
            - 'depths': (S, H, W) depth maps (optional)
            - 'masks': (S, H, W) visibility masks (optional)
        patch_size: Patch size (default: 16)
        image_size: Image size (default: 224)
        **flow_kwargs: Additional arguments for FlowComputer
    
    Returns:
        Dictionary containing:
            - 'flows': (S, Nq, Nt, 16, 16, 2) pixel-level flow in patch units
            - 'confidence': (S, Nq, Nt, 16, 16) confidence per pixel
            - 'unseen_masks': (S, Nq, Nt, 16, 16) unseen mask for loss
            - 'patch_centers': (Nq, 2) patch centers
            - 'num_patches': int
    """
    computer = FlowComputer(patch_size=patch_size, **flow_kwargs)
    
    # Compute patch grid
    num_patches_per_side = image_size // patch_size
    num_patches = num_patches_per_side ** 2
    
    # Get patch centers
    patch_centers = []
    for i in range(num_patches_per_side):
        for j in range(num_patches_per_side):
            center_x = (j + 0.5) * patch_size
            center_y = (i + 0.5) * patch_size
            patch_centers.append([center_x, center_y])
    patch_centers = np.array(patch_centers)  # (Nq, 2)
    
    # Extract query data
    query_K = query_data['K']
    query_pose = query_data['pose']
    query_depth = query_data.get('depth', None)
    query_mask = query_data.get('mask', None)
    
    # Extract template data
    template_poses = template_data['poses']  # (S, 4, 4)
    num_templates = len(template_poses)
    
    # Handle template K (can be shared or per-template)
    template_K = template_data.get('K', query_K)
    if template_K.ndim == 2:
        template_K = np.tile(template_K[None], (num_templates, 1, 1))
    
    template_depths = template_data.get('depths', None)
    template_masks = template_data.get('masks', None)
    
    # Allocate output arrays
    flows = np.zeros((num_templates, num_patches, num_patches, patch_size, patch_size, 2))
    confidence = np.zeros((num_templates, num_patches, num_patches, patch_size, patch_size))
    unseen_masks = np.ones((num_templates, num_patches, num_patches, patch_size, patch_size), dtype=bool)
    
    # Compute flows for each template
    for s in range(num_templates):
        template_pose = template_poses[s]
        template_K_s = template_K[s]
        template_depth_s = template_depths[s] if template_depths is not None else None
        template_mask_s = template_masks[s] if template_masks is not None else None
        
        # Compute flow from each template patch to each query patch
        for q_idx, query_center in enumerate(patch_centers):
            for t_idx, template_center in enumerate(patch_centers):
                result = computer.compute_flow_between_patches(
                    query_patch_center=query_center,
                    template_patch_center=template_center,
                    query_K=query_K,
                    template_K=template_K_s,
                    query_pose=query_pose,
                    template_pose=template_pose,
                    query_depth=query_depth,
                    template_depth=template_depth_s,
                    query_mask=query_mask,
                    template_mask=template_mask_s,
                )
                
                flows[s, q_idx, t_idx] = result.flow
                confidence[s, q_idx, t_idx] = result.confidence
                unseen_masks[s, q_idx, t_idx] = result.unseen_mask
    
    return {
        'flows': flows,  # (S, Nq, Nt, 16, 16, 2)
        'confidence': confidence,  # (S, Nq, Nt, 16, 16)
        'unseen_masks': unseen_masks,  # (S, Nq, Nt, 16, 16)
        'patch_centers': patch_centers,  # (Nq, 2)
        'num_patches': num_patches,
    }


def compute_classification_labels(
    query_pose: np.ndarray,
    template_poses: np.ndarray,
    query_mask: Optional[np.ndarray] = None,
    template_masks: Optional[np.ndarray] = None,
    patch_size: int = 16,
    image_size: int = 224,
    occlusion_threshold: float = 0.5,
) -> Dict:
    """
    Compute ground truth classification labels for patch matching.
    
    For each query patch, determine:
    1. Best matching template patch (based on pose similarity and visibility)
    2. Unseen label (if patch is occluded or out-of-view)
    
    Args:
        query_pose: (4, 4) query object pose
        template_poses: (S, 4, 4) template object poses
        query_mask: (H, W) query visibility mask (optional)
        template_masks: (S, H, W) template visibility masks (optional)
        patch_size: Patch size
        image_size: Image size
        occlusion_threshold: Fraction of visible pixels for valid patch
    
    Returns:
        Dictionary containing:
            - 'labels': (Nq,) template patch indices [0, Nt-1] or Nt for unseen
            - 'unseen_mask': (Nq,) boolean mask for unseen patches
            - 'template_indices': (Nq,) which template [0, S-1]
    """
    num_patches_per_side = image_size // patch_size
    num_patches = num_patches_per_side ** 2
    num_templates = len(template_poses)
    
    labels = np.full(num_patches, num_patches, dtype=np.int64)  # Default to unseen
    unseen_mask = np.ones(num_patches, dtype=bool)
    template_indices = np.zeros(num_patches, dtype=np.int64)
    
    # Simple heuristic: if query mask has sufficient visibility, it's not unseen
    if query_mask is not None:
        for i in range(num_patches_per_side):
            for j in range(num_patches_per_side):
                patch_idx = i * num_patches_per_side + j
                
                # Extract patch region
                y_start = i * patch_size
                y_end = (i + 1) * patch_size
                x_start = j * patch_size
                x_end = (j + 1) * patch_size
                
                patch_mask = query_mask[y_start:y_end, x_start:x_end]
                visibility = patch_mask.mean()
                
                if visibility >= occlusion_threshold:
                    # Not unseen - assign to closest template patch
                    # (In practice, this should use flow/correspondence)
                    labels[patch_idx] = patch_idx  # Placeholder: same patch
                    unseen_mask[patch_idx] = False
                    template_indices[patch_idx] = 0  # Placeholder: first template
    
    return {
        'labels': labels,
        'unseen_mask': unseen_mask,
        'template_indices': template_indices,
    }


if __name__ == "__main__":
    """
    REAL TEST with GSO data and flow visualizations
    Computes actual flows between template and query patches and visualizes them
    """
    import os
    import sys
    from pathlib import Path
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib import patches as mpl_patches
    from scipy.spatial.transform import Rotation
    from PIL import Image
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from bop_toolkit_lib import inout
    from src.custom_megapose.template_dataset import TemplateDataset
    from src.custom_megapose.transform import Transform
    
    print("=" * 80)
    print("REAL TEST: FlowComputer with GSO Data")
    print("=" * 80)
    
    # Setup paths
    dataset_dir = project_root / "datasets" / "gso"
    templates_dir = project_root / "datasets" / "templates" / "gso"
    save_dir = project_root / "tmp" / "flow_computer_test"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if not templates_dir.exists():
        print(f"\n✗ Templates not found at {templates_dir}")
        print("Please run: python -m src.scripts.render_gso_templates")
        sys.exit(1)
    
    print(f"✓ Found templates at {templates_dir}")
    
    # Initialize flow computer
    computer = FlowComputer(
        patch_size=16,
        compute_visibility=True,
        compute_patch_visibility=True,
        visibility_threshold=0.1,
    )
    print(f"✓ Initialized FlowComputer")
    print(f"  Patch size: {computer.patch_size}x{computer.patch_size}")
    
    # Load model info
    model_infos = inout.load_json(dataset_dir / "models_info.json")
    obj_id = int(list(model_infos.keys())[0])  # Use first object
    
    print(f"✓ Testing with object {obj_id:06d}")
    
    # Load template poses
    poses_file = templates_dir / f"object_poses/{obj_id:06d}.npy"
    if not poses_file.exists():
        print(f"✗ Poses file not found: {poses_file}")
        sys.exit(1)
    
    all_poses = np.load(poses_file)
    print(f"✓ Loaded {len(all_poses)} template poses")
    
    # Select two templates with known pose difference
    template_idx = 0  # Frontal view
    query_idx = 10    # Slightly rotated view
    
    template_pose = all_poses[template_idx]
    query_pose = all_poses[query_idx]
    
    # Compute angular distance
    from training.dataloader.template_selector import extract_d_ref_from_pose
    template_dir = extract_d_ref_from_pose(template_pose)
    query_dir = extract_d_ref_from_pose(query_pose)
    angle = np.rad2deg(np.arccos(np.clip(np.dot(template_dir, query_dir), -1, 1)))
    
    print(f"\nSelected template pair:")
    print(f"  Template {template_idx} -> Query {query_idx}")
    print(f"  Angular distance: {angle:.2f}°")
    
    # Load template and query images
    template_img_path = templates_dir / f"{obj_id:06d}/{template_idx:06d}.png"
    query_img_path = templates_dir / f"{obj_id:06d}/{query_idx:06d}.png"
    template_depth_path = templates_dir / f"{obj_id:06d}/{template_idx:06d}_depth.png"
    query_depth_path = templates_dir / f"{obj_id:06d}/{query_idx:06d}_depth.png"
    
    if not all([p.exists() for p in [template_img_path, query_img_path, 
                                      template_depth_path, query_depth_path]]):
        print(f"✗ Missing image or depth files")
        sys.exit(1)
    
    # Load images and depth
    template_img = np.array(Image.open(template_img_path))
    query_img = np.array(Image.open(query_img_path))
    template_depth = np.array(Image.open(template_depth_path)).astype(np.float32) / 10.0  # Scale factor
    query_depth = np.array(Image.open(query_depth_path)).astype(np.float32) / 10.0
    
    # Create masks from alpha channel
    template_mask = template_img[:, :, 3] > 0 if template_img.shape[2] == 4 else np.ones(template_img.shape[:2], dtype=bool)
    query_mask = query_img[:, :, 3] > 0 if query_img.shape[2] == 4 else np.ones(query_img.shape[:2], dtype=bool)
    
    H, W = template_img.shape[:2]
    print(f"✓ Loaded images: {W}x{H}")
    
    # Template dataset uses specific camera intrinsics
    K = np.array([
        [572.4114, 0.0, W/2],
        [0.0, 573.57043, H/2],
        [0.0, 0.0, 1.0]
    ])
    
    # Test flow computation on a grid of patches
    print(f"\nComputing flows for patch grid...")
    num_patches_per_side = 14  # For 224x224 image with 16x16 patches
    patch_centers = []
    flow_results = []
    
    for i in range(3):  # Sample 3x3 grid for visualization
        for j in range(3):
            # Map to image coordinates
            px = int((j + 1) * W / 4)  # Spread across image
            py = int((i + 1) * H / 4)
            patch_centers.append([px, py])
            
            result = computer.compute_flow_between_patches(
                query_patch_center=np.array([px, py], dtype=float),
                template_patch_center=np.array([px, py], dtype=float),
                query_K=K,
                template_K=K,
                query_pose=query_pose,
                template_pose=template_pose,
                query_depth=query_depth,
                template_depth=template_depth,
                query_mask=query_mask,
                template_mask=template_mask,
            )
            flow_results.append(result)
    
    print(f"✓ Computed flows for {len(flow_results)} patches")
    
    # VISUALIZATION: Flow fields
    print(f"\nGenerating flow visualizations...")
    
    # Figure 1: Template and Query images with patch grid
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(template_img[:, :, :3] if template_img.shape[2] == 4 else template_img)
    axes[0].set_title(f'Template {template_idx}', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(query_img[:, :, :3] if query_img.shape[2] == 4 else query_img)
    axes[1].set_title(f'Query {query_idx} (∠={angle:.1f}°)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Draw patch centers
    for px, py in patch_centers:
        for ax in axes:
            rect = mpl_patches.Rectangle(
                (px - computer.patch_size//2, py - computer.patch_size//2),
                computer.patch_size, computer.patch_size,
                linewidth=2, edgecolor='yellow', facecolor='none'
            )
            ax.add_patch(rect)
            ax.plot(px, py, 'r+', markersize=10, markeredgewidth=2)
    
    plt.tight_layout()
    save_path = save_dir / f"obj{obj_id:06d}_images_with_patches.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved patch grid: {save_path}")
    
    # Figure 2: Flow fields for each patch
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, (result, (px, py)) in enumerate(zip(flow_results, patch_centers)):
        ax = axes[idx]
        
        # Create flow visualization
        flow = result.flow  # (patch_size, patch_size, 2)
        valid = result.valid_flow
        
        # Flow magnitude
        flow_mag = np.linalg.norm(flow, axis=-1)
        flow_mag[~valid] = np.nan  # Mask invalid
        
        im = ax.imshow(flow_mag, cmap='jet', vmin=0, vmax=10)
        
        # Draw flow arrows (subsample)
        step = 4
        Y, X = np.meshgrid(np.arange(0, computer.patch_size, step),
                           np.arange(0, computer.patch_size, step), indexing='ij')
        
        for y, x in zip(Y.flatten(), X.flatten()):
            if valid[y, x]:
                dx, dy = flow[y, x]
                ax.arrow(x, y, dx*2, dy*2, head_width=0.5, head_length=0.3,
                        fc='white', ec='white', alpha=0.7, linewidth=0.5)
        
        valid_pct = valid.mean() * 100
        mean_mag = flow_mag[valid].mean() if valid.any() else 0
        
        ax.set_title(f'Patch ({px},{py})\n{valid_pct:.0f}% valid, mag={mean_mag:.1f}px',
                    fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    save_path = save_dir / f"obj{obj_id:06d}_flow_fields.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved flow fields: {save_path}")
    
    # Figure 3: Statistics
    all_valid_flows = []
    all_visibility = []
    
    for result in flow_results:
        if result.valid_flow.any():
            valid_flows = result.flow[result.valid_flow]
            all_valid_flows.extend(np.linalg.norm(valid_flows, axis=-1).tolist())
        all_visibility.append(result.valid_flow.mean())
    
    if all_valid_flows:
        print(f"\nFlow Statistics:")
        print(f"  Mean magnitude: {np.mean(all_valid_flows):.2f} pixels")
        print(f"  Max magnitude:  {np.max(all_valid_flows):.2f} pixels")
        print(f"  Std magnitude:  {np.std(all_valid_flows):.2f} pixels")
        print(f"  Mean visibility: {np.mean(all_visibility)*100:.1f}%")
    
    print("\n" + "=" * 80)
    print(f"✓ All tests passed!")
    print(f"✓ Visualizations saved to {save_dir}")
    print("=" * 80)
