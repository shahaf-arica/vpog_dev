"""
VPOG Training Dataset
Main dataloader for Visual Patch-wise Object pose estimation with Groups of templates

Loads:
- Query images from GSO/ShapeNet training data
- S = S_p + S_n templates per query
- Computes flow labels and patch correspondences
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# GigaPose imports
from src.megapose.datasets.scene_dataset import SceneObservation
from src.custom_megapose.web_scene_dataset import WebSceneDataset, IterableWebSceneDataset
from src.custom_megapose.template_dataset import TemplateDataset
from src.custom_megapose.transform import Transform
from src.utils.bbox import BoundingBox
from src.utils.logging import get_logger
import src.megapose.utils.tensor_collection as tc
from bop_toolkit_lib import inout

# VPOG imports
from training.dataloader.template_selector import TemplateSelector, extract_d_ref_from_pose
from training.dataloader.flow_computer import FlowComputer

logger = get_logger(__name__)


@dataclass
class VPOGBatch:
    """Container for a VPOG training batch"""

    # Core tensors
    images: torch.Tensor                 # [B, S+1, 3, H, W]
    masks: torch.Tensor                  # [B, S+1, H, W]
    K: torch.Tensor                      # [B, S+1, 3, 3]
    poses: torch.Tensor                  # [B, S+1, 4, 4]
    d_ref: torch.Tensor                  # [B, 3]
    template_indices: torch.Tensor       # [B, S]
    template_types: torch.Tensor         # [B, S]

    # Existing supervision
    flows: torch.Tensor                  # [B, S, H_p, W_p, 2]
    visibility: torch.Tensor             # [B, S, H_p, W_p]
    patch_visibility: torch.Tensor       # [B, S, H_p, W_p]

    # NEW: per-patch classification label
    #  -1                    → background (no object at patch center)
    #  0 .. H_p*W_p-1        → buddy template patch index (flattened)
    #  H_p*W_p               → object patch but unseen
    patch_cls: torch.Tensor              # [B, S, H_p, W_p]

    # NEW: dense flow and weights (template→query, in query-patch coords)
    dense_flow: torch.Tensor             # [B, S, H_p, W_p, ps, ps, 2]
    dense_weight: torch.Tensor           # [B, S, H_p, W_p, ps, ps]

    # Metadata / extras
    infos: object
    full_rgb: Optional[torch.Tensor] = None        # [B, 3, H_orig, W_orig]
    centered_rgb: Optional[torch.Tensor] = None    # [B, 3, 224, 224]
    bboxes: Optional[torch.Tensor] = None          # [B, 4]
    query_depth: Optional[torch.Tensor] = None     # [B, 224, 224]
    template_depth: Optional[torch.Tensor] = None  # [B, S, 224, 224]

    def to(self, device: torch.device) -> "VPOGBatch":
        """Move all tensors to the specified device and return a new VPOGBatch."""
        return VPOGBatch(
            images=self.images.to(device),
            masks=self.masks.to(device),
            K=self.K.to(device),
            poses=self.poses.to(device),
            d_ref=self.d_ref.to(device),
            template_indices=self.template_indices.to(device),
            template_types=self.template_types.to(device),
            flows=self.flows.to(device),
            visibility=self.visibility.to(device),
            patch_visibility=self.patch_visibility.to(device),
            patch_cls=self.patch_cls.to(device),
            dense_flow=self.dense_flow.to(device),
            dense_weight=self.dense_weight.to(device),
            infos=self.infos,  # usually small / non-tensor, keep as-is
            full_rgb=self.full_rgb.to(device) if self.full_rgb is not None else None,
            centered_rgb=self.centered_rgb.to(device) if self.centered_rgb is not None else None,
            bboxes=self.bboxes.to(device) if self.bboxes is not None else None,
            query_depth=self.query_depth.to(device) if self.query_depth is not None else None,
            template_depth=self.template_depth.to(device) if self.template_depth is not None else None,
        )




class VPOGTrainDataset:
    """
    VPOG Training Dataset
    
    For each query:
    1. Load query image and GT pose
    2. Select S_p nearest templates + S_n random negatives
    3. Extract d_ref from nearest template
    4. Compute flow labels between templates and query
    5. Return batch in format [B, S+1, C, H, W]
    """
    
    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        template_config: Dict,
        num_positive_templates: int = 4,
        num_negative_templates: int = 0,
        min_negative_angle_deg: float = 90.0,
        d_ref_random_ratio: float = 0.0,
        patch_size: int = 16,
        image_size: int = 224,
        flow_config: Optional[Dict] = None,
        transforms: Optional[Dict] = None,
        batch_size: int = 8,
        depth_scale: float = 10.0,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            root_dir: Root directory for datasets
            dataset_name: 'gso' or 'shapenet'
            template_config: Configuration for template dataset
            num_positive_templates: S_p - number of nearest templates
            num_negative_templates: S_n - number of random negative templates
            min_negative_angle_deg: Minimum angle for negative templates
            d_ref_random_ratio: Ratio of random vs nearest d_ref selection
            patch_size: Size of patches (default 16 for CroCo)
            image_size: Input image size
            flow_config: Configuration for flow computation
            transforms: Data augmentation transforms
            batch_size: Batch size
            depth_scale: Depth scale factor
            seed: Random seed for reproducibility (None = no fixed seed)
        """
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.dataset_dir = Path(root_dir) / dataset_name
        self.transforms = transforms if transforms is not None else {}
        self.patch_size = patch_size
        self.image_size = image_size
        self.depth_scale = depth_scale
        self.seed = seed
        
        # Number of patches
        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        
        logger.info(f"Initializing VPOGTrainDataset for {dataset_name}")
        logger.info(f"  Image size: {image_size}, Patch size: {patch_size}")
        logger.info(f"  Patches per side: {self.num_patches_per_side}, Total patches: {self.num_patches}")
        
        # Set random seed for reproducibility if provided
        if seed is not None:
            import random
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            logger.info(f"  Set random seed to {seed} for reproducibility")
        else:
        
            logger.info(f"  No seed set - results will vary between runs")
        # Load the web dataset (query images)
        web_dataset = WebSceneDataset(
            self.dataset_dir / "train_pbr_web",
            depth_scale=depth_scale
        )
        self.web_dataloader = IterableWebSceneDataset(web_dataset)
        
        # Load template dataset
        model_infos = inout.load_json(self.dataset_dir / "models_info.json")
        template_config['dir'] = f"{template_config['dir']}/{dataset_name}"
        
        # Create config object with required attributes
        # TemplateDataset.from_config expects: dir, pose_name, num_templates, scale_factor
        config_obj = type('Config', (), {
            'dir': template_config['dir'],
            'pose_name': template_config.get('pose_name', 'object_poses/OBJECT_ID.npy'),
            'num_templates': template_config.get('num_templates', 162),
            'scale_factor': template_config.get('scale_factor', 10.0),
            'level_templates': template_config.get('level_templates', 1),
            'pose_distribution': template_config.get('pose_distribution', 'all'),
        })()
        
        self.template_dataset = TemplateDataset.from_config(
            model_infos, config_obj
        )
        
        logger.info(f"  Loaded {len(self.template_dataset)} object templates")
        
        # Initialize template selector with seed for reproducibility
        self.template_selector = TemplateSelector(
            level_templates=template_config.get('level_templates', 1),
            pose_distribution=template_config.get('pose_distribution', 'all'),
            num_positive=num_positive_templates,
            num_negative=num_negative_templates,
            min_negative_angle_deg=min_negative_angle_deg,
            d_ref_random_ratio=d_ref_random_ratio,
            seed=seed,  # Use same seed for reproducibility
        )
        
        self.num_templates = num_positive_templates + num_negative_templates
        logger.info(f"  Templates per query: S_p={num_positive_templates}, S_n={num_negative_templates}, S={self.num_templates}")
        
        # Initialize flow computer
        flow_config = flow_config or {}
        self.flow_computer = FlowComputer(
            patch_size=patch_size,
            compute_visibility=flow_config.get('compute_visibility', True),
            compute_patch_visibility=flow_config.get('compute_patch_visibility', True),
            visibility_threshold=flow_config.get('visibility_threshold', 0.1),
        )
        
        logger.info(f"  Flow computation: visibility={self.flow_computer.compute_visibility}, "
                   f"patch_visibility={self.flow_computer.compute_patch_visibility}")
        
        # Setup transforms
        self._setup_transforms()

        self.debug_mode = True if "debug" in kwargs and kwargs["debug"] else False
    
    def _setup_transforms(self):
        """Setup data augmentation transforms"""
        transforms = self.transforms
        
        # Normalize transform
        if 'normalize' in transforms:
            self.normalize = transforms['normalize']
        else:
            # Default normalization (ImageNet stats)
            import torchvision.transforms as T
            self.normalize = T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        
        # Crop transform
        if 'crop_transform' in transforms:
            self.crop_transform = transforms['crop_transform']
        else:
            from src.utils.crop import CropResizePad
            self.crop_transform = CropResizePad(target_size=self.image_size)
        
        # RGB augmentation
        self.rgb_augmentation = transforms.get('rgb_augmentation', False)
        if self.rgb_augmentation and 'rgb_transform' in transforms:
            self.rgb_transform = transforms['rgb_transform']
        else:
            self.rgb_transform = None
        
        # Inplane augmentation (important for VPOG)
        self.inplane_augmentation = transforms.get('inplane_augmentation', False)
    
    def process_query(
        self,
        batch: SceneObservation,
        min_box_size: int = 50,
    ) -> tc.PandasTensorCollection:
        """
        Process query images from the batch.
        Similar to GigaPose's process_real but adapted for VPOG.
        
        Args:
            batch: Batch of scene observations
            min_box_size: Minimum bounding box size
        
        Returns:
            Processed query data
        """
        # Get ground truth data
        rgb = batch["rgb"] / 255.0
        depth = batch["depth"].squeeze(1) if "depth" in batch else None
        detections = batch["gt_detections"]
        data = batch["gt_data"]
        
        # Make bounding boxes square
        bboxes = BoundingBox(detections.bboxes, "xywh")
        
        # Sample up to batch_size objects
        idx_selected = np.random.choice(
            np.arange(len(detections.bboxes)),
            min(self.batch_size, len(detections.bboxes)),
            replace=False,
        )
        
        bboxes = bboxes.reset(idx_selected)
        batch_im_id = detections[idx_selected].infos.batch_im_id
        masks = data.masks[idx_selected]
        K = data.K[idx_selected].float()
        pose = data.TWO[idx_selected].float()
        
        rgb = rgb[batch_im_id]
        if depth is not None:
            depth = depth[batch_im_id]
        
        # Crop RGB WITH background (model input)
        centered_rgb_data = self.crop_transform(bboxes.xyxy_box, images=rgb)
        centered_rgb = centered_rgb_data["images"]  # [B, 3, 224, 224] WITH background
        M_crop = centered_rgb_data["M"]  # Crop transformation matrix [B, 3, 3]
        
        # Crop depth if available and apply mask
        if depth is not None:
            # Apply mask to depth first (zero out background)
            masked_depth = depth * masks  # [B, H, W] - zero out background
            
            # Depth is [B, H, W], need to add channel dim for crop_transform
            depth_4d = masked_depth.unsqueeze(1)  # [B, 1, H, W]
            centered_depth_data = self.crop_transform(bboxes.xyxy_box, images=depth_4d)
            centered_depth = centered_depth_data["images"].squeeze(1)  # [B, H, W]
        else:
            centered_depth = None
        
        # Apply crop transformation to intrinsics: K_cropped = M @ K
        K_cropped = torch.bmm(M_crop, K)  # [B, 3, 3]
        
        # Masked RGB (for visualization)
        m_rgb = rgb * masks[:, None, :, :]
        m_rgba = torch.cat([m_rgb, masks[:, None, :, :]], dim=1)
        
        # Crop masked RGB
        cropped_data = self.crop_transform(bboxes.xyxy_box, images=m_rgba)
        
        out_data = tc.PandasTensorCollection(
            full_rgb=rgb,
            centered_rgb=centered_rgb,  # Cropped WITH background
            full_depth=depth,
            depth=centered_depth,  # Cropped depth for flow computation
            K=K_cropped,  # CORRECTED: Transformed intrinsics for cropped images
            K_original=K,  # ORIGINAL intrinsics before crop
            mask_original=masks,  # ORIGINAL mask before crop
            rgb=cropped_data["images"][:, :3],  # Cropped masked
            mask=cropped_data["images"][:, -1],
            M=cropped_data["M"],  # Crop transformation matrix
            pose=pose,
            infos=data[idx_selected].infos,
            bboxes=bboxes.xyxy_box,  # Store bboxes for visualization
        )
        
        return out_data
    
    def process_templates(
        self,
        query_data: tc.PandasTensorCollection,
    ) -> Tuple[tc.PandasTensorCollection, Dict]:
        """
        Process templates for each query:
        1. Select S_p + S_n templates
        2. Load template images
        3. Extract d_ref
        
        Args:
            query_data: Processed query data
        
        Returns:
            Tuple of (template_data, selection_info)
        """
        batch_size = len(query_data)
        
        # Storage for template data
        template_rgbas = []
        template_depths = []
        template_depths_original = []  # Store ORIGINAL uncropped depths
        template_Ks = []
        template_Ms = []
        template_poses = []
        template_indices_list = []
        template_types_list = []
        d_refs = []
        
        labels = query_data.infos.label
        query_poses = query_data.pose.cpu().numpy()
        
        K_template = torch.from_numpy(self.template_dataset.K).float()
        
        for label, query_pose in zip(labels, query_poses):
            # Load template object first
            template_object = self.template_dataset.get_object_templates(label)
            
            # STEP 1: Find nearest OOP (out-of-plane) template t* using GigaPose's method
            # This finds the template with the same viewing direction, ignoring in-plane rotation
            from src.custom_megapose.template_dataset import NearestTemplateFinder
            
            config_obj = type('Config', (), {
                'level_templates': self.template_selector.level_templates,
                'pose_distribution': self.template_selector.pose_distribution,
            })()
            
            template_finder = NearestTemplateFinder(config_obj)
            nearest_info = template_finder.search_nearest_template(query_pose[:3, :3])
            nearest_template_idx = nearest_info['view_id']
            
            # DEBUG: Log for verification
            if label == '733' or (len(template_indices_list) == 0 and logger.level <= 20):
                logger.info(f"  Object {label}: Nearest OOP template t* = {nearest_template_idx}")
            
            # STEP 2: Find S_p-1 additional templates that are closest to t* in full SO(3)
            # These provide small perturbations around t* to reveal more pixels
            all_template_poses = self.template_selector.template_poses
            nearest_template_pose = all_template_poses[nearest_template_idx]
            
            # Compute SO(3) distances from ALL templates to t*
            angles_to_nearest = np.zeros(len(all_template_poses))
            R_nearest = nearest_template_pose[:3, :3]
            
            for i, template_pose in enumerate(all_template_poses):
                R_template = template_pose[:3, :3]
                R_rel = R_nearest.T @ R_template
                trace = np.trace(R_rel)
                cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
                angles_to_nearest[i] = np.rad2deg(np.arccos(cos_angle))
            
            # Sort by distance to t* and select S_p-1 nearest (excluding t* itself)
            sorted_indices = np.argsort(angles_to_nearest)
            positive_indices = [nearest_template_idx]  # t* is first
            for idx in sorted_indices[1:]:  # Skip index 0 which is t* itself
                if len(positive_indices) >= self.template_selector.num_positive:
                    break
                if int(idx) != nearest_template_idx:  # Double check we're not adding t* again
                    positive_indices.append(int(idx))
            
            # STEP 3: Select negative templates (far from query in OOP)
            # Compute angles from query to all templates
            angles_to_query = np.zeros(len(all_template_poses))
            R_query = query_pose[:3, :3]
            for i, template_pose in enumerate(all_template_poses):
                R_template = template_pose[:3, :3]
                R_rel = R_query.T @ R_template
                trace = np.trace(R_rel)
                cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
                angles_to_query[i] = np.rad2deg(np.arccos(cos_angle))
            
            negative_candidates = np.where(angles_to_query >= self.template_selector.min_negative_angle_deg)[0]
            if len(negative_candidates) < self.template_selector.num_negative:
                sorted_by_distance = np.argsort(-angles_to_query)
                negative_indices = sorted_by_distance[:self.template_selector.num_negative].tolist()
            else:
                sampled = self.template_selector.rng.choice(
                    negative_candidates, size=self.template_selector.num_negative, replace=False
                )
                negative_indices = sampled.tolist()
            
            # Build selection result
            selection = {
                'nearest_template_idx': nearest_template_idx,
                'positive_indices': positive_indices,
                'negative_indices': negative_indices,
                'all_indices': positive_indices + negative_indices,
            }
            
            # Extract d_ref from t*
            d_ref_pose = all_template_poses[nearest_template_idx]
            from src.lib3d.numpy import opencv2opengl
            d_ref_pose_opengl = opencv2opengl(d_ref_pose)
            d_ref = d_ref_pose_opengl[:3, 2]
            d_ref = d_ref / np.linalg.norm(d_ref)
            selection['d_ref'] = d_ref
            
            positive_indices = selection['positive_indices']
            negative_indices = selection['negative_indices']
            all_indices = selection['all_indices']
            d_ref = selection['d_ref']
            
            # Load each selected template
            batch_template_data = {
                'rgba': [],
                'depth': [],
                'K': [],
                'M': [],
                'pose': [],
                'box': [],
            }
            
            # Load ACTUAL template object poses (used during rendering)
            # These have z=diameter and match the rendered depth maps
            # Get template_dir from the template object
            template_dir_base = Path(template_object.template_dir).parent
            template_poses_path = template_dir_base / "object_poses" / f"{int(label):06d}.npy"
            template_poses_actual = np.load(template_poses_path)  # [N, 4, 4], z=diameter
            
            for idx in all_indices:
                # Get view ID for this template
                view_id = self.template_selector.get_template_view_id(idx)
                
                # Apply inplane augmentation if enabled
                if self.inplane_augmentation:
                    inplane = np.random.randint(0, 360)
                else:
                    inplane = 0
                
                # Load template data
                template_data = template_object.read_train_mode(
                    transform=None,
                    object_info={'view_id': view_id, 'inplane': inplane},
                    num_negatives=0,
                )
                
                # Use ACTUAL template pose (used during rendering, z=diameter)
                template_pose = template_poses_actual[idx]
                
                # Extract positive template data
                batch_template_data['rgba'].append(template_data['pos_rgba'])
                batch_template_data['depth'].append(template_data['pos_depth'])
                batch_template_data['box'].append(template_data['pos_box'])
                # Use template pose from selector (as torch tensor)
                batch_template_data['pose'].append(torch.from_numpy(template_pose).float())
            
            # Stack template data
            for key in ['rgba', 'depth', 'box', 'pose']:
                batch_template_data[key] = torch.stack(batch_template_data[key], dim=0)
            
            # Crop templates (including depth)
            cropped = self.crop_transform(
                batch_template_data['box'],
                images=batch_template_data['rgba']
            )
            
            # Also crop depth using the same transform
            cropped_depth = self.crop_transform(
                batch_template_data['box'],
                images=batch_template_data['depth']
            )
            
            # Apply crop transformation to template intrinsics
            M_crop = cropped['M']  # [S, 3, 3]
            K_template_batch = K_template.unsqueeze(0).repeat(self.num_templates, 1, 1)  # [S, 3, 3]
            K_template_cropped = torch.bmm(M_crop, K_template_batch)  # [S, 3, 3]
            
            # Store CROPPED template data
            template_rgbas.append(cropped['images'][:, :3])
            template_depths.append(cropped_depth['images'].squeeze(1))  # Remove channel dim (dim 1, not 2)
            template_Ks.append(K_template_cropped)  # CORRECTED: Use transformed intrinsics
            template_Ms.append(cropped['M'])
            template_poses.append(batch_template_data['pose'])
            
            # Store ORIGINAL template data (before crop) [S, H_orig, W_orig]
            template_depths_original.append(batch_template_data['depth'].squeeze(1) if batch_template_data['depth'].dim() == 4 else batch_template_data['depth'])
            
            # Store selection info
            template_indices_list.append(all_indices)
            template_types = [0] * len(positive_indices) + [1] * len(negative_indices)
            template_types_list.append(template_types)
            d_refs.append(d_ref)
        
        # Stack all data
        template_data = tc.PandasTensorCollection(
            rgb=torch.stack(template_rgbas, dim=0),  # [B, S, C, H, W]
            depth=torch.stack(template_depths, dim=0),  # [B, S, H, W] - cropped
            depth_original=torch.stack(template_depths_original, dim=0),  # [B, S, H_orig, W_orig] - ORIGINAL
            K=torch.stack(template_Ks, dim=0),  # [B, S, 3, 3] - cropped
            K_original=K_template,  # [3, 3] ORIGINAL template K (same for all templates)
            M=torch.stack(template_Ms, dim=0),  # [B, S, 3, 3]
            pose=torch.stack(template_poses, dim=0),  # [B, S, 4, 4]
            infos=query_data.infos,
        )
        
        selection_info = {
            'template_indices': torch.tensor(template_indices_list),  # [B, S]
            'template_types': torch.tensor(template_types_list),  # [B, S]
            'd_ref': torch.tensor(np.stack(d_refs, axis=0)).float(),  # [B, 3]
        }
        
        return template_data, selection_info
    
    def unproject_query_depth(
        self,
        depth: torch.Tensor,
        K: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Unproject query depth image to 3D point cloud in camera frame.
        
        Args:
            depth: [H, W] depth map in millimeters (BOP scale)
            K: [3, 3] camera intrinsics
            mask: [H, W] optional mask (only unproject masked pixels)
        
        Returns:
            points_cam: [H, W, 3] 3D points in camera frame (mm)
        """
        H, W = depth.shape
        device = depth.device
        
        # Create pixel grid
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Extract intrinsics
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        
        # Unproject to 3D
        z = depth
        x = (x_grid - cx) * z / fx
        y = (y_grid - cy) * z / fy
        
        points_cam = torch.stack([x, y, z], dim=-1)  # [H, W, 3]
        
        # Apply mask if provided
        if mask is not None:
            points_cam = points_cam * mask.unsqueeze(-1)
        
        return points_cam
    
    def transform_query_to_template_space(
        self,
        query_points_cam: torch.Tensor,
        query_pose: torch.Tensor,
        template_pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transform 3D points from query camera frame to template camera frame.
        
        Key insight from debugging:
        - Query: BOP model (scale 0.1) at real poses → depths in mm
        - Template: Normalized model × 10 at scaled poses → depths in "template units"
        - To match: transform geometrically, NO scaling needed!
        
        The template depth maps were rendered with:
        1. Normalized model (100x smaller than orig, same as BOP × 10)
        2. At pose with z = diameter (4775mm for obj 733)
        3. Depth values are stored directly from rendering
        
        Args:
            query_points_cam: [..., 3] points in query camera frame (BOP scale, mm)
            query_pose: [4, 4] TWO matrix (world-to-query-camera)
            template_pose: [4, 4] TWO matrix (world-to-template-camera)
        
        Returns:
            points_template_cam: [..., 3] points in template camera frame (same units as template depth)
        """
        original_shape = query_points_cam.shape
        points_flat = query_points_cam.reshape(-1, 3)  # [N, 3]
        
        # Step 1: Transform from query camera to world frame
        # query_cam = TWO_q @ world  =>  world = inv(TWO_q) @ query_cam
        T_q2w = torch.inverse(query_pose)
        points_homo = torch.cat([points_flat, torch.ones(len(points_flat), 1, device=points_flat.device)], dim=1)
        points_world = (T_q2w @ points_homo.T).T[:, :3]  # [N, 3] in BOP scale (mm)
        
        # Step 2: Scale world points by 10 (BOP model → Normalized × 10 model)
        points_world_scaled = points_world * 10.0
        
        # Step 3: Transform from scaled world to template camera frame
        # template_cam = TWO_t @ world_scaled
        points_homo_world = torch.cat([points_world_scaled, torch.ones(len(points_world_scaled), 1, device=points_world_scaled.device)], dim=1)
        points_template_cam = (template_pose @ points_homo_world.T).T[:, :3]  # [N, 3]
        
        # NO division needed! The template depth was rendered with a 10x scaled model,
        # so after scaling world points by 10x and transforming to template camera,
        # the depths are already in the correct scale (matching template PNG).
        
        return points_template_cam.reshape(original_shape)
    
    def project_to_template_image(
        self,
        points_template_cam: torch.Tensor,
        K_template: torch.Tensor,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project 3D points to template image coordinates.
        
        Args:
            points_template_cam: [..., 3] points in template camera frame
            K_template: [3, 3] template camera intrinsics
            H, W: template image size
        
        Returns:
            u: [...] pixel x coordinates
            v: [...] pixel y coordinates  
            z: [...] depth values in template units
            valid: [...] boolean mask for valid projections
        """
        # Extract coordinates
        x, y, z = points_template_cam[..., 0], points_template_cam[..., 1], points_template_cam[..., 2]
        
        # Extract intrinsics
        fx, fy, cx, cy = K_template[0, 0], K_template[1, 1], K_template[0, 2], K_template[1, 2]
        
        # Project to image
        u = (x / z) * fx + cx
        v = (y / z) * fy + cy
        
        # Check valid projections
        valid = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        
        return u, v, z, valid
    
    def _project_pcd_helper(self, pcd_world, pose, K, H, W):
        """
        Helper for visualization - project world point cloud to depth image.
        
        Args:
            pcd_world: [N, 3] point cloud in world frame
            pose: [4, 4] TWO matrix (world-to-camera)
            K: [3, 3] camera intrinsics
            H, W: output image size
        
        Returns:
            [H, W] depth image with depth values
        """
        pcd_homo = np.concatenate([pcd_world, np.ones((len(pcd_world), 1))], axis=1)
        pcd_cam = (pose @ pcd_homo.T).T
        x, y, z = pcd_cam[:, 0], pcd_cam[:, 1], pcd_cam[:, 2]
        
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        u = (x / z) * fx + cx
        v = (y / z) * fy + cy
        
        depth_img = np.zeros((H, W))
        valid = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        for u_i, v_i, z_i in zip(u[valid].astype(int), v[valid].astype(int), z[valid]):
            if depth_img[v_i, u_i] == 0 or z_i < depth_img[v_i, u_i]:
                depth_img[v_i, u_i] = z_i
        return depth_img
    
    def _visualize_correspondences(
        self,
        query_data,
        template_data,
        points_q_cam,
        correspondences,
        q_pose,
        t_pose,
        q_K_orig,
        t_K_orig,
        q_depth_orig,
        t_depth_orig,
        q_mask_orig,
        batch_idx,
        template_idx,
    ):
        """
        Visualize correspondences between query and template (CORRECT LOGIC ONLY).
        
        Uses mesh projection to validate transformation - this is the GOOD method.
        """
        import matplotlib.pyplot as plt
        
        # Get metadata
        obj_label = query_data.infos.label[batch_idx]
        H_orig, W_orig = q_depth_orig.shape
        H_t_orig, W_t_orig = t_depth_orig.shape
        
        # Pose info
        q_pose_np = q_pose.cpu().numpy()
        t_pose_np = t_pose.cpu().numpy()
        
        logger.info(f"  Visualizing object {obj_label}:")
        logger.info(f"    Query pose t: {q_pose_np[:3, 3]}")
        logger.info(f"    Template pose t: {t_pose_np[:3, 3]}")
        
        # ===== COMPUTE VALIDATION DEPTHS (GOOD METHOD) =====
        query_depth_render = None
        template_depth_render = None
        
        try:
            # Extract query point cloud and transform to world
            valid_mask = (q_depth_orig > 0) & (q_mask_orig > 0.5)
            pcd_query_cam = points_q_cam[valid_mask].cpu().numpy()  # [N, 3]
            
            T_cam2world = np.linalg.inv(q_pose_np)
            pcd_homo = np.concatenate([pcd_query_cam, np.ones((len(pcd_query_cam), 1))], axis=1)
            pcd_world = (T_cam2world @ pcd_homo.T).T[:, :3]  # [N, 3] in BOP scale (mm)
            
            # Project to query (should match PNG)
            query_depth_render = self._project_pcd_helper(
                pcd_world, q_pose_np, q_K_orig.cpu().numpy(), H_orig, W_orig
            )
            
            # Project to template with 10x scaling (GOOD METHOD)
            template_depth_render = self._project_pcd_helper(
                pcd_world * 10, t_pose_np, t_K_orig.cpu().numpy(), H_t_orig, W_t_orig
            )
            
        except Exception as e:
            logger.warning(f"  Mesh projection failed: {e}")
        
        # ===== CREATE VISUALIZATION =====
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # ROW 0: RGB
        q_rgb = query_data.full_rgb[batch_idx].cpu().permute(1, 2, 0).numpy()
        q_mask_np = q_mask_orig.cpu().numpy()
        
        axes[0, 0].imshow(q_rgb)
        axes[0, 0].set_title(f'Query RGB\n{H_orig}x{W_orig}', fontweight='bold')
        axes[0, 0].axis('off')
        
        q_rgb_masked = q_rgb * q_mask_np[:, :, None]
        axes[0, 1].imshow(q_rgb_masked)
        axes[0, 1].set_title(f'Query Masked\nObj {obj_label}', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Show correspondence count
        if correspondences:
            axes[0, 2].text(0.5, 0.5, f"{correspondences['count']} correspondences\n\n"
                           f"Query: [{correspondences['query_z'].min():.0f}, {correspondences['query_z'].max():.0f}] mm\n"
                           f"Template: [{correspondences['template_z'].min():.0f}, {correspondences['template_z'].max():.0f}] units",
                           ha='center', va='center', fontsize=12, fontweight='bold')
        else:
            axes[0, 2].text(0.5, 0.5, 'NO CORRESPONDENCES', ha='center', va='center', 
                           fontsize=14, color='red', fontweight='bold')
        axes[0, 2].axis('off')
        axes[0, 3].axis('off')
        
        # ROW 1: Depth comparison (PNG vs Rendered)
        q_depth_np = (q_depth_orig * q_mask_orig).cpu().numpy()
        t_depth_np = t_depth_orig.cpu().numpy()
        
        # Query depth PNG
        im1 = axes[1, 0].imshow(q_depth_np, cmap='jet', vmin=0, 
                                 vmax=q_depth_np[q_depth_np>0].max() if (q_depth_np>0).any() else 1)
        axes[1, 0].set_title(f'Query Depth (PNG)\n[{q_depth_np[q_depth_np>0].min():.0f}, {q_depth_np[q_depth_np>0].max():.0f}] mm',
                            fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
        
        # Query rendered (should match PNG)
        if query_depth_render is not None and (query_depth_render > 0).any():
            im2 = axes[1, 1].imshow(query_depth_render, cmap='jet', vmin=0, 
                                     vmax=query_depth_render[query_depth_render>0].max())
            axes[1, 1].set_title(f'Query Rendered (VALIDATION)\n[{query_depth_render[query_depth_render>0].min():.0f}, {query_depth_render[query_depth_render>0].max():.0f}] mm',
                                fontweight='bold', color='green')
            plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
        axes[1, 1].axis('off')
        
        # Template depth PNG (CORRECT)
        im3 = axes[1, 2].imshow(t_depth_np, cmap='jet', vmin=0, vmax=t_depth_np.max())
        axes[1, 2].set_title(f'Template Depth (PNG - CORRECT)\n[{t_depth_np[t_depth_np>0].min():.0f}, {t_depth_np[t_depth_np>0].max():.0f}] units',
                            fontweight='bold', color='green')
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
        
        # Template rendered (CORRECT - should match PNG)
        if template_depth_render is not None and (template_depth_render > 0).any():
            im4 = axes[1, 3].imshow(template_depth_render, cmap='jet', vmin=0,
                                     vmax=template_depth_render[template_depth_render>0].max())
            axes[1, 3].set_title(f'Template Rendered (CORRECT)\n[{template_depth_render[template_depth_render>0].min():.0f}, {template_depth_render[template_depth_render>0].max():.0f}] units',
                                fontweight='bold', color='green')
            plt.colorbar(im4, ax=axes[1, 3], fraction=0.046)
        axes[1, 3].axis('off')
        
        # Title
        plt.suptitle(f'Object {obj_label}: Query-Template Correspondence Visualization\n'
                    f'Query Pose: {q_pose_np[:3, 3].astype(int)} mm | Template Pose: {t_pose_np[:3, 3].astype(int)} mm\n'
                    f'CORRECT Logic: template_depth_render matches template PNG!',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = Path(f"tmp/vpog_dataset_test/correspondences_b{batch_idx}_s{template_idx}.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved visualization to {save_path}")
    
    def _visualize_dense_patch_flow(
        self,
        query_rgb: torch.Tensor,          # [3, H, W]
        template_rgb: torch.Tensor,       # [3, H, W]
        q_mask: torch.Tensor,             # [H, W]
        t_depth: torch.Tensor,            # [H, W]
        flow_grid: torch.Tensor,          # [H_p, W_p, ps, ps, 2]
        weight_grid: torch.Tensor,        # [H_p, W_p, ps, ps]
        patch_has_object: torch.Tensor,   # [H_p, W_p]
        patch_is_visible: torch.Tensor,   # [H_p, W_p]
        patch_buddy_i: torch.Tensor,      # [H_p, W_p]
        patch_buddy_j: torch.Tensor,      # [H_p, W_p]
        batch_idx: int,
        template_idx: int,
        obj_label: int,
    ):
        """
        Visualize dense template→query correspondences:
        - Shows template patch and query patch side by side
        - Draws lines between corresponding pixels (template → query)
        - Colors correspond to different pixels for clarity
        - Red marks indicate pixels with no correspondence
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        from matplotlib.collections import LineCollection
        
        ps = self.patch_size
        H_p = self.num_patches_per_side
        
        # Find visible patches (using the first logic)
        visible_indices = torch.nonzero(patch_is_visible, as_tuple=False)
        if visible_indices.numel() == 0:
            logger.info(f"  [DENSE VIZ] No visible patches to visualize")
            return
        
        # Select up to 4 patches for visualization
        num_patches = min(4, len(visible_indices))
        selected_indices = visible_indices[:num_patches]
        
        # Convert images to numpy
        q_rgb_np = query_rgb.cpu().permute(1, 2, 0).numpy()
        q_rgb_np = np.clip(q_rgb_np, 0, 1)
        t_rgb_np = template_rgb.cpu().permute(1, 2, 0).numpy()
        t_rgb_np = np.clip(t_rgb_np, 0, 1)
        t_depth_np = t_depth.cpu().numpy()
        
        # Create figure with full images on top
        fig = plt.figure(figsize=(15, 4 + 4 * num_patches))
        
        # Top row: Full query and template images with patch boxes
        ax_query_full = plt.subplot2grid((num_patches + 1, 3), (0, 0), colspan=1)
        ax_query_full.imshow(q_rgb_np)
        ax_query_full.set_title('Query Image (with selected patches)', fontsize=11, fontweight='bold')
        ax_query_full.axis('off')
        
        ax_template_full = plt.subplot2grid((num_patches + 1, 3), (0, 1), colspan=1)
        ax_template_full.imshow(t_rgb_np)
        ax_template_full.set_title('Template Image (with buddy patches)', fontsize=11, fontweight='bold')
        ax_template_full.axis('off')
        
        # Draw boxes on full images for selected patches
        for idx in range(num_patches):
            i_q, j_q = selected_indices[idx].tolist()
            i_t = patch_buddy_i[i_q, j_q].item()
            j_t = patch_buddy_j[i_q, j_q].item()
            
            # Query patch box
            q_rect = mpatches.Rectangle(
                (j_q * ps, i_q * ps), ps, ps,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax_query_full.add_patch(q_rect)
            ax_query_full.text(
                j_q * ps + ps // 2, i_q * ps - 3,
                f'{idx+1}', color='red', fontsize=11, fontweight='bold',
                ha='center', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            # Template buddy patch box
            t_rect = mpatches.Rectangle(
                (j_t * ps, i_t * ps), ps, ps,
                linewidth=2, edgecolor='blue', facecolor='none'
            )
            ax_template_full.add_patch(t_rect)
            ax_template_full.text(
                j_t * ps + ps // 2, i_t * ps - 3,
                f'{idx+1}', color='blue', fontsize=11, fontweight='bold',
                ha='center', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        # Info panel in top right
        ax_info = plt.subplot2grid((num_patches + 1, 3), (0, 2), colspan=1)
        info_text = (
            f'Dense Flow Visualization\n'
            f'═══════════════════════\n\n'
            f'Direction: Template → Query\n'
            f'Coordinate System:\n'
            f'  Query patch center = (0, 0)\n'
            f'  Flow normalized by patch size\n\n'
            f'Patches shown: {num_patches}\n'
            f'Patch size: {ps}×{ps} pixels\n\n'
            f'Red boxes: Query patches\n'
            f'Blue boxes: Template buddies\n\n'
            f'In correspondence views:\n'
            f'  Blue dots: Template pixels\n'
            f'  Green dots: Query projections\n'
            f'  Red X: No correspondence'
        )
        ax_info.text(0.5, 0.5, info_text, ha='center', va='center',
                    fontsize=9, fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        ax_info.axis('off')
        
        # Create axes for patch details
        axes = []
        for idx in range(num_patches):
            row_axes = []
            for col in range(3):
                ax = plt.subplot2grid((num_patches + 1, 3), (idx + 1, col), colspan=1)
                row_axes.append(ax)
            axes.append(row_axes)
        if num_patches == 1:
            axes = [axes[0]]
        
        # Process each selected patch
        for idx in range(num_patches):
            i_q, j_q = selected_indices[idx].tolist()
            i_t = patch_buddy_i[i_q, j_q].item()
            j_t = patch_buddy_j[i_q, j_q].item()
            
            flow_patch = flow_grid[i_q, j_q].cpu().numpy()    # [ps, ps, 2]
            weight_patch = weight_grid[i_q, j_q].cpu().numpy()  # [ps, ps]
            
            # Extract patches
            # Query patch
            qy0, qy1 = i_q * ps, (i_q + 1) * ps
            qx0, qx1 = j_q * ps, (j_q + 1) * ps
            q_patch = q_rgb_np[qy0:qy1, qx0:qx1]
            
            # Template patch
            ty0, ty1 = i_t * ps, (i_t + 1) * ps
            tx0, tx1 = j_t * ps, (j_t + 1) * ps
            t_patch = t_rgb_np[ty0:ty1, tx0:tx1]
            t_patch_depth = t_depth_np[ty0:ty1, tx0:tx1]
            t_patch_mask = t_patch_depth > 0
            
            # Column 0: Template patch
            ax_t = axes[idx][0]
            ax_t.imshow(t_patch)
            ax_t.set_title(f'#{idx+1} Template Patch [{i_t},{j_t}]', fontsize=11, fontweight='bold')
            ax_t.axis('off')
            
            # Column 1: Query patch
            ax_q = axes[idx][1]
            ax_q.imshow(q_patch)
            ax_q.set_title(f'#{idx+1} Query Patch [{i_q},{j_q}]', fontsize=11, fontweight='bold')
            ax_q.axis('off')
            
            # Column 2: Correspondences (template left, query right)
            ax_corr = axes[idx][2]
            
            # Create side-by-side view
            combined = np.concatenate([t_patch, q_patch], axis=1)  # [ps, 2*ps, 3]
            ax_corr.imshow(combined)
            
            # Draw correspondences
            # Build grid of template pixel positions
            ty_grid, tx_grid = np.meshgrid(np.arange(ps), np.arange(ps), indexing='ij')
            
            # For each template pixel, check if it has a correspondence
            valid = weight_patch > 0.1
            
            # Mark template pixels without correspondence in red
            no_corr_y, no_corr_x = np.where(t_patch_mask & (~valid))
            ax_corr.scatter(no_corr_x, no_corr_y, c='red', s=5, alpha=0.6, marker='x')
            
            if valid.any():
                # Get valid correspondences
                ty_valid = ty_grid[valid]
                tx_valid = tx_grid[valid]
                
                # Flow gives offset from query patch center in normalized coords
                # Convert to pixel coords in query patch
                flow_u = flow_patch[..., 0][valid] * ps  # [N]
                flow_v = flow_patch[..., 1][valid] * ps  # [N]
                
                # Query projection: patch center + flow
                qy_proj = ps // 2 + flow_v
                qx_proj = ps // 2 + flow_u
                
                # Draw lines from template (left) to query (right, offset by ps)
                lines = []
                colors = []
                
                # Use colormap for variety
                cmap = plt.cm.get_cmap('tab20')
                num_valid = len(tx_valid)
                
                for i in range(num_valid):
                    # Template point (left side)
                    t_x, t_y = tx_valid[i], ty_valid[i]
                    # Query point (right side, offset by ps)
                    q_x, q_y = qx_proj[i] + ps, qy_proj[i]
                    
                    lines.append([(t_x, t_y), (q_x, q_y)])
                    colors.append(cmap(i % 20))
                
                # Draw lines
                lc = LineCollection(lines, colors=colors, linewidths=0.5, alpha=0.7)
                ax_corr.add_collection(lc)
                
                # Mark endpoints
                ax_corr.scatter(tx_valid, ty_valid, c='blue', s=10, alpha=0.8, marker='o')
                ax_corr.scatter(qx_proj + ps, qy_proj, c='green', s=10, alpha=0.8, marker='o')
            
            # Draw dividing line
            ax_corr.axvline(x=ps, color='white', linewidth=2, linestyle='--', alpha=0.7)
            
            # Add labels
            ax_corr.text(ps // 2, -2, 'Template', ha='center', va='bottom', 
                        fontsize=10, fontweight='bold', color='blue')
            ax_corr.text(ps + ps // 2, -2, 'Query', ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color='green')
            
            num_corr = valid.sum()
            num_total = t_patch_mask.sum()
            ax_corr.set_title(f'Correspondences: {num_corr}/{num_total} pixels\n'
                            f'(Template→Query, red x = no corr)', 
                            fontsize=10, fontweight='bold')
            ax_corr.axis('off')
        
        plt.suptitle(
            f'Dense Template→Query Flow: Object {obj_label}, Batch {batch_idx}, Template {template_idx}\n'
            f'Flow in query patch coordinates (center=0,0, normalized by patch size)',
            fontsize=13, fontweight='bold'
        )
        
        save_dir = Path("tmp/vpog_dataset_test")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"dense_flow_obj{obj_label}_b{batch_idx}_t{template_idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  [DENSE VIZ] ✓ Saved dense flow visualization to {save_path}")

    
    def _visualize_train_correspondences(
        self,
        query_data,
        template_data,
        points_q_cam,
        correspondences,
        q_pose,
        t_pose,
        q_K,
        t_K,
        q_depth,
        t_depth,
        q_mask,
        query_seen_map,
        template_seen_map,
        template_not_seen_map,
        patch_has_object,
        patch_is_visible,
        patch_flow_x,
        patch_flow_y,
        patch_buddy_i,
        patch_buddy_j,
        patch_center_mass_u,
        patch_center_mass_v,
        batch_idx,
        template_idx,
    ):
        """
        Visualize correspondences for TRAINING data (224x224 cropped).
        Uses mesh projection to validate transformation on cropped data.
        Also shows visibility maps.
        """
        import matplotlib.pyplot as plt
        
        # Get metadata
        obj_label = query_data.infos.label[batch_idx]
        H, W = q_depth.shape  # 224x224
        
        # Pose info
        q_pose_np = q_pose.cpu().numpy()
        t_pose_np = t_pose.cpu().numpy()
        
        logger.info(f"  [TRAIN VIZ] Object {obj_label}:")
        logger.info(f"    Query pose t: {q_pose_np[:3, 3]}")
        logger.info(f"    Template pose t: {t_pose_np[:3, 3]}")
        
        # ===== COMPUTE VALIDATION DEPTHS (GOOD METHOD) =====
        query_depth_render = None
        template_depth_render = None
        
        try:
            # Extract query point cloud and transform to world
            valid_mask = (q_depth > 0) & (q_mask > 0.5)
            pcd_query_cam = points_q_cam[valid_mask].cpu().numpy()  # [N, 3]
            
            T_cam2world = np.linalg.inv(q_pose_np)
            pcd_homo = np.concatenate([pcd_query_cam, np.ones((len(pcd_query_cam), 1))], axis=1)
            pcd_world = (T_cam2world @ pcd_homo.T).T[:, :3]  # [N, 3] in BOP scale (mm)
            
            # Project to query (should match cropped PNG)
            query_depth_render = self._project_pcd_helper(
                pcd_world, q_pose_np, q_K.cpu().numpy(), H, W
            )
            
            # Project to template with 10x scaling (GOOD METHOD)
            template_depth_render = self._project_pcd_helper(
                pcd_world * 10, t_pose_np, t_K.cpu().numpy(), H, W
            )
            
        except Exception as e:
            logger.warning(f"  [TRAIN VIZ] Mesh projection failed: {e}")
        
        # ===== CREATE VISUALIZATION =====
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        # ROW 0: RGB (from centered_rgb)
        q_rgb = query_data.centered_rgb[batch_idx].cpu().permute(1, 2, 0).numpy()
        
        axes[0, 0].imshow(np.clip(q_rgb, 0, 1))
        axes[0, 0].set_title(f'Query RGB (Centered)\n{H}x{W}', fontweight='bold')
        axes[0, 0].axis('off')
        
        q_mask_np = q_mask.cpu().numpy()
        q_rgb_masked = q_rgb * q_mask_np[:, :, None]
        axes[0, 1].imshow(q_rgb_masked)
        axes[0, 1].set_title(f'Query Masked\nObj {obj_label}', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Template RGB (cropped)
        t_rgb = template_data.rgb[batch_idx, template_idx].cpu().permute(1, 2, 0).numpy()
        axes[0, 2].imshow(np.clip(t_rgb, 0, 1))
        axes[0, 2].set_title(f'Template RGB\n{H}x{W}', fontweight='bold')
        axes[0, 2].axis('off')
        
        # Show correspondence count
        if correspondences:
            axes[0, 3].text(0.5, 0.5, f"{correspondences['count']} correspondences\n\n"
                           f"Query: [{correspondences['query_z'].min():.0f}, {correspondences['query_z'].max():.0f}] mm\n"
                           f"Template: [{correspondences['template_z'].min():.0f}, {correspondences['template_z'].max():.0f}] units",
                           ha='center', va='center', fontsize=12, fontweight='bold')
        else:
            axes[0, 3].text(0.5, 0.5, 'NO CORRESPONDENCES', ha='center', va='center', 
                           fontsize=14, color='red', fontweight='bold')
        axes[0, 3].axis('off')
        
        # ROW 1: Depth comparison (PNG vs Rendered) - CROPPED 224x224
        q_depth_np = (q_depth * q_mask).cpu().numpy()
        t_depth_np = t_depth.cpu().numpy()
        
        # Query depth PNG (cropped)
        im1 = axes[1, 0].imshow(q_depth_np, cmap='jet', vmin=0, 
                                 vmax=q_depth_np[q_depth_np>0].max() if (q_depth_np>0).any() else 1)
        axes[1, 0].set_title(f'Query Depth (PNG 224x224)\n[{q_depth_np[q_depth_np>0].min():.0f}, {q_depth_np[q_depth_np>0].max():.0f}] mm',
                            fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
        
        # Query rendered (should match PNG)
        if query_depth_render is not None and (query_depth_render > 0).any():
            im2 = axes[1, 1].imshow(query_depth_render, cmap='jet', vmin=0, 
                                     vmax=query_depth_render[query_depth_render>0].max())
            axes[1, 1].set_title(f'Query Rendered (VALIDATION)\n[{query_depth_render[query_depth_render>0].min():.0f}, {query_depth_render[query_depth_render>0].max():.0f}] mm',
                                fontweight='bold', color='green')
            plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
        axes[1, 1].axis('off')
        
        # Template depth PNG (CORRECT) - cropped
        im3 = axes[1, 2].imshow(t_depth_np, cmap='jet', vmin=0, vmax=t_depth_np.max())
        axes[1, 2].set_title(f'Template Depth (PNG 224x224 - CORRECT)\n[{t_depth_np[t_depth_np>0].min():.0f}, {t_depth_np[t_depth_np>0].max():.0f}] units',
                            fontweight='bold', color='green')
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
        
        # Template rendered (CORRECT - should match PNG)
        if template_depth_render is not None and (template_depth_render > 0).any():
            im4 = axes[1, 3].imshow(template_depth_render, cmap='jet', vmin=0,
                                     vmax=template_depth_render[template_depth_render>0].max())
            axes[1, 3].set_title(f'Template Rendered (CORRECT)\n[{template_depth_render[template_depth_render>0].min():.0f}, {template_depth_render[template_depth_render>0].max():.0f}] units',
                                fontweight='bold', color='green')
            plt.colorbar(im4, ax=axes[1, 3], fraction=0.046)
        axes[1, 3].axis('off')
        
        # ROW 2: VISIBILITY MAPS (only meaningful within object masks)
        query_seen_np = query_seen_map.cpu().numpy()
        query_not_seen_np = (q_mask.cpu().numpy() > 0.5) & (~query_seen_np)  # Within mask but not seen
        template_seen_np = template_seen_map.cpu().numpy()
        template_not_seen_np = template_not_seen_map.cpu().numpy()
        
        # Get valid pixel counts
        q_mask_np = (q_mask.cpu().numpy() > 0.5)
        t_mask_np = (t_depth_np > 0)
        
        # Query Seen Map: which query pixels (within mask) are visible in template
        # Create RGB visualization: gray=outside mask, green=seen, black=not computed
        query_seen_viz = np.zeros((H, W, 3))
        query_seen_viz[~q_mask_np] = [0.3, 0.3, 0.3]  # Gray for outside mask
        query_seen_viz[query_seen_np] = [0, 1, 0]  # Green for seen
        query_seen_viz[query_not_seen_np] = [1, 0, 0]  # Red for not seen (within mask)
        
        axes[2, 0].imshow(query_seen_viz)
        axes[2, 0].set_title(f'Query Seen Map\n{query_seen_np.sum()}/{q_mask_np.sum()} pixels visible\nGreen=Seen, Red=Occluded, Gray=Outside',
                            fontweight='bold')
        axes[2, 0].axis('off')
        
        # Query NOT seen (within mask only)
        query_not_seen_viz = np.zeros((H, W, 3))
        query_not_seen_viz[~q_mask_np] = [0.3, 0.3, 0.3]  # Gray for outside mask
        query_not_seen_viz[query_not_seen_np] = [1, 0, 0]  # Red for not seen
        query_not_seen_viz[query_seen_np] = [0, 0.5, 0]  # Dark green for seen (for reference)
        
        axes[2, 1].imshow(query_not_seen_viz)
        axes[2, 1].set_title(f'Query NOT Seen (Occluded)\n{query_not_seen_np.sum()}/{q_mask_np.sum()} pixels\nRed=Occluded, Gray=Outside',
                            fontweight='bold', color='red')
        axes[2, 1].axis('off')
        
        # Template Seen Map: which template pixels have query projections
        template_seen_viz = np.zeros((H, W, 3))
        template_seen_viz[~t_mask_np] = [0.3, 0.3, 0.3]  # Gray for outside mask
        template_seen_viz[template_seen_np] = [0, 1, 0]  # Green for seen
        template_seen_viz[template_not_seen_np] = [1, 0.5, 0]  # Orange for not in query
        
        axes[2, 2].imshow(template_seen_viz)
        axes[2, 2].set_title(f'Template Seen Map\n{template_seen_np.sum()}/{t_mask_np.sum()} pixels\nGreen=Has Query, Orange=No Query, Gray=Outside',
                            fontweight='bold')
        axes[2, 2].axis('off')
        
        # Template NOT Seen Map: template pixels (within mask) not in query
        template_not_seen_viz = np.zeros((H, W, 3))
        template_not_seen_viz[~t_mask_np] = [0.3, 0.3, 0.3]  # Gray for outside mask
        template_not_seen_viz[template_not_seen_np] = [1, 0.5, 0]  # Orange for not in query
        template_not_seen_viz[template_seen_np] = [0, 0.5, 0]  # Dark green for has query (for reference)
        
        axes[2, 3].imshow(template_not_seen_viz)
        axes[2, 3].set_title(f'Template NOT in Query\n{template_not_seen_np.sum()}/{t_mask_np.sum()} pixels\nOrange=Not in Query, Gray=Outside',
                            fontweight='bold', color='orange')
        axes[2, 3].axis('off')
        
        # Title
        plt.suptitle(f'[TRAIN] Object {obj_label}: Query-Template Correspondence & Visibility Maps (224x224)\n'
                    f'Query Pose: {q_pose_np[:3, 3].astype(int)} mm | Template Pose: {t_pose_np[:3, 3].astype(int)} mm\n'
                    f'Row 1: Depth Validation (CORRECT) | Row 2: Visibility Maps',
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save with unique identifier (object label + batch + template indices)
        # Get actual template index from selection_info if available
        save_path = Path(f"tmp/vpog_dataset_test/train_corr_obj{obj_label}_b{batch_idx}_t{template_idx}.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  [TRAIN VIZ] ✓ Saved visualization to {save_path}")
        
        # Also create patch correspondence visualization
        self._visualize_patch_correspondences(
            query_data=query_data,
            template_data=template_data,
            q_mask=q_mask,
            patch_has_object=patch_has_object,
            patch_is_visible=patch_is_visible,
            patch_flow_x=patch_flow_x,
            patch_flow_y=patch_flow_y,
            patch_buddy_i=patch_buddy_i,
            patch_buddy_j=patch_buddy_j,
            patch_center_mass_u=patch_center_mass_u,
            patch_center_mass_v=patch_center_mass_v,
            batch_idx=batch_idx,
            template_idx=template_idx,
            obj_label=obj_label,
        )
    
    def _visualize_patch_correspondences(
        self,
        query_data,
        template_data,
        q_mask,
        patch_has_object,
        patch_is_visible,
        patch_flow_x,
        patch_flow_y,
        patch_buddy_i,
        patch_buddy_j,
        patch_center_mass_u,
        patch_center_mass_v,
        batch_idx,
        template_idx,
        obj_label,
    ):
        """
        Visualize patch-level correspondences between query and template.
        Shows query patches (green=visible, red=not visible) and zooms into examples.
        For template, shows 3x3 grid of patches centered on buddy patch with actual query projection.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import matplotlib.patches as mpatches
        
        ps = self.patch_size
        H_p = self.num_patches_per_side
        
        # Get RGB images
        q_rgb = query_data.centered_rgb[batch_idx].cpu().permute(1, 2, 0).numpy()
        t_rgb = template_data.rgb[batch_idx, template_idx].cpu().permute(1, 2, 0).numpy()
        
        # Create figure with patch visualization + 3 zoom examples
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # ===== ROW 0: Query and Template with patch boxes =====
        ax_query = fig.add_subplot(gs[0, 0:2])
        ax_template = fig.add_subplot(gs[0, 2:4])
        
        # Query with patch boxes
        ax_query.imshow(np.clip(q_rgb, 0, 1))
        ax_query.set_title(f'Query Patches (Obj {obj_label})\nGreen=Visible, Red=Occluded, No box=No object',
                          fontsize=12, fontweight='bold')
        ax_query.axis('off')
        
        # Draw patch boxes on query
        for i_p in range(H_p):
            for j_p in range(H_p):
                if patch_has_object[i_p, j_p]:
                    y_start, x_start = i_p * ps, j_p * ps
                    
                    if patch_is_visible[i_p, j_p]:
                        color = 'green'
                        linewidth = 2
                    else:
                        color = 'red'
                        linewidth = 1
                    
                    rect = Rectangle((x_start, y_start), ps, ps,
                                   linewidth=linewidth, edgecolor=color, facecolor='none')
                    ax_query.add_patch(rect)
        
        # Template with patch boxes
        ax_template.imshow(np.clip(t_rgb, 0, 1))
        ax_template.set_title(f'Template Patches\nShowing buddy patches for visible query patches',
                             fontsize=12, fontweight='bold')
        ax_template.axis('off')
        
        # Draw buddy patch boxes on template (only for visible query patches)
        for i_p in range(H_p):
            for j_p in range(H_p):
                if patch_has_object[i_p, j_p] and patch_is_visible[i_p, j_p]:
                    buddy_i = patch_buddy_i[i_p, j_p].item()
                    buddy_j = patch_buddy_j[i_p, j_p].item()
                    
                    y_start, x_start = buddy_i * ps, buddy_j * ps
                    
                    rect = Rectangle((x_start, y_start), ps, ps,
                                   linewidth=2, edgecolor='blue', facecolor='none', alpha=0.7)
                    ax_template.add_patch(rect)
        
        # ===== ROW 1-3: Zoom into 3 randomly sampled patches =====
        # Find visible patches and randomly sample 3 of them
        visible_patches = torch.nonzero(patch_is_visible, as_tuple=False)
        
        if len(visible_patches) > 0:
            # Randomly sample up to 3 patches
            num_samples = min(3, len(visible_patches))
            
            # Random sampling without replacement
            perm = torch.randperm(len(visible_patches))[:num_samples]
            sampled_patches = visible_patches[perm]  # [num_samples, 2]
            
            # For first sample (to maintain variable names below)
            sample_i, sample_j = sampled_patches[0]
            sample_i, sample_j = sample_i.item(), sample_j.item()
            
            # Visualize each sampled patch
            for idx in range(num_samples):
                sample_i, sample_j = sampled_patches[idx]
                sample_i, sample_j = sample_i.item(), sample_j.item()
                
                buddy_i = patch_buddy_i[sample_i, sample_j].item()
                buddy_j = patch_buddy_j[sample_i, sample_j].item()
                
                flow_x_px = patch_flow_x[sample_i, sample_j].item() * ps
                flow_y_px = patch_flow_y[sample_i, sample_j].item() * ps
                
                # Get center of mass projection for this patch
                center_mass_u = patch_center_mass_u[sample_i, sample_j].item()
                center_mass_v = patch_center_mass_v[sample_i, sample_j].item()
                
                # Extract query patch
                q_y_start, q_y_end = sample_i * ps, (sample_i + 1) * ps
                q_x_start, q_x_end = sample_j * ps, (sample_j + 1) * ps
                q_patch = q_rgb[q_y_start:q_y_end, q_x_start:q_x_end]
                
                # Extract 3x3 template patches centered on buddy patch
                # Clamp to valid range
                H_t, W_t = t_rgb.shape[:2]
                t_y_start_3x3 = max(0, (buddy_i - 1) * ps)
                t_y_end_3x3 = min(H_t, (buddy_i + 2) * ps)
                t_x_start_3x3 = max(0, (buddy_j - 1) * ps)
                t_x_end_3x3 = min(W_t, (buddy_j + 2) * ps)
                t_patch_3x3 = t_rgb[t_y_start_3x3:t_y_end_3x3, t_x_start_3x3:t_x_end_3x3]
                
                row = idx + 1  # Start from row 1 (row 0 has full images)
                
                # Query patch zoom
                ax_q_zoom = fig.add_subplot(gs[row, 0])
                ax_q_zoom.imshow(np.clip(q_patch, 0, 1))
                # Mark center
                q_center = ps // 2
                ax_q_zoom.plot(q_center, q_center, 'r+', markersize=20, markeredgewidth=3)
                ax_q_zoom.set_title(f'Sample {idx+1}: Query Patch [{sample_i},{sample_j}]\nCenter marked with red +',
                                   fontsize=10, fontweight='bold')
                ax_q_zoom.axis('off')
                
                # Template 3x3 patches with buddy in center
                ax_t_zoom = fig.add_subplot(gs[row, 1])
                ax_t_zoom.imshow(np.clip(t_patch_3x3, 0, 1))
                
                # Mark buddy patch center (in 3x3 grid coords)
                # Buddy patch is at position [buddy_i, buddy_j] in original grid
                # In 3x3 crop, it's offset by the crop start
                buddy_center_x_in_crop = (buddy_j * ps + ps // 2) - t_x_start_3x3
                buddy_center_y_in_crop = (buddy_i * ps + ps // 2) - t_y_start_3x3
                ax_t_zoom.plot(buddy_center_x_in_crop, buddy_center_y_in_crop, 'b+', markersize=20, markeredgewidth=3)
                
                # Mark where query center actually projects (center of mass)
                query_proj_x_in_crop = center_mass_u - t_x_start_3x3
                query_proj_y_in_crop = center_mass_v - t_y_start_3x3
                ax_t_zoom.plot(query_proj_x_in_crop, query_proj_y_in_crop, 'rx', markersize=15, markeredgewidth=3)
                
                # Draw grid lines to show patch boundaries
                for i in range(1, 3):
                    y_line = i * ps - (buddy_i - 1) * ps if buddy_i > 0 else i * ps
                    if 0 <= y_line <= t_patch_3x3.shape[0]:
                        ax_t_zoom.axhline(y=y_line, color='white', linewidth=1, alpha=0.5)
                for j in range(1, 3):
                    x_line = j * ps - (buddy_j - 1) * ps if buddy_j > 0 else j * ps
                    if 0 <= x_line <= t_patch_3x3.shape[1]:
                        ax_t_zoom.axvline(x=x_line, color='white', linewidth=1, alpha=0.5)
                
                # Draw box around center (buddy) patch
                center_patch_x_start = ps if buddy_j > 0 else 0
                center_patch_y_start = ps if buddy_i > 0 else 0
                center_patch_rect = Rectangle(
                    (center_patch_x_start, center_patch_y_start), ps, ps,
                    linewidth=2, edgecolor='yellow', facecolor='none', linestyle='-'
                )
                ax_t_zoom.add_patch(center_patch_rect)
                
                ax_t_zoom.set_title(f'Template 3×3 Patches (Buddy [{buddy_i},{buddy_j}] in center)\nBlue +=buddy center, Red x=query projection, Yellow box=buddy patch',
                                   fontsize=10, fontweight='bold')
                ax_t_zoom.axis('off')
                
                # Flow visualization
                ax_flow = fig.add_subplot(gs[row, 2])
                ax_flow.text(0.5, 0.5, 
                            f'Patch Correspondence:\n\n'
                            f'Query: [{sample_i}, {sample_j}]\n'
                            f'Template Buddy: [{buddy_i}, {buddy_j}]\n\n'
                            f'Flow (pixels):\n'
                            f'  Δx = {flow_x_px:.2f} px\n'
                            f'  Δy = {flow_y_px:.2f} px\n\n'
                            f'Flow (normalized):\n'
                            f'  Δx = {patch_flow_x[sample_i, sample_j].item():.3f}\n'
                            f'  Δy = {patch_flow_y[sample_i, sample_j].item():.3f}',
                            ha='center', va='center', fontsize=10, fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax_flow.axis('off')
                
                # Empty subplot for aesthetics
                if idx < num_samples - 1:
                    ax_empty = fig.add_subplot(gs[row, 3])
                    ax_empty.axis('off')
            
            # Statistics in last row, last column
            ax_stats = fig.add_subplot(gs[num_samples, 3])
            num_with_obj = patch_has_object.sum().item()
            num_visible = patch_is_visible.sum().item()
            num_occluded = (patch_has_object & ~patch_is_visible).sum().item()
            
            ax_stats.text(0.5, 0.5,
                         f'Patch Statistics:\n\n'
                         f'Total: {H_p}×{H_p} = {H_p*H_p}\n'
                         f'With object: {num_with_obj}\n'
                         f'Visible: {num_visible}\n'
                         f'Occluded: {num_occluded}\n\n'
                         f'Visibility:\n'
                         f'{num_visible}/{num_with_obj}\n'
                         f'= {100*num_visible/max(num_with_obj,1):.1f}%',
                         ha='center', va='center', fontsize=10, fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            ax_stats.axis('off')
        else:
            # No visible patches
            for row in range(1, 4):
                for col in range(4):
                    ax = fig.add_subplot(gs[row, col])
                    if row == 1 and col == 0:
                        ax.text(0.5, 0.5, 'NO VISIBLE PATCHES', ha='center', va='center',
                               fontsize=14, color='red', fontweight='bold')
                    ax.axis('off')
        
        plt.suptitle(f'[TRAIN] Object {obj_label}: Patch-Level Correspondences\n'
                    f'Patch size: {ps}×{ps} pixels, Grid: {H_p}×{H_p} patches',
                    fontsize=14, fontweight='bold')
        
        # Save with unique identifier
        obj_label = query_data.infos.label[batch_idx]
        save_path = Path(f"tmp/vpog_dataset_test/train_patches_obj{obj_label}_b{batch_idx}_t{template_idx}.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  [TRAIN VIZ] ✓ Saved patch visualization to {save_path}")
    
    # def compute_flow_labels_for_train(
    #     self,
    #     query_data: tc.PandasTensorCollection,
    #     template_data: tc.PandasTensorCollection,
    # ) -> Dict[str, torch.Tensor]:
    #     """
    #     Compute flow labels between query and templates.
    #     Uses CROPPED/CENTERED 224x224 depth/K for training (matches model input resolution).
        
    #     This is the TRAINING version - operates on cropped data.
    #     """
    #     device = query_data.K.device
    #     B = len(query_data)
    #     S = self.num_templates
    #     H, W = self.image_size, self.image_size  # 224x224
    #     ps = self.patch_size
        
    #     with torch.inference_mode():
    #         # Initialize outputs [B, S, H_p, W_p, ...]
    #         H_p = self.num_patches_per_side
    #         flows = torch.zeros(B, S, H_p, H_p, 2, device=device)
    #         patch_vis = torch.zeros(B, S, H_p, H_p, dtype=torch.bool, device=device)
            
    #         for b in range(B):
    #             # Use CROPPED centered data (224x224)
    #             q_K = query_data.K[b]  # [3, 3] - cropped K
    #             q_depth = query_data.depth[b]  # [224, 224] - cropped depth
    #             q_mask = query_data.mask[b]  # [224, 224] - cropped mask
    #             q_pose = query_data.pose[b]  # [4, 4]
                
    #             # Unproject query depth to 3D point cloud in query camera frame
    #             points_q_cam = self.unproject_query_depth(
    #                 depth=q_depth,
    #                 K=q_K,
    #                 mask=q_mask,
    #             )  # [224, 224, 3] in query camera frame (BOP scale, mm)
                
    #             # Valid query pixels
    #             valid_q = (q_depth > 0) & (q_mask > 0.5)
                
    #             for s in range(S):
    #                 # Template data (CROPPED)
    #                 t_K = template_data.K[b, s]  # [3, 3] - cropped K
    #                 t_pose = template_data.pose[b, s]  # [4, 4]
    #                 t_depth = template_data.depth[b, s]  # [224, 224] - cropped depth
                    
    #                 # Transform query point cloud to template camera frame
    #                 points_t_cam = self.transform_query_to_template_space(
    #                     query_points_cam=points_q_cam,
    #                     query_pose=q_pose,
    #                     template_pose=t_pose,
    #                 )  # [224, 224, 3] in template camera frame (template scale)
                    
    #                 # Project to template image
    #                 u_t, v_t, z_t, valid_proj = self.project_to_template_image(
    #                     points_template_cam=points_t_cam,
    #                     K_template=t_K,
    #                     H=H,
    #                     W=W,
    #                 )
                    
    #                 # Combine with query validity
    #                 valid_proj = valid_proj & valid_q
                    
    #                 # Check occlusion
    #                 u_int = u_t.long().clamp(0, W-1)
    #                 v_int = v_t.long().clamp(0, H-1)
    #                 z_template_gt = t_depth[v_int, u_int]
    #                 not_occluded = (z_t <= z_template_gt + self.flow_computer.depth_tolerance)

    #                 # Final visibility
    #                 visible = valid_proj & not_occluded
                    
    #                 num_visible = visible.sum().item()
    #                 logger.info(f"  [TRAIN] Sample {b}, Template {s}: {num_visible} pixels visible")
                    
    #                 # ========== COMPUTE VISIBILITY MAPS ==========
    #                 # Valid masks (only consider pixels inside object masks)
    #                 query_valid_mask = (q_depth > 0) & (q_mask > 0.5)  # Query object pixels
    #                 template_valid_mask = (t_depth > 0)  # Template object pixels
                    
    #                 # 1. Query Seen Map: which query pixels (within mask) are visible in template
    #                 #    Only meaningful for query pixels within the query mask
    #                 query_seen_map = torch.zeros(H, W, dtype=torch.bool, device=device)
    #                 query_seen_map[query_valid_mask] = visible[query_valid_mask]  # True if query pixel is seen in template
                    
    #                 # 2. Template Seen Map: which template pixels (within mask) have query projections
    #                 #    Build by marking all template pixels where query projects
    #                 template_seen_map = torch.zeros(H, W, dtype=torch.bool, device=device)  # [H, W]
    #                 if visible.any():
    #                     # Mark template pixels where visible query pixels project
    #                     visible_u = u_int[visible]
    #                     visible_v = v_int[visible]
    #                     template_seen_map[visible_v, visible_u] = True
                    
    #                 # Template pixels NOT seen: in template mask but not in template_seen_map
    #                 template_not_seen_map = template_valid_mask & (~template_seen_map)
                    
    #                 # Query pixels NOT seen: in query mask but not visible in template
    #                 query_not_seen_map = query_valid_mask & (~query_seen_map)
                    
    #                 # Log statistics (only count within masks)
    #                 num_query_pixels = query_valid_mask.sum().item()
    #                 num_query_seen = query_seen_map.sum().item()
    #                 num_query_not_seen = query_not_seen_map.sum().item()
                    
    #                 num_template_pixels = template_valid_mask.sum().item()
    #                 num_template_seen = template_seen_map.sum().item()
    #                 num_template_not_seen = template_not_seen_map.sum().item()
                    
    #                 logger.info(f"  [TRAIN] Visibility statistics:")
    #                 logger.info(f"    Query: {num_query_seen}/{num_query_pixels} pixels visible in template ({num_query_not_seen} occluded)")
    #                 logger.info(f"    Template: {num_template_seen}/{num_template_pixels} pixels have query projection ({num_template_not_seen} NOT in query)")
                    
    #                 # ========== COMPUTE PATCH-LEVEL CORRESPONDENCES ==========
    #                 H_p = self.num_patches_per_side  # 14 patches per side for 224x224 with patch_size=16
    #                 ps = self.patch_size  # 16
                    
    #                 # For each query patch, compute correspondence to template patch
    #                 # Patch indices: i_p, j_p in [0, H_p) for patch grid
    #                 patch_i, patch_j = torch.meshgrid(
    #                     torch.arange(H_p, device=device),
    #                     torch.arange(H_p, device=device),
    #                     indexing='ij'
    #                 )  # [H_p, H_p]
                    
    #                 # Patch centers in pixel coordinates
    #                 patch_center_y = patch_i * ps + ps // 2  # [H_p, H_p]
    #                 patch_center_x = patch_j * ps + ps // 2  # [H_p, H_p]
                    
    #                 # For each patch, determine if CENTER pixel belongs to object mask
    #                 patch_has_object = torch.zeros(H_p, H_p, dtype=torch.bool, device=device)
    #                 patch_is_visible = torch.zeros(H_p, H_p, dtype=torch.bool, device=device)
    #                 patch_flow_x = torch.zeros(H_p, H_p, device=device)
    #                 patch_flow_y = torch.zeros(H_p, H_p, device=device)
    #                 patch_buddy_i = torch.zeros(H_p, H_p, dtype=torch.long, device=device)
    #                 patch_buddy_j = torch.zeros(H_p, H_p, dtype=torch.long, device=device)
    #                 # Store center of mass projections for visualization
    #                 patch_center_mass_u = torch.zeros(H_p, H_p, device=device)
    #                 patch_center_mass_v = torch.zeros(H_p, H_p, device=device)
                    
    #                 # Process all patches at once (vectorized)
    #                 for i_p in range(H_p):
    #                     for j_p in range(H_p):
    #                         # Pixel range for this patch
    #                         y_start, y_end = i_p * ps, (i_p + 1) * ps
    #                         x_start, x_end = j_p * ps, (j_p + 1) * ps
                            
    #                         # Check if CENTER pixel of patch belongs to object mask
    #                         center_y = i_p * ps + ps // 2
    #                         center_x = j_p * ps + ps // 2
                            
    #                         # Only process patch if its CENTER belongs to object mask
    #                         if query_valid_mask[center_y, center_x]:
    #                             patch_has_object[i_p, j_p] = True
                                
    #                             # Extract patch regions for visibility check
    #                             patch_visible = query_seen_map[y_start:y_end, x_start:x_end]
                                
    #                             # Check if any pixels are visible in template
    #                             if patch_visible.any():
    #                                 patch_is_visible[i_p, j_p] = True
                                    
    #                                 # Get template projections for visible pixels in this patch
    #                                 patch_u = u_int[y_start:y_end, x_start:x_end]  # [ps, ps]
    #                                 patch_v = v_int[y_start:y_end, x_start:x_end]  # [ps, ps]
                                    
    #                                 # Only consider visible pixels
    #                                 visible_u = patch_u[patch_visible].float()
    #                                 visible_v = patch_v[patch_visible].float()
                                    
    #                                 # Compute center of mass of template projections
    #                                 center_mass_u = visible_u.mean()
    #                                 center_mass_v = visible_v.mean()
                                    
    #                                 # Store for visualization
    #                                 patch_center_mass_u[i_p, j_p] = center_mass_u
    #                                 patch_center_mass_v[i_p, j_p] = center_mass_v
                                    
    #                                 # Find template patch whose center is closest to center of mass
    #                                 # Template patch centers
    #                                 template_patch_centers_x = torch.arange(H_p, device=device) * ps + ps // 2  # [H_p]
    #                                 template_patch_centers_y = torch.arange(H_p, device=device) * ps + ps // 2  # [H_p]
                                    
    #                                 # Distance from center of mass to each template patch center
    #                                 dist_x = template_patch_centers_x - center_mass_u  # [H_p]
    #                                 dist_y = template_patch_centers_y - center_mass_v  # [H_p]
                                    
    #                                 # Grid of distances [H_p, H_p]
    #                                 dist_grid = dist_x[None, :] ** 2 + dist_y[:, None] ** 2
                                    
    #                                 # Find closest template patch
    #                                 min_idx = dist_grid.argmin()
    #                                 buddy_i = min_idx // H_p
    #                                 buddy_j = min_idx % H_p
                                    
    #                                 patch_buddy_i[i_p, j_p] = buddy_i
    #                                 patch_buddy_j[i_p, j_p] = buddy_j
                                    
    #                                 # Compute flow: from query patch center to template patch center
    #                                 query_center_x = j_p * ps + ps // 2
    #                                 query_center_y = i_p * ps + ps // 2
    #                                 template_buddy_center_x = buddy_j * ps + ps // 2
    #                                 template_buddy_center_y = buddy_i * ps + ps // 2
                                    
    #                                 # Flow in pixels
    #                                 flow_x_px = template_buddy_center_x - query_center_x
    #                                 flow_y_px = template_buddy_center_y - query_center_y
                                    
    #                                 # Normalized by patch size
    #                                 patch_flow_x[i_p, j_p] = flow_x_px / ps
    #                                 patch_flow_y[i_p, j_p] = flow_y_px / ps
                    
    #                 num_patches_with_object = patch_has_object.sum().item()
    #                 num_patches_visible = patch_is_visible.sum().item()
                    
    #                 logger.info(f"  [TRAIN] Patch-level statistics:")
    #                 logger.info(f"    Patches with object: {num_patches_with_object}/{H_p*H_p}")
    #                 logger.info(f"    Patches visible in template: {num_patches_visible}/{num_patches_with_object}")
    #                 if num_patches_visible > 0:
    #                     logger.info(f"    Flow range: x=[{patch_flow_x[patch_is_visible].min():.2f}, {patch_flow_x[patch_is_visible].max():.2f}] patches")
    #                     logger.info(f"                y=[{patch_flow_y[patch_is_visible].min():.2f}, {patch_flow_y[patch_is_visible].max():.2f}] patches")
                    
    #                 # ========== VISUALIZATION (first batch, all templates) ==========
    #                 if self.debug_mode:
    #                     # Extract correspondences
    #                     valid_corr_mask = visible.flatten()
                        
    #                     if valid_corr_mask.any():
    #                         # Query pixel grid
    #                         query_y, query_x = torch.meshgrid(
    #                             torch.arange(H, device=device),
    #                             torch.arange(W, device=device),
    #                             indexing='ij'
    #                         )
                            
    #                         # Extract visible correspondences
    #                         q_x_corr = query_x.flatten()[valid_corr_mask]  # [N_corr]
    #                         q_y_corr = query_y.flatten()[valid_corr_mask]  # [N_corr]
    #                         q_z_corr = q_depth.flatten()[valid_corr_mask]  # [N_corr] in mm (BOP scale)
                            
    #                         t_u_corr = u_int.flatten()[valid_corr_mask]  # [N_corr]
    #                         t_v_corr = v_int.flatten()[valid_corr_mask]  # [N_corr]
    #                         t_z_corr = z_t.flatten()[valid_corr_mask]  # [N_corr] in template units
                            
    #                         # Correspondence summary
    #                         num_corr = len(q_x_corr)
    #                         logger.info(f"  [TRAIN] ✓ Extracted {num_corr} correspondences")
    #                         logger.info(f"    Query depths: [{q_z_corr.min():.1f}, {q_z_corr.max():.1f}] mm")
    #                         logger.info(f"    Template depths: [{t_z_corr.min():.1f}, {t_z_corr.max():.1f}] template units")
                            
    #                         correspondences = {
    #                             'query_x': q_x_corr,
    #                             'query_y': q_y_corr,
    #                             'query_z': q_z_corr,
    #                             'template_u': t_u_corr,
    #                             'template_v': t_v_corr,
    #                             'template_z': t_z_corr,
    #                             'count': num_corr,
    #                         }
    #                     else:
    #                         logger.warning(f"  [TRAIN] ✗ No valid correspondences found!")
    #                         correspondences = None
                        
    #                     # Visualize with mesh validation AND visibility maps AND patch correspondences
    #                     self._visualize_train_correspondences(
    #                         query_data=query_data,
    #                         template_data=template_data,
    #                         points_q_cam=points_q_cam,
    #                         correspondences=correspondences,
    #                         q_pose=q_pose,
    #                         t_pose=t_pose,
    #                         q_K=q_K,
    #                         t_K=t_K,
    #                         q_depth=q_depth,
    #                         t_depth=t_depth,
    #                         q_mask=q_mask,
    #                         query_seen_map=query_seen_map,
    #                         template_seen_map=template_seen_map,
    #                         template_not_seen_map=template_not_seen_map,
    #                         patch_has_object=patch_has_object,
    #                         patch_is_visible=patch_is_visible,
    #                         patch_flow_x=patch_flow_x,
    #                         patch_flow_y=patch_flow_y,
    #                         patch_buddy_i=patch_buddy_i,
    #                         patch_buddy_j=patch_buddy_j,
    #                         patch_center_mass_u=patch_center_mass_u,
    #                         patch_center_mass_v=patch_center_mass_v,
    #                         batch_idx=b,
    #                         template_idx=s,
    #                     )
                    
    #                 # For now: simple patch visibility
    #                 if num_visible > 0:
    #                     patch_vis[b, s, :, :] = True
        
    #     return {
    #         'flows': flows,
    #         'visibility': patch_vis,
    #         'patch_visibility': patch_vis,
    #     }
    def compute_flow_labels_for_train(
        self,
        query_data: tc.PandasTensorCollection,
        template_data: tc.PandasTensorCollection,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training labels between query and templates using CROPPED 224x224 depth/K.

        Outputs:
            - flows        : [B, S, H_p, W_p, 2]  (coarse query→template flow, patch centers)
            - visibility   : [B, S, H_p, W_p]     (patch is visible in template)
            - patch_visibility: same as visibility
            - patch_cls    : [B, S, H_p, W_p]     (coarse buddy class per patch)
            - dense_flow   : [B, S, H_p, W_p, ps, ps, 2]   (dense flow inside patch)
            - dense_weight : [B, S, H_p, W_p, ps, ps]      (weight / visibility per sub-pixel)
        """
        device = query_data.K.device
        B = len(query_data)
        S = self.num_templates
        H, W = self.image_size, self.image_size  # 224x224
        ps = self.patch_size                     # 16
        H_p = self.num_patches_per_side          # 14
        
        unseen_class = H_p * H_p  # class id for "object patch but unseen"
        
        with torch.inference_mode():
            # Coarse flows & visibility
            flows = torch.zeros(B, S, H_p, H_p, 2, device=device)
            patch_vis = torch.zeros(B, S, H_p, H_p, dtype=torch.bool, device=device)
            patch_cls = torch.full(
                (B, S, H_p, H_p), unseen_class,
                dtype=torch.long, device=device
            )
            
            # Dense (per sub-pixel) flow & weights
            dense_flow = torch.zeros(B, S, H_p, H_p, ps, ps, 2, device=device)
            dense_weight = torch.zeros(B, S, H_p, H_p, ps, ps, device=device)
            
            # Precompute query pixel grid and patch indices for 224x224
            query_y, query_x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )  # [H, W]
            
            # Patch index per pixel
            pix_patch_i = query_y // ps  # [H, W] in [0, H_p)
            pix_patch_j = query_x // ps  # [H, W] in [0, H_p)
            pix_patch_idx_flat = (pix_patch_i * H_p + pix_patch_j).view(-1)  # [H*W]
            
            # Patch grid & centers (same for all b,s)
            patch_i, patch_j = torch.meshgrid(
                torch.arange(H_p, device=device),
                torch.arange(H_p, device=device),
                indexing='ij'
            )  # [H_p, H_p]
            patch_center_y = patch_i * ps + ps // 2  # [H_p, H_p]
            patch_center_x = patch_j * ps + ps // 2  # [H_p, H_p]
            
            # Per-pixel patch centers (for dense flow)
            patch_center_y_px = patch_center_y[pix_patch_i, pix_patch_j]  # [H, W]
            patch_center_x_px = patch_center_x[pix_patch_i, pix_patch_j]  # [H, W]
            
            for b in range(B):
                # CROPPED query data
                q_K = query_data.K[b]         # [3, 3]
                q_depth = query_data.depth[b] # [224, 224]
                q_mask = query_data.mask[b]   # [224, 224]
                q_pose = query_data.pose[b]   # [4, 4]
                
                # Query object pixels
                query_valid_mask = (q_depth > 0) & (q_mask > 0.5)  # [H, W]
                
                # Unproject query depth to 3D in query camera frame
                points_q_cam = self.unproject_query_depth(
                    depth=q_depth,
                    K=q_K,
                    mask=q_mask,
                )  # [H, W, 3] in query camera frame (BOP scale, mm)
                
                for s in range(S):
                    # CROPPED template data
                    t_K = template_data.K[b, s]         # [3, 3]
                    t_pose = template_data.pose[b, s]   # [4, 4]
                    t_depth = template_data.depth[b, s] # [224, 224]
                    template_valid_mask = (t_depth > 0) # [H, W]
                    
                    # ========== EFFICIENT: Transform only query PATCH CENTERS to template ==========
                    # Extract center pixels of each query patch
                    patch_center_depths = q_depth[patch_center_y, patch_center_x]  # [H_p, H_p]
                    patch_center_valid = query_valid_mask[patch_center_y, patch_center_x]  # [H_p, H_p]
                    
                    # Unproject patch centers only
                    fx, fy = q_K[0, 0], q_K[1, 1]
                    cx, cy = q_K[0, 2], q_K[1, 2]
                    
                    # [H_p, H_p] pixel coordinates
                    x_centers = patch_center_x.float()
                    y_centers = patch_center_y.float()
                    z_centers = patch_center_depths.float()
                    
                    # Unproject to 3D in query camera frame
                    X_centers = (x_centers - cx) * z_centers / fx
                    Y_centers = (y_centers - cy) * z_centers / fy
                    Z_centers = z_centers
                    
                    points_q_centers = torch.stack([X_centers, Y_centers, Z_centers], dim=-1)  # [H_p, H_p, 3]
                    
                    # Transform query patch centers to template camera frame
                    points_t_centers = self.transform_query_to_template_space(
                        query_points_cam=points_q_centers,
                        query_pose=q_pose,
                        template_pose=t_pose,
                    )  # [H_p, H_p, 3] in template camera frame
                    
                    # Project to template image
                    u_t_centers, v_t_centers, z_t_centers, valid_proj_centers = self.project_to_template_image(
                        points_template_cam=points_t_centers,
                        K_template=t_K,
                        H=H,
                        W=W,
                    )  # [H_p, H_p]
                    
                    # Combine with query validity
                    valid_proj_centers = valid_proj_centers & patch_center_valid
                    
                    # Occlusion test for patch centers
                    u_int_centers = u_t_centers.long().clamp(0, W - 1)
                    v_int_centers = v_t_centers.long().clamp(0, H - 1)
                    z_template_gt_centers = t_depth[v_int_centers, u_int_centers]
                    not_occluded_centers = (z_t_centers <= z_template_gt_centers + self.flow_computer.depth_tolerance)
                    
                    # Final visibility for patch centers
                    visible_centers = valid_proj_centers & not_occluded_centers  # [H_p, H_p]
                    
                    # For debug/vis: optionally compute full query→template
                    if self.debug_mode:
                        # Transform all query pixels to template
                        points_t_cam = self.transform_query_to_template_space(
                            query_points_cam=points_q_cam,
                            query_pose=q_pose,
                            template_pose=t_pose,
                        )  # [H, W, 3] in template camera frame
                        
                        # Project to template image
                        u_t, v_t, z_t, valid_proj = self.project_to_template_image(
                            points_template_cam=points_t_cam,
                            K_template=t_K,
                            H=H,
                            W=W,
                        )
                        
                        # Combine with query validity
                        valid_proj = valid_proj & query_valid_mask
                        
                        # Occlusion test
                        u_int = u_t.long().clamp(0, W - 1)
                        v_int = v_t.long().clamp(0, H - 1)
                        z_template_gt = t_depth[v_int, u_int]
                        not_occluded = (z_t <= z_template_gt + self.flow_computer.depth_tolerance)
                        
                        # Final visibility at pixel level
                        visible = valid_proj & not_occluded  # [H, W]
                        num_visible = int(visible.sum().item())
                        logger.info(f"  [TRAIN] Sample {b}, Template {s}: {num_visible} pixels visible (full)")
                    
                    if num_visible == 0:
                        # Nothing visible: keep zeros / unseen_class
                        continue
                    
                    # ========== VISIBILITY MAPS (only for debug) ==========
                    if self.debug_mode:
                        # Query seen map: only for query object pixels
                        query_seen_map = torch.zeros(H, W, dtype=torch.bool, device=device)
                        query_seen_map[query_valid_mask] = visible[query_valid_mask]
                        
                        # Template seen map: which template pixels get any query projection
                        template_seen_map = torch.zeros(H, W, dtype=torch.bool, device=device)
                        visible_u = u_int[visible]
                        visible_v = v_int[visible]
                        template_seen_map[visible_v, visible_u] = True
                        template_not_seen_map = template_valid_mask & (~template_seen_map)
                        query_not_seen_map = query_valid_mask & (~query_seen_map)
                        
                        num_query_pixels = int(query_valid_mask.sum().item())
                        num_query_seen = int(query_seen_map.sum().item())
                        num_query_not_seen = int(query_not_seen_map.sum().item())
                        
                        num_template_pixels = int(template_valid_mask.sum().item())
                        num_template_seen = int(template_seen_map.sum().item())
                        num_template_not_seen = int(template_not_seen_map.sum().item())
                        
                        logger.info(f"  [TRAIN] Visibility statistics:")
                        logger.info(f"    Query: {num_query_seen}/{num_query_pixels} pixels visible in template "
                                    f"({num_query_not_seen} occluded)")
                        logger.info(f"    Template: {num_template_seen}/{num_template_pixels} pixels have query projection "
                                    f"({num_template_not_seen} NOT in query)")
                    
                    # ========== PATCH-LEVEL: which patches have object & are visible ==========
                    # 1) patch_has_object: CENTER pixel in object mask
                    patch_has_object = patch_center_valid  # [H_p, H_p]
                    
                    # 2) patch_is_visible: center is visible in template
                    patch_is_visible = visible_centers & patch_has_object  # [H_p, H_p]
                    
                    num_patches_with_object = int(patch_has_object.sum().item())
                    num_patches_visible = int(patch_is_visible.sum().item())
                    logger.info(f"  [TRAIN] Patch-level statistics:")
                    logger.info(f"    Patches with object center: {num_patches_with_object}/{H_p*H_p}")
                    logger.info(f"    Patches visible in template: {num_patches_visible}/{num_patches_with_object or 1}")
                    
                    if num_patches_visible == 0:
                        # No visible patches: mark background / unseen
                        patch_cls[b, s][~patch_has_object] = -1  # background
                        patch_vis[b, s] = False
                        continue
                    
                    # ========== COARSE QUERY→TEMPLATE: Find buddy patch by closest center (VECTORIZED) ==========
                    # For each visible query patch center, find closest template patch center
                    # Template patch centers in pixels [H_p, H_p]
                    t_patch_centers_x = (torch.arange(H_p, device=device) * ps + ps // 2).view(1, H_p).expand(H_p, H_p)  # [H_p, H_p]
                    t_patch_centers_y = (torch.arange(H_p, device=device) * ps + ps // 2).view(H_p, 1).expand(H_p, H_p)  # [H_p, H_p]
                    
                    # Compute distance from each query projection to ALL template patch centers
                    # u_t_centers, v_t_centers are [H_p, H_p] - query patch centers projected to template
                    # We need to broadcast and compute distance to all [H_p, H_p] template patch centers
                    
                    # Reshape for broadcasting: [H_p, H_p, 1, 1] vs [1, 1, H_p, H_p]
                    u_proj = u_t_centers[:, :, None, None]  # [H_p, H_p, 1, 1] - query projections
                    v_proj = v_t_centers[:, :, None, None]  # [H_p, H_p, 1, 1]
                    t_centers_x = t_patch_centers_x[None, None, :, :]  # [1, 1, H_p, H_p] - template centers
                    t_centers_y = t_patch_centers_y[None, None, :, :]  # [1, 1, H_p, H_p]
                    
                    # Distance: [H_p, H_p, H_p, H_p] where [i_q, j_q, i_t, j_t] is distance from query patch [i_q,j_q] to template patch [i_t,j_t]
                    dist_sq = (u_proj - t_centers_x) ** 2 + (v_proj - t_centers_y) ** 2  # [H_p, H_p, H_p, H_p]
                    
                    # Find closest template patch for each query patch
                    dist_sq_flat = dist_sq.view(H_p, H_p, -1)  # [H_p, H_p, H_p*H_p]
                    min_idx = dist_sq_flat.argmin(dim=2)  # [H_p, H_p]
                    
                    buddy_i = min_idx // H_p  # [H_p, H_p]
                    buddy_j = min_idx % H_p   # [H_p, H_p]
                    
                    # Template buddy centers in pixels
                    template_buddy_center_x = buddy_j * ps + ps // 2  # [H_p, H_p]
                    template_buddy_center_y = buddy_i * ps + ps // 2  # [H_p, H_p]
                    
                    # Coarse flow: from query patch center → template buddy patch center
                    flow_x_px = template_buddy_center_x - patch_center_x  # [H_p, H_p]
                    flow_y_px = template_buddy_center_y - patch_center_y  # [H_p, H_p]
                    flow_x_norm = flow_x_px.float() / ps
                    flow_y_norm = flow_y_px.float() / ps
                    
                    # Store coarse flows only where patch_has_object
                    valid_patch = patch_has_object  # [H_p, H_p] bool

                    flows[b, s, :, :, 0][valid_patch] = flow_x_norm[valid_patch]
                    flows[b, s, :, :, 1][valid_patch] = flow_y_norm[valid_patch]
                    
                    # Patch visibility mask
                    patch_vis[b, s] = patch_is_visible
                    
                    # Classification label:
                    #   - flatten buddy index for visible patches
                    #   - unseen_class for object-but-unseen
                    #   - -1 for background (no object at center)
                    cls = torch.full((H_p, H_p), unseen_class, dtype=torch.long, device=device)
                    cls[patch_is_visible] = (buddy_i * H_p + buddy_j)[patch_is_visible]
                    cls[patch_has_object & (~patch_is_visible)] = unseen_class
                    cls[~patch_has_object] = -1  # background
                    patch_cls[b, s] = cls
                    
                    # ========== DENSE TEMPLATE→QUERY FLOW ==========
                    # Project ALL template pixels to query once (vectorized, GPU-friendly)
                    fx_t, fy_t = t_K[0, 0], t_K[1, 1]
                    cx_t, cy_t = t_K[0, 2], t_K[1, 2]
                    
                    # Create pixel grid for template
                    t_y_grid, t_x_grid = torch.meshgrid(
                        torch.arange(H, device=device),
                        torch.arange(W, device=device),
                        indexing='ij'
                    )  # [H, W]
                    
                    # Only process template pixels with valid depth (object pixels)
                    template_object_mask = t_depth > 0  # [H, W]
                    
                    # Unproject ALL template pixels to 3D
                    t_z = t_depth  # [H, W]
                    t_X = (t_x_grid.float() - cx_t) * t_z / fx_t
                    t_Y = (t_y_grid.float() - cy_t) * t_z / fy_t
                    t_Z = t_z
                    
                    points_t_cam = torch.stack([t_X, t_Y, t_Z], dim=-1)  # [H, W, 3]
                    
                    # Transform ALL template→query in one operation
                    points_q_from_t = self.transform_template_to_query_space(
                        template_points_cam=points_t_cam,
                        template_pose=t_pose,
                        query_pose=q_pose,
                    )  # [H, W, 3] in query camera frame
                    
                    # Project ALL to query image
                    q_X, q_Y, q_Z = points_q_from_t[..., 0], points_q_from_t[..., 1], points_q_from_t[..., 2]
                    q_u = fx * q_X / q_Z + cx
                    q_v = fy * q_Y / q_Z + cy
                    
                    # Check if projection is valid (must be template object pixel)
                    valid_proj = (q_Z > 0) & (q_u >= 0) & (q_u < W) & (q_v >= 0) & (q_v < H) & template_object_mask
                    
                    # Occlusion check: is the projected point in front of query depth?
                    q_u_int = q_u.long().clamp(0, W - 1)
                    q_v_int = q_v.long().clamp(0, H - 1)
                    q_depth_at_proj = q_depth[q_v_int, q_u_int]
                    not_occluded = (q_Z <= q_depth_at_proj + self.flow_computer.depth_tolerance) | (q_depth_at_proj == 0)
                    
                    # Final visibility: valid projection AND not occluded AND lands on query mask
                    visible_t2q = valid_proj & not_occluded & query_valid_mask[q_v_int, q_u_int]
                    
                    # Now organize by query patches using buddy relationships (VECTORIZED)
                    # Reshape template to patch grid [H_p, H_p, ps, ps]
                    visible_patches = visible_t2q.view(H_p, ps, H_p, ps).permute(0, 2, 1, 3)  # [H_p, H_p, ps, ps]
                    q_u_patches = q_u.view(H_p, ps, H_p, ps).permute(0, 2, 1, 3)  # [H_p, H_p, ps, ps]
                    q_v_patches = q_v.view(H_p, ps, H_p, ps).permute(0, 2, 1, 3)  # [H_p, H_p, ps, ps]
                    
                    # For each query patch [i_q, j_q], we want flow from its buddy template patch [i_t, j_t]
                    # Create mapping using advanced indexing
                    dense_flow_output = torch.zeros(H_p, H_p, ps, ps, 2, device=device)
                    dense_weight_output = torch.zeros(H_p, H_p, ps, ps, device=device)
                    
                    # Get all query patch centers [H_p, H_p, 2]
                    q_patch_centers_j = (torch.arange(H_p, device=device) * ps + ps // 2).view(1, H_p).expand(H_p, H_p)
                    q_patch_centers_i = (torch.arange(H_p, device=device) * ps + ps // 2).view(H_p, 1).expand(H_p, H_p)
                    
                    # For visible query patches, extract buddy template patch data
                    # Use buddy_i, buddy_j to index into template patches
                    # This is done via advanced indexing (vectorized)
                    dense_weight_output = visible_patches[buddy_i, buddy_j]  # [H_p, H_p, ps, ps]
                    
                    # Compute flow for all pixels
                    # Get query projections for buddy template patches
                    q_u_buddy = q_u_patches[buddy_i, buddy_j]  # [H_p, H_p, ps, ps]
                    q_v_buddy = q_v_patches[buddy_i, buddy_j]  # [H_p, H_p, ps, ps]
                    
                    # Broadcast query patch centers for flow computation
                    q_centers_u = q_patch_centers_j[:, :, None, None]  # [H_p, H_p, 1, 1]
                    q_centers_v = q_patch_centers_i[:, :, None, None]  # [H_p, H_p, 1, 1]
                    
                    # Compute flow: (projection - query_patch_center) / ps
                    flow_u = (q_u_buddy - q_centers_u) / ps  # [H_p, H_p, ps, ps]
                    flow_v = (q_v_buddy - q_centers_v) / ps  # [H_p, H_p, ps, ps]
                    
                    dense_flow_output[:, :, :, :, 0] = flow_u
                    dense_flow_output[:, :, :, :, 1] = flow_v
                    
                    # Only keep for query patches with object
                    # Apply mask to weight only (flow values don't matter where weight=0)
                    valid_patch_mask = patch_has_object.unsqueeze(-1).unsqueeze(-1)  # [H_p, H_p, 1, 1]
                    dense_weight_output = dense_weight_output * valid_patch_mask.float()
                    
                    dense_flow[b, s] = dense_flow_output
                    dense_weight[b, s] = dense_weight_output
                    
                    num_t2q_correspondences = int(visible_t2q.sum().item())
                    logger.info(f"  [TRAIN] Template→Query: {num_t2q_correspondences} correspondences")
                    
                    # ========== DEBUG VISUALIZATION ==========
                    if self.debug_mode:
                        # Full visibility + patch correspondence visualization
                        self._visualize_train_correspondences(
                            query_data=query_data,
                            template_data=template_data,
                            points_q_cam=points_q_cam,
                            correspondences=None,
                            q_pose=q_pose,
                            t_pose=t_pose,
                            q_K=q_K,
                            t_K=t_K,
                            q_depth=q_depth,
                            t_depth=t_depth,
                            q_mask=q_mask,
                            query_seen_map=query_seen_map,
                            template_seen_map=template_seen_map,
                            template_not_seen_map=template_not_seen_map,
                            patch_has_object=patch_has_object,
                            patch_is_visible=patch_is_visible,
                            patch_flow_x=flow_x_norm,
                            patch_flow_y=flow_y_norm,
                            patch_buddy_i=buddy_i,
                            patch_buddy_j=buddy_j,
                            patch_center_mass_u=u_t_centers,  # Use projected centers
                            patch_center_mass_v=v_t_centers,
                            batch_idx=b,
                            template_idx=s,
                        )
                        
                        # Visualize dense template→query flow
                        obj_label = query_data.infos.label[b]
                        self._visualize_dense_patch_flow(
                            query_rgb=query_data.centered_rgb[b],
                            template_rgb=template_data.rgb[b, s],
                            q_mask=q_mask,
                            t_depth=t_depth,
                            flow_grid=dense_flow[b, s],
                            weight_grid=dense_weight[b, s],
                            patch_has_object=patch_has_object,
                            patch_is_visible=patch_is_visible,
                            patch_buddy_i=buddy_i,
                            patch_buddy_j=buddy_j,
                            batch_idx=b,
                            template_idx=s,
                            obj_label=obj_label,
                        )
            
            return {
                'flows': flows,
                'visibility': patch_vis,
                'patch_visibility': patch_vis,
                'patch_cls': patch_cls,
                'dense_flow': dense_flow,
                'dense_weight': dense_weight,
            }

    def compute_flow_labels(
        self,
        query_data: tc.PandasTensorCollection,
        template_data: tc.PandasTensorCollection,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute flow labels between query and templates.
        Uses ORIGINAL full-resolution depth/K/mask for accurate visibility.
        """
        device = query_data.K.device
        B = len(query_data)
        S = self.num_templates
        H, W = self.image_size, self.image_size
        ps = self.patch_size
        
        with torch.inference_mode():
            # Initialize outputs [B, S, H_p, W_p, ...]
            H_p = self.num_patches_per_side
            flows = torch.zeros(B, S, H_p, H_p, 2, device=device)
            patch_vis = torch.zeros(B, S, H_p, H_p, dtype=torch.bool, device=device)
            
            for b in range(B):
                # Use ORIGINAL uncropped data
                q_K_orig = query_data.K_original[b]  # [3, 3]
                q_depth_orig = query_data.full_depth[b]  # [H_orig, W_orig]
                q_mask_orig = query_data.mask_original[b]  # [H_orig, W_orig]
                q_pose = query_data.pose[b]  # [4, 4]
                
                H_orig, W_orig = q_depth_orig.shape
                
                # Unproject query depth to 3D point cloud in query camera frame
                points_q_cam = self.unproject_query_depth(
                    depth=q_depth_orig,
                    K=q_K_orig,
                    mask=q_mask_orig,
                )  # [H_orig, W_orig, 3] in query camera frame (BOP scale, mm)
                # Valid query pixels
                valid_q = (q_depth_orig > 0) & (q_mask_orig > 0.5)
                
                for s in range(S):
                    # Template data
                    t_K_orig = template_data.K_original  # [3, 3]
                    t_pose = template_data.pose[b, s]  # [4, 4]
                    t_depth_orig = template_data.depth_original[b, s]  # [H_t, W_t]
                    H_t_orig, W_t_orig = t_depth_orig.shape
                    
                    # Transform query point cloud to template camera frame
                    points_t_cam = self.transform_query_to_template_space(
                        query_points_cam=points_q_cam,
                        query_pose=q_pose,
                        template_pose=t_pose,
                    )  # [H_orig, W_orig, 3] in template camera frame (template scale)
                    
                    # Project to template image
                    u_t, v_t, z_t, valid_proj = self.project_to_template_image(
                        points_template_cam=points_t_cam,
                        K_template=t_K_orig,
                        H=H_t_orig,
                        W=W_t_orig,
                    )
                    
                    # Combine with query validity
                    valid_proj = valid_proj & valid_q
                    
                    # Check occlusion
                    u_int = u_t.long().clamp(0, W_t_orig-1)
                    v_int = v_t.long().clamp(0, H_t_orig-1)
                    z_template_gt = t_depth_orig[v_int, u_int]
                    not_occluded = (z_t <= z_template_gt + self.flow_computer.depth_tolerance)
                    
                    # Final visibility
                    visible = valid_proj & not_occluded
                    
                    num_visible = visible.sum().item()
                    logger.info(f"  Sample {b}, Template {s}: {num_visible} pixels visible")
                    
                    # ========== COMPUTE CORRESPONDENCES (CORE LOGIC) ==========
                    # For first sample/template, compute and visualize correspondences
                    if self.debug_mode:
                        # Extract visible correspondences: query pixel → template pixel
                        valid_corr_mask = visible.flatten()
                        
                        if valid_corr_mask.any():
                            # Query pixel grid
                            query_y, query_x = torch.meshgrid(
                                torch.arange(H_orig, device=device),
                                torch.arange(W_orig, device=device),
                                indexing='ij'
                            )
                            
                            # Extract visible correspondences
                            q_x_corr = query_x.flatten()[valid_corr_mask]  # [N_corr]
                            q_y_corr = query_y.flatten()[valid_corr_mask]  # [N_corr]
                            q_z_corr = q_depth_orig.flatten()[valid_corr_mask]  # [N_corr] in mm (BOP scale)
                            
                            t_u_corr = u_int.flatten()[valid_corr_mask]  # [N_corr]
                            t_v_corr = v_int.flatten()[valid_corr_mask]  # [N_corr]
                            t_z_corr = z_t.flatten()[valid_corr_mask]  # [N_corr] in template units
                            
                            # Correspondence summary
                            num_corr = len(q_x_corr)
                            logger.info(f"  ✓ Extracted {num_corr} correspondences")
                            logger.info(f"    Query depths: [{q_z_corr.min():.1f}, {q_z_corr.max():.1f}] mm")
                            logger.info(f"    Template depths: [{t_z_corr.min():.1f}, {t_z_corr.max():.1f}] template units")
                            
                            correspondences = {
                                'query_x': q_x_corr,
                                'query_y': q_y_corr,
                                'query_z': q_z_corr,
                                'template_u': t_u_corr,
                                'template_v': t_v_corr,
                                'template_z': t_z_corr,
                                'count': num_corr,
                            }
                        else:
                            logger.warning(f"  ✗ No valid correspondences found!")
                            correspondences = None
                        
                        # ========== VISUALIZATION (AFTER LOGIC) ==========
                        self._visualize_correspondences(
                            query_data=query_data,
                            template_data=template_data,
                            points_q_cam=points_q_cam,
                            correspondences=correspondences,
                            q_pose=q_pose,
                            t_pose=template_data.pose[b, s],
                            q_K_orig=q_K_orig,
                            t_K_orig=t_K_orig,
                            q_depth_orig=q_depth_orig,
                            t_depth_orig=t_depth_orig,
                            q_mask_orig=q_mask_orig,
                            batch_idx=b,
                            template_idx=s,
                        )

                    
                    # For now: simple patch visibility
                    if num_visible > 0:
                        patch_vis[b, s, :, :] = True
        
        return {
            'flows': flows,
            'visibility': patch_vis,
            'patch_visibility': patch_vis,
        }

    def transform_template_to_query_space(
        self,
        template_points_cam: torch.Tensor,
        template_pose: torch.Tensor,
        query_pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transform 3D points from template camera frame to query camera frame.

        Mirrors transform_query_to_template_space but in reverse:

        Template depth was rendered with a normalized×10 model at pose TWO_t,
        so points in template camera frame live in the same "template units".
        We do:
            template_cam -> world_scaled -> world (divide by 10) -> query_cam
        """
        original_shape = template_points_cam.shape
        points_flat = template_points_cam.reshape(-1, 3)  # [N, 3]

        # Step 1: template_cam -> scaled world frame
        T_t2w = torch.inverse(template_pose)  # inverse of TWO_t
        points_homo = torch.cat(
            [points_flat, torch.ones(len(points_flat), 1, device=points_flat.device)],
            dim=1,
        )
        points_world_scaled = (T_t2w @ points_homo.T).T[:, :3]  # [N, 3]

        # Step 2: undo the 10x scale (world_scaled -> BOP world)
        points_world = points_world_scaled / 10.0

        # Step 3: world -> query_cam
        points_homo_world = torch.cat(
            [points_world, torch.ones(len(points_world), 1, device=points_world.device)],
            dim=1,
        )
        points_query_cam = (query_pose @ points_homo_world.T).T[:, :3]

        return points_query_cam.reshape(original_shape)

    
    def collate_fn(
        self,
        batch: List[SceneObservation],
    ) -> Optional[VPOGBatch]:
        """
        Collate function for dataloader.
        
        Processes a batch of scene observations into VPOG format:
        1. Process query images
        2. Select and load templates
        3. Compute flow labels
        4. Combine into [B, S+1, ...] format
        
        Args:
            batch: List of SceneObservations
        
        Returns:
            VPOGBatch or None if error
        """
        try:
            # Apply RGB augmentation if enabled
            if self.rgb_augmentation and self.rgb_transform is not None:
                batch = [self.rgb_transform(data) for data in batch]
            
            # Convert to tensor collection
            batch = SceneObservation.collate_fn(batch)
            
            # Process query
            query_data = self.process_query(batch)
            
            # Process templates
            template_data, selection_info = self.process_templates(query_data)
            
            # Compute flow labels using TRAINING version (cropped 224x224)
            flow_labels = self.compute_flow_labels_for_train(query_data, template_data)
            
            batch_size = len(query_data)
            
            # Combine query and templates into [B, S+1, ...] format
            # Query is first, then S templates
            
            # Images: [B, S+1, 3, H, W]
            query_images = query_data.rgb.unsqueeze(1)  # [B, 1, 3, H, W]
            template_images = template_data.rgb  # [B, S, 3, H, W]
            images = torch.cat([query_images, template_images], dim=1)
            
            # Normalize images
            images = self.normalize(images.flatten(0, 1)).reshape(images.shape)
            
            # Masks: [B, S+1, H, W]
            query_masks = query_data.mask.unsqueeze(1)  # [B, 1, H, W]
            # Templates don't have separate masks in this simplified version
            # In full implementation, extract from template RGBA alpha channel
            template_masks = torch.ones_like(template_data.rgb[:, :, 0, :, :])  # [B, S, H, W]
            masks = torch.cat([query_masks, template_masks], dim=1)
            
            # Camera intrinsics: [B, S+1, 3, 3]
            query_K = query_data.K.unsqueeze(1)  # [B, 1, 3, 3]
            template_K = template_data.K  # [B, S, 3, 3]
            K = torch.cat([query_K, template_K], dim=1)
            
            # Poses: [B, S+1, 4, 4]
            query_poses = query_data.pose.unsqueeze(1)  # [B, 1, 4, 4]
            template_poses = template_data.pose  # [B, S, 4, 4]
            poses = torch.cat([query_poses, template_poses], dim=1)
            
            # # Create VPOGBatch
            # vpog_batch = VPOGBatch(
            #     images=images,
            #     masks=masks,
            #     K=K,
            #     poses=poses,
            #     d_ref=selection_info['d_ref'],
            #     template_indices=selection_info['template_indices'],
            #     template_types=selection_info['template_types'],
            #     flows=flow_labels['flows'],
            #     visibility=flow_labels['visibility'],
            #     patch_visibility=flow_labels['patch_visibility'],
            #     infos=query_data.infos,
            #     full_rgb=query_data.full_rgb,  # Original full images before cropping
            #     centered_rgb=query_data.centered_rgb,  # Center crop WITH background (model input)
            #     bboxes=query_data.bboxes,  # Bounding boxes used for cropping
            #     query_depth=query_data.depth,  # Keep in mm
            #     template_depth=template_data.depth,  # Keep in mm
            # )
            vpog_batch = VPOGBatch(
                images=images,
                masks=masks,
                K=K,
                poses=poses,
                d_ref=selection_info['d_ref'],
                template_indices=selection_info['template_indices'],
                template_types=selection_info['template_types'],
                flows=flow_labels['flows'],
                visibility=flow_labels['visibility'],
                patch_visibility=flow_labels['patch_visibility'],
                patch_cls=flow_labels['patch_cls'],
                dense_flow=flow_labels['dense_flow'],
                dense_weight=flow_labels['dense_weight'],
                infos=query_data.infos,
                full_rgb=query_data.full_rgb,          # Original full images
                centered_rgb=query_data.centered_rgb,  # Center crop WITH background
                bboxes=query_data.bboxes,              # Cropping boxes
                query_depth=query_data.depth,          # Cropped query depth
                template_depth=template_data.depth,    # Cropped template depth
            )
        
            return vpog_batch
        
        except Exception as e:
            logger.error(f"Error in collate_fn: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    """
    REAL TEST: Full dataset integration with GSO data
    Tests complete pipeline from WebSceneDataset through VPOGBatch creation
    """
    import sys
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    print("=" * 80)
    print("REAL TEST: VPOGTrainDataset with GSO Data")
    print("=" * 80)
    
    # Setup paths
    SEED = 2  # Fixed seed for reproducible test
    dataset_dir = project_root / "datasets" / "gso"
    templates_dir = project_root / "datasets" / "templates" / "gso"
    save_dir = project_root / "tmp" / "vpog_dataset_test"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if not templates_dir.exists():
        print(f"\n✗ Templates not found at {templates_dir}")
        print("Please run: python -m src.scripts.render_gso_templates")
        sys.exit(1)
    
    print(f"✓ Found templates at {templates_dir}")
    
    print(f"\n✓ Initializing VPOGTrainDataset...")
    try:
        template_config = {
            'dir': str(project_root / "datasets" / "templates"),
            'level_templates': 1,
            'pose_distribution': 'all',
            'scale_factor': 10.0,
            'num_templates': 162,
            'pose_name': 'object_poses/OBJECT_ID.npy',
        }
        
        dataset = VPOGTrainDataset(
            root_dir=str(project_root / "datasets"),
            dataset_name='gso',
            template_config=template_config,
            num_positive_templates=3,
            num_negative_templates=2,
            min_negative_angle_deg=90.0,
            d_ref_random_ratio=0.0,
            patch_size=16,
            image_size=224,
            flow_config={
                'compute_visibility': True,
                'compute_patch_visibility': True,
                'visibility_threshold': 0.1,
            },
            batch_size=2,
            depth_scale=10.0,
            seed=SEED,
            debug=True
        )
        print(f"✓ Dataset initialized successfully")
        print(f"  - TemplateDataset: {len(dataset.template_dataset)} templates")
        print(f"  - Num templates per query: {dataset.num_templates}")
    except Exception as e:
        print(f"✗ Failed to initialize dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create dataloader using the web_dataloader from dataset
    print(f"\n✓ Creating DataLoader...")
    from src.utils.dataloader import NoneFilteringDataLoader
    
    dataloader = NoneFilteringDataLoader(
        dataset.web_dataloader.datapipeline,
        batch_size=2,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    
    # Test batch loading
    print(f"\n✓ Loading first batch...")
    try:
        batch = next(iter(dataloader))
        
        if batch is None:
            print("✗ Batch is None!")
            sys.exit(1)
        
        print(f"✓ Batch loaded successfully!")
        print(f"\nBatch structure:")
        print(f"  images:           {batch.images.shape} (B, S+1, 3, H, W)")
        print(f"  masks:            {batch.masks.shape} (B, S+1, H, W)")
        print(f"  K:                {batch.K.shape} (B, S+1, 3, 3)")
        print(f"  poses:            {batch.poses.shape} (B, S+1, 4, 4)")
        print(f"  d_ref:            {batch.d_ref.shape} (B, 3)")
        print(f"  template_indices: {batch.template_indices.shape} (B, S)")
        print(f"  template_types:   {batch.template_types.shape} (B, S)")
        print(f"  flows:            {batch.flows.shape} (B, S, H_p, W_p, 2)")
        print(f"  visibility:       {batch.visibility.shape} (B, S, H_p, W_p)")
        print(f"  patch_visibility: {batch.patch_visibility.shape} (B, S, H_p, W_p)")
        
        # Verify data ranges
        print(f"\nData validation:")
        print(f"  ✓ Images range: [{batch.images.min():.3f}, {batch.images.max():.3f}]")
        print(f"  ✓ Masks range: [{batch.masks.min():.3f}, {batch.masks.max():.3f}]")
        print(f"  ✓ d_ref norm: {torch.norm(batch.d_ref, dim=1).tolist()}")
        
        # Count positive vs negative templates
        num_pos = (batch.template_types == 0).sum(dim=1)
        num_neg = (batch.template_types == 1).sum(dim=1)
        print(f"  ✓ Positive templates per sample: {num_pos.tolist()}")
        print(f"  ✓ Negative templates per sample: {num_neg.tolist()}")
        
        # Original full RGB info
        if batch.full_rgb is not None:
            print(f"\nOriginal full RGB (before cropping):")
            print(f"  ✓ Shape: {batch.full_rgb.shape}")
            print(f"  ✓ Range: [{batch.full_rgb.min():.3f}, {batch.full_rgb.max():.3f}]")
            if batch.bboxes is not None:
                print(f"  ✓ Bounding boxes (xyxy): {batch.bboxes.shape}")
                for i, bbox in enumerate(batch.bboxes[:2]):
                    print(f"    Sample {i}: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
        
        # Object info
        print(f"\nQuery objects:")
        try:
            if hasattr(batch.infos, 'infos'):
                for i in range(len(batch.infos.infos)):
                    label = batch.infos.infos.iloc[i]['label']
                    print(f"  Sample {i}: {label}")
            else:
                print(f"  Info format: {type(batch.infos)}")
        except Exception as e:
            print(f"  Could not extract labels: {e}")
        
    except Exception as e:
        print(f"✗ Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Visualize batch (simple version - full viz in test_vpog_labels.py)
    print(f"\n✓ Generating basic visualizations...")
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        def denorm_img(tensor):
            img = tensor.cpu().numpy()
            mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
            std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
            img = img * std + mean
            img = np.transpose(img, (1, 2, 0))
            return np.clip(img, 0, 1)
        
        def compute_pose_angle(pose1, pose2):
            """Compute SO(3) angular distance between two poses"""
            R1 = pose1[:3, :3].cpu().numpy()
            R2 = pose2[:3, :3].cpu().numpy()
            R_rel = R1.T @ R2
            trace = np.trace(R_rel)
            cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
            angle_deg = np.rad2deg(np.arccos(cos_angle))
            return angle_deg
        
        # Figure 1: Original full RGB + centered query + cropped query + templates with angles
        fig, axes = plt.subplots(2, 8, figsize=(24, 8))
        
        for b in range(min(2, batch.images.shape[0])):
            # Original full RGB with bbox
            if batch.full_rgb is not None:
                full_img = batch.full_rgb[b].permute(1, 2, 0).cpu().numpy()
                axes[b, 0].imshow(np.clip(full_img, 0, 1))
                axes[b, 0].set_title(f'Sample {b}:\nFull RGB', fontsize=9, fontweight='bold')
                
                # Draw bbox
                if batch.bboxes is not None:
                    bbox = batch.bboxes[b].cpu().numpy()
                    rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                   linewidth=2, edgecolor='red', facecolor='none')
                    axes[b, 0].add_patch(rect)
                axes[b, 0].axis('off')
            
            # Centered query (object-centered, before cropping)
            # Extract just the object region from full_rgb
            if batch.full_rgb is not None and batch.bboxes is not None:
                bbox = batch.bboxes[b].cpu().numpy().astype(int)
                centered_img = full_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                axes[b, 1].imshow(np.clip(centered_img, 0, 1))
                axes[b, 1].set_title(f'Centered\n(pre-crop)', fontsize=9, fontweight='bold')
                axes[b, 1].axis('off')
            
            # Cropped & normalized query (final input)
            query_img = denorm_img(batch.images[b, 0])
            axes[b, 2].imshow(query_img)
            axes[b, 2].set_title(f'Query\n(224x224)', fontsize=9, fontweight='bold', color='blue')
            axes[b, 2].axis('off')
            
            # Templates with angles
            query_pose = batch.poses[b, 0]
            # First positive template is t* (nearest OOP)
            t_star_pose = batch.poses[b, 1]  # T0 is always t*
            
            for s in range(min(5, batch.images.shape[1] - 1)):
                template_pose = batch.poses[b, s+1]
                angle_to_query = compute_pose_angle(query_pose, template_pose)
                angle_to_tstar = compute_pose_angle(t_star_pose, template_pose)
                template_idx = batch.template_indices[b, s].item()
                
                axes[b, s+3].imshow(denorm_img(batch.images[b, s+1]))
                is_pos = batch.template_types[b, s].item() == 0
                color = 'green' if is_pos else 'red'
                # Show angle to t* for POS templates, angle to query for NEG templates
                angle_display = angle_to_tstar if is_pos else angle_to_query
                axes[b, s+3].set_title(f'T{s} ({"POS" if is_pos else "NEG"}) idx={template_idx}\nto {"t*" if is_pos else "Q"}:{angle_display:.1f}°', 
                                      fontsize=8, color=color, fontweight='bold' if is_pos else 'normal')
                axes[b, s+3].axis('off')
        
        plt.suptitle('VPOG Batch: Full Pipeline Visualization\n' + 
                     'Check if positive templates (POS) have smallest angles to query!',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Save with seed in filename for traceability
        output_filename = f'batch_with_full_rgb_seed{SEED}.png'
        plt.savefig(save_dir / output_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved to {save_dir / output_filename}")
        
        # Print detailed angle information
        print(f"\n  Template angles (POS→t*, NEG→Query):")
        for b in range(min(2, batch.images.shape[0])):
            query_pose = batch.poses[b, 0]
            t_star_pose = batch.poses[b, 1]  # T0 is always t*
            print(f"    Sample {b}:")
            
            for s in range(min(5, batch.images.shape[1] - 1)):
                template_pose = batch.poses[b, s+1]
                angle_to_query = compute_pose_angle(query_pose, template_pose)
                angle_to_tstar = compute_pose_angle(t_star_pose, template_pose)
                template_idx = batch.template_indices[b, s].item()
                is_pos = batch.template_types[b, s].item() == 0
                type_str = "POS" if is_pos else "NEG"
                
                if is_pos:
                    print(f"      T{s} ({type_str}) idx={template_idx:3d}: to t*={angle_to_tstar:6.2f}° (to Q={angle_to_query:6.2f}°)")
                else:
                    print(f"      T{s} ({type_str}) idx={template_idx:3d}: to Q={angle_to_query:6.2f}°)")
        
        print(f"\n  Check that:")
        print(f"    1. Full RGB shows the complete scene")
        print(f"    2. Centered shows object region (red bbox)")
        print(f"    3. Query is the cropped & normalized version")
        print(f"    4. POS templates: T0 (t*) should have angle=0° to itself, T1/T2 should be <30° to t*")
        print(f"    5. NEG templates have LARGE angles to query (should be >90°)")
        print(f"    6. Same images appear each run (seed set)")
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    # # Label visualization
    # print(f"\n✓ Generating label visualizations...")
    # try:
    #     from training.dataloader.visualize_labels import LabelVisualizer
    #     visualizer = LabelVisualizer(device="cuda" if torch.cuda.is_available() else "cpu")
    #     label_save_path = save_dir / "labels.png"
    #     compute_time = visualizer.visualize_batch_labels(
    #         batch=batch,
    #         save_path=label_save_path,
    #         num_samples=min(2, batch.images.shape[0]),
    #         num_random_patches=8,
    #     )
    # except Exception as e:
    #     print(f"✗ Label visualization failed: {e}")
    #     import traceback
    #     traceback.print_exc()
    
    # # Test multiple batches
    # print(f"\n✓ Testing multiple batches...")
    # num_test_batches = 1
    # for i, batch in enumerate(dataloader):
    #     if i >= num_test_batches:
    #         break
    #     if batch is None:
    #         print(f"  Batch {i}: None (skipped)")
    #         continue
    #     print(f"  Batch {i}: {batch.images.shape[0]} samples")
    
    # print("\n" + "=" * 80)
    # print("✓ All VPOGTrainDataset tests passed!")
    # print(f"✓ Full pipeline works: WebSceneDataset → Template Selection → VPOGBatch")
    # print(f"✓ Visualizations saved to {save_dir}")
    # print("=" * 80)
