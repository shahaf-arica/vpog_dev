"""
Template Manager for VPOG Inference

Handles template loading, caching, and selection for 6D pose estimation.
Supports two modes:
- "all": Load all available templates (162 for level 1)
- "subset": Select S_p nearest + S_n negative templates

Author: VPOG Team
Date: December 2025
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import logging

# Import from training dataloader
from training.dataloader.template_selector import TemplateSelector, extract_d_ref_from_pose
from src.lib3d.template_transform import get_obj_poses_from_template_level

logger = logging.getLogger(__name__)


class TemplateManager:
    """
    Manages template loading, caching, and selection for VPOG inference.
    
    Features:
    - Two modes: "all" (load all 162 templates) or "subset" (select S_p + S_n)
    - Template caching for performance
    - Batch loading support
    - Integration with TemplateSelector
    """
    
    def __init__(
        self,
        templates_dir: Union[str, Path],
        dataset_name: str = "gso",
        mode: str = "all",
        level_templates: int = 1,
        pose_distribution: str = "all",
        num_positive: int = 4,
        num_negative: int = 2,
        min_negative_angle_deg: float = 90.0,
        cache_size: int = 10,
        device: str = "cpu",
    ):
        """
        Args:
            templates_dir: Root directory containing template images
                Expected structure: templates_dir/dataset_name/object_id/NNNNNN.png
            dataset_name: Name of dataset ("gso", "lmo", "ycbv", etc.)
            mode: "all" (load all templates) or "subset" (select S_p + S_n)
            level_templates: Icosphere level (1=162 views, 2=642 views)
            pose_distribution: "all" or "upper" hemisphere
            num_positive: S_p - number of nearest templates (subset mode)
            num_negative: S_n - number of negative templates (subset mode)
            min_negative_angle_deg: Minimum angle for negatives (subset mode)
            cache_size: Number of objects to cache in memory
            device: Device for tensors ("cpu" or "cuda")
        """
        self.templates_dir = Path(templates_dir)
        self.dataset_name = dataset_name
        self.mode = mode
        self.level_templates = level_templates
        self.pose_distribution = pose_distribution
        self.device = device
        self.cache_size = cache_size
        
        # Verify templates directory exists
        self.dataset_dir = self.templates_dir / dataset_name
        if not self.dataset_dir.exists():
            raise ValueError(f"Templates directory not found: {self.dataset_dir}")
        
        # Initialize template selector for subset mode
        if mode == "subset":
            self.selector = TemplateSelector(
                level_templates=level_templates,
                pose_distribution=pose_distribution,
                num_positive=num_positive,
                num_negative=num_negative,
                min_negative_angle_deg=min_negative_angle_deg,
            )
            self.num_templates = num_positive + num_negative
        else:
            self.selector = None
            # Get number of templates from pose distribution
            avail_index, _ = get_obj_poses_from_template_level(
                level_templates, pose_distribution, return_cam=False, return_index=True
            )
            self.num_templates = len(avail_index)
        
        # Cache: {object_id: template_data}
        self.cache: Dict[str, Dict] = {}
        self.cache_order: List[str] = []  # LRU tracking
        
        logger.info(f"TemplateManager initialized:")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Dataset: {dataset_name}")
        logger.info(f"  Templates per object: {self.num_templates}")
        logger.info(f"  Cache size: {cache_size}")
    
    def load_object_templates(
        self,
        object_id: str,
        query_pose: Optional[np.ndarray] = None,
        return_metadata: bool = True,
    ) -> Dict:
        """
        Load templates for a specific object.
        
        Args:
            object_id: Object identifier (e.g., "000733" for GSO object 733)
            query_pose: Query pose for subset mode (4x4 matrix, required for subset mode)
            return_metadata: Whether to return template indices and metadata
        
        Returns:
            Dictionary containing:
                - images: [N, H, W, 3] RGB images (uint8)
                - masks: [N, H, W] binary masks (uint8)
                - poses: [N, 4, 4] template poses
                - template_indices: [N] indices into full template set
                - d_ref: [3] reference direction (subset mode only)
                (if return_metadata=True)
        """
        # Check cache first
        if object_id in self.cache:
            cached_data = self.cache[object_id]
            
            # For "all" mode, return cached data directly
            if self.mode == "all":
                return self._move_to_device(cached_data) if return_metadata else {
                    "images": self._move_to_device(cached_data["images"]),
                    "masks": self._move_to_device(cached_data["masks"]),
                    "poses": self._move_to_device(cached_data["poses"]),
                }
            
            # For "subset" mode, need to reselect based on query_pose
            # (don't cache subset selection, only the full templates)
        
        # Load templates from disk
        object_dir = self.dataset_dir / object_id
        if not object_dir.exists():
            raise ValueError(f"Object directory not found: {object_dir}")
        
        # Load poses
        pose_file = self.templates_dir / self.dataset_name / "object_poses" / f"{object_id}.npy"
        if not pose_file.exists():
            raise ValueError(f"Pose file not found: {pose_file}")
        
        all_poses = np.load(pose_file)
        
        if self.mode == "all":
            # Load all available templates
            avail_index, template_poses = get_obj_poses_from_template_level(
                self.level_templates,
                self.pose_distribution,
                return_cam=False,
                return_index=True,
            )
            
            template_data = self._load_templates_by_indices(
                object_dir, avail_index, all_poses
            )
            template_data["template_indices"] = avail_index
            
            # Cache the data
            self._add_to_cache(object_id, template_data)
            
        else:  # subset mode
            if query_pose is None:
                raise ValueError("query_pose required for subset mode")
            
            # Select templates using TemplateSelector
            selection_result = self.selector.select_templates(
                query_pose, return_d_ref=True
            )
            
            positive_indices = selection_result["positive_indices"]
            negative_indices = selection_result["negative_indices"]
            selected_indices = positive_indices + negative_indices
            
            # Load selected templates
            template_data = self._load_templates_by_indices(
                object_dir, selected_indices, all_poses
            )
            template_data["template_indices"] = np.array(selected_indices)
            template_data["d_ref"] = selection_result["d_ref"]
            template_data["template_types"] = np.array(
                [0] * len(positive_indices) + [1] * len(negative_indices)
            )  # 0=positive, 1=negative
        
        return self._move_to_device(template_data) if return_metadata else {
            "images": self._move_to_device(template_data["images"]),
            "masks": self._move_to_device(template_data["masks"]),
            "poses": self._move_to_device(template_data["poses"]),
        }
    
    def _load_templates_by_indices(
        self,
        object_dir: Path,
        indices: Union[List[int], np.ndarray],
        all_poses: np.ndarray,
    ) -> Dict:
        """
        Load template images and poses for specific view indices.
        
        Args:
            object_dir: Path to object template directory
            indices: List of view indices to load
            all_poses: Full array of all template poses
        
        Returns:
            Dictionary with images, masks, poses
        """
        images = []
        masks = []
        poses = []
        
        for idx in indices:
            # Load RGBA image
            img_path = object_dir / f"{idx:06d}.png"
            if not img_path.exists():
                raise ValueError(f"Template image not found: {img_path}")
            
            rgba = np.array(Image.open(img_path))
            
            # Split RGB and alpha
            rgb = rgba[:, :, :3]
            alpha = rgba[:, :, 3] if rgba.shape[2] == 4 else np.ones(rgba.shape[:2], dtype=np.uint8) * 255
            
            # Binary mask (threshold alpha at 128)
            mask = (alpha > 128).astype(np.uint8)
            
            images.append(rgb)
            masks.append(mask)
            poses.append(all_poses[idx])
        
        return {
            "images": np.stack(images),  # [N, H, W, 3]
            "masks": np.stack(masks),    # [N, H, W]
            "poses": np.stack(poses),    # [N, 4, 4]
        }
    
    def _add_to_cache(self, object_id: str, data: Dict):
        """Add data to cache with LRU eviction."""
        # Remove oldest if cache is full
        if len(self.cache) >= self.cache_size:
            oldest_id = self.cache_order.pop(0)
            del self.cache[oldest_id]
            logger.debug(f"Evicted {oldest_id} from cache")
        
        # Add to cache
        self.cache[object_id] = data
        self.cache_order.append(object_id)
        logger.debug(f"Cached templates for object {object_id}")
    
    def _move_to_device(self, data: Dict) -> Dict:
        """Convert numpy arrays to tensors and move to device."""
        result = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                result[key] = torch.from_numpy(value).to(self.device)
            else:
                result[key] = value
        return result
    
    def preload_objects(self, object_ids: List[str]):
        """
        Preload templates for multiple objects into cache.
        Only works in "all" mode.
        
        Args:
            object_ids: List of object IDs to preload
        """
        if self.mode != "all":
            logger.warning("Preloading only supported in 'all' mode")
            return
        
        logger.info(f"Preloading {len(object_ids)} objects...")
        for obj_id in object_ids:
            if obj_id not in self.cache:
                try:
                    self.load_object_templates(obj_id)
                except Exception as e:
                    logger.error(f"Failed to preload {obj_id}: {e}")
        
        logger.info(f"Preloaded {len(self.cache)} objects")
    
    def clear_cache(self):
        """Clear the template cache."""
        self.cache.clear()
        self.cache_order.clear()
        logger.info("Cache cleared")
    
    def get_available_objects(self) -> List[str]:
        """
        Get list of available object IDs in the dataset.
        
        Returns:
            List of object ID strings
        """
        object_dirs = sorted([d.name for d in self.dataset_dir.iterdir() if d.is_dir()])
        return object_dirs
    
    def get_cache_info(self) -> Dict:
        """Get information about cache usage."""
        return {
            "cached_objects": len(self.cache),
            "cache_size": self.cache_size,
            "cache_order": self.cache_order.copy(),
        }


def create_template_manager(
    templates_dir: str,
    dataset_name: str = "gso",
    mode: str = "all",
    **kwargs
) -> TemplateManager:
    """
    Factory function to create a TemplateManager with common defaults.
    
    Args:
        templates_dir: Path to templates directory
        dataset_name: Dataset name ("gso", "lmo", etc.)
        mode: "all" or "subset"
        **kwargs: Additional arguments passed to TemplateManager
    
    Returns:
        Initialized TemplateManager instance
    """
    return TemplateManager(
        templates_dir=templates_dir,
        dataset_name=dataset_name,
        mode=mode,
        **kwargs
    )


if __name__ == "__main__":
    """
    Basic test of TemplateManager functionality.
    Run with: PYTHONPATH=/data/home/ssaricha/gigapose:$PYTHONPATH python vpog/inference/template_manager.py
    """
    import sys
    
    # Test configuration
    templates_dir = Path(__file__).parent.parent.parent / "datasets" / "templates"
    
    print("=" * 60)
    print("TemplateManager Basic Test")
    print("=" * 60)
    
    # Test 1: All mode
    print("\n=== Test 1: All Mode ===")
    manager_all = TemplateManager(
        templates_dir=templates_dir,
        dataset_name="gso",
        mode="all",
        cache_size=5,
    )
    
    # Load templates for object 733
    try:
        data = manager_all.load_object_templates("000733")
        print(f"✓ Loaded {len(data['images'])} templates for object 733")
        print(f"  Images shape: {data['images'].shape}")
        print(f"  Masks shape: {data['masks'].shape}")
        print(f"  Poses shape: {data['poses'].shape}")
        print(f"  Template indices: {data['template_indices'][:5]}... ({len(data['template_indices'])} total)")
    except Exception as e:
        print(f"✗ Failed to load templates: {e}")
        sys.exit(1)
    
    # Test 2: Subset mode
    print("\n=== Test 2: Subset Mode ===")
    manager_subset = TemplateManager(
        templates_dir=templates_dir,
        dataset_name="gso",
        mode="subset",
        num_positive=4,
        num_negative=2,
    )
    
    # Create a query pose
    from scipy.spatial.transform import Rotation
    R_query = Rotation.random().as_matrix()
    t_query = np.array([0.0, 0.0, 1.0])
    query_pose = np.eye(4)
    query_pose[:3, :3] = R_query
    query_pose[:3, 3] = t_query
    
    try:
        data = manager_subset.load_object_templates("000733", query_pose=query_pose)
        print(f"✓ Selected {len(data['images'])} templates (S_p + S_n)")
        print(f"  Images shape: {data['images'].shape}")
        print(f"  Template indices: {data['template_indices']}")
        print(f"  Template types: {data['template_types']} (0=pos, 1=neg)")
        print(f"  d_ref: {data['d_ref']}")
    except Exception as e:
        print(f"✗ Failed to select templates: {e}")
        sys.exit(1)
    
    # Test 3: Cache functionality
    print("\n=== Test 3: Cache Functionality ===")
    cache_info = manager_all.get_cache_info()
    print(f"Cache info: {cache_info}")
    
    # Load another object
    try:
        manager_all.load_object_templates("000001")
        cache_info = manager_all.get_cache_info()
        print(f"After loading object 1: {cache_info}")
    except Exception as e:
        print(f"Note: Object 000001 may not exist: {e}")
    
    # Test 4: Available objects
    print("\n=== Test 4: Available Objects ===")
    objects = manager_all.get_available_objects()
    print(f"Found {len(objects)} objects in dataset")
    print(f"First 10: {objects[:10]}")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)
