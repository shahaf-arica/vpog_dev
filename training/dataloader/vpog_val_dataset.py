"""
VPOG Validation Dataset
Validation dataloader for BOP datasets (YCBV, etc.)

Loads:
- BOP test/validation images
- Templates for each object
- Ground truth poses for evaluation
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import from VPOGTrainDataset to reuse batch structure
from training.dataloader.vpog_dataset import VPOGBatch, VPOGTrainDataset
from src.custom_megapose.web_scene_dataset import WebSceneDataset, IterableWebSceneDataset
from src.custom_megapose.template_dataset import TemplateDataset
from bop_toolkit_lib import inout
from src.utils.logging import get_logger

logger = get_logger(__name__)


class VPOGValDataset(VPOGTrainDataset):
    """
    Validation dataset for VPOG using BOP datasets.
    
    Inherits from VPOGTrainDataset but:
    - Uses test/val split from BOP
    - No data augmentation
    - Fixed template selection (no randomness)
    - Evaluates against ground truth poses
    """
    
    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        template_config: Dict,
        depth_scale: float = 10.0,
        patch_size: int = 16,
        image_size: int = 224,
        num_positive_templates: int = 3,
        num_negative_templates: int = 2,
        test_setting: str = 'localization',
        load_gt: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            root_dir: Root directory containing BOP datasets
            dataset_name: BOP dataset name (e.g., 'ycbv', 'tless', 'lmo')
            template_config: Template configuration dict
            depth_scale: Scale factor for depth values
            patch_size: Size of patches for vision transformer
            image_size: Input image size
            num_positive_templates: Number of positive (nearby) templates
            num_negative_templates: Number of negative (far) templates
            test_setting: 'localization' or 'detection'
            load_gt: Whether to load ground truth annotations
            seed: Random seed for reproducibility
        """
        logger.info(f"Initializing VPOG Validation Dataset for {dataset_name}...")
        
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.dataset_dir = self.root_dir / dataset_name
        self.depth_scale = depth_scale
        self.patch_size = patch_size
        self.image_size = image_size
        self.test_setting = test_setting
        self.load_gt = load_gt
        
        # Template selection params
        self.num_positive_templates = num_positive_templates
        self.num_negative_templates = num_negative_templates
        self.num_templates = num_positive_templates + num_negative_templates
        
        # Calculate patch grid dimensions
        assert image_size % patch_size == 0, f"Image size {image_size} must be divisible by patch size {patch_size}"
        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        
        logger.info(f"  Dataset: {dataset_name}")
        logger.info(f"  Root dir: {root_dir}")
        logger.info(f"  Image size: {image_size}, Patch size: {patch_size}")
        logger.info(f"  Patches per side: {self.num_patches_per_side}, Total patches: {self.num_patches}")
        logger.info(f"  Test setting: {test_setting}")
        
        # Initialize attributes to match parent VPOGTrainDataset
        self.seed = seed
        self.batch_size = 1  # Validation typically uses batch_size=1
        self.transforms = {}  # No augmentation for validation
        self.debug_mode = False
        
        # Set random seed for reproducibility
        if seed is not None:
            import random
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            logger.info(f"  Set random seed to {seed} for reproducibility")
        
        # Determine split name (test-all, test, val, etc.)
        split_name = self._get_split_name(dataset_name)
        
        # Load the web dataset (validation/test images)
        web_dataset = WebSceneDataset(
            self.dataset_dir / split_name,
            depth_scale=depth_scale
        )
        self.size = len(web_dataset)
        self.web_dataloader = IterableWebSceneDataset(web_dataset, set_length=True)
        
        # Load template dataset
        model_infos = inout.load_json(self.dataset_dir / "models" / "models_info.json")
        model_infos = [{"obj_id": int(obj_id)} for obj_id in model_infos.keys()]
        
        # Update template config with dataset-specific path and convert to namespace for attribute access
        from types import SimpleNamespace
        template_config = dict(template_config)
        template_config['dir'] = str(Path(template_config['dir']) / dataset_name)
        template_config_obj = SimpleNamespace(**template_config)
        
        self.template_dataset = TemplateDataset.from_config(model_infos, template_config_obj)
        
        # Initialize template selector (for validation, use deterministic selection)
        from training.dataloader.vpog_dataset import TemplateSelector
        self.template_selector = TemplateSelector(
            level_templates=template_config.get('level_templates', 1),
            pose_distribution=template_config.get('pose_distribution', 'all'),
            num_positive=num_positive_templates,
            num_negative=num_negative_templates,
            min_negative_angle_deg=90.0,
            d_ref_random_ratio=0.0,  # Always use nearest for validation
            seed=seed,  # Use same seed for reproducible validation
        )
        
        # Initialize flow computer (same as parent but no augmentation)
        from training.dataloader.vpog_dataset import FlowComputer
        self.flow_computer = FlowComputer(
            patch_size=patch_size,
            compute_visibility=True,
            compute_patch_visibility=True,
            visibility_threshold=0.1,
        )
        
        # Setup transforms (no augmentation for validation)
        self._setup_transforms()
        
        logger.info(f"  Loaded {len(model_infos)} objects")
        logger.info(f"  Loaded {self.size} validation samples")
        logger.info(f"  Templates per sample: {self.num_templates} ({self.num_positive_templates} positive + {self.num_negative_templates} negative)")

    def __len__(self) -> int:
        """Return number of validation samples."""
        return self.size
    
    def _setup_transforms(self):
        """Setup transforms for validation (no augmentation)."""
        import torchvision.transforms as T
        
        # Default normalization (CLIP stats)
        self.normalize = T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        
        # Crop transform
        from src.utils.crop import CropResizePad
        self.crop_transform = CropResizePad(target_size=self.image_size)
        
        # No RGB augmentation for validation
        self.rgb_augmentation = False
        self.rgb_transform = None
        
        # No inplane augmentation for validation
        self.inplane_augmentation = False
    
    def _get_split_name(self, dataset_name: str) -> str:
        """Get the appropriate split name for BOP dataset."""
        # Most BOP datasets use 'test' or 'test_all'
        # Some have separate 'val' splits
        possible_splits = ['test_all', 'test', 'val']
        
        for split in possible_splits:
            if (self.dataset_dir / split).exists():
                logger.info(f"  Using split: {split}")
                return split
        
        # Default to 'test'
        logger.warning(f"  Could not find standard split, defaulting to 'test'")
        return 'test'


if __name__ == "__main__":
    # Test the validation dataset
    import hydra
    from omegaconf import DictConfig, OmegaConf
    from torch.utils.data import DataLoader
    
    # Simple test
    template_config = {
        'dir': '/data/home/ssaricha/gigapose/datasets/templates',
        'level_templates': 1,
        'pose_distribution': 'all',
        'scale_factor': 1.0,
        'num_templates': 162,
        'image_name': 'OBJECT_ID/VIEW_ID.png',
        'pose_name': 'object_poses/OBJECT_ID.npy',
    }
    
    dataset = VPOGValDataset(
        root_dir='/data/home/ssaricha/gigapose/datasets',
        dataset_name='ycbv',
        template_config=template_config,
        depth_scale=10.0,
        patch_size=16,
        image_size=224,
        num_positive_templates=3,
        num_negative_templates=2,
        test_setting='localization',
        load_gt=True,
        seed=42,
    )
    
    print(f"\nValidation dataset created successfully!")
    print(f"Total samples: {len(dataset)}")
