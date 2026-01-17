"""
BOP Test Dataloader for VPOG

Loads BOP test images with default detections (CNOS) and provides:
- Cropped query images per detection
- All 162 templates for each object
- Detection metadata (scene_id, im_id, obj_id, bbox, etc.)
- Reuses src.dataloader logic where possible

Author: VPOG Team
Date: January 2026
"""

from __future__ import annotations

import os
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from bop_toolkit_lib import inout

from src.utils.logging import get_logger
from src.utils.inout import load_test_list_and_cnos_detections
from src.custom_megapose.web_scene_dataset import WebSceneDataset, IterableWebSceneDataset
from src.custom_megapose.template_dataset import TemplateDataset
from src.megapose.utils.tensor_collection import PandasTensorCollection

logger = get_logger(__name__)


@dataclass
class BOPDetection:
    """Single BOP test detection."""
    scene_id: int
    im_id: int
    obj_id: int
    bbox: np.ndarray  # [4] (x1, y1, x2, y2) or (x, y, w, h)
    score: float
    time: float  # Detection time
    inst_count: int  # Number of instances of this object in this image
    image_id: int  # Unique image identifier
    
    def __repr__(self) -> str:
        return (f"BOPDetection(scene={self.scene_id}, im={self.im_id}, "
                f"obj={self.obj_id}, bbox={self.bbox}, score={self.score:.3f})")


class BOPTestDataset(Dataset):
    """
    BOP Test Dataset for VPOG evaluation.
    
    Loads BOP test images with CNOS default detections and provides:
    - Cropped query images (one per detection)
    - Full template set for each object (162 templates)
    - Detection metadata for BOP submission
    
    Reuses existing logic from src.dataloader.test.GigaPoseTestSet
    """
    
    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        templates_dir: str,
        test_setting: str = "localization",  # "localization" or "detection"
        depth_scale: float = 1000.0,
        image_size: int = 224,
        max_det_per_object: Optional[int] = None,
        normalize_rgb: bool = True,
    ):
        """
        Args:
            root_dir: Root directory containing BOP datasets
            dataset_name: BOP dataset name (ycbv, tless, lmo, etc.)
            templates_dir: Directory containing rendered templates
            test_setting: "localization" (use test_targets) or "detection" (all detections)
            depth_scale: Depth scaling factor
            image_size: Image size for cropping (224 for VPOG)
            max_det_per_object: Max detections per object (None = all)
            normalize_rgb: Whether to normalize RGB to [0,1]
        """
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.templates_dir = Path(templates_dir)
        self.test_setting = test_setting
        self.depth_scale = depth_scale
        self.image_size = image_size
        self.normalize_rgb = normalize_rgb
        
        # Load test split name
        split, model_name = self._get_split_name(dataset_name)
        self.split = split
        self.model_name = model_name
        
        # Load web dataset (BOP test images)
        webdataset_dir = self.root_dir / self.dataset_name
        web_dataset = WebSceneDataset(
            webdataset_dir / split,
            depth_scale=depth_scale
        )
        self.web_dataloader = IterableWebSceneDataset(web_dataset, set_length=True)
        logger.info(f"Loaded web dataset from {webdataset_dir / split}")
        
        # Load model info
        model_infos = inout.load_json(
            self.root_dir / self.dataset_name / model_name / "models_info.json"
        )
        self.model_infos = [{"obj_id": int(obj_id)} for obj_id in model_infos.keys()]
        self.obj_ids = sorted([int(obj_id) for obj_id in model_infos.keys()])
        logger.info(f"Found {len(self.obj_ids)} objects: {self.obj_ids}")
        
        # Load default detections (CNOS)
        self._load_detections(max_det_per_object)
        
        # Build detection list (flat list of all detections)
        self._build_detection_list()
        
        logger.info(
            f"BOPTestDataset initialized: {len(self.detections)} detections "
            f"from {len(self.test_list)} test targets"
        )
    
    def _get_split_name(self, dataset_name: str) -> Tuple[str, str]:
        """Get split and model name for dataset."""
        if dataset_name in ["hope", "handal", "hot3d"]:
            split = "test"
            model_name = "models"
        else:
            split = "test_primesense" if dataset_name == "tless" else "test"
            model_name = "models_cad" if dataset_name == "tless" else "models"
        return split, model_name
    
    def _load_detections(self, max_det_per_object: Optional[int]):
        """Load CNOS detections and test list."""
        # Set max detections per object (for localization)
        if self.test_setting == "localization":
            if max_det_per_object is None:
                max_det_per_object = 32 if self.dataset_name == "icbin" else 16
        else:
            max_det_per_object = None
        
        # Load using existing utility
        self.test_list, self.cnos_dets = load_test_list_and_cnos_detections(
            self.root_dir,
            self.dataset_name,
            self.test_setting,
            max_det_per_object_id=max_det_per_object,
        )
        
        logger.info(f"Loaded {len(self.test_list)} test targets")
        logger.info(f"Loaded detections for {len(self.cnos_dets)} images")
    
    def _build_detection_list(self):
        """Build flat list of all detections."""
        self.detections = []
        
        for test_target in self.test_list:
            scene_id = test_target["scene_id"]
            im_id = test_target["im_id"]
            obj_id = test_target["obj_id"]
            inst_count = test_target["inst_count"]
            image_id = test_target["image_id"]
            
            # Get detections for this image
            image_key = f"{scene_id:06d}_{im_id:06d}"
            if image_key not in self.cnos_dets:
                logger.warning(f"No detections found for {image_key}")
                continue
            
            image_dets = self.cnos_dets[image_key]
            
            # Filter detections for this object
            obj_dets = [d for d in image_dets if d["category_id"] == obj_id]
            
            # Create detection objects
            for det in obj_dets:
                bbox = np.array(det["bbox"], dtype=np.float32)
                self.detections.append(BOPDetection(
                    scene_id=scene_id,
                    im_id=im_id,
                    obj_id=obj_id,
                    bbox=bbox,
                    score=det.get("score", 1.0),
                    time=det.get("time", 0.0),
                    inst_count=inst_count,
                    image_id=image_id,
                ))
    
    def __len__(self) -> int:
        """Number of detections (not images)."""
        return len(self.detections)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single detection with query image and templates.
        
        Returns:
            dict with:
                - query_image: [3, H, W] RGB tensor
                - query_mask: [H, W] mask tensor
                - query_K: [3, 3] intrinsics (after cropping)
                - templates: dict with template data
                - detection: BOPDetection metadata
        """
        detection = self.detections[idx]
        
        # Load query image
        image_key = f"{detection.scene_id:06d}_{detection.im_id:06d}"
        query_data = self._load_query_image(image_key, detection.bbox)
        
        # Load templates for this object
        template_data = self._load_templates(detection.obj_id)
        
        return {
            "query_image": query_data["image"],
            "query_mask": query_data["mask"],
            "query_K": query_data["K"],
            "query_M": query_data["M"],
            "templates": template_data,
            "detection": detection,
        }
    
    def _load_query_image(
        self,
        image_key: str,
        bbox: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """Load and crop query image."""
        # Load RGB and camera data from web dataset
        scene_data = self.web_dataset_dict[image_key]
        rgb = scene_data["rgb"]  # [H, W, 3] uint8
        K = scene_data["K"]  # [3, 3]
        
        # Convert bbox format if needed (handle both [x, y, w, h] and [x1, y1, x2, y2])
        if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
            # [x, y, w, h] format
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
        else:
            # [x1, y1, x2, y2] format
            x1, y1, x2, y2 = bbox
        
        # Crop image
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(rgb.shape[1], x2)), int(min(rgb.shape[0], y2))
        cropped_rgb = rgb[y1:y2, x1:x2]
        
        # Resize to target size
        from PIL import Image
        pil_img = Image.fromarray(cropped_rgb)
        pil_img = pil_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        cropped_rgb = np.array(pil_img)
        
        # Convert to tensor and normalize
        if self.normalize_rgb:
            image_tensor = torch.from_numpy(cropped_rgb).float() / 255.0
        else:
            image_tensor = torch.from_numpy(cropped_rgb).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # [3, H, W]
        
        # Create mask (all ones for now, could be refined with segmentation)
        mask_tensor = torch.ones((self.image_size, self.image_size), dtype=torch.float32)
        
        # Compute crop transformation matrix M
        # Scale from crop size to target size
        crop_w, crop_h = x2 - x1, y2 - y1
        scale_x = self.image_size / crop_w
        scale_y = self.image_size / crop_h
        
        M = torch.tensor([
            [scale_x, 0, -x1 * scale_x],
            [0, scale_y, -y1 * scale_y],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        # Update intrinsics for cropped image
        K_cropped = M @ torch.from_numpy(K).float()
        
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "K": K_cropped,
            "M": M,
        }
    
    def _load_templates(self, obj_id: int) -> Dict[str, torch.Tensor]:
        """Load all templates for an object."""
        # Get template object for this obj_id
        label = f"obj_{obj_id:06d}"
        template_object = self.template_dataset.get_object_templates(label)
        
        # Load all templates (all view_ids)
        view_ids = list(range(template_object.num_templates))
        template_data = template_object.load_set_of_templates(
            view_ids=view_ids,
            reload=True,
            inplanes=None,
            reset=False,
        )
        
        # Load poses
        template_poses = template_object.load_pose(view_ids=None, inplanes=[0])
        
        # Apply crop transform if needed
        if hasattr(self, 'crop_transform') and self.crop_transform is not None:
            template_data = template_object.apply_transform(
                self.crop_transform,
                template_data
            )
        
        # Normalize RGB channels (templates are RGBA)
        template_rgb = template_data["rgba"][:, :3]  # [N, 3, H, W]
        if self.normalize_rgb:
            template_rgb = template_rgb  # Already in [0, 1] from load_set_of_templates
        
        return {
            "images": template_rgb,  # [N, 3, H, W]
            "masks": template_data["rgba"][:, 3],  # [N, H, W]
            "depths": template_data["depth"],  # [N, 1, H, W]
            "poses": template_poses,  # [N, 4, 4]
            "K": torch.from_numpy(self.template_dataset.K).float(),  # [3, 3]
            "num_templates": len(view_ids),
        }
    
    def get_test_list(self) -> List[Dict]:
        """Get test list for BOP evaluation."""
        return self.test_list
    
    def get_detections_by_image(self) -> Dict[str, List[BOPDetection]]:
        """Group detections by image for batch processing."""
        detections_by_image = {}
        for det in self.detections:
            image_key = f"{det.scene_id:06d}_{det.im_id:06d}"
            if image_key not in detections_by_image:
                detections_by_image[image_key] = []
            detections_by_image[image_key].append(det)
        return detections_by_image


def collate_bop_test(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for BOPTestDataset.
    
    Since templates can be large, we keep batch_size=1 or handle carefully.
    """
    if len(batch) == 1:
        return batch[0]
    
    # For batch_size > 1, stack tensors
    return {
        "query_images": torch.stack([b["query_image"] for b in batch]),
        "query_masks": torch.stack([b["query_mask"] for b in batch]),
        "query_K": torch.stack([b["query_K"] for b in batch]),
        "query_M": torch.stack([b["query_M"] for b in batch]),
        "templates": [b["templates"] for b in batch],  # List of template dicts
        "detections": [b["detection"] for b in batch],
    }


if __name__ == "__main__":
    # Test dataloader
    from src.utils.logging import setup_logger
    setup_logger()
    
    root_dir = "/path/to/bop/datasets"
    templates_dir = "/path/to/templates"
    
    dataset = BOPTestDataset(
        root_dir=root_dir,
        dataset_name="ycbv",
        templates_dir=templates_dir,
        test_setting="localization",
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"First detection: {dataset.detections[0]}")
