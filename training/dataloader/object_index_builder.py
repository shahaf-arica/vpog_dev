"""
Object index builder for VPOG dataset.

Builds object-level index from scene-level datasets to enable:
1. Object-centric iteration (not scene-centric)
2. Deterministic validation (iterate all objects)
3. Proper shuffling in training
4. Predictable memory usage
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
import webdataset as wds

from src.custom_megapose.web_scene_dataset import WebSceneDataset
from src.megapose.utils.webdataset import tarfile_to_samples
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ObjectIndexBuilder:
    """Builds object-level index from scene-level dataset."""
    
    def __init__(
        self,
        root_dir: Path,
        dataset_name: str,
        split: str,
        min_visib_fract: float = 0.1,
        depth_scale: float = 10.0,
        validate_templates: bool = True,
    ):
        """
        Args:
            root_dir: Root directory for datasets
            dataset_name: Dataset name (gso, shapenet, ycbv, etc.)
            split: Split name (train_pbr_web, test, etc.)
            min_visib_fract: Minimum visibility fraction to include object
            depth_scale: Depth scale factor for loading
            validate_templates: If True, skip objects with missing template images
        """
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.split = split
        self.min_visib_fract = min_visib_fract
        self.depth_scale = depth_scale
        self.validate_templates = validate_templates
        
        self.dataset_dir = self.root_dir / dataset_name
        self.split_dir = self.dataset_dir / split
        self.output_path = self.split_dir / "object_index.json"
        self.templates_dir = self.root_dir / "templates" / dataset_name
        
        # Validate paths
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
    
    def build_index(self) -> Dict:
        """
        Build object index by iterating through webdataset tar archives.
        
        Returns:
            Dictionary containing object index metadata and object list
        """
        
        logger.info(f"Building object index for {self.dataset_name}/{self.split}")
        logger.info(f"Minimum visibility: {self.min_visib_fract}")
        
        # Get tar file list
        web_dataset = WebSceneDataset(
            self.split_dir,
            depth_scale=self.depth_scale,
            load_depth=False,
        )
        tar_files = web_dataset.get_tar_list()
        
        logger.info(f"Processing {len(tar_files)} tar shards...")
        
        # Create webdataset pipeline to iterate through samples
        dataset = wds.DataPipeline(
            wds.SimpleShardList(tar_files),
            tarfile_to_samples(),
        )
        
        objects = []
        filtered_count = 0
        missing_templates_count = 0
        scene_count = 0
        
        # Iterate through all samples
        for sample in tqdm(dataset, desc="Indexing objects"):
            scene_key = sample["__key__"]
            
            # Check if GT data exists
            if "gt.json" not in sample or "gt_info.json" not in sample:
                continue
            
            try:
                # Parse GT files
                objects_gt = json.loads(sample["gt.json"])
                objects_gt_info = json.loads(sample["gt_info.json"])
                
                # Process each object in this scene
                for obj_idx, (obj_data, obj_info) in enumerate(zip(objects_gt, objects_gt_info)):
                    visib_fract = obj_info['visib_fract']
                    
                    # Filter by visibility (matching WebSceneDataset logic)
                    if visib_fract <= 0.1:
                        filtered_count += 1
                        continue
                    
                    # Additional filter by min_visib_fract
                    if visib_fract < self.min_visib_fract:
                        filtered_count += 1
                        continue
                    
                    obj_id = obj_data['obj_id']
                    
                    # Validate template images exist
                    if self.validate_templates:
                        template_obj_dir = self.templates_dir / f"{obj_id:06d}"
                        # Check if directory exists and has at least one .png file
                        if not template_obj_dir.exists():
                            logger.warning(f"Missing template directory for obj_id {obj_id:06d}, skipping")
                            missing_templates_count += 1
                            continue
                        
                        png_files = list(template_obj_dir.glob("*.png"))
                        if len(png_files) == 0:
                            logger.warning(f"Empty template directory for obj_id {obj_id:06d}, skipping")
                            missing_templates_count += 1
                            continue
                    
                    objects.append({
                        "global_idx": len(objects),
                        "scene_key": scene_key,
                        "scene_idx": scene_count,
                        "obj_idx": obj_idx,
                        "obj_id": int(obj_id),
                        "visib_fract": float(visib_fract),
                    })
                
                scene_count += 1
                
            except Exception as e:
                logger.warning(f"Error processing scene {scene_key}: {e}")
                continue
        
        index_data = {
            "version": "1.0",
            "dataset": self.dataset_name,
            "split": self.split,
            "created_at": datetime.now().isoformat(),
            "total_scenes": scene_count,
            "total_objects": len(objects),
            "filtered_objects": filtered_count,
            "missing_templates": missing_templates_count,
            "min_visib_fract": self.min_visib_fract,
            "validated_templates": self.validate_templates,
            "objects": objects,
        }
        
        logger.info(f"✓ Found {len(objects)} objects across {scene_count} scenes")
        logger.info(f"  Filtered {filtered_count} objects (visib < {self.min_visib_fract})")
        if self.validate_templates:
            logger.info(f"  Skipped {missing_templates_count} objects (missing templates)")
        
        return index_data
    
    def save_index(self, index_data: Dict):
        """Save index to JSON file."""
        with open(self.output_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        logger.info(f"✓ Saved to {self.output_path}")
        logger.info(f"  Size: {self.output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    def build_and_save(self):
        """Build and save object index."""
        index_data = self.build_index()
        self.save_index(index_data)


def load_object_index(index_path: Path) -> List[Dict]:
    """
    Load object index from JSON file.
    
    Args:
        index_path: Path to object_index.json
        
    Returns:
        List of object metadata dictionaries
    """
    with open(index_path) as f:
        data = json.load(f)
    
    # Validate version
    if data.get('version') != '1.0':
        logger.warning(f"Object index version mismatch: {data.get('version')} (expected 1.0)")
    
    return data['objects']
