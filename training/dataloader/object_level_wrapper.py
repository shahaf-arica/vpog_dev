"""
Object-level dataset wrapper.

Wraps scene-level WebSceneDataset to provide object-level access,
where each __getitem__ returns a scene with a single object.
"""

from typing import List, Dict
from functools import lru_cache

from src.custom_megapose.web_scene_dataset import WebSceneDataset
from src.megapose.datasets.scene_dataset import SceneObservation


class ObjectLevelDataset:
    """
    Wraps WebSceneDataset to provide object-level access.
    
    Instead of iterating over scenes (each containing N objects),
    iterates over individual objects (each SceneObservation has 1 object).
    
    This enables:
    - Proper shuffling at object level (not scene level)
    - Deterministic validation over all objects
    - Predictable batch sizes and memory usage
    
    Example:
        web_dataset = WebSceneDataset(split_dir)
        object_index = load_object_index(split_dir / "object_index.json")
        dataset = ObjectLevelDataset(web_dataset, object_index)
        
        # Now dataset[i] returns a scene with exactly 1 object
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    """
    
    def __init__(
        self,
        web_dataset: WebSceneDataset,
        object_index: List[Dict],
    ):
        """
        Args:
            web_dataset: Scene-level WebSceneDataset
            object_index: List of object metadata from object_index.json
        """
        self.web_dataset = web_dataset
        self.object_index = object_index
        
        # Build mapping from scene_key to positional index (not DataFrame index)
        self.scene_key_to_idx = {
            row['key']: pos_idx 
            for pos_idx, (_, row) in enumerate(web_dataset.frame_index.iterrows())
        }
    
    def __len__(self) -> int:
        """Number of objects (not scenes)."""
        return len(self.object_index)
    
    @lru_cache(maxsize=128)
    def _get_scene(self, scene_key: str) -> SceneObservation:
        """
        Load scene with LRU cache.
        
        When multiple objects come from the same scene (common in batches),
        this prevents redundant loading.
        """
        # Look up frame_index position using scene_key
        frame_idx = self.scene_key_to_idx[scene_key]
        return self.web_dataset[frame_idx]
    
    def __getitem__(self, idx: int) -> SceneObservation:
        """
        Return scene observation with single object.
        
        Args:
            idx: Object index (0 to len(object_index)-1)
            
        Returns:
            SceneObservation with exactly 1 object in object_datas
        """
        obj_info = self.object_index[idx]
        scene_key = obj_info['scene_key']
        target_obj_id = obj_info['obj_id']  # Use obj_id to match, not obj_idx
        
        # Load full scene (cached if recently accessed)
        scene = self._get_scene(scene_key)
        
        # Find the object by obj_id (since indices don't match after filtering)
        target_obj_data = None
        target_obj_idx = None
        for i, obj_data in enumerate(scene.object_datas):
            if int(obj_data.label) == target_obj_id:
                target_obj_data = obj_data
                target_obj_idx = i
                break
        
        if target_obj_data is None:
            raise ValueError(
                f"Object with obj_id={target_obj_id} not found in scene {scene_key}. "
                f"Available obj_ids: {[int(o.label) for o in scene.object_datas]}"
            )
        
        # Extract single object
        # Use the object's unique_id (original index) to get the correct mask
        # Then re-key it to '0' since we're returning a scene with just 1 object
        original_unique_id = target_obj_data.unique_id
        single_obj_masks = {}
        if original_unique_id in scene.binary_masks:
            single_obj_masks['0'] = scene.binary_masks[original_unique_id]
        
        # Update the object's unique_id to '0' for the single-object scene
        target_obj_data.unique_id = '0'
        
        single_obj_scene = SceneObservation(
            rgb=scene.rgb,
            depth=scene.depth,
            infos=scene.infos,
            object_datas=[target_obj_data],  # Single object
            camera_data=scene.camera_data,
            binary_masks=single_obj_masks,
        )
        
        return single_obj_scene
