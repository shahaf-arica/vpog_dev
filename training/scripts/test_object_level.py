#!/usr/bin/env python3
"""
Quick test to verify object-level iteration works correctly.

Usage:
    python training/scripts/test_object_level.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from training.dataloader.object_index_builder import load_object_index
from training.dataloader.object_level_wrapper import ObjectLevelDataset
from src.custom_megapose.web_scene_dataset import WebSceneDataset


def test_object_index(dataset_name='ycbv', split='test'):
    """Test that object index can be loaded and used."""
    
    root_dir = Path('/strg/E/shared-data/Shahaf/gigapose/datasets')
    split_dir = root_dir / dataset_name / split
    
    print("="*80)
    print(f"Testing Object-Level Iteration: {dataset_name}/{split}")
    print("="*80)
    
    # Check if index exists
    index_path = split_dir / "object_index.json"
    if not index_path.exists():
        print(f"❌ Object index not found: {index_path}")
        print(f"\nBuild it first:")
        print(f"  python training/scripts/build_object_index.py --dataset {dataset_name} --split {split}")
        return False
    
    print(f"✓ Object index found: {index_path}")
    
    # Load scene dataset
    web_dataset = WebSceneDataset(split_dir, depth_scale=10.0, load_depth=False)
    print(f"✓ WebSceneDataset loaded: {len(web_dataset)} scenes")
    
    # Load object index
    object_index = load_object_index(index_path)
    print(f"✓ Object index loaded: {len(object_index)} objects")
    
    # Create object-level dataset
    dataset = ObjectLevelDataset(web_dataset, object_index)
    print(f"✓ ObjectLevelDataset created: {len(dataset)} objects")
    
    # Test access
    print(f"\n{'='*80}")
    print("Testing object access:")
    print("="*80)
    
    for i in [0, 1, 2]:
        obj_info = object_index[i]
        scene = dataset[i]
        
        print(f"\nObject {i}:")
        print(f"  Scene key:    {obj_info['scene_key']}")
        print(f"  Scene idx:    {obj_info['scene_idx']}")
        print(f"  Object idx:   {obj_info['obj_idx']}")
        print(f"  Object ID:    {obj_info['obj_id']}")
        print(f"  Visibility:   {obj_info['visib_fract']:.2f}")
        print(f"  RGB shape:    {scene.rgb.shape}")
        print(f"  Objects:      {len(scene.object_datas)}")
        
        if len(scene.object_datas) != 1:
            print(f"  ❌ ERROR: Expected 1 object, got {len(scene.object_datas)}")
            return False
    
    print(f"\n{'='*80}")
    print("✓ All tests passed!")
    print("="*80)
    
    return True


if __name__ == '__main__':
    success = test_object_index()
    sys.exit(0 if success else 1)
