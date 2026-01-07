#!/usr/bin/env python3
"""
Build object index for VPOG training.

This script generates object_index.json which maps object-level access
to scene-level data, enabling efficient object-centric iteration.

Usage:
    python training/scripts/build_object_index.py --dataset gso --split train_pbr_web
    python training/scripts/build_object_index.py --dataset ycbv --split test --root_dir /path/to/datasets
    
    # Build for multiple datasets
    python training/scripts/build_object_index.py --dataset gso --split train_pbr_web
    python training/scripts/build_object_index.py --dataset shapenet --split train_pbr_web
    python training/scripts/build_object_index.py --dataset ycbv --split test
"""

import argparse
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from training.dataloader.object_index_builder import ObjectIndexBuilder


def main():
    parser = argparse.ArgumentParser(
        description='Build object index for VPOG dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training/scripts/build_object_index.py --dataset gso --split train_pbr_web
  python training/scripts/build_object_index.py --dataset ycbv --split test --min_visib 0.1
        """
    )
    parser.add_argument('--dataset', required=True, 
                       help='Dataset name (gso, shapenet, ycbv, tless, lmo, etc.)')
    parser.add_argument('--split', required=True, 
                       help='Split name (train_pbr_web, test, test_all, etc.)')
    parser.add_argument('--root_dir', default='datasets',
                       help='Root directory for datasets')
    parser.add_argument('--min_visib', type=float, default=0.1, 
                       help='Minimum visibility fraction (default: 0.1)')
    parser.add_argument('--depth_scale', type=float, default=10.0, 
                       help='Depth scale factor (default: 10.0)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("VPOG Object Index Builder")
    print("="*80)
    print(f"Dataset:    {args.dataset}")
    print(f"Split:      {args.split}")
    print(f"Root dir:   {args.root_dir}")
    print(f"Min visib:  {args.min_visib}")
    print("="*80)
    
    builder = ObjectIndexBuilder(
        root_dir=Path(args.root_dir),
        dataset_name=args.dataset,
        split=args.split,
        min_visib_fract=args.min_visib,
        depth_scale=args.depth_scale
    )
    
    builder.build_and_save()
    
    print("="*80)
    print("✓ Object index built successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
