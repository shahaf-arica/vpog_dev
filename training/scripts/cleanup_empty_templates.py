#!/usr/bin/env python3
"""
Cleanup empty template directories.

Usage:
    python training/scripts/cleanup_empty_templates.py --dataset shapenet
    python training/scripts/cleanup_empty_templates.py --dataset shapenet --dry-run
"""

import argparse
import shutil
from pathlib import Path

def cleanup_empty_dirs(templates_dir: Path, dry_run: bool = True):
    """Remove empty template directories."""
    
    if not templates_dir.exists():
        print(f"Error: {templates_dir} does not exist")
        return
    
    empty_dirs = []
    for obj_dir in templates_dir.iterdir():
        if obj_dir.is_dir():
            # Check if directory is empty or has no .png files
            png_files = list(obj_dir.glob("*.png"))
            if len(png_files) == 0:
                empty_dirs.append(obj_dir)
    
    print(f"Found {len(empty_dirs)} empty template directories")
    
    if len(empty_dirs) == 0:
        print("Nothing to cleanup!")
        return
    
    if dry_run:
        print("\nDRY RUN - would remove:")
        for i, d in enumerate(empty_dirs[:10]):
            print(f"  {d.name}")
        if len(empty_dirs) > 10:
            print(f"  ... and {len(empty_dirs) - 10} more")
        print("\nRun without --dry-run to actually remove these directories")
    else:
        print("Removing empty directories...")
        for obj_dir in empty_dirs:
            shutil.rmtree(obj_dir)
        print(f"✓ Removed {len(empty_dirs)} empty directories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cleanup empty template directories')
    parser.add_argument('--dataset', required=True, help='Dataset name (gso, shapenet)')
    parser.add_argument('--root-dir', default='datasets', help='Root directory for datasets')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be removed without removing')
    
    args = parser.parse_args()
    
    templates_dir = Path(args.root_dir) / "templates" / args.dataset
    
    print("="*80)
    print("Template Directory Cleanup")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Path: {templates_dir}")
    print(f"Dry run: {args.dry_run}")
    print("="*80)
    
    cleanup_empty_dirs(templates_dir, dry_run=args.dry_run)
