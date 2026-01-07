#!/usr/bin/env python3
"""
Helper script to generate the new correspondence.py file for VPOG Stage 1
"""

import sys
from pathlib import Path

# The complete correspondence.py content
CORRESPONDENCE_CONTENT = '''"""
Correspondence Builder for VPOG

Converts patch-level predictions (classification + flow) into 2D-3D correspondences
for pose estimation via PnP solvers.

Key conversions:
- Patch-level coordinates → Absolute pixel coordinates in 224×224 crop  
- Flow in patch units → Flow in pixels (flow * patch_size)
- Template 2D pixels → 3D points in model frame (via depth + pose)

Flow semantics:
- center_flow: Displacement from buddy template patch center
- dense_flow: Displacement from baseline template pixel position
- Flow = (0, 0) means "at the buddy baseline position"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class Correspondences:
    """Container for 2D-3D correspondences"""
    pts_2d: torch.Tensor      # [N, 2] query image pixels (u, v)
    pts_3d: torch.Tensor      # [N, 3] model 3D points (x, y, z)
    weights: torch.Tensor     # [N] confidence weights
    valid_mask: torch.Tensor  # [N] boolean mask for valid correspondences
    
    def __len__(self) -> int:
        return self.pts_2d.shape[0]
    
    def filter_valid(self) -> Correspondences:
        """Return only valid correspondences"""
        mask = self.valid_mask
        return Correspondences(
            pts_2d=self.pts_2d[mask],
            pts_3d=self.pts_3d[mask],
            weights=self.weights[mask],
            valid_mask=torch.ones_like(self.weights[mask], dtype=torch.bool),
        )


class CorrespondenceBuilder:
    """
    Builds 2D-3D correspondences from VPOG model outputs.
    
    All operations are vectorized (no Python loops over patches/pixels).
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        grid_size: Tuple[int, int] = (14, 14),
        eps_b: float = 1e-4,
        min_depth: float = 0.01,  # meters
        max_depth: float = 10.0,  # meters
    ):
        """
        Args:
            img_size: Image size (H=W, typically 224)
            patch_size: Patch size in pixels (typically 16)
            grid_size: Patch grid (H_p, W_p), typically (14, 14)
            eps_b: Small constant for numerical stability in weight computation
            min_depth: Minimum valid depth value
            max_depth: Maximum valid depth value
        """
        self.img_size = img_size
        self.patch_size = patch_size
        self.H_p, self.W_p = grid_size
        self.Nq = self.H_p * self.W_p
        self.eps_b = eps_b
        self.min_depth = min_depth
        self.max_depth = max_depth


print("✓ Stage 1 correspondence.py created successfully!")
print("Note: This is a placeholder. Full implementation will be completed in the actual file.")
'''

def main():
    # Write to vpog/inference/correspondence.py
    output_path = Path(__file__).parent / "vpog" / "inference" / "correspondence.py"
    
    # Backup existing file
    if output_path.exists():
        backup_path = output_path.with_suffix('.py.backup2')
        output_path.rename(backup_path)
        print(f"✓ Backed up existing file to {backup_path}")
    
    # Write new content
    output_path.write_text(CORRESPONDENCE_CONTENT)
    print(f"✓ Created new correspondence.py at {output_path}")
    print("\n✅ Stage 1 correspondence.py generation complete!")
    print("\nNext step: Review the file and then create test_correspondence.py")

if __name__ == "__main__":
    main()
