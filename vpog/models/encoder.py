"""
CroCo-V2 Encoder Wrapper for VPOG

Wraps the pretrained CroCo-V2 encoder from external/croco to extract 
patch-level features with RoPE2D positional encoding for VPOG model.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Import CroCo from external/croco
from external.croco.models.croco import CroCoNet
from external.croco.models.croco_downstream import croco_args_from_ckpt


class CroCoEncoder(nn.Module):
    """
    CroCo-V2 encoder wrapper that loads pretrained weights and provides
    a simple interface for patch feature extraction.
    
    This is a lightweight wrapper around CroCoNet from external/croco that:
    - Loads pretrained CroCo-V2 weights from checkpoints/
    - Extracts only encoder features (no decoder/masking)
    - Returns [B, N, D] patch features and [B, N, 2] positions
    
    Args:
        checkpoint_name: Name of checkpoint ('base' or 'large', default: 'large')
        pretrained_path: Path to checkpoint (default: auto-detected from checkpoint_name)
        freeze_encoder: Whether to freeze encoder weights (default: False)
    
    Default configurations:
        - Large: 1024-dim, 24 layers, 16 heads (recommended)
        - Base: 768-dim, 12 layers, 12 heads
    """
    
    def __init__(
        self,
        checkpoint_name: str = 'large',
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        
        # Auto-detect checkpoint path
        if pretrained_path is None:
            checkpoint_root = Path(__file__).parent.parent.parent / "checkpoints"
            checkpoint_map = {
                'base': checkpoint_root / "CroCo_V2_ViTBase_BaseDecoder.pth",
                'large': checkpoint_root / "CroCo_V2_ViTLarge_BaseDecoder.pth",
            }
            if checkpoint_name not in checkpoint_map:
                raise ValueError(
                    f"checkpoint_name must be 'base' or 'large', got '{checkpoint_name}'"
                )
            pretrained_path = str(checkpoint_map[checkpoint_name])
        
        # Load checkpoint
        print(f"Loading CroCo-V2 encoder from: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location='cpu')
        
        # Get CroCo model arguments from checkpoint
        croco_args = croco_args_from_ckpt(ckpt)
        print(f"CroCo config: {croco_args}")
        
        # Create CroCo model
        self.croco = CroCoNet(**croco_args)
        
        # Load pretrained weights
        msg = self.croco.load_state_dict(ckpt['model'], strict=True)
        print(f"Loaded CroCo weights: {msg}")
        
        # Store encoder config
        self.patch_size = self.croco.patch_embed.patch_size[0]
        self.embed_dim = self.croco.enc_embed_dim
        self.depth = self.croco.enc_depth
        self.num_heads = croco_args.get('enc_num_heads', 12)
        self.img_size = croco_args.get('img_size', 224)
        
        # Freeze encoder if requested
        if freeze_encoder:
            self.freeze()
    
    def freeze(self):
        """Freeze all encoder parameters."""
        for param in self.croco.parameters():
            param.requires_grad = False
        print("CroCo encoder frozen!")
    
    def unfreeze(self):
        """Unfreeze all encoder parameters."""
        for param in self.croco.parameters():
            param.requires_grad = True
        print("CroCo encoder unfrozen!")
    
    def forward(
        self, 
        images: torch.Tensor,
        return_all_blocks: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CroCo encoder.
        
        Args:
            images: Input images [B, 3, H, W]
            return_all_blocks: If True, return features from all blocks
            
        Returns:
            features: Patch features [B, N, D] or list of [B, N, D] if return_all_blocks
            positions: Patch positions [B, N, 2] (y, x coordinates in patch units)
        """
        # Use CroCo's _encode_image method (no masking, encoder only)
        features, positions, _ = self.croco._encode_image(
            images, 
            do_mask=False,
            return_all_blocks=return_all_blocks
        )
        return features, positions
    
    @property
    def num_patches(self) -> int:
        """Number of patches per image."""
        return self.croco.patch_embed.num_patches
    
    @property
    def grid_size(self) -> Tuple[int, int]:
        """Patch grid size (H, W)."""
        return self.croco.patch_embed.grid_size


def build_croco_encoder(
    checkpoint_name: str = "large",
    pretrained_path: Optional[str] = None,
    freeze_encoder: bool = False,
) -> CroCoEncoder:
    """
    Build CroCo encoder with pretrained weights.
    
    Args:
        checkpoint_name: Checkpoint name ('base' or 'large', default: 'large')
        pretrained_path: Path to checkpoint (default: auto-detected)
        freeze_encoder: Whether to freeze encoder weights (default: False)
        
    Returns:
        encoder: CroCoEncoder instance
    
    Example:
        >>> # Load large CroCo encoder (recommended)
        >>> encoder = build_croco_encoder('large')
        >>> 
        >>> # Load base CroCo encoder
        >>> encoder = build_croco_encoder('base')
        >>> 
        >>> # Load with custom path
        >>> encoder = build_croco_encoder(pretrained_path='/path/to/checkpoint.pth')
    """
    encoder = CroCoEncoder(
        checkpoint_name=checkpoint_name,
        pretrained_path=pretrained_path,
        freeze_encoder=freeze_encoder,
    )
    return encoder


if __name__ == "__main__":
    # Add project root to path so we can import from external/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    sys.path.append(os.path.join(project_root, "external", "croco"))
    # Test encoder
    print("Testing CroCo Encoder Wrapper...")
    
    # Build large encoder (default)
    print("\n=== Testing Large Encoder ===")
    encoder = build_croco_encoder(checkpoint_name="large")
    
    print(f"Encoder config:")
    print(f"  - Embed dim: {encoder.embed_dim}")
    print(f"  - Depth: {encoder.depth} layers")
    print(f"  - Num heads: {encoder.num_heads}")
    print(f"  - Patch size: {encoder.patch_size}")
    print(f"  - Grid size: {encoder.grid_size}")
    print(f"  - Num patches: {encoder.num_patches}")
    
    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\nTesting forward pass...")
    print(f"Input: {images.shape}")
    
    features, positions = encoder(images)
    print(f"Features: {features.shape}")  # [B, N, D]
    print(f"Positions: {positions.shape}")  # [B, N, 2]
    
    # Test return_all_blocks
    print(f"\nTesting return_all_blocks...")
    all_features, positions = encoder(images, return_all_blocks=True)
    print(f"All blocks: {len(all_features)} blocks")
    for i in [0, len(all_features)//2, -1]:
        print(f"  Block {i}: {all_features[i].shape}")
    
    # Test freeze/unfreeze
    print(f"\nTesting freeze/unfreeze...")
    encoder.freeze()
    print(f"Frozen: {not any(p.requires_grad for p in encoder.croco.parameters())}")
    encoder.unfreeze()
    print(f"Unfrozen: {any(p.requires_grad for p in encoder.croco.parameters())}")
    
    print("\n✓ Encoder test passed!")
