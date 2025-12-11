"""
Generic Encoder Wrapper for VPOG

Supports multiple encoder backbones:
- CroCo-V2 (from external/croco)
- DINOv2 (from torch.hub or local)

This wrapper provides a unified interface regardless of the underlying encoder,
ensuring consistent output format: [B, N, D] features and [B, N, 2] positions.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

# Import CroCo from external/croco
from external.croco.models.croco import CroCoNet
from external.croco.models.croco_downstream import croco_args_from_ckpt


class EncoderWrapper(nn.Module):
    """
    Generic encoder wrapper supporting multiple backbones.
    
    Provides unified interface:
    - Input: [B, 3, H, W] images
    - Output: [B, N, D] features, [B, N, 2] positions
    
    Args:
        encoder_type: Type of encoder ('croco' or 'dinov2')
        **kwargs: Encoder-specific configuration
    """
    
    def __init__(
        self,
        encoder_type: str = 'croco',
        **kwargs,
    ):
        super().__init__()
        
        self.encoder_type = encoder_type.lower()
        
        if self.encoder_type == 'croco':
            self._build_croco_encoder(**kwargs)
        elif self.encoder_type == 'dinov2':
            self._build_dinov2_encoder(**kwargs)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Choose 'croco' or 'dinov2'")
    
    def _build_croco_encoder(
        self,
        checkpoint_name: str = 'large',
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = False,
        **kwargs,
    ):
        """Build CroCo-V2 encoder."""
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
        self.encoder = CroCoNet(**croco_args)
        
        # Load pretrained weights
        msg = self.encoder.load_state_dict(ckpt['model'], strict=True)
        print(f"Loaded CroCo weights: {msg}")
        
        # Store encoder config
        self.patch_size = self.encoder.patch_embed.patch_size[0]
        self.embed_dim = self.encoder.enc_embed_dim
        self.depth = self.encoder.enc_depth
        self.num_heads = croco_args.get('enc_num_heads', 12)
        self.img_size = croco_args.get('img_size', 224)
        
        # Freeze encoder if requested
        if freeze_encoder:
            self.freeze()
    
    def _build_dinov2_encoder(
        self,
        model_name: str = 'dinov2_vitl14',
        pretrained: bool = True,
        freeze_encoder: bool = False,
        patch_size: int = 14,
        img_size: int = 224,
        **kwargs,
    ):
        """Build DINOv2 encoder."""
        print(f"Loading DINOv2 encoder: {model_name}")
        
        try:
            # Try to load from torch.hub
            self.encoder = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=pretrained)
        except Exception as e:
            print(f"Failed to load from torch.hub: {e}")
            raise RuntimeError(
                f"Could not load DINOv2 model '{model_name}'. "
                "Make sure you have internet connection or the model is cached."
            )
        
        # Store encoder config
        self.patch_size = patch_size
        self.img_size = img_size
        
        # Get embed_dim from model
        if hasattr(self.encoder, 'embed_dim'):
            self.embed_dim = self.encoder.embed_dim
        elif hasattr(self.encoder, 'num_features'):
            self.embed_dim = self.encoder.num_features
        else:
            # Default dims for DINOv2
            dim_map = {
                'dinov2_vits14': 384,
                'dinov2_vitb14': 768,
                'dinov2_vitl14': 1024,
                'dinov2_vitg14': 1536,
            }
            self.embed_dim = dim_map.get(model_name, 768)
        
        # Get depth and num_heads if available
        self.depth = getattr(self.encoder, 'n_blocks', 12)
        self.num_heads = getattr(self.encoder, 'num_heads', 12)
        
        print(f"DINOv2 config: dim={self.embed_dim}, depth={self.depth}, heads={self.num_heads}")
        
        # Freeze encoder if requested
        if freeze_encoder:
            self.freeze()
    
    def freeze(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print(f"{self.encoder_type.upper()} encoder frozen!")
    
    def unfreeze(self):
        """Unfreeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print(f"{self.encoder_type.upper()} encoder unfrozen!")
    
    def forward(
        self, 
        images: torch.Tensor,
        return_all_blocks: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            images: Input images [B, 3, H, W]
            return_all_blocks: If True, return features from all blocks
            
        Returns:
            features: Patch features [B, N, D] or list of [B, N, D] if return_all_blocks
            positions: Patch positions [B, N, 2] (y, x coordinates in patch units)
        """
        if self.encoder_type == 'croco':
            return self._forward_croco(images, return_all_blocks)
        elif self.encoder_type == 'dinov2':
            return self._forward_dinov2(images, return_all_blocks)
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")
    
    def _forward_croco(
        self,
        images: torch.Tensor,
        return_all_blocks: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through CroCo encoder."""
        # Use CroCo's _encode_image method (no masking, encoder only)
        features, positions, _ = self.encoder._encode_image(
            images, 
            do_mask=False,
            return_all_blocks=return_all_blocks
        )
        return features, positions
    
    def _forward_dinov2(
        self,
        images: torch.Tensor,
        return_all_blocks: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through DINOv2 encoder."""
        B, C, H, W = images.shape
        
        # DINOv2 forward pass
        if return_all_blocks:
            # Get intermediate features from all blocks
            features = self.encoder.get_intermediate_layers(images, n=self.depth)
            # Each element is [B, N+1, D] (includes CLS token)
            # Remove CLS token and return list
            features = [f[:, 1:, :] for f in features]
        else:
            # Standard forward pass
            features = self.encoder.forward_features(images)
            # Remove CLS token: [B, N+1, D] -> [B, N, D]
            if features.shape[1] > (H // self.patch_size) * (W // self.patch_size):
                features = features[:, 1:, :]  # Remove CLS token
        
        # Generate position grid [B, N, 2]
        H_p, W_p = H // self.patch_size, W // self.patch_size
        N = H_p * W_p
        
        # Create position grid (y, x) in patch coordinates
        y_pos = torch.arange(H_p, device=images.device).repeat_interleave(W_p)
        x_pos = torch.arange(W_p, device=images.device).repeat(H_p)
        positions = torch.stack([y_pos, x_pos], dim=-1).float()  # [N, 2]
        positions = positions.unsqueeze(0).expand(B, -1, -1)  # [B, N, 2]
        
        return features, positions
    
    @property
    def num_patches(self) -> int:
        """Number of patches per image."""
        return (self.img_size // self.patch_size) ** 2
    
    @property
    def grid_size(self) -> Tuple[int, int]:
        """Patch grid size (H, W)."""
        size = self.img_size // self.patch_size
        return (size, size)


def build_encoder(
    encoder_type: str = 'croco',
    **kwargs,
) -> EncoderWrapper:
    """
    Build encoder wrapper with specified backend.
    
    Args:
        encoder_type: Type of encoder ('croco' or 'dinov2')
        **kwargs: Encoder-specific configuration
        
    Returns:
        encoder: EncoderWrapper instance
    
    Example:
        >>> # CroCo-V2 Large
        >>> encoder = build_encoder('croco', checkpoint_name='large')
        >>> 
        >>> # DINOv2 Large
        >>> encoder = build_encoder('dinov2', model_name='dinov2_vitl14')
    """
    return EncoderWrapper(encoder_type=encoder_type, **kwargs)


if __name__ == "__main__":
    # Test encoder wrapper
    print("Testing Encoder Wrapper...")
    
    # Test CroCo
    print("\n=== Testing CroCo Encoder ===")
    croco_encoder = build_encoder('croco', checkpoint_name='large')
    print(f"Embed dim: {croco_encoder.embed_dim}")
    print(f"Patch size: {croco_encoder.patch_size}")
    print(f"Grid size: {croco_encoder.grid_size}")
    
    images = torch.randn(2, 3, 224, 224)
    features, positions = croco_encoder(images)
    print(f"Features: {features.shape}")
    print(f"Positions: {positions.shape}")
    
    # Test DINOv2 (skip if no internet)
    # print("\n=== Testing DINOv2 Encoder ===")
    # dinov2_encoder = build_encoder('dinov2', model_name='dinov2_vitl14')
    # features, positions = dinov2_encoder(images)
    # print(f"Features: {features.shape}")
    # print(f"Positions: {positions.shape}")
    
    print("\n✓ Encoder wrapper test passed!")
