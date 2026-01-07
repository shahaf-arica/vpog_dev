"""
VPOG Model Builder
Factory for building VPOG model components from Hydra configs
"""

from typing import Optional
import torch.nn as nn
from omegaconf import DictConfig
from hydra.utils import instantiate


def build_vpog_model(cfg: DictConfig) -> nn.Module:
    """Build VPOG model from Hydra config.
    
    Args:
        cfg: Model configuration from Hydra
        
    Returns:
        VPOGModel instance
    """
    # Simply use Hydra's instantiate - the config should have _target_ defined
    return instantiate(cfg)


def build_croco_encoder(
    checkpoint_path: str,
    patch_size: int = 16,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
) -> nn.Module:
    """Build CroCo encoder backbone.
    
    Args:
        checkpoint_path: Path to CroCo checkpoint
        patch_size: Patch size for vision transformer
        embed_dim: Embedding dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        
    Returns:
        CroCo encoder module
    """
    from external.croco.models.croco import CroCoNet
    import torch
    
    # Load CroCo model
    model = CroCoNet(
        output_mode='dense',
        head_type='linear',
        patch_embed_cls='PatchEmbedDust3R',
        patch_size=patch_size,
        enc_embed_dim=embed_dim,
        enc_depth=depth,
        enc_num_heads=num_heads,
    )
    
    # Load pretrained weights
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
    
    return model.encoder
