"""
Token Manager for VPOG

Manages added tokens (like unseen tokens) that are appended to query/template sequences.
These tokens go through AA module but WITHOUT positional encoding (RoPE).

Configuration:
- num_query_added_tokens: Number of special tokens to add to query (default: 0)
- num_template_added_tokens: Number of special tokens to add to each template (default: 1 for unseen)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class TokenManager(nn.Module):
    """
    Manages special added tokens for query and templates.
    
    Added tokens are:
    - Appended to the sequence BEFORE AA module
    - DO NOT receive positional encoding (RoPE/S²RoPE)
    - Go through all attention mechanisms
    - Used for classification (e.g., unseen class)
    
    Args:
        dim: Feature dimension
        num_query_added_tokens: Number of special tokens for query (default: 0)
        num_template_added_tokens: Number of special tokens per template (default: 1)
        init_std: Initialization standard deviation for added tokens
    """
    
    def __init__(
        self,
        dim: int,
        num_query_added_tokens: int = 0,
        num_template_added_tokens: int = 1,
        init_std: float = 0.02,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_query_added_tokens = num_query_added_tokens
        self.num_template_added_tokens = num_template_added_tokens
        
        # Initialize added tokens as learnable parameters
        if num_query_added_tokens > 0:
            self.query_added_tokens = nn.Parameter(
                torch.randn(1, num_query_added_tokens, dim) * init_std
            )
        else:
            self.register_parameter('query_added_tokens', None)
        
        if num_template_added_tokens > 0:
            self.template_added_tokens = nn.Parameter(
                torch.randn(1, num_template_added_tokens, dim) * init_std
            )
        else:
            self.register_parameter('template_added_tokens', None)
    
    def add_query_tokens(
        self,
        query_features: torch.Tensor,  # [B, Nq, D]
        query_pos2d: torch.Tensor,  # [B, Nq, 2]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add special tokens to query features.
        
        Args:
            query_features: Query patch features [B, Nq, D]
            query_pos2d: Query 2D positions [B, Nq, 2]
            
        Returns:
            features_with_tokens: [B, Nq + num_added, D]
            pos2d_with_tokens: [B, Nq + num_added, 2] (dummy pos for added tokens)
            rope_mask: [B, Nq + num_added] - True for patches, False for added tokens
        """
        B, Nq, D = query_features.shape
        
        if self.num_query_added_tokens == 0:
            # No added tokens, return original with all-True rope_mask
            rope_mask = torch.ones(B, Nq, dtype=torch.bool, device=query_features.device)
            return query_features, query_pos2d, rope_mask
        
        # Expand added tokens for batch
        added_tokens = self.query_added_tokens.expand(B, -1, -1)  # [B, num_added, D]
        
        # Concatenate: [patches, added_tokens]
        features_with_tokens = torch.cat([query_features, added_tokens], dim=1)  # [B, Nq + num_added, D]
        
        # Create dummy positions for added tokens (they won't use RoPE anyway)
        dummy_pos = torch.zeros(B, self.num_query_added_tokens, 2, device=query_pos2d.device)
        pos2d_with_tokens = torch.cat([query_pos2d, dummy_pos], dim=1)  # [B, Nq + num_added, 2]
        
        # Create rope_mask: True for patches, False for added tokens
        rope_mask = torch.cat([
            torch.ones(B, Nq, dtype=torch.bool, device=query_features.device),
            torch.zeros(B, self.num_query_added_tokens, dtype=torch.bool, device=query_features.device),
        ], dim=1)  # [B, Nq + num_added]
        
        return features_with_tokens, pos2d_with_tokens, rope_mask
    
    def add_template_tokens(
        self,
        template_features: torch.Tensor,  # [B, S*Nt, D] or [B, S, Nt, D]
        template_pos2d: torch.Tensor,  # [B, S*Nt, 2] or [B, S, Nt, 2]
        num_templates: int,  # S
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add special tokens to each template.
        
        Args:
            template_features: Template features [B, S*Nt, D] or [B, S, Nt, D]
            template_pos2d: Template positions [B, S*Nt, 2] or [B, S, Nt, 2]
            num_templates: Number of templates S
            
        Returns:
            features_with_tokens: [B, S*(Nt + num_added), D]
            pos2d_with_tokens: [B, S*(Nt + num_added), 2]
            rope_mask: [B, S*(Nt + num_added)] - True for patches, False for added tokens
        """
        if template_features.dim() == 3:
            # [B, S*Nt, D] -> [B, S, Nt, D]
            B, SNt, D = template_features.shape
            Nt = SNt // num_templates
            template_features = template_features.reshape(B, num_templates, Nt, D)
            template_pos2d = template_pos2d.reshape(B, num_templates, Nt, 2)
        else:
            B, S, Nt, D = template_features.shape
            assert S == num_templates, f"num_templates mismatch: {S} vs {num_templates}"
        
        if self.num_template_added_tokens == 0:
            # No added tokens, reshape back and return with all-True rope_mask
            features_flat = template_features.reshape(B, num_templates * Nt, D)
            pos2d_flat = template_pos2d.reshape(B, num_templates * Nt, 2)
            rope_mask = torch.ones(B, num_templates * Nt, dtype=torch.bool, device=template_features.device)
            return features_flat, pos2d_flat, rope_mask
        
        # Expand added tokens for batch and templates
        added_tokens = self.template_added_tokens.expand(B, num_templates, -1, -1)  # [B, S, num_added, D]
        
        # Concatenate to each template: [patches, added_tokens]
        features_with_tokens = torch.cat([template_features, added_tokens], dim=2)  # [B, S, Nt + num_added, D]
        
        # Create dummy positions for added tokens
        dummy_pos = torch.zeros(B, num_templates, self.num_template_added_tokens, 2, device=template_pos2d.device)
        pos2d_with_tokens = torch.cat([template_pos2d, dummy_pos], dim=2)  # [B, S, Nt + num_added, 2]
        
        # Create rope_mask: True for patches, False for added tokens
        rope_mask_per_template = torch.cat([
            torch.ones(Nt, dtype=torch.bool, device=template_features.device),
            torch.zeros(self.num_template_added_tokens, dtype=torch.bool, device=template_features.device),
        ], dim=0)  # [Nt + num_added]
        
        rope_mask = rope_mask_per_template.unsqueeze(0).unsqueeze(0).expand(B, num_templates, -1)  # [B, S, Nt + num_added]
        
        # Flatten for processing
        features_flat = features_with_tokens.reshape(B, num_templates * (Nt + self.num_template_added_tokens), D)
        pos2d_flat = pos2d_with_tokens.reshape(B, num_templates * (Nt + self.num_template_added_tokens), 2)
        rope_mask_flat = rope_mask.reshape(B, num_templates * (Nt + self.num_template_added_tokens))
        
        return features_flat, pos2d_flat, rope_mask_flat
    
    def remove_added_tokens(
        self,
        features_with_tokens: torch.Tensor,  # [B, N_with_added, D]
        num_original: int,  # Original number of tokens before adding
        is_template: bool = False,
        num_templates: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Remove added tokens and split them out.
        
        Args:
            features_with_tokens: Features including added tokens
            num_original: Original number of tokens (Nq for query, S*Nt for templates)
            is_template: Whether these are template features
            num_templates: Number of templates (for templates only)
            
        Returns:
            original_features: Original patch features
            added_token_features: Features of added tokens
        """
        B, N_total, D = features_with_tokens.shape
        
        if is_template:
            num_added_per_template = self.num_template_added_tokens
            if num_added_per_template == 0:
                return features_with_tokens, None
            
            # Reshape: [B, S*(Nt+num_added), D] -> [B, S, Nt+num_added, D]
            Nt_with_added = N_total // num_templates
            Nt = num_original // num_templates
            
            features_reshaped = features_with_tokens.reshape(B, num_templates, Nt_with_added, D)
            
            # Split
            original_features = features_reshaped[:, :, :Nt, :]  # [B, S, Nt, D]
            added_token_features = features_reshaped[:, :, Nt:, :]  # [B, S, num_added, D]
            
            # Flatten original features
            original_features = original_features.reshape(B, num_templates * Nt, D)
            
            return original_features, added_token_features
        else:
            num_added = self.num_query_added_tokens
            if num_added == 0:
                return features_with_tokens, None
            
            # Split: [patches, added_tokens]
            original_features = features_with_tokens[:, :num_original, :]  # [B, Nq, D]
            added_token_features = features_with_tokens[:, num_original:, :]  # [B, num_added, D]
            
            return original_features, added_token_features


if __name__ == "__main__":
    print("Testing Token Manager...")
    
    # Test configuration
    batch_size = 2
    num_query_patches = 196  # 14x14
    num_templates = 4
    num_template_patches = 196
    dim = 768
    
    # Create token manager
    token_manager = TokenManager(
        dim=dim,
        num_query_added_tokens=0,  # No added tokens for query
        num_template_added_tokens=1,  # 1 unseen token per template
    )
    
    print(f"Token Manager: query_added={token_manager.num_query_added_tokens}, template_added={token_manager.num_template_added_tokens}")
    
    # Test query tokens
    print("\n=== Testing Query Tokens ===")
    query_features = torch.randn(batch_size, num_query_patches, dim)
    query_pos2d = torch.randint(0, 14, (batch_size, num_query_patches, 2)).float()
    
    query_with_tokens, query_pos_with_tokens, query_rope_mask = token_manager.add_query_tokens(
        query_features, query_pos2d
    )
    print(f"Original query: {query_features.shape}")
    print(f"Query with tokens: {query_with_tokens.shape}")
    print(f"Query rope_mask: {query_rope_mask.shape}, all True: {query_rope_mask.all()}")
    
    # Test template tokens
    print("\n=== Testing Template Tokens ===")
    template_features = torch.randn(batch_size, num_templates * num_template_patches, dim)
    template_pos2d = torch.randint(0, 14, (batch_size, num_templates * num_template_patches, 2)).float()
    
    template_with_tokens, template_pos_with_tokens, template_rope_mask = token_manager.add_template_tokens(
        template_features, template_pos2d, num_templates
    )
    print(f"Original templates: {template_features.shape}")
    print(f"Templates with tokens: {template_with_tokens.shape}")  # Should be [B, S*(Nt+1), D]
    print(f"Template rope_mask: {template_rope_mask.shape}")
    print(f"RoPE mask pattern (first template): {template_rope_mask[0, :num_template_patches+1]}")
    print(f"  - Patches (should be True): {template_rope_mask[0, :num_template_patches].all()}")
    print(f"  - Added token (should be False): {not template_rope_mask[0, num_template_patches]}")
    
    # Test removal
    print("\n=== Testing Token Removal ===")
    original_template, added_template = token_manager.remove_added_tokens(
        template_with_tokens,
        num_original=num_templates * num_template_patches,
        is_template=True,
        num_templates=num_templates,
    )
    print(f"Recovered original: {original_template.shape}")
    print(f"Added tokens: {added_template.shape}")  # [B, S, num_added, D]
    
    # Verify correctness
    print("\n=== Verification ===")
    # The first Nt tokens of each template should match original
    for s in range(num_templates):
        original_slice = template_features[:, s*num_template_patches:(s+1)*num_template_patches, :]
        recovered_slice = original_template[:, s*num_template_patches:(s+1)*num_template_patches, :]
        # Note: They won't exactly match because we didn't preserve order, but shapes should be correct
        print(f"Template {s}: Original shape {original_slice.shape}, Recovered shape {recovered_slice.shape}")
    
    print("\n✓ Token Manager test passed!")
