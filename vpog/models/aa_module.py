"""
Attention Aggregation (AA) Module for VPOG

Implements global-local attention mechanism inspired by VGGT paper.
- Global attention: Full attention with S²RoPE for templates, RoPE2D for query
- Local attention: Windowed attention with RoPE2D only
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add external/croco to path for RoPE2D
croco_path = Path(__file__).parent.parent.parent / "external" / "croco"
if str(croco_path) not in sys.path:
    sys.path.insert(0, str(croco_path))

from external.croco.models.pos_embed import RoPE2D

# Import S²RoPE from vpog
from vpog.models.s2rope.s2rope_module import S2RoPE


class GlobalAttention(nn.Module):
    """
    Global attention with different positional encodings for templates vs query.
    
    - Templates: S²RoPE (spherical encoding for viewpoints)
    - Query: RoPE2D (2D spatial encoding)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope_freq: float = 100.0,
        n_faces: int = 6,
        s2rope_px: int = None,
        s2rope_py: int = None,
        s2rope_ps: int = None,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        # QKV projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Positional encodings
        self.rope2d = RoPE2D(freq=rope_freq)  # For query
        self.s2rope = S2RoPE(head_dim=self.head_dim, n_faces=n_faces, px=s2rope_px, py=s2rope_py, ps=s2rope_ps)  # For templates
    
    def forward(
        self,
        x: torch.Tensor,
        pos2d: Optional[torch.Tensor] = None,  # [B, N, 2] for query patches
        phase_x: Optional[torch.Tensor] = None,  # [B, N, px] for templates
        phase_y: Optional[torch.Tensor] = None,  # [B, N, py] for templates
        phase_sph: Optional[torch.Tensor] = None,  # [B, N, F, ps] for templates
        is_template: bool = False,  # Whether this is template or query tokens
        rope_mask: Optional[torch.Tensor] = None,  # [B, N] - True = apply RoPE, False = skip RoPE
    ) -> torch.Tensor:
        """
        Args:
            x: Input tokens [B, N, D]
            pos2d: 2D positions for RoPE2D [B, N, 2] (for query)
            phase_x, phase_y, phase_sph: Phase encodings for S²RoPE (for templates)
            is_template: Whether tokens are template (use S²RoPE) or query (use RoPE2D)
            rope_mask: [B, N] mask indicating which tokens get RoPE (True) or skip it (False)
                       Used for unseen tokens that should not receive positional encoding
            
        Returns:
            out: Attention output [B, N, D]
        """
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D_h]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, H, N, D_h]
        
        # Apply positional encoding (respecting rope_mask for unseen tokens)
        if is_template:
            # Apply S²RoPE to templates
            # S²RoPE expects [B, N, H, D_h], we have [B, H, N, D_h]
            q = q.transpose(1, 2)  # [B, N, H, D_h]
            k = k.transpose(1, 2)
            
            if rope_mask is not None:
                # Apply S²RoPE only to masked tokens, keep others unchanged
                q_rope = self.s2rope(q, phase_x, phase_y, phase_sph)
                k_rope = self.s2rope(k, phase_x, phase_y, phase_sph)
                # rope_mask: [B, N] -> [B, N, 1, 1]
                mask = rope_mask.unsqueeze(-1).unsqueeze(-1)
                q = torch.where(mask, q_rope, q)
                k = torch.where(mask, k_rope, k)
            else:
                q = self.s2rope(q, phase_x, phase_y, phase_sph)
                k = self.s2rope(k, phase_x, phase_y, phase_sph)
            
            q = q.transpose(1, 2)  # [B, H, N, D_h]
            k = k.transpose(1, 2)
        else:
            # Apply RoPE2D to query
            # RoPE2D expects [B, H, N, D_h], pos2d: [B, N, 2]
            if pos2d is not None:
                if rope_mask is not None:
                    # Apply RoPE2D only to masked tokens
                    q_rope = self.rope2d(q, pos2d)
                    k_rope = self.rope2d(k, pos2d)
                    # rope_mask: [B, N] -> [B, 1, N, 1]
                    mask = rope_mask.unsqueeze(1).unsqueeze(-1)
                    q = torch.where(mask, q_rope, q)
                    k = torch.where(mask, k_rope, k)
                else:
                    q = self.rope2d(q, pos2d)
                    k = self.rope2d(k, pos2d)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, D]
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class LocalAttention(nn.Module):
    """
    Windowed local attention with RoPE2D only.
    
    Both templates and query use RoPE2D in local attention.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 7,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope_freq: float = 100.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        # QKV projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # RoPE2D for both templates and query
        self.rope2d = RoPE2D(freq=rope_freq)
    
    def window_partition(self, x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Partition tokens into non-overlapping windows.
        
        Args:
            x: [B, H, W, C]
            window_size: Window size
            
        Returns:
            windows: [B * num_windows, window_size, window_size, C]
            (H_pad, W_pad): Padded height and width
        """
        B, H, W, C = x.shape
        
        # Pad if needed
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        H_pad, W_pad = H + pad_h, W + pad_w
        
        # Partition
        x = x.view(B, H_pad // window_size, window_size, W_pad // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        
        return windows, (H_pad, W_pad)
    
    def window_reverse(self, windows: torch.Tensor, window_size: int, H_pad: int, W_pad: int, H: int, W: int) -> torch.Tensor:
        """
        Reverse window partition.
        
        Args:
            windows: [B * num_windows, window_size, window_size, C]
            window_size: Window size
            H_pad, W_pad: Padded dimensions
            H, W: Original dimensions
            
        Returns:
            x: [B, H, W, C]
        """
        B_total = windows.shape[0]
        nH = H_pad // window_size
        nW = W_pad // window_size
        B = B_total // (nH * nW)
        
        x = windows.view(B, nH, nW, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H_pad, W_pad, -1)
        
        # Remove padding
        if H_pad > H or W_pad > W:
            x = x[:, :H, :W, :].contiguous()
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        pos2d: torch.Tensor,  # [B, N, 2]
        grid_size: Tuple[int, int],  # (H, W)
        rope_mask: Optional[torch.Tensor] = None,  # [B, N] - True = apply RoPE, False = skip
        num_added_tokens: int = 0,  # Number of added tokens that don't form spatial grid
    ) -> torch.Tensor:
        """
        Args:
            x: Input tokens [B, N, D] where N = H*W + num_added_tokens
            pos2d: 2D positions [B, N, 2]
            grid_size: (H, W) spatial dimensions
            rope_mask: [B, N] mask for which tokens get RoPE (False for added tokens)
            num_added_tokens: Number of non-spatial tokens (e.g., unseen tokens)
            
        Returns:
            out: Attention output [B, N, D]
        """
        B, N, C = x.shape
        H, W = grid_size
        expected_N = H * W + num_added_tokens
        assert N == expected_N, f"N={N} must equal H*W + num_added={expected_N}"
        
        if num_added_tokens > 0:
            # Separate spatial tokens from added tokens
            x_spatial = x[:, :H*W, :]  # [B, H*W, C]
            x_added = x[:, H*W:, :]  # [B, num_added, C]
            pos2d_spatial = pos2d[:, :H*W, :]  # [B, H*W, 2]
            # pos2d for added tokens is just placeholder, won't be used with RoPE
            
            # rope_mask for spatial and added tokens
            rope_mask_spatial = rope_mask[:, :H*W] if rope_mask is not None else None
            # Note: rope_mask_added should always be False (no RoPE for added tokens)
            
            # Process spatial tokens with windowed local attention
            x_2d = x_spatial.view(B, H, W, C)  # [B, H, W, C]
            pos_2d = pos2d_spatial.view(B, H, W, 2)  # [B, H, W, 2]
        else:
            # No added tokens, standard processing
            x_2d = x.view(B, H, W, C)  # [B, H, W, C]
            pos_2d = pos2d.view(B, H, W, 2)  # [B, H, W, 2]
            rope_mask_spatial = rope_mask
        
        # Partition into windows
        x_windows, (H_pad, W_pad) = self.window_partition(x_2d, self.window_size)  # [B*nW, ws, ws, C]
        pos_windows, _ = self.window_partition(pos_2d, self.window_size)  # [B*nW, ws, ws, 2]
        
        nW = x_windows.shape[0] // B
        ws = self.window_size
        
        # Flatten windows
        x_win = x_windows.view(-1, ws * ws, C)  # [B*nW, ws*ws, C]
        pos_win = pos_windows.view(-1, ws * ws, 2)  # [B*nW, ws*ws, 2]
        
        # Convert continuous positions to discrete grid indices for RoPE
        # pos_win is in normalized coordinates [0, 1] or pixel coordinates
        # We need integer grid positions for RoPE embedding
        # Assuming pos_win is in range [0, H-1] x [0, W-1] for a window
        pos_win_int = pos_win.long()  # Convert to integer indices
        
        # QKV projection
        qkv = self.qkv(x_win).reshape(-1, ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*nW, H, ws*ws, D_h]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B*nW, H, ws*ws, D_h]
        
        # Apply RoPE2D (respecting rope_mask if provided)
        if rope_mask_spatial is not None:
            # Reshape rope_mask for windows: [B, H*W] -> [B, H, W] -> windows -> [B*nW, ws*ws]
            rope_mask_2d = rope_mask_spatial.view(B, H, W)
            rope_mask_windows, _ = self.window_partition(rope_mask_2d.unsqueeze(-1).float(), self.window_size)
            rope_mask_win = rope_mask_windows.squeeze(-1).view(-1, ws * ws).bool()
            
            q_rope = self.rope2d(q, pos_win_int)
            k_rope = self.rope2d(k, pos_win_int)
            # rope_mask_win: [B*nW, ws*ws] -> [B*nW, 1, ws*ws, 1]
            mask = rope_mask_win.unsqueeze(1).unsqueeze(-1)
            q = torch.where(mask, q_rope, q)
            k = torch.where(mask, k_rope, k)
        else:
            q = self.rope2d(q, pos_win_int)
            k = self.rope2d(k, pos_win_int)
        
        # Attention within windows
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*nW, H, ws*ws, ws*ws]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x_win = (attn @ v).transpose(1, 2).reshape(-1, ws * ws, C)  # [B*nW, ws*ws, C]
        x_win = self.proj(x_win)
        x_win = self.proj_drop(x_win)
        
        # Reshape back to windows
        x_windows = x_win.view(-1, ws, ws, C)
        
        # Reverse window partition
        x_2d = self.window_reverse(x_windows, ws, H_pad, W_pad, H, W)  # [B, H, W, C]
        
        # Flatten back
        x_spatial_out = x_2d.view(B, H * W, C)  # [B, H*W, C]
        
        if num_added_tokens > 0:
            # For added tokens, apply self-attention with spatial tokens
            # Added tokens attend to all spatial tokens (global context within the template)
            # This allows unseen token to gather information but doesn't apply RoPE to it
            
            # x_added was already extracted at the beginning
            # QKV for added tokens
            qkv_added = self.qkv(x_added).reshape(B, num_added_tokens, 3, self.num_heads, self.head_dim)
            qkv_added = qkv_added.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, num_added, head_dim]
            q_added, k_added, v_added = qkv_added[0], qkv_added[1], qkv_added[2]
            
            # Keys and values from spatial tokens (already processed)
            # We need to project spatial tokens to get K, V for cross-attention
            # Recompute QKV for spatial tokens
            qkv_spatial = self.qkv(x_spatial_out).reshape(B, H*W, 3, self.num_heads, self.head_dim)
            qkv_spatial = qkv_spatial.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, H*W, head_dim]
            q_spatial, k_spatial, v_spatial = qkv_spatial[0], qkv_spatial[1], qkv_spatial[2]
            
            # Added tokens query, spatial tokens provide keys/values
            # Cross-attention: added tokens attend to spatial tokens
            attn_added = (q_added @ k_spatial.transpose(-2, -1)) * self.scale  # [B, num_heads, num_added, H*W]
            attn_added = attn_added.softmax(dim=-1)
            attn_added = self.attn_drop(attn_added)
            
            x_added_out = (attn_added @ v_spatial).transpose(1, 2).reshape(B, num_added_tokens, C)
            x_added_out = self.proj(x_added_out)
            x_added_out = self.proj_drop(x_added_out)
            
            # Also allow spatial tokens to attend to added tokens (bidirectional)
            attn_spatial_to_added = (q_spatial @ k_added.transpose(-2, -1)) * self.scale  # [B, num_heads, H*W, num_added]
            attn_spatial_to_added = attn_spatial_to_added.softmax(dim=-1)
            attn_spatial_to_added = self.attn_drop(attn_spatial_to_added)
            
            update_from_added = (attn_spatial_to_added @ v_added).transpose(1, 2).reshape(B, H*W, C)
            update_from_added = self.proj(update_from_added)
            update_from_added = self.proj_drop(update_from_added)
            
            # Combine: spatial tokens get updates from both windowed attention and added tokens
            x_spatial_out = x_spatial_out + update_from_added * 0.5  # Scale down the cross-attention
            
            # Concatenate spatial and added tokens
            x = torch.cat([x_spatial_out, x_added_out], dim=1)  # [B, N, C]
        else:
            x = x_spatial_out
        
        return x


class AABlock(nn.Module):
    """
    Attention Aggregation block with global and local attention.
    
    Can be configured to use only global, only local, or both.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        window_size: int = 7,
        use_global: bool = True,
        use_local: bool = True,
        rope_freq: float = 100.0,
        n_faces: int = 6,
        s2rope_px: int = None,
        s2rope_py: int = None,
        s2rope_ps: int = None,
    ):
        super().__init__()
        
        self.dim = dim
        self.use_global = use_global
        self.use_local = use_local
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        if use_global and use_local:
            self.norm1_local = nn.LayerNorm(dim)
        
        # Attention layers
        if use_global:
            self.global_attn = GlobalAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                rope_freq=rope_freq,
                n_faces=n_faces,
                s2rope_px=s2rope_px,
                s2rope_py=s2rope_py,
                s2rope_ps=s2rope_ps,
            )
        
        if use_local:
            self.local_attn = LocalAttention(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                rope_freq=rope_freq,
            )
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        pos2d: Optional[torch.Tensor] = None,
        phase_x: Optional[torch.Tensor] = None,
        phase_y: Optional[torch.Tensor] = None,
        phase_sph: Optional[torch.Tensor] = None,
        is_template: bool = False,
        grid_size: Optional[Tuple[int, int]] = None,
        rope_mask: Optional[torch.Tensor] = None,  # [B, N] - mask for RoPE application
    ) -> torch.Tensor:
        """
        Args:
            x: Input tokens [B, N, D]
            pos2d: 2D positions [B, N, 2] (for query)
            phase_x, phase_y, phase_sph: S²RoPE phases (for templates)
            is_template: Whether tokens are templates or query
            grid_size: (H, W) for local attention
            rope_mask: [B, N] mask indicating which tokens get RoPE (for unseen tokens)
            
        Returns:
            out: Output tokens [B, N, D]
        """
        # Global attention
        if self.use_global:
            x = x + self.drop_path(
                self.global_attn(
                    self.norm1(x),
                    pos2d=pos2d,
                    phase_x=phase_x,
                    phase_y=phase_y,
                    phase_sph=phase_sph,
                    is_template=is_template,
                    rope_mask=rope_mask,
                )
            )
        
        # Local attention
        if self.use_local:
            assert grid_size is not None, "grid_size required for local attention"
            assert pos2d is not None, "pos2d required for local attention"
            
            # Check if we have added tokens (N != H*W)
            B, N, C = x.shape
            H, W = grid_size
            expected_N = H * W
            
            if N != expected_N:
                # We have added tokens (e.g., unseen tokens)
                # Added tokens participate in attention but don't form part of spatial grid
                num_added = N - expected_N
                assert num_added > 0, f"N={N} < H*W={expected_N}, this shouldn't happen"
                
                # Separate base tokens and added tokens
                x_base = x[:, :expected_N, :]  # [B, H*W, C]
                x_added = x[:, expected_N:, :]  # [B, num_added, C]
                pos2d_base = pos2d[:, :expected_N, :]  # [B, H*W, 2]
                
                # Added tokens get dummy positions (they won't receive RoPE anyway due to rope_mask)
                # Use position [0, 0] for added tokens as placeholder
                pos2d_added = torch.zeros(B, num_added, 2, device=pos2d.device, dtype=pos2d.dtype)
                
                # Apply normalization
                norm_x_base = self.norm1_local(x_base) if (self.use_global and self.use_local) else self.norm1(x_base)
                norm_x_added = self.norm1_local(x_added) if (self.use_global and self.use_local) else self.norm1(x_added)
                
                # Combine for attention (base tokens form spatial grid, added tokens attend globally)
                norm_x_combined = torch.cat([norm_x_base, norm_x_added], dim=1)  # [B, N, C]
                pos2d_combined = torch.cat([pos2d_base, pos2d_added], dim=1)  # [B, N, 2]
                
                # Apply local attention with extended grid that includes added tokens
                # The added tokens will participate in attention but won't receive RoPE (rope_mask=False for them)
                x_combined_out = self.local_attn(
                    norm_x_combined, 
                    pos2d_combined, 
                    grid_size, 
                    rope_mask=rope_mask,  # rope_mask should be False for added tokens
                    num_added_tokens=num_added
                )
                
                # Split back
                x_base_out = x_combined_out[:, :expected_N, :]
                x_added_out = x_combined_out[:, expected_N:, :]
                
                # Apply residual
                x = torch.cat([x_base + self.drop_path(x_base_out), x_added + self.drop_path(x_added_out)], dim=1)
            else:
                # No added tokens, apply local attention normally
                norm_x = self.norm1_local(x) if (self.use_global and self.use_local) else self.norm1(x)
                x = x + self.drop_path(
                    self.local_attn(norm_x, pos2d, grid_size, rope_mask=rope_mask)
                )
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class AAModule(nn.Module):
    """
    Full Attention Aggregation module with multiple AA blocks.
    """
    
    def __init__(
        self,
        dim: int,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.1,
        window_size: int = 7,
        use_global: bool = True,
        use_local: bool = True,
        rope_freq: float = 100.0,
        n_faces: int = 6,
        s2rope_px: int = None,
        s2rope_py: int = None,
        s2rope_ps: int = None,
    ):
        super().__init__()
        
        self.dim = dim
        self.depth = depth
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # AA blocks
        self.blocks = nn.ModuleList([
            AABlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
                window_size=window_size,
                use_global=use_global,
                use_local=use_local,
                rope_freq=rope_freq,
                n_faces=n_faces,
                s2rope_px=s2rope_px,
                s2rope_py=s2rope_py,
                s2rope_ps=s2rope_ps,
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        x: torch.Tensor,
        pos2d: Optional[torch.Tensor] = None,
        phase_x: Optional[torch.Tensor] = None,
        phase_y: Optional[torch.Tensor] = None,
        phase_sph: Optional[torch.Tensor] = None,
        is_template: bool = False,
        grid_size: Optional[Tuple[int, int]] = None,
        rope_mask: Optional[torch.Tensor] = None,  # [B, N] - mask for RoPE
    ) -> torch.Tensor:
        """
        Args:
            x: Input tokens [B, N, D]
            pos2d: 2D positions [B, N, 2]
            phase_x, phase_y, phase_sph: S²RoPE phases
            is_template: Whether tokens are templates
            grid_size: (H, W) for local attention
            rope_mask: [B, N] mask for which tokens get RoPE (for unseen tokens)
            
        Returns:
            out: Output tokens [B, N, D]
        """
        for blk in self.blocks:
            x = blk(
                x,
                pos2d=pos2d,
                phase_x=phase_x,
                phase_y=phase_y,
                phase_sph=phase_sph,
                is_template=is_template,
                grid_size=grid_size,
                rope_mask=rope_mask,
            )
        
        x = self.norm(x)
        return x


# Helper modules (copied from CroCo for standalone use)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class Mlp(nn.Module):
    """MLP as used in Vision Transformer."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


if __name__ == "__main__":
    print("Testing AA Module...")
    
    # Test configuration
    batch_size = 2
    num_templates = 4
    num_query_patches = 196  # 14x14
    dim = 768
    grid_size = (14, 14)
    
    # Build AA module
    aa_module = AAModule(
        dim=dim,
        depth=4,
        num_heads=12,
        window_size=7,
        use_global=True,
        use_local=True,
    )
    
    print(f"AA Module: {aa_module.depth} layers, {dim}D")
    
    # Test with query tokens (RoPE2D)
    print("\n=== Testing Query (RoPE2D) ===")
    query_tokens = torch.randn(batch_size, num_query_patches, dim)
    pos2d = torch.randint(0, 14, (batch_size, num_query_patches, 2))
    
    output = aa_module(query_tokens, pos2d=pos2d, is_template=False, grid_size=grid_size)
    print(f"Input: {query_tokens.shape}")
    print(f"Output: {output.shape}")
    
    # Test with template tokens (S²RoPE)
    print("\n=== Testing Templates (S²RoPE) ===")
    template_tokens = torch.randn(batch_size, num_templates * num_query_patches, dim)
    # Mock S²RoPE phases (simplified for testing)
    phase_x = torch.randn(batch_size, num_templates * num_query_patches, aa_module.blocks[0].global_attn.s2rope.px)
    phase_y = torch.randn(batch_size, num_templates * num_query_patches, aa_module.blocks[0].global_attn.s2rope.py)
    phase_sph = torch.randn(batch_size, num_templates * num_query_patches, 6, aa_module.blocks[0].global_attn.s2rope.ps)
    pos2d_template = torch.randint(0, 14, (batch_size, num_templates * num_query_patches, 2))
    
    # For templates, grid_size is per-template
    grid_size_template = (num_templates * 14, 14)  # Treat as extended spatial grid
    
    output = aa_module(
        template_tokens,
        pos2d=pos2d_template,
        phase_x=phase_x,
        phase_y=phase_y,
        phase_sph=phase_sph,
        is_template=True,
        grid_size=grid_size_template,
    )
    print(f"Input: {template_tokens.shape}")
    print(f"Output: {output.shape}")
    
    print("\n✓ AA Module test passed!")
