# """
# Attention Aggregation (AA) Module for VPOG

# Implements global-local attention mechanism inspired by VGGT paper.
# - Global attention: Full attention with S²RoPE for templates, RoPE2D for query
# - Local attention: Windowed attention with RoPE2D only
# """

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# RoPE2D from CroCo
from external.croco.models.pos_embed import RoPE2D

# Phase generator (we changed forward() to return phases)
from vpog.models.pos_embed import S2RopeSequencePositionalEncoding

# Compiled S2RoPE kernel for q/k rotation
from vpog.models.s2rope.s2rope_module import S2RoPE


@dataclass
class AAConfig:
    dim: int
    depth: int
    num_heads: int
    window_size: int
    use_s2rope: bool = True
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class GlobalSelfAttention(nn.Module):
    """
    Shared global self-attention module.
    Positional encoding is applied to q/k on image-token prefix only:
      - Query: RoPE2D always
      - Template: S²RoPE if enabled (global only), else RoPE2D
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float,
        proj_drop: float,
        use_s2rope: bool,
        rope2d: RoPE2D,
        s2rope_seq: S2RopeSequencePositionalEncoding,
        s2rope_kernel: S2RoPE,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_s2rope = use_s2rope
        self.rope2d = rope2d
        self.s2rope_seq = s2rope_seq
        self.s2rope_kernel = s2rope_kernel

    def _attend(self, x: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q,k,v: [B, H, N, Dh]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # [B, H, N, Dh]
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.dim)  # [B, N, C]
        out = self.proj_drop(self.proj(out))
        return out

    def _qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)          # [B, N, H, Dh]
        q = q.permute(0, 2, 1, 3).contiguous()  # [B, H, N, Dh]
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        return q, k, v

    def forward_query(self, x_q: torch.Tensor, pos2d: torch.Tensor, num_img: int) -> torch.Tensor:
        """
        x_q: [B, Nq, C], where Nq == num_img (query has only image tokens)
        pos2d: [B, num_img, 2]
        """
        q, k, v = self._qkv(x_q)  # [B, H, N, Dh]
        # Apply RoPE2D to all query tokens (they are all image tokens)
        q = self.rope2d(q, pos2d)
        k = self.rope2d(k, pos2d)
        return self._attend(x_q, q, k, v)

    def forward_template(
        self,
        x_t: torch.Tensor,                  # [B*S, Nt_total, C] (includes unseen)
        pos2d_img: torch.Tensor,            # [B*S, num_img, 2] for image-token prefix
        num_img: int,                       # HW
        # S² metadata (flattened to B*S where relevant)
        phase_x: Optional[torch.Tensor],     # [B*S, num_img, px] or None
        phase_y: Optional[torch.Tensor],     # [B*S, num_img, py] or None
        phase_s2: Optional[torch.Tensor],    # [B*S, num_img, F, ps] or None
        s2_mask: Optional[torch.Tensor],     # [B*S, num_img] bool or None
    ) -> torch.Tensor:
        """
        Template global self-attention over full sequence (image tokens + unseen),
        but positional encoding is applied ONLY on image-token prefix [0:num_img].
        """
        q, k, v = self._qkv(x_t)  # [B*S, H, N, Dh]
        # Split prefix vs tail (tail contains unseen token(s))
        q_img, q_tail = q[:, :, :num_img, :], q[:, :, num_img:, :]
        k_img, k_tail = k[:, :, :num_img, :], k[:, :, num_img:, :]

        if self.use_s2rope:
            # Apply S²RoPE to q/k for image tokens only.
            # S2RoPE kernel expects [B, N, H, Dh], not [B, H, N, Dh].
            q_img_bnHD = q_img.permute(0, 2, 1, 3).contiguous()  # [B*S, num_img, H, Dh]
            k_img_bnHD = k_img.permute(0, 2, 1, 3).contiguous()

            # Apply kernel (phases already computed for image tokens)
            # Masking: s2_mask is token-wise boolean on [B*S, num_img]
            q_img_bnHD = self.s2rope_kernel(q_img_bnHD, phase_x, phase_y, phase_s2, s2_mask)
            k_img_bnHD = self.s2rope_kernel(k_img_bnHD, phase_x, phase_y, phase_s2, s2_mask)

            q_img = q_img_bnHD.permute(0, 2, 1, 3).contiguous()  # [B*S, H, num_img, Dh]
            k_img = k_img_bnHD.permute(0, 2, 1, 3).contiguous()
        else:
            # Fallback: apply RoPE2D on template image tokens only (not unseen)
            q_img = self.rope2d(q_img, pos2d_img)
            k_img = self.rope2d(k_img, pos2d_img)

        # Stitch back
        q = torch.cat([q_img, q_tail], dim=2)
        k = torch.cat([k_img, k_tail], dim=2)

        return self._attend(x_t, q, k, v)


class LocalWindowAttention(nn.Module):
    """
    Windowed self-attention over image tokens (HW) with optional per-image "unseen" token.

    If unseen is provided:
      - unseen participates in every window (so it can bind to that template),
      - but unseen gets NO RoPE2D.
      - unseen outputs from all windows are reduced (mean) back into one token.

    Inputs:
      x_img:   [B, HW, C]
      pos2d:   [B, HW, 2]  (y,x)
      unseen:  None OR [B, C] OR [B, 1, C]

    Returns:
      out_img:    [B, HW, C]
      out_unseen: None OR [B, C]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        attn_drop: float,
        proj_drop: float,
        rope2d: "RoPE2D",
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope2d = rope2d

    def _qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [Bseq, Nseq, C]
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # [B, N, H, Dh]
        q = q.permute(0, 2, 1, 3).contiguous()  # [B, H, N, Dh]
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        return q, k, v

    def _attend(self, x: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # x: [Bseq, Nseq, C], q/k/v: [Bseq, H, Nseq, Dh]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # [Bseq, H, Nseq, Dh]
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.dim)  # [Bseq, Nseq, C]
        out = self.proj_drop(self.proj(out))
        return out

    def forward(
        self,
        x_img: torch.Tensor,                     # [B, HW, C]
        pos2d_img: torch.Tensor,                 # [B, HW, 2]
        grid_size: Tuple[int, int],              # (H, W)
        unseen: Optional[torch.Tensor] = None,   # None or [B, C] or [B, 1, C]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B, HW, C = x_img.shape
        H, W = grid_size
        ws = self.window_size

        assert HW == H * W, f"Expected HW==H*W, got {HW} vs {H}*{W}"
        assert H % ws == 0 and W % ws == 0, f"Grid {H}x{W} must be divisible by window_size={ws}"

        # Normalize unseen to [B,1,C] if provided
        if unseen is not None:
            if unseen.dim() == 2:
                unseen = unseen[:, None, :]  # [B,1,C]
            assert unseen.shape == (B, 1, C), f"unseen must be [B,C] or [B,1,C], got {tuple(unseen.shape)}"

        # Reshape to grid
        xg = x_img.view(B, H, W, C)
        pg = pos2d_img.view(B, H, W, 2)

        # Partition into windows
        # [B, H//ws, ws, W//ws, ws, C] -> [B, nWh, nWw, ws, ws, C]
        xw = xg.view(B, H // ws, ws, W // ws, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        pw = pg.view(B, H // ws, ws, W // ws, ws, 2).permute(0, 1, 3, 2, 4, 5).contiguous()

        nWh = H // ws
        nWw = W // ws
        Bwin = B * nWh * nWw
        Nwin = ws * ws

        # Flatten windows: [Bwin, Nwin, C] and [Bwin, Nwin, 2]
        xw = xw.view(Bwin, Nwin, C)
        pw = pw.view(Bwin, Nwin, 2)

        # Append unseen token to every window sequence
        if unseen is not None:
            # unseen: [B,1,C] -> [B,nWh,nWw,1,C] -> [Bwin,1,C]
            unseen_w = unseen.view(B, 1, 1, 1, C).expand(B, nWh, nWw, 1, C).contiguous()
            unseen_w = unseen_w.view(Bwin, 1, C)

            # concat along token axis
            xw = torch.cat([xw, unseen_w], dim=1)  # [Bwin, Nwin+1, C]

        # QKV
        q, k, v = self._qkv(xw)  # q/k/v: [Bwin, Hh, Nseq, Dh]
        Nseq = xw.shape[1]       # Nwin or Nwin+1

        # Apply RoPE2D ONLY to the first Nwin tokens (image tokens)
        q_img = q[:, :, :Nwin, :]
        k_img = k[:, :, :Nwin, :]
        q_tail = q[:, :, Nwin:, :]  # empty if unseen is None
        k_tail = k[:, :, Nwin:, :]

        q_img = self.rope2d(q_img, pw)
        k_img = self.rope2d(k_img, pw)

        q = torch.cat([q_img, q_tail], dim=2)
        k = torch.cat([k_img, k_tail], dim=2)

        # Attention
        out = self._attend(xw, q, k, v)  # [Bwin, Nseq, C]

        if unseen is None:
            # Reconstruct image tokens
            out_img = out.view(B, nWh, nWw, ws, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous()
            out_img = out_img.view(B, H, W, C).view(B, HW, C)
            return out_img, None

        # Split outputs: image tokens and unseen token
        out_img = out[:, :Nwin, :]       # [Bwin, Nwin, C]
        out_u = out[:, Nwin, :]          # [Bwin, C]  (the appended unseen per window)

        # Reconstruct image grid
        out_img = out_img.view(B, nWh, nWw, ws, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        out_img = out_img.view(B, H, W, C).view(B, HW, C)

        # Reduce unseen outputs across windows -> one token per sample
        out_u = out_u.view(B, nWh, nWw, C)          # [B, nWh, nWw, C]
        out_unseen = out_u.mean(dim=(1, 2))         # [B, C]

        return out_img, out_unseen


class AABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        attn_drop: float,
        proj_drop: float,
        use_s2rope: bool,
        rope2d: RoPE2D,
        s2rope_seq: S2RopeSequencePositionalEncoding,
        s2rope_kernel: S2RoPE,
    ):
        super().__init__()

        # Separate norms for query/template streams (safer)
        self.norm_q1 = nn.LayerNorm(dim)
        self.norm_t1 = nn.LayerNorm(dim)
        self.g_attn = GlobalSelfAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_s2rope=use_s2rope,
            rope2d=rope2d,
            s2rope_seq=s2rope_seq,
            s2rope_kernel=s2rope_kernel,
        )

        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_t2 = nn.LayerNorm(dim)
        self.l_attn = LocalWindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            rope2d=rope2d,
        )

        self.norm_q3 = nn.LayerNorm(dim)
        self.norm_t3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp_q = MLP(dim, hidden, drop=proj_drop)
        self.mlp_t = MLP(dim, hidden, drop=proj_drop)

    def forward(
        self,
        q_tokens: torch.Tensor,                 # [B, HW, C]
        t_tokens: torch.Tensor,                 # [B, S, HW+1, C]  (last token is unseen)
        pos2d: torch.Tensor,                    # [B, HW, 2]
        grid_size: Tuple[int, int],             # (H, W)
        template_frame_dirs: Optional[torch.Tensor] = None,    # [B, S, 3]
        template_frame_has_s2: Optional[torch.Tensor] = None,  # [B, S]
        ref_dirs: Optional[torch.Tensor] = None,               # broadcastable
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, HW, C = q_tokens.shape
        B2, S, Nt, C2 = t_tokens.shape
        assert B2 == B and C2 == C
        assert Nt == HW + 1, "Expected templates to include exactly one unseen token (HW+1)."

        # ============================================================
        # 1) Global attention
        #    - Query: RoPE2D on all query image tokens
        #    - Templates: S²RoPE on template image tokens only (prefix),
        #                unseen participates but gets no PE.
        # ============================================================

        # Query global
        q_in = self.norm_q1(q_tokens)
        q_tokens = q_tokens + self.g_attn.forward_query(q_in, pos2d, num_img=HW)

        # Template global (flatten S -> batch)
        t_flat = t_tokens.view(B * S, HW + 1, C)
        t_in = self.norm_t1(t_flat)

        # pos2d per template: [B*S, HW, 2]
        pos2d_t = pos2d[:, None, :, :].expand(B, S, HW, 2).reshape(B * S, HW, 2)

        # Compute S² phases once for all templates (vectorized), if enabled
        phase_x = phase_y = phase_s2 = s2_mask = None
        if self.g_attn.use_s2rope:
            # phase generator expects pos: [B, S, HW, 2]
            pos_seq = pos2d[:, None, :, :].expand(B, S, HW, 2)

            phase_x, phase_y, phase_s2, mask = self.g_attn.s2rope_seq(
                pos=pos_seq,
                frame_dirs=template_frame_dirs,
                frame_has_s2=template_frame_has_s2,
                ref_dirs=ref_dirs,
                token_s2_mask=None,
            )

            # Flatten to [B*S, HW, ...]
            if phase_x is not None:
                phase_x = phase_x.reshape(B * S, HW, -1)
            if phase_y is not None:
                phase_y = phase_y.reshape(B * S, HW, -1)
            if phase_s2 is not None:
                # [B,S,HW,F,ps] -> [B*S,HW,F,ps]
                phase_s2 = phase_s2.reshape(B * S, HW, phase_s2.shape[-2], phase_s2.shape[-1])
            if mask is not None:
                s2_mask = mask.reshape(B * S, HW)

        # Global template attention (over HW+1 tokens)
        t_out = self.g_attn.forward_template(
            x_t=t_in,                 # [B*S, HW+1, C]
            pos2d_img=pos2d_t,         # [B*S, HW, 2]
            num_img=HW,
            phase_x=phase_x,
            phase_y=phase_y,
            phase_s2=phase_s2,
            s2_mask=s2_mask,
        )
        t_flat = t_flat + t_out
        t_tokens = t_flat.view(B, S, HW + 1, C)

        # ============================================================
        # 2) Local attention
        #    Requirement (corrected): template unseen token MUST be
        #    included with its own image tokens in local attention,
        #    but MUST NOT get RoPE.
        #
        #    Query has only image tokens -> standard local attention.
        # ============================================================

        # Query local (image tokens only)
        q_in = self.norm_q2(q_tokens)
        q_local, _ = self.l_attn(q_in, pos2d, grid_size, unseen=None)
        q_tokens = q_tokens + q_local

        # Template local: include unseen
        t_img = t_tokens[:, :, :HW, :].reshape(B * S, HW, C)    # [B*S, HW, C]
        t_unseen = t_tokens[:, :, HW, :].reshape(B * S, C)      # [B*S, C]

        t_img_in = self.norm_t2(t_img)
        t_img_local, t_unseen_local = self.l_attn(
            t_img_in,
            pos2d_t,                      # [B*S, HW, 2]
            grid_size,
            unseen=t_unseen,              # <-- key: unseen participates, but gets no RoPE inside l_attn
        )
        # Residuals:
        t_img = t_img + t_img_local                   # [B*S, HW, C]
        t_unseen = t_unseen + t_unseen_local          # [B*S, C]

        # Stitch back to [B, S, HW+1, C]
        t_img = t_img.view(B, S, HW, C)
        t_unseen = t_unseen.view(B, S, 1, C)
        t_tokens = torch.cat([t_img, t_unseen], dim=2)

        # ============================================================
        # 3) MLP (full sequences)
        # ============================================================

        q_tokens = q_tokens + self.mlp_q(self.norm_q3(q_tokens))

        t_flat = t_tokens.view(B * S, HW + 1, C)
        t_flat = t_flat + self.mlp_t(self.norm_t3(t_flat))
        t_tokens = t_flat.view(B, S, HW + 1, C)

        return q_tokens, t_tokens



class AAModule(nn.Module):
    """
    Final AA module:
      input:
        q_tokens: [B, HW, C]
        t_tokens: [B, S, HW+1, C]  (must include unseen as last token)
        pos2d:    [B, HW, 2]
      output:
        q_tokens, t_tokens updated through depth blocks
    """

    def __init__(self, cfg: AAConfig):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.use_s2rope = cfg.use_s2rope

        # Pos encoders/kernels
        self.rope2d = RoPE2D(freq=100.0)  # CroCo default freq=100
        self.s2rope_seq = S2RopeSequencePositionalEncoding(head_dim=self.dim // cfg.num_heads)
        self.s2rope_kernel = S2RoPE()

        self.blocks = nn.ModuleList([
            AABlock(
                dim=cfg.dim,
                num_heads=cfg.num_heads,
                window_size=cfg.window_size,
                mlp_ratio=cfg.mlp_ratio,
                attn_drop=cfg.attn_drop,
                proj_drop=cfg.proj_drop,
                use_s2rope=cfg.use_s2rope,
                rope2d=self.rope2d,
                s2rope_seq=self.s2rope_seq,
                s2rope_kernel=self.s2rope_kernel,
            )
            for _ in range(cfg.depth)
        ])

    def forward(
        self,
        q_tokens: torch.Tensor,                 # [B, HW, C]
        t_tokens: torch.Tensor,                 # [B, S, HW+1, C]
        pos2d: torch.Tensor,                    # [B, HW, 2]
        grid_size: Tuple[int, int],             # (H, W)
        template_frame_dirs: Optional[torch.Tensor] = None,   # [B,S,3]
        template_frame_has_s2: Optional[torch.Tensor] = None, # [B,S]
        ref_dirs: Optional[torch.Tensor] = None,              # broadcastable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for blk in self.blocks:
            q_tokens, t_tokens = blk(
                q_tokens=q_tokens,
                t_tokens=t_tokens,
                pos2d=pos2d,
                grid_size=grid_size,
                template_frame_dirs=template_frame_dirs,
                template_frame_has_s2=template_frame_has_s2,
                ref_dirs=ref_dirs,
            )
        return q_tokens, t_tokens
