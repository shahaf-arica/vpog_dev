# aa_module.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from external.croco.models.pos_embed import RoPE2D

from .so3prope.so3prope import SO3PropeKernel

@dataclass
class AAConfig:
    dim: int
    depth: int
    num_heads: int
    window_size: int

    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    # global encodings
    use_so3prope: bool = False

    # SO3 channel fraction within head_dim (must produce multiple-of-3 channels)
    # Best-practice: keep configurable, but default is 1/2 of head_dim.
    so3_frac: float = 0.5

    # RoPE2D freq (CroCo default typically 100)
    rope2d_freq: float = 100.0


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, drop: float):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class GlobalJointAttention(nn.Module):
    """
    One global attention over a JOINT sequence:
      x_joint = [query_tokens || template_tokens_flat]

    Encoding policy (as you requested):
      - Query image tokens: RoPE2D
      - Template image tokens: SO3Prope (if enabled)
      - Template special tokens: no encoding (for now)

    Inputs:
      q_tokens: [B, HW, C]
      t_tokens: [B, S, Nt, C], where Nt = HW + K (K >= 0)
      pos2d:    [B, HW, 2]  (positions for query image tokens ONLY)
      template_poses: [B, S, 4, 4], object->camera; we use R = T[:3,:3]
    """

    def __init__(self, cfg: AAConfig, rope2d: RoPE2D):
        super().__init__()
        assert cfg.dim % cfg.num_heads == 0, "dim must be divisible by num_heads"
        self.dim = cfg.dim
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.dim // cfg.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=True)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=True)
        self.attn_drop = nn.Dropout(cfg.attn_drop)
        self.proj_drop = nn.Dropout(cfg.proj_drop)

        self.rope2d = rope2d
        self.use_so3prope = cfg.use_so3prope

        # derive so3_dim from frac with constraints
        raw = int(round(self.head_dim * cfg.so3_frac))
        so3_dim = (raw // 3) * 3  # enforce multiple-of-3
        if so3_dim > self.head_dim:
            so3_dim = (self.head_dim // 3) * 3
        self.so3_dim = so3_dim

        self.so3_kernel = SO3PropeKernel(head_dim=self.head_dim, so3_dim=self.so3_dim) if self.use_so3prope else None

    def _qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        return qkv[0], qkv[1], qkv[2]  # [B,Hh,N,Dh]

    def forward(
        self,
        q_tokens: torch.Tensor,                # [B, HW, C]
        t_tokens: torch.Tensor,                # [B, S, Nt, C]
        pos2d: torch.Tensor,                   # [B, HW, 2]
        *,
        template_poses: Optional[torch.Tensor] = None,  # [B,S,4,4]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, HW, C = q_tokens.shape
        B2, S, Nt, C2 = t_tokens.shape
        assert B2 == B and C2 == C
        K = Nt - HW
        assert K >= 0

        # joint tokens
        t_flat = t_tokens.view(B, S * Nt, C)
        x_joint = torch.cat([q_tokens, t_flat], dim=1)  # [B, N, C]
        N = x_joint.shape[1]

        q, k, v = self._qkv(x_joint)  # [B,Hh,N,Dh]

        # ---- Apply RoPE2D ONLY to query image tokens (first HW tokens) ----
        q_q = q[:, :, :HW, :]
        k_q = k[:, :, :HW, :]
        q_t = q[:, :, HW:, :]
        k_t = k[:, :, HW:, :]

        q_q = self.rope2d(q_q, pos2d)
        k_q = self.rope2d(k_q, pos2d)

        q = torch.cat([q_q, q_t], dim=2)
        k = torch.cat([k_q, k_t], dim=2)

        # ---- Apply SO3Prope ONLY to template IMAGE tokens ----
        if self.use_so3prope:
            assert template_poses is not None, "template_poses is required when use_so3prope=True"
            assert template_poses.shape == (B, S, 4, 4)

            R = template_poses[:, :, :3, :3].contiguous()  # [B,S,3,3]

            # reshape template part to [B,Hh,S,Nt,Dh]
            q_t = q[:, :, HW:, :].view(B, self.num_heads, S, Nt, self.head_dim)
            k_t = k[:, :, HW:, :].view(B, self.num_heads, S, Nt, self.head_dim)

            # apply SO3 to first HW tokens in each template chunk
            q_t = self.so3_kernel(q_t, R, num_img=HW)
            k_t = self.so3_kernel(k_t, R, num_img=HW)

            # stitch back
            q[:, :, HW:, :] = q_t.view(B, self.num_heads, S * Nt, self.head_dim)
            k[:, :, HW:, :] = k_t.view(B, self.num_heads, S * Nt, self.head_dim)

        # attention
        # attn = (q * self.scale) @ k.transpose(-2, -1)  # [B,Hh,N,N]
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        # out = attn @ v  # [B,Hh,N,Dh]

        # attention (memory-efficient SDPA; avoids explicit [B,Hh,N,N])
        dropout_p = self.attn_drop.p if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False,
        )  # [B,Hh,N,Dh]


        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.proj_drop(self.proj(out))

        out_q = out[:, :HW, :]
        out_t = out[:, HW:, :].view(B, S, Nt, C)
        return out_q, out_t


class LocalWindowAttention(nn.Module):
    """
    Local attention on image tokens with optional special tokens per template.
    Best-practice default:
      - apply RoPE2D to image tokens only
      - special tokens participate without RoPE2D and are window-aggregated (mean)

    This is orthogonal to your requested global policy; we keep it stable and generic.
    """

    def __init__(self, cfg: AAConfig, rope2d):
        super().__init__()
        assert cfg.dim % cfg.num_heads == 0
        self.dim = cfg.dim
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.dim // cfg.num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = cfg.window_size

        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=True)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=True)
        self.attn_drop = nn.Dropout(cfg.attn_drop)
        self.proj_drop = nn.Dropout(cfg.proj_drop)
        self.rope2d = rope2d

    def _qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        return qkv[0], qkv[1], qkv[2]

    def forward(
        self,
        x_img: torch.Tensor,                 # [B, HW, C]
        pos2d: torch.Tensor,                 # [B, HW, 2]
        grid_size: Tuple[int, int],          # (H, W)
        *,
        special: Optional[torch.Tensor] = None,  # [B, K, C]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B, HW, C = x_img.shape
        H, W = grid_size
        assert HW == H * W
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0

        nWh = H // ws
        nWw = W // ws
        Bwin = B * nWh * nWw
        Nwin = ws * ws

        xw = x_img.view(B, H, W, C)
        pw = pos2d.view(B, H, W, 2)

        xw = xw.view(B, nWh, ws, nWw, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        pw = pw.view(B, nWh, ws, nWw, ws, 2).permute(0, 1, 3, 2, 4, 5).contiguous()

        xw = xw.view(Bwin, Nwin, C)
        pw = pw.view(Bwin, Nwin, 2)

        K = 0
        if special is not None:
            assert special.shape[0] == B and special.shape[2] == C
            K = special.shape[1]
            sp = special.view(B, 1, 1, K, C).expand(B, nWh, nWw, K, C).contiguous()
            sp = sp.view(Bwin, K, C)
            xw = torch.cat([xw, sp], dim=1)  # [Bwin, Nwin+K, C]

        q, k, v = self._qkv(xw)  # [Bwin,Hh,Nseq,Dh]
        Nseq = xw.shape[1]

        # RoPE2D only on image tokens in window (first Nwin)
        q_img, k_img = q[:, :, :Nwin, :], k[:, :, :Nwin, :]
        q_sp, k_sp = q[:, :, Nwin:, :], k[:, :, Nwin:, :]

        q_img = self.rope2d(q_img, pw)
        k_img = self.rope2d(k_img, pw)

        q = torch.cat([q_img, q_sp], dim=2)
        k = torch.cat([k_img, k_sp], dim=2)

        # attn = (q * self.scale) @ k.transpose(-2, -1)
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # out = attn @ v

        dropout_p = self.attn_drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False,
        )  # [Bwin, Hh, Nseq, Dh]

        out = out.transpose(1, 2).contiguous().view(Bwin, Nseq, C)
        out = self.proj_drop(self.proj(out))

        out_img = out[:, :Nwin, :].view(B, nWh, nWw, ws, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        out_img = out_img.view(B, H, W, C).view(B, HW, C)

        if special is None or K == 0:
            return out_img, None

        out_sp = out[:, Nwin:, :].view(B, nWh, nWw, K, C).mean(dim=(1, 2))
        return out_img, out_sp


class AABlock(nn.Module):
    def __init__(self, cfg: AAConfig, rope2d: RoPE2D):
        super().__init__()
        self.cfg = cfg

        self.norm_q1 = nn.LayerNorm(cfg.dim)
        self.norm_t1 = nn.LayerNorm(cfg.dim)
        self.g_attn = GlobalJointAttention(cfg, rope2d)

        self.norm_q2 = nn.LayerNorm(cfg.dim)
        self.norm_t2 = nn.LayerNorm(cfg.dim)
        self.l_attn = LocalWindowAttention(cfg, rope2d)

        self.norm_q3 = nn.LayerNorm(cfg.dim)
        self.norm_t3 = nn.LayerNorm(cfg.dim)
        self.mlp_q = MLP(cfg.dim, cfg.mlp_ratio, cfg.proj_drop)
        self.mlp_t = MLP(cfg.dim, cfg.mlp_ratio, cfg.proj_drop)

    def forward(
        self,
        q_tokens: torch.Tensor,                 # [B, HW, C]
        t_tokens: torch.Tensor,                 # [B, S, Nt, C]
        pos2d: torch.Tensor,                    # [B, HW, 2]
        grid_size: Tuple[int, int],             # (H, W)
        *,
        template_poses: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, HW, C = q_tokens.shape
        B2, S, Nt, C2 = t_tokens.shape
        assert B2 == B and C2 == C
        K = Nt - HW
        assert K >= 0

        # ---- Global ----
        q_in = self.norm_q1(q_tokens)
        t_in = self.norm_t1(t_tokens.view(B * S, Nt, C)).view(B, S, Nt, C)
        dq, dt = self.g_attn(q_in, t_in, pos2d, template_poses=template_poses)
        q_tokens = q_tokens + dq
        t_tokens = t_tokens + dt

        # ---- Local (query) ----
        q_local_in = self.norm_q2(q_tokens)
        dq_local, _ = self.l_attn(q_local_in, pos2d, grid_size, special=None)
        q_tokens = q_tokens + dq_local

        # ---- Local (templates) ----
        t_img = t_tokens[:, :, :HW, :].reshape(B * S, HW, C)
        t_sp = t_tokens[:, :, HW:, :].reshape(B * S, K, C) if K > 0 else None
        pos2d_t = pos2d[:, None, :, :].expand(B, S, HW, 2).reshape(B * S, HW, 2)

        # t_img_in = self.norm_t2(t_img)
        # dt_img, dt_sp = self.l_attn(t_img_in, pos2d_t, grid_size, special=t_sp)

        t_tokens_in = self.norm_t2(t_tokens)
        t_img_in = t_tokens_in[:, :, :HW, :].reshape(B * S, HW, C)
        t_sp_in  = t_tokens_in[:, :, HW:, :].reshape(B * S, K, C) if K > 0 else None

        if torch.isnan(t_tokens_in).any():
            raise RuntimeError("NaN in t_tokens BEFORE local attention")

        dt_img, dt_sp = self.l_attn(t_img_in, pos2d_t, grid_size, special=t_sp_in)

        t_img = t_img + dt_img
        if K > 0:
            t_sp = t_sp + dt_sp

        t_img = t_img.view(B, S, HW, C)
        if K > 0:
            t_sp = t_sp.view(B, S, K, C)
            t_tokens = torch.cat([t_img, t_sp], dim=2)
        else:
            t_tokens = t_img

        # ---- MLP ----
        q_tokens = q_tokens + self.mlp_q(self.norm_q3(q_tokens))
        t_flat = t_tokens.view(B * S, t_tokens.shape[2], C)
        t_flat = t_flat + self.mlp_t(self.norm_t3(t_flat))
        t_tokens = t_flat.view(B, S, t_tokens.shape[2], C)

        return q_tokens, t_tokens


class AAModule(nn.Module):
    def __init__(self, cfg: AAConfig):
        super().__init__()
        self.cfg = cfg
        self.rope2d = RoPE2D(freq=cfg.rope2d_freq)
        self.blocks = nn.ModuleList([AABlock(cfg, self.rope2d) for _ in range(cfg.depth)])

    def forward(
        self,
        q_tokens: torch.Tensor,
        t_tokens: torch.Tensor,
        pos2d: torch.Tensor,
        grid_size: Tuple[int, int],
        *,
        template_poses: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.cfg.use_so3prope:
            assert template_poses is not None, "template_poses required when use_so3prope=True"

        for blk in self.blocks:
            q_tokens, t_tokens = blk(
                q_tokens, t_tokens, pos2d, grid_size, template_poses=template_poses
            )
        return q_tokens, t_tokens
