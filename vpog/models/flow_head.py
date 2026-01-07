# vpog/models/flow_head.py
#
# Buddy-only flow heads with PACKED valid (q,t) pairs.
#
# Goal:
#   Avoid computing dense flow for all (B,S,Nq) pairs.
#   Instead, pack only valid pairs where patch_cls in [0..Nt-1].
#
# patch_cls semantics:
#   -1            => background (ignore)
#    0..Nt-1      => buddy template patch index (flattened)
#    Nt (=Nq)     => object patch but unseen-in-template (ignore flow)
#
# Dense flow is defined in "patch coordinates":
#   dense flow is t->q per template pixel, expressed in query-patch coords.
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

############### DEBUGGING UTILITIES ###############
def first_bad_row(x: torch.Tensor) -> int | None:
    if x.numel() == 0:
        return None
    bad = ~torch.isfinite(x).view(x.shape[0], -1).all(dim=1)
    if bad.any():
        return int(bad.nonzero(as_tuple=False)[0].item())
    return None

def assert_finite(name: str, x: torch.Tensor):
    if x is None:
        return
    if not torch.isfinite(x).all():
        # try row-wise first if it looks like [M,...] or [B,...]
        m = None
        if x.dim() >= 2:
            m = first_bad_row(x.view(x.shape[0], -1))
        raise RuntimeError(f"[NaN DEBUG] {name} has non-finite values. first_bad_row={m}, shape={tuple(x.shape)}")
##########################################################


# -------------------------
# Packing utilities (NO LOOPS)
# -------------------------
@torch.no_grad()
def pack_valid_qt_pairs(
    q_tokens: torch.Tensor,     # [B, Nq, C]
    t_tokens: torch.Tensor,     # [B, S, Nt, C]  (image tokens only; no unseen)
    patch_cls: torch.Tensor,    # [B, S, Nq]     (-1 bg, 0..Nt-1 buddy, Nt unseen)
) -> Dict[str, torch.Tensor]:
    """
    Packs only valid (b,s,q)->t_idx pairs into a compact M-size batch.

    Returns dict with:
      q_tok:   [M, C]
      t_tok:   [M, C]
      b_idx:   [M]
      s_idx:   [M]
      q_idx:   [M]
      t_idx:   [M]   (buddy template patch index in [0..Nt-1])
      M:       scalar tensor (#pairs)
    """
    if q_tokens.dim() != 3:
        raise ValueError(f"q_tokens must be [B,Nq,C], got {tuple(q_tokens.shape)}")
    if t_tokens.dim() != 4:
        raise ValueError(f"t_tokens must be [B,S,Nt,C], got {tuple(t_tokens.shape)}")
    if patch_cls.dim() != 3:
        raise ValueError(f"patch_cls must be [B,S,Nq], got {tuple(patch_cls.shape)}")

    B, Nq, C = q_tokens.shape
    Bt, S, Nt, Ct = t_tokens.shape
    if Bt != B or Ct != C:
        raise ValueError(f"Shape mismatch: q={tuple(q_tokens.shape)} t={tuple(t_tokens.shape)}")
    if patch_cls.shape != (B, S, Nq):
        raise ValueError(f"patch_cls must be [B,S,Nq], got {tuple(patch_cls.shape)}")

    # valid iff buddy index in [0..Nt-1]
    valid = (patch_cls >= 0) & (patch_cls < Nt)  # [B,S,Nq]

    # indices of valid pairs (vectorized)
    b_idx, s_idx, q_idx = valid.nonzero(as_tuple=True)  # each [M]

    ############# for DEBUGGING
    HW = q_tokens.shape[1]   # 196
    Nt = t_tokens.shape[2]   # 196 or 197

    # q_idx must always be in [0, HW-1]
    if not ((q_idx >= 0).all() and (q_idx < HW).all()):
        bad = ~((q_idx >= 0) & (q_idx < HW))
        m = int(bad.nonzero(as_tuple=False)[0].item())
        raise RuntimeError(f"[NaN DEBUG] q_idx out of range at m={m}: q_idx={int(q_idx[m])}, HW={HW}")

    # t_idx is derived from patch_cls; compute it explicitly here for debug
    t_idx = patch_cls[b_idx, s_idx, q_idx]  # [M] long

    if not ((t_idx >= 0).all() and (t_idx < Nt).all()):
        bad = ~((t_idx >= 0) & (t_idx < Nt))
        m = int(bad.nonzero(as_tuple=False)[0].item())
        raise RuntimeError(f"[NaN DEBUG] t_idx out of range at m={m}: t_idx={int(t_idx[m])}, Nt={Nt}")
    # This is diagnostic only:
    num_unseen = int((t_idx == HW).sum().item()) if Nt == HW + 1 else 0
    if num_unseen > 0:
        print(f"[NaN DEBUG] pack_valid_qt_pairs: unseen pairs included: {num_unseen}/{t_idx.numel()} (t_idx==HW)")
    #############

    if b_idx.numel() == 0:
        # Return empty packed tensors (keep device/dtype consistent)
        empty = q_tokens.new_empty((0, C))
        empty_i = patch_cls.new_empty((0,), dtype=torch.long)
        return {
            "q_tok": empty,
            "t_tok": empty,
            "b_idx": empty_i,
            "s_idx": empty_i,
            "q_idx": empty_i,
            "t_idx": empty_i,
            "M": torch.tensor(0, device=q_tokens.device),
        }

    t_idx = patch_cls[b_idx, s_idx, q_idx].long()  # [M] in [0..Nt-1]

    # gather q tokens: [M,C]
    q_tok = q_tokens[b_idx, q_idx]  # advanced indexing

    # gather buddy template tokens: [M,C]
    t_tok = t_tokens[b_idx, s_idx, t_idx]

    ############ DEBUGGING
    assert_finite("packed q_tok", q_tok)
    assert_finite("packed t_tok", t_tok)

    # Pinpoint first bad packed row and print its mapping
    bad_m_q = first_bad_row(q_tok)
    bad_m_t = first_bad_row(t_tok)
    if bad_m_q is not None or bad_m_t is not None:
        m = bad_m_q if bad_m_q is not None else bad_m_t
        print("[NaN DEBUG] BAD packed row m=", m)
        print("  b_idx,s_idx,q_idx,t_idx =",
            int(b_idx[m]), int(s_idx[m]), int(q_idx[m]), int(t_idx[m]))
        raise RuntimeError("[NaN DEBUG] Non-finite token after gather in pack_valid_qt_pairs")
    #######################

    return {
        "q_tok": q_tok,
        "t_tok": t_tok,
        "b_idx": b_idx.long(),
        "s_idx": s_idx.long(),
        "q_idx": q_idx.long(),
        "t_idx": t_idx.long(),
        "M": torch.tensor(int(b_idx.numel()), device=q_tokens.device),
    }


def _make_mlp(in_dim: int, hidden_dims: Tuple[int, ...], dropout: float, use_layernorm: bool) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h, bias=True))
        if use_layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.GELU())
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    return nn.Sequential(*layers)


def _fuse_qt_packed(q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Packed fusion.
      q: [M,C]
      t: [M,C]
    returns: [M,4C] as [q, t, q-t, q*t]
    """
    return torch.cat([q, t, q - t, q * t], dim=-1)


# -------------------------
# Heads
# -------------------------
class BuddyDenseFlowHead(nn.Module):
    """
    Dense flow head over PACKED valid pairs only.

    forward_packed:
      input:
        q_tok: [M,C]
        t_tok: [M,C]
      output:
        dense_flow: [M,ps,ps,2]
        dense_b:    [M,ps,ps]    (Laplace scale, positive)
        dense_w:    [M,ps,ps]    (= 1/(b+eps), unbounded)
    """

    def __init__(
        self,
        ps: int = 16,
        in_dim: int = 128,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.0,
        use_layernorm: bool = True,
        eps_b: float = 1e-4,
    ):
        super().__init__()
        self.ps = int(ps)
        self.eps_b = float(eps_b)

        fuse_dim = 4 * in_dim
        self.backbone = _make_mlp(
            in_dim=fuse_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )
        last_dim = hidden_dims[-1] if len(hidden_dims) > 0 else fuse_dim

        self.flow_head = nn.Linear(last_dim, self.ps * self.ps * 2, bias=True)
        nn.init.normal_(self.flow_head.weight, std=1e-3)
        nn.init.zeros_(self.flow_head.bias)

        self.b_head = nn.Linear(last_dim, self.ps * self.ps, bias=True)
        nn.init.normal_(self.b_head.weight, std=1e-3)
        nn.init.zeros_(self.b_head.bias)

    def forward_packed(
        self,
        q_tok: torch.Tensor,   # [M,C]
        t_tok: torch.Tensor,   # [M,C]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if q_tok.dim() != 2 or t_tok.dim() != 2:
            raise ValueError(f"q_tok,t_tok must be [M,C]; got {tuple(q_tok.shape)}, {tuple(t_tok.shape)}")
        if q_tok.shape != t_tok.shape:
            raise ValueError(f"q_tok shape {tuple(q_tok.shape)} != t_tok shape {tuple(t_tok.shape)}")
        
        ############# for DEBUGGING
        assert_finite("dense_head q_tok input", q_tok)
        assert_finite("dense_head t_tok input", t_tok)
        ############################


        M, C = q_tok.shape
        if M == 0:
            empty_flow = q_tok.new_empty((0, self.ps, self.ps, 2))
            empty_b = q_tok.new_empty((0, self.ps, self.ps))
            empty_w = q_tok.new_empty((0, self.ps, self.ps))
            return empty_flow, empty_b, empty_w

        fused = _fuse_qt_packed(q_tok, t_tok)  # [M,4C]

        ############# for DEBUGGING
        assert_finite("dense_head fused", fused)
        ############################

        h = self.backbone(fused)

        ############# for DEBUGGING
        assert_finite("dense_head backbone out", h)
        ############################

        flow_flat = self.flow_head(h)  # [M, ps*ps*2]
        b_flat = self.b_head(h)        # [M, ps*ps]

        ############## for DEBUGGING
        assert_finite("dense_head flow_flat", flow_flat)
        assert_finite("dense_head b_flat", b_flat)
        ############################

        dense_flow = flow_flat.view(M, self.ps, self.ps, 2)

        raw_b = b_flat.view(M, self.ps, self.ps)
        dense_b = F.softplus(raw_b) + self.eps_b

        dense_w = 1.0 / (dense_b + self.eps_b)  # unbounded weight

        return dense_flow, dense_b, dense_w


class CenterFlowHead(nn.Module):
    """
    Optional future head for patch-center flow (q->t) over PACKED valid pairs.

    forward_packed:
      input:
        q_tok: [M,C]
        t_tok: [M,C]
      output:
        center_flow: [M,2]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128),
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        fuse_dim = 4 * in_dim
        self.backbone = _make_mlp(fuse_dim, hidden_dims, dropout, use_layernorm)
        last_dim = hidden_dims[-1] if len(hidden_dims) > 0 else fuse_dim

        self.head = nn.Linear(last_dim, 2, bias=True)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward_packed(self, q_tok: torch.Tensor, t_tok: torch.Tensor) -> torch.Tensor:
        if q_tok.dim() != 2 or t_tok.dim() != 2:
            raise ValueError(f"q_tok,t_tok must be [M,C]; got {tuple(q_tok.shape)}, {tuple(t_tok.shape)}")
        if q_tok.shape != t_tok.shape:
            raise ValueError(f"q_tok shape {tuple(q_tok.shape)} != t_tok shape {tuple(t_tok.shape)}")

        M, _ = q_tok.shape
        if M == 0:
            return q_tok.new_empty((0, 2))

        fused = _fuse_qt_packed(q_tok, t_tok)
        h = self.backbone(fused)
        out = self.head(h)  # [M,2]
        return out