# pos_embed.py
#
# S2RoPE sequence + single-image positional encoding
#
# IMPORTANT (Option A):
#   - NO automatic inference of d_ref.
#   - ref_dirs MUST be provided if any S² encoding is used.
#   - If `frame_has_s2` contains True anywhere, then ref_dirs is REQUIRED.

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from vpog.models.s2rope.s2rope_module import S2RoPE


# ---------------------------------------------------------------
# helper
# ---------------------------------------------------------------
def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


# ---------------------------------------------------------------
# 2D Patch Positions
# ---------------------------------------------------------------
class PositionGetter:
    def __init__(self):
        self.cache = {}

    def __call__(self, b, h, w, device):
        if (h, w) not in self.cache:
            ys = torch.arange(h, device=device)
            xs = torch.arange(w, device=device)
            grid = torch.cartesian_prod(ys, xs)
            self.cache[(h, w)] = grid
        pos = self.cache[(h, w)].view(1, h * w, 2).expand(b, -1, 2).clone()
        return pos


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.Hp = img_size[0] // patch_size[0]
        self.Wp = img_size[1] // patch_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.Identity()
        self.get_pos = PositionGetter()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]

        x = self.proj(x)  # [B, C, Hp, Wp]
        pos = self.get_pos(B, self.Hp, self.Wp, x.device)
        x = x.flatten(2).transpose(1, 2)  # [B, Hp*Wp, C]
        x = self.norm(x)
        return x, pos


# ---------------------------------------------------------------
# Pure torch S2RoPE (reference) — no CUDA kernel
# ---------------------------------------------------------------
class TorchS2RoPE(nn.Module):
    """
    Pure PyTorch implementation of the S2RoPE rotation.

    Assumes channel layout:
      D = 2 * (px + py + F * ps)

      tokens[..., 0 : 2*px]                 -> X RoPE (px pairs)
      tokens[..., 2*px : 2*(px+py)]         -> Y RoPE (py pairs)
      tokens[..., offset + f*2*ps : ...]    -> S² RoPE for face f, ps pairs

    Same interface as S2RoPE.forward:
      tokens:    [B, N, H, D]
      phase_x:   [B, N, px] or empty
      phase_y:   [B, N, py] or empty
      phase_sph: [B, N, F, ps] or empty
    """

    def __init__(self, head_dim: int, n_faces: int, px: int, py: int, ps: int):
        super().__init__()
        self.n_faces = n_faces
        self.px = px
        self.py = py
        self.ps = ps

        expected = 2 * (px + py + n_faces * ps)
        if expected != head_dim:
            raise ValueError(
                f"TorchS2RoPE: head_dim={head_dim} but expected {expected} "
                f"for px={px}, py={py}, ps={ps}, n_faces={n_faces}"
            )
        self.head_dim = head_dim

    @staticmethod
    def _rotate_pairs(slice_tokens: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """
        slice_tokens: [B, N, H, 2*K]
        angles:       [B, N, K]  (phase per complex pair)
        Returns rotated slice with same shape as slice_tokens.
        """
        if slice_tokens.numel() == 0:
            return slice_tokens

        B, N, H, twoK = slice_tokens.shape
        K = twoK // 2

        # [B, N, H, K, 2]
        x = slice_tokens.view(B, N, H, K, 2)
        a = x[..., 0]  # [B, N, H, K]
        b = x[..., 1]  # [B, N, H, K]

        # angles: [B, N, K] -> [B, N, 1, K] for broadcast over H
        cos = angles.cos().unsqueeze(2)  # [B, N, 1, K]
        sin = angles.sin().unsqueeze(2)  # [B, N, 1, K]

        # Broadcast to [B, N, H, K]
        a_new = a * cos - b * sin
        b_new = a * sin + b * cos

        out = torch.stack([a_new, b_new], dim=-1)  # [B, N, H, K, 2]
        return out.view(B, N, H, twoK)

    def forward(
        self,
        tokens: torch.Tensor,
        phase_x: Optional[torch.Tensor] = None,
        phase_y: Optional[torch.Tensor] = None,
        phase_sph: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokens.dim() != 4:
            raise ValueError("TorchS2RoPE: tokens must have shape [B,N,H,D]")
        B, N, H, D = tokens.shape
        if D != self.head_dim:
            raise ValueError(
                f"TorchS2RoPE: tokens.shape[-1]={D} but head_dim={self.head_dim}"
            )

        out = tokens.clone()

        # Ensure non-None phases
        if phase_x is None or phase_x.numel() == 0 or self.px == 0:
            phase_x_used = None
        else:
            phase_x_used = phase_x  # [B,N,px]

        if phase_y is None or phase_y.numel() == 0 or self.py == 0:
            phase_y_used = None
        else:
            phase_y_used = phase_y  # [B,N,py]

        if phase_sph is None or phase_sph.numel() == 0 or self.ps == 0 or self.n_faces == 0:
            phase_sph_used = None
        else:
            phase_sph_used = phase_sph  # [B,N,F,ps]

        offset = 0

        # X part
        if self.px > 0 and phase_x_used is not None:
            x_slice = out[..., offset : offset + 2 * self.px]
            x_rot = self._rotate_pairs(x_slice, phase_x_used)
            out[..., offset : offset + 2 * self.px] = x_rot
            offset += 2 * self.px

        # Y part
        if self.py > 0 and phase_y_used is not None:
            y_slice = out[..., offset : offset + 2 * self.py]
            y_rot = self._rotate_pairs(y_slice, phase_y_used)
            out[..., offset : offset + 2 * self.py] = y_rot
            offset += 2 * self.py

        # S² part
        if self.ps > 0 and phase_sph_used is not None:
            F = self.n_faces
            for f in range(F):
                # phase_sph: [B,N,F,ps] -> [B,N,ps] for this face
                angles_f = phase_sph_used[:, :, f, :]  # [B,N,ps]
                start = offset + 2 * self.ps * f
                end = start + 2 * self.ps
                s_slice = out[..., start:end]
                s_rot = self._rotate_pairs(s_slice, angles_f)
                out[..., start:end] = s_rot

        return out


# ---------------------------------------------------------------
# S2RopePositionalEncoding
#   *** ref_dirs IS REQUIRED when s2_mask contains True ***
# ---------------------------------------------------------------
class S2RopePositionalEncoding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        n_faces: int = 6,
        px: Optional[int] = None,
        py: Optional[int] = None,
        ps: Optional[int] = None,
        base_xy: float = 10000.0,
        f0_xy: float = 1.0,
        base_s2: float = 10000.0,
        f0_s2: float = 1.0,
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")

        if n_faces != 6:
            raise ValueError("S2RopePositionalEncoding currently assumes n_faces=6")

        # Auto-split
        if px is None or py is None or ps is None:
            D_pairs = head_dim // 2
            ps = max(1, (D_pairs // 4) // n_faces)
            s2_pairs = ps * n_faces
            remaining = D_pairs - s2_pairs
            px = remaining // 2
            py = remaining - px
            assert 2 * (px + py + n_faces * ps) == head_dim

        self.px = px
        self.py = py
        self.ps = ps
        self.n_faces = n_faces
        self.head_dim = head_dim
        self.base_xy = base_xy
        self.f0_xy = f0_xy
        self.base_s2 = base_s2
        self.f0_s2 = f0_s2

        # Kernel-based S2RoPE
        self.rope = S2RoPE(head_dim=head_dim, n_faces=n_faces, px=px, py=py, ps=ps)

    # -------------------------------------------------
    # XY phases
    # -------------------------------------------------
    def _xy_phases(self, pos: torch.Tensor):
        B, N, _ = pos.shape
        pos = pos.float()
        device = pos.device

        phase_x = None
        phase_y = None

        if self.px > 0:
            idx = torch.arange(self.px, device=device)
            inv = self.f0_xy / (self.base_xy ** (idx / max(1.0, float(self.px))))
            # pos[...,1] is x
            phase_x = pos[..., 1:2] * inv  # [B,N,px]

        if self.py > 0:
            idx = torch.arange(self.py, device=device)
            inv = self.f0_xy / (self.base_xy ** (idx / max(1.0, float(self.py))))
            # pos[...,0] is y
            phase_y = pos[..., 0:1] * inv  # [B,N,py]

        return phase_x, phase_y

    # -------------------------------------------------
    # S² phases
    #   ref_dirs is REQUIRED if any mask True
    # -------------------------------------------------
    def _s2_phases(
        self,
        view_dirs: Optional[torch.Tensor],
        ref_dirs: Optional[torch.Tensor],
        s2_mask: Optional[torch.Tensor],
    ):
        if self.ps == 0:
            return None
        if view_dirs is None:
            return None

        B, N, _ = view_dirs.shape
        device = view_dirs.device
        dtype = view_dirs.dtype

        if s2_mask is not None and s2_mask.any() and ref_dirs is None:
            raise ValueError(
                "ref_dirs is REQUIRED when S² RoPE is active. "
                "You chose Option A: no auto-inference from view_dirs."
            )

        if s2_mask is None:
            use_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        else:
            use_mask = s2_mask.bool()

        # Normalize directions
        dirs = view_dirs / (view_dirs.norm(dim=-1, keepdim=True) + 1e-8)

        # ref_dirs must be [B,3]
        if ref_dirs is None:
            return None
        if ref_dirs.dim() == 3 and ref_dirs.size(1) == 1:
            ref = ref_dirs[:, 0, :]
        else:
            ref = ref_dirs
        ref = F.normalize(ref, dim=-1)  # [B,3]

        faces = self._build_faces(ref)  # [B,6,3]

        dot = torch.einsum("bnc,bfc->bnf", dirs, faces).clamp(-1, 1)
        gamma = torch.acos(dot)  # [B,N,6]

        dirs_e = dirs.unsqueeze(2)   # [B,N,1,3]
        faces_e = faces.unsqueeze(1) # [B,1,6,3]
        diff = dirs_e - faces_e * dot.unsqueeze(-1)
        t = diff / (diff.norm(dim=-1, keepdim=True) + 1e-8)
        logv = gamma.unsqueeze(-1) * t  # [B,N,6,3]

        v_face, w_face = self._face_basis(faces)
        v_e = v_face.unsqueeze(1)  # [B,1,6,3]
        w_e = w_face.unsqueeze(1)  # [B,1,6,3]

        u1 = (logv * v_e).sum(dim=-1)
        u2 = (logv * w_e).sum(dim=-1)
        phi = torch.atan2(u2, u1)  # [B,N,6]

        ps = self.ps
        nr = ps // 2
        na = ps - nr

        phases = []

        if nr > 0:
            idx = torch.arange(nr, device=device, dtype=dtype)
            inv = self.f0_s2 / (self.base_s2 ** (idx / max(1.0, float(nr))))
            phases.append(gamma.unsqueeze(-1) * inv.view(1, 1, 1, -1))

        if na > 0:
            idx = torch.arange(na, device=device, dtype=dtype)
            inv = self.f0_s2 / (self.base_s2 ** (idx / max(1.0, float(na))))
            phases.append(phi.unsqueeze(-1) * inv.view(1, 1, 1, -1))

        if not phases:
            return None

        out = torch.cat(phases, dim=-1)  # [B,N,6,ps]

        # mask out tokens with s2_mask=False
        mask = use_mask.float().unsqueeze(-1).unsqueeze(-1)
        out = out * mask

        return out

    # -------------------------------------------------
    # face basis
    # -------------------------------------------------
    def _face_basis(self, faces: torch.Tensor):
        B, F, _ = faces.shape
        device = faces.device
        dtype = faces.dtype

        z = torch.tensor([0, 0, 1], device=device, dtype=dtype).view(1, 1, 3)
        y = torch.tensor([0, 1, 0], device=device, dtype=dtype).view(1, 1, 3)

        dotz = (faces * z).sum(-1, keepdim=True).abs()
        usez = dotz < 0.9
        a = torch.where(usez, z.expand(B, F, -1), y.expand(B, F, -1))

        proj = (a * faces).sum(-1, keepdim=True) * faces
        v = a - proj
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

        w = torch.cross(faces, v, dim=-1)
        w = w / (w.norm(dim=-1, keepdim=True) + 1e-8)
        return v, w

    # -------------------------------------------------
    # build faces
    # -------------------------------------------------
    def _build_faces(self, ref: torch.Tensor):
        B = ref.shape[0]
        e1, e2 = self._orthonormal_frame(ref)
        return torch.stack([ref, -ref, e1, -e1, e2, -e2], dim=1)

    def _orthonormal_frame(self, ref: torch.Tensor):
        B = ref.size(0)
        device = ref.device
        dtype = ref.dtype

        z = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 3)
        dot = (ref * z).sum(-1, keepdim=True).abs()
        usez = dot < 0.9
        a = torch.where(
            usez,
            z.expand(B, 3),
            torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).expand(B, 3),
        )

        proj = (a * ref).sum(-1, keepdim=True) * ref
        v = a - proj
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        w = torch.cross(ref, v, dim=-1)
        w = w / (w.norm(dim=-1, keepdim=True) + 1e-8)
        return v, w

    # -------------------------------------------------
    # forward
    # -------------------------------------------------
    def forward(
        self,
        tokens: torch.Tensor,
        pos: torch.Tensor,
        view_dirs: Optional[torch.Tensor],
        ref_dirs: Optional[torch.Tensor],
        s2_mask: Optional[torch.Tensor],
    ):
        B, N, C = tokens.shape
        if C != self.head_dim:
            raise ValueError("Wrong head_dim.")

        phase_x, phase_y = self._xy_phases(pos)
        phase_s2 = self._s2_phases(view_dirs, ref_dirs, s2_mask)

        t4 = tokens.view(B, N, 1, C)
        out = self.rope(t4, phase_x, phase_y, phase_s2)
        return out.view(B, N, C)


# ---------------------------------------------------------------
# Pure-torch S2RopePositionalEncoding using TorchS2RoPE
# ---------------------------------------------------------------
class S2RopePositionalEncodingTorch(S2RopePositionalEncoding):
    """
    Same interface as S2RopePositionalEncoding, but uses TorchS2RoPE
    (pure PyTorch) instead of the CUDA/C++ kernel.

    Intended for:
      - correctness cross-check vs kernel
      - CPU-only environments without s2rope_ext
    """

    def __init__(
        self,
        head_dim: int,
        n_faces: int = 6,
        px: Optional[int] = None,
        py: Optional[int] = None,
        ps: Optional[int] = None,
        base_xy: float = 10000.0,
        f0_xy: float = 1.0,
        base_s2: float = 10000.0,
        f0_s2: float = 1.0,
    ):
        super().__init__(
            head_dim=head_dim,
            n_faces=n_faces,
            px=px,
            py=py,
            ps=ps,
            base_xy=base_xy,
            f0_xy=f0_xy,
            base_s2=base_s2,
            f0_s2=f0_s2,
        )
        # Override kernel-based rope with pure torch reference
        self.rope = TorchS2RoPE(
            head_dim=self.head_dim,
            n_faces=self.n_faces,
            px=self.px,
            py=self.py,
            ps=self.ps,
        )


class S2RopeSequencePositionalEncoding(nn.Module):
    """
    Phase generator for template sequences.

    Intended usage:
      - You call forward(...) to GET PHASES (for q/k rotation inside attention),
        not to rotate embeddings.
      - AA will apply phases only to image-tokens (no unseen token included here).

    Shapes:
      pos:          [B, S, T, 2]    where T = H*W (image tokens only)
      frame_dirs:   [B, S, 3]       per-template view direction (or camera dir)
      frame_has_s2: [B, S] bool     whether S² is active per template (optional)
      ref_dirs:     [B, 3] (or broadcastable) REQUIRED if any S² is active
      token_s2_mask:[B, S, T] bool  optional per-token enabling mask

    Returns:
      phase_x:   [B, S, T, px] or None if px==0
      phase_y:   [B, S, T, py] or None if py==0
      phase_s2:  [B, S, T, F, ps] or None if S² inactive / ps==0 / frame_dirs is None
      s2_mask:   [B, S, T] bool or None if S² inactive
    """

    def __init__(self, head_dim: int, **kw):
        super().__init__()
        self.head_dim = head_dim
        self.inner = S2RopePositionalEncoding(head_dim=head_dim, **kw)

    @torch.no_grad()
    def _validate_inputs(
        self,
        pos: torch.Tensor,
        frame_dirs: Optional[torch.Tensor],
        frame_has_s2: Optional[torch.Tensor],
        token_s2_mask: Optional[torch.Tensor],
    ):
        # pos: [B,S,T,2]
        assert pos.dim() == 4 and pos.size(-1) == 2, f"pos must be [B,S,T,2], got {tuple(pos.shape)}"
        B, S, T, _ = pos.shape

        if frame_dirs is not None:
            assert frame_dirs.shape[:2] == (B, S) and frame_dirs.size(-1) == 3, \
                f"frame_dirs must be [B,S,3], got {tuple(frame_dirs.shape)}"

        if frame_has_s2 is not None:
            assert frame_has_s2.shape == (B, S), \
                f"frame_has_s2 must be [B,S], got {tuple(frame_has_s2.shape)}"

        if token_s2_mask is not None:
            assert token_s2_mask.shape == (B, S, T), \
                f"token_s2_mask must be [B,S,T], got {tuple(token_s2_mask.shape)}"

    def forward(
        self,
        pos: torch.Tensor,                          # [B,S,T,2] image tokens only
        frame_dirs: Optional[torch.Tensor] = None,  # [B,S,3]
        frame_has_s2: Optional[torch.Tensor] = None,# [B,S] bool
        ref_dirs: Optional[torch.Tensor] = None,    # [B,3] (or broadcastable)
        token_s2_mask: Optional[torch.Tensor] = None,# [B,S,T] bool
    ):
        self._validate_inputs(pos, frame_dirs, frame_has_s2, token_s2_mask)

        B, S, T, _ = pos.shape
        device = pos.device

        # -----------------------
        # XY phases (always well-defined for image tokens)
        # -----------------------
        pos_f = pos.reshape(B, S * T, 2)
        phase_x_f, phase_y_f = self.inner._xy_phases(pos_f)

        phase_x = None if phase_x_f is None else phase_x_f.reshape(B, S, T, -1)
        phase_y = None if phase_y_f is None else phase_y_f.reshape(B, S, T, -1)

        # -----------------------
        # S² phases (only if enabled)
        # -----------------------
        if self.inner.ps == 0 or frame_dirs is None:
            # no spherical component
            return phase_x, phase_y, None, None

        if frame_has_s2 is None:
            frame_mask = torch.ones(B, S, dtype=torch.bool, device=device)
        else:
            frame_mask = frame_has_s2.bool()

        if token_s2_mask is None:
            token_mask = torch.ones(B, S, T, dtype=torch.bool, device=device)
        else:
            token_mask = token_s2_mask.bool()

        # final token-wise mask: [B,S,T]
        final_mask = frame_mask[:, :, None] & token_mask
        s2_mask_f = final_mask.reshape(B, S * T)  # [B, S*T]

        # Expand per-template dirs to per-token dirs without loops:
        # frame_dirs: [B,S,3] -> [B,S,T,3] -> [B,S*T,3]
        vdirs = frame_dirs[:, :, None, :].expand(B, S, T, 3).reshape(B, S * T, 3)

        # Compute spherical phases for flattened tokens.
        phase_s2_f = self.inner._s2_phases(
            view_dirs=vdirs,
            ref_dirs=ref_dirs,
            s2_mask=s2_mask_f,
        )
        phase_s2 = None if phase_s2_f is None else phase_s2_f.reshape(B, S, T, self.inner.n_faces, -1)

        return phase_x, phase_y, phase_s2, final_mask