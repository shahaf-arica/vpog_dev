# so3prope.py
from __future__ import annotations
import torch
import torch.nn as nn


class SO3PropeKernel(nn.Module):
    """
    Applies a per-template SO(3) rotation to the first `so3_dim` channels of template IMAGE tokens only.

    Expected input:
      x_t: [B, Hh, S, Nt, Dh]  (template tokens only)
      R  : [B, S, 3, 3]        (R = T[:3,:3] per template)
    Only tokens [0:num_img) are treated as template image tokens; tokens after that are "special" and unchanged.

    We apply multiplication by R^T (stable, unit-free, orthonormal). This aligns with the "rotate features"
    view and is symmetric for q/k in your chosen policy.
    """

    def __init__(self, *, head_dim: int, so3_dim: int):
        super().__init__()
        assert 0 <= so3_dim <= head_dim
        assert so3_dim % 3 == 0, "so3_dim must be divisible by 3"
        self.head_dim = head_dim
        self.so3_dim = so3_dim
        self.reps = so3_dim // 3

    def forward(
        self,
        x_t: torch.Tensor,   # [B, Hh, S, Nt, Dh]
        R: torch.Tensor,     # [B, S, 3, 3]
        *,
        num_img: int,        # HW
    ) -> torch.Tensor:
        if self.so3_dim == 0 or num_img == 0:
            return x_t

        B, Hh, S, Nt, Dh = x_t.shape
        assert Dh == self.head_dim
        assert R.shape == (B, S, 3, 3)
        assert num_img <= Nt

        Rt = R.transpose(-1, -2).contiguous()  # [B,S,3,3]

        # split channels
        x_so3 = x_t[..., : self.so3_dim]      # [B,Hh,S,Nt,so3_dim]
        x_rest = x_t[..., self.so3_dim :]     # [B,Hh,S,Nt,Dh-so3_dim]

        # reshape into (..., reps, 3)
        x_so3 = x_so3.view(B, Hh, S, Nt, self.reps, 3)

        # apply only to template image tokens [0:num_img)
        x_img = x_so3[:, :, :, :num_img, :, :]  # [B,Hh,S,num_img,reps,3]
        x_img = torch.einsum("bsij,bhsnkj->bhsnki", Rt, x_img)  # multiply last dim 3

        # keep special tokens unchanged
        if num_img < Nt:
            x_sp = x_so3[:, :, :, num_img:, :, :]
            x_so3 = torch.cat([x_img, x_sp], dim=3)
        else:
            x_so3 = x_img

        x_so3 = x_so3.contiguous().view(B, Hh, S, Nt, self.so3_dim)
        return torch.cat([x_so3, x_rest], dim=-1)

