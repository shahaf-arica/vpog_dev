# # so3prope.py
# from __future__ import annotations

# import torch
# from typing import Optional, Tuple


# class SO3PRoPE(torch.nn.Module):
#     """
#     SO(3)-only PRoPE:
#     - Applies rotation encoding (SO3) using camera rotation matrices
#     - Applies 2D RoPE on image tokens only
#     - Supports arbitrary number of special tokens per template
#     """

#     def __init__(
#         self,
#         *,
#         head_dim: int,
#         patches_x: int,
#         patches_y: int,
#         apply_so3_to_special: bool = True,
#         apply_rope2d_to_special: bool = False,
#         rope_freq_base: float = 100.0,
#         rope_freq_scale: float = 1.0,
#     ):
#         super().__init__()

#         assert head_dim % 4 == 0, "head_dim must be divisible by 4"
#         self.head_dim = head_dim
#         self.patches_x = patches_x
#         self.patches_y = patches_y

#         # channel split (same spirit as PRoPE)
#         self.dim_so3 = head_dim // 2
#         self.dim_rope_x = head_dim // 4
#         self.dim_rope_y = head_dim // 4

#         # SO3 block must be divisible by 3
#         assert self.dim_so3 % 3 == 0, "SO3 block must be divisible by 3"
#         self.so3_repeats = self.dim_so3 // 3

#         self.apply_so3_to_special = apply_so3_to_special
#         self.apply_rope2d_to_special = apply_rope2d_to_special

#         # Precompute 2D RoPE coefficients for image tokens
#         self.register_buffer(
#             "rope_x_cos",
#             self._build_rope_coeffs_x(freq_base=rope_freq_base, freq_scale=rope_freq_scale),
#             persistent=False,
#         )
#         self.register_buffer(
#             "rope_x_sin",
#             self._build_rope_coeffs_x(freq_base=rope_freq_base, freq_scale=rope_freq_scale, sin=True),
#             persistent=False,
#         )
#         self.register_buffer(
#             "rope_y_cos",
#             self._build_rope_coeffs_y(freq_base=rope_freq_base, freq_scale=rope_freq_scale),
#             persistent=False,
#         )
#         self.register_buffer(
#             "rope_y_sin",
#             self._build_rope_coeffs_y(freq_base=rope_freq_base, freq_scale=rope_freq_scale, sin=True),
#             persistent=False,
#         )

#     # ------------------------------------------------------------
#     # Public API
#     # ------------------------------------------------------------

#     def apply_to_qkv(
#         self,
#         q: torch.Tensor,
#         k: torch.Tensor,
#         v: torch.Tensor,
#         *,
#         rotations: torch.Tensor,
#         num_special_tokens: int,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         q,k,v: (B, H, N, head_dim)
#         rotations: (B, 3, 3)  -- one rotation per template (already selected per template group)
#         """

#         q = self._apply(q, rotations, num_special_tokens, mode="q")
#         k = self._apply(k, rotations, num_special_tokens, mode="k")
#         v = self._apply(v, rotations, num_special_tokens, mode="v")
#         return q, k, v

#     # ------------------------------------------------------------
#     # Core logic
#     # ------------------------------------------------------------

#     def _apply(
#         self,
#         x: torch.Tensor,
#         rotations: torch.Tensor,
#         num_special_tokens: int,
#         *,
#         mode: str,
#     ) -> torch.Tensor:
#         """
#         mode: 'q', 'k', or 'v'
#         """
#         B, H, N, D = x.shape
#         HW = self.patches_x * self.patches_y
#         assert D == self.head_dim

#         # Split channels
#         x_so3, x_rx, x_ry = torch.split(
#             x, [self.dim_so3, self.dim_rope_x, self.dim_rope_y], dim=-1
#         )

#         # --------------------------------------------------
#         # SO(3) block
#         # --------------------------------------------------
#         if mode == "q":
#             R = rotations.transpose(-1, -2)  # R^T
#         else:
#             R = rotations  # R^{-1} == R^T if rotations are camera->object, but we keep symmetric

#         x_so3 = self._apply_so3(
#             x_so3,
#             R,
#             HW,
#             num_special_tokens,
#             apply_to_special=self.apply_so3_to_special,
#         )

#         # --------------------------------------------------
#         # 2D RoPE blocks
#         # --------------------------------------------------
#         x_rx = self._apply_rope(
#             x_rx,
#             self.rope_x_cos,
#             self.rope_x_sin,
#             HW,
#             num_special_tokens,
#             apply_to_special=self.apply_rope2d_to_special,
#             inverse=(mode == "o"),
#         )

#         x_ry = self._apply_rope(
#             x_ry,
#             self.rope_y_cos,
#             self.rope_y_sin,
#             HW,
#             num_special_tokens,
#             apply_to_special=self.apply_rope2d_to_special,
#             inverse=(mode == "o"),
#         )

#         return torch.cat([x_so3, x_rx, x_ry], dim=-1)

#     # ------------------------------------------------------------
#     # SO(3) application
#     # ------------------------------------------------------------

#     def _apply_so3(
#         self,
#         x: torch.Tensor,
#         R: torch.Tensor,
#         HW: int,
#         num_special: int,
#         *,
#         apply_to_special: bool,
#     ) -> torch.Tensor:
#         """
#         x: (B,H,N,dim_so3)
#         R: (B,3,3)
#         """

#         B, H, N, _ = x.shape
#         x = x.view(B, H, N, self.so3_repeats, 3)

#         # image tokens
#         x_img = torch.einsum("bij,bhnkj->bhnki", R, x[:, :, :HW])

#         if num_special == 0:
#             return x_img.view(B, H, HW, self.dim_so3)

#         x_special = x[:, :, HW : HW + num_special]

#         if apply_to_special:
#             x_special = torch.einsum("bij,bhnkj->bhnki", R, x_special)

#         x_out = torch.cat([x_img, x_special], dim=2)
#         return x_out.view(B, H, HW + num_special, self.dim_so3)

#     # ------------------------------------------------------------
#     # RoPE application
#     # ------------------------------------------------------------

#     def _apply_rope(
#         self,
#         x: torch.Tensor,
#         cos: torch.Tensor,
#         sin: torch.Tensor,
#         HW: int,
#         num_special: int,
#         *,
#         apply_to_special: bool,
#         inverse: bool = False,
#     ) -> torch.Tensor:
#         B, H, N, D = x.shape
#         half = D // 2

#         x1 = x[..., :half]
#         x2 = x[..., half:]

#         cos = cos[:HW].view(1, 1, HW, half)
#         sin = sin[:HW].view(1, 1, HW, half)

#         if inverse:
#             y1 = cos * x1[:, :, :HW] - sin * x2[:, :, :HW]
#             y2 = sin * x1[:, :, :HW] + cos * x2[:, :, :HW]
#         else:
#             y1 = cos * x1[:, :, :HW] + sin * x2[:, :, :HW]
#             y2 = -sin * x1[:, :, :HW] + cos * x2[:, :, :HW]

#         x_img = torch.cat([y1, y2], dim=-1)

#         if num_special == 0:
#             return x_img

#         x_special = x[:, :, HW : HW + num_special]
#         if apply_to_special:
#             return torch.cat([x_img, x_special], dim=2)
#         else:
#             return torch.cat([x_img, x_special], dim=2)

#     # ------------------------------------------------------------
#     # RoPE coeff builders
#     # ------------------------------------------------------------

#     def _build_rope_coeffs_x(self, *, freq_base, freq_scale, sin=False):
#         x = torch.tile(torch.arange(self.patches_x), (self.patches_y,))
#         return self._rope_coeffs(x, self.dim_rope_x // 2, freq_base, freq_scale, sin)

#     def _build_rope_coeffs_y(self, *, freq_base, freq_scale, sin=False):
#         y = torch.repeat_interleave(torch.arange(self.patches_y), self.patches_x)
#         return self._rope_coeffs(y, self.dim_rope_y // 2, freq_base, freq_scale, sin)

#     @staticmethod
#     def _rope_coeffs(pos, dim, freq_base, freq_scale, sin):
#         freqs = freq_scale * freq_base ** (
#             -torch.arange(dim, device=pos.device) / dim
#         )
#         angles = pos[:, None] * freqs[None]
#         return torch.sin(angles) if sin else torch.cos(angles)
