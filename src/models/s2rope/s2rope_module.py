# s2rope_module.py
import torch

try:
    import s2rope_ext as _kernels  # after `python setup_s2rope.py install`
except ModuleNotFoundError:
    from . import s2rope_ext as _kernels  # after in-place build


class S2RoPEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, phase_x, phase_y, phase_sph):
        """
        tokens:   [B, N, H, D]  (float32, in-place rotated)
        phase_x:  [B, N, Px]    or None / empty
        phase_y:  [B, N, Py]
        phase_sph:[B, N, F, Ps]
        """
        if phase_x is None:
            phase_x = torch.empty(0, device=tokens.device, dtype=tokens.dtype)
        if phase_y is None:
            phase_y = torch.empty(0, device=tokens.device, dtype=tokens.dtype)
        if phase_sph is None:
            phase_sph = torch.empty(0, device=tokens.device, dtype=tokens.dtype)

        ctx.save_for_backward(phase_x, phase_y, phase_sph)
        # in-place rotation
        _kernels.s2rope(tokens, phase_x, phase_y, phase_sph, 1.0)
        ctx.mark_dirty(tokens)
        return tokens

    @staticmethod
    def backward(ctx, grad_tokens):
        phase_x, phase_y, phase_sph = ctx.saved_tensors
        # apply inverse rotation in-place on gradient
        _kernels.s2rope(grad_tokens, phase_x, phase_y, phase_sph, -1.0)
        ctx.mark_dirty(grad_tokens)
        return grad_tokens, None, None, None


class S2RoPE(torch.nn.Module):
    """
    Generic S2 RoPE module.

    It assumes that the last dim D of tokens is:
        D = 2 * (Px + Py + F * Ps)

    You are responsible for:
      - choosing Px, Py, F, Ps (via constructor)
      - arranging tokens accordingly
      - providing phase_x, phase_y, phase_sph in forward()
    """
    def __init__(
        self,
        head_dim: int = 1024,
        n_faces: int = 6,
        px: int = None,
        py: int = None,
        ps: int = None,
    ):
        super().__init__()
        self.n_faces = n_faces

        total_pairs = head_dim // 2  # complex slots
        # Default split ~ our earlier suggestion:
        # ~53% pairs to xy, ~47% to S^2, equal split x/y, equal per face
        if px is None or py is None or ps is None:
            xy_pairs_total = int(round(total_pairs * 0.53))
            xy_pairs_total = max(xy_pairs_total, 2)
            # ensure even for x,y split
            if xy_pairs_total % 2 != 0:
                xy_pairs_total -= 1
            px_default = xy_pairs_total // 2
            py_default = xy_pairs_total // 2
            remaining_pairs = total_pairs - (px_default + py_default)
            ps_default = max(1, remaining_pairs // n_faces)
            self.px = px_default
            self.py = py_default
            self.ps = ps_default
        else:
            self.px = px
            self.py = py
            self.ps = ps

        # for sanity
        expected_head_dim = 2 * (self.px + self.py + self.n_faces * self.ps)
        if expected_head_dim != head_dim:
            raise ValueError(
                f"head_dim={head_dim} inconsistent with px={self.px}, "
                f"py={self.py}, ps={self.ps}, n_faces={self.n_faces}; "
                f"expected D=2*(px+py+F*ps)={expected_head_dim}"
            )

    def forward(self, tokens, phase_x=None, phase_y=None, phase_sph=None):
        """
        tokens:    [B, N, H, D]
        phase_x:   [B, N, px]   or None
        phase_y:   [B, N, py]   or None
        phase_sph: [B, N, F, ps] or None
        """
        if tokens.dim() != 4:
            raise ValueError("tokens must have shape [B,N,H,D]")
        D = tokens.size(-1)
        expected_D = 2 * (self.px + self.py + self.n_faces * self.ps)
        if D != expected_D:
            raise ValueError(
                f"tokens.shape[-1]={D} but expected {expected_D} "
                f"for px={self.px}, py={self.py}, ps={self.ps}, F={self.n_faces}"
            )

        return S2RoPEFunc.apply(tokens, phase_x, phase_y, phase_sph)
