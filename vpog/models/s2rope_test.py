# ======================================================================
# s2rope_test.py  —  Exhaustive correctness tests for S2RoPE
# CPU reference vs CUDA, geometry checks, gradient checks, torch-ref check
# ======================================================================

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.s2rope.s2rope_module import S2RoPE
from src.models.pos_embed import (
    S2RopePositionalEncoding,
    S2RopePositionalEncodingTorch,
)

EPS = 1e-6


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def assert_close(x, y, tol=1e-5, msg=""):
    diff = (x - y).abs().max().item()
    print(f"{msg} max diff = {diff}")
    if diff > tol:
        raise AssertionError(f"FAILED: {msg} — diff={diff}")


def _random_dirs(B, N, device):
    v = torch.randn(B, N, 3, device=device)
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)


# ----------------------------------------------------------------------
# 1. X-only, CPU vs CUDA
# ----------------------------------------------------------------------
def test_x_only():
    print("\n==== test_x_only ====")
    if not torch.cuda.is_available():
        print("CUDA missing → skipping")
        return

    B, N, H = 2, 5, 3
    Px, Py, Ps, F = 16, 0, 0, 0
    head_dim = 2 * Px

    tokens = torch.randn(B, N, H, head_dim)
    phase_x = torch.randn(B, N, Px)
    phase_y = torch.zeros(0)
    phase_sph = torch.zeros(0)

    rope_cpu = S2RoPE(head_dim=head_dim, px=Px, py=Py, ps=Ps)
    out_cpu = rope_cpu(tokens.clone(), phase_x, phase_y, phase_sph)

    rope_cuda = S2RoPE(head_dim=head_dim, px=Px, py=Py, ps=Ps).cuda()
    out_cuda = rope_cuda(
        tokens.clone().cuda(),
        phase_x.cuda(),
        phase_y.cuda(),
        phase_sph.cuda(),
    ).cpu()

    assert_close(out_cpu, out_cuda, tol=1e-5, msg="x-only CPU vs CUDA")


# ----------------------------------------------------------------------
# 2. Y-only, CPU vs CUDA
# ----------------------------------------------------------------------
def test_y_only():
    print("\n==== test_y_only ====")
    if not torch.cuda.is_available():
        print("CUDA missing → skipping")
        return

    B, N, H = 2, 5, 3
    Px, Py, Ps, F = 0, 16, 0, 0
    head_dim = 2 * Py

    tokens = torch.randn(B, N, H, head_dim)
    phase_y = torch.randn(B, N, Py)
    phase_x = torch.zeros(0)
    phase_sph = torch.zeros(0)

    rope_cpu = S2RoPE(head_dim=head_dim, px=Px, py=Py, ps=Ps)
    out_cpu = rope_cpu(tokens.clone(), phase_x, phase_y, phase_sph)

    rope_cuda = S2RoPE(head_dim=head_dim, px=Px, py=Py, ps=Ps).cuda()
    out_cuda = rope_cuda(
        tokens.clone().cuda(),
        phase_x.cuda(),
        phase_y.cuda(),
        phase_sph.cuda(),
    ).cpu()

    assert_close(out_cpu, out_cuda, tol=1e-5, msg="y-only CPU vs CUDA")


# ----------------------------------------------------------------------
# 3. S²-only CPU vs CUDA
# ----------------------------------------------------------------------
def test_s2_only():
    print("\n==== test_s2_only ====")
    if not torch.cuda.is_available():
        print("CUDA missing → skipping")
        return

    B, N, H = 2, 5, 3
    Px, Py, F, Ps = 0, 0, 6, 8
    head_dim = 2 * (F * Ps)

    tokens = torch.randn(B, N, H, head_dim)
    phase_sph = torch.randn(B, N, F, Ps)
    phase_x = torch.zeros(0)
    phase_y = torch.zeros(0)

    rope_cpu = S2RoPE(head_dim=head_dim, px=Px, py=Py, ps=Ps)
    out_cpu = rope_cpu(tokens.clone(), phase_x, phase_y, phase_sph)

    rope_cuda = S2RoPE(head_dim=head_dim, px=Px, py=Py, ps=Ps).cuda()
    out_cuda = rope_cuda(
        tokens.clone().cuda(),
        phase_x.cuda(),
        phase_y.cuda(),
        phase_sph.cuda(),
    ).cpu()

    assert_close(out_cpu, out_cuda, tol=1e-5, msg="S2-only CPU vs CUDA")


# ----------------------------------------------------------------------
# 4. Full X+Y+S² CPU vs CUDA
# ----------------------------------------------------------------------
def test_full_xy_s2():
    print("\n==== test_full_xy_s2 ====")
    if not torch.cuda.is_available():
        print("CUDA missing → skipping")
        return

    B, N, H = 2, 5, 3
    Px, Py, Ps, F = 8, 8, 4, 6
    head_dim = 2 * (Px + Py + F * Ps)

    tokens = torch.randn(B, N, H, head_dim)
    phase_x = torch.randn(B, N, Px)
    phase_y = torch.randn(B, N, Py)
    phase_sph = torch.randn(B, N, F, Ps)

    rope_cpu = S2RoPE(head_dim=head_dim, px=Px, py=Py, ps=Ps)
    out_cpu = rope_cpu(tokens.clone(), phase_x, phase_y, phase_sph)

    rope_cuda = S2RoPE(head_dim=head_dim, px=Px, py=Py, ps=Ps).cuda()
    out_cuda = rope_cuda(
        tokens.clone().cuda(),
        phase_x.cuda(),
        phase_y.cuda(),
        phase_sph.cuda(),
    ).cpu()

    assert_close(out_cpu, out_cuda, tol=1e-5, msg="Full XY+S2 CPU vs CUDA")


# ----------------------------------------------------------------------
# 5. Norm preservation (CPU)
# ----------------------------------------------------------------------
def test_norm_preservation():
    print("\n==== test_norm_preservation (CPU) ====")

    B, N, H = 2, 5, 3
    Px, Py, Ps, F = 8, 8, 4, 6
    head_dim = 2 * (Px + Py + F * Ps)

    tokens = torch.randn(B, N, H, head_dim)
    phase_x = torch.randn(B, N, Px)
    phase_y = torch.randn(B, N, Py)
    phase_sph = torch.randn(B, N, F, Ps)

    rope_cpu = S2RoPE(head_dim=head_dim, px=Px, py=Py, ps=Ps)
    before = tokens.clone()
    after = rope_cpu(tokens.clone(), phase_x, phase_y, phase_sph)

    diff = ((after ** 2).sum(-1) - (before ** 2).sum(-1)).abs().max().item()
    print("norm diff =", diff)
    assert diff < 1e-4


# ----------------------------------------------------------------------
# 6. S² geometry consistency (CPU) — local ref invariance
# ----------------------------------------------------------------------
def test_s2_local_ref_invariance():
    """
    Check local consistency of the S² RoPE under a change of reference direction,
    when everything stays on the SAME cube face (+Z).
    """
    print("\n==== test_s2_local_ref_invariance ====")

    device = "cpu"
    head_dim = 256
    pe = S2RopePositionalEncoding(head_dim=head_dim).to(device)

    def v_alpha(alpha_deg: float) -> torch.Tensor:
        alpha = math.radians(alpha_deg)
        return torch.tensor(
            [math.sin(alpha), 0.0, math.cos(alpha)],
            dtype=torch.float32,
            device=device,
        )

    # Two view directions on +Z face
    v1 = v_alpha(15.0)
    v2 = v_alpha(45.0)

    # Two nearby reference directions
    d_ref_A = v_alpha(10.0)
    d_ref_B = v_alpha(20.0)

    B, N = 1, 2
    dirs = torch.stack([v1, v2], dim=0).view(1, N, 3)  # [1,2,3]

    s2_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    pos = torch.zeros(B, N, 2, device=device)

    tokens = torch.arange(B * N * head_dim, dtype=torch.float32, device=device)
    tokens = tokens.view(B, N, head_dim)

    refA = d_ref_A.view(1, 3)
    refB = d_ref_B.view(1, 3)

    outA = pe(tokens.clone(), pos, dirs, refA, s2_mask)
    outB = pe(tokens.clone(), pos, dirs, refB, s2_mask)

    dA = (outA[:, 0, :] - outA[:, 1, :]).norm(dim=-1).item()
    dB = (outB[:, 0, :] - outB[:, 1, :]).norm(dim=-1).item()

    print(f"dA (ref A) = {dA:.6f}, dB (ref B) = {dB:.6f}")

    if min(dA, dB) < 1e-6:
        raise AssertionError("Distances collapsed to ~0; bad S² behavior.")

    ratio = max(dA, dB) / (min(dA, dB) + 1e-12)
    print(f"ratio = {ratio:.6f}")
    assert ratio < 1.1, f"S² local ref invariance too distorted, ratio={ratio}"


# ----------------------------------------------------------------------
# 7. Attention + backward test (CUDA if available)
# ----------------------------------------------------------------------
class MiniAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.pe = S2RopePositionalEncoding(head_dim=d)

    def forward(self, x, pos, vd, ref):
        """
        x:   [B,N,D]
        pos: [B,N,2]
        vd:  [B,N,3]
        ref: [B,1,3] or [B,3]
        """
        B, N, D = x.shape
        device = x.device

        s2_mask = torch.ones(B, N, dtype=torch.bool, device=device)

        if ref.dim() == 3:
            ref_dirs = ref[:, 0, :]  # [B,3]
        else:
            ref_dirs = ref

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = self.pe(q, pos, vd, ref_dirs, s2_mask)
        k = self.pe(k, pos, vd, ref_dirs, s2_mask)

        att = (q @ k.transpose(1, 2)) / math.sqrt(D)
        att = att.softmax(-1)
        return self.o(att @ v)


def test_attention_backward():
    print("\n==== test_attention_backward ====")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, N, D = 2, 8, 256
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    pos = torch.zeros(B, N, 2, device=device)
    vd = _random_dirs(B, N, device)
    ref = _random_dirs(B, 1, device)  # [B,1,3]

    model = MiniAttn(D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    y = model(x, pos, vd, ref)
    loss = y.mean()
    print("loss:", loss.item())

    opt.zero_grad()
    loss.backward()
    opt.step()

    assert torch.isfinite(x.grad).all()


# ----------------------------------------------------------------------
# 8. Kernel vs pure-torch S2RopePositionalEncoding
# ----------------------------------------------------------------------
def test_s2rope_vs_torch_ref():
    print("\n==== test_s2rope_vs_torch_ref ====")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    B, N, D = 2, 128, 1024

    tokens = torch.randn(B, N, D, device=device)
    pos = torch.zeros(B, N, 2, device=device)

    vd = _random_dirs(B, N, device)
    ref = _random_dirs(B, 1, device)  # [B,1,3]
    ref_dirs = ref[:, 0, :]           # [B,3]

    s2_mask = torch.ones(B, N, dtype=torch.bool, device=device)

    pe_kernel = S2RopePositionalEncoding(head_dim=D).to(device).eval()
    pe_torch = S2RopePositionalEncodingTorch(head_dim=D).to(device).eval()

    with torch.no_grad():
        out_kernel = pe_kernel(tokens.clone(), pos, vd, ref_dirs, s2_mask)
        out_torch = pe_torch(tokens.clone(), pos, vd, ref_dirs, s2_mask)

    diff = (out_kernel - out_torch).abs().max().item()
    print(f"Kernel vs Torch ref max diff: {diff:.6e}")
    assert diff < 1e-5

    # Runtime comparison on CUDA if available
    if torch.cuda.is_available():
        print("Benchmarking on CUDA...")
        n_runs = 50

        tokens_cuda = tokens
        pos_cuda = pos
        vd_cuda = vd
        ref_cuda = ref_dirs
        s2_mask_cuda = s2_mask

        # Warm-up
        with torch.no_grad():
            _ = pe_kernel(tokens_cuda, pos_cuda, vd_cuda, ref_cuda, s2_mask_cuda)
            _ = pe_torch(tokens_cuda, pos_cuda, vd_cuda, ref_cuda, s2_mask_cuda)
            torch.cuda.synchronize()

        # Kernel timing
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = pe_kernel(tokens_cuda, pos_cuda, vd_cuda, ref_cuda, s2_mask_cuda)
            torch.cuda.synchronize()
        kernel_time = (time.time() - t0) / n_runs

        # Torch timing
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = pe_torch(tokens_cuda, pos_cuda, vd_cuda, ref_cuda, s2_mask_cuda)
            torch.cuda.synchronize()
        torch_time = (time.time() - t0) / n_runs

        print(f"Avg kernel-based S2RoPE time: {kernel_time*1000:.3f} ms")
        print(f"Avg torch-only S2RoPE time:  {torch_time*1000:.3f} ms")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    test_x_only()
    test_y_only()
    test_s2_only()
    test_full_xy_s2()
    test_norm_preservation()
    test_s2_local_ref_invariance()
    test_attention_backward()
    test_s2rope_vs_torch_ref()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
