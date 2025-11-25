"""
s2rope_seq_test.py

Sanity tests for:
  - S2RoPE kernel (CU/CPP)
  - S2RopePositionalEncoding (single image)
  - S2RopeSequencePositionalEncoding (sequence [B,S,T,C])

Covers:
  * Determinism
  * XY-only vs S² behavior
  * Support vs query frames (frame_has_s2)
  * Per-token S² masking
  * CPU vs CUDA consistency (if CUDA available)
  * Backward / gradient sanity
  * Basic CPU vs CUDA timing for sequence
"""

import time
import torch
import torch.nn.functional as F

from src.models.pos_embed import (
    S2RopePositionalEncoding,
    S2RopeSequencePositionalEncoding,
)
from src.models.s2rope.s2rope_module import S2RoPE


# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------
def has_cuda() -> bool:
    return torch.cuda.is_available()


def device_str():
    return "cuda" if has_cuda() else "cpu"


def build_single_dummy(
    B: int = 2,
    N: int = 128,
    C: int = 1024,
    with_view_dirs: bool = True,
    device=None,
):
    """
    Simple [B,N,C] dummy for single-image S2RopePositionalEncoding tests.
    pos: fake integer (y,x) grid folded into N.
    """
    if device is None:
        device = torch.device(device_str())

    torch.manual_seed(0)
    tokens = torch.randn(B, N, C, device=device)
    # make a square-ish grid for positions if possible
    H = int(N**0.5)
    W = N // H
    if H * W < N:
        W += 1
    ys = torch.arange(H, device=device)
    xs = torch.arange(W, device=device)
    grid = torch.cartesian_prod(ys, xs)[:N]  # [N,2]
    pos = grid.view(1, N, 2).expand(B, -1, 2).clone()  # [B,N,2]

    if with_view_dirs:
        view_dirs = F.normalize(torch.randn(B, N, 3, device=device), dim=-1)
    else:
        view_dirs = None

    ref_dirs = F.normalize(torch.randn(B, 3, device=device), dim=-1)

    return tokens, pos, view_dirs, ref_dirs


def build_sequence_dummy(
    B: int = 2,
    S: int = 8,
    C: int = 1024,
    H: int = 14,
    W: int = 14,
    extra_tokens: int = 0,
    device=None,
):
    """
    Returns:
      tokens:       [B,S,T,C]
      pos:          [B,S,T,2]
      frame_dirs:   [B,S,3]
      frame_has_s2: [B,S] (False for frame 0, True for 1..S-1)
      ref_dirs:     [B,3]
    """
    if device is None:
        device = torch.device(device_str())

    torch.manual_seed(1)

    # Fake encoder output [B,S,C,H,W]
    enc = torch.randn(B, S, C, H, W, device=device)

    # Spatial tokens [B,S,H*W,C]
    spatial = enc.permute(0, 1, 3, 4, 2).contiguous().reshape(B, S, H * W, C)

    if extra_tokens > 0:
        extra = torch.randn(B, S, extra_tokens, C, device=device)
        tokens = torch.cat([extra, spatial], dim=2)
    else:
        tokens = spatial

    B_, S_, T, C_ = tokens.shape
    assert (B_, S_, C_) == (B, S, C)
    T = tokens.shape[2]

    pos = torch.zeros(B, S, T, 2, device=device, dtype=torch.long)
    ys = torch.arange(H, device=device)
    xs = torch.arange(W, device=device)
    grid = torch.cartesian_prod(ys, xs).view(1, 1, H * W, 2).expand(B, S, -1, 2)
    pos[:, :, extra_tokens:, :] = grid

    frame_dirs = F.normalize(torch.randn(B, S, 3, device=device), dim=-1)

    frame_has_s2 = torch.zeros(B, S, dtype=torch.bool, device=device)
    frame_has_s2[:, 1:] = True  # frame 0 is query

    ref_dirs = F.normalize(torch.randn(B, 3, device=device), dim=-1)

    return tokens, pos, frame_dirs, frame_has_s2, ref_dirs


# -------------------------------------------------------------
# Test 1: S2RoPE kernel – XY + S² determinism & effect
# -------------------------------------------------------------
def test_kernel_s2rope_basic():
    print("\n== test_kernel_s2rope_basic ==")
    device = torch.device(device_str())
    torch.manual_seed(0)

    B, N, H_heads, D = 2, 16, 1, 1024

    # IMPORTANT: do NOT rely on S2RoPE's internal auto-split (it’s inconsistent).
    # Use the same split logic as S2RopePositionalEncoding:
    #   D_pairs = D // 2 = 512
    #   ps = max(1, (D_pairs // 4) // n_faces) = 21
    #   s2_pairs = ps * n_faces = 126
    #   remaining = 512 - 126 = 386
    #   px = 193, py = 193
    px = 193
    py = 193
    ps = 21
    n_faces = 6

    rope = S2RoPE(head_dim=D, n_faces=n_faces, px=px, py=py, ps=ps).to(device)
    print(f"px={rope.px}, py={rope.py}, ps={rope.ps}, F={rope.n_faces}")

    tokens = torch.randn(B, N, H_heads, D, device=device)

    # XY phases
    phase_x = torch.randn(B, N, rope.px, device=device)
    phase_y = torch.randn(B, N, rope.py, device=device)

    # S² phases
    phase_sph1 = torch.zeros(B, N, rope.n_faces, rope.ps, device=device)
    phase_sph2 = torch.randn(B, N, rope.n_faces, rope.ps, device=device)

    with torch.no_grad():
        out1 = rope(tokens.clone(), phase_x, phase_y, phase_sph1).clone()
        out2 = rope(tokens.clone(), phase_x, phase_y, phase_sph2).clone()
        out3 = rope(tokens.clone(), phase_x, phase_y, phase_sph2).clone()

    # determinism given same inputs
    diff_same = (out2 - out3).abs().max().item()
    print(f"max diff same phases: {diff_same:.6e}")
    assert diff_same < 1e-7

    # S² should change output
    diff_s2 = (out1 - out2).abs().max().item()
    print(f"max diff different S² phases: {diff_s2:.6e}")
    assert diff_s2 > 1e-5

    print("OK")


# -------------------------------------------------------------
# Test 2: Single-image S2RopePositionalEncoding – XY only
# -------------------------------------------------------------
def test_single_xy_only():
    print("\n== test_single_xy_only ==")
    device = torch.device(device_str())
    torch.manual_seed(0)

    B, N, C = 2, 128, 1024
    tokens, pos, view_dirs, ref_dirs = build_single_dummy(
        B=B, N=N, C=C, with_view_dirs=False, device=device
    )

    pe = S2RopePositionalEncoding(head_dim=C).to(device)
    pe.eval()

    with torch.no_grad():
        out1 = pe(tokens.clone(), pos, view_dirs=None, ref_dirs=None, s2_mask=None)
        out2 = pe(tokens.clone(), pos, view_dirs=None, ref_dirs=None, s2_mask=None)

    assert out1.shape == tokens.shape
    diff = (out1 - out2).abs().max().item()
    print(f"max diff (xy-only, same input): {diff:.6e}")
    assert diff < 1e-7

    print("OK")


# -------------------------------------------------------------
# Test 3: Single-image S2RopePositionalEncoding – S² effect
# -------------------------------------------------------------
def test_single_s2_effect():
    print("\n== test_single_s2_effect ==")
    device = torch.device(device_str())
    torch.manual_seed(0)

    B, N, C = 2, 128, 1024
    tokens, pos, view_dirs, ref_dirs = build_single_dummy(
        B=B, N=N, C=C, with_view_dirs=True, device=device
    )

    s2_mask = torch.ones(B, N, dtype=torch.bool, device=device)

    pe = S2RopePositionalEncoding(head_dim=C).to(device)
    pe.eval()

    with torch.no_grad():
        out1 = pe(tokens.clone(), pos, view_dirs, ref_dirs, s2_mask)

    # change view_dirs
    view_dirs2 = view_dirs.clone()
    noise = torch.randn_like(view_dirs2) * 0.5
    view_dirs2 = F.normalize(view_dirs2 + noise, dim=-1)

    with torch.no_grad():
        out2 = pe(tokens.clone(), pos, view_dirs2, ref_dirs, s2_mask)

    diff = (out1 - out2).abs().amax(dim=-1)  # [B,N]
    max_diff = diff.max().item()
    print(f"max diff single-image S² change: {max_diff:.6e}")
    assert max_diff > 1e-5

    print("OK")


# -------------------------------------------------------------
# Test 4: Sequence – support vs query frames
# -------------------------------------------------------------
def test_seq_support_vs_query():
    print("\n== test_seq_support_vs_query ==")
    device = torch.device(device_str())
    torch.manual_seed(1)

    tokens, pos, frame_dirs, frame_has_s2, ref_dirs = build_sequence_dummy(
        B=2, S=8, C=1024, H=14, W=14, extra_tokens=0, device=device
    )
    B, S, T, C = tokens.shape

    seq_pos = S2RopeSequencePositionalEncoding(head_dim=C).to(device)
    seq_pos.eval()

    with torch.no_grad():
        out1 = seq_pos(tokens.clone(), pos, frame_dirs, frame_has_s2, ref_dirs)

    # modify frame_dirs only for support frames in batch=1
    frame_dirs2 = frame_dirs.clone()
    noise = torch.randn_like(frame_dirs2[1, 1:, :]) * 0.5
    frame_dirs2[1, 1:, :] = F.normalize(frame_dirs2[1, 1:, :] + noise, dim=-1)

    with torch.no_grad():
        out2 = seq_pos(tokens.clone(), pos, frame_dirs2, frame_has_s2, ref_dirs)

    diff = (out1 - out2).abs().amax(dim=-1)  # [B,S,T]
    support_diff = diff[1, 1:, :].max().item()
    query_diff = diff[1, 0, :].max().item()

    print(f"support diff (batch 1, frames 1..S-1): {support_diff:.6e}")
    print(f"query   diff (batch 1, frame 0):       {query_diff:.6e}")

    assert support_diff > 1e-5
    assert query_diff < support_diff * 0.5
    print("OK")


# -------------------------------------------------------------
# Test 5: Sequence – per-token S² mask
# -------------------------------------------------------------
def test_seq_token_s2_mask():
    print("\n== test_seq_token_s2_mask ==")
    device = torch.device(device_str())
    torch.manual_seed(2)

    tokens, pos, frame_dirs, frame_has_s2, ref_dirs = build_sequence_dummy(
        B=2, S=4, C=1024, H=8, W=8, extra_tokens=4, device=device
    )
    B, S, T, C = tokens.shape

    # enable S² in all frames for this test
    frame_has_s2[:, :] = True

    # token_s2_mask: we mask half tokens in frame 1, batch 0
    token_s2_mask = torch.ones(B, S, T, dtype=torch.bool, device=device)
    half = T // 2
    token_s2_mask[0, 1, :half] = False

    seq_pos = S2RopeSequencePositionalEncoding(head_dim=C).to(device)
    seq_pos.eval()

    with torch.no_grad():
        out1 = seq_pos(tokens.clone(), pos, frame_dirs, frame_has_s2, ref_dirs, token_s2_mask)

    # change frame_dirs for batch 0 (all frames)
    frame_dirs2 = frame_dirs.clone()
    noise = torch.randn_like(frame_dirs2[0]) * 0.5
    frame_dirs2[0] = F.normalize(frame_dirs2[0] + noise, dim=-1)

    with torch.no_grad():
        out2 = seq_pos(tokens.clone(), pos, frame_dirs2, frame_has_s2, ref_dirs, token_s2_mask)

    diff = (out1 - out2).abs().amax(dim=-1)  # [B,S,T]
    diffs_frame1 = diff[0, 1]                # [T]

    masked = ~token_s2_mask[0, 1]
    unmasked = token_s2_mask[0, 1]

    masked_max = diffs_frame1[masked].max().item()
    unmasked_max = diffs_frame1[unmasked].max().item()

    print(f"masked   tokens diff:   {masked_max:.6e}")
    print(f"unmasked tokens diff:   {unmasked_max:.6e}")

    assert unmasked_max > 1e-5
    assert masked_max < unmasked_max * 0.2
    print("OK")


# -------------------------------------------------------------
# Test 6: CPU vs CUDA – single-image encoding
# -------------------------------------------------------------
def test_cpu_cuda_single():
    print("\n== test_cpu_cuda_single ==")
    if not has_cuda():
        print("CUDA not available – skipping.")
        return

    torch.manual_seed(3)

    B, N, C = 2, 128, 1024

    # Build on CPU and copy to CUDA to ensure exact same values
    tokens_cpu, pos_cpu, vdirs_cpu, ref_cpu = build_single_dummy(
        B=B, N=N, C=C, with_view_dirs=True, device=torch.device("cpu")
    )
    s2_mask_cpu = torch.ones(B, N, dtype=torch.bool)

    tokens_cuda = tokens_cpu.to("cuda")
    pos_cuda = pos_cpu.to("cuda")
    vdirs_cuda = vdirs_cpu.to("cuda")
    ref_cuda = ref_cpu.to("cuda")
    s2_mask_cuda = s2_mask_cpu.to("cuda")

    pe_cpu = S2RopePositionalEncoding(head_dim=C).to("cpu").eval()
    pe_cuda = S2RopePositionalEncoding(head_dim=C).to("cuda").eval()

    with torch.no_grad():
        out_cpu = pe_cpu(tokens_cpu, pos_cpu, vdirs_cpu, ref_cpu, s2_mask_cpu)
        out_cuda = pe_cuda(tokens_cuda, pos_cuda, vdirs_cuda, ref_cuda, s2_mask_cuda)
        out_cuda_cpu = out_cuda.to("cpu")

    max_diff = (out_cpu - out_cuda_cpu).abs().max().item()
    print(f"CPU vs CUDA max diff (single-image): {max_diff:.6e}")
    assert max_diff < 1e-5

    print("OK")


# -------------------------------------------------------------
# Test 7: CPU vs CUDA – sequence encoding
# -------------------------------------------------------------
def test_cpu_cuda_sequence():
    print("\n== test_cpu_cuda_sequence ==")
    if not has_cuda():
        print("CUDA not available – skipping.")
        return

    torch.manual_seed(4)

    B, S, C, H, W = 2, 6, 1024, 10, 10

    tokens_cpu, pos_cpu, frame_dirs_cpu, frame_has_s2_cpu, ref_cpu = build_sequence_dummy(
        B=B, S=S, C=C, H=H, W=W, extra_tokens=3, device=torch.device("cpu")
    )

    # Copy everything to CUDA
    tokens_cuda = tokens_cpu.to("cuda")
    pos_cuda = pos_cpu.to("cuda")
    frame_dirs_cuda = frame_dirs_cpu.to("cuda")
    frame_has_s2_cuda = frame_has_s2_cpu.to("cuda")
    ref_cuda = ref_cpu.to("cuda")

    seq_cpu = S2RopeSequencePositionalEncoding(head_dim=C).to("cpu").eval()
    seq_cuda = S2RopeSequencePositionalEncoding(head_dim=C).to("cuda").eval()

    with torch.no_grad():
        out_cpu = seq_cpu(tokens_cpu, pos_cpu, frame_dirs_cpu, frame_has_s2_cpu, ref_cpu)
        out_cuda = seq_cuda(tokens_cuda, pos_cuda, frame_dirs_cuda, frame_has_s2_cuda, ref_cuda)
        out_cuda_cpu = out_cuda.to("cpu")

    max_diff = (out_cpu - out_cuda_cpu).abs().max().item()
    print(f"CPU vs CUDA max diff (sequence): {max_diff:.6e}")
    assert max_diff < 1e-5

    print("OK")


# -------------------------------------------------------------
# Test 8: Backward/grad sanity for sequence
# -------------------------------------------------------------
def test_backward_sequence():
    print("\n== test_backward_sequence ==")
    device = torch.device(device_str())
    torch.manual_seed(5)

    # Build dummy sequence (tokens does NOT require grad yet)
    tokens, pos, frame_dirs, frame_has_s2, ref_dirs = build_sequence_dummy(
        B=2, S=4, C=256, H=6, W=6, extra_tokens=2, device=device
    )

    # We want gradients w.r.t. some base tensor, but the tensor that actually
    # goes through S2RoPE must be NON-LEAF (because the kernel mutates in-place).
    #
    # Strategy:
    #   base: leaf with requires_grad=True (we'll inspect base.grad)
    #   x   : clone of base (non-leaf, requires_grad=True, grad_fn from clone)
    #         -> this is passed to the sequence encoder and mutated in-place.
    base = tokens.clone().detach().requires_grad_(True)
    base.retain_grad()  # so that autograd keeps grad on this non-output leaf

    x = base.clone()    # non-leaf (has grad_fn), safe for in-place ops

    seq_pos = S2RopeSequencePositionalEncoding(head_dim=256).to(device)
    seq_pos.train()

    out = seq_pos(x, pos, frame_dirs, frame_has_s2, ref_dirs)  # uses x, not base
    loss = out.pow(2).mean()
    loss.backward()

    # Check that we actually got gradients back on base
    assert base.grad is not None
    max_grad = base.grad.abs().max().item()
    print(f"max grad on base tokens: {max_grad:.6e}")
    assert not torch.isnan(base.grad).any()

    print("OK")


# -------------------------------------------------------------
# Basic timing comparison CPU vs CUDA for sequence (if CUDA)
# -------------------------------------------------------------
def bench_cpu_vs_cuda_sequence():
    print("\n== bench_cpu_vs_cuda_sequence ==")
    if not has_cuda():
        print("CUDA not available – skipping.")
        return

    torch.manual_seed(6)

    B, S, C, H, W = 2, 8, 1024, 14, 14
    tokens_cpu, pos_cpu, frame_dirs_cpu, frame_has_s2_cpu, ref_cpu = build_sequence_dummy(
        B=B, S=S, C=C, H=H, W=W, extra_tokens=4, device=torch.device("cpu")
    )

    tokens_cuda = tokens_cpu.to("cuda")
    pos_cuda = pos_cpu.to("cuda")
    frame_dirs_cuda = frame_dirs_cpu.to("cuda")
    frame_has_s2_cuda = frame_has_s2_cpu.to("cuda")
    ref_cuda = ref_cpu.to("cuda")

    seq_cpu = S2RopeSequencePositionalEncoding(head_dim=C).to("cpu").eval()
    seq_cuda = S2RopeSequencePositionalEncoding(head_dim=C).to("cuda").eval()

    # Warmup
    with torch.no_grad():
        _ = seq_cpu(tokens_cpu, pos_cpu, frame_dirs_cpu, frame_has_s2_cpu, ref_cpu)
        _ = seq_cuda(tokens_cuda, pos_cuda, frame_dirs_cuda, frame_has_s2_cuda, ref_cuda)
        torch.cuda.synchronize()

    n_runs = 20

    # CPU timing
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = seq_cpu(tokens_cpu, pos_cpu, frame_dirs_cpu, frame_has_s2_cpu, ref_cpu)
    cpu_time = (time.time() - t0) / n_runs

    # CUDA timing
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = seq_cuda(tokens_cuda, pos_cuda, frame_dirs_cuda, frame_has_s2_cuda, ref_cuda)
        torch.cuda.synchronize()
    cuda_time = (time.time() - t0) / n_runs

    print(f"avg CPU time per fwd:  {cpu_time*1000:.3f} ms")
    print(f"avg CUDA time per fwd: {cuda_time*1000:.3f} ms")


# -------------------------------------------------------------
if __name__ == "__main__":
    test_kernel_s2rope_basic()
    test_single_xy_only()
    test_single_s2_effect()
    test_seq_support_vs_query()
    test_seq_token_s2_mask()
    test_cpu_cuda_single()
    test_cpu_cuda_sequence()
    test_backward_sequence()
    bench_cpu_vs_cuda_sequence()
    print("\nAll S2RoPE tests completed.\n")
