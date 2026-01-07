#!/usr/bin/env python3
"""
Debug script to reproduce NaN issue from saved batch.

Usage:
    python training/debug_nan.py /path/to/nan_batch_XXX.pt
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def _fmt_tensor(x: torch.Tensor, max_elems: int = 12) -> str:
    """Compact tensor formatting for debug prints."""
    if not isinstance(x, torch.Tensor):
        return str(x)
    x_cpu = x.detach().cpu()
    if x_cpu.numel() == 0:
        return f"Tensor(shape={tuple(x_cpu.shape)}, empty)"
    if x_cpu.numel() <= max_elems:
        return f"{x_cpu}"
    flat = x_cpu.reshape(-1)
    head = flat[:max_elems].tolist()
    return f"Tensor(shape={tuple(x_cpu.shape)}, dtype={x_cpu.dtype}, head={head}...)"

def print_debug_dict(title: str, d: dict | None, max_slice_elems: int = 12):
    print("\n" + "="*80)
    print(title)
    print("="*80)
    if d is None:
        print("  (None)")
        return
    if not isinstance(d, dict):
        print(f"  (Not a dict) type={type(d)} value={d}")
        return

    # Stable ordering: scalars first, then others
    keys = sorted(d.keys())
    for k in keys:
        v = d[k]
        # Scalars (0-dim tensors) and python scalars
        if isinstance(v, torch.Tensor) and v.ndim == 0:
            # print as python scalar
            try:
                vv = v.item()
            except Exception:
                vv = _fmt_tensor(v, max_elems=max_slice_elems)
            print(f"  {k}: {vv}")
        elif isinstance(v, (int, float, bool, str)):
            print(f"  {k}: {v}")
        elif isinstance(v, torch.Tensor):
            print(f"  {k}: {_fmt_tensor(v, max_elems=max_slice_elems)}")
        else:
            print(f"  {k}: (type={type(v)}) {v}")


def inspect_saved_batch(checkpoint_path: str):
    """Load and inspect a saved NaN batch."""
    print("="*80)
    print(f"Loading debug checkpoint: {checkpoint_path}")
    print("="*80)
    
    data = torch.load(checkpoint_path, map_location='cpu')

    # Print classification head debug dicts if available
    cls_dbg_amp = data.get("cls_head_debug_amp", None)
    cls_dbg_fp32 = data.get("cls_head_debug_fp32", None)

    print_debug_dict("CLASSIFICATION HEAD DEBUG (AMP)", cls_dbg_amp, max_slice_elems=12)
    print_debug_dict("CLASSIFICATION HEAD DEBUG (FP32)", cls_dbg_fp32, max_slice_elems=12)

    def _get(dct, key):
        if isinstance(dct, dict) and key in dct:
            v = dct[key]
            return v.item() if isinstance(v, torch.Tensor) and v.ndim == 0 else v
        return None

    if isinstance(cls_dbg_amp, dict) and isinstance(cls_dbg_fp32, dict):
        print("\n" + "="*80)
        print("QUICK INTERPRETATION")
        print("="*80)

        amp_q_inf = _get(cls_dbg_amp, "q_proj_has_inf")
        amp_t_inf = _get(cls_dbg_amp, "t_proj_has_inf")
        amp_q_nan = _get(cls_dbg_amp, "q_proj_has_nan")
        amp_t_nan = _get(cls_dbg_amp, "t_proj_has_nan")
        amp_logits_nan = _get(cls_dbg_amp, "logits_final_has_nan")

        fp_q_inf = _get(cls_dbg_fp32, "q_proj_has_inf")
        fp_t_inf = _get(cls_dbg_fp32, "t_proj_has_inf")
        fp_q_nan = _get(cls_dbg_fp32, "q_proj_has_nan")
        fp_t_nan = _get(cls_dbg_fp32, "t_proj_has_nan")
        fp_logits_nan = _get(cls_dbg_fp32, "logits_final_has_nan")

        print(f"  AMP:  q_proj(has_nan={amp_q_nan}, has_inf={amp_q_inf})  "
              f"t_proj(has_nan={amp_t_nan}, has_inf={amp_t_inf})  logits_nan={amp_logits_nan}")
        print(f"  FP32: q_proj(has_nan={fp_q_nan}, has_inf={fp_q_inf})  "
              f"t_proj(has_nan={fp_t_nan}, has_inf={fp_t_inf})  logits_nan={fp_logits_nan}")

        if (amp_logits_nan is True) and (fp_logits_nan is False):
            print("  Likely AMP numerical overflow/instability in classification head path.")
        elif (amp_logits_nan is True) and (fp_logits_nan is True):
            print("  NaNs persist in FP32: likely upstream corruption (weights/tokens) or logic bug.")
        else:
            print("  No NaNs in these debug dicts (check saved tensors or mismatch between saved outputs and replay).")

    
    print(f"\nGlobal step: {data['global_step']}")
    print(f"Batch index: {data['batch_idx']}")
    
    print("\n" + "="*80)
    print("LOSS COMPONENTS (scalar values)")
    print("="*80)
    for name, value in data['loss_tensors'].items():
        print(f"  {name}: {value.item()}")
    
    print("\n" + "="*80)
    print("TENSOR STATISTICS")
    print("="*80)
    for name, value in data['stats'].items():
        print(f"  {name}: {value}")
    
    print("\n" + "="*80)
    print("CHECKING FOR NaN IN INTERMEDIATE TENSORS")
    print("="*80)
    
    tensors_to_check = {
        'pred_flow': data['pred_flow'],
        'pred_b': data['pred_b'],
        'gt_flow': data['gt_flow'],
        'gt_vis': data['gt_vis'],
        'q_tokens_aa': data['outputs']['q_tokens_aa'],
        't_img_aa': data['outputs']['t_img_aa'],
        'classification_logits': data['outputs']['classification_logits'],
    }
    
    for name, tensor in tensors_to_check.items():
        if tensor is None:
            print(f"  {name}: None")
            continue
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        print(f"  {name}: shape={tuple(tensor.shape)}, has_nan={has_nan}, has_inf={has_inf}")
        if has_nan or has_inf:
            print(f"    -> NaN count: {torch.isnan(tensor).sum().item()}")
            print(f"    -> Inf count: {torch.isinf(tensor).sum().item()}")
            print(f"    -> Min: {tensor[torch.isfinite(tensor)].min().item() if torch.isfinite(tensor).any() else 'all nan/inf'}")
            print(f"    -> Max: {tensor[torch.isfinite(tensor)].max().item() if torch.isfinite(tensor).any() else 'all nan/inf'}")
    
    print("\n" + "="*80)
    print("BATCH INFO")
    print("="*80)
    batch = data['batch']
    print(f"  images: {tuple(batch.images.shape)}")
    print(f"  poses: {tuple(batch.poses.shape)}")
    if hasattr(batch, 'scene_keys'):
        print(f"  scene_keys: {batch.scene_keys}")
    
    return data


def replay_forward_pass(data, device='cuda'):
    """Replay the forward pass with anomaly detection."""
    print("\n" + "="*80)
    print("REPLAYING FORWARD PASS WITH ANOMALY DETECTION")
    print("="*80)
    
    from training.lightning_module import VPOGLightningModule
    from hydra import compose, initialize
    
    # This is simplified - you'd need to properly reconstruct the model
    print("\nTo replay with full model:")
    print("1. Load model config from training run")
    print("2. Create VPOGLightningModule")
    print("3. Load model_state_dict from checkpoint")
    print("4. Run forward pass with torch.autograd.set_detect_anomaly(True)")
    print("5. Compute loss and backward to see exact operation causing NaN")
    
    print(f"\nModel state keys: {list(data['model_state'].keys())[:5]}...")
    print(f"Optimizer state available: {data['optimizer_state'] is not None}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python training/debug_nan.py /path/to/nan_batch_XXX.pt")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    if not Path(checkpoint_path).exists():
        print(f"Error: File not found: {checkpoint_path}")
        sys.exit(1)
    
    data = inspect_saved_batch(checkpoint_path)
    
    # Uncomment to replay forward pass
    # replay_forward_pass(data)
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Check which tensor has NaN (see above)")
    print("2. If pred_b has very small values -> increase eps_b in loss config")
    print("3. If gt_flow/gt_vis has NaN -> data preprocessing issue")
    print("4. If tokens have NaN -> encoder/AA module issue")
    print("5. Run with detect_anomaly=True in config to get exact line")
