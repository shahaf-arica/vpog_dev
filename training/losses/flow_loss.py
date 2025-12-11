# vpog/losses/flow_loss.py
#
# Flow loss for pixel-level correspondence prediction
#
# Input:
#   - pred_flow: [B, S, Nq, Nt, 16, 16, 2] - predicted flow in patch units
#   - gt_flow: [B, S, Nq, Nt, 16, 16, 2] - ground truth flow in patch units
#   - pred_conf: [B, S, Nq, Nt, 16, 16] - predicted confidence [0, 1]
#   - unseen_mask: [B, S, Nq, Nt, 16, 16] - True for unseen pixels
#
# Behavior:
#   - Compute flow error only on seen pixels (unseen_mask=False)
#   - Support L1 or Huber loss
#   - Optional confidence weighting
#
# Output:
#   - Scalar loss (mean over valid pixels)

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowLoss(nn.Module):
    """
    Flow loss for pixel-level correspondence prediction.
    
    Computes flow error only on visible (non-unseen) pixels.
    Supports L1 and Huber loss variants.
    """
    
    def __init__(
        self,
        loss_type: str = 'l1',
        huber_delta: float = 1.0,
        use_confidence_weighting: bool = True,
        min_valid_pixels: int = 1,
    ):
        """
        Args:
            loss_type: 'l1' or 'huber' (default 'l1')
            huber_delta: Delta for Huber loss (default 1.0 patch unit)
            use_confidence_weighting: Weight loss by predicted confidence (default True)
            min_valid_pixels: Minimum valid pixels to compute loss (default 1)
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        self.huber_delta = huber_delta
        self.use_confidence_weighting = use_confidence_weighting
        self.min_valid_pixels = min_valid_pixels
        
        if self.loss_type not in ['l1', 'huber']:
            raise ValueError(f"loss_type must be 'l1' or 'huber', got {loss_type}")
    
    def forward(
        self,
        pred_flow: torch.Tensor,
        gt_flow: torch.Tensor,
        pred_conf: torch.Tensor,
        unseen_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute flow loss.
        
        Args:
            pred_flow: [B, S, Nq, Nt, 16, 16, 2] - predicted flow
            gt_flow: [B, S, Nq, Nt, 16, 16, 2] - ground truth flow
            pred_conf: [B, S, Nq, Nt, 16, 16] - predicted confidence
            unseen_mask: [B, S, Nq, Nt, 16, 16] - True for unseen pixels
        
        Returns:
            Scalar loss
        """
        # Mask for valid pixels (not unseen)
        valid_mask = ~unseen_mask  # [B, S, Nq, Nt, 16, 16]
        
        num_valid = valid_mask.sum()
        if num_valid < self.min_valid_pixels:
            # No valid pixels, return zero loss
            return torch.tensor(0.0, device=pred_flow.device, dtype=pred_flow.dtype)
        
        # Compute per-pixel flow error
        flow_diff = pred_flow - gt_flow  # [B, S, Nq, Nt, 16, 16, 2]
        
        if self.loss_type == 'l1':
            # L1 norm per pixel
            error = flow_diff.abs().sum(dim=-1)  # [B, S, Nq, Nt, 16, 16]
        else:  # huber
            # Huber loss per pixel
            flow_diff_norm = flow_diff.norm(dim=-1)  # [B, S, Nq, Nt, 16, 16]
            is_small = flow_diff_norm <= self.huber_delta
            error = torch.where(
                is_small,
                0.5 * flow_diff_norm ** 2,
                self.huber_delta * (flow_diff_norm - 0.5 * self.huber_delta)
            )
        
        # Apply confidence weighting
        if self.use_confidence_weighting:
            # Higher confidence predictions should have higher weight
            weights = pred_conf  # [B, S, Nq, Nt, 16, 16]
            error = error * weights
        
        # Mask out unseen pixels
        error = error * valid_mask.float()
        
        # Mean over valid pixels
        loss = error.sum() / num_valid
        
        return loss
    
    def compute_metrics(
        self,
        pred_flow: torch.Tensor,
        gt_flow: torch.Tensor,
        unseen_mask: torch.Tensor,
    ) -> dict:
        """
        Compute flow prediction metrics.
        
        Returns:
            dict with 'mae', 'rmse', 'valid_ratio'
        """
        valid_mask = ~unseen_mask
        num_valid = valid_mask.sum()
        
        if num_valid == 0:
            return {
                'mae': 0.0,
                'rmse': 0.0,
                'valid_ratio': 0.0,
            }
        
        # Compute error on valid pixels
        flow_diff = pred_flow - gt_flow
        flow_error = flow_diff.norm(dim=-1)  # [B, S, Nq, Nt, 16, 16]
        
        # Mask and compute metrics
        valid_errors = flow_error[valid_mask]
        
        mae = valid_errors.mean().item()
        rmse = (valid_errors ** 2).mean().sqrt().item()
        valid_ratio = (num_valid.float() / unseen_mask.numel()).item()
        
        return {
            'mae': mae,
            'rmse': rmse,
            'valid_ratio': valid_ratio,
        }
