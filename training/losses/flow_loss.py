# training/losses/flow_loss.py
#
# Packed buddy-only flow losses.
#
# Dense flow loss is computed ONLY on packed valid pairs:
#   pred_dense_flow: [M,ps,ps,2]
#   pred_dense_b:    [M,ps,ps]
#   gt_dense_flow:   [M,ps,ps,2]   gathered by buddy t_idx
#   gt_vis:          [M,ps,ps]     (0/1 visibility mask; optional soft weights)
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from vpog.models.flow_head import assert_finite


@dataclass
class DenseFlowLossConfig:
    eps_b: float = 1e-4
    eps_norm: float = 1e-6
    lambda_invis: float = 0.1   # set 0 to disable invisible regularizer
    clamp_b_min: float = 1e-4


class DenseFlowLoss(nn.Module):
    def __init__(
            self,
            eps_b =1e-4,
            eps_norm =1e-6,
            lambda_invis =0.1,
            clamp_b_min =1e-4,
        ):
        super().__init__()
        self.eps_b = eps_b
        self.eps_norm = eps_norm
        self.lambda_invis = lambda_invis
        self.clamp_b_min = clamp_b_min

    def forward(
        self,
        pred_flow: torch.Tensor,   # [M,ps,ps,2]
        pred_b: torch.Tensor,      # [M,ps,ps] (positive)
        gt_flow: torch.Tensor,     # [M,ps,ps,2]
        gt_vis: torch.Tensor,      # [M,ps,ps] (0/1 or soft)
        return_pnp_weights: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if pred_flow.shape != gt_flow.shape:
            raise ValueError(f"pred_flow {tuple(pred_flow.shape)} != gt_flow {tuple(gt_flow.shape)}")
        if pred_flow.dim() != 4 or pred_flow.shape[-1] != 2:
            raise ValueError(f"pred_flow must be [M,ps,ps,2], got {tuple(pred_flow.shape)}")
        if pred_b.shape != pred_flow.shape[:-1]:
            raise ValueError(f"pred_b must be [M,ps,ps], got {tuple(pred_b.shape)}")
        if gt_vis.shape != pred_b.shape:
            raise ValueError(f"gt_vis must be [M,ps,ps], got {tuple(gt_vis.shape)}")
        

        ############### for DEBUGGING
        assert_finite("loss pred_flow", pred_flow)
        assert_finite("loss pred_b", pred_b)
        assert_finite("loss gt_flow", gt_flow)
        assert_finite("loss gt_vis", gt_vis)
        # vis_sum = gt_vis.float().sum(dim=(-2, -1))  # [M]
        # if (vis_sum == 0).any():
        #     m = int((vis_sum == 0).nonzero(as_tuple=False)[0].item())
        #     print(f"[NaN DEBUG] gt_vis is all-zero at m={m}. This can cause 0/0 if loss divides by vis_sum.")
        ################

        M = pred_flow.shape[0]
        if M == 0:
            z = pred_flow.new_tensor(0.0)
            out = {"dense_flow_loss": z, "dense_invis_reg": z}
            if return_pnp_weights:
                out["pnp_weights"] = pred_flow.new_empty((0, pred_b.shape[1], pred_b.shape[2]))
            return out

        # clamp b
        b = pred_b.clamp_min(self.clamp_b_min)

        # per-pixel L1 error over (dx,dy)
        err = (pred_flow - gt_flow).abs().sum(dim=-1)  # [M,ps,ps]

        # Laplace NLL: err/b + 2*log(b)
        nll = err / (b + self.eps_b) + 2.0 * torch.log(b + self.eps_b)

        w = gt_vis.clamp_min(0.0)  # [M,ps,ps]
        weighted = w * nll
        denom = w.sum().clamp_min(self.eps_norm)
        dense_flow_loss = weighted.sum() / denom

        dense_invis_reg = pred_flow.new_tensor(0.0)
        if self.lambda_invis > 0:
            inv_w = (1.0 - w).clamp_min(0.0)
            denom2 = inv_w.sum().clamp_min(self.eps_norm)
            reg = (inv_w * (1.0 / (b + self.eps_b))).sum() / denom2
            dense_invis_reg = self.lambda_invis * reg
        out = {
            "dense_flow_loss": dense_flow_loss,
            "dense_invis_reg": dense_invis_reg,
        }

        if return_pnp_weights:
            # unbounded weights for PnP: visibility * 1/b
            out["pnp_weights"] = w * (1.0 / (b + self.eps_b))  # [M,ps,ps]

        return out


@dataclass
class CenterFlowLossConfig:
    eps_norm: float = 1e-6
    loss_type: str = "l1"    # "l1" or "huber"
    huber_delta: float = 1.0


class CenterFlowLoss(nn.Module):
    """
    Optional future: loss over packed center-flow pairs.
      pred_center: [M,2]
      gt_center:   [M,2]
      valid_w:     [M] (0/1), e.g. patch_visibility
    """

    def __init__(
            self,
            eps_norm =1e-6,
            loss_type ="l1",
            huber_delta =1.0,):
        super().__init__()
        self.eps_norm = eps_norm
        self.loss_type = loss_type
        self.huber_delta = huber_delta

    def forward(
        self,
        pred_center: torch.Tensor,  # [M,2]
        gt_center: torch.Tensor,    # [M,2]
        valid_w: torch.Tensor,      # [M] float/bool
    ) -> Dict[str, torch.Tensor]:
        if pred_center.shape != gt_center.shape:
            raise ValueError(f"pred_center {tuple(pred_center.shape)} != gt_center {tuple(gt_center.shape)}")
        if pred_center.dim() != 2 or pred_center.shape[-1] != 2:
            raise ValueError(f"pred_center must be [M,2], got {tuple(pred_center.shape)}")
        if valid_w.dim() != 1 or valid_w.shape[0] != pred_center.shape[0]:
            raise ValueError(f"valid_w must be [M], got {tuple(valid_w.shape)}")

        M = pred_center.shape[0]
        if M == 0:
            return {"center_flow_loss": pred_center.new_tensor(0.0)}

        w = valid_w.float().clamp_min(0.0)
        e = pred_center - gt_center

        if self.loss_type.lower() == "huber":
            abs_e = e.abs()
            d = float(self.huber_delta)
            huber = torch.where(abs_e < d, 0.5 * (abs_e ** 2) / d, abs_e - 0.5 * d)
            per = huber.sum(dim=-1)  # [M]
        else:
            per = e.abs().sum(dim=-1)  # [M]

        denom = w.sum().clamp_min(self.eps_norm)
        loss = (w * per).sum() / denom
        return {"center_flow_loss": loss}
