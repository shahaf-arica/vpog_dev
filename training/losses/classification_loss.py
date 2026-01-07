
# vpog/losses/classification_loss.py
#
# Patched for VPOG patch classification semantics:
#   patch_cls: [B,S,Nq]
#     -1           -> ignore/background
#      0..Nt-1     -> buddy template patch index
#      Nt          -> unseen-in-template
#
# logits: [B,S,Nq,Nt+1]
#
# Provides:
#   forward(...) -> dict with loss + accuracies (overall/seen/unseen)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ClassificationLossConfig:
    tau: float = 1.0
    weight_unseen: float = 1.0
    label_smoothing: float = 0.0
    ignore_index: int = -1


class ClassificationLoss(nn.Module):
    def __init__(self, cfg: ClassificationLossConfig = ClassificationLossConfig()):
        super().__init__()
        self.cfg = cfg

    def forward(self, logits: torch.Tensor, patch_cls: torch.Tensor) -> Dict[str, torch.Tensor]:
        if logits.dim() != 4:
            raise ValueError(f"logits must be [B,S,Nq,C], got {tuple(logits.shape)}")
        if patch_cls.shape != logits.shape[:3]:
            raise ValueError(f"patch_cls must be [B,S,Nq], got {tuple(patch_cls.shape)} vs logits {tuple(logits.shape)}")

        B, S, Nq, C = logits.shape
        Nt = C - 1

        x = logits / float(self.cfg.tau)

        x_flat = x.reshape(B * S * Nq, C)
        y_flat = patch_cls.reshape(B * S * Nq)

        loss_flat = F.cross_entropy(
            x_flat,
            y_flat,
            ignore_index=int(self.cfg.ignore_index),
            reduction="none",
            label_smoothing=float(self.cfg.label_smoothing),
        )
        loss = loss_flat.view(B, S, Nq)

        valid = patch_cls != int(self.cfg.ignore_index)
        unseen = valid & (patch_cls == Nt)
        seen = valid & (patch_cls >= 0) & (patch_cls < Nt)

        if float(self.cfg.weight_unseen) != 1.0:
            w = torch.ones_like(loss)
            w[unseen] = float(self.cfg.weight_unseen)
            loss = loss * w

        denom = valid.sum().clamp_min(1).to(loss.dtype)
        loss_mean = (loss * valid.to(loss.dtype)).sum() / denom

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc_overall = ((pred == patch_cls) & valid).sum().float() / valid.sum().clamp_min(1).float()
            acc_seen = (((pred == patch_cls) & seen).sum().float() / seen.sum().clamp_min(1).float()) if seen.any() else pred.new_tensor(0.0)
            acc_unseen = (((pred == patch_cls) & unseen).sum().float() / unseen.sum().clamp_min(1).float()) if unseen.any() else pred.new_tensor(0.0)

        return {
            "loss_cls": loss_mean,
            "acc_overall": acc_overall,
            "acc_seen": acc_seen,
            "acc_unseen": acc_unseen,
        }
