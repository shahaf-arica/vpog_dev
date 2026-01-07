# vpog/utils/confidence_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch


ConfMode = Literal["inv", "exp", "prob"]


@dataclass
class ConfidenceMapConfig:
    """
    How to map Laplace scale b (>0) to a [0,1] confidence.

    mode:
      - "inv":  conf = 1 / (1 + b)             (no hyperparams, robust default)
      - "exp":  conf = exp(-b / tau)           (tau controls softness)
      - "prob": conf = 1 - exp(-delta / b)     (interpretable: P(|e|<delta) under Laplace)

    Notes:
      - conf is then multiplied by visibility/weight mask if provided.
      - eps is used for numerical stability.
    """
    mode: ConfMode = "inv"
    eps: float = 1e-4
    tau: float = 1.0       # used if mode == "exp"
    delta: float = 1.0     # used if mode == "prob"
    clamp_wmax: Optional[float] = None  # optional clamp for PnP weights


def laplace_scale_to_pnp_weight(
    b: torch.Tensor,                       # [..., ps, ps] or any shape
    visibility: Optional[torch.Tensor] = None,  # same shape as b, typically dense_weight in [0,1]
    eps: float = 1e-4,
    clamp_wmax: Optional[float] = None,
) -> torch.Tensor:
    """
    Converts Laplace scale b to an unbounded weight for PnP: w = 1/(b+eps),
    optionally masked by visibility.
    """
    w = 1.0 / (b + eps)
    if visibility is not None:
        w = w * visibility
    if clamp_wmax is not None:
        w = w.clamp(max=float(clamp_wmax))
    return w


def laplace_scale_to_conf01(
    b: torch.Tensor,
    visibility: Optional[torch.Tensor] = None,
    cfg: ConfidenceMapConfig = ConfidenceMapConfig(),
) -> torch.Tensor:
    """
    Map Laplace scale b (>0) to confidence in [0,1], then apply visibility mask (if provided).
    """
    if cfg.mode == "inv":
        conf = 1.0 / (1.0 + b)
    elif cfg.mode == "exp":
        conf = torch.exp(-b / float(cfg.tau))
    elif cfg.mode == "prob":
        conf = 1.0 - torch.exp(-float(cfg.delta) / (b + cfg.eps))
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    # Ensure numeric range (should already be within [0,1] up to eps)
    conf = conf.clamp(min=0.0, max=1.0)

    if visibility is not None:
        conf = conf * visibility

    return conf


def reduce_patch_confidence(
    conf01: torch.Tensor,
    reduce: Literal["mean", "max"] = "mean",
    dims: Tuple[int, int] = (-2, -1),  # ps dims
) -> torch.Tensor:
    """
    Reduce per-pixel confidence map to a scalar per correspondence (e.g., for logging).
    """
    if reduce == "mean":
        return conf01.mean(dim=dims)
    if reduce == "max":
        return conf01.max(dim=dims[0]).values.max(dim=dims[1]).values
    raise ValueError(f"Unknown reduce: {reduce}")
