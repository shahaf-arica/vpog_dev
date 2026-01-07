
# vpog/training/lightning_module.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from training.losses.flow_loss import (
    DenseFlowLoss,
    DenseFlowLossConfig,
    CenterFlowLoss,
    CenterFlowLossConfig,
)
from training.losses.classification_loss import ClassificationLoss, ClassificationLossConfig
from src.utils.logging import get_logger

from vpog.models.flow_head import pack_valid_qt_pairs

logger = get_logger(__name__)


@dataclass
class LossWeights:
    cls: float = 1.0
    dense: float = 1.0
    invis_reg: float = 1.0
    center: float = 1.0


class VPOGLightningModule(pl.LightningModule):
    """
    Trains VPOG with:
      - patch classification (buddy vs unseen)
      - dense buddy-only flow on PACKED valid pairs (Laplace scale b)
      - (optional future) center-flow on packed pairs
    """

    def __init__(
        self,
        model: nn.Module,
        cls_loss: nn.Module,
        dense_flow_loss: nn.Module,
        center_flow_loss: nn.Module = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 0.999),
        loss_weights: LossWeights = LossWeights(),
        use_center_flow: bool = False,
        log_every_n_steps: int = 50,
        use_param_groups: bool = False,
        backbone_lr_multiplier: float = 0.1,
        new_components_lr_multiplier: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.betas = betas
        self.loss_weights = loss_weights
        self.log_every_n_steps = int(log_every_n_steps)

        self.cls_loss = cls_loss
        self.dense_flow_loss = dense_flow_loss
        self.central_flow_loss = center_flow_loss

        self.use_center_flow = use_center_flow

        assert not self.use_center_flow or (self.central_flow_loss is not None), "Center flow loss must be provided if use_center_flow is True"

    @staticmethod
    def _flatten_patch_grid(x: torch.Tensor) -> torch.Tensor:
        """
        Flattens [B,S,H_p,W_p,...] -> [B,S,N,...] where N = H_p*W_p.
        """
        if x.dim() < 4:
            raise ValueError(f"Expected tensor with at least 4 dims [B,S,H_p,W_p,...], got {tuple(x.shape)}")
        B, S, Hp, Wp = x.shape[:4]
        rest = x.shape[4:]
        return x.view(B, S, Hp * Wp, *rest)

    def forward(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Convenience forward that accepts VPOGBatch.
        Produces logits and AA tokens needed for head calls.
        """
        images = batch.images            # [B,S+1,3,H,W]
        poses = batch.poses              # [B,S+1,4,4]

        B, Sp1, _, _, _ = images.shape
        S = Sp1 - 1

        query_images = images[:, 0]      # [B,3,H,W]
        template_images = images[:, 1:]  # [B,S,3,H,W]
        template_poses = poses[:, 1:]    # [B,S,4,4]

        out = self.model(
            query_images=query_images,
            template_images=template_images,
            template_poses=template_poses,
        )

        q_tokens_aa = out["query_tokens_aa"]          # [B, Nq, C]
        t_tokens_aa = out["template_tokens_aa"]       # [B, S, Nt+1, C]
        HW = q_tokens_aa.shape[1]
        Nt = HW  # by construction

        classification_logits = self.model.classification_head(
            q_tokens=q_tokens_aa,
            t_tokens=t_tokens_aa,
        )  # [B,S,Nq,Nt+1]

        # Image tokens only for flow heads
        t_img_aa = t_tokens_aa[:, :, :Nt, :].contiguous()  # [B,S,Nt,C]

        return {
            "classification_logits": classification_logits,
            "q_tokens_aa": q_tokens_aa,
            "t_img_aa": t_img_aa,
            "t_tokens_aa": t_tokens_aa,
        }

        # return {
        #     "classification_logits": classification_logits,
        #     "q_tokens_aa": q_tokens_aa,
        #     "t_img_aa": t_img_aa,
        # }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if isinstance(batch, list):
            if len(batch) == 0:
                logger.warning(f"Training batch {batch_idx} is empty list, skipping...")
                return None
            batch = batch[0]
        if batch is None:
            logger.warning(f"Training batch {batch_idx} is None, skipping...")
            return None

        # Check for NaN/Inf in input data
        if not torch.isfinite(batch.images).all():
            logger.error(f"NaN/Inf detected in input images at batch {batch_idx}, skipping...")
            return None
        if hasattr(batch, 'dense_flow') and batch.dense_flow is not None:
            if not torch.isfinite(batch.dense_flow).all():
                logger.error(f"NaN/Inf detected in dense_flow labels at batch {batch_idx}, skipping...")
                return None

        outputs = self.forward(batch)

        # -------------------------
        # GT flatten
        # -------------------------
        patch_cls = self._flatten_patch_grid(batch.patch_cls).long()        # [B,S,Nq]
        gt_dense_all = self._flatten_patch_grid(batch.dense_flow).float()   # [B,S,Nt,ps,ps,2]  (stored per TEMPLATE patch)

        # visibility/weight for dense pixels:
        gt_vis_all = self._flatten_patch_grid(batch.dense_visibility).float()  # [B,S,Nt,ps,ps]
        
        B, S, Nq = patch_cls.shape
        _, _, Nt, ps, ps2, _ = gt_dense_all.shape
        assert ps == ps2
        if Nt != Nq:
            raise ValueError(f"Expected Nt==Nq (template/query patch grids match). Got Nt={Nt}, Nq={Nq}")

        # -------------------------
        # Classification loss (as you already do)
        # -------------------------
        cls_stats = self.cls_loss(outputs["classification_logits"], patch_cls)
        loss_cls = cls_stats["loss_cls"]

        # -------------------------
        # PACK valid pairs for dense flow
        # -------------------------
        pack = pack_valid_qt_pairs(
            q_tokens=outputs["q_tokens_aa"],   # [B,Nq,C]
            t_tokens=outputs["t_img_aa"],      # [B,S,Nt,C]
            patch_cls=patch_cls,               # [B,S,Nq]
        )

        q_tok = pack["q_tok"]   # [M,C]
        t_tok = pack["t_tok"]   # [M,C]
        b_idx = pack["b_idx"]   # [M]
        s_idx = pack["s_idx"]   # [M]
        t_idx = pack["t_idx"]   # [M]  (buddy template patch index)

        # Gather GT dense flow/vis by buddy template patch index (t_idx)
        gt_flow = gt_dense_all[b_idx, s_idx, t_idx]  # [M,ps,ps,2]
        gt_vis = gt_vis_all[b_idx, s_idx, t_idx]     # [M,ps,ps]

        # Dense head only on packed pairs
        pred_flow, pred_b, pred_w = self.model.dense_flow_head.forward_packed(q_tok, t_tok)

        dense_stats = self.dense_flow_loss(
            pred_flow=pred_flow,
            pred_b=pred_b,
            gt_flow=gt_flow,
            gt_vis=gt_vis,
            return_pnp_weights=False,
        )
        loss_dense = dense_stats["dense_flow_loss"]
        loss_invis = dense_stats["dense_invis_reg"]

        # -------------------------
        # Optional future: center-flow packed
        # -------------------------
        # loss_center = pred_flow.new_tensor(0.0)
        if self.use_center_flow:
            gt_center_all = self._flatten_patch_grid(batch.coarse_flows).float()   # [B,S,Nq,2] (stored per QUERY patch)
            patch_vis_all = self._flatten_patch_grid(batch.patch_visibility).float()  # [B,S,Nq]
            # gather packed gt center by q_idx (not t_idx)
            q_idx = pack["q_idx"]
            gt_center = gt_center_all[b_idx, s_idx, q_idx]     # [M,2]
            valid_w = patch_vis_all[b_idx, s_idx, q_idx]       # [M]
            pred_center = self.model.center_flow_head.forward_packed(q_tok, t_tok)  # [M,2]
            center_stats = self.central_flow_loss(pred_center, gt_center, valid_w)
            loss_center = center_stats["center_flow_loss"]

        # -------------------------
        # Total loss
        # -------------------------
        lw = self.loss_weights
        loss_total = (
            lw.cls * loss_cls +
            lw.dense * loss_dense +
            lw.invis_reg * loss_invis
        )
        if self.use_center_flow:
            loss_total += lw.center * loss_center
        
        # NaN/Inf detection - CRITICAL: Stop training immediately
        # Once NaN appears, it corrupts model weights/optimizer state and persists forever
        if not torch.isfinite(loss_total):
            from pathlib import Path
            
            # Save everything for debugging
            debug_dir = Path(self.trainer.log_dir if hasattr(self.trainer, 'log_dir') else ".") / "nan_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            save_path = debug_dir / f"nan_batch_{self.global_step}_{batch_idx}.pt"
            
            logger.error("="*80)
            logger.error(f"NaN DETECTED at step {self.global_step}, batch {batch_idx}")
            logger.error(f"Loss components: cls={loss_cls.item()}, dense={loss_dense.item()}, invis={loss_invis.item()}")
            logger.error(f"Saving debug data to: {save_path}")
            # Extra classification-head diagnostics (small slices)
            cls_head_debug_amp = None
            cls_head_debug_fp32 = None
            try:
                head = self.model.classification_head
                qaa = outputs["q_tokens_aa"]
                taa = outputs["t_tokens_aa"]

                # AMP path (matches training numerics)
                with torch.no_grad():
                    _logits_amp, cls_head_debug_amp = head(
                        q_tokens=qaa,
                        t_tokens=taa,
                        return_debug=True,
                    )

                # FP32 recompute (helps confirm AMP overflow/Inf -> NaN)
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                    _logits_fp32, cls_head_debug_fp32 = head(
                        q_tokens=qaa.float(),
                        t_tokens=taa.float(),
                        return_debug=True,
                    )
            except Exception as _e:
                logger.error(f"Failed to collect cls_head_debug: {_e}")
            try:
                
                torch.save({
                    'batch': batch,
                    'batch_idx': batch_idx,
                    'global_step': self.global_step,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.trainer.optimizers[0].state_dict() if self.trainer.optimizers else None,
                    
                    # Intermediate tensors (detached to avoid graph)
                    'outputs': {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()},
                    'pred_flow': pred_flow.detach(),
                    'pred_b': pred_b.detach(),
                    'pred_w': pred_w.detach() if pred_w is not None else None,
                    'gt_flow': gt_flow.detach(),
                    'gt_vis': gt_vis.detach(),
                    'patch_cls': patch_cls.detach(),
                    'pack_indices': {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in pack.items()},

                    'cls_head_debug_amp': cls_head_debug_amp,
                    'cls_head_debug_fp32': cls_head_debug_fp32,
                    
                    # Loss tensors
                    'loss_tensors': {
                        'cls': loss_cls.detach(),
                        'dense': loss_dense.detach(),
                        'invis': loss_invis.detach(),
                        'total': loss_total.detach(),
                    },
                    
                    # Statistics
                    'stats': {
                        'pred_flow_min': pred_flow.min().item(),
                        'pred_flow_max': pred_flow.max().item(),
                        'pred_b_min': pred_b.min().item(),
                        'pred_b_max': pred_b.max().item(),
                        'q_tokens_min': outputs['q_tokens_aa'].min().item(),
                        'q_tokens_max': outputs['q_tokens_aa'].max().item(),
                    }
                }, save_path)
                logger.error(f"Debug data saved successfully")
            except Exception as e:
                logger.error(f"Failed to save debug data: {e}")
            
            logger.error("="*80)
            raise RuntimeError(
                f"NaN loss detected at step {self.global_step}. "
                f"Training stopped to prevent optimizer corruption. "
                f"Debug data saved to {save_path}"
            )
        
        # Logging
        self.log("train/loss_total", loss_total, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss_cls", loss_cls, on_step=True, on_epoch=True)
        self.log("train/loss_dense", loss_dense, on_step=True, on_epoch=True)
        self.log("train/loss_invis_reg", loss_invis, on_step=True, on_epoch=True)
        if self.use_center_flow:
            self.log("train/loss_center", loss_center, on_step=True, on_epoch=True)

        self.log("train/acc_overall", cls_stats["acc_overall"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc_seen", cls_stats["acc_seen"], on_step=True, on_epoch=True)
        self.log("train/acc_unseen", cls_stats["acc_unseen"], on_step=True, on_epoch=True)

        # Useful debug: how many packed pairs
        self.log("train/M_pairs", pack["M"].float(), prog_bar=False, on_step=True, on_epoch=True)

        return loss_total

    @torch.no_grad()
    def validation_step(self, batch: Any, batch_idx: int) -> None:
        # Keep your current val losses for now (real pose metrics come later)
        if isinstance(batch, list):
            if len(batch) == 0:
                logger.warning(f"Validation batch {batch_idx} is empty list, skipping...")
                return None
            batch = batch[0]
        if batch is None:
            logger.warning(f"Validation batch {batch_idx} is None, skipping...")
            return None

        outputs = self.forward(batch)

        patch_cls = self._flatten_patch_grid(batch.patch_cls).long()        # [B,S,Nq]
        gt_dense_all = self._flatten_patch_grid(batch.dense_flow).float()   # [B,S,Nt,ps,ps,2]
        if hasattr(batch, "dense_visibility") and batch.dense_visibility is not None:
            gt_vis_all = self._flatten_patch_grid(batch.dense_visibility).float()
        else:
            gt_vis_all = self._flatten_patch_grid(batch.dense_weight).float()

        cls_stats = self.cls_loss(outputs["classification_logits"], patch_cls)
        loss_cls = cls_stats["loss_cls"]

        pack = pack_valid_qt_pairs(outputs["q_tokens_aa"], outputs["t_img_aa"], patch_cls)
        b_idx, s_idx, t_idx = pack["b_idx"], pack["s_idx"], pack["t_idx"]

        gt_flow = gt_dense_all[b_idx, s_idx, t_idx]
        gt_vis = gt_vis_all[b_idx, s_idx, t_idx]

        pred_flow, pred_b, _ = self.model.dense_flow_head.forward_packed(pack["q_tok"], pack["t_tok"])

        dense_stats = self.dense_flow_loss(
            pred_flow=pred_flow,
            pred_b=pred_b,
            gt_flow=gt_flow,
            gt_vis=gt_vis,
            return_pnp_weights=False,
        )
        loss_dense = dense_stats["dense_flow_loss"]
        loss_invis = dense_stats["dense_invis_reg"]

        lw = self.loss_weights
        total = lw.cls * loss_cls + lw.dense * loss_dense + lw.invis_reg * loss_invis

        self.log("val/loss_total", total, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/loss_cls", loss_cls, on_step=False, on_epoch=True)
        self.log("val/loss_dense", loss_dense, on_step=False, on_epoch=True)
        self.log("val/loss_invis_reg", loss_invis, on_step=False, on_epoch=True)

        self.log("val/acc_overall", cls_stats["acc_overall"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc_seen", cls_stats["acc_seen"], on_step=False, on_epoch=True)
        self.log("val/acc_unseen", cls_stats["acc_unseen"], on_step=False, on_epoch=True)
        self.log("val/M_pairs", pack["M"].float(), prog_bar=False, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        use_param_groups = self.hparams.get("use_param_groups", False)

        if use_param_groups:
            backbone_lr_mult = self.hparams.get("backbone_lr_multiplier", 0.1)
            new_components_lr_mult = self.hparams.get("new_components_lr_multiplier", 1.0)

            backbone_params = []
            new_component_params = []

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if any(component in name for component in ["croco_encoder", "enc"]):
                    backbone_params.append(param)
                else:
                    new_component_params.append(param)

            param_groups = [
                {"params": backbone_params, "lr": self.lr * backbone_lr_mult, "name": "backbone"},
                {"params": new_component_params, "lr": self.lr * new_components_lr_mult, "name": "new_components"},
            ]

            logger.info("Using parameter groups:")
            logger.info(f"  Backbone: {len(backbone_params)} params, LR={self.lr * backbone_lr_mult:.2e}")
            logger.info(f"  New components: {len(new_component_params)} params, LR={self.lr * new_components_lr_mult:.2e}")

            opt = torch.optim.AdamW(param_groups, betas=self.betas, weight_decay=self.weight_decay)
        else:
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
            logger.info(f"Using single LR={self.lr:.2e} for all parameters")

        return opt




# # vpog/training/lightning_module.py

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Any, Dict, Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pytorch_lightning as pl

# from training.losses.flow_loss import DenseBuddyFlowLoss, DenseBuddyFlowLossConfig, CenterFlowLoss, CenterFlowLossConfig
# from training.losses.classification_loss import ClassificationLoss, ClassificationLossConfig
# from src.utils.logging import get_logger

# from vpog.models.flow_head import gather_buddy_tokens

# logger = get_logger(__name__)


# @dataclass
# class LossWeights:
#     cls: float = 1.0
#     dense: float = 1.0
#     invis_reg: float = 1.0
#     center: float = 1.0


# class VPOGLightningModule(pl.LightningModule):
#     """
#     Lightning module that trains VPOG with:
#       - patch classification (buddy vs unseen)
#       - dense buddy-only patch flow (Laplace scale b)
#       - center-flow
#     """

#     def __init__(
#         self,
#         model: nn.Module,
#         lr: float = 1e-4,
#         weight_decay: float = 1e-2,
#         betas: Tuple[float, float] = (0.9, 0.999),
#         loss_weights: LossWeights = LossWeights(),
#         dense_loss_cfg: DenseBuddyFlowLossConfig = DenseBuddyFlowLossConfig(),
#         center_loss_cfg: CenterFlowLossConfig = CenterFlowLossConfig(),
#         cls_loss_cfg: ClassificationLossConfig = ClassificationLossConfig(),
#         use_center_flow: bool = False,
#         log_every_n_steps: int = 50,
#         # Parameter groups for differential learning rates
#         use_param_groups: bool = False,
#         backbone_lr_multiplier: float = 0.1,
#         new_components_lr_multiplier: float = 1.0,
#     ):
#         super().__init__()
#         self.save_hyperparameters(ignore=["model"])

#         self.model = model
#         self.lr = float(lr)
#         self.weight_decay = float(weight_decay)
#         self.betas = betas
#         self.loss_weights = loss_weights
#         self.log_every_n_steps = int(log_every_n_steps)

#         self.dense_loss = DenseBuddyFlowLoss(dense_loss_cfg)
#         self.center_loss = CenterFlowLoss(center_loss_cfg)
#         self.cls_loss = ClassificationLoss(cls_loss_cfg)

#         self.use_center_flow = use_center_flow

#     @staticmethod
#     def _flatten_patch_grid(x: torch.Tensor) -> torch.Tensor:
#         """
#         Flattens [B,S,H_p,W_p,...] -> [B,S,Nq,...] where Nq = H_p*W_p.
#         """
#         if x.dim() < 4:
#             raise ValueError(f"Expected tensor with at least 4 dims [B,S,H_p,W_p,...], got {tuple(x.shape)}")
#         B, S, Hp, Wp = x.shape[:4]
#         rest = x.shape[4:]
#         return x.view(B, S, Hp * Wp, *rest)

#     @staticmethod
#     def _classification_loss_and_acc(
#         logits: torch.Tensor,   # [B,S,Nq,Nt+1]
#         patch_cls: torch.Tensor # [B,S,Nq] with -1 ignore, 0..Nt-1 buddy, Nt unseen
#     ) -> Dict[str, torch.Tensor]:
#         B, S, Nq, num_classes = logits.shape
#         if patch_cls.shape != (B, S, Nq):
#             raise ValueError(f"patch_cls must be [B,S,Nq], got {tuple(patch_cls.shape)}")

#         # CE with ignore_index=-1
#         loss = F.cross_entropy(
#             logits.reshape(B * S * Nq, num_classes),
#             patch_cls.reshape(B * S * Nq),
#             ignore_index=-1,
#             reduction="mean",
#         )

#         # Accuracy metrics (ignore -1)
#         with torch.no_grad():
#             pred = logits.argmax(dim=-1)  # [B,S,Nq]
#             valid = patch_cls != -1
#             correct = (pred == patch_cls) & valid

#             denom = valid.sum().clamp_min(1)
#             acc_overall = correct.sum().float() / denom.float()

#             # seen vs unseen split
#             Nt = num_classes - 1
#             seen = valid & (patch_cls >= 0) & (patch_cls < Nt)
#             unseen = valid & (patch_cls == Nt)

#             acc_seen = (correct & seen).sum().float() / seen.sum().clamp_min(1).float()
#             acc_unseen = (correct & unseen).sum().float() / unseen.sum().clamp_min(1).float()

#         return {
#             "loss_cls": loss,
#             "acc_overall": acc_overall,
#             "acc_seen": acc_seen,
#             "acc_unseen": acc_unseen,
#         }

#     def forward(self, batch: Any) -> Dict[str, torch.Tensor]:
#         """
#         Convenience forward that accepts VPOGBatch.
#         """
#         images = batch.images              # [B,S+1,3,H,W]
#         poses = batch.poses                # [B,S+1,4,4]
#         # ref_dirs = batch.d_ref             # [B,3]

#         B, Sp1, _, _, _ = images.shape
#         S = Sp1 - 1

#         query_images = images[:, 0]        # [B,3,H,W]
#         template_images = images[:, 1:]    # [B,S,3,H,W]
#         # query_poses = poses[:, 0]          # [B,4,4]
#         template_poses = poses[:, 1:]      # [B,S,4,4]

#         patch_cls = self._flatten_patch_grid(batch.patch_cls).long()  # [B,S,Nq]

#         out = self.model(
#                 query_images=query_images,
#                 template_images=template_images,
#                 template_poses=template_poses,
#             )

#         q_tokens_aa = out["query_tokens_aa"]      # [B,S,Nq,C]
#         t_tokens_aa = out["template_tokens_aa"]  # [B,S,Nt+1,C]
#         Nt = out["num_tokens_per_template"]

#         classification_logits = self.model.classification_head(
#             q_tokens=q_tokens_aa,
#             t_tokens=t_tokens_aa,
#         )  # [B,S,Nq,Nt+1]

#         t_img_aa = t_tokens_aa[:, :, :Nt, :].contiguous()  # [B,S,Nt,C]


#         t_buddy, valid = gather_buddy_tokens(t_img_aa, patch_cls)
#         if self.use_center_flow:
#             # compute coarse flows for the centers of patches (q->t)
#             center_flow_out = self.model.center_flow_head(
#                 q_tokens=q_tokens_aa,
#                 t_buddy=t_buddy
#             )
#         else:
#             center_flow_out = None

#         # compute refined flows from each template patch to it's buddy (t->q)
#         # Note: since this is training stage we take patch_cls directly from ground truth
#         (
#             dense_flow_out,
#             dense_b,
#         ) = self.model.dense_flow_head(
#             q_tokens=q_tokens_aa,
#             t_tokens=t_img_aa,
#             t_buddy=t_buddy,
#         )

#         out = {
#             "classification_logits": classification_logits,
#             "center_flow": center_flow_out,
#             "dense_flow": dense_flow_out,
#             "dense_b": dense_b,
#             "center_flow": center_flow_out,
#             "flow_valid": valid,
#         }

#         return out

#         # # Preferred model signature (after your model patch):
#         # try:
#         #     return self.model(
#         #         query_images=query_images,
#         #         template_images=template_images,
#         #         # query_poses=query_poses,
#         #         template_poses=template_poses,
#         #         # ref_dirs=ref_dirs,
#         #         patch_cls=patch_cls,
#         #     )
#         # except TypeError:
#         #     # Fallback for older signatures (will likely be removed once model is updated)
#         #     return self.model(
#         #         query_images=query_images,
#         #         template_images=template_images,
#         #         # query_poses=query_poses,
#         #         template_poses=template_poses,
#         #         ref_dirs=ref_dirs,
#         #     )

#     def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
#         # Handle multiple dataloaders (multi-dataset training)
#         # PyTorch Lightning wraps batches from multiple dataloaders in a list
#         if isinstance(batch, list):
#             if len(batch) == 0:
#                 logger.warning(f"Training batch {batch_idx} is empty list, skipping...")
#                 return None
#             # Use first batch (or could iterate and accumulate losses)
#             batch = batch[0]
        
#         # Skip None batches (collate_fn failed)
#         if batch is None:
#             logger.warning(f"Training batch {batch_idx} is None, skipping...")
#             return None
        
#         outputs = self.forward(batch)
#         # reshape the dense_visibility from shape [B,S,H_p,W_p,ps,ps] to [B,S,Nq,ps,ps]
#         dense_visibility = batch.dense_visibility.reshape(
#             batch.dense_weight.shape[0],
#             batch.dense_weight.shape[1],
#             -1,
#             batch.dense_weight.shape[4],
#             batch.dense_weight.shape[5],
#         )
#         # Flatten batch supervision to match buddy-only heads
#         patch_cls = self._flatten_patch_grid(batch.patch_cls).long()                     # [B,S,Nq]
#         gt_center = self._flatten_patch_grid(batch.coarse_flows).float()                 # [B,S,Nq,2]
#         patch_vis = self._flatten_patch_grid(batch.patch_visibility).float()             # [B,S,Nq]
#         gt_dense = self._flatten_patch_grid(batch.dense_flow).float()                    # [B,S,Nq,ps,ps,2]
#         dense_w = self._flatten_patch_grid(dense_visibility).float()                   # [B,S,Nq,ps,ps]

#         # --- Classification ---
#         cls_stats = self.cls_loss(outputs["classification_logits"], patch_cls)
#         loss_cls = cls_stats["loss_cls"]

#         # --- Dense flow (Laplace with b) ---
#         dense_stats = self.dense_loss(
#             pred_flow=outputs["dense_flow"],
#             pred_b=outputs["dense_b"],
#             gt_flow=gt_dense,
#             dense_weight=dense_w,
#             patch_cls=patch_cls,
#             return_pnp_weights=False,  # PnP deferred
#         )
#         loss_dense = dense_stats["dense_flow_loss"]
#         loss_invis = dense_stats["dense_invis_reg"]
        
#         lw = self.loss_weights
#         all_losses = [lw.cls * loss_cls, lw.dense * loss_dense, lw.invis_reg * loss_invis]

#         # --- Center flow ---
#         if self.use_center_flow:
#             center_stats = self.center_loss(
#                 pred_center=outputs["center_flow"],
#                 gt_center=gt_center,
#                 patch_cls=patch_cls,
#                 patch_visibility=patch_vis,
#             )
#             loss_center = center_stats["center_flow_loss"]
#             all_losses.append(lw.center * loss_center)

#         # Total
#         loss_total = torch.stack(all_losses).sum()

#         # Logging
#         self.log("train/loss_total", loss_total, prog_bar=True, on_step=True, on_epoch=True)
#         self.log("train/loss_cls", loss_cls, prog_bar=False, on_step=True, on_epoch=True)
#         self.log("train/loss_dense", loss_dense, prog_bar=False, on_step=True, on_epoch=True)
#         self.log("train/loss_invis_reg", loss_invis, prog_bar=False, on_step=True, on_epoch=True)
#         self.log("train/loss_center", loss_center, prog_bar=False, on_step=True, on_epoch=True)

#         self.log("train/acc_overall", cls_stats["acc_overall"], prog_bar=True, on_step=True, on_epoch=True)
#         self.log("train/acc_seen", cls_stats["acc_seen"], prog_bar=False, on_step=True, on_epoch=True)
#         self.log("train/acc_unseen", cls_stats["acc_unseen"], prog_bar=False, on_step=True, on_epoch=True)

#         return loss_total

#     @torch.no_grad()
#     def validation_step(self, batch: Any, batch_idx: int) -> None:
#         # Handle multiple dataloaders (if validation has multiple datasets)
#         if isinstance(batch, list):
#             if len(batch) == 0:
#                 logger.warning(f"Validation batch {batch_idx} is empty list, skipping...")
#                 return None
#             batch = batch[0]
        
#         # Skip None batches (collate_fn failed)
#         if batch is None:
#             logger.warning(f"Validation batch {batch_idx} is None, skipping...")
#             return None
        
#         outputs = self.forward(batch)

#         patch_cls = self._flatten_patch_grid(batch.patch_cls).long()
#         gt_center = self._flatten_patch_grid(batch.coarse_flows).float()
#         patch_vis = self._flatten_patch_grid(batch.patch_visibility).float()
#         gt_dense = self._flatten_patch_grid(batch.dense_flow).float()
#         dense_w = self._flatten_patch_grid(batch.dense_weight).float()

#         cls_stats = self.cls_loss(outputs["classification_logits"], patch_cls)
#         dense_stats = self.dense_loss(
#             pred_flow=outputs["dense_flow"],
#             pred_b=outputs["dense_b"],
#             gt_flow=gt_dense,
#             dense_weight=dense_w,
#             patch_cls=patch_cls,
#             return_pnp_weights=False,
#         )
#         center_stats = self.center_loss(
#             pred_center=outputs["center_flow"],
#             gt_center=gt_center,
#             patch_cls=patch_cls,
#             patch_visibility=patch_vis,
#         )

#         lw = self.loss_weights
#         total = (
#             lw.cls * cls_stats["loss_cls"] +
#             lw.dense * dense_stats["dense_flow_loss"] +
#             lw.invis_reg * dense_stats["dense_invis_reg"] +
#             lw.center * center_stats["center_flow_loss"]
#         )

#         self.log("val/loss_total", total, prog_bar=True, on_step=False, on_epoch=True)
#         self.log("val/loss_cls", cls_stats["loss_cls"], prog_bar=False, on_step=False, on_epoch=True)
#         self.log("val/loss_dense", dense_stats["dense_flow_loss"], prog_bar=False, on_step=False, on_epoch=True)
#         self.log("val/loss_invis_reg", dense_stats["dense_invis_reg"], prog_bar=False, on_step=False, on_epoch=True)
#         self.log("val/loss_center", center_stats["center_flow_loss"], prog_bar=False, on_step=False, on_epoch=True)

#         self.log("val/acc_overall", cls_stats["acc_overall"], prog_bar=True, on_step=False, on_epoch=True)
#         self.log("val/acc_seen", cls_stats["acc_seen"], prog_bar=False, on_step=False, on_epoch=True)
#         self.log("val/acc_unseen", cls_stats["acc_unseen"], prog_bar=False, on_step=False, on_epoch=True)

#     def configure_optimizers(self):
#         """Configure optimizer with parameter groups for different learning rates.
        
#         Supports:
#         - Backbone (CroCo encoder) with lower LR
#         - New components (AA module + heads) with higher LR
#         """
#         # Check if we should use parameter groups
#         use_param_groups = self.hparams.get('use_param_groups', False)
        
#         if use_param_groups:
#             # Get LR multipliers from config
#             backbone_lr_mult = self.hparams.get('backbone_lr_multiplier', 0.1)
#             new_components_lr_mult = self.hparams.get('new_components_lr_multiplier', 1.0)
            
#             # Separate parameters by component
#             backbone_params = []
#             new_component_params = []
            
#             for name, param in self.model.named_parameters():
#                 if not param.requires_grad:
#                     continue
                    
#                 # Backbone components (CroCo encoder)
#                 if any(component in name for component in ['croco_encoder', 'enc']):
#                     backbone_params.append(param)
#                 # New components (AA module, heads)
#                 else:
#                     new_component_params.append(param)
            
#             # Create parameter groups with different learning rates
#             param_groups = [
#                 {
#                     'params': backbone_params,
#                     'lr': self.lr * backbone_lr_mult,
#                     'name': 'backbone',
#                 },
#                 {
#                     'params': new_component_params,
#                     'lr': self.lr * new_components_lr_mult,
#                     'name': 'new_components',
#                 },
#             ]
            
#             logger.info(f"Using parameter groups:")
#             logger.info(f"  Backbone: {len(backbone_params)} params, LR={self.lr * backbone_lr_mult:.2e}")
#             logger.info(f"  New components: {len(new_component_params)} params, LR={self.lr * new_components_lr_mult:.2e}")
            
#             opt = torch.optim.AdamW(
#                 param_groups,
#                 betas=self.betas,
#                 weight_decay=self.weight_decay,
#             )
#         else:
#             # Single learning rate for all parameters
#             opt = torch.optim.AdamW(
#                 self.parameters(),
#                 lr=self.lr,
#                 betas=self.betas,
#                 weight_decay=self.weight_decay,
#             )
#             logger.info(f"Using single LR={self.lr:.2e} for all parameters")
        
#         return opt
