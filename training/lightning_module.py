
# vpog/training/lightning_module.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import cv2

from training.losses.flow_loss import (
    DenseFlowLoss,
    DenseFlowLossConfig,
    CenterFlowLoss,
    CenterFlowLossConfig,
)
from training.losses.classification_loss import ClassificationLoss, ClassificationLossConfig
from src.utils.logging import get_logger

from vpog.models.flow_head import pack_valid_qt_pairs
from utils.bop_evaluation import BOPEvaluator
from vpog.inference.correspondence import CorrespondenceBuilder

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
        enable_pose_eval: bool = True,
        dataset_name: str = "ycbv",  # BOP dataset name for pose evaluation
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
        
        # BOP evaluation for validation
        self.enable_pose_eval = enable_pose_eval
        self.bop_evaluator = None
        self.correspondence_builder = None
        if self.enable_pose_eval:
            try:
                self.bop_evaluator = BOPEvaluator(dataset_name=dataset_name)
                self.correspondence_builder = CorrespondenceBuilder(
                    img_size=224,
                    patch_size=16,
                    grid_size=(14, 14),
                )
                logger.info(f"BOP evaluator initialized for dataset: {dataset_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize BOP evaluator: {e}. Pose metrics disabled.")
                self.enable_pose_eval = False

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
        gt_dense_all = self._flatten_patch_grid(batch.dense_flow).float()   # [B,S,Nq,ps,ps,2]  (stored per QUERY patch, Design B)

        # visibility/weight for dense pixels:
        gt_vis_all = self._flatten_patch_grid(batch.dense_visibility).float()  # [B,S,Nq,ps,ps]
        
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
        q_idx = pack["q_idx"]   # [M]  (query patch index)
        t_idx = pack["t_idx"]   # [M]  (buddy template patch index)

        # Gather GT dense flow/vis by QUERY patch index (q_idx) - Design B storage
        # GT is stored as dense_flow[b,s,q] = flow from buddy_template(q) → query_patch(q)
        gt_flow = gt_dense_all[b_idx, s_idx, q_idx]  # [M,ps,ps,2]
        gt_vis = gt_vis_all[b_idx, s_idx, q_idx]     # [M,ps,ps]

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
        dtype = torch.bfloat16
        with torch.cuda.amp.autocast(dtype=dtype):
            outputs = self.forward(batch)

        # GT labels for loss computation only
        patch_cls_gt = self._flatten_patch_grid(batch.patch_cls).long()        # [B,S,Nq]
        gt_dense_all = self._flatten_patch_grid(batch.dense_flow).float()   # [B,S,Nq,ps,ps,2]  (stored per QUERY patch, Design B)
        if hasattr(batch, "dense_visibility") and batch.dense_visibility is not None:
            gt_vis_all = self._flatten_patch_grid(batch.dense_visibility).float()  # [B,S,Nq,ps,ps]
        else:
            gt_vis_all = self._flatten_patch_grid(batch.dense_weight).float()  # [B,S,Nq,ps,ps]

        # Classification loss (compare predictions vs GT)
        cls_stats = self.cls_loss(outputs["classification_logits"], patch_cls_gt)
        loss_cls = cls_stats["loss_cls"]

        # CRITICAL: Use PREDICTED classifications for correspondence building (not GT!)
        # This simulates real inference where GT is unavailable
        patch_cls_pred = outputs["classification_logits"].argmax(dim=-1)  # [B,S,Nq]
        pack = pack_valid_qt_pairs(outputs["q_tokens_aa"], outputs["t_img_aa"], patch_cls_pred)
        b_idx, s_idx, q_idx, t_idx = pack["b_idx"], pack["s_idx"], pack["q_idx"], pack["t_idx"]

        # Gather GT by QUERY patch index (q_idx) - Design B storage
        gt_flow = gt_dense_all[b_idx, s_idx, q_idx]  # [M,ps,ps,2]
        gt_vis = gt_vis_all[b_idx, s_idx, q_idx]     # [M,ps,ps]
        

        with torch.cuda.amp.autocast(dtype=dtype):
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

        # Lightning automatically aggregates on_epoch=True metrics (mean reduction by default)
        self.log("val/loss_total", total, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss_cls", loss_cls, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss_dense", loss_dense, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss_invis_reg", loss_invis, on_step=False, on_epoch=True, sync_dist=True)

        self.log("val/acc_overall", cls_stats["acc_overall"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/acc_seen", cls_stats["acc_seen"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/acc_unseen", cls_stats["acc_unseen"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/M_pairs", pack["M"].float(), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        # BOP Pose Evaluation (if enabled)
        if self.enable_pose_eval and self.bop_evaluator is not None:
            try:
                self._evaluate_poses(batch, outputs, pack, pred_flow, pred_b, batch_idx)
            except Exception as e:
                logger.warning(f"Pose evaluation failed for batch {batch_idx}: {e}")
    
    def _evaluate_poses(self, batch: Any, outputs: Dict[str, torch.Tensor], 
                       pack: Dict[str, torch.Tensor], pred_flow: torch.Tensor, 
                       pred_b: torch.Tensor, batch_idx: int) -> None:
        """
        Evaluate poses using BOP metrics (MSPD, MSSD).
        
        This method:
        1. Builds 2D-3D correspondences from packed model outputs (matched patches only)
        2. Runs RANSAC PnP to estimate poses
        3. Computes MSPD and MSSD errors
        4. Adds results to the BOP evaluator
        
        Args:
            batch: VPOGBatch with depth, poses, K
            outputs: Model outputs (not used directly - we use packed pairs)
            pack: Packed indices from pack_valid_qt_pairs
            pred_flow: Dense flow predictions [M, ps, ps, 2]
            pred_b: Laplace scale [M, ps, ps]
            batch_idx: Batch index for logging
        """
        import numpy as np
        import cv2
        
        # Get batch metadata
        B = batch.images.shape[0]
        
        # Process each query in the batch
        for b in range(B):
            try:
                # Extract metadata from batch.infos (pandas DataFrame)
                obj_id = int(batch.infos.label.iloc[b])
                scene_id = int(batch.infos.scene_id.iloc[b])
                im_id = int(batch.infos.view_id.iloc[b])
                
                # Get ground truth pose and camera intrinsics
                query_pose_gt = batch.poses[b, 0].cpu().numpy()  # [4, 4] - query is first image
                R_gt = query_pose_gt[:3, :3]
                t_gt = query_pose_gt[:3, 3] * 1000.0  # Convert to mm for BOP
                
                # Build 2D-3D correspondences from packed pairs (matched patches only)
                corr_2d, corr_3d, weights = self._build_correspondences_from_packed(
                    batch, pack, pred_flow, pred_b, b
                )
                
                if len(corr_2d) < 4:
                    logger.debug(f"Batch {batch_idx}, sample {b}: Not enough correspondences ({len(corr_2d)})")
                    continue
                
                # Run RANSAC PnP to estimate pose
                R_est, t_est, num_inliers = self._solve_pnp_ransac(
                    corr_2d, corr_3d, K, weights
                )
                
                if num_inliers < 4:
                    logger.debug(f"Batch {batch_idx}, sample {b}: Not enough inliers ({num_inliers})")
                    continue
                
                # Convert translation to mm for BOP
                t_est = t_est * 1000.0
                
                # Add evaluation to BOP evaluator
                inlier_ratio = num_inliers / len(corr_2d) if len(corr_2d) > 0 else 0.0
                self.bop_evaluator.add_evaluation(
                    R_est=R_est,
                    t_est=t_est,
                    R_gt=R_gt,
                    t_gt=t_gt,
                    K=batch.K[b].cpu().numpy(),
                    obj_id=obj_id,
                    scene_id=scene_id,
                    im_id=im_id,
                    score=inlier_ratio,
                )
                
            except Exception as e:
                logger.debug(f"Failed to evaluate pose for batch {batch_idx}, sample {b}: {e}")
                continue
    
    def _build_correspondences_from_packed(
        self, 
        batch: Any, 
        pack: Dict[str, torch.Tensor],
        pred_flow: torch.Tensor,  # [M, ps, ps, 2]
        pred_b: torch.Tensor,     # [M, ps, ps]
        b_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build 2D-3D correspondences from PACKED pairs (matched patches only).
        
        This is the correct approach: we only build correspondences for patches that 
        have valid buddy matches (predicted by classification head).
        
        Args:
            batch: VPOGBatch with template depth and poses
            pack: Packed indices from pack_valid_qt_pairs (b_idx, s_idx, q_idx, t_idx, M)
            pred_flow: Dense flow predictions [M, ps, ps, 2]
            pred_b: Laplace scale (uncertainty) [M, ps, ps]
            b_idx: Batch index for this sample
            K: Camera intrinsics [3, 3]
            
        Returns:
            corr_2d: [N, 2] 2D points in query image (u, v)
            corr_3d: [N, 3] 3D points in model frame (x, y, z)
            weights: [N] correspondence confidence weights
        """
        try:
            device = pred_flow.device
            ps = pred_flow.shape[1]
            img_size = 224
            patch_size = 16
            grid_size = img_size // patch_size  # 14
            
            # Filter packed pairs for this batch sample
            mask = pack["b_idx"] == b_idx
            if mask.sum() == 0:
                logger.debug(f"No matched pairs for sample {b_idx}")
                return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3), np.array([])
            
            # Extract this sample's packed data
            s_idx = pack["s_idx"][mask]      # [M_sample] which template
            q_idx = pack["q_idx"][mask]      # [M_sample] which query patch
            t_idx = pack["t_idx"][mask]      # [M_sample] which template patch (buddy)
            flow = pred_flow[mask]           # [M_sample, ps, ps, 2]
            b_scale = pred_b[mask]           # [M_sample, ps, ps]
            
            M_sample = len(q_idx)
            
            # Check if batch has required data
            if not hasattr(batch, 'template_depth') or batch.template_depth is None:
                logger.debug(f"No template depth available for sample {b_idx}")
                return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3), np.array([])
            
            # Convert patch indices to grid coordinates
            def patch_idx_to_coords(idx):
                """Convert flat patch index to (y, x) grid coordinates."""
                y = idx // grid_size
                x = idx % grid_size
                return y, x
            
            qy, qx = patch_idx_to_coords(q_idx)  # [M_sample]
            ty, tx = patch_idx_to_coords(t_idx)  # [M_sample]
            
            # Build dense pixel grids within each patch
            delta_coords = torch.arange(ps, device=device)  # [ps]
            delta_v, delta_u = torch.meshgrid(delta_coords, delta_coords, indexing='ij')  # [ps, ps]
            delta_uv = torch.stack([delta_u, delta_v], dim=-1).float()  # [ps, ps, 2]
            
            # Query 2D pixels: [M_sample, ps, ps, 2]
            query_base_x = qx[:, None, None].float() * patch_size  # [M_sample, 1, 1]
            query_base_y = qy[:, None, None].float() * patch_size  # [M_sample, 1, 1]
            query_base = torch.stack([query_base_x, query_base_y], dim=-1)  # [M_sample, 1, 1, 2]
            query_pixels = query_base + delta_uv[None, :, :, :]  # [M_sample, ps, ps, 2]
            
            # Template baseline pixels (before flow): [M_sample, ps, ps, 2]
            template_base_x = tx[:, None, None].float() * patch_size
            template_base_y = ty[:, None, None].float() * patch_size
            template_base = torch.stack([template_base_x, template_base_y], dim=-1)
            template_baseline = template_base + delta_uv[None, :, :, :]
            
            # Apply dense flow: template_pixels = baseline + flow * patch_size
            template_pixels = template_baseline + flow * patch_size  # [M_sample, ps, ps, 2]
            
            # Check bounds
            valid_bounds = (
                (template_pixels[..., 0] >= 0) & (template_pixels[..., 0] < img_size) &
                (template_pixels[..., 1] >= 0) & (template_pixels[..., 1] < img_size)
            )  # [M_sample, ps, ps]
            
            # Sample depth from template images
            # For each correspondence, we need to know which template image to sample from
            # Build list of correspondences with their template indices
            corr_2d_list = []
            corr_3d_list = []
            weights_list = []
            
            for i in range(M_sample):
                s = s_idx[i].item()  # template index
                
                # Get template data for this specific template
                template_depth_map = batch.template_depth[b_idx, s]  # [H, W]
                template_pose = batch.poses[b_idx, s + 1]  # [4, 4] (+1 because first pose is query)
                K_template = batch.K[b_idx, s + 1]  # [3, 3]
                
                # Sample depth at template pixels
                t_pixels = template_pixels[i]  # [ps, ps, 2]
                t_u = t_pixels[..., 0]  # [ps, ps]
                t_v = t_pixels[..., 1]  # [ps, ps]
                
                # Bilinear sampling (simple version - could use grid_sample)
                t_u_floor = t_u.long().clamp(0, img_size - 2)
                t_v_floor = t_v.long().clamp(0, img_size - 2)
                depth_vals = template_depth_map[t_v_floor, t_u_floor]  # [ps, ps]
                
                # Check valid depth
                valid_depth = (depth_vals > 0.01) & (depth_vals < 10.0)  # meters
                valid_mask = valid_bounds[i] & valid_depth  # [ps, ps]
                
                if valid_mask.sum() == 0:
                    continue
                
                # Backproject template pixels to 3D in template camera frame
                fx, fy = K_template[0, 0].item(), K_template[1, 1].item()
                cx, cy = K_template[0, 2].item(), K_template[1, 2].item()
                
                X = (t_u - cx) * depth_vals / fx  # [ps, ps]
                Y = (t_v - cy) * depth_vals / fy  # [ps, ps]
                Z = depth_vals                     # [ps, ps]
                pts_3d_cam = torch.stack([X, Y, Z], dim=-1)  # [ps, ps, 3]
                
                # Transform to model frame: T_m2c^-1 @ [X, Y, Z, 1]
                T_inv = torch.inverse(template_pose)  # [4, 4]
                R_inv = T_inv[:3, :3]  # [3, 3]
                t_inv = T_inv[:3, 3]   # [3]
                
                pts_3d_model = torch.matmul(pts_3d_cam, R_inv.T) + t_inv  # [ps, ps, 3]
                
                # Compute weights: 1 / (b + eps)
                conf_weights = 1.0 / (b_scale[i] + 1e-4)  # [ps, ps]
                
                # Flatten and filter valid
                q_pix_flat = query_pixels[i].reshape(-1, 2)[valid_mask.flatten()]  # [N_valid, 2]
                pts_3d_flat = pts_3d_model.reshape(-1, 3)[valid_mask.flatten()]    # [N_valid, 3]
                weights_flat = conf_weights.flatten()[valid_mask.flatten()]        # [N_valid]
                
                corr_2d_list.append(q_pix_flat)
                corr_3d_list.append(pts_3d_flat)
                weights_list.append(weights_flat)
            
            if len(corr_2d_list) == 0:
                logger.debug(f"No valid correspondences after depth filtering for sample {b_idx}")
                return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3), np.array([])
            
            # Concatenate all correspondences
            corr_2d = torch.cat(corr_2d_list, dim=0).cpu().numpy()  # [N, 2]
            corr_3d = torch.cat(corr_3d_list, dim=0).cpu().numpy()  # [N, 3]
            weights = torch.cat(weights_list, dim=0).cpu().numpy()  # [N]
            
            logger.debug(f"Built {len(corr_2d)} correspondences from {M_sample} packed pairs for sample {b_idx}")
            return corr_2d, corr_3d, weights
            
        except Exception as e:
            logger.debug(f"Correspondence building failed for sample {b_idx}: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3), np.array([])
    
    def _solve_pnp_ransac(
        self,
        pts_2d: np.ndarray,
        pts_3d: np.ndarray,
        K: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Solve PnP with RANSAC.
        
        Returns:
            R: [3, 3] rotation matrix
            t: [3] translation vector (meters)
            num_inliers: Number of inliers
        """
        if len(pts_2d) < 4:
            return np.eye(3), np.zeros(3), 0
        
        # RANSAC parameters
        ransac_threshold = 8.0  # pixels
        ransac_iterations = 1000
        ransac_confidence = 0.99
        
        # Reshape for OpenCV
        pts_2d_cv = pts_2d.astype(np.float32).reshape(-1, 1, 2)
        pts_3d_cv = pts_3d.astype(np.float32).reshape(-1, 1, 3)
        camera_matrix = K.astype(np.float32)
        dist_coeffs = np.zeros(4, dtype=np.float32)
        
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=pts_3d_cv,
                imagePoints=pts_2d_cv,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
                reprojectionError=ransac_threshold,
                iterationsCount=ransac_iterations,
                confidence=ransac_confidence,
                flags=cv2.SOLVEPNP_EPNP,
            )
            
            if not success or inliers is None:
                return np.eye(3), np.zeros(3), 0
            
            # Convert rotation vector to matrix
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.flatten()
            num_inliers = len(inliers)
            
            return R, t, num_inliers
            
        except cv2.error:
            return np.eye(3), np.zeros(3), 0
    
    def on_validation_epoch_start(self) -> None:
        """Reset BOP evaluator at start of validation epoch."""
        if self.enable_pose_eval and self.bop_evaluator is not None:
            self.bop_evaluator.reset()
            logger.info("BOP evaluator reset for new validation epoch")
    
    def on_validation_epoch_end(self) -> None:
        """Compute and log BOP metrics at end of validation epoch."""
        if self.enable_pose_eval and self.bop_evaluator is not None:
            try:
                # Get metrics summary
                metrics = self.bop_evaluator.get_metrics()
                
                # Log main metrics
                if metrics['num_samples'] > 0:
                    self.log("val/ar_mspd", metrics['ar_mspd'], prog_bar=True, on_epoch=True)
                    self.log("val/ar_mssd", metrics['ar_mssd'], prog_bar=True, on_epoch=True)
                    self.log("val/ar_combined", metrics['ar_combined'], prog_bar=True, on_epoch=True)
                    self.log("val/mean_mspd_error", metrics['mean_mspd_error'], on_epoch=True)
                    self.log("val/mean_mssd_error", metrics['mean_mssd_error'], on_epoch=True)
                    self.log("val/num_pose_samples", float(metrics['num_samples']), on_epoch=True)
                    
                    logger.info(
                        f"Validation BOP Metrics: "
                        f"AR_MSPD={metrics['ar_mspd']:.3f}, "
                        f"AR_MSSD={metrics['ar_mssd']:.3f}, "
                        f"AR_combined={metrics['ar_combined']:.3f}, "
                        f"N={metrics['num_samples']}"
                    )
                else:
                    logger.warning("No poses successfully estimated during validation epoch")
                    
            except Exception as e:
                logger.warning(f"Failed to compute BOP metrics: {e}")

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
