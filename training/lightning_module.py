# vpog/training/lightning_module.py
#
# PyTorch Lightning training module for VPOG
#
# Features:
#   - DDP multi-GPU support
#   - Integrated loss functions (classification, flow, regularization, EPro-PnP)
#   - Optimizer and scheduler configuration
#   - Training and validation steps
#   - Metric logging
#   - Hydra configuration

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '../..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vpog.models.vpog_model import VPOGModel
from training.losses import ClassificationLoss, FlowLoss, WeightRegularization, EProPnPLoss
from training.dataloader.flow_computer import FlowComputer


class VPOGLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for VPOG training.
    
    Integrates model, losses, optimizer, and training logic.
    """
    
    def __init__(self, cfg: Any):
        """
        Args:
            cfg: Hydra configuration object
        """
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        # Build model
        self.model = VPOGModel(
            croco_checkpoint=cfg.model.croco_checkpoint,
            patch_size=cfg.model.patch_size,
            enc_embed_dim=cfg.model.enc_embed_dim,
            enc_depth=cfg.model.enc_depth,
            enc_num_heads=cfg.model.enc_num_heads,
            aa_depth=cfg.model.aa_depth,
            aa_num_heads=cfg.model.aa_num_heads,
            aa_window_size=cfg.model.aa_window_size,
            num_query_added_tokens=cfg.model.num_query_added_tokens,
            num_template_added_tokens=cfg.model.num_template_added_tokens,
            cls_hidden_dim=cfg.model.get('cls_hidden_dim', None),
            cls_tau=cfg.model.cls_tau,
            flow_hidden_dim=cfg.model.flow_hidden_dim,
            flow_num_layers=cfg.model.flow_num_layers,
        )
        
        # Build losses
        self.classification_loss = ClassificationLoss(
            tau=cfg.loss.classification.tau,
            weight_unseen=cfg.loss.classification.weight_unseen,
            label_smoothing=cfg.loss.classification.label_smoothing,
        )
        
        self.flow_loss = FlowLoss(
            loss_type=cfg.loss.flow.loss_type,
            huber_delta=cfg.loss.flow.huber_delta,
            use_confidence_weighting=cfg.loss.flow.use_confidence_weighting,
            min_valid_pixels=cfg.loss.flow.min_valid_pixels,
        )
        
        self.weight_regularization = WeightRegularization(
            weight_decay=cfg.loss.regularization.weight_decay,
            exclude_bias=cfg.loss.regularization.exclude_bias,
            exclude_norm=cfg.loss.regularization.exclude_norm,
        )
        
        self.epro_pnp_loss = EProPnPLoss(
            conf_threshold=cfg.loss.epro_pnp.conf_threshold,
            min_correspondences=cfg.loss.epro_pnp.min_correspondences,
            rotation_weight=cfg.loss.epro_pnp.rotation_weight,
            translation_weight=cfg.loss.epro_pnp.translation_weight,
            use_epropnp=cfg.loss.epro_pnp.use_epropnp,
        )
        
        # Loss weights
        self.loss_weights = {
            'classification': cfg.loss.weights.classification,
            'flow': cfg.loss.weights.flow,
            'regularization': cfg.loss.weights.regularization,
            'epro_pnp': cfg.loss.weights.epro_pnp,
        }
        
        # FlowComputer for GT labels
        self.flow_computer = FlowComputer(
            patch_size=cfg.model.patch_size,
            depth_tolerance=cfg.data.depth_tolerance,
            min_visible_pixels=cfg.data.min_visible_pixels,
        )
        
        # Validation metric tracking
        self.validation_step_outputs = []
    
    def forward(self, query_data: dict, template_data: dict):
        """Forward pass through model."""
        return self.model(query_data, template_data)
    
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: dict with 'query', 'template', 'gt_pose', etc.
            batch_idx: batch index
        
        Returns:
            Total loss
        """
        query_data = batch['query']
        template_data = batch['template']
        
        # Forward pass
        outputs = self(query_data, template_data)
        
        # Extract outputs
        classification_logits = outputs['classification_logits']  # [B, S, Nq, Nt+added]
        flow = outputs['flow']  # [B, S, Nq, Nt, 16, 16, 2]
        flow_confidence = outputs['flow_confidence']  # [B, S, Nq, Nt, 16, 16]
        
        # Compute ground truth labels
        gt_labels = self._compute_gt_labels(query_data, template_data)
        
        # Compute losses
        losses = {}
        
        # Classification loss
        if self.loss_weights['classification'] > 0:
            cls_loss = self.classification_loss(
                classification_logits,
                gt_labels['classification_labels'],
                gt_labels['query_unseen_mask'],
                unseen_token_idx=template_data['features'].shape[2],  # Nt
            )
            losses['classification'] = cls_loss
            self.log('train/loss_classification', cls_loss, prog_bar=True)
        
        # Flow loss
        if self.loss_weights['flow'] > 0:
            flow_loss = self.flow_loss(
                flow,
                gt_labels['flow'],
                flow_confidence,
                gt_labels['flow_unseen_mask'],
            )
            losses['flow'] = flow_loss
            self.log('train/loss_flow', flow_loss, prog_bar=True)
        
        # Weight regularization
        if self.loss_weights['regularization'] > 0:
            reg_loss = self.weight_regularization(self.model)
            losses['regularization'] = reg_loss
            self.log('train/loss_regularization', reg_loss)
        
        # EPro-PnP loss (optional, may be disabled)
        if self.loss_weights['epro_pnp'] > 0:
            epro_loss = self.epro_pnp_loss(
                classification_logits,
                flow,
                flow_confidence,
                query_data,
                template_data,
                batch['camera_intrinsics'],
                batch['gt_pose'],
                gt_labels['flow_unseen_mask'],
            )
            losses['epro_pnp'] = epro_loss
            self.log('train/loss_epro_pnp', epro_loss)
        
        # Total loss
        total_loss = sum(
            self.loss_weights[k] * v
            for k, v in losses.items()
            if k in self.loss_weights
        )
        
        self.log('train/loss_total', total_loss, prog_bar=True)
        
        # Log learning rate
        opt = self.optimizers()
        self.log('train/lr', opt.param_groups[0]['lr'])
        
        return total_loss
    
    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Validation step.
        
        Args:
            batch: dict with 'query', 'template', etc.
            batch_idx: batch index
        
        Returns:
            Metrics dict
        """
        query_data = batch['query']
        template_data = batch['template']
        
        # Forward pass
        outputs = self(query_data, template_data)
        
        # Extract outputs
        classification_logits = outputs['classification_logits']
        flow = outputs['flow']
        flow_confidence = outputs['flow_confidence']
        
        # Compute ground truth labels
        gt_labels = self._compute_gt_labels(query_data, template_data)
        
        # Compute losses
        cls_loss = self.classification_loss(
            classification_logits,
            gt_labels['classification_labels'],
            gt_labels['query_unseen_mask'],
            unseen_token_idx=template_data['features'].shape[2],
        )
        
        flow_loss = self.flow_loss(
            flow,
            gt_labels['flow'],
            flow_confidence,
            gt_labels['flow_unseen_mask'],
        )
        
        # Compute metrics
        cls_metrics = self.classification_loss.compute_accuracy(
            classification_logits,
            gt_labels['classification_labels'],
            gt_labels['query_unseen_mask'],
            unseen_token_idx=template_data['features'].shape[2],
        )
        
        flow_metrics = self.flow_loss.compute_metrics(
            flow,
            gt_labels['flow'],
            gt_labels['flow_unseen_mask'],
        )
        
        metrics = {
            'val_loss_classification': cls_loss,
            'val_loss_flow': flow_loss,
            'val_acc_seen': cls_metrics['acc_seen'],
            'val_acc_unseen': cls_metrics['acc_unseen'],
            'val_acc_overall': cls_metrics['acc_overall'],
            'val_flow_mae': flow_metrics['mae'],
            'val_flow_rmse': flow_metrics['rmse'],
            'val_valid_ratio': flow_metrics['valid_ratio'],
        }
        
        self.validation_step_outputs.append(metrics)
        return metrics
    
    def on_validation_epoch_end(self):
        """Aggregate validation metrics."""
        if not self.validation_step_outputs:
            return
        
        # Average metrics
        avg_metrics = {}
        for key in self.validation_step_outputs[0].keys():
            values = [x[key] for x in self.validation_step_outputs]
            if isinstance(values[0], torch.Tensor):
                avg_metrics[key] = torch.stack(values).mean()
            else:
                avg_metrics[key] = sum(values) / len(values)
        
        # Log metrics
        for key, value in avg_metrics.items():
            self.log(key, value, prog_bar=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        cfg = self.cfg.optimizer
        
        # Build optimizer
        if cfg.name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2),
                weight_decay=cfg.weight_decay,
                eps=cfg.eps,
            )
        elif cfg.name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2),
                weight_decay=cfg.weight_decay,
                eps=cfg.eps,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.name}")
        
        # Build scheduler
        scheduler_cfg = self.cfg.scheduler
        
        if scheduler_cfg.name.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_cfg.T_max,
                eta_min=scheduler_cfg.eta_min,
            )
        elif scheduler_cfg.name.lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_cfg.step_size,
                gamma=scheduler_cfg.gamma,
            )
        elif scheduler_cfg.name.lower() == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=scheduler_cfg.milestones,
                gamma=scheduler_cfg.gamma,
            )
        elif scheduler_cfg.name.lower() == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_cfg.factor,
                patience=scheduler_cfg.patience,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss_total',
                }
            }
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_cfg.name}")
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
    
    def _compute_gt_labels(self, query_data: dict, template_data: dict) -> dict:
        """
        Compute ground truth labels for losses.
        
        Returns:
            dict with 'classification_labels', 'flow', 'flow_unseen_mask', 'query_unseen_mask'
        """
        # Extract data
        B = query_data['features'].shape[0]
        S = query_data['features'].shape[1]
        
        # Use FlowComputer to generate GT labels
        classification_labels, flow, confidence, unseen_mask = self.flow_computer.compute_classification_labels(
            query_xyz=query_data['xyz'],  # [B, S, Nq, 3]
            template_xyz=template_data['xyz'],  # [B, S, Nt, 3]
            query_depth=query_data['depth'],  # [B, S, H, W]
            template_depth=template_data['depth'],  # [B, S, H, W]
            query_K=query_data['camera_intrinsics'],  # [B, S, 3, 3]
            template_K=template_data['camera_intrinsics'],  # [B, S, 3, 3]
            template_pose=template_data['pose'],  # [B, S, 4, 4]
        )
        
        # Compute query unseen mask (aggregate from pixel-level)
        # A query patch is unseen if all its pixels are unseen
        query_unseen_mask = unseen_mask.all(dim=(-2, -1))  # [B, S, Nq, Nt] -> [B, S, Nq]
        query_unseen_mask = query_unseen_mask.all(dim=-1)  # Unseen in all templates
        
        return {
            'classification_labels': classification_labels,
            'flow': flow,
            'flow_confidence': confidence,
            'flow_unseen_mask': unseen_mask,
            'query_unseen_mask': query_unseen_mask,
        }
