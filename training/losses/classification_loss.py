# vpog/losses/classification_loss.py
#
# Classification loss for template patch matching
#
# Input:
#   - logits: [B, S, Nq, Nt+num_added] - classification logits including added tokens
#   - labels: [B, S, Nq] - ground truth template patch indices (0 to Nt-1)
#   - unseen_mask: [B, S, Nq] - True for unseen query patches
#   - unseen_token_idx: int - index of unseen token (typically Nt)
#
# Behavior:
#   - For seen patches (unseen_mask=False): CE loss against GT label
#   - For unseen patches (unseen_mask=True): CE loss targeting unseen_token_idx
#   - Temperature scaling: tau (default 1.0)
#
# Output:
#   - Scalar loss (mean over valid patches)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """
    Cross-entropy classification loss for template patch matching.
    
    Handles both seen and unseen query patches:
    - Seen patches: Match against GT template patch index
    - Unseen patches: Should predict unseen token
    """
    
    def __init__(
        self,
        tau: float = 1.0,
        weight_unseen: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            tau: Temperature for softmax scaling (default 1.0)
            weight_unseen: Weight for unseen patches vs seen patches (default 1.0)
            label_smoothing: Label smoothing factor (default 0.0)
        """
        super().__init__()
        self.tau = tau
        self.weight_unseen = weight_unseen
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        unseen_mask: torch.Tensor,
        unseen_token_idx: int,
    ) -> torch.Tensor:
        """
        Compute classification loss.
        
        Args:
            logits: [B, S, Nq, Nt+num_added] - classification logits
            labels: [B, S, Nq] - GT template patch indices (0 to Nt-1)
            unseen_mask: [B, S, Nq] - True for unseen query patches
            unseen_token_idx: int - index of unseen token
        
        Returns:
            Scalar loss
        """
        B, S, Nq, num_classes = logits.shape
        
        # Apply temperature scaling
        logits = logits / self.tau
        
        # Create full target labels
        # For seen patches: use GT labels
        # For unseen patches: use unseen_token_idx
        targets = torch.where(
            unseen_mask,
            torch.full_like(labels, unseen_token_idx),
            labels
        )  # [B, S, Nq]
        
        # Flatten for CE loss
        logits_flat = logits.reshape(-1, num_classes)  # [B*S*Nq, num_classes]
        targets_flat = targets.reshape(-1)  # [B*S*Nq]
        
        # Compute CE loss
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )  # [B*S*Nq]
        
        # Reshape and apply unseen weighting
        loss = loss.reshape(B, S, Nq)
        
        if self.weight_unseen != 1.0:
            weights = torch.where(
                unseen_mask,
                torch.full_like(loss, self.weight_unseen),
                torch.ones_like(loss),
            )
            loss = loss * weights
        
        # Mean over all patches
        return loss.mean()
    
    def compute_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        unseen_mask: torch.Tensor,
        unseen_token_idx: int,
    ) -> dict:
        """
        Compute classification accuracy metrics.
        
        Returns:
            dict with 'acc_seen', 'acc_unseen', 'acc_overall'
        """
        B, S, Nq, num_classes = logits.shape
        
        # Get predictions
        preds = logits.argmax(dim=-1)  # [B, S, Nq]
        
        # Create full target labels
        targets = torch.where(
            unseen_mask,
            torch.full_like(labels, unseen_token_idx),
            labels
        )
        
        # Compute accuracy for seen patches
        seen_mask = ~unseen_mask
        if seen_mask.any():
            acc_seen = (preds[seen_mask] == targets[seen_mask]).float().mean().item()
        else:
            acc_seen = 0.0
        
        # Compute accuracy for unseen patches
        if unseen_mask.any():
            acc_unseen = (preds[unseen_mask] == targets[unseen_mask]).float().mean().item()
        else:
            acc_unseen = 0.0
        
        # Overall accuracy
        acc_overall = (preds == targets).float().mean().item()
        
        return {
            'acc_seen': acc_seen,
            'acc_unseen': acc_unseen,
            'acc_overall': acc_overall,
        }
