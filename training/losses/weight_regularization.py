# vpog/losses/weight_regularization.py
#
# Weight regularization (L2) for model parameters
#
# Input:
#   - model: nn.Module
#   - exclude_bias: bool - whether to exclude bias terms (default True)
#   - exclude_norm: bool - whether to exclude norm layers (default True)
#
# Output:
#   - Scalar L2 regularization loss

import torch
import torch.nn as nn


class WeightRegularization(nn.Module):
    """
    L2 weight regularization loss.
    
    Computes L2 norm of model parameters, with options to exclude
    bias terms and normalization layers.
    """
    
    def __init__(
        self,
        weight_decay: float = 1e-4,
        exclude_bias: bool = True,
        exclude_norm: bool = True,
    ):
        """
        Args:
            weight_decay: L2 regularization coefficient (default 1e-4)
            exclude_bias: Exclude bias parameters (default True)
            exclude_norm: Exclude LayerNorm/BatchNorm parameters (default True)
        """
        super().__init__()
        self.weight_decay = weight_decay
        self.exclude_bias = exclude_bias
        self.exclude_norm = exclude_norm
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute L2 regularization loss.
        
        Args:
            model: PyTorch model
        
        Returns:
            Scalar L2 loss
        """
        l2_loss = 0.0
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Skip bias terms if requested
            if self.exclude_bias and 'bias' in name:
                continue
            
            # Skip normalization layers if requested
            if self.exclude_norm:
                if any(x in name.lower() for x in ['norm', 'bn', 'ln']):
                    continue
            
            # Add L2 norm
            l2_loss = l2_loss + param.pow(2).sum()
        
        return self.weight_decay * l2_loss
    
    def get_regularized_params(self, model: nn.Module) -> list:
        """
        Get list of parameter names that will be regularized.
        
        Useful for debugging and verification.
        """
        regularized = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if self.exclude_bias and 'bias' in name:
                continue
            
            if self.exclude_norm:
                if any(x in name.lower() for x in ['norm', 'bn', 'ln']):
                    continue
            
            regularized.append(name)
        
        return regularized
