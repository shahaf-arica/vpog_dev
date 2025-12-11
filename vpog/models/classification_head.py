"""
Classification Head for VPOG

Classifies each query patch as matching one of the template patches or being unseen.
Uses projection-based matching with optional MLP and temperature scaling.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    Classification head for patch matching.
    
    For each query patch, classify which template patch it matches, or if it's unseen.
    
    Architecture:
    1. Project query and template features to matching space
    2. Optional MLP for further processing
    3. Compute similarity between query patches and template patches + unseen token
    4. Apply temperature scaling
    5. Output logits [B, S, Nq, Nt+1] where last class is "unseen"
    
    Args:
        dim: Feature dimension
        num_templates: Number of template patches per sample (S * Nt)
        proj_dim: Projection dimension for matching (default: same as dim)
        use_mlp: Whether to use MLP after projection (default: False)
        mlp_hidden_dim: MLP hidden dimension (default: dim)
        temperature: Temperature for logit scaling (default: 1.0)
        learnable_temperature: Whether temperature is learnable (default: False)
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        dim: int,
        num_templates: Optional[int] = None,  # Can be dynamic
        proj_dim: Optional[int] = None,
        use_mlp: bool = False,
        mlp_hidden_dim: Optional[int] = None,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.proj_dim = proj_dim or dim
        self.use_mlp = use_mlp
        self.num_templates = num_templates
        
        # Query projection
        self.query_proj = nn.Sequential(
            nn.Linear(dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
        )
        
        # Template projection
        self.template_proj = nn.Sequential(
            nn.Linear(dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
        )
        
        # Optional MLP
        if use_mlp:
            mlp_hidden_dim = mlp_hidden_dim or dim
            self.query_mlp = nn.Sequential(
                nn.Linear(self.proj_dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_dim, self.proj_dim),
                nn.Dropout(dropout),
            )
            self.template_mlp = nn.Sequential(
                nn.Linear(self.proj_dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_dim, self.proj_dim),
                nn.Dropout(dropout),
            )
        else:
            self.query_mlp = nn.Identity()
            self.template_mlp = nn.Identity()
        
        # Unseen token (learnable)
        self.unseen_token = nn.Parameter(torch.randn(1, 1, self.proj_dim) * 0.02)
        
        # Temperature
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
    
    def forward(
        self,
        query_features: torch.Tensor,  # [B, Nq, D]
        template_features: torch.Tensor,  # [B, S*Nt, D]
        num_templates_per_sample: int,  # S (number of selected templates)
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query_features: Query patch features [B, Nq, D]
            template_features: Template patch features [B, S*Nt, D]
            num_templates_per_sample: Number of selected templates S
            
        Returns:
            logits: Classification logits [B, S, Nq, Nt+1]
                    Last dimension: [template_0_patches, ..., template_S-1_patches, unseen]
        """
        B, Nq, D = query_features.shape
        B_t, SNt, D_t = template_features.shape
        
        assert B == B_t, f"Batch size mismatch: {B} vs {B_t}"
        assert D == D_t, f"Feature dim mismatch: {D} vs {D_t}"
        
        # Infer Nt (patches per template)
        Nt = SNt // num_templates_per_sample
        S = num_templates_per_sample
        
        assert SNt == S * Nt, f"Template features {SNt} must equal S*Nt ({S}*{Nt})"
        
        # Project features
        query_proj = self.query_proj(query_features)  # [B, Nq, proj_dim]
        template_proj = self.template_proj(template_features)  # [B, S*Nt, proj_dim]
        
        # Apply MLP
        query_proj = self.query_mlp(query_proj)  # [B, Nq, proj_dim]
        template_proj = self.template_mlp(template_proj)  # [B, S*Nt, proj_dim]
        
        # Reshape template features: [B, S*Nt_with_added, proj_dim] -> [B, S, Nt_with_added, proj_dim]
        # NOTE: Template features already include added tokens (e.g., unseen) from TokenManager
        Nt_with_added = SNt // S
        template_proj = template_proj.view(B, S, Nt_with_added, self.proj_dim)
        
        # Normalize features (L2 normalization for cosine similarity)
        query_proj_norm = F.normalize(query_proj, p=2, dim=-1)  # [B, Nq, proj_dim]
        template_proj_norm = F.normalize(template_proj, p=2, dim=-1)  # [B, S, Nt_with_added, proj_dim]
        
        # Compute similarity: dot product between query and templates
        # query: [B, Nq, proj_dim] -> [B, 1, Nq, proj_dim]
        # template: [B, S, Nt_with_added, proj_dim] (includes added tokens like unseen)
        # We want: [B, S, Nq, Nt_with_added]
        
        query_expanded = query_proj_norm.unsqueeze(1)  # [B, 1, Nq, proj_dim]
        template_expanded = template_proj_norm  # [B, S, Nt_with_added, proj_dim]
        
        # Compute similarity for each (query_patch, template)
        # [B, 1, Nq, proj_dim] @ [B, S, proj_dim, Nt_with_added] -> [B, S, Nq, Nt_with_added]
        similarity = torch.einsum('binc,bsmc->bsnm', query_expanded, template_expanded)
        
        # Apply temperature scaling
        logits = similarity / self.temperature
        
        return logits  # [B, S, Nq, Nt_with_added] (last indices are added tokens like unseen)
    
    def get_predictions(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get predicted matches from logits.
        
        Args:
            logits: [B, S, Nq, Nt+1]
            
        Returns:
            predictions: [B, S, Nq] - predicted class indices (Nt means unseen)
            confidences: [B, S, Nq] - confidence scores (max probability)
        """
        probs = F.softmax(logits, dim=-1)  # [B, S, Nq, Nt+1]
        confidences, predictions = probs.max(dim=-1)  # [B, S, Nq]
        
        return predictions, confidences
    
    def get_unseen_mask(self, logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary mask for unseen patches.
        
        Args:
            logits: [B, S, Nq, Nt+1]
            threshold: Confidence threshold for unseen class
            
        Returns:
            unseen_mask: [B, S, Nq] - True for unseen patches
        """
        probs = F.softmax(logits, dim=-1)  # [B, S, Nq, Nt+1]
        unseen_probs = probs[..., -1]  # [B, S, Nq] - probability of unseen class
        unseen_mask = unseen_probs > threshold
        
        return unseen_mask


class ProjectionOnlyHead(nn.Module):
    """
    Simpler classification head with projection only (no MLP).
    
    This is a lightweight version that directly computes cosine similarity
    between projected features.
    """
    
    def __init__(
        self,
        dim: int,
        proj_dim: Optional[int] = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.proj_dim = proj_dim or dim
        
        # Single projection layer for both query and template
        self.projection = nn.Linear(dim, self.proj_dim)
        
        # Unseen token
        self.unseen_token = nn.Parameter(torch.randn(1, 1, self.proj_dim) * 0.02)
        
        # Temperature
        self.register_buffer('temperature', torch.tensor(temperature))
    
    def forward(
        self,
        query_features: torch.Tensor,  # [B, Nq, D]
        template_features: torch.Tensor,  # [B, S*Nt, D]
        num_templates_per_sample: int,  # S
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query_features: [B, Nq, D]
            template_features: [B, S*Nt, D]
            num_templates_per_sample: S
            
        Returns:
            logits: [B, S, Nq, Nt+1]
        """
        B, Nq, D = query_features.shape
        SNt = template_features.shape[1]
        S = num_templates_per_sample
        Nt = SNt // S
        
        # Project
        query_proj = self.projection(query_features)  # [B, Nq, proj_dim]
        template_proj = self.projection(template_features)  # [B, S*Nt, proj_dim]
        
        # Reshape templates
        template_proj = template_proj.view(B, S, Nt, self.proj_dim)
        
        # Add unseen token
        unseen_tokens = self.unseen_token.expand(B, S, -1, -1)
        template_proj_with_unseen = torch.cat([template_proj, unseen_tokens], dim=2)  # [B, S, Nt+1, proj_dim]
        
        # Normalize
        query_proj_norm = F.normalize(query_proj, p=2, dim=-1)
        template_proj_norm = F.normalize(template_proj_with_unseen, p=2, dim=-1)
        
        # Compute similarity
        query_expanded = query_proj_norm.unsqueeze(1)  # [B, 1, Nq, proj_dim]
        similarity = torch.einsum('binc,bsmc->bsnm', query_expanded, template_proj_norm)
        
        # Temperature scaling
        logits = similarity / self.temperature
        
        return logits


def build_classification_head(
    head_type: str = "full",  # "full" or "projection_only"
    dim: int = 768,
    proj_dim: Optional[int] = None,
    use_mlp: bool = False,
    temperature: float = 1.0,
    **kwargs,
) -> nn.Module:
    """
    Build classification head.
    
    Args:
        head_type: "full" (with separate query/template projections) or "projection_only"
        dim: Feature dimension
        proj_dim: Projection dimension
        use_mlp: Whether to use MLP (only for "full")
        temperature: Temperature for logit scaling
        **kwargs: Additional arguments
        
    Returns:
        head: Classification head module
    """
    if head_type == "full":
        return ClassificationHead(
            dim=dim,
            proj_dim=proj_dim,
            use_mlp=use_mlp,
            temperature=temperature,
            **kwargs,
        )
    elif head_type == "projection_only":
        return ProjectionOnlyHead(
            dim=dim,
            proj_dim=proj_dim,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unknown head_type: {head_type}")


if __name__ == "__main__":
    print("Testing Classification Head...")
    
    # Test configuration
    batch_size = 2
    num_query_patches = 196  # 14x14
    num_templates = 4  # S
    num_patches_per_template = 196  # Nt
    dim = 768
    
    # Create features
    query_features = torch.randn(batch_size, num_query_patches, dim)
    template_features = torch.randn(batch_size, num_templates * num_patches_per_template, dim)
    
    # Test full head
    print("\n=== Testing Full Classification Head ===")
    head_full = build_classification_head(
        head_type="full",
        dim=dim,
        proj_dim=512,
        use_mlp=True,
        temperature=1.0,
    )
    
    logits_full = head_full(query_features, template_features, num_templates)
    print(f"Query: {query_features.shape}")
    print(f"Templates: {template_features.shape}")
    print(f"Logits: {logits_full.shape}")  # [B, S, Nq, Nt+1]
    
    # Get predictions
    predictions, confidences = head_full.get_predictions(logits_full)
    print(f"Predictions: {predictions.shape}")  # [B, S, Nq]
    print(f"Confidences: {confidences.shape}")  # [B, S, Nq]
    print(f"Max prediction index: {predictions.max().item()} (should be < {num_patches_per_template + 1})")
    
    # Get unseen mask
    unseen_mask = head_full.get_unseen_mask(logits_full, threshold=0.5)
    print(f"Unseen mask: {unseen_mask.shape}, Num unseen: {unseen_mask.sum().item()}")
    
    # Test projection-only head
    print("\n=== Testing Projection-Only Head ===")
    head_proj = build_classification_head(
        head_type="projection_only",
        dim=dim,
        proj_dim=512,
        temperature=1.0,
    )
    
    logits_proj = head_proj(query_features, template_features, num_templates)
    print(f"Logits: {logits_proj.shape}")  # [B, S, Nq, Nt+1]
    
    # Check temperature effect
    print("\n=== Testing Temperature Effect ===")
    head_temp = ClassificationHead(dim=dim, temperature=0.1)  # Lower temp = sharper
    logits_sharp = head_temp(query_features, template_features, num_templates)
    
    probs_normal = F.softmax(logits_full, dim=-1)
    probs_sharp = F.softmax(logits_sharp, dim=-1)
    
    print(f"Normal temp entropy: {-(probs_normal * (probs_normal + 1e-10).log()).sum(-1).mean().item():.4f}")
    print(f"Sharp temp entropy: {-(probs_sharp * (probs_sharp + 1e-10).log()).sum(-1).mean().item():.4f}")
    
    print("\n✓ Classification head test passed!")
