"""
Flow Head for VPOG

Predicts 16x16 pixel-level optical flow within each patch pair.
Flow represents Template → Query displacement with delta_x=1.0 meaning one full patch movement.
Also outputs per-pixel confidence scores.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    """
    Flow head for predicting dense optical flow within patches.
    
    For each (query_patch, template_patch) pair, predict:
    - 16x16 flow vectors (dx, dy) representing pixel displacement
    - 16x16 confidence scores
    
    Flow convention:
    - delta_x = 1.0 means displacement by one full patch (16 pixels)
    - Flow is from Template → Query
    - Positive dx means rightward motion, positive dy means downward
    
    Args:
        dim: Feature dimension
        patch_size: Patch size in pixels (default: 16)
        flow_resolution: Flow prediction resolution (default: 16x16)
        hidden_dims: List of hidden dimensions for decoder (default: [512, 256, 128])
        use_confidence: Whether to predict confidence (default: True)
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        dim: int,
        patch_size: int = 16,
        flow_resolution: int = 16,
        hidden_dims: Optional[list[int]] = None,
        use_confidence: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.patch_size = patch_size
        self.flow_resolution = flow_resolution
        self.use_confidence = use_confidence
        
        # Default hidden dims
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        # Feature fusion: concatenate query and template features
        fusion_dim = dim * 2
        
        # Build decoder network
        layers = []
        in_dim = fusion_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        
        # Flow prediction head
        # Output: flow_resolution^2 * 2 (for dx, dy)
        self.flow_predictor = nn.Linear(in_dim, flow_resolution * flow_resolution * 2)
        
        # Confidence prediction head (optional)
        if use_confidence:
            self.confidence_predictor = nn.Sequential(
                nn.Linear(in_dim, flow_resolution * flow_resolution),
                nn.Sigmoid(),  # Confidence in [0, 1]
            )
        else:
            self.confidence_predictor = None
        
        # Initialize flow predictor with small weights (near-zero initialization)
        nn.init.normal_(self.flow_predictor.weight, std=0.001)
        nn.init.zeros_(self.flow_predictor.bias)
    
    def forward(
        self,
        query_features: torch.Tensor,  # [B, Nq, D]
        template_features: torch.Tensor,  # [B, S*Nt, D]
        num_templates_per_sample: int,  # S
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            query_features: Query patch features [B, Nq, D]
            template_features: Template patch features [B, S*Nt, D]
            num_templates_per_sample: Number of selected templates S
            
        Returns:
            flow: Flow predictions [B, S, Nq, flow_res, flow_res, 2]
                  Values represent displacement in patch units (delta_x=1.0 = one patch)
            confidence: Confidence scores [B, S, Nq, flow_res, flow_res, 1] or None
                       Values in [0, 1], higher means more confident
        """
        B, Nq, D = query_features.shape
        SNt = template_features.shape[1]
        S = num_templates_per_sample
        Nt = SNt // S
        
        # Reshape template features: [B, S*Nt, D] -> [B, S, Nt, D]
        template_features = template_features.view(B, S, Nt, D)
        
        # Expand query features for broadcasting: [B, Nq, D] -> [B, 1, Nq, D] -> [B, S, Nq, D]
        query_expanded = query_features.unsqueeze(1).expand(B, S, Nq, D)
        
        # For each template, compute flow for all query patches vs all template patches
        # We want: [B, S, Nq, Nt, D] for query and template pairs
        # But flow is only meaningful for matched pairs, so we compute flow for all pairs
        # and let the classification head select which flows to use
        
        # Expand for all pairs: query [B, S, Nq, 1, D], template [B, S, 1, Nt, D]
        query_for_pairs = query_expanded.unsqueeze(3)  # [B, S, Nq, 1, D]
        template_for_pairs = template_features.unsqueeze(2)  # [B, S, 1, Nt, D]
        
        # Broadcast to [B, S, Nq, Nt, D]
        query_for_pairs = query_for_pairs.expand(B, S, Nq, Nt, D)
        template_for_pairs = template_for_pairs.expand(B, S, Nq, Nt, D)
        
        # Concatenate features
        fused_features = torch.cat([query_for_pairs, template_for_pairs], dim=-1)  # [B, S, Nq, Nt, 2D]
        
        # Flatten for decoder
        fused_flat = fused_features.view(B * S * Nq * Nt, -1)  # [B*S*Nq*Nt, 2D]
        
        # Decode
        decoded = self.decoder(fused_flat)  # [B*S*Nq*Nt, hidden_dim]
        
        # Predict flow
        flow_flat = self.flow_predictor(decoded)  # [B*S*Nq*Nt, flow_res^2 * 2]
        flow_flat = flow_flat.view(B, S, Nq, Nt, self.flow_resolution, self.flow_resolution, 2)
        
        # Predict confidence
        if self.use_confidence:
            confidence_flat = self.confidence_predictor(decoded)  # [B*S*Nq*Nt, flow_res^2]
            confidence_flat = confidence_flat.view(B, S, Nq, Nt, self.flow_resolution, self.flow_resolution, 1)
        else:
            confidence_flat = None
        
        return flow_flat, confidence_flat
    
    def get_flow_for_matches(
        self,
        flow_all: torch.Tensor,  # [B, S, Nq, Nt, flow_res, flow_res, 2]
        matches: torch.Tensor,  # [B, S, Nq] - matched template patch indices
        confidence_all: Optional[torch.Tensor] = None,  # [B, S, Nq, Nt, flow_res, flow_res, 1]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract flow predictions for matched pairs only.
        
        Args:
            flow_all: Flow for all pairs [B, S, Nq, Nt, flow_res, flow_res, 2]
            matches: Matched template indices [B, S, Nq]
            confidence_all: Confidence for all pairs (optional)
            
        Returns:
            flow_matched: Flow for matched pairs [B, S, Nq, flow_res, flow_res, 2]
            confidence_matched: Confidence for matched pairs [B, S, Nq, flow_res, flow_res, 1] or None
        """
        B, S, Nq, Nt = flow_all.shape[:4]
        
        # Gather flow for matched indices
        # matches: [B, S, Nq] -> [B, S, Nq, 1, 1, 1, 1]
        matches_expanded = matches.view(B, S, Nq, 1, 1, 1, 1).expand(
            B, S, Nq, 1, self.flow_resolution, self.flow_resolution, 2
        )
        
        # Gather
        flow_matched = torch.gather(flow_all, 3, matches_expanded).squeeze(3)  # [B, S, Nq, flow_res, flow_res, 2]
        
        if confidence_all is not None:
            conf_matches_expanded = matches.view(B, S, Nq, 1, 1, 1, 1).expand(
                B, S, Nq, 1, self.flow_resolution, self.flow_resolution, 1
            )
            confidence_matched = torch.gather(confidence_all, 3, conf_matches_expanded).squeeze(3)
        else:
            confidence_matched = None
        
        return flow_matched, confidence_matched


class LightweightFlowHead(nn.Module):
    """
    Lightweight flow head with shared decoder for query patches.
    
    This version processes all template patches for a query patch together,
    reducing computation compared to the full pairwise version.
    """
    
    def __init__(
        self,
        dim: int,
        patch_size: int = 16,
        flow_resolution: int = 16,
        hidden_dim: int = 256,
        use_confidence: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.patch_size = patch_size
        self.flow_resolution = flow_resolution
        self.use_confidence = use_confidence
        
        # Simple decoder
        self.decoder = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Flow predictor
        self.flow_predictor = nn.Linear(hidden_dim // 2, flow_resolution * flow_resolution * 2)
        
        # Confidence predictor
        if use_confidence:
            self.confidence_predictor = nn.Sequential(
                nn.Linear(hidden_dim // 2, flow_resolution * flow_resolution),
                nn.Sigmoid(),
            )
        
        # Initialize
        nn.init.normal_(self.flow_predictor.weight, std=0.001)
        nn.init.zeros_(self.flow_predictor.bias)
    
    def forward(
        self,
        query_features: torch.Tensor,
        template_features: torch.Tensor,
        num_templates_per_sample: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass - same interface as FlowHead."""
        B, Nq, D = query_features.shape
        SNt = template_features.shape[1]
        S = num_templates_per_sample
        Nt = SNt // S
        
        # Reshape
        template_features = template_features.view(B, S, Nt, D)
        query_expanded = query_features.unsqueeze(1).expand(B, S, Nq, D)
        
        # Pairwise fusion
        query_for_pairs = query_expanded.unsqueeze(3).expand(B, S, Nq, Nt, D)
        template_for_pairs = template_features.unsqueeze(2).expand(B, S, Nq, Nt, D)
        fused = torch.cat([query_for_pairs, template_for_pairs], dim=-1)
        
        # Process
        fused_flat = fused.view(B * S * Nq * Nt, -1)
        decoded = self.decoder(fused_flat)
        
        # Predict
        flow = self.flow_predictor(decoded)
        flow = flow.view(B, S, Nq, Nt, self.flow_resolution, self.flow_resolution, 2)
        
        if self.use_confidence:
            confidence = self.confidence_predictor(decoded)
            confidence = confidence.view(B, S, Nq, Nt, self.flow_resolution, self.flow_resolution, 1)
        else:
            confidence = None
        
        return flow, confidence


def build_flow_head(
    head_type: str = "full",  # "full" or "lightweight"
    dim: int = 768,
    patch_size: int = 16,
    flow_resolution: int = 16,
    **kwargs,
) -> nn.Module:
    """
    Build flow head.
    
    Args:
        head_type: "full" (multi-layer decoder) or "lightweight" (simpler decoder)
        dim: Feature dimension
        patch_size: Patch size
        flow_resolution: Flow prediction resolution
        **kwargs: Additional arguments
        
    Returns:
        head: Flow head module
    """
    if head_type == "full":
        return FlowHead(
            dim=dim,
            patch_size=patch_size,
            flow_resolution=flow_resolution,
            **kwargs,
        )
    elif head_type == "lightweight":
        return LightweightFlowHead(
            dim=dim,
            patch_size=patch_size,
            flow_resolution=flow_resolution,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown head_type: {head_type}")


if __name__ == "__main__":
    print("Testing Flow Head...")
    
    # Test configuration
    batch_size = 2
    num_query_patches = 196  # 14x14
    num_templates = 4  # S
    num_patches_per_template = 196  # Nt
    dim = 768
    flow_resolution = 16
    
    # Create features
    query_features = torch.randn(batch_size, num_query_patches, dim)
    template_features = torch.randn(batch_size, num_templates * num_patches_per_template, dim)
    
    # Test full flow head
    print("\n=== Testing Full Flow Head ===")
    head_full = build_flow_head(
        head_type="full",
        dim=dim,
        flow_resolution=flow_resolution,
        use_confidence=True,
    )
    
    flow, confidence = head_full(query_features, template_features, num_templates)
    print(f"Query: {query_features.shape}")
    print(f"Templates: {template_features.shape}")
    print(f"Flow: {flow.shape}")  # [B, S, Nq, Nt, flow_res, flow_res, 2]
    print(f"Confidence: {confidence.shape}")  # [B, S, Nq, Nt, flow_res, flow_res, 1]
    
    # Check flow statistics
    print(f"\nFlow statistics:")
    print(f"  Mean: {flow.mean().item():.4f}")
    print(f"  Std: {flow.std().item():.4f}")
    print(f"  Min: {flow.min().item():.4f}")
    print(f"  Max: {flow.max().item():.4f}")
    
    print(f"Confidence statistics:")
    print(f"  Mean: {confidence.mean().item():.4f}")
    print(f"  Min: {confidence.min().item():.4f}")
    print(f"  Max: {confidence.max().item():.4f}")
    
    # Test flow extraction for matches
    print("\n=== Testing Flow Extraction ===")
    matches = torch.randint(0, num_patches_per_template, (batch_size, num_templates, num_query_patches))
    flow_matched, conf_matched = head_full.get_flow_for_matches(flow, matches, confidence)
    
    print(f"Matches: {matches.shape}")
    print(f"Matched flow: {flow_matched.shape}")  # [B, S, Nq, flow_res, flow_res, 2]
    print(f"Matched confidence: {conf_matched.shape}")  # [B, S, Nq, flow_res, flow_res, 1]
    
    # Test lightweight head
    print("\n=== Testing Lightweight Flow Head ===")
    head_light = build_flow_head(
        head_type="lightweight",
        dim=dim,
        flow_resolution=flow_resolution,
        hidden_dim=256,
    )
    
    flow_light, conf_light = head_light(query_features, template_features, num_templates)
    print(f"Flow: {flow_light.shape}")
    print(f"Confidence: {conf_light.shape}")
    
    # Test without confidence
    print("\n=== Testing Without Confidence ===")
    head_no_conf = FlowHead(dim=dim, flow_resolution=flow_resolution, use_confidence=False)
    flow_nc, conf_nc = head_no_conf(query_features, template_features, num_templates)
    print(f"Flow: {flow_nc.shape}")
    print(f"Confidence: {conf_nc}")  # Should be None
    
    print("\n✓ Flow head test passed!")
