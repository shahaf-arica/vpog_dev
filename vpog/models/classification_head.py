"""
Classification Head for VPOG

Classifies each query patch as matching one of the template patches or being unseen.
Uses projection-based matching with optional MLP and temperature scaling.
"""
# classification_head.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ClassificationHeadConfig:
    in_dim: int                 # encoder dim C
    proj_dim: int = 256         # d
    hidden_dim: int = 512       # MLP hidden
    num_layers: int = 2         # MLP depth (>=1)
    dropout: float = 0.0
    use_layernorm: bool = True
    normalize: bool = True      # L2-normalize embeddings before dot product
    learnable_temperature: bool = True
    init_temperature: float = 1 / 0.07  # typical CLIP-ish starting point


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
        use_layernorm: bool,
    ):
        super().__init__()
        assert num_layers >= 1

        layers = []
        d0 = in_dim
        for li in range(num_layers - 1):
            layers.append(nn.Linear(d0, hidden_dim, bias=True))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d0 = hidden_dim

        layers.append(nn.Linear(d0, out_dim, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassificationHead(nn.Module):
    """
    Patch-wise classification head.

    Inputs:
      q_tokens: [B, Nq, C]
      t_tokens: [B, S, Nt1, C]   where Nt1 = Nt + 1 (includes unseen token)
        OR
      t_tokens: [B, S*Nt1, C] with num_templates=S

    Output:
      logits: [B, S, Nq, Nt1]
        logits[b,s,i,j] is the dot-product similarity between query patch i
        and template token j (j in 0..Nt plus unseen at Nt).
    """

    def __init__(self, cfg: ClassificationHeadConfig):
        super().__init__()
        self.cfg = cfg

        self.q_proj = MLP(
            in_dim=cfg.in_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.proj_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            use_layernorm=cfg.use_layernorm,
        )
        self.t_proj = MLP(
            in_dim=cfg.in_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.proj_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            use_layernorm=cfg.use_layernorm,
        )

        # Learnable temperature (logit scale), CLIP-style.
        if cfg.learnable_temperature:
            # Store log(scale) so scale stays positive.
            init = float(cfg.init_temperature)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(init), dtype=torch.float32))
        else:
            self.register_buffer("logit_scale", torch.tensor(0.0), persistent=False)

    def _reshape_templates(
        self,
        t_tokens: torch.Tensor,
        num_templates: Optional[int],
    ) -> torch.Tensor:
        """
        Returns t_tokens_4d: [B, S, Nt1, C]
        """
        if t_tokens.dim() == 4:
            return t_tokens

        if t_tokens.dim() != 3:
            raise ValueError(f"t_tokens must be [B,S,Nt1,C] or [B,S*Nt1,C]. Got {tuple(t_tokens.shape)}")

        if num_templates is None:
            raise ValueError("num_templates must be provided when t_tokens is flattened [B,S*Nt1,C].")

        B, SNt1, C = t_tokens.shape
        S = int(num_templates)
        if SNt1 % S != 0:
            raise ValueError(f"Flattened template tokens length {SNt1} not divisible by S={S}.")
        Nt1 = SNt1 // S
        return t_tokens.view(B, S, Nt1, C)
    
    def forward(
        self,
        q_tokens: torch.Tensor,                 # [B, Nq, C]
        t_tokens: torch.Tensor,                 # [B, S, Nt1, C] OR [B, S*Nt1, C]
        num_templates: Optional[int] = None,    # required only for flattened case
        return_debug: bool = False,
        debug_max_vec_elems: int = 32,
    ):
        """
        Returns:
        - logits: [B, S, Nq, Nt1]
        - if return_debug=True: (logits, debug_dict) with small slices and finiteness stats
        """
        if q_tokens.dim() != 3:
            raise ValueError(f"q_tokens must be [B,Nq,C]. Got {tuple(q_tokens.shape)}")

        t_tokens = self._reshape_templates(t_tokens, num_templates=num_templates)

        B, Nq, Cq = q_tokens.shape
        Bt, S, Nt1, Ct = t_tokens.shape
        if Bt != B:
            raise ValueError(f"Batch mismatch: q B={B} vs t B={Bt}")
        if Cq != Ct:
            raise ValueError(f"Channel mismatch: q C={Cq} vs t C={Ct}")

        debug = {} if return_debug else None

        # ---- Basic input stats (these are the tokens coming from AA) ----
        if return_debug:
            debug.update({
                "q_tokens_absmax": q_tokens.detach().abs().max().float().cpu(),
                "t_tokens_absmax": t_tokens.detach().abs().max().float().cpu(),
                "q_tokens_has_nan": torch.isnan(q_tokens).any().cpu(),
                "q_tokens_has_inf": torch.isinf(q_tokens).any().cpu(),
                "t_tokens_has_nan": torch.isnan(t_tokens).any().cpu(),
                "t_tokens_has_inf": torch.isinf(t_tokens).any().cpu(),
            })

        # ---- Project ----
        q_proj = self.q_proj(q_tokens)      # [B, Nq, d]
        t_proj = self.t_proj(t_tokens)      # [B, S, Nt1, d]

        if return_debug:
            q_norm = q_proj.detach().norm(dim=-1)          # [B,Nq]
            t_norm = t_proj.detach().norm(dim=-1)          # [B,S,Nt1]
            debug.update({
                "q_proj_absmax": q_proj.detach().abs().max().float().cpu(),
                "t_proj_absmax": t_proj.detach().abs().max().float().cpu(),
                "q_proj_has_nan": torch.isnan(q_proj).any().cpu(),
                "q_proj_has_inf": torch.isinf(q_proj).any().cpu(),
                "t_proj_has_nan": torch.isnan(t_proj).any().cpu(),
                "t_proj_has_inf": torch.isinf(t_proj).any().cpu(),
                "q_proj_norm_min": q_norm.min().float().cpu(),
                "q_proj_norm_max": q_norm.max().float().cpu(),
                "t_proj_norm_min": t_norm.min().float().cpu(),
                "t_proj_norm_max": t_norm.max().float().cpu(),
            })

            # Max-abs query token slice (b, i)
            q_absmax_per_tok = q_proj.detach().abs().amax(dim=-1)  # [B,Nq]
            q_flat_idx = int(q_absmax_per_tok.reshape(-1).argmax().item())
            qb = q_flat_idx // Nq
            qi = q_flat_idx % Nq
            debug.update({
                "q_proj_maxloc_b": torch.tensor(qb),
                "q_proj_maxloc_i": torch.tensor(qi),
                "q_proj_maxloc_absmax": q_absmax_per_tok[qb, qi].float().cpu(),
                "q_proj_maxloc_slice": q_proj.detach()[qb, qi, :debug_max_vec_elems].float().cpu(),
            })

            # Max-abs template token slice (b, s, j)
            t_absmax_per_tok = t_proj.detach().abs().amax(dim=-1)  # [B,S,Nt1]
            t_flat_idx = int(t_absmax_per_tok.reshape(-1).argmax().item())
            tb = t_flat_idx // (S * Nt1)
            rem = t_flat_idx % (S * Nt1)
            ts = rem // Nt1
            tj = rem % Nt1
            debug.update({
                "t_proj_maxloc_b": torch.tensor(tb),
                "t_proj_maxloc_s": torch.tensor(ts),
                "t_proj_maxloc_j": torch.tensor(tj),
                "t_proj_maxloc_absmax": t_absmax_per_tok[tb, ts, tj].float().cpu(),
                "t_proj_maxloc_slice": t_proj.detach()[tb, ts, tj, :debug_max_vec_elems].float().cpu(),
            })

        # ---- Normalize (recommended) ----
        if self.cfg.normalize:
            q = F.normalize(q_proj, dim=-1)
            t = F.normalize(t_proj, dim=-1)
        else:
            q, t = q_proj, t_proj

        if return_debug:
            debug.update({
                "q_normed_has_nan": torch.isnan(q).any().cpu(),
                "q_normed_has_inf": torch.isinf(q).any().cpu(),
                "t_normed_has_nan": torch.isnan(t).any().cpu(),
                "t_normed_has_inf": torch.isinf(t).any().cpu(),
            })

        # ---- Dot-product logits ----
        # logits[b,s,i,j] = sum_d q[b,i,d] * t[b,s,j,d]
        logits = torch.einsum("bid,bsjd->bsij", q, t)  # [B, S, Nq, Nt1]

        if return_debug:
            nan_mask = torch.isnan(logits)
            inf_mask = torch.isinf(logits)
            debug.update({
                "logits_has_nan": nan_mask.any().cpu(),
                "logits_has_inf": inf_mask.any().cpu(),
                "logits_absmax": logits.detach().abs().max().float().cpu(),
            })

            if nan_mask.any():
                b, s, i, j = nan_mask.nonzero(as_tuple=False)[0].tolist()
                debug.update({
                    "first_nan_b": torch.tensor(b),
                    "first_nan_s": torch.tensor(s),
                    "first_nan_i": torch.tensor(i),
                    "first_nan_j": torch.tensor(j),
                    "q_at_first_nan_slice": q.detach()[b, i, :debug_max_vec_elems].float().cpu(),
                    "t_at_first_nan_slice": t.detach()[b, s, j, :debug_max_vec_elems].float().cpu(),
                })

                # “196 NaNs” signature: identify template token indices j responsible for many NaNs
                per_j = nan_mask[b, s].sum(dim=0).to(torch.int32).cpu()  # [Nt1]
                topk = min(5, per_j.numel())
                vals, idxs = torch.topk(per_j, k=topk)
                debug.update({
                    "nan_per_j_topk_counts": vals,
                    "nan_per_j_topk_js": idxs,
                })

        # ---- Apply temperature / scaling ----
        if self.cfg.learnable_temperature:
            scale = self.logit_scale.exp().clamp(min=1e-3, max=100.0)
            logits = logits * scale
            if return_debug:
                debug.update({
                    "logit_scale_value": self.logit_scale.detach().float().cpu(),
                    "scale_value": scale.detach().float().cpu(),
                    "logit_scale_has_nan": torch.isnan(self.logit_scale).any().cpu(),
                    "logit_scale_has_inf": torch.isinf(self.logit_scale).any().cpu(),
                })
        else:
            if not self.cfg.normalize:
                logits = logits / math.sqrt(self.cfg.proj_dim)

        if return_debug:
            debug["logits_final_has_nan"] = torch.isnan(logits).any().cpu()
            debug["logits_final_has_inf"] = torch.isinf(logits).any().cpu()
            return logits, debug

        return logits

    # def forward(
    #     self,
    #     q_tokens: torch.Tensor,                 # [B, Nq, C]
    #     t_tokens: torch.Tensor,                 # [B, S, Nt1, C] OR [B, S*Nt1, C]
    #     num_templates: Optional[int] = None,    # required only for flattened case
    # ) -> torch.Tensor:
    #     if q_tokens.dim() != 3:
    #         raise ValueError(f"q_tokens must be [B,Nq,C]. Got {tuple(q_tokens.shape)}")

    #     t_tokens = self._reshape_templates(t_tokens, num_templates=num_templates)

    #     B, Nq, Cq = q_tokens.shape
    #     Bt, S, Nt1, Ct = t_tokens.shape
    #     if Bt != B:
    #         raise ValueError(f"Batch mismatch: q B={B} vs t B={Bt}")
    #     if Cq != Ct:
    #         raise ValueError(f"Channel mismatch: q C={Cq} vs t C={Ct}")

    #     # Project
    #     q = self.q_proj(q_tokens)      # [B, Nq, d]
    #     t = self.t_proj(t_tokens)      # [B, S, Nt1, d]

    #     # Normalize (recommended)
    #     if self.cfg.normalize:
    #         q = F.normalize(q, dim=-1)
    #         t = F.normalize(t, dim=-1)

    #     # Dot-product logits over embedding dim (no loops):
    #     # logits[b,s,i,j] = sum_d q[b,i,d] * t[b,s,j,d]
    #     logits = torch.einsum("bid,bsjd->bsij", q, t)  # [B, S, Nq, Nt1]

    #     # Apply temperature / scaling
    #     if self.cfg.learnable_temperature:
    #         # clamp helps prevent exploding scales early in training
    #         scale = self.logit_scale.exp().clamp(min=1e-3, max=100.0)
    #         logits = logits * scale
    #     else:
    #         # If not learnable, use standard transformer scaling if not normalized
    #         if not self.cfg.normalize:
    #             logits = logits / math.sqrt(self.cfg.proj_dim)

    #     return logits





# from typing import Optional

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class ClassificationHead(nn.Module):
#     """
#     Classification head for patch matching.
    
#     For each query patch, classify which template patch it matches, or if it's unseen.
    
#     Architecture:
#     1. Project query and template features to matching space
#     2. Optional MLP for further processing
#     3. Compute similarity between query patches and template patches + unseen token
#     4. Apply temperature scaling
#     5. Output logits [B, S, Nq, Nt+1] where last class is "unseen"
    
#     Args:
#         dim: Feature dimension
#         num_templates: Number of template patches per sample (S * Nt)
#         proj_dim: Projection dimension for matching (default: same as dim)
#         use_mlp: Whether to use MLP after projection (default: False)
#         mlp_hidden_dim: MLP hidden dimension (default: dim)
#         temperature: Temperature for logit scaling (default: 1.0)
#         learnable_temperature: Whether temperature is learnable (default: False)
#         dropout: Dropout rate (default: 0.0)
#     """
    
#     def __init__(
#         self,
#         dim: int,
#         num_templates: Optional[int] = None,  # Can be dynamic
#         proj_dim: Optional[int] = None,
#         use_mlp: bool = False,
#         mlp_hidden_dim: Optional[int] = None,
#         temperature: float = 1.0,
#         learnable_temperature: bool = False,
#         dropout: float = 0.0,
#     ):
#         super().__init__()
        
#         self.dim = dim
#         self.proj_dim = proj_dim or dim
#         self.use_mlp = use_mlp
#         self.num_templates = num_templates
        
#         # Query projection
#         self.query_proj = nn.Sequential(
#             nn.Linear(dim, self.proj_dim),
#             nn.LayerNorm(self.proj_dim),
#         )
        
#         # Template projection
#         self.template_proj = nn.Sequential(
#             nn.Linear(dim, self.proj_dim),
#             nn.LayerNorm(self.proj_dim),
#         )
        
#         # Optional MLP
#         if use_mlp:
#             mlp_hidden_dim = mlp_hidden_dim or dim
#             self.query_mlp = nn.Sequential(
#                 nn.Linear(self.proj_dim, mlp_hidden_dim),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(mlp_hidden_dim, self.proj_dim),
#                 nn.Dropout(dropout),
#             )
#             self.template_mlp = nn.Sequential(
#                 nn.Linear(self.proj_dim, mlp_hidden_dim),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(mlp_hidden_dim, self.proj_dim),
#                 nn.Dropout(dropout),
#             )
#         else:
#             self.query_mlp = nn.Identity()
#             self.template_mlp = nn.Identity()
        
#         # Unseen token (learnable)
#         self.unseen_token = nn.Parameter(torch.randn(1, 1, self.proj_dim) * 0.02)
        
#         # Temperature
#         if learnable_temperature:
#             self.temperature = nn.Parameter(torch.tensor(temperature))
#         else:
#             self.register_buffer('temperature', torch.tensor(temperature))
    
#     def forward(
#         self,
#         query_features: torch.Tensor,  # [B, Nq, D]
#         template_features: torch.Tensor,  # [B, S*Nt, D]
#         num_templates_per_sample: int,  # S (number of selected templates)
#     ) -> torch.Tensor:
#         """
#         Forward pass.
        
#         Args:
#             query_features: Query patch features [B, Nq, D]
#             template_features: Template patch features [B, S*Nt, D]
#             num_templates_per_sample: Number of selected templates S
            
#         Returns:
#             logits: Classification logits [B, S, Nq, Nt+1]
#                     Last dimension: [template_0_patches, ..., template_S-1_patches, unseen]
#         """
#         B, Nq, D = query_features.shape
#         B_t, SNt, D_t = template_features.shape
        
#         assert B == B_t, f"Batch size mismatch: {B} vs {B_t}"
#         assert D == D_t, f"Feature dim mismatch: {D} vs {D_t}"
        
#         # Infer Nt (patches per template)
#         Nt = SNt // num_templates_per_sample
#         S = num_templates_per_sample
        
#         assert SNt == S * Nt, f"Template features {SNt} must equal S*Nt ({S}*{Nt})"
        
#         # Project features
#         query_proj = self.query_proj(query_features)  # [B, Nq, proj_dim]
#         template_proj = self.template_proj(template_features)  # [B, S*Nt, proj_dim]
        
#         # Apply MLP
#         query_proj = self.query_mlp(query_proj)  # [B, Nq, proj_dim]
#         template_proj = self.template_mlp(template_proj)  # [B, S*Nt, proj_dim]
        
#         # Reshape template features: [B, S*Nt_with_added, proj_dim] -> [B, S, Nt_with_added, proj_dim]
#         # NOTE: Template features already include added tokens (e.g., unseen) from TokenManager
#         Nt_with_added = SNt // S
#         template_proj = template_proj.view(B, S, Nt_with_added, self.proj_dim)
        
#         # Normalize features (L2 normalization for cosine similarity)
#         query_proj_norm = F.normalize(query_proj, p=2, dim=-1)  # [B, Nq, proj_dim]
#         template_proj_norm = F.normalize(template_proj, p=2, dim=-1)  # [B, S, Nt_with_added, proj_dim]
        
#         # Compute similarity: dot product between query and templates
#         # query: [B, Nq, proj_dim] -> [B, 1, Nq, proj_dim]
#         # template: [B, S, Nt_with_added, proj_dim] (includes added tokens like unseen)
#         # We want: [B, S, Nq, Nt_with_added]
        
#         query_expanded = query_proj_norm.unsqueeze(1)  # [B, 1, Nq, proj_dim]
#         template_expanded = template_proj_norm  # [B, S, Nt_with_added, proj_dim]
        
#         # Compute similarity for each (query_patch, template)
#         # [B, 1, Nq, proj_dim] @ [B, S, proj_dim, Nt_with_added] -> [B, S, Nq, Nt_with_added]
#         similarity = torch.einsum('binc,bsmc->bsnm', query_expanded, template_expanded)
        
#         # Apply temperature scaling
#         logits = similarity / self.temperature
        
#         return logits  # [B, S, Nq, Nt_with_added] (last indices are added tokens like unseen)
    
#     def get_predictions(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         Get predicted matches from logits.
        
#         Args:
#             logits: [B, S, Nq, Nt+1]
            
#         Returns:
#             predictions: [B, S, Nq] - predicted class indices (Nt means unseen)
#             confidences: [B, S, Nq] - confidence scores (max probability)
#         """
#         probs = F.softmax(logits, dim=-1)  # [B, S, Nq, Nt+1]
#         confidences, predictions = probs.max(dim=-1)  # [B, S, Nq]
        
#         return predictions, confidences
    
#     def get_unseen_mask(self, logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
#         """
#         Get binary mask for unseen patches.
        
#         Args:
#             logits: [B, S, Nq, Nt+1]
#             threshold: Confidence threshold for unseen class
            
#         Returns:
#             unseen_mask: [B, S, Nq] - True for unseen patches
#         """
#         probs = F.softmax(logits, dim=-1)  # [B, S, Nq, Nt+1]
#         unseen_probs = probs[..., -1]  # [B, S, Nq] - probability of unseen class
#         unseen_mask = unseen_probs > threshold
        
#         return unseen_mask


# class ProjectionOnlyHead(nn.Module):
#     """
#     Simpler classification head with projection only (no MLP).
    
#     This is a lightweight version that directly computes cosine similarity
#     between projected features.
#     """
    
#     def __init__(
#         self,
#         dim: int,
#         proj_dim: Optional[int] = None,
#         temperature: float = 1.0,
#     ):
#         super().__init__()
        
#         self.dim = dim
#         self.proj_dim = proj_dim or dim
        
#         # Single projection layer for both query and template
#         self.projection = nn.Linear(dim, self.proj_dim)
        
#         # Unseen token
#         self.unseen_token = nn.Parameter(torch.randn(1, 1, self.proj_dim) * 0.02)
        
#         # Temperature
#         self.register_buffer('temperature', torch.tensor(temperature))
    
#     def forward(
#         self,
#         query_features: torch.Tensor,  # [B, Nq, D]
#         template_features: torch.Tensor,  # [B, S*Nt, D]
#         num_templates_per_sample: int,  # S
#     ) -> torch.Tensor:
#         """
#         Forward pass.
        
#         Args:
#             query_features: [B, Nq, D]
#             template_features: [B, S*Nt, D]
#             num_templates_per_sample: S
            
#         Returns:
#             logits: [B, S, Nq, Nt+1]
#         """
#         B, Nq, D = query_features.shape
#         SNt = template_features.shape[1]
#         S = num_templates_per_sample
#         Nt = SNt // S
        
#         # Project
#         query_proj = self.projection(query_features)  # [B, Nq, proj_dim]
#         template_proj = self.projection(template_features)  # [B, S*Nt, proj_dim]
        
#         # Reshape templates
#         template_proj = template_proj.view(B, S, Nt, self.proj_dim)
        
#         # Add unseen token
#         unseen_tokens = self.unseen_token.expand(B, S, -1, -1)
#         template_proj_with_unseen = torch.cat([template_proj, unseen_tokens], dim=2)  # [B, S, Nt+1, proj_dim]
        
#         # Normalize
#         query_proj_norm = F.normalize(query_proj, p=2, dim=-1)
#         template_proj_norm = F.normalize(template_proj_with_unseen, p=2, dim=-1)
        
#         # Compute similarity
#         query_expanded = query_proj_norm.unsqueeze(1)  # [B, 1, Nq, proj_dim]
#         similarity = torch.einsum('binc,bsmc->bsnm', query_expanded, template_proj_norm)
        
#         # Temperature scaling
#         logits = similarity / self.temperature
        
#         return logits


# def build_classification_head(
#     head_type: str = "full",  # "full" or "projection_only"
#     dim: int = 768,
#     proj_dim: Optional[int] = None,
#     use_mlp: bool = False,
#     temperature: float = 1.0,
#     **kwargs,
# ) -> nn.Module:
#     """
#     Build classification head.
    
#     Args:
#         head_type: "full" (with separate query/template projections) or "projection_only"
#         dim: Feature dimension
#         proj_dim: Projection dimension
#         use_mlp: Whether to use MLP (only for "full")
#         temperature: Temperature for logit scaling
#         **kwargs: Additional arguments
        
#     Returns:
#         head: Classification head module
#     """
#     if head_type == "full":
#         return ClassificationHead(
#             dim=dim,
#             proj_dim=proj_dim,
#             use_mlp=use_mlp,
#             temperature=temperature,
#             **kwargs,
#         )
#     elif head_type == "projection_only":
#         return ProjectionOnlyHead(
#             dim=dim,
#             proj_dim=proj_dim,
#             temperature=temperature,
#         )
#     else:
#         raise ValueError(f"Unknown head_type: {head_type}")


# if __name__ == "__main__":
#     print("Testing Classification Head...")
    
#     # Test configuration
#     batch_size = 2
#     num_query_patches = 196  # 14x14
#     num_templates = 4  # S
#     num_patches_per_template = 196  # Nt
#     dim = 768
    
#     # Create features
#     query_features = torch.randn(batch_size, num_query_patches, dim)
#     template_features = torch.randn(batch_size, num_templates * num_patches_per_template, dim)
    
#     # Test full head
#     print("\n=== Testing Full Classification Head ===")
#     head_full = build_classification_head(
#         head_type="full",
#         dim=dim,
#         proj_dim=512,
#         use_mlp=True,
#         temperature=1.0,
#     )
    
#     logits_full = head_full(query_features, template_features, num_templates)
#     print(f"Query: {query_features.shape}")
#     print(f"Templates: {template_features.shape}")
#     print(f"Logits: {logits_full.shape}")  # [B, S, Nq, Nt+1]
    
#     # Get predictions
#     predictions, confidences = head_full.get_predictions(logits_full)
#     print(f"Predictions: {predictions.shape}")  # [B, S, Nq]
#     print(f"Confidences: {confidences.shape}")  # [B, S, Nq]
#     print(f"Max prediction index: {predictions.max().item()} (should be < {num_patches_per_template + 1})")
    
#     # Get unseen mask
#     unseen_mask = head_full.get_unseen_mask(logits_full, threshold=0.5)
#     print(f"Unseen mask: {unseen_mask.shape}, Num unseen: {unseen_mask.sum().item()}")
    
#     # Test projection-only head
#     print("\n=== Testing Projection-Only Head ===")
#     head_proj = build_classification_head(
#         head_type="projection_only",
#         dim=dim,
#         proj_dim=512,
#         temperature=1.0,
#     )
    
#     logits_proj = head_proj(query_features, template_features, num_templates)
#     print(f"Logits: {logits_proj.shape}")  # [B, S, Nq, Nt+1]
    
#     # Check temperature effect
#     print("\n=== Testing Temperature Effect ===")
#     head_temp = ClassificationHead(dim=dim, temperature=0.1)  # Lower temp = sharper
#     logits_sharp = head_temp(query_features, template_features, num_templates)
    
#     probs_normal = F.softmax(logits_full, dim=-1)
#     probs_sharp = F.softmax(logits_sharp, dim=-1)
    
#     print(f"Normal temp entropy: {-(probs_normal * (probs_normal + 1e-10).log()).sum(-1).mean().item():.4f}")
#     print(f"Sharp temp entropy: {-(probs_sharp * (probs_sharp + 1e-10).log()).sum(-1).mean().item():.4f}")
    
#     print("\n✓ Classification head test passed!")
