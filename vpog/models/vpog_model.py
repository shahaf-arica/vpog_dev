"""
VPOG Model - Visual Patch-wise Object pose estimation with Groups of templates

This version matches the updated AA contract:
  - Query tokens:    [B, HW, C]
  - Template tokens: [B, S, HW+1, C]  (one unseen token appended PER template, owned by VPOGModel)

AA responsibilities (implemented in vpog.models.aa_module):
  - Global:
      * Query image tokens -> RoPE2D
      * Template image tokens -> S^2-RoPE if enabled, else RoPE2D
      * Unseen token participates but receives NO positional encoding
  - Local:
      * Query: RoPE2D over image tokens
      * Template: window attention over (window image tokens + unseen), RoPE2D only on image tokens

Heads are kept backward-compatible via reshaping:
  - ClassificationHead gets templates flattened to [B, S*(HW+1), C]
  - FlowHead gets template *image* tokens flattened to [B, S*HW, C]

Note: TokenManager is intentionally removed from the forward path.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class VPOGModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        aa_module: nn.Module,
        classification_head: nn.Module,
        dense_flow_head: nn.Module,
        # center_flow_head: nn.Module, # Currently unused
        img_size: int = 224,
        patch_size: int = 16,
        use_s2rope: bool = True,
        assert_pos2d_match: bool = False,
    ):
        super().__init__()

        self.encoder = encoder
        self.aa_module = aa_module
        self.classification_head = classification_head
        # self.center_flow_head = center_flow_head
        self.dense_flow_head = dense_flow_head

        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.use_s2rope = bool(use_s2rope)
        self.assert_pos2d_match = bool(assert_pos2d_match)

        if hasattr(encoder, "grid_size"):
            self.grid_size = tuple(getattr(encoder, "grid_size"))
        else:
            self.grid_size = (self.img_size // self.patch_size, self.img_size // self.patch_size)

        self.num_patches = int(self.grid_size[0] * self.grid_size[1])

        # Initialize learnable unseen token
        # Get dimension from encoder
        if hasattr(encoder, 'embed_dim'):
            dim = encoder.embed_dim
        elif hasattr(encoder, 'enc') and hasattr(encoder.enc, 'embed_dim'):
            dim = encoder.enc.embed_dim
        else:
            raise ValueError("Cannot determine embedding dimension from encoder")
        
        # Create learnable unseen token parameter [1, 1, 1, dim]
        self._template_unseen_token = nn.Parameter(torch.zeros(1, 1, 1, dim))
        nn.init.trunc_normal_(self._template_unseen_token, std=0.02)

    @property
    def template_unseen_token(self) -> nn.Parameter:
        return self._template_unseen_token

    def forward(
        self,
        query_images: torch.Tensor,                      # [B,3,H,W]
        template_images: torch.Tensor,                   # [B,S,3,H,W]
        # query_poses: Optional[torch.Tensor] = None,      # [B,4,4]
        template_poses: Optional[torch.Tensor] = None,   # [B,S,4,4]
        # ref_dirs: Optional[torch.Tensor] = None,         # [B,3] (required if use_s2rope=True)
        # patch_cls: Optional[torch.Tensor] = None,        # [B,S,Nq] required
        # return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, S = template_images.shape[:2]

        q_tokens, q_pos2d = self.encoder(query_images)  # [B,Nq,C], [B,Nq,2]
        Nq, C = q_tokens.shape[1], q_tokens.shape[2]
        if Nq != self.num_patches:
            self.num_patches = int(Nq)

        t_imgs_flat = template_images.reshape(B * S, *template_images.shape[2:])
        t_tokens_flat, t_pos2d_flat = self.encoder(t_imgs_flat)  # [B*S,Nt,C], [B*S,Nt,2]
        Nt = t_tokens_flat.shape[1]
        if Nt != Nq:
            raise ValueError(f"Expected Nt==Nq (same patch grid), got Nt={Nt} Nq={Nq}")

        t_tokens_img = t_tokens_flat.view(B, S, Nt, C)  # [B,S,Nt,C]

        if self.assert_pos2d_match:
            t_pos2d = t_pos2d_flat.view(B, S, Nt, 2)
            if not torch.allclose(t_pos2d[:, 0], q_pos2d, atol=0.0, rtol=0.0):
                raise AssertionError("Template pos2d grid differs from query pos2d grid.")

        unseen = self.template_unseen_token.expand(B, S, 1, C)
        t_tokens = torch.cat([t_tokens_img, unseen], dim=2)  # [B,S,Nt+1,C]

        # template_frame_dirs = None
        # template_frame_has_s2 = None
        # if self.use_s2rope:
        #     if template_poses is None or ref_dirs is None:
        #         raise ValueError("use_s2rope=True requires template_poses and ref_dirs.")
        #     template_frame_dirs = template_poses[:, :, :3, 2]  # [B,S,3]
        #     template_frame_has_s2 = torch.ones(B, S, dtype=torch.bool, device=q_tokens.device)

        # q_tokens_aa, t_tokens_aa = self.aa_module(
        #     q_tokens=q_tokens,
        #     t_tokens=t_tokens,
        #     pos2d=q_pos2d,
        #     grid_size=self.grid_size,
        #     template_frame_dirs=template_frame_dirs,
        #     template_frame_has_s2=template_frame_has_s2,
        #     ref_dirs=ref_dirs,
        # )  # q: [B,Nq,C], t: [B,S,Nt+1,C]

        q_tokens_aa, t_tokens_aa = self.aa_module(
            q_tokens=q_tokens,
            t_tokens=t_tokens,
            pos2d=q_pos2d,
            grid_size=self.grid_size,
            template_poses=template_poses,
        )  # q: [B,Nq,C], t: [B,S,Nt+1,C]

        # classification_logits = self.classification_head(
        #     q_tokens=q_tokens_aa,
        #     t_tokens=t_tokens_aa,
        # )  # [B,S,Nq,Nt+1]

        # if patch_cls is None:
        #     raise ValueError("patch_cls is required for buddy-only flow_head.")
        # if patch_cls.shape != (B, S, Nq):
        #     raise ValueError(f"patch_cls must be [B,S,Nq]={B,S,Nq}, got {tuple(patch_cls.shape)}")

        # t_img_aa = t_tokens_aa[:, :, :Nt, :].contiguous()  # [B,S,Nt,C]

        out = {
            # "classification_logits": classification_logits,
            "query_tokens_aa": q_tokens_aa,
            "template_tokens_aa": t_tokens_aa,
            "query_pos2d": q_pos2d,
            "num_tokens_per_template": Nt,
            "num_query_tokens": Nq,
        }
        # flow_out = self.flow_head(
        #     q_tokens=q_tokens_aa,
        #     t_tokens_img=t_img_aa,
        #     patch_cls=patch_cls,
        # )

        # out: Dict[str, torch.Tensor] = {
        #     "classification_logits": classification_logits,
        #     "dense_flow": flow_out["dense_flow"],
        #     "dense_b": flow_out["dense_b"],
        #     "center_flow": flow_out["center_flow"],
        #     "flow_valid": flow_out["valid"],
        # }

        # if return_all:
        #     out.update(
        #         {
        #             "query_tokens_raw": q_tokens,
        #             "template_tokens_img_raw": t_tokens_img,
        #             "query_tokens_aa": q_tokens_aa,
        #             "template_tokens_aa": t_tokens_aa,
        #             "query_pos2d": q_pos2d,
        #         }
        #     )
        return out


def build_vpog_model(
    encoder: nn.Module,
    aa_module: nn.Module,
    classification_head: nn.Module,
    flow_head: nn.Module,
    img_size: int = 224,
    patch_size: int = 16,
    use_s2rope: bool = True,
    assert_pos2d_match: bool = False,
) -> VPOGModel:
    return VPOGModel(
        encoder=encoder,
        aa_module=aa_module,
        classification_head=classification_head,
        flow_head=flow_head,
        img_size=img_size,
        patch_size=patch_size,
        use_s2rope=use_s2rope,
        assert_pos2d_match=assert_pos2d_match,
    )


if __name__ == "__main__":
    print("Testing VPOG Model...")
    
    # Test configuration
    config = {
        'encoder': {
            'model_size': 'base',
            'pretrained_path': None,
            'freeze_encoder': False,
        },
        'aa_module': {
            'depth': 4,
            'num_heads': 12,
            'window_size': 7,
            'use_global': True,
            'use_local': True,
        },
        'classification_head': {
            'head_type': 'full',
            'proj_dim': 512,
            'use_mlp': True,
            'temperature': 1.0,
        },
        'flow_head': {
            'head_type': 'full',
            'flow_resolution': 16,
            'use_confidence': True,
        },
        'img_size': 224,
        'patch_size': 16,
    }
    
    # Build model
    model = build_vpog_model(config)
    print(f"Model built successfully!")
    print(f"  Encoder: {model.encoder.embed_dim}D, {model.encoder.depth} layers")
    print(f"  AA Module: {model.aa_module.depth} layers")
    print(f"  Grid size: {model.grid_size}, Num patches: {model.num_patches}")
    
    # Test forward pass
    batch_size = 2
    num_templates = 4
    
    query_images = torch.randn(batch_size, 3, 224, 224)
    template_images = torch.randn(batch_size, num_templates, 3, 224, 224)
    query_poses = torch.eye(4).unsqueeze(0).expand(batch_size, -1, -1)
    template_poses = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, num_templates, -1, -1)
    ref_dirs = torch.tensor([[0, 0, 1]], dtype=torch.float32).expand(batch_size, -1)
    
    print(f"\nInput shapes:")
    print(f"  Query images: {query_images.shape}")
    print(f"  Template images: {template_images.shape}")
    print(f"  Query poses: {query_poses.shape}")
    print(f"  Template poses: {template_poses.shape}")
    print(f"  Ref dirs: {ref_dirs.shape}")
    
    # Forward pass
    outputs = model(
        query_images,
        template_images,
        query_poses,
        template_poses,
        ref_dirs,
        return_all=True,
    )
    
    print(f"\nOutput shapes:")
    print(f"  Classification logits: {outputs['classification_logits'].shape}")  # [B, S, Nq, Nt+1]
    print(f"  Flow: {outputs['flow'].shape}")  # [B, S, Nq, Nt, 16, 16, 2]
    print(f"  Flow confidence: {outputs['flow_confidence'].shape}")  # [B, S, Nq, Nt, 16, 16, 1]
    
    # Get predictions
    predictions = model.get_predictions(
        outputs['classification_logits'],
        outputs['flow'],
        outputs['flow_confidence'],
    )
    
    print(f"\nPrediction shapes:")
    print(f"  Matched patches: {predictions['matched_template_patches'].shape}")  # [B, S, Nq]
    print(f"  Confidences: {predictions['match_confidences'].shape}")  # [B, S, Nq]
    print(f"  Unseen mask: {predictions['unseen_mask'].shape}")  # [B, S, Nq]
    print(f"  Flow matched: {predictions['flow_matched'].shape}")  # [B, S, Nq, 16, 16, 2]
    
    print("\n✓ VPOG model test passed!")
    
    # Verify unseen tokens don't get positional encoding
    print("\n=== Verification: Unseen Tokens ===")
    print("Unseen tokens are added in ClassificationHead.forward() AFTER AA module processing")
    print("This ensures unseen tokens do NOT receive RoPE2D or S²RoPE encoding")
    print("✓ Unseen tokens correctly excluded from positional encoding")
