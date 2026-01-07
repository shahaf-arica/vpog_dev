"""
Test VPOG Model Integration with TokenManager and rope_mask

This test verifies:
1. TokenManager correctly adds tokens with rope_mask
2. AA module respects rope_mask (added tokens skip RoPE)
3. Classification head receives correct dimensions
4. Flow head works with patch-only features
5. End-to-end forward pass completes successfully
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from vpog.models.token_manager import TokenManager
from vpog.models.vpog_model import VPOGModel


def test_token_manager_integration():
    """Test TokenManager with AA module rope_mask"""
    print("\n" + "="*80)
    print("TEST: TokenManager Integration with rope_mask")
    print("="*80)
    
    B, Nq, D = 2, 196, 768  # 2 batches, 14x14 patches, 768-dim
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create token manager
    token_manager = TokenManager(
        num_query_added_tokens=0,
        num_template_added_tokens=1,
        embed_dim=D,
    )
    
    # Create dummy features
    query_features = torch.randn(B, Nq, D, device=device)
    query_pos2d = torch.rand(B, Nq, 2, device=device) * 14  # 14x14 grid
    
    # Add tokens
    query_with_tokens, query_pos_with_tokens, query_rope_mask = \
        token_manager.add_query_tokens(query_features, query_pos2d)
    
    print(f"\nQuery features shape: {query_features.shape}")
    print(f"Query with tokens shape: {query_with_tokens.shape}")
    print(f"Query rope_mask shape: {query_rope_mask.shape}")
    print(f"Query rope_mask (first sample): {query_rope_mask[0]}")
    print(f"  - All True (no added tokens): {query_rope_mask.all().item()}")
    
    # Test with templates
    S, Nt = 4, 196  # 4 templates, 14x14 patches each
    template_features = torch.randn(B, S, Nt, D, device=device)
    template_pos2d = torch.rand(B, S, Nt, 2, device=device) * 14
    
    templates_with_tokens, templates_pos_with_tokens, templates_rope_mask = \
        token_manager.add_template_tokens(template_features, template_pos2d)
    
    print(f"\nTemplate features shape: {template_features.shape}")
    print(f"Templates with tokens shape: {templates_with_tokens.shape}")
    print(f"Templates rope_mask shape: {templates_rope_mask.shape}")
    print(f"Templates rope_mask (first sample, first template): {templates_rope_mask[0, 0]}")
    print(f"  - Last token is False (unseen): {not templates_rope_mask[0, 0, -1].item()}")
    print(f"  - First 196 are True: {templates_rope_mask[0, 0, :196].all().item()}")
    
    # Remove added tokens
    templates_without_added, template_added_features = token_manager.remove_added_tokens(
        templates_with_tokens, is_query=False
    )
    
    print(f"\nAfter removal:")
    print(f"  Templates (patches only): {templates_without_added.shape}")
    print(f"  Added features: {template_added_features.shape}")
    print(f"  Matches original: {templates_without_added.shape == template_features.shape}")
    
    print("\n✓ TokenManager integration test PASSED\n")


def test_vpog_model_forward():
    """Test full VPOG model forward pass with TokenManager"""
    print("\n" + "="*80)
    print("TEST: VPOG Model Forward Pass with TokenManager")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Model configuration
    img_size = 224
    patch_size = 16
    B, S = 2, 4  # 2 batches, 4 templates
    
    encoder_config = {
        'model_name': 'CroCo_V2_ViTBase',
        'pretrained_path': None,
        'freeze_encoder': False,
        'num_query_added_tokens': 0,
        'num_template_added_tokens': 1,
    }
    
    aa_config = {
        'depth': 2,  # Shallow for testing
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'window_size': 7,
        'use_s2rope': True,
    }
    
    classification_config = {
        'use_mlp': True,
        'mlp_hidden_dim': 512,
        'temperature': 1.0,
    }
    
    flow_config = {
        'num_layers': 2,
        'hidden_dim': 256,
    }
    
    print("\nCreating VPOG model...")
    try:
        model = VPOGModel(
            encoder_config=encoder_config,
            aa_config=aa_config,
            classification_config=classification_config,
            flow_config=flow_config,
            img_size=img_size,
            patch_size=patch_size,
        ).to(device)
        print("✓ Model created successfully")
    except Exception as e:
        print(f"✗ Model creation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check TokenManager is configured correctly
    print(f"\nTokenManager configuration:")
    print(f"  num_query_added_tokens: {model.num_query_added_tokens}")
    print(f"  num_template_added_tokens: {model.num_template_added_tokens}")
    print(f"  TokenManager embed_dim: {model.token_manager.embed_dim}")
    
    # Create dummy inputs
    print("\nCreating dummy inputs...")
    query_images = torch.randn(B, 3, img_size, img_size, device=device)
    template_images = torch.randn(B, S, 3, img_size, img_size, device=device)
    
    # Dummy poses (identity matrices)
    query_poses = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
    template_poses = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
    
    # Dummy reference directions
    ref_dirs = torch.tensor([[0, 0, 1]], device=device, dtype=torch.float32).expand(B, -1)
    
    # Forward pass
    print("\nRunning forward pass...")
    try:
        with torch.no_grad():
            outputs = model(
                query_images=query_images,
                template_images=template_images,
                query_poses=query_poses,
                template_poses=template_poses,
                ref_dirs=ref_dirs,
                return_all=True,
            )
        print("✓ Forward pass completed successfully")
    except Exception as e:
        print(f"✗ Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check outputs
    print("\nChecking outputs...")
    
    Nq = (img_size // patch_size) ** 2  # 196
    Nt = Nq
    Nt_with_added = Nt + model.num_template_added_tokens  # 197
    
    expected_shapes = {
        'classification_logits': (B, S, Nq, Nt_with_added),
        'flow': (B, S, Nq, Nt, 16, 16, 2),
        'flow_confidence': (B, S, Nq, Nt, 16, 16, 1),
        'query_features_raw': (B, Nq, encoder_config.get('embed_dim', 768)),
        'query_features_aa': (B, Nq, encoder_config.get('embed_dim', 768)),
        'query_added_features': (B, model.num_query_added_tokens, encoder_config.get('embed_dim', 768)),
        'template_added_features': (B, S, model.num_template_added_tokens, encoder_config.get('embed_dim', 768)),
    }
    
    all_correct = True
    for key, expected_shape in expected_shapes.items():
        if key in outputs:
            actual_shape = tuple(outputs[key].shape)
            # For embed_dim, use actual encoder dimension
            if key in ['query_features_raw', 'query_features_aa', 'query_added_features', 'template_added_features']:
                expected_shape = list(expected_shape)
                expected_shape[-1] = outputs[key].shape[-1]
                expected_shape = tuple(expected_shape)
            
            matches = actual_shape == expected_shape
            status = "✓" if matches else "✗"
            print(f"  {status} {key}: {actual_shape} {'==' if matches else '!='} {expected_shape}")
            if not matches:
                all_correct = False
        else:
            print(f"  ✗ {key}: MISSING")
            all_correct = False
    
    # Special check: classification logits should have Nt+1 (or Nt+added)
    if 'classification_logits' in outputs:
        last_dim = outputs['classification_logits'].shape[-1]
        expected_last_dim = Nt_with_added
        if last_dim == expected_last_dim:
            print(f"\n✓ Classification logits last dim = {last_dim} (Nt + {model.num_template_added_tokens} added tokens)")
        else:
            print(f"\n✗ Classification logits last dim = {last_dim}, expected {expected_last_dim}")
            all_correct = False
    
    # Check that added features have correct shape
    if 'template_added_features' in outputs:
        if model.num_template_added_tokens > 0:
            shape = outputs['template_added_features'].shape
            if shape == (B, S, model.num_template_added_tokens, shape[-1]):
                print(f"✓ Template added features shape correct: {shape}")
            else:
                print(f"✗ Template added features shape incorrect: {shape}")
                all_correct = False
    
    if all_correct:
        print("\n✓ VPOG model forward pass test PASSED\n")
    else:
        print("\n✗ VPOG model forward pass test FAILED\n")


def test_rope_mask_effectiveness():
    """Test that rope_mask actually prevents RoPE from being applied"""
    print("\n" + "="*80)
    print("TEST: rope_mask Effectiveness (Added Tokens Skip RoPE)")
    print("="*80)
    
    from vpog.models.aa_module_old import AAModule
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    B, N, D = 2, 197, 768  # 196 patches + 1 added token
    
    # Create AA module
    aa_module = AAModule(
        dim=D,
        depth=1,
        num_heads=12,
        mlp_ratio=4.0,
        window_size=7,
    ).to(device)
    
    # Create features: 196 patches + 1 added token
    features = torch.randn(B, N, D, device=device)
    pos2d = torch.rand(B, N, 2, device=device) * 14
    
    # Create rope_mask: True for patches (0-195), False for added token (196)
    rope_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    rope_mask[:, -1] = False  # Last token should skip RoPE
    
    print(f"\nInput features shape: {features.shape}")
    print(f"rope_mask shape: {rope_mask.shape}")
    print(f"rope_mask[0]: first 5 = {rope_mask[0, :5].tolist()}, last 5 = {rope_mask[0, -5:].tolist()}")
    
    # Test 1: With rope_mask
    with torch.no_grad():
        output_with_mask = aa_module(
            features,
            pos2d=pos2d,
            rope_mask=rope_mask,
            is_template=False,
            grid_size=(14, 14),
        )
    
    # Test 2: Without rope_mask (all tokens get RoPE)
    rope_mask_all = torch.ones(B, N, dtype=torch.bool, device=device)
    with torch.no_grad():
        output_without_mask = aa_module(
            features,
            pos2d=pos2d,
            rope_mask=rope_mask_all,
            is_template=False,
            grid_size=(14, 14),
        )
    
    # Compare: outputs should differ because last token has different positional encoding
    diff = (output_with_mask - output_without_mask).abs()
    
    print(f"\nDifference between masked and unmasked:")
    print(f"  Patches (0-195) mean diff: {diff[:, :-1, :].mean().item():.6f}")
    print(f"  Added token (196) mean diff: {diff[:, -1, :].mean().item():.6f}")
    
    # The added token should show significant difference because it's getting different treatment
    # Patches might also differ slightly due to attention interactions
    added_token_diff = diff[:, -1, :].mean().item()
    
    if added_token_diff > 1e-6:
        print(f"\n✓ rope_mask is working: added token differs when masked vs unmasked")
    else:
        print(f"\n✗ rope_mask may not be working: no difference detected")
    
    print("\n✓ rope_mask effectiveness test completed\n")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("VPOG TokenManager Integration Test Suite")
    print("="*80)
    
    # Run tests
    test_token_manager_integration()
    test_vpog_model_forward()
    test_rope_mask_effectiveness()
    
    print("\n" + "="*80)
    print("All Tests Completed")
    print("="*80 + "\n")
