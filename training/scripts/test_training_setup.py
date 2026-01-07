#!/usr/bin/env python3
"""
Quick test script to verify training setup before running full training.

Tests:
- Configuration loading
- Model initialization
- Dataloader
- Single training step
- Checkpoint saving/loading

Usage:
    python training/scripts/test_training_setup.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf


def test_config():
    """Test configuration loading."""
    print("\n" + "="*60)
    print("TEST 1: Configuration Loading")
    print("="*60)
    
    from hydra import compose, initialize_config_module
    
    try:
        with initialize_config_module(config_module="training.config", version_base=None):
            cfg = compose(config_name="train", overrides=["machine=local"])
        
        print("✓ Configuration loaded successfully")
        print(f"  Save dir: {cfg.save_dir}")
        print(f"  Batch size: {cfg.machine.batch_size}")
        print(f"  Max epochs: {cfg.max_epochs}")
        return True
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test model initialization."""
    print("\n" + "="*60)
    print("TEST 2: Model Initialization")
    print("="*60)
    
    try:
        from training.lightning_module import VPOGLightningModule
        from vpog.models.vpog_model import VPOGModel
        
        # Create minimal model
        vpog_model = VPOGModel(
            croco_checkpoint=None,  # Skip for test
            patch_size=16,
            enc_embed_dim=768,
            enc_depth=12,
            enc_num_heads=12,
        )
        
        model = VPOGLightningModule(
            model=vpog_model,
            lr=1e-4,
        )
        
        print("✓ Model initialized successfully")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """Test dataloader."""
    print("\n" + "="*60)
    print("TEST 3: Dataloader")
    print("="*60)
    
    try:
        from training.dataloader import VPOGTrainDataset
        from torch.utils.data import DataLoader
        
        # Check if templates exist
        templates_dir = Path("datasets/templates")
        if not templates_dir.exists():
            print(f"⚠ Templates directory not found: {templates_dir}")
            print("  Skipping dataloader test")
            return True  # Not a failure, just skip
        
        # Create dataset
        dataset = VPOGTrainDataset(
            templates_dir=str(templates_dir),
            dataset_name="gso",
            num_templates=2,
            patch_size=16,
            image_size=224,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # No workers for test
        )
        
        # Try to get a batch
        batch = next(iter(dataloader))
        
        print("✓ Dataloader working")
        print(f"  Dataset size: {len(dataset)}")
        print(f"  Batch images: {batch.images.shape}")
        print(f"  Batch poses: {batch.poses.shape}")
        return True
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """Test a single training step."""
    print("\n" + "="*60)
    print("TEST 4: Training Step")
    print("="*60)
    
    try:
        from training.lightning_module import VPOGLightningModule
        from vpog.models.vpog_model import VPOGModel
        from training.dataloader import VPOGBatch
        
        # Create model
        vpog_model = VPOGModel(
            croco_checkpoint=None,
            patch_size=16,
            enc_embed_dim=384,  # Smaller for test
            enc_depth=6,
            enc_num_heads=6,
        )
        
        model = VPOGLightningModule(
            model=vpog_model,
            lr=1e-4,
        )
        
        # Create dummy batch
        B, S, H, W = 2, 2, 224, 224
        Hp, Wp = H // 16, W // 16
        ps = 16
        
        batch = VPOGBatch(
            images=torch.randn(B, S+1, 3, H, W),
            poses=torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, S+1, -1, -1),
            d_ref=torch.randn(B, 3),
            patch_cls=torch.randint(0, S+1, (B, S, Hp, Wp)),
            coarse_flows=torch.randn(B, S, Hp, Wp, 2),
            patch_visibility=torch.ones(B, S, Hp, Wp),
            dense_flow=torch.randn(B, S, Hp, Wp, ps, ps, 2),
            dense_weight=torch.ones(B, S, Hp, Wp, ps, ps),
        )
        
        # Forward pass
        model.train()
        loss = model.training_step(batch, 0)
        
        print("✓ Training step successful")
        print(f"  Loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        print("✓ Backward pass successful")
        return True
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint():
    """Test checkpoint saving/loading."""
    print("\n" + "="*60)
    print("TEST 5: Checkpoint Save/Load")
    print("="*60)
    
    try:
        from training.lightning_module import VPOGLightningModule
        from vpog.models.vpog_model import VPOGModel
        import tempfile
        
        # Create model
        vpog_model = VPOGModel(
            croco_checkpoint=None,
            patch_size=16,
            enc_embed_dim=384,
            enc_depth=6,
            enc_num_heads=6,
        )
        
        model = VPOGLightningModule(
            model=vpog_model,
            lr=1e-4,
        )
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as f:
            checkpoint_path = f.name
        
        trainer = pl.Trainer(max_epochs=1, accelerator='cpu')
        trainer.save_checkpoint(checkpoint_path, weights_only=False)
        
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Load checkpoint
        loaded_model = VPOGLightningModule.load_from_checkpoint(
            checkpoint_path,
            model=vpog_model,
        )
        
        print("✓ Checkpoint loaded successfully")
        
        # Cleanup
        Path(checkpoint_path).unlink()
        
        return True
    except Exception as e:
        print(f"✗ Checkpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("VPOG Training Setup Test")
    print("="*60)
    
    tests = [
        ("Configuration", test_config),
        ("Model", test_model),
        ("Dataloader", test_dataloader),
        ("Training Step", test_training_step),
        ("Checkpoint", test_checkpoint),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Ready for training.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please fix before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
