"""
CroCo V2 Model Loading and Testing Snippet
===========================================

This script demonstrates how to load and use the CroCo V2 model in different modes:
1. Encoder-only mode (for extracting features from a single image)
2. Full encoder+decoder mode (for cross-view completion with two images)

The script uses the pretrained checkpoints in the checkpoints/ folder.

For encoder-only usage (recommended for your use case):
- Input: Single RGB image of size (B, 3, H, W)
- Preprocessing: Resize to 224x224, normalize with ImageNet stats
- Output: Encoded features (B, N_patches, embed_dim) where N_patches = (224/16)^2 = 196

Model configurations available:
- ViT-Base encoder + Small decoder: enc_embed_dim=768, enc_depth=12
- ViT-Base encoder + Base decoder: enc_embed_dim=768, enc_depth=12  
- ViT-Large encoder + Base decoder: enc_embed_dim=1024, enc_depth=24
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Compose, Resize

# Add external/croco to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'external', 'croco'))

from models.croco import CroCoNet
from models.croco_downstream import CroCoDownstreamMonocularEncoder, croco_args_from_ckpt


# ==================== CONFIGURATION ====================
# Change these parameters for your experiments

# Model selection - choose one of the available checkpoints
MODEL_PATH = "checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth"  # Options:
# "checkpoints/CroCo_V2_ViTBase_SmallDecoder.pth"  # ViT-Base + Small decoder
# "checkpoints/CroCo_V2_ViTBase_BaseDecoder.pth"   # ViT-Base + Base decoder
# "checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth"  # ViT-Large + Base decoder (largest, best)

# Mode selection
USE_ENCODER_ONLY = True  # Set to True for encoder-only mode, False for full encoder+decoder

# Image settings
IMAGE_SIZE = 224  # CroCo expects 224x224 images
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ImageNet normalization (required for CroCo models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Test image paths (modify these to your own images)
TEST_IMAGE_1 = "external/croco/assets/Chateau1.png"  # For encoder-only or as image1 in full mode
TEST_IMAGE_2 = "external/croco/assets/Chateau2.png"  # For full encoder+decoder mode

# Debugging flags
PRINT_MODEL_INFO = True  # Print model architecture details
PRINT_FEATURE_SHAPES = True  # Print output feature shapes
VISUALIZE_OUTPUT = True  # Save visualization (only works in full encoder+decoder mode)

# ==================== END CONFIGURATION ====================


def load_and_preprocess_image(image_path, device):
    """
    Load and preprocess an image for CroCo model.
    
    Args:
        image_path: Path to the image file
        device: Device to load the tensor on
        
    Returns:
        Preprocessed image tensor of shape (1, 3, 224, 224)
    """
    # Define transform pipeline
    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Load and transform image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device, non_blocking=True)
    
    return img_tensor


def load_croco_full_model(checkpoint_path, device):
    """
    Load the full CroCo model (encoder + decoder) for cross-view completion.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Device to load the model on
        
    Returns:
        Loaded CroCo model
    """
    print(f"Loading full CroCo model from {checkpoint_path}...")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model with checkpoint config
    model = CroCoNet(**ckpt.get('croco_kwargs', {}))
    
    # Load weights
    msg = model.load_state_dict(ckpt['model'], strict=True)
    print(f"Model loaded with message: {msg}")
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    if PRINT_MODEL_INFO:
        print("\n=== Full Model Configuration ===")
        print(f"Encoder embed_dim: {model.enc_embed_dim}")
        print(f"Encoder depth: {model.enc_depth}")
        print(f"Decoder embed_dim: {model.dec_embed_dim}")
        print(f"Decoder depth: {model.dec_depth}")
        print(f"Position embedding: {model.pos_embed}")
        print(f"Patch size: {model.patch_embed.patch_size}")
        print(f"Number of patches: {model.patch_embed.num_patches}")
        print("================================\n")
    
    return model


class SimpleHead(torch.nn.Module):
    """
    A simple identity head for the encoder-only model.
    This just returns the features as-is.
    """
    def setup(self, model):
        """Called during model initialization to setup the head."""
        self.embed_dim = model.enc_embed_dim
        self.num_patches = model.patch_embed.num_patches
        
    def forward(self, features, img_info):
        """
        Args:
            features: Encoded features (B, N_patches, embed_dim)
            img_info: Dictionary with 'height' and 'width' keys
            
        Returns:
            features as-is
        """
        return features


def load_croco_encoder_only(checkpoint_path, device):
    """
    Load only the encoder part of CroCo for feature extraction.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Device to load the model on
        
    Returns:
        Encoder-only model
    """
    print(f"Loading CroCo encoder-only from {checkpoint_path}...")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model arguments from checkpoint
    croco_kwargs = croco_args_from_ckpt(ckpt)
    
    # Create encoder-only model with a simple identity head
    head = SimpleHead()
    model = CroCoDownstreamMonocularEncoder(head=head, **croco_kwargs)
    
    # Load encoder weights (decoder weights will be ignored)
    msg = model.load_state_dict(ckpt['model'], strict=False)
    print(f"Encoder loaded with message: {msg}")
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    if PRINT_MODEL_INFO:
        print("\n=== Encoder-Only Configuration ===")
        print(f"Encoder embed_dim: {model.enc_embed_dim}")
        print(f"Encoder depth: {model.enc_depth}")
        print(f"Position embedding: {model.pos_embed}")
        print(f"Patch size: {model.patch_embed.patch_size}")
        print(f"Number of patches: {model.patch_embed.num_patches}")
        print("===================================\n")
    
    return model


def test_encoder_only(model, image_path, device):
    """
    Test the encoder-only model on a single image.
    
    Args:
        model: Encoder-only model
        image_path: Path to test image
        device: Device to run inference on
    """
    print(f"\n=== Testing Encoder-Only Mode ===")
    print(f"Input image: {image_path}")
    
    # Load and preprocess image
    img_tensor = load_and_preprocess_image(image_path, device)
    print(f"Input shape: {img_tensor.shape}")
    
    # Run inference
    with torch.inference_mode():
        features = model(img_tensor)
    
    if PRINT_FEATURE_SHAPES:
        print(f"Output features shape: {features.shape}")
        print(f"  Batch size: {features.shape[0]}")
        print(f"  Number of patches: {features.shape[1]}")
        print(f"  Feature dimension: {features.shape[2]}")
    
    # Compute some statistics
    print(f"\nFeature statistics:")
    print(f"  Mean: {features.mean().item():.4f}")
    print(f"  Std: {features.std().item():.4f}")
    print(f"  Min: {features.min().item():.4f}")
    print(f"  Max: {features.max().item():.4f}")
    
    return features


def test_full_model(model, image1_path, image2_path, device):
    """
    Test the full encoder+decoder model on two images.
    
    Args:
        model: Full CroCo model
        image1_path: Path to first image (will be masked and reconstructed)
        image2_path: Path to second image (reference view)
        device: Device to run inference on
    """
    print(f"\n=== Testing Full Encoder+Decoder Mode ===")
    print(f"Image 1 (to reconstruct): {image1_path}")
    print(f"Image 2 (reference): {image2_path}")
    
    # Load and preprocess images
    img1 = load_and_preprocess_image(image1_path, device)
    img2 = load_and_preprocess_image(image2_path, device)
    print(f"Input shapes: {img1.shape}, {img2.shape}")
    
    # Run inference
    with torch.inference_mode():
        out, mask, target = model(img1, img2)
    
    if PRINT_FEATURE_SHAPES:
        print(f"\nOutput shapes:")
        print(f"  Reconstruction: {out.shape}")
        print(f"  Mask: {mask.shape}")
        print(f"  Target: {target.shape}")
        print(f"  Mask ratio: {mask.float().mean().item():.2%}")
    
    # Optionally create visualization
    if VISUALIZE_OUTPUT:
        visualize_reconstruction(model, img1, img2, out, mask, device)
    
    return out, mask, target


def visualize_reconstruction(model, img1, img2, out, mask, device):
    """
    Create a visualization of the reconstruction (similar to demo.py).
    
    Args:
        model: CroCo model
        img1: First input image tensor
        img2: Second input image tensor
        out: Model output (reconstruction)
        mask: Mask tensor
        device: Device
    """
    import torchvision
    
    print("\nCreating visualization...")
    
    # Denormalize: the output is normalized, use mean/std of actual image
    imagenet_mean_tensor = torch.tensor(IMAGENET_MEAN).view(1,3,1,1).to(device, non_blocking=True)
    imagenet_std_tensor = torch.tensor(IMAGENET_STD).view(1,3,1,1).to(device, non_blocking=True)
    
    patchified = model.patchify(img1)
    mean = patchified.mean(dim=-1, keepdim=True)
    var = patchified.var(dim=-1, keepdim=True)
    decoded_image = model.unpatchify(out * (var + 1.e-6)**.5 + mean)
    
    # Undo imagenet normalization
    decoded_image = decoded_image * imagenet_std_tensor + imagenet_mean_tensor
    input_image = img1 * imagenet_std_tensor + imagenet_mean_tensor
    ref_image = img2 * imagenet_std_tensor + imagenet_mean_tensor
    
    # Create masked input image
    image_masks = model.unpatchify(model.patchify(torch.ones_like(ref_image)) * mask[:,:,None])
    masked_input_image = ((1 - image_masks) * input_image)
    
    # Create visualization: [reference | masked_input | reconstruction | ground_truth]
    visualization = torch.cat((ref_image, masked_input_image, decoded_image, input_image), dim=3)
    B, C, H, W = visualization.shape
    visualization = visualization.permute(1, 0, 2, 3).reshape(C, B*H, W)
    visualization = torchvision.transforms.functional.to_pil_image(torch.clamp(visualization, 0, 1))
    
    output_path = "croco_output_visualization.png"
    visualization.save(output_path)
    print(f"Visualization saved to: {output_path}")
    print("  Layout: [Reference | Masked Input | Reconstruction | Ground Truth]")


def main():
    """
    Main function to run the CroCo model tests.
    """
    print("=" * 60)
    print("CroCo V2 Model Loading and Testing")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Mode: {'Encoder-Only' if USE_ENCODER_ONLY else 'Full (Encoder+Decoder)'}")
    print("=" * 60)
    
    # Check if checkpoint exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Checkpoint not found at {MODEL_PATH}")
        print("Available checkpoints in checkpoints/:")
        if os.path.exists("checkpoints"):
            for f in os.listdir("checkpoints"):
                if f.endswith(".pth"):
                    print(f"  - {f}")
        return
    
    if USE_ENCODER_ONLY:
        # Test encoder-only mode
        model = load_croco_encoder_only(MODEL_PATH, DEVICE)
        
        if not os.path.exists(TEST_IMAGE_1):
            print(f"\nWARNING: Test image not found at {TEST_IMAGE_1}")
            print("Please update TEST_IMAGE_1 in the configuration section.")
            return
            
        features = test_encoder_only(model, TEST_IMAGE_1, DEVICE)
        
        print("\n" + "=" * 60)
        print("Encoder-only test completed successfully!")
        print("=" * 60)
        print("\nHow to use the encoder in your dataloader:")
        print("1. Resize images to 224x224")
        print("2. Normalize with ImageNet mean/std:")
        print(f"   mean = {IMAGENET_MEAN}")
        print(f"   std = {IMAGENET_STD}")
        print("3. Pass tensor of shape (B, 3, 224, 224) to model")
        print(f"4. Get features of shape (B, 196, {model.enc_embed_dim})")
        print("   where 196 = (224/16)^2 is the number of patches")
        print("=" * 60)
        
    else:
        # Test full encoder+decoder mode
        model = load_croco_full_model(MODEL_PATH, DEVICE)
        
        if not os.path.exists(TEST_IMAGE_1) or not os.path.exists(TEST_IMAGE_2):
            print(f"\nWARNING: Test images not found")
            print(f"  Image 1: {TEST_IMAGE_1}")
            print(f"  Image 2: {TEST_IMAGE_2}")
            print("Please update TEST_IMAGE_1 and TEST_IMAGE_2 in the configuration section.")
            return
            
        out, mask, target = test_full_model(model, TEST_IMAGE_1, TEST_IMAGE_2, DEVICE)
        
        print("\n" + "=" * 60)
        print("Full model test completed successfully!")
        print("=" * 60)


if __name__ == "__main__":
    main()
