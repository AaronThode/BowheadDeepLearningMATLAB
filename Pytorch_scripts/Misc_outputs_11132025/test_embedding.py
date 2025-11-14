#!/usr/bin/env python3
"""
Test TensorBoard embedding with synthetic data to isolate the shape issue.
"""
import os
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Create synthetic latent vectors and label images with different shapes
latents = np.random.randn(100, 16)  # 100 samples, 16-dim latents

# Simulate the problematic case: images with different shapes
images = []
# Image 1: 2392x2392x3
img1 = torch.randn(3, 2392, 2392)
# Image 2: 2662x2392x3  
img2 = torch.randn(3, 2662, 2392)
# Image 3: different size again
img3 = torch.randn(3, 1500, 1800)

images = [img1, img2, img3]

# Test the robust normalization approach
def normalize_images_for_embedding(images, target_size=(64, 64)):
    """Normalize all images to same size and format for TensorBoard embedding."""
    normalized = []
    target_H, target_W = target_size
    
    for img in images:
        # Ensure float tensor
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(np.asarray(img)).float()
        else:
            img = img.float()
        
        # Ensure CHW format
        if img.ndim == 2:
            img = img.unsqueeze(0)  # Add channel dim -> [1,H,W]
        elif img.ndim == 3:
            pass  # Already [C,H,W]
        else:
            img = img.view(-1, img.size(-2), img.size(-1))  # Force to CHW
        
        print(f"Original shape: {img.shape}")
        
        # Force to 3 channels for RGB thumbnails
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)  # [1,H,W] -> [3,H,W]
        elif img.size(0) > 3:
            img = img[:3]  # Take first 3 channels
        elif img.size(0) == 2:
            img = torch.cat([img, img[:1]], dim=0)  # Duplicate first channel to get 3
        
        # Resize to fixed target size using interpolation
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), 
            size=(target_H, target_W), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # Normalize to [0,1] range for thumbnails
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        print(f"Normalized shape: {img.shape}")
        normalized.append(img.unsqueeze(0))  # Add batch dim -> [1,3,64,64]
    
    return normalized

# Test the function
try:
    normalized_images = normalize_images_for_embedding(images)
    
    # Try concatenation
    label_tensor = torch.cat(normalized_images, dim=0)
    print(f"Successfully concatenated to shape: {label_tensor.shape}")
    
    # Test with TensorBoard
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(repo_root, 'runs', f'embedding_test_{time.strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(run_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=run_dir)
    
    # Sample matching number of latents
    sample_latents = latents[:len(normalized_images)]
    
    writer.add_embedding(
        torch.from_numpy(sample_latents).float(),
        label_img=label_tensor,
        global_step=0,
        tag='Test/Embedding'
    )
    writer.flush()
    writer.close()
    
    print(f"✓ TensorBoard embedding test successful! Run dir: {run_dir}")
    
except Exception as e:
    print(f"✗ TensorBoard embedding test failed: {e}")
    import traceback
    traceback.print_exc()