#!/usr/bin/env python3
"""
Diagnostic script to analyze autoencoder reconstruction quality.
Checks data preprocessing, architecture, and visualizes reconstruction issues.
"""
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SpectrogramDataset(Dataset):
    def __init__(self, mat_files, transform=None):
        self.files = mat_files
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            mat = loadmat(self.files[idx])
            image = mat.get('SNR_gram', None)
            if image is None:
                raise ValueError("SNR_gram not found")
            
            if self.transform:
                image = self.transform(image)
            else:
                image = torch.from_numpy(image).float().unsqueeze(0)
            return image, self.files[idx]
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            # Return zeros as fallback
            return torch.zeros(1, 121, 104), self.files[idx]

def analyze_data_distribution(dataset, n_samples=10):
    """Analyze the distribution of input data."""
    print("=== DATA DISTRIBUTION ANALYSIS ===")
    
    values = []
    for i in range(min(n_samples, len(dataset))):
        img, filepath = dataset[i]
        img_np = img.squeeze().numpy()
        values.extend(img_np.flatten())
        
        print(f"Sample {i+1}: {os.path.basename(filepath)}")
        print(f"  Shape: {img_np.shape}")
        print(f"  Range: [{img_np.min():.3f}, {img_np.max():.3f}]")
        print(f"  Mean: {img_np.mean():.3f}, Std: {img_np.std():.3f}")
    
    values = np.array(values)
    print(f"\nOverall distribution:")
    print(f"  Range: [{values.min():.3f}, {values.max():.3f}]")
    print(f"  Mean: {values.mean():.3f}, Std: {values.std():.3f}")
    print(f"  Percentiles [5%, 25%, 50%, 75%, 95%]: {np.percentile(values, [5, 25, 50, 75, 95])}")
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(values, bins=50, alpha=0.7)
    plt.title('Data Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')
    
    plt.subplot(1, 3, 2)
    plt.hist(values, bins=50, alpha=0.7, log=True)
    plt.title('Data Distribution (log scale)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Count (log)')
    
    plt.subplot(1, 3, 3)
    sample_img = dataset[0][0].squeeze().numpy()
    plt.imshow(sample_img, cmap='viridis', origin='lower')
    plt.title('Sample Spectrogram')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('data_analysis.png', dpi=150)
    plt.show()

def build_autoencoder(nrow, ncol, n_channels=16, latent_dim=16):
    """Build the autoencoder architecture."""
    nrow_reduced = int(nrow / 8)
    ncol_reduced = int(ncol / 8)
    nel_reduced = nrow_reduced * ncol_reduced * n_channels
    
    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            # Encoder
            self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
            self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
            self.conv3 = nn.Conv2d(8, n_channels, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            
            # Latent space
            self.fc1 = nn.Linear(nel_reduced, latent_dim)
            self.fc2 = nn.Linear(latent_dim, nel_reduced)
            
            # Decoder
            self.t_conv1 = nn.ConvTranspose2d(n_channels, 8, 2, stride=2)
            self.t_conv2 = nn.ConvTranspose2d(8, 4, 2, stride=2)
            # Fix: use output_padding to match input dimensions exactly
            self.t_conv3 = nn.ConvTranspose2d(4, 1, 2, stride=2, output_padding=(1, 0))
        
        def forward(self, x):
            # Encoder
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.relu(self.conv3(x))
            x = self.pool(x)
            
            # Flatten and encode
            x = x.view(-1, nel_reduced)
            latent = torch.relu(self.fc1(x))
            
            # Decode
            x = torch.relu(self.fc2(latent))
            x = x.view(-1, n_channels, nrow_reduced, ncol_reduced)
            
            # Decoder
            x = torch.relu(self.t_conv1(x))
            x = torch.relu(self.t_conv2(x))
            output = torch.sigmoid(self.t_conv3(x))  # Sigmoid may be problematic!
            
            return output, latent
    
    return Autoencoder(), nel_reduced

def test_architecture(nrow, ncol):
    """Test if the autoencoder architecture preserves dimensions."""
    print(f"\n=== ARCHITECTURE TEST ===")
    print(f"Input dimensions: {nrow} x {ncol}")
    
    autoencoder, nel_reduced = build_autoencoder(nrow, ncol)
    autoencoder.eval()
    
    # Test with dummy input
    dummy_input = torch.randn(1, 1, nrow, ncol)
    print(f"Dummy input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        try:
            output, latent = autoencoder(dummy_input)
            print(f"Output shape: {output.shape}")
            print(f"Latent shape: {latent.shape}")
            
            if output.shape == dummy_input.shape:
                print("✓ Architecture preserves dimensions correctly")
                return True
            else:
                print(f"✗ Dimension mismatch: {output.shape} != {dummy_input.shape}")
                return False
        except Exception as e:
            print(f"✗ Architecture test failed: {e}")
            return False

def analyze_reconstruction_quality(dataset, model_path, device, n_samples=5):
    """Analyze reconstruction quality with a trained model."""
    print(f"\n=== RECONSTRUCTION QUALITY ANALYSIS ===")
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    # Load model
    sample_img, _ = dataset[0]
    nrow, ncol = sample_img.shape[1], sample_img.shape[2]
    autoencoder, _ = build_autoencoder(nrow, ncol)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        autoencoder.load_state_dict(state_dict)
        autoencoder = autoencoder.to(device).eval()
        print(f"✓ Loaded model from {model_path}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Test reconstruction on sample images
    fig, axes = plt.subplots(3, n_samples, figsize=(2*n_samples, 6))
    
    with torch.no_grad():
        for i in range(n_samples):
            img, filepath = dataset[i]
            img_batch = img.unsqueeze(0).to(device)
            
            recon, latent = autoencoder(img_batch)
            
            # Move to CPU for plotting
            original = img.squeeze().cpu().numpy()
            reconstructed = recon.squeeze().cpu().numpy()
            diff = np.abs(original - reconstructed)
            
            # Calculate metrics
            mse = np.mean((original - reconstructed)**2)
            mae = np.mean(np.abs(original - reconstructed))
            
            # Plot
            axes[0, i].imshow(original, cmap='viridis', origin='lower')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(reconstructed, cmap='viridis', origin='lower')
            axes[1, i].set_title(f'Recon (MSE:{mse:.3f})')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(diff, cmap='hot', origin='lower')
            axes[2, i].set_title(f'Diff (MAE:{mae:.3f})')
            axes[2, i].axis('off')
            
            print(f"Sample {i+1}: MSE={mse:.4f}, MAE={mae:.4f}")
            print(f"  Original range: [{original.min():.3f}, {original.max():.3f}]")
            print(f"  Recon range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    
    plt.tight_layout()
    plt.savefig('reconstruction_analysis.png', dpi=150)
    plt.show()

def main():
    # Setup paths
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(repo_root, "Spectrogram_Image_Database.dir", "Unsupervised_images.dir")
    
    # Find dataset - try common patterns
    possible_dirs = [
        data_dir,
        os.path.join(repo_root, "Unsupervised_database_With_Airguns.dir"),
        "."
    ]
    
    mat_files = []
    for d in possible_dirs:
        if os.path.isdir(d):
            files = glob.glob(os.path.join(d, "**", "*.mat"), recursive=True)
            if files:
                mat_files = sorted(files)
                data_dir = d
                break
    
    if not mat_files:
        print("No .mat files found. Please check data directory.")
        return
    
    print(f"Found {len(mat_files)} .mat files in {data_dir}")
    
    # Create dataset
    transform = transforms.ToTensor()
    dataset = SpectrogramDataset(mat_files[:100], transform=None)  # Limit for speed
    
    # Get dimensions from first valid sample
    sample_img, _ = dataset[0]
    nrow, ncol = sample_img.shape[1], sample_img.shape[2]
    
    # Run analyses
    analyze_data_distribution(dataset)
    architecture_ok = test_architecture(nrow, ncol)
    
    # Look for trained model
    model_patterns = [
        "conv_autoencoder.pth",
        "*_model.pth",
        "model.pth"
    ]
    
    model_path = None
    for pattern in model_patterns:
        matches = glob.glob(pattern)
        if matches:
            model_path = matches[0]
            break
    
    if model_path and architecture_ok:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        analyze_reconstruction_quality(dataset, model_path, device)
    else:
        print("\nSkipping reconstruction analysis (no model found or architecture issues)")
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. Check if data preprocessing is appropriate (range, normalization)")
    print("2. Consider removing sigmoid activation in decoder for unbounded outputs")
    print("3. Try different loss functions (L1 vs L2)")
    print("4. Increase model capacity or latent dimension")
    print("5. Adjust learning rate and training epochs")

if __name__ == "__main__":
    main()