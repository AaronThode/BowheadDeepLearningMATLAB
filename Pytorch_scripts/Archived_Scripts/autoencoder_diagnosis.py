#!/usr/bin/env python3
"""
Improved autoencoder with better architecture and training strategies.
"""
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

class ImprovedAutoencoder(nn.Module):
    def __init__(self, nrow, ncol, latent_dim=64):
        super().__init__()
        
        # Calculate dimensions after pooling
        self.nrow, self.ncol = nrow, ncol
        self.nrow_reduced = nrow // 8
        self.ncol_reduced = ncol // 8
        
        # Encoder
        self.encoder = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
            
            # Block 2: 32 -> 64 channels  
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4
            
            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8
        )
        
        # Calculate flattened size
        self.flat_size = 128 * self.nrow_reduced * self.ncol_reduced
        
        # Latent space
        self.to_latent = nn.Sequential(
            nn.Linear(self.flat_size, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, self.flat_size),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Block 1: 128 -> 64 channels
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 2: 64 -> 32 channels
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 3: 32 -> 1 channel (output)
            nn.ConvTranspose2d(32, 1, 2, stride=2, output_padding=(nrow % 8, ncol % 8)),
            # No activation - let output be unbounded!
        )
        
        self.latent_dim = latent_dim
    
    def forward(self, x):
        # Encode
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        latent = self.to_latent(x_flat)
        
        # Decode
        x_recon = self.from_latent(latent)
        x_recon = x_recon.view(x_recon.size(0), 128, self.nrow_reduced, self.ncol_reduced)
        output = self.decoder(x_recon)
        
        return output, latent

def analyze_gsi_data():
    """Analyze the available GSI header data to understand the dataset."""
    print("=== ANALYZING AVAILABLE DATA ===")
    
    # Check GSI header table
    gsi_path = "/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/GSI_header_table.mat"
    if os.path.exists(gsi_path):
        try:
            data = loadmat(gsi_path)
            print(f"GSI header table contents:")
            for key, value in data.items():
                if not key.startswith('__'):
                    if hasattr(value, 'shape'):
                        print(f"  {key}: shape {value.shape}, type {type(value)}")
                    else:
                        print(f"  {key}: {type(value)}")
        except Exception as e:
            print(f"Error reading GSI header: {e}")
    
    # Look for spectrogram directories mentioned in training script
    repo_root = "/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB"
    potential_dirs = [
        "Spectrogram_Image_Database.dir/Unsupervised_images.dir",
        "Unsupervised_database_With_Airguns.dir",
        "Spectrogram_Image_Database.dir"
    ]
    
    for d in potential_dirs:
        full_path = os.path.join(repo_root, d)
        if os.path.exists(full_path):
            print(f"\nFound directory: {full_path}")
            try:
                contents = os.listdir(full_path)
                mat_files = [f for f in contents if f.endswith('.mat')]
                print(f"  Contains {len(mat_files)} .mat files")
                if mat_files:
                    # Analyze first file
                    sample_path = os.path.join(full_path, mat_files[0])
                    sample_data = loadmat(sample_path)
                    for key, value in sample_data.items():
                        if not key.startswith('__') and hasattr(value, 'shape'):
                            print(f"    {key}: {value.shape}, range [{value.min():.3f}, {value.max():.3f}]")
            except Exception as e:
                print(f"  Error analyzing directory: {e}")
        else:
            print(f"Directory not found: {full_path}")

def test_model_architecture():
    """Test the improved autoencoder architecture."""
    print("\n=== TESTING IMPROVED ARCHITECTURE ===")
    
    # Test with standard spectrogram dimensions
    nrow, ncol = 121, 104
    model = ImprovedAutoencoder(nrow, ncol, latent_dim=64)
    
    # Test forward pass
    dummy_input = torch.randn(4, 1, nrow, ncol)  # Batch of 4
    print(f"Input shape: {dummy_input.shape}")
    print(f"Input range: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")
    
    with torch.no_grad():
        output, latent = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Latent shape: {latent.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Check if dimensions match
        if output.shape == dummy_input.shape:
            print("✓ Architecture preserves input/output dimensions")
        else:
            print(f"✗ Dimension mismatch: {output.shape} vs {dummy_input.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

def analyze_existing_models():
    """Analyze existing trained models and their performance."""
    print("\n=== ANALYZING EXISTING MODELS ===")
    
    script_dir = "/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/Pytorch_scripts"
    
    # Look at loss plots
    loss_plots = [f for f in os.listdir(script_dir) if f.endswith('_loss_plot.png')]
    print(f"Found {len(loss_plots)} loss plots:")
    for plot in loss_plots:
        print(f"  {plot}")
    
    # Look at reconstruction error histograms
    error_hists = [f for f in os.listdir(script_dir) if f.endswith('_recon_error_hist.png')]
    print(f"Found {len(error_hists)} reconstruction error histograms:")
    for hist in error_hists:
        print(f"  {hist}")
    
    # Check model files
    models = [f for f in os.listdir(script_dir) if f.endswith('.pth')]
    print(f"Found {len(models)} model files:")
    for model in models:
        model_path = os.path.join(script_dir, model)
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                print(f"  {model}: state dict with {len(checkpoint)} keys")
                # Look at layer shapes to understand architecture
                conv_layers = [k for k in checkpoint.keys() if 'conv' in k and 'weight' in k]
                print(f"    Conv layers: {len(conv_layers)}")
                for layer in conv_layers[:3]:  # Show first few
                    shape = checkpoint[layer].shape
                    print(f"      {layer}: {shape}")
            else:
                print(f"  {model}: unknown format")
        except Exception as e:
            print(f"  {model}: error loading - {e}")

def create_synthetic_test():
    """Create a synthetic test to verify autoencoder learning."""
    print("\n=== SYNTHETIC LEARNING TEST ===")
    
    # Create synthetic spectrogram-like data with clear patterns
    def make_synthetic_spectrogram(batch_size=16):
        """Create synthetic spectrograms with recognizable patterns."""
        specs = []
        for i in range(batch_size):
            spec = np.zeros((121, 104))
            
            # Add some structured patterns
            # Horizontal lines (frequency bands)
            for freq_line in [20, 40, 60, 80]:
                if freq_line < 121:
                    spec[freq_line, :] = 0.8 + 0.2 * np.random.randn(104)
            
            # Vertical lines (time events)
            for time_line in [20, 50, 80]:
                if time_line < 104:
                    spec[:, time_line] = 0.6 + 0.3 * np.random.randn(121)
            
            # Add some noise
            spec += 0.1 * np.random.randn(121, 104)
            
            # Add a diagonal pattern
            for t in range(min(104, 121)):
                if t + 30 < 121:
                    spec[t + 30, t] = 1.0
            
            specs.append(spec)
        
        return np.array(specs)
    
    # Test learning on synthetic data
    synthetic_data = make_synthetic_spectrogram(32)
    print(f"Created synthetic data: {synthetic_data.shape}")
    print(f"Data range: [{synthetic_data.min():.3f}, {synthetic_data.max():.3f}]")
    
    # Convert to torch
    data_tensor = torch.from_numpy(synthetic_data).float().unsqueeze(1)  # Add channel dim
    
    # Create model and test learning
    model = ImprovedAutoencoder(121, 104, latent_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Training on synthetic data for quick test...")
    model.train()
    losses = []
    
    for epoch in range(20):  # Quick test
        optimizer.zero_grad()
        output, _ = model(data_tensor)
        loss = criterion(output, data_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Test reconstruction quality
    model.eval()
    with torch.no_grad():
        final_output, _ = model(data_tensor)
        final_loss = criterion(final_output, data_tensor).item()
        print(f"Final reconstruction loss: {final_loss:.4f}")
        
        # Visualize one example
        orig = data_tensor[0, 0].numpy()
        recon = final_output[0, 0].numpy()
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(orig, cmap='viridis', origin='lower')
        plt.title('Synthetic Original')
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(recon, cmap='viridis', origin='lower')
        plt.title('Reconstruction')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(orig - recon), cmap='hot', origin='lower')
        plt.title('Absolute Error')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig('synthetic_test.png', dpi=150)
        plt.show()
        
        if final_loss < 0.01:
            print("✓ Model can learn synthetic patterns successfully")
        else:
            print("⚠ Model struggles to learn even synthetic patterns")

def main():
    print("AUTOENCODER DIAGNOSTIC ANALYSIS")
    print("=" * 50)
    
    analyze_gsi_data()
    test_model_architecture()
    analyze_existing_models()
    create_synthetic_test()
    
    print("\n=== DIAGNOSIS SUMMARY ===")
    print("Key issues with original autoencoder:")
    print("1. Sigmoid activation constrains output to [0,1] - may not match data range")
    print("2. Small latent dimension (16) may be insufficient for 121×104 spectrograms")
    print("3. No batch normalization - training may be unstable")
    print("4. No data normalization strategy")
    print("\nImproved architecture addresses:")
    print("1. Removed output activation (unbounded reconstruction)")
    print("2. Larger latent space (64D)")
    print("3. Added batch normalization and dropout")
    print("4. Deeper encoder/decoder with skip connections potential")

if __name__ == "__main__":
    main()