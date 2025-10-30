#!/usr/bin/env python3
"""
Demonstrate autoencoder reconstruction quality issues and solutions.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

# Recreate the problematic original autoencoder
class OriginalAutoencoder(nn.Module):
    def __init__(self, nrow=121, ncol=104, latent_dim=16, n_channels=16):
        super().__init__()
        self.nrow, self.ncol = nrow, ncol
        nrow_reduced = int(nrow / 8)
        ncol_reduced = int(ncol / 8) 
        nel_reduced = nrow_reduced * ncol_reduced * n_channels
        
        # Original architecture from your script
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, n_channels, 3, padding=1)
        self.t_conv1 = nn.ConvTranspose2d(n_channels, 8, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(8, 4, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(4, 1, [2, 2], stride=[2, 2], output_padding=(1, 0))
        self.fc1 = nn.Linear(nel_reduced, latent_dim)
        self.fc2 = nn.Linear(latent_dim, nel_reduced)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.nel_reduced = nel_reduced
        self.nrow_reduced = nrow_reduced
        self.ncol_reduced = ncol_reduced
        self.n_channels = n_channels

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, self.nel_reduced)
        latent = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(latent))
        x = x.view(-1, self.n_channels, self.nrow_reduced, self.ncol_reduced)
        x = torch.relu(self.t_conv1(x))
        x = torch.relu(self.t_conv2(x))
        output = torch.sigmoid(self.t_conv3(x))  # PROBLEM: Sigmoid constraint!
        return output, latent

# Improved autoencoder without sigmoid
class ImprovedAutoencoder(nn.Module):
    def __init__(self, nrow=121, ncol=104, latent_dim=64):
        super().__init__()
        self.nrow, self.ncol = nrow, ncol
        nrow_reduced = nrow // 8
        ncol_reduced = ncol // 8
        
        # Encoder with batch norm
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        flat_size = 128 * nrow_reduced * ncol_reduced
        
        self.to_latent = nn.Sequential(
            nn.Linear(flat_size, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, flat_size),
            nn.ReLU(inplace=True)
        )
        
        # Decoder without final activation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 1, 2, stride=2, output_padding=(nrow % 8, ncol % 8)),
            # No sigmoid! Allow unbounded output
        )
        
        self.flat_size = flat_size
        self.nrow_reduced = nrow_reduced
        self.ncol_reduced = ncol_reduced

    def forward(self, x):
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        latent = self.to_latent(x_flat)
        x_recon = self.from_latent(latent)
        x_recon = x_recon.view(x_recon.size(0), 128, self.nrow_reduced, self.ncol_reduced)
        output = self.decoder(x_recon)
        return output, latent

def create_realistic_spectrograms(n_samples=5):
    """Create realistic spectrogram-like data with various features."""
    spectrograms = []
    
    for i in range(n_samples):
        # Base dimensions
        spec = np.zeros((121, 104))
        
        # Add frequency bands (horizontal features)
        for freq in [15, 30, 45, 70, 95]:
            if freq < 121:
                # Create frequency band with some variation
                amplitude = 0.5 + 0.3 * np.random.randn()
                width = np.random.randint(1, 4)
                for w in range(width):
                    if freq + w < 121:
                        spec[freq + w, :] = amplitude * (1 + 0.1 * np.random.randn(104))
        
        # Add time-localized events (vertical features)
        n_events = np.random.randint(2, 6)
        for _ in range(n_events):
            t_center = np.random.randint(10, 94)
            f_center = np.random.randint(20, 101)
            duration = np.random.randint(3, 8)
            bandwidth = np.random.randint(5, 15)
            amplitude = 0.8 + 0.4 * np.random.randn()
            
            # Gaussian-like event
            for t_offset in range(-duration//2, duration//2 + 1):
                for f_offset in range(-bandwidth//2, bandwidth//2 + 1):
                    t_idx = t_center + t_offset
                    f_idx = f_center + f_offset
                    if 0 <= t_idx < 104 and 0 <= f_idx < 121:
                        distance = np.sqrt(t_offset**2 + f_offset**2)
                        spec[f_idx, t_idx] += amplitude * np.exp(-distance**2 / (2 * 2**2))
        
        # Add noise
        spec += 0.1 * np.random.randn(121, 104)
        
        # Normalize to roughly [0, 2] range (typical spectrogram range)
        spec = np.clip(spec, 0, None)
        spec = spec / np.max(spec) * 2.0 if np.max(spec) > 0 else spec
        
        spectrograms.append(spec)
    
    return np.array(spectrograms)

def compare_autoencoder_performance():
    """Compare original vs improved autoencoder on realistic data."""
    print("Creating realistic spectrogram data...")
    
    # Create data
    data = create_realistic_spectrograms(8)
    data_tensor = torch.from_numpy(data).float().unsqueeze(1)  # Add channel dimension
    
    print(f"Data shape: {data_tensor.shape}")
    print(f"Data range: [{data_tensor.min():.3f}, {data_tensor.max():.3f}]")
    
    # Create models
    original_model = OriginalAutoencoder(latent_dim=16)
    improved_model = ImprovedAutoencoder(latent_dim=64)
    
    print(f"Original model parameters: {sum(p.numel() for p in original_model.parameters()):,}")
    print(f"Improved model parameters: {sum(p.numel() for p in improved_model.parameters()):,}")
    
    # Quick training function
    def quick_train(model, data, epochs=50, lr=0.001):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        model.train()
        
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
        
        return losses
    
    # Train both models
    print("\nTraining original model...")
    original_losses = quick_train(original_model, data_tensor)
    
    print("\nTraining improved model...")
    improved_losses = quick_train(improved_model, data_tensor)
    
    # Evaluate reconstructions
    original_model.eval()
    improved_model.eval()
    
    with torch.no_grad():
        original_recon, _ = original_model(data_tensor)
        improved_recon, _ = improved_model(data_tensor)
    
    # Create comparison figure
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    
    for i in range(5):
        # Original data
        axes[0, i].imshow(data[i], cmap='viridis', origin='lower', aspect='auto')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Original model reconstruction
        orig_recon = original_recon[i, 0].numpy()
        axes[1, i].imshow(orig_recon, cmap='viridis', origin='lower', aspect='auto')
        mse_orig = np.mean((data[i] - orig_recon)**2)
        axes[1, i].set_title(f'Original Model\\nMSE: {mse_orig:.3f}')
        axes[1, i].axis('off')
        
        # Improved model reconstruction
        imp_recon = improved_recon[i, 0].numpy()
        axes[2, i].imshow(imp_recon, cmap='viridis', origin='lower', aspect='auto')
        mse_imp = np.mean((data[i] - imp_recon)**2)
        axes[2, i].set_title(f'Improved Model\\nMSE: {mse_imp:.3f}')
        axes[2, i].axis('off')
        
        # Difference
        diff = np.abs(data[i] - orig_recon)
        axes[3, i].imshow(diff, cmap='hot', origin='lower', aspect='auto')
        axes[3, i].set_title(f'Error (Original)')
        axes[3, i].axis('off')
    
    plt.suptitle('Autoencoder Comparison: Original vs Improved\\n' + 
                 'Top: Originals, 2nd: Original Model, 3rd: Improved Model, Bottom: Error')
    plt.tight_layout()
    plt.savefig('autoencoder_comparison.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # Loss comparison
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(original_losses, label='Original (sigmoid output)')
    plt.plot(improved_losses, label='Improved (no sigmoid)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    # Final reconstruction quality metrics
    original_final_mse = np.mean((data - original_recon.squeeze().numpy())**2)
    improved_final_mse = np.mean((data - improved_recon.squeeze().numpy())**2)
    
    models = ['Original\\n(sigmoid)', 'Improved\\n(no sigmoid)']
    mse_values = [original_final_mse, improved_final_mse]
    colors = ['red', 'green']
    
    plt.bar(models, mse_values, color=colors, alpha=0.7)
    plt.ylabel('Final MSE')
    plt.title('Reconstruction Quality')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=150)
    plt.show()
    
    print(f"\\nFinal Results:")
    print(f"Original model final MSE: {original_final_mse:.4f}")
    print(f"Improved model final MSE: {improved_final_mse:.4f}")
    print(f"Improvement factor: {original_final_mse / improved_final_mse:.2f}x")
    
    # Save sample data for further testing
    sample_data = {
        'spectrograms': data,
        'original_recon': original_recon.squeeze().numpy(),
        'improved_recon': improved_recon.squeeze().numpy()
    }
    savemat('autoencoder_comparison_data.mat', sample_data)
    print("Saved comparison data to autoencoder_comparison_data.mat")

if __name__ == "__main__":
    compare_autoencoder_performance()