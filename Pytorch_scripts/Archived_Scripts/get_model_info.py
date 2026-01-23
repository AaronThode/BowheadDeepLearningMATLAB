#!/usr/bin/env python3
"""
Extract model information from a trained autoencoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class ImprovedAutoencoder(nn.Module):
    """Autoencoder with batch normalization and no sigmoid constraint."""
    
    def __init__(self, nrow=121, ncol=104, latent_dim=64, 
                 base_channels=64, extra_conv=False):
        super().__init__()
        self.nrow, self.ncol = nrow, ncol
        self.extra_conv = extra_conv
        
        if extra_conv:
            nrow_reduced = nrow // 16
            ncol_reduced = ncol // 16
        else:
            nrow_reduced = nrow // 8
            ncol_reduced = ncol // 8
        
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        
        if extra_conv:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(c3, c4, 3, padding=1), nn.BatchNorm2d(c4), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            )
            flat_size = c4 * nrow_reduced * ncol_reduced
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            )
            flat_size = c3 * nrow_reduced * ncol_reduced
        
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
        
        if extra_conv:
            pad_h = (nrow - nrow_reduced * 16) % 2
            pad_w = (ncol - ncol_reduced * 16) % 2
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(c4, c3, 2, stride=2), nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c3, c2, 2, stride=2), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c2, c1, 2, stride=2), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c1, 1, 2, stride=2, output_padding=(pad_h, pad_w)),
            )
        else:
            pad_h = (nrow - nrow_reduced * 8) % 2
            pad_w = (ncol - ncol_reduced * 8) % 2
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(c3, c2, 2, stride=2), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c2, c1, 2, stride=2), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c1, 1, 2, stride=2, output_padding=(pad_h, pad_w)),
            )
        
        self.flat_size = flat_size
        self.nrow_reduced = nrow_reduced
        self.ncol_reduced = ncol_reduced
        self.base_channels = base_channels
        self.c_out = c4 if extra_conv else c3

    def forward(self, x):
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        latent = self.to_latent(x_flat)
        x_recon = self.from_latent(latent)
        x_recon = x_recon.view(x_recon.size(0), self.c_out, self.nrow_reduced, self.ncol_reduced)
        output = self.decoder(x_recon)
        return output, latent


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "autoencoder_clean.pth"
    
    # Load model
    print(f"Loading model from: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    
    # Create model (standard config for v08)
    model = ImprovedAutoencoder(nrow=121, ncol=104, latent_dim=32, 
                                base_channels=64, extra_conv=False)
    model.load_state_dict(state_dict, strict=True)
    
    # Count parameters
    n_params = count_parameters(model)
    
    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70)
    print(f"Number of model parameters: {n_params:,}")
    print(f"Architecture: latent_dim=32, base_channels=64, extra_conv=False")
    print(f"Input shape: 121 x 104")
    print("="*70)
